from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import load_trainer_checkpoint


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trial_plan(design: Mapping[str, Any]) -> dict[str, Any]:
    growth = design.get("growth_evidence") if isinstance(design.get("growth_evidence"), Mapping) else {}
    topology = design.get("topology_trial") if isinstance(design.get("topology_trial"), Mapping) else {}
    return {
        "surface": "binding_candidate_hub_topology_plan.v1",
        "candidate_column_ids": list(growth.get("candidate_column_ids") or []),
        "max_total_edge_delta": int(topology.get("max_total_edge_delta", 0)),
        "proposed_total_edge_delta": int(topology.get("proposed_total_edge_delta", 0)),
        "proposed_edges": list(topology.get("proposed_edges") or []),
        "baseline_topology_hash": topology.get("baseline_topology_hash"),
        "plan_hash": topology.get("binding_plan_hash"),
    }


def _stored_patterns(trainer: Any, *, max_samples: int) -> list[tuple[torch.Tensor, str | None]]:
    store = trainer.model.memory_store
    patterns: list[tuple[torch.Tensor, str | None]] = []
    for pattern, raw_window in zip(store.slow_input_patterns, store.slow_raw_windows):
        if not isinstance(pattern, torch.Tensor):
            continue
        patterns.append((pattern.detach().clone(), None if raw_window is None else str(raw_window)))
        if len(patterns) >= max(1, int(max_samples)):
            break
    return patterns


def _run_clone(
    checkpoint_path: Path,
    *,
    plan: Mapping[str, Any] | None,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    trainer, metadata = load_trainer_checkpoint(checkpoint_path)
    patterns = _stored_patterns(trainer, max_samples=max_samples)
    if not patterns:
        return {
            "available": False,
            "reason": "checkpoint_contains_no_stored_input_patterns",
            "sample_count": 0,
            "metadata": metadata,
        }
    application = None
    if plan is not None:
        binding = trainer.model.binding_layer
        apply_plan = getattr(binding, "apply_candidate_hub_topology_plan", None)
        if not callable(apply_plan):
            return {
                "available": False,
                "reason": "binding_layer_does_not_support_exact_plan_application",
                "sample_count": len(patterns),
                "metadata": metadata,
            }
        application = apply_plan(dict(plan), reason="isolated checkpoint-clone binding growth trial")

    recon_errors: list[float] = []
    binding_strengths: list[float] = []
    step_latencies_ms: list[float] = []
    started = time.perf_counter()
    for pattern, raw_window in patterns:
        step_started = time.perf_counter()
        metrics = trainer.train_step(
            pattern,
            raw_window=raw_window,
            allow_sleep_maintenance=False,
        )
        if trainer.model.device.type == "cuda":
            torch.cuda.synchronize()
        step_latencies_ms.append((time.perf_counter() - step_started) * 1000.0)
        recon_errors.append(float(metrics.get("recon_error", 0.0)))
        binding_strengths.append(float(metrics.get("binding_strength", 0.0)))
    total_latency_ms = (time.perf_counter() - started) * 1000.0
    predictive = trainer.model.predictive
    spike_health = trainer.model.competitive.spike_health_report()
    peak_cuda_bytes = (
        int(torch.cuda.max_memory_allocated())
        if trainer.model.device.type == "cuda"
        else 0
    )
    return {
        "available": True,
        "reason": None,
        "sample_count": len(patterns),
        "device": str(trainer.model.device),
        "application": application,
        "metrics": {
            "reconstruction_error_mean": statistics.fmean(recon_errors),
            "prediction_error_mean": float(predictive.prediction_error.mean().item()),
            "prediction_error_max": float(predictive.prediction_error.max().item()),
            "binding_strength_mean": statistics.fmean(binding_strengths),
            "step_latency_median_ms": statistics.median(step_latencies_ms),
            "step_latency_p95_ms": sorted(step_latencies_ms)[
                max(0, int(len(step_latencies_ms) * 0.95) - 1)
            ],
            "total_latency_ms": total_latency_ms,
            "peak_cuda_memory_bytes": peak_cuda_bytes,
        },
        "spike_health": spike_health,
        "metadata": metadata,
    }


def run_binding_growth_trial(
    *,
    checkpoint_path: str | Path,
    trial_design: Mapping[str, Any],
    max_samples: int = 16,
    seed: int = 17,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    gate = (
        trial_design.get("promotion_gate")
        if isinstance(trial_design.get("promotion_gate"), Mapping)
        else {}
    )
    required = {
        "checkpoint_available": checkpoint.is_file(),
        "trial_design_surface_available": trial_design.get("surface")
        == "binding_growth_trial_design.v1",
        "trial_design_ready": bool(gate.get("eligible_for_isolated_trial")),
        "trial_design_non_executable": not bool(trial_design.get("executable")),
        "trial_design_non_mutating": not bool(trial_design.get("mutates_runtime_state")),
        "trial_design_hash_available": len(
            str(trial_design.get("binding_growth_trial_design_hash") or "")
        )
        == 64,
    }
    if not all(required.values()):
        report = {
            "schema_version": 1,
            "artifact_kind": "marulho_binding_growth_trial",
            "surface": "binding_growth_trial_evaluation.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "blocked_missing_binding_growth_trial_evidence",
            "passed": False,
            "required_evidence": required,
            "mutates_live_runtime": False,
        }
    else:
        plan = _trial_plan(trial_design)
        baseline = _run_clone(
            checkpoint,
            plan=None,
            max_samples=max_samples,
            seed=seed,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        variant = _run_clone(
            checkpoint,
            plan=plan,
            max_samples=max_samples,
            seed=seed,
        )
        baseline_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), Mapping) else {}
        variant_metrics = variant.get("metrics") if isinstance(variant.get("metrics"), Mapping) else {}
        prediction_delta = float(variant_metrics.get("prediction_error_mean", 0.0)) - float(
            baseline_metrics.get("prediction_error_mean", 0.0)
        )
        reconstruction_delta = float(
            variant_metrics.get("reconstruction_error_mean", 0.0)
        ) - float(baseline_metrics.get("reconstruction_error_mean", 0.0))
        latency_delta = float(variant_metrics.get("total_latency_ms", 0.0)) - float(
            baseline_metrics.get("total_latency_ms", 0.0)
        )
        evaluated = bool(baseline.get("available")) and bool(variant.get("available"))
        improvement = evaluated and prediction_delta < 0.0 and reconstruction_delta <= 0.0
        report = {
            "schema_version": 1,
            "artifact_kind": "marulho_binding_growth_trial",
            "surface": "binding_growth_trial_evaluation.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": (
                "evidence_supported_for_operator_review"
                if improvement
                else "evaluated_without_cognitive_improvement"
                if evaluated
                else "blocked_missing_checkpoint_replay_patterns"
            ),
            "passed": improvement,
            "required_evidence": required,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _checkpoint_hash(checkpoint),
            },
            "trial_design_hash": trial_design.get("binding_growth_trial_design_hash"),
            "baseline": baseline,
            "variant": variant,
            "deltas": {
                "prediction_error_mean": prediction_delta,
                "reconstruction_error_mean": reconstruction_delta,
                "total_latency_ms": latency_delta,
            },
            "promotion_gate": {
                "eligible_for_structural_mutation_design": improvement,
                "requires_operator_review": True,
                "requires_live_checkpoint_transaction": True,
                "required_improvement": {
                    "prediction_error_reduced": prediction_delta < 0.0,
                    "reconstruction_error_not_regressed": reconstruction_delta <= 0.0,
                },
            },
            "mutates_live_runtime": False,
            "writes_live_checkpoint": False,
            "hot_path_effect": "none_explicit_evaluation_runner",
        }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="Binding Growth Trial Evaluation",
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a binding growth plan on sequential checkpoint clones."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trial-design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    design = json.loads(args.trial_design.read_text(encoding="utf-8"))
    report = run_binding_growth_trial(
        checkpoint_path=args.checkpoint,
        trial_design=design,
        max_samples=args.max_samples,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "blocked_missing_binding_growth_trial_evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
