"""Grounded sensory-fallback gate for the promoted fused route/vote mode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from marulho.evaluation.inplace_grounded_quality_benchmark import (
    _make_grounded_stream,
    _run_grounded_arm,
    compare_grounded_arms,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.checkpointing import load_trainer_checkpoint


def run_fused_route_vote_grounded_benchmark(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    samples: int = 128,
    warmup_steps: int = 8,
    concepts: int = 8,
    seed: int = 20260621,
) -> dict[str, object]:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("fused route/vote grounded benchmark requires CUDA")
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    stream = _make_grounded_stream(
        input_dim=trainer.config.input_dim,
        visual_dim=trainer.config.cross_modal_dim_visual,
        audio_dim=trainer.config.cross_modal_dim_audio,
        samples=samples + warmup_steps,
        concepts=concepts,
        seed=seed + 1,
    )
    n_columns = int(trainer.config.n_columns)
    del trainer

    baseline_a, baseline_state_a = _run_grounded_arm(
        checkpoint,
        executor="runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    variant_a, variant_state_a = _run_grounded_arm(
        checkpoint,
        executor="fused_triton_text_runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    variant_b, variant_state_b = _run_grounded_arm(
        checkpoint,
        executor="fused_triton_text_runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    baseline_b, baseline_state_b = _run_grounded_arm(
        checkpoint,
        executor="runtime",
        stream=stream,
        warmup_steps=warmup_steps,
    )
    comparison_a = compare_grounded_arms(
        baseline_a,
        variant_a,
        baseline_state_a,
        variant_state_a,
    )
    comparison_b = compare_grounded_arms(
        baseline_b,
        variant_b,
        baseline_state_b,
        variant_state_b,
    )
    baseline_mean = (
        float(baseline_a["throughput_ticks_per_second"])
        + float(baseline_b["throughput_ticks_per_second"])
    ) / 2.0
    variant_mean = (
        float(variant_a["throughput_ticks_per_second"])
        + float(variant_b["throughput_ticks_per_second"])
    ) / 2.0
    reversed_speedup = variant_mean / max(baseline_mean, 1e-9)
    comparison = comparison_a
    comparison["reversed_pair"] = {
        "baseline_mean_ticks_per_second": baseline_mean,
        "variant_mean_ticks_per_second": variant_mean,
        "speedup": reversed_speedup,
        "second_pair_quality_preserved": bool(
            comparison_b["quality_preserved"]
            or all(
                value
                for name, value in comparison_b["gates"].items()
                if name != "speedup_at_least_1_10x"
            )
        ),
        "second_pair_grounding_quality_preserved": bool(
            comparison_b["grounding_quality_preserved"]
        ),
    }
    variant_reports = (
        variant_a["column_transition_runtime"],
        variant_b["column_transition_runtime"],
    )
    route_execution_count = sum(
        int(report["route_vote_execution_count"])
        for report in variant_reports
    )
    sensory_fallback_count = sum(
        int(report["route_vote_sensory_fallback_count"])
        for report in variant_reports
    )
    expected_sensory_fallback_count = 2 * (samples + warmup_steps)
    comparison["gates"].pop("speedup_at_least_1_10x", None)
    comparison["gates"]["sensory_fallback_throughput_within_10_percent"] = (
        reversed_speedup >= 0.90
    )
    comparison["gates"]["fused_route_not_executed_on_sensory_ticks"] = (
        route_execution_count == 0
    )
    comparison["gates"]["all_sensory_ticks_reported_as_fallback"] = (
        sensory_fallback_count == expected_sensory_fallback_count
    )
    comparison["sensory_fallback_evidence"] = {
        "route_vote_execution_count": route_execution_count,
        "sensory_fallback_count": sensory_fallback_count,
        "expected_sensory_fallback_count": expected_sensory_fallback_count,
        "throughput_floor": 0.90,
    }
    comparison["quality_preserved"] = all(comparison["gates"].values())
    comparison["promotion_eligible"] = bool(
        comparison["quality_preserved"]
        and comparison["grounding_quality_preserved"]
        and comparison["reversed_pair"]["second_pair_quality_preserved"]
        and comparison["reversed_pair"]["second_pair_grounding_quality_preserved"]
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "marulho_fused_route_vote_grounded_benchmark",
        "surface": "fused_route_vote_grounded_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "production_text_route_vote_sensory_fallback_supported"
            if comparison["promotion_eligible"]
            else "blocked_grounded_quality_or_throughput_regression"
        ),
        "passed": bool(comparison["promotion_eligible"]),
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "seed": seed,
        "n_columns": n_columns,
        "concepts": concepts,
        "samples": samples,
        "warmup_steps": warmup_steps,
        "baseline": baseline_a,
        "variant": variant_a,
        "reversed_pair_arms": {
            "variant_b": variant_b,
            "baseline_b": baseline_b,
        },
        "comparison": comparison,
        "promotion_gate": {
            "runtime_default_changed": False,
            "checkpoint_opt_in_required": True,
            "production_owned_lifecycle": True,
            "eligible": bool(comparison["promotion_eligible"]),
        },
        "mutates_live_runtime": False,
        "writes_live_checkpoint": False,
        "claim_boundary": (
            "synthetic correlated visual/audio spike associations on isolated "
            "checkpoint clones; fused text routing is disabled on sensory ticks "
            "and the gate excludes service/source throughput and real-world "
            "semantic grounding"
        ),
    }
    if output_path is not None:
        write_json_report_with_readme(
            output_path,
            report,
            title="Fused Route Vote Grounded Benchmark",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--concepts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()
    report = run_fused_route_vote_grounded_benchmark(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        concepts=args.concepts,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
