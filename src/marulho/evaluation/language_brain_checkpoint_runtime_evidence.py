"""Evidence runner for reviewed LM checkpoints installed into MarulhoBrain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_brain_checkpoint_runtime_evidence.v1"
ARTIFACT_KIND = "marulho_language_brain_checkpoint_runtime_evidence"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiny_brain_config(*, device: str) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=16,
        column_latent_dim=16,
        window_size=4,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=128,
        enable_learned_chunking=False,
        micro_sleep_interval_tokens=10_000,
        deep_sleep_interval_tokens=10_000,
        trainer_telemetry_interval_tokens=10_000,
        device=str(device),
    )


def _compact_generation(generation: Mapping[str, Any]) -> dict[str, Any]:
    decode = (
        generation.get("generation_decode")
        if isinstance(generation.get("generation_decode"), Mapping)
        else {}
    )
    continuation = str(generation.get("continuation_text") or "")
    return {
        "surface": "marulho_brain_generation_probe_summary.v1",
        "active_language_path": generation.get("active_language_path"),
        "external_llm_used": bool(generation.get("external_llm_used", False)),
        "checkpointed_language_components": bool(
            generation.get("checkpointed_language_components", False)
        ),
        "prompt_token_count": int(generation.get("prompt_token_count", 0) or 0),
        "generated_token_count": int(generation.get("generated_token_count", 0) or 0),
        "emitted_tokens": int(generation.get("emitted_tokens", 0) or 0),
        "continuation_preview": continuation[:240],
        "generation_decode": dict(decode),
    }


def _status_read_mutates(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return (
        int(before.get("token_count", 0) or 0) != int(after.get("token_count", 0) or 0)
        or int(before.get("queued_tokens", 0) or 0)
        != int(after.get("queued_tokens", 0) or 0)
        or str(before.get("active_language_path") or "")
        != str(after.get("active_language_path") or "")
    )


def _run_brain_generation_window(
    brain: MarulhoBrain,
    *,
    prompt: str,
    target_tokens: int,
    chunk_tokens: int,
    timeout_seconds: float,
    generation_repetition_penalty: float,
    generation_no_repeat_ngram_size: int,
) -> dict[str, Any]:
    target = max(0, int(target_tokens))
    chunk = max(1, int(chunk_tokens))
    timeout = max(0.0, float(timeout_seconds))
    started = time.perf_counter()
    token_delta = 0
    call_count = 0
    failure_reason: str | None = None
    last_generation: dict[str, Any] = {}
    while token_delta < target:
        elapsed = time.perf_counter() - started
        if timeout > 0.0 and elapsed >= timeout:
            failure_reason = "target_tokens_not_reached_before_timeout"
            break
        request_tokens = min(chunk, target - token_delta)
        generation = brain.generate(
            prompt=prompt,
            max_tokens=request_tokens,
            generation_repetition_penalty=generation_repetition_penalty,
            generation_no_repeat_ngram_size=generation_no_repeat_ngram_size,
        )
        call_count += 1
        emitted = int(generation.get("emitted_tokens", 0) or 0)
        last_generation = _compact_generation(generation)
        if emitted <= 0:
            failure_reason = "generation_emitted_no_tokens"
            break
        token_delta += emitted
        if emitted < request_tokens:
            failure_reason = "generation_stopped_before_request"
            break
    elapsed = time.perf_counter() - started
    success = token_delta >= target
    if success:
        failure_reason = None
    return {
        "surface": "marulho_brain_generation_window.v1",
        "target_tokens": target,
        "chunk_tokens": chunk,
        "token_delta": int(token_delta),
        "generation_call_count": int(call_count),
        "elapsed_seconds": float(elapsed),
        "tokens_per_second": (float(token_delta) / elapsed if elapsed > 0.0 else 0.0),
        "success": bool(success),
        "failure_reason": failure_reason,
        "last_generation": last_generation,
    }


def _compact_sustained_report(report: Mapping[str, Any], *, output_path: Path) -> dict[str, Any]:
    execution = (
        report.get("execution_evidence")
        if isinstance(report.get("execution_evidence"), Mapping)
        else {}
    )
    promotion = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    return {
        "surface": "marulho_brain_restored_training_owned_sustained_summary.v1",
        "output_path": str(output_path),
        "report_status": report.get("report_status"),
        "success": bool(report.get("success", False)),
        "target_tokens": int(report.get("target_tokens", 0) or 0),
        "token_delta": int(report.get("token_delta", 0) or 0),
        "elapsed_seconds": float(report.get("elapsed_seconds", 0.0) or 0.0),
        "tokens_per_second": float(report.get("tokens_per_second", 0.0) or 0.0),
        "failure_reason": report.get("failure_reason"),
        "active_language_path": report.get("active_language_path"),
        "device": report.get("device"),
        "backend": execution.get("backend"),
        "mode": execution.get("mode"),
        "cuda_graph_burst_used": bool(execution.get("cuda_graph_burst_used", False)),
        "cuda_graph_burst_tokens": int(execution.get("cuda_graph_burst_tokens", 0) or 0),
        "cuda_graph_burst_replay_count": int(
            execution.get("cuda_graph_burst_replay_count", 0) or 0
        ),
        "tracked_triton_kernel_used_names": list(
            execution.get("tracked_triton_kernel_used_names") or []
        ),
        "tracked_triton_kernel_failure_count": int(
            execution.get("tracked_triton_kernel_failure_count", 0) or 0
        ),
        "promotes_runtime_claim": bool(
            promotion.get("promotes_runtime_claim", False)
        ),
    }


def _base_report(
    *,
    output_path: str | Path,
    promotion_review_path: str | Path,
    brain_checkpoint_path: str | Path,
    target_tokens: int,
    chunk_tokens: int,
    device: str,
) -> dict[str, Any]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "report_status": "partial",
        "output_path": str(output_path),
        "promotion_review_path": str(promotion_review_path),
        "brain_checkpoint_path": str(brain_checkpoint_path),
        "target_tokens": int(target_tokens),
        "chunk_tokens": int(chunk_tokens),
        "runtime_owner": "MarulhoBrain",
        "requested_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "service_owned_cognition": False,
        "status_read_mutation": False,
        "promotes_runtime_claim": False,
    }


def build_language_brain_checkpoint_runtime_evidence(
    *,
    output_path: str | Path,
    promotion_review_path: str | Path,
    brain_checkpoint_path: str | Path,
    operator_approved: bool,
    operator_id: str | None = None,
    approval_note: str = "",
    artifact_base_dir: str | Path | None = None,
    device: str | None = None,
    prompt: str = "MARULHO",
    target_tokens: int = 8192,
    chunk_tokens: int = 128,
    timeout_seconds: float = 600.0,
    run_training_owned_sustained: bool = True,
    training_owned_target_tokens: int | None = None,
    training_owned_timeout_seconds: float | None = None,
    training_owned_output_path: str | Path | None = None,
    generation_repetition_penalty: float = 1.15,
    generation_no_repeat_ngram_size: int = 3,
) -> dict[str, Any]:
    """Install a reviewed child checkpoint into MarulhoBrain and verify restore."""

    selected_device = str(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output = Path(output_path)
    brain_checkpoint = Path(brain_checkpoint_path)
    report = _base_report(
        output_path=output,
        promotion_review_path=promotion_review_path,
        brain_checkpoint_path=brain_checkpoint,
        target_tokens=target_tokens,
        chunk_tokens=chunk_tokens,
        device=selected_device,
    )
    exception: BaseException | None = None
    try:
        brain = MarulhoBrain.fresh(_tiny_brain_config(device=selected_device))
        install = brain.install_language_checkpoint_from_promotion_review(
            promotion_review_path,
            operator_approved=bool(operator_approved),
            operator_id=operator_id,
            approval_note=approval_note,
            artifact_base_dir=artifact_base_dir,
        )
        report["installation"] = dict(install)
        if not bool(install.get("installed")):
            report["status"] = "blocked_language_brain_checkpoint_runtime_evidence"
            report["report_status"] = "partial"
            report["failure_reason"] = "language_checkpoint_installation_blocked"
            return report

        saved = brain.save(brain_checkpoint)
        saved_path = Path(str(saved["path"]))
        brain_checkpoint_hash = _sha256_file(saved_path)
        restored = MarulhoBrain.load(saved_path)
        before_status = restored.status()
        after_status = restored.status()
        status_read_mutation = _status_read_mutates(before_status, after_status)
        generation_window = _run_brain_generation_window(
            restored,
            prompt=prompt,
            target_tokens=int(target_tokens),
            chunk_tokens=int(chunk_tokens),
            timeout_seconds=float(timeout_seconds),
            generation_repetition_penalty=float(generation_repetition_penalty),
            generation_no_repeat_ngram_size=int(generation_no_repeat_ngram_size),
        )
        training_owned_sustained: dict[str, Any] = {
            "surface": "marulho_brain_restored_training_owned_sustained_summary.v1",
            "enabled": False,
            "reason": "training_owned_sustained_not_requested",
        }
        if bool(run_training_owned_sustained):
            if restored.status().get("active_language_path") != "marulho_lm_head":
                training_owned_sustained = {
                    "surface": (
                        "marulho_brain_restored_training_owned_sustained_summary.v1"
                    ),
                    "enabled": True,
                    "success": False,
                    "failure_reason": "restored_language_runtime_not_installed",
                }
            else:
                sustained_output = Path(
                    training_owned_output_path
                    or output.with_name(
                        f"{output.stem}-training-owned-sustained.json"
                    )
                )
                sustained_generation = restored.generate_sustained_language(
                    output_path=sustained_output,
                    target_tokens=int(training_owned_target_tokens or target_tokens),
                    prompt=prompt,
                    tick_tokens=int(chunk_tokens),
                    quantum_tokens=16,
                    timeout_seconds=float(
                        training_owned_timeout_seconds
                        if training_owned_timeout_seconds is not None
                        else timeout_seconds
                    ),
                    generation_repetition_penalty=float(generation_repetition_penalty),
                    generation_no_repeat_ngram_size=int(
                        generation_no_repeat_ngram_size
                    ),
                    collect_environment=False,
                )
                sustained_report = {
                    **sustained_generation,
                    "surface": sustained_generation.get("language_model_surface"),
                }
                training_owned_sustained = _compact_sustained_report(
                    sustained_report,
                    output_path=sustained_output,
                )
        restored_status = restored.status()
        checkpoint_restore_verified = (
            str(before_status.get("active_language_path") or "") == "marulho_lm_head"
            and bool(
                before_status.get("language_model", {}).get(
                    "checkpointed_language_components"
                )
            )
            and int(
                before_status.get("language_model", {}).get(
                    "checkpoint_installation_count",
                    0,
                )
                or 0
            )
            >= 1
        )
        success = (
            bool(generation_window.get("success"))
            and checkpoint_restore_verified
            and not status_read_mutation
            and (
                not bool(run_training_owned_sustained)
                or bool(training_owned_sustained.get("success", False))
            )
        )
        brain_api_tps = float(generation_window.get("tokens_per_second", 0.0) or 0.0)
        training_owned_tps = float(
            training_owned_sustained.get("tokens_per_second", 0.0) or 0.0
        )
        report.update(
            {
                "status": (
                    "final"
                    if success
                    else "blocked_language_brain_checkpoint_runtime_evidence"
                ),
                "report_status": "final" if success else "partial",
                "failure_reason": generation_window.get("failure_reason"),
                "brain_checkpoint": {
                    "surface": "marulho_brain_installed_language_checkpoint.v1",
                    "path": str(saved_path),
                    "sha256": brain_checkpoint_hash,
                    "save_report": dict(saved),
                    "restore_verified": bool(checkpoint_restore_verified),
                },
                "status_read": {
                    "surface": "marulho_brain_status_read_check.v1",
                    "mutates_runtime_state": bool(status_read_mutation),
                    "before_token_count": int(before_status.get("token_count", 0) or 0),
                    "after_token_count": int(after_status.get("token_count", 0) or 0),
                    "active_language_path": before_status.get("active_language_path"),
                },
                "restored_brain": {
                    "surface": "marulho_restored_brain_language_runtime.v1",
                    "active_language_path": restored_status.get("active_language_path"),
                    "device": restored_status.get("device"),
                    "language_model": dict(restored_status.get("language_model") or {}),
                    "last_trace": dict(restored_status.get("last_trace") or {}),
                },
                "generation_window": generation_window,
                "training_owned_sustained_window": training_owned_sustained,
                "brain_api_tokens_per_second": brain_api_tps,
                "training_owned_tokens_per_second": training_owned_tps,
                "training_owned_speedup_over_brain_api": (
                    training_owned_tps / brain_api_tps if brain_api_tps > 0.0 else 0.0
                ),
                "target_tokens": int(generation_window.get("target_tokens", 0) or 0),
                "token_delta": int(generation_window.get("token_delta", 0) or 0),
                "elapsed_seconds": float(
                    generation_window.get("elapsed_seconds", 0.0) or 0.0
                ),
                "tokens_per_second": float(
                    generation_window.get("tokens_per_second", 0.0) or 0.0
                ),
                "active_language_path": restored_status.get("active_language_path"),
                "status_read_mutation": bool(status_read_mutation),
            }
        )
        gate = {
            "surface": "marulho_language_brain_checkpoint_runtime_gate.v1",
            "installed_reviewed_checkpoint": bool(install.get("installed")),
            "candidate_checkpoint_hash_verified": bool(
                install.get("candidate_checkpoint", {}).get("hash_verified")
            ),
            "brain_checkpoint_saved": saved_path.is_file(),
            "brain_checkpoint_restore_verified": bool(checkpoint_restore_verified),
            "status_read_mutation_absent": not bool(status_read_mutation),
            "restored_generation_uses_marulho_lm_head": (
                restored_status.get("active_language_path") == "marulho_lm_head"
            ),
            "external_llm_absent": not bool(report.get("external_llm_used", False)),
            "service_owned_cognition_absent": not bool(
                report.get("service_owned_cognition", False)
            ),
            "target_tokens_reached": bool(generation_window.get("success")),
            "training_owned_sustained_target_reached": bool(
                training_owned_sustained.get("success", False)
            ),
            "diagnostic_8192_boundary_reached": int(
                generation_window.get("token_delta", 0) or 0
            )
            >= 8192,
            "long_run_131072_boundary_reached": int(
                generation_window.get("token_delta", 0) or 0
            )
            >= 131072,
            "house_scale_524288_boundary_reached": int(
                generation_window.get("token_delta", 0) or 0
            )
            >= 524288,
            "training_owned_diagnostic_8192_boundary_reached": int(
                training_owned_sustained.get("token_delta", 0) or 0
            )
            >= 8192,
            "training_owned_long_run_131072_boundary_reached": int(
                training_owned_sustained.get("token_delta", 0) or 0
            )
            >= 131072,
            "training_owned_house_scale_524288_boundary_reached": int(
                training_owned_sustained.get("token_delta", 0) or 0
            )
            >= 524288,
            "ready_for_runtime_claim_review": False,
            "promotes_runtime_claim": False,
        }
        report["promotion_gate"] = gate
        return report
    except BaseException as exc:  # pragma: no cover - report persistence guard
        exception = exc
        report.update(
            {
                "status": "exception",
                "report_status": "exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return report
    finally:
        if exception is not None:
            report.setdefault(
                "promotion_gate",
                {
                    "surface": "marulho_language_brain_checkpoint_runtime_gate.v1",
                    "ready_for_runtime_claim_review": False,
                    "promotes_runtime_claim": False,
                },
            )
        write_json_report_with_readme(output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-review", type=Path, required=True)
    parser.add_argument("--brain-checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--operator-id", type=str, default="codex-operator")
    parser.add_argument("--approval-note", type=str, default="")
    parser.add_argument("--operator-approved", action="store_true")
    parser.add_argument("--prompt", type=str, default="MARULHO")
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--skip-training-owned-sustained", action="store_true")
    parser.add_argument("--training-owned-target-tokens", type=int, default=None)
    parser.add_argument("--training-owned-timeout-seconds", type=float, default=None)
    parser.add_argument("--training-owned-output", type=Path, default=None)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    report = build_language_brain_checkpoint_runtime_evidence(
        output_path=args.output,
        promotion_review_path=args.promotion_review,
        brain_checkpoint_path=args.brain_checkpoint,
        operator_approved=bool(args.operator_approved),
        operator_id=args.operator_id,
        approval_note=args.approval_note,
        artifact_base_dir=args.artifact_base_dir,
        device=args.device,
        prompt=args.prompt,
        target_tokens=args.target_tokens,
        chunk_tokens=args.chunk_tokens,
        timeout_seconds=args.timeout_seconds,
        run_training_owned_sustained=not bool(args.skip_training_owned_sustained),
        training_owned_target_tokens=args.training_owned_target_tokens,
        training_owned_timeout_seconds=args.training_owned_timeout_seconds,
        training_owned_output_path=args.training_owned_output,
        generation_repetition_penalty=args.generation_repetition_penalty,
        generation_no_repeat_ngram_size=args.generation_no_repeat_ngram_size,
    )
    return 0 if report.get("report_status") == "final" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
