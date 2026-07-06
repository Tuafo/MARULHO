"""Checkpoint-backed generation repair through an installed MarulhoBrain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.brain import MarulhoBrain
from marulho.evaluation.language_brain_generation_evidence import (
    _BrainGenerationAdapter,
    _language_state,
    _language_tokenizer,
    _prompt_case_from_arg,
    _read_text,
    _sha256_file,
    _status_read_mutates,
)
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    default_generation_coherence_prompt_cases,
    run_language_generation_coherence_report,
)
from marulho.evaluation.language_quality_replay_experiment import (
    _build_hard_prompt_corpus,
    _coherence_delta,
)
from marulho.evaluation.language_training_experiment import (
    _apply_cuda_math_policy,
    _resolve_device,
    _restore_cuda_math_policy,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
)
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
)


SURFACE = "marulho_language_brain_installed_generation_repair_evidence.v1"
ARTIFACT_KIND = "marulho_language_brain_installed_generation_repair_evidence"


@dataclass(frozen=True)
class BrainInstalledGenerationRepairEvidenceConfig:
    sequence_length: int = 64
    stride: int = 16
    batch_size: int = 16
    eval_fraction: float = 0.2
    hard_prompt_repeat: int = 16
    hard_prompt_context_chars: int = 192
    max_new_batches: int = 8
    max_replay_batches: int = 8
    max_old_eval_batches: int = 8
    max_new_eval_batches: int = 8
    learning_rate: float = 5e-4
    max_steps: int = 6
    replay_loss_weight: float = 1.5
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 8
    dense_adamw_backend: str = "default"
    forgetting_tolerance: float = 100.0
    replay_retention_tolerance: float = 100.0
    rollback_on_forgetting: bool = False
    collect_training_telemetry: bool = False
    min_case_pass_rate: float = 1.0
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3
    run_post_repair_sustained: bool = False
    sustained_target_tokens: int = 8192
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 600.0
    sustained_prompt: str = "MARULHO"
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
    seed: int = 20260706
    device: str = "auto"


def _trim(batches: Sequence[LanguageBatch], limit: int) -> tuple[LanguageBatch, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _language_model_status(status: Mapping[str, Any]) -> Mapping[str, Any]:
    language_model = status.get("language_model")
    return language_model if isinstance(language_model, Mapping) else {}


def _score_generation(
    brain: MarulhoBrain,
    *,
    tokenizer,
    prompt_cases: Sequence[LanguageGenerationPromptCase],
    checkpoint_path: str | Path,
    config: BrainInstalledGenerationRepairEvidenceConfig,
) -> dict[str, Any]:
    adapter = _BrainGenerationAdapter(brain, tokenizer)
    return run_language_generation_coherence_report(
        adapter,  # type: ignore[arg-type]
        tokenizer,
        prompt_cases=prompt_cases,
        min_case_pass_rate=float(config.min_case_pass_rate),
        checkpoint_path=checkpoint_path,
        generation_repetition_penalty=float(config.generation_repetition_penalty),
        generation_no_repeat_ngram_size=int(config.generation_no_repeat_ngram_size),
    )


def _learning_summary(learning: Mapping[str, Any]) -> dict[str, Any]:
    report = learning.get("report") if isinstance(learning.get("report"), Mapping) else {}
    evidence = (
        report.get("learning_evidence")
        if isinstance(report.get("learning_evidence"), Mapping)
        else {}
    )
    rollback = (
        report.get("rollback_evidence")
        if isinstance(report.get("rollback_evidence"), Mapping)
        else {}
    )
    memory_slots = (
        evidence.get("memory_slots")
        if isinstance(evidence.get("memory_slots"), Mapping)
        else {}
    )
    triton_accounting = (
        evidence.get("training_window_triton_accounting")
        if isinstance(evidence.get("training_window_triton_accounting"), Mapping)
        else {}
    )
    return {
        "surface": "marulho_brain_installed_generation_repair_learning_summary.v1",
        "brain_surface": learning.get("surface"),
        "training_surface": report.get("surface"),
        "status": report.get("status"),
        "trace_event": (learning.get("trace") or {}).get("event")
        if isinstance(learning.get("trace"), Mapping)
        else None,
        "mutates_language_model_weights": bool(
            report.get("mutates_language_model_weights", False)
        ),
        "update_token_count": int(evidence.get("update_token_count", 0) or 0),
        "tokens_per_second": float(evidence.get("tokens_per_second", 0.0) or 0.0),
        "total_window_tokens_per_second": float(
            evidence.get("total_window_tokens_per_second", 0.0) or 0.0
        ),
        "new_domain_loss_delta": float(
            evidence.get("new_domain_loss_delta", 0.0) or 0.0
        ),
        "old_domain_forgetting": float(
            evidence.get("old_domain_forgetting", 0.0) or 0.0
        ),
        "general_replay_retention_delta": float(
            evidence.get("general_replay_retention_delta", 0.0) or 0.0
        ),
        "final_parameter_delta_l2": float(
            evidence.get("final_parameter_delta_l2", 0.0) or 0.0
        ),
        "device": evidence.get("device"),
        "memory_slots": {
            "enabled": bool(memory_slots.get("enabled", False)),
            "candidate_slots_scored": int(
                memory_slots.get("candidate_slots_scored", 0) or 0
            ),
            "runs_all_slots": bool(memory_slots.get("runs_all_slots", False)),
            "bounded_memory_slot_path": bool(
                memory_slots.get("bounded_memory_slot_path", False)
            ),
            "memory_slot_retrieval_backend": memory_slots.get(
                "memory_slot_retrieval_backend"
            ),
        },
        "training_window_triton_accounting": {
            "tracked_triton_kernel_used_names": list(
                triton_accounting.get("tracked_triton_kernel_used_names") or []
            ),
            "tracked_torch_fallback_call_count": int(
                triton_accounting.get("tracked_torch_fallback_call_count", 0) or 0
            ),
            "tracked_triton_failure_count": int(
                triton_accounting.get("tracked_triton_failure_count", 0) or 0
            ),
        },
        "rollback_evidence": dict(rollback),
    }


def _compact_checkpoint(
    *,
    path: str | Path,
    status: Mapping[str, Any],
    surface: str,
) -> dict[str, Any]:
    checkpoint = Path(path)
    language_model = _language_model_status(status)
    return {
        "surface": surface,
        "path": str(checkpoint),
        "sha256": _sha256_file(checkpoint) if checkpoint.is_file() else "",
        "restore_verified": bool(
            checkpoint.is_file()
            and status.get("active_language_path") == "marulho_lm_head"
            and bool(language_model.get("checkpointed_language_components", False))
        ),
        "active_language_path": status.get("active_language_path"),
        "language_model": dict(language_model),
        "last_trace": dict(status.get("last_trace") or {}),
    }


def _compact_sustained_generation(
    report: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    execution = (
        report.get("execution_evidence")
        if isinstance(report.get("execution_evidence"), Mapping)
        else {}
    )
    return {
        "surface": "marulho_brain_post_generation_repair_sustained_summary.v1",
        "enabled": True,
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
        "cuda_graph_burst_replay_count": int(
            execution.get("cuda_graph_burst_replay_count", 0) or 0
        ),
        "tracked_triton_kernel_used_names": list(
            execution.get("tracked_triton_kernel_used_names") or []
        ),
        "tracked_triton_kernel_failure_count": int(
            execution.get("tracked_triton_kernel_failure_count", 0) or 0
        ),
        "external_llm_used": bool(report.get("external_llm_used", False)),
        "service_owned_cognition": bool(report.get("service_owned_cognition", False)),
        "promotes_runtime_claim": bool(report.get("promotes_runtime_claim", False)),
    }


def build_language_brain_installed_generation_repair_evidence(
    *,
    output_path: str | Path,
    brain_checkpoint_path: str | Path,
    repaired_brain_checkpoint_path: str | Path | None = None,
    post_repair_sustained_output_path: str | Path | None = None,
    source_path: str | Path | None = None,
    replay_source_path: str | Path | None = None,
    prompt_cases: Sequence[LanguageGenerationPromptCase] | None = None,
    config: BrainInstalledGenerationRepairEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Learn on hard prompt replay through MarulhoBrain and rescore generation."""

    cfg = config or BrainInstalledGenerationRepairEvidenceConfig()
    output = Path(output_path)
    repaired_checkpoint = Path(
        repaired_brain_checkpoint_path
        or output.with_name(f"{output.stem}-repaired-brain.pt")
    )
    selected_device = _resolve_device(str(cfg.device))
    torch.manual_seed(int(cfg.seed))
    cuda_math_policy: dict[str, Any] = {"before": None}
    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "report_status": "partial",
        "output_path": str(output),
        "brain_checkpoint_path": str(brain_checkpoint_path),
        "repaired_brain_checkpoint_path": str(repaired_checkpoint),
        "runtime_owner": "MarulhoBrain",
        "requested_device": str(selected_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": asdict(cfg),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "service_owned_cognition": False,
        "status_read_mutation": False,
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }
    exception: BaseException | None = None
    try:
        cuda_math_policy = _apply_cuda_math_policy(selected_device, cfg)
        checkpoint = Path(brain_checkpoint_path)
        if not checkpoint.is_file():
            report.update(
                {
                    "status": "blocked_brain_installed_generation_repair_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_checkpoint_missing",
                }
            )
            return report
        brain = MarulhoBrain.load(checkpoint)
        restored_status = brain.status()
        language_model = _language_model_status(restored_status)
        if not bool(language_model.get("available", False)):
            report.update(
                {
                    "status": "blocked_brain_installed_generation_repair_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_language_runtime_missing",
                }
            )
            return report
        before_status = brain.status()
        after_status = brain.status()
        status_read_mutation = _status_read_mutates(before_status, after_status)
        language_state = _language_state(brain)
        tokenizer = _language_tokenizer(language_state)
        tokenizer_hash_matches_installed = (
            tokenizer.vocabulary_hash() == language_model.get("tokenizer_hash")
        )
        source_text, source = _read_text(source_path)
        replay_text, replay_source = _read_text(replay_source_path)
        cases = tuple(prompt_cases or default_generation_coherence_prompt_cases(source_text))
        pre_generation = _score_generation(
            brain,
            tokenizer=tokenizer,
            prompt_cases=cases,
            checkpoint_path=checkpoint,
            config=cfg,
        )
        hard_text, hard_corpus = _build_hard_prompt_corpus(
            cases,
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            repeat=int(cfg.hard_prompt_repeat),
            context_chars=int(cfg.hard_prompt_context_chars),
        )
        new_split = build_language_model_splits(
            [hard_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=selected_device,
        )
        replay_split = build_language_model_splits(
            [replay_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=selected_device,
        )
        used_new_batches = _trim(new_split.train, int(cfg.max_new_batches))
        used_replay_batches = _trim(replay_split.train, int(cfg.max_replay_batches))
        used_old_eval_batches = _trim(replay_split.eval, int(cfg.max_old_eval_batches))
        used_new_eval_batches = _trim(new_split.eval, int(cfg.max_new_eval_batches))
        learning_config = LanguageContinualLearningConfig(
            learning_rate=float(cfg.learning_rate),
            max_steps=int(cfg.max_steps),
            replay_loss_weight=float(cfg.replay_loss_weight),
            max_grad_norm=float(cfg.max_grad_norm),
            gradient_clip_interval=max(0, int(cfg.gradient_clip_interval)),
            dense_adamw_backend=str(cfg.dense_adamw_backend),
            forgetting_tolerance=float(cfg.forgetting_tolerance),
            replay_retention_tolerance=float(cfg.replay_retention_tolerance),
            rollback_on_forgetting=bool(cfg.rollback_on_forgetting),
            collect_training_telemetry=bool(cfg.collect_training_telemetry),
        )
        learning = brain.learn_language_window(
            new_batches=used_new_batches,
            old_eval_batches=used_old_eval_batches,
            new_eval_batches=used_new_eval_batches,
            replay_batches=used_replay_batches,
            config=learning_config,
        )
        learned_status = brain.status()
        repaired_save = brain.save(repaired_checkpoint)
        repaired_restored = MarulhoBrain.load(repaired_save["path"])
        repaired_status = repaired_restored.status()
        post_generation = _score_generation(
            repaired_restored,
            tokenizer=tokenizer,
            prompt_cases=cases,
            checkpoint_path=repaired_checkpoint,
            config=cfg,
        )
        delta = _coherence_delta(pre_generation, post_generation)
        learning_summary = _learning_summary(learning)
        post_repair_sustained: dict[str, Any] = {
            "surface": "marulho_brain_post_generation_repair_sustained_summary.v1",
            "enabled": False,
            "reason": "post_repair_sustained_not_requested",
        }
        if bool(cfg.run_post_repair_sustained):
            sustained_output = Path(
                post_repair_sustained_output_path
                or output.with_name(f"{output.stem}-post-repair-sustained.json")
            )
            sustained_report = repaired_restored.generate_sustained_language(
                output_path=sustained_output,
                target_tokens=int(cfg.sustained_target_tokens),
                prompt=str(cfg.sustained_prompt),
                tick_tokens=int(cfg.sustained_tick_tokens),
                quantum_tokens=int(cfg.sustained_quantum_tokens),
                timeout_seconds=float(cfg.sustained_timeout_seconds),
                generation_repetition_penalty=float(cfg.generation_repetition_penalty),
                generation_no_repeat_ngram_size=int(
                    cfg.generation_no_repeat_ngram_size
                ),
                collect_environment=False,
            )
            post_repair_sustained = _compact_sustained_generation(
                sustained_report,
                output_path=sustained_output,
            )
        pre_summary = (
            pre_generation.get("summary")
            if isinstance(pre_generation.get("summary"), Mapping)
            else {}
        )
        post_summary = (
            post_generation.get("summary")
            if isinstance(post_generation.get("summary"), Mapping)
            else {}
        )
        update_tokens = int(learning_summary.get("update_token_count", 0) or 0)
        quality_repair_observed = (
            int(delta.get("passed_case_count_delta", 0) or 0) > 0
            or float(delta.get("mean_prefix_match_chars_delta", 0.0) or 0.0) > 0.0
        )
        repaired_checkpoint_report = _compact_checkpoint(
            path=repaired_checkpoint,
            status=repaired_status,
            surface="marulho_brain_generation_repair_checkpoint.v1",
        )
        success = (
            not bool(status_read_mutation)
            and bool(tokenizer_hash_matches_installed)
            and pre_generation.get("active_language_path") == "marulho_lm_head"
            and post_generation.get("active_language_path") == "marulho_lm_head"
            and pre_generation.get("owned_by_marulho") is True
            and post_generation.get("owned_by_marulho") is True
            and pre_generation.get("external_llm_used") is False
            and post_generation.get("external_llm_used") is False
            and learning.get("surface") == "marulho_brain_language_learning_window.v1"
            and learning_summary.get("trace_event") == "language_learn"
            and update_tokens > 0
            and bool(repaired_checkpoint_report["restore_verified"])
            and (
                not bool(cfg.run_post_repair_sustained)
                or bool(post_repair_sustained.get("success", False))
            )
        )
        report.update(
            {
                "status": (
                    "final"
                    if success
                    else "blocked_brain_installed_generation_repair_evidence"
                ),
                "report_status": "final" if success else "partial",
                "failure_reason": None if success else "evidence_gate_not_satisfied",
                "cuda_math_policy": cuda_math_policy,
                "brain_checkpoint": _compact_checkpoint(
                    path=checkpoint,
                    status=restored_status,
                    surface="marulho_brain_generation_repair_source_checkpoint.v1",
                ),
                "repaired_brain_checkpoint": repaired_checkpoint_report,
                "status_read": {
                    "surface": "marulho_brain_status_read_check.v1",
                    "mutates_runtime_state": bool(status_read_mutation),
                    "before_token_count": int(before_status.get("token_count", 0) or 0),
                    "after_token_count": int(after_status.get("token_count", 0) or 0),
                    "active_language_path": before_status.get("active_language_path"),
                },
                "tokenizer": {
                    "surface": "marulho_brain_generation_repair_tokenizer_check.v1",
                    "tokenizer_hash": tokenizer.vocabulary_hash(),
                    "tokenizer_hash_matches_installed_runtime": bool(
                        tokenizer_hash_matches_installed
                    ),
                    "used_for_batch_tokenization_only": True,
                },
                "source": {
                    "path": source,
                    "character_count": len(source_text),
                },
                "replay_source": {
                    "path": replay_source,
                    "character_count": len(replay_text),
                },
                "hard_prompt_corpus": hard_corpus,
                "split": {
                    "new": new_split.report,
                    "replay": replay_split.report,
                    "used_new_train_batches": len(used_new_batches),
                    "used_replay_batches": len(used_replay_batches),
                    "used_old_eval_batches": len(used_old_eval_batches),
                    "used_new_eval_batches": len(used_new_eval_batches),
                },
                "continual_learning_config": asdict(learning_config),
                "learning_window": dict(learning),
                "learning_summary": learning_summary,
                "pre_generation_coherence": dict(pre_generation),
                "post_generation_coherence": dict(post_generation),
                "generation_quality_delta": dict(delta),
                "post_repair_sustained_window": post_repair_sustained,
                "repaired_brain": {
                    "surface": "marulho_brain_generation_repair_restore_summary.v1",
                    "active_language_path": repaired_status.get(
                        "active_language_path"
                    ),
                    "device": repaired_status.get("device"),
                    "language_model": dict(_language_model_status(repaired_status)),
                    "last_trace": dict(repaired_status.get("last_trace") or {}),
                },
                "active_language_path": repaired_status.get("active_language_path"),
                "status_read_mutation": bool(status_read_mutation),
                "update_token_count": update_tokens,
                "tokens_per_second": float(
                    learning_summary.get("tokens_per_second", 0.0) or 0.0
                ),
                "total_window_tokens_per_second": float(
                    learning_summary.get("total_window_tokens_per_second", 0.0)
                    or 0.0
                ),
            }
        )
        report["promotion_gate"] = {
            "surface": "marulho_language_brain_installed_generation_repair_gate.v1",
            "loaded_installed_brain_checkpoint": True,
            "brain_checkpoint_restore_verified": bool(
                report["brain_checkpoint"]["restore_verified"]
            ),
            "batch_tokenizer_matches_installed_runtime": bool(
                tokenizer_hash_matches_installed
            ),
            "status_read_mutation_absent": not bool(status_read_mutation),
            "pre_generation_runs_through_marulho_brain": bool(
                pre_generation.get("owned_by_marulho") is True
                and pre_generation.get("active_language_path") == "marulho_lm_head"
            ),
            "learning_runs_through_marulho_brain": bool(
                learning.get("surface") == "marulho_brain_language_learning_window.v1"
            ),
            "language_learn_trace_recorded": learning_summary.get("trace_event")
            == "language_learn",
            "records_actual_continual_learning": update_tokens > 0,
            "repaired_brain_checkpoint_restore_verified": bool(
                repaired_checkpoint_report["restore_verified"]
            ),
            "post_generation_runs_through_marulho_brain": bool(
                post_generation.get("owned_by_marulho") is True
                and post_generation.get("active_language_path") == "marulho_lm_head"
            ),
            "case_count": int(post_summary.get("case_count", 0) or 0),
            "pre_passed_case_count": int(pre_summary.get("passed_case_count", 0) or 0),
            "post_passed_case_count": int(
                post_summary.get("passed_case_count", 0) or 0
            ),
            "passed_case_count_delta": int(
                delta.get("passed_case_count_delta", 0) or 0
            ),
            "pre_case_pass_rate": float(pre_summary.get("case_pass_rate", 0.0) or 0.0),
            "post_case_pass_rate": float(
                post_summary.get("case_pass_rate", 0.0) or 0.0
            ),
            "mean_prefix_match_chars_delta": float(
                delta.get("mean_prefix_match_chars_delta", 0.0) or 0.0
            ),
            "quality_repair_observed": bool(quality_repair_observed),
            "post_repair_sustained_enabled": bool(cfg.run_post_repair_sustained),
            "post_repair_sustained_target_reached": bool(
                post_repair_sustained.get("success", False)
            ),
            "post_repair_sustained_8192_boundary_reached": int(
                post_repair_sustained.get("token_delta", 0) or 0
            )
            >= 8192,
            "post_repair_sustained_131072_boundary_reached": int(
                post_repair_sustained.get("token_delta", 0) or 0
            )
            >= 131072,
            "post_repair_sustained_524288_boundary_reached": int(
                post_repair_sustained.get("token_delta", 0) or 0
            )
            >= 524288,
            "external_llm_absent": not bool(report.get("external_llm_used", False)),
            "service_owned_cognition_absent": not bool(
                report.get("service_owned_cognition", False)
            ),
            "ready_for_runtime_claim_review": False,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
        }
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
        before_policy = cuda_math_policy.get("before")
        if isinstance(before_policy, Mapping):
            _restore_cuda_math_policy(dict(before_policy))
        if exception is not None:
            report.setdefault(
                "promotion_gate",
                {
                    "surface": (
                        "marulho_language_brain_installed_generation_repair_gate.v1"
                    ),
                    "ready_for_runtime_claim_review": False,
                    "promotes_runtime_claim": False,
                    "promotes_generation_quality_claim": False,
                },
            )
        write_json_report_with_readme(output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain-checkpoint", type=Path, required=True)
    parser.add_argument("--repaired-brain-checkpoint", type=Path, default=None)
    parser.add_argument("--post-repair-sustained-output", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--replay-source", type=Path, default=None)
    parser.add_argument(
        "--prompt-case",
        action="append",
        default=[],
        help=(
            "Prompt case as prompt|max_new_tokens|min_prefix_chars|min_prefix_fraction. "
            "Defaults to the standard MARULHO prompt suite."
        ),
    )
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--hard-prompt-repeat", type=int, default=16)
    parser.add_argument("--hard-prompt-context-chars", type=int, default=192)
    parser.add_argument("--max-new-batches", type=int, default=8)
    parser.add_argument("--max-replay-batches", type=int, default=8)
    parser.add_argument("--max-old-eval-batches", type=int, default=8)
    parser.add_argument("--max-new-eval-batches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--replay-loss-weight", type=float, default=1.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=8)
    parser.add_argument("--dense-adamw-backend", default="default")
    parser.add_argument("--forgetting-tolerance", type=float, default=100.0)
    parser.add_argument("--replay-retention-tolerance", type=float, default=100.0)
    parser.add_argument("--rollback-on-forgetting", action="store_true")
    parser.add_argument("--collect-training-telemetry", action="store_true")
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--run-post-repair-sustained", action="store_true")
    parser.add_argument("--sustained-target-tokens", type=int, default=8192)
    parser.add_argument("--sustained-tick-tokens", type=int, default=128)
    parser.add_argument("--sustained-quantum-tokens", type=int, default=16)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--sustained-prompt", default="MARULHO")
    parser.add_argument("--cuda-allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cuda-float32-matmul-precision", default="high")
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    source_text, _source = _read_text(args.source)
    cases = (
        tuple(_prompt_case_from_arg(value, source_text=source_text) for value in args.prompt_case)
        if args.prompt_case
        else default_generation_coherence_prompt_cases(source_text)
    )
    report = build_language_brain_installed_generation_repair_evidence(
        output_path=args.output,
        brain_checkpoint_path=args.brain_checkpoint,
        repaired_brain_checkpoint_path=args.repaired_brain_checkpoint,
        post_repair_sustained_output_path=args.post_repair_sustained_output,
        source_path=args.source,
        replay_source_path=args.replay_source,
        prompt_cases=cases,
        config=BrainInstalledGenerationRepairEvidenceConfig(
            sequence_length=int(args.sequence_length),
            stride=int(args.stride),
            batch_size=int(args.batch_size),
            eval_fraction=float(args.eval_fraction),
            hard_prompt_repeat=max(1, int(args.hard_prompt_repeat)),
            hard_prompt_context_chars=max(0, int(args.hard_prompt_context_chars)),
            max_new_batches=int(args.max_new_batches),
            max_replay_batches=int(args.max_replay_batches),
            max_old_eval_batches=int(args.max_old_eval_batches),
            max_new_eval_batches=int(args.max_new_eval_batches),
            learning_rate=float(args.learning_rate),
            max_steps=int(args.max_steps),
            replay_loss_weight=float(args.replay_loss_weight),
            max_grad_norm=float(args.max_grad_norm),
            gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
            dense_adamw_backend=str(args.dense_adamw_backend),
            forgetting_tolerance=float(args.forgetting_tolerance),
            replay_retention_tolerance=float(args.replay_retention_tolerance),
            rollback_on_forgetting=bool(args.rollback_on_forgetting),
            collect_training_telemetry=bool(args.collect_training_telemetry),
            min_case_pass_rate=float(args.min_case_pass_rate),
            generation_repetition_penalty=max(
                1.0,
                float(args.generation_repetition_penalty),
            ),
            generation_no_repeat_ngram_size=max(
                0,
                int(args.generation_no_repeat_ngram_size),
            ),
            run_post_repair_sustained=bool(args.run_post_repair_sustained),
            sustained_target_tokens=max(0, int(args.sustained_target_tokens)),
            sustained_tick_tokens=max(1, int(args.sustained_tick_tokens)),
            sustained_quantum_tokens=max(1, int(args.sustained_quantum_tokens)),
            sustained_timeout_seconds=float(args.sustained_timeout_seconds),
            sustained_prompt=str(args.sustained_prompt),
            cuda_allow_tf32=bool(args.cuda_allow_tf32),
            cuda_float32_matmul_precision=str(args.cuda_float32_matmul_precision),
            seed=int(args.seed),
            device=str(args.device),
        ),
    )
    return 0 if report.get("report_status") == "final" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
