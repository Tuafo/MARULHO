"""Evidence runner for reviewed LM checkpoints learning through MarulhoBrain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.evaluation.language_continual_learning_experiment import (
    DEFAULT_NEW_CORPUS,
    DEFAULT_OLD_CORPUS,
    _read_text,
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
    load_language_model_checkpoint,
)


SURFACE = "marulho_language_brain_installed_continual_learning_evidence.v1"
ARTIFACT_KIND = "marulho_language_brain_installed_continual_learning_evidence"


@dataclass(frozen=True)
class BrainInstalledContinualLearningEvidenceConfig:
    sequence_length: int = 64
    stride: int = 16
    batch_size: int = 16
    eval_fraction: float = 0.2
    max_old_eval_batches: int = 22
    max_new_eval_batches: int = 27
    max_new_batches: int = 4
    max_replay_batches: int = 4
    learning_rate: float = 2e-3
    max_steps: int = 64
    replay_loss_weight: float = 0.25
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 8
    recurrent_gradient_horizon: int | None = None
    dense_adamw_backend: str = "default"
    forgetting_tolerance: float = 100.0
    replay_retention_tolerance: float = 100.0
    rollback_on_forgetting: bool = False
    collect_training_telemetry: bool = False
    sampled_vocab_ce_triton_training: bool = False
    memory_slots_triton_training: bool = False
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
    run_post_learning_sustained: bool = True
    sustained_target_tokens: int = 8192
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 600.0
    sustained_prompt: str = "MARULHO"
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3
    seed: int = 20260705
    device: str = "auto"


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


def _trim(batches: Sequence[LanguageBatch], limit: int) -> tuple[LanguageBatch, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _status_read_mutates(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return (
        int(before.get("token_count", 0) or 0) != int(after.get("token_count", 0) or 0)
        or int(before.get("queued_tokens", 0) or 0)
        != int(after.get("queued_tokens", 0) or 0)
        or str(before.get("active_language_path") or "")
        != str(after.get("active_language_path") or "")
    )


def _training_backend_policy(
    config: BrainInstalledContinualLearningEvidenceConfig,
) -> dict[str, Any]:
    env_names = {
        "sampled_vocab_ce_triton_training": (
            "MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING"
        ),
        "memory_slots_triton_training": (
            "MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING"
        ),
    }
    requested = {
        "sampled_vocab_ce_triton_training": bool(
            config.sampled_vocab_ce_triton_training
        ),
        "memory_slots_triton_training": bool(config.memory_slots_triton_training),
    }
    previous = {key: os.environ.get(name) for key, name in env_names.items()}
    for key, name in env_names.items():
        os.environ[name] = "1" if requested[key] else "0"
    return {
        "surface": "marulho_language_continual_training_backend_policy.v1",
        "scope": "brain_installed_continual_learning_evidence_process",
        "env_names": env_names,
        "previous_env": previous,
        "requested": requested,
        "active": {key: os.environ.get(name) for key, name in env_names.items()},
    }


def _restore_training_backend_policy(policy: Mapping[str, Any]) -> None:
    env_names = policy.get("env_names")
    previous = policy.get("previous_env")
    if not isinstance(env_names, Mapping) or not isinstance(previous, Mapping):
        return
    for key, name in env_names.items():
        if not isinstance(name, str):
            continue
        old_value = previous.get(key)
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(old_value)


def _compact_checkpoint(
    *,
    surface: str,
    save_report: Mapping[str, Any],
    status: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(str(save_report["path"]))
    language_model = (
        status.get("language_model")
        if isinstance(status.get("language_model"), Mapping)
        else {}
    )
    return {
        "surface": surface,
        "path": str(path),
        "sha256": _sha256_file(path) if path.is_file() else "",
        "save_report": dict(save_report),
        "restore_verified": bool(
            path.is_file()
            and status.get("active_language_path") == "marulho_lm_head"
            and bool(language_model.get("checkpointed_language_components", False))
        ),
        "active_language_path": status.get("active_language_path"),
        "continual_learning_window_count": int(
            language_model.get("continual_learning_window_count", 0) or 0
        ),
        "recurrent_gradient_horizon": (
            None
            if language_model.get("recurrent_gradient_horizon") is None
            else int(language_model.get("recurrent_gradient_horizon", 0) or 0)
        ),
        "state_block_recurrent_gradient_horizon": (
            None
            if language_model.get("state_block_recurrent_gradient_horizon") is None
            else int(
                language_model.get(
                    "state_block_recurrent_gradient_horizon",
                    0,
                )
                or 0
            )
        ),
        "last_trace": dict(status.get("last_trace") or {}),
    }


def _disabled_recurrent_horizon_override() -> dict[str, Any]:
    return {
        "surface": "marulho_brain_language_recurrent_horizon_override.v1",
        "enabled": False,
        "applied": False,
        "reason": "recurrent_gradient_horizon_override_not_requested",
        "mutates_language_model_config": False,
        "mutates_language_model_weights": False,
        "runtime_owner": "MarulhoBrain",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "service_owned_cognition": False,
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
        "surface": "marulho_brain_post_learning_sustained_generation_summary.v1",
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
    torch_fallback_calls = _tracked_torch_fallback_calls(triton_accounting)
    return {
        "surface": "marulho_brain_installed_continual_learning_summary.v1",
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
        "optimizer_policy": evidence.get("optimizer_policy"),
        "dense_adamw_backend": evidence.get("dense_adamw_backend"),
        "metric_readback_mode": evidence.get("metric_readback_mode"),
        "per_step_metric_cpu_sync": bool(
            evidence.get("per_step_metric_cpu_sync", True)
        ),
        "batch_device_staging": dict(
            evidence.get("batch_device_staging")
            if isinstance(evidence.get("batch_device_staging"), Mapping)
            else {}
        ),
        "measured_update_loop_caller_device_transfer_calls": int(
            evidence.get("measured_update_loop_caller_device_transfer_calls", 0) or 0
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
            "tracked_torch_fallback_calls": int(torch_fallback_calls),
            "tracked_torch_fallback_call_count": int(
                torch_fallback_calls
            ),
            "tracked_triton_failure_count": int(
                triton_accounting.get("tracked_triton_failure_count", 0) or 0
            ),
        },
        "rollback_evidence": dict(rollback),
    }


def _tracked_torch_fallback_calls(accounting: Mapping[str, Any]) -> int:
    return int(
        accounting.get("tracked_torch_fallback_calls")
        or accounting.get("tracked_torch_fallback_call_count")
        or 0
    )


def _base_report(
    *,
    output_path: str | Path,
    promotion_review_path: str | Path | None,
    language_checkpoint_path: str | Path | None,
    language_checkpoint_sha256: str | None,
    pre_learning_brain_checkpoint_path: str | Path,
    learned_brain_checkpoint_path: str | Path,
    device: str,
    config: BrainInstalledContinualLearningEvidenceConfig,
) -> dict[str, Any]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "report_status": "partial",
        "output_path": str(output_path),
        "promotion_review_path": (
            None if promotion_review_path is None else str(promotion_review_path)
        ),
        "language_checkpoint_path": (
            None if language_checkpoint_path is None else str(language_checkpoint_path)
        ),
        "language_checkpoint_sha256": language_checkpoint_sha256,
        "checkpoint_install_source": (
            "promotion_review"
            if promotion_review_path is not None
            else "direct_checkpoint_review"
            if language_checkpoint_path is not None
            else "missing_checkpoint_install_source"
        ),
        "pre_learning_brain_checkpoint_path": str(pre_learning_brain_checkpoint_path),
        "learned_brain_checkpoint_path": str(learned_brain_checkpoint_path),
        "runtime_owner": "MarulhoBrain",
        "requested_device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "config": asdict(config),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "service_owned_cognition": False,
        "status_read_mutation": False,
        "promotes_runtime_claim": False,
    }


def build_language_brain_installed_continual_learning_evidence(
    *,
    output_path: str | Path,
    promotion_review_path: str | Path | None = None,
    language_checkpoint_path: str | Path | None = None,
    language_checkpoint_sha256: str | None = None,
    operator_approved: bool,
    operator_id: str | None = None,
    approval_note: str = "",
    artifact_base_dir: str | Path | None = None,
    pre_learning_brain_checkpoint_path: str | Path | None = None,
    learned_brain_checkpoint_path: str | Path | None = None,
    post_learning_sustained_output_path: str | Path | None = None,
    old_corpus_path: str | Path | None = None,
    new_corpus_path: str | Path | None = None,
    config: BrainInstalledContinualLearningEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Install a reviewed parent, learn through MarulhoBrain, and verify restore."""

    cfg = config or BrainInstalledContinualLearningEvidenceConfig()
    output = Path(output_path)
    pre_checkpoint = Path(
        pre_learning_brain_checkpoint_path
        or output.with_name(f"{output.stem}-pre-learning-brain.pt")
    )
    learned_checkpoint = Path(
        learned_brain_checkpoint_path
        or output.with_name(f"{output.stem}-learned-brain.pt")
    )
    selected_device = str(_resolve_device(str(cfg.device)))
    torch.manual_seed(int(cfg.seed))
    training_backend_policy = _training_backend_policy(cfg)
    cuda_math_policy: dict[str, Any] = {"before": None}
    report = _base_report(
        output_path=output,
        promotion_review_path=promotion_review_path,
        language_checkpoint_path=language_checkpoint_path,
        language_checkpoint_sha256=language_checkpoint_sha256,
        pre_learning_brain_checkpoint_path=pre_checkpoint,
        learned_brain_checkpoint_path=learned_checkpoint,
        device=selected_device,
        config=cfg,
    )
    exception: BaseException | None = None
    try:
        cuda_math_policy = _apply_cuda_math_policy(torch.device(selected_device), cfg)
        brain = MarulhoBrain.fresh(_tiny_brain_config(device=selected_device))
        if promotion_review_path is not None:
            install = brain.install_language_checkpoint_from_promotion_review(
                promotion_review_path,
                operator_approved=bool(operator_approved),
                operator_id=operator_id,
                approval_note=approval_note,
                artifact_base_dir=artifact_base_dir,
            )
        elif language_checkpoint_path is not None:
            install = brain.install_language_checkpoint_from_direct_review(
                language_checkpoint_path,
                expected_sha256=str(language_checkpoint_sha256 or ""),
                operator_approved=bool(operator_approved),
                operator_id=operator_id,
                approval_note=approval_note,
                artifact_base_dir=artifact_base_dir,
            )
        else:
            install = {
                "surface": "marulho_brain_language_checkpoint_installation_missing_source.v1",
                "status": "blocked_language_checkpoint_installation",
                "installed": False,
                "missing_evidence": [
                    "promotion_review_or_language_checkpoint_required"
                ],
                "mutates_runtime_state": False,
                "service_owned_cognition": False,
                "external_llm_used": False,
                "promotes_runtime_claim": False,
            }
        report["installation"] = dict(install)
        if not bool(install.get("installed")):
            report.update(
                {
                    "status": "blocked_brain_installed_continual_learning_evidence",
                    "report_status": "partial",
                    "failure_reason": "language_checkpoint_installation_blocked",
                }
            )
            return report

        candidate_checkpoint = (
            install.get("candidate_checkpoint")
            if isinstance(install.get("candidate_checkpoint"), Mapping)
            else {}
        )
        candidate_path = Path(str(candidate_checkpoint.get("resolved_path") or ""))
        _candidate_model, tokenizer, candidate_metadata = load_language_model_checkpoint(
            candidate_path,
            map_location="cpu",
        )
        del _candidate_model
        recurrent_horizon_override = _disabled_recurrent_horizon_override()
        if cfg.recurrent_gradient_horizon is not None:
            recurrent_horizon_override = brain.set_language_recurrent_gradient_horizon(
                int(cfg.recurrent_gradient_horizon)
            )
        installed_status = brain.status()
        installed_language = (
            installed_status.get("language_model")
            if isinstance(installed_status.get("language_model"), Mapping)
            else {}
        )
        tokenizer_hash_matches_installed = (
            tokenizer.vocabulary_hash() == installed_language.get("tokenizer_hash")
        )

        pre_save = brain.save(pre_checkpoint)
        pre_restored = MarulhoBrain.load(pre_save["path"])
        before_status = pre_restored.status()
        after_status = pre_restored.status()
        status_read_mutation = _status_read_mutates(before_status, after_status)
        pre_checkpoint_report = _compact_checkpoint(
            surface="marulho_brain_pre_learning_installed_checkpoint.v1",
            save_report=pre_save,
            status=before_status,
        )

        old_text, old_source = _read_text(old_corpus_path, default=DEFAULT_OLD_CORPUS)
        new_text, new_source = _read_text(new_corpus_path, default=DEFAULT_NEW_CORPUS)
        old_split = build_language_model_splits(
            [old_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=torch.device(selected_device),
        )
        new_split = build_language_model_splits(
            [new_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=torch.device(selected_device),
        )
        used_new_batches = _trim(new_split.train, int(cfg.max_new_batches))
        used_replay_batches = _trim(old_split.train, int(cfg.max_replay_batches))
        used_old_eval_batches = _trim(old_split.eval, int(cfg.max_old_eval_batches))
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
        learning = pre_restored.learn_language_window(
            new_batches=used_new_batches,
            old_eval_batches=used_old_eval_batches,
            new_eval_batches=used_new_eval_batches,
            replay_batches=used_replay_batches,
            config=learning_config,
        )
        learned_status = pre_restored.status()
        learned_save = pre_restored.save(learned_checkpoint)
        learned_restored = MarulhoBrain.load(learned_save["path"])
        learned_restored_status = learned_restored.status()
        learned_checkpoint_report = _compact_checkpoint(
            surface="marulho_brain_post_learning_checkpoint.v1",
            save_report=learned_save,
            status=learned_restored_status,
        )
        horizon_override_requested = cfg.recurrent_gradient_horizon is not None
        requested_horizon = (
            int(cfg.recurrent_gradient_horizon) if horizon_override_requested else None
        )
        pre_horizon_matches = (
            not horizon_override_requested
            or (
                pre_checkpoint_report.get("recurrent_gradient_horizon")
                == requested_horizon
                and pre_checkpoint_report.get("state_block_recurrent_gradient_horizon")
                == requested_horizon
            )
        )
        learned_horizon_matches = (
            not horizon_override_requested
            or (
                learned_checkpoint_report.get("recurrent_gradient_horizon")
                == requested_horizon
                and learned_checkpoint_report.get(
                    "state_block_recurrent_gradient_horizon"
                )
                == requested_horizon
            )
        )
        recurrent_horizon_override_applied = (
            not horizon_override_requested
            or (
                bool(recurrent_horizon_override.get("applied", False))
                and int(
                    recurrent_horizon_override.get(
                        "current_recurrent_gradient_horizon",
                        -1,
                    )
                )
                == requested_horizon
                and int(
                    recurrent_horizon_override.get(
                        "current_state_block_recurrent_gradient_horizon",
                        -1,
                    )
                )
                == requested_horizon
            )
        )
        post_learning_sustained: dict[str, Any] = {
            "surface": "marulho_brain_post_learning_sustained_generation_summary.v1",
            "enabled": False,
            "reason": "post_learning_sustained_not_requested",
        }
        if bool(cfg.run_post_learning_sustained):
            sustained_output = Path(
                post_learning_sustained_output_path
                or output.with_name(f"{output.stem}-post-learning-sustained.json")
            )
            sustained_generation = learned_restored.generate_sustained_language(
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
            post_learning_sustained = _compact_sustained_generation(
                sustained_generation,
                output_path=sustained_output,
            )

        learning_report = (
            learning.get("report") if isinstance(learning.get("report"), Mapping) else {}
        )
        evidence = (
            learning_report.get("learning_evidence")
            if isinstance(learning_report.get("learning_evidence"), Mapping)
            else {}
        )
        learning_summary = _learning_summary(learning)
        update_tokens = int(evidence.get("update_token_count", 0) or 0)
        learning_surface_ok = learning.get("surface") == (
            "marulho_brain_language_learning_window.v1"
        )
        post_restore_learning_count = int(
            (
                learned_restored_status.get("language_model", {})
                if isinstance(learned_restored_status.get("language_model"), Mapping)
                else {}
            ).get("continual_learning_window_count", 0)
            or 0
        )
        success = (
            bool(install.get("installed"))
            and bool(pre_checkpoint_report["restore_verified"])
            and bool(learned_checkpoint_report["restore_verified"])
            and not bool(status_read_mutation)
            and bool(tokenizer_hash_matches_installed)
            and bool(learning_surface_ok)
            and update_tokens > 0
            and learning_summary["trace_event"] == "language_learn"
            and post_restore_learning_count >= 1
            and bool(recurrent_horizon_override_applied)
            and bool(pre_horizon_matches)
            and bool(learned_horizon_matches)
            and (
                not bool(cfg.run_post_learning_sustained)
                or bool(post_learning_sustained.get("success", False))
            )
        )
        report.update(
            {
                "status": (
                    "final"
                    if success
                    else "blocked_brain_installed_continual_learning_evidence"
                ),
                "report_status": "final" if success else "partial",
                "failure_reason": None if success else "evidence_gate_not_satisfied",
                "cuda_math_policy": cuda_math_policy,
                "training_backend_policy": training_backend_policy,
                "candidate_checkpoint": {
                    "path": str(candidate_path),
                    "sha256": _sha256_file(candidate_path)
                    if candidate_path.is_file()
                    else "",
                    "metadata": dict(candidate_metadata),
                    "tokenizer_hash": tokenizer.vocabulary_hash(),
                    "tokenizer_hash_matches_installed_runtime": bool(
                        tokenizer_hash_matches_installed
                    ),
                    "used_for_batch_tokenization_only": True,
                },
                "pre_learning_brain_checkpoint": pre_checkpoint_report,
                "learned_brain_checkpoint": learned_checkpoint_report,
                "recurrent_gradient_horizon_override": recurrent_horizon_override,
                "status_read": {
                    "surface": "marulho_brain_status_read_check.v1",
                    "mutates_runtime_state": bool(status_read_mutation),
                    "before_token_count": int(before_status.get("token_count", 0) or 0),
                    "after_token_count": int(after_status.get("token_count", 0) or 0),
                    "active_language_path": before_status.get("active_language_path"),
                },
                "split": {
                    "old": old_split.report,
                    "new": new_split.report,
                    "used_new_train_batches": len(used_new_batches),
                    "used_replay_batches": len(used_replay_batches),
                    "used_old_eval_batches": len(used_old_eval_batches),
                    "used_new_eval_batches": len(used_new_eval_batches),
                },
                "corpus": {
                    "old_source": old_source,
                    "new_source": new_source,
                    "old_character_count": len(old_text),
                    "new_character_count": len(new_text),
                },
                "continual_learning_config": asdict(learning_config),
                "learning_window": dict(learning),
                "learning_summary": learning_summary,
                "learning_evidence": dict(evidence),
                "post_learning_sustained_window": post_learning_sustained,
                "learned_brain": {
                    "surface": "marulho_brain_post_learning_restore_summary.v1",
                    "active_language_path": learned_restored_status.get(
                        "active_language_path"
                    ),
                    "device": learned_restored_status.get("device"),
                    "language_model": dict(
                        learned_restored_status.get("language_model") or {}
                    ),
                    "last_trace": dict(learned_restored_status.get("last_trace") or {}),
                },
                "active_language_path": learned_restored_status.get(
                    "active_language_path"
                ),
                "status_read_mutation": bool(status_read_mutation),
                "update_token_count": update_tokens,
                "tokens_per_second": float(
                    evidence.get("tokens_per_second", 0.0) or 0.0
                ),
                "total_window_tokens_per_second": float(
                    evidence.get("total_window_tokens_per_second", 0.0) or 0.0
                ),
            }
        )
        report["promotion_gate"] = {
            "surface": "marulho_language_brain_installed_continual_learning_gate.v1",
            "installed_reviewed_checkpoint": bool(install.get("installed")),
            "candidate_checkpoint_hash_verified": bool(
                candidate_checkpoint.get("hash_verified", False)
            ),
            "batch_tokenizer_matches_installed_runtime": bool(
                tokenizer_hash_matches_installed
            ),
            "pre_learning_brain_checkpoint_restore_verified": bool(
                pre_checkpoint_report["restore_verified"]
            ),
            "learning_runs_through_marulho_brain": bool(learning_surface_ok),
            "language_learn_trace_recorded": learning_summary["trace_event"]
            == "language_learn",
            "recurrent_gradient_horizon_override_requested": bool(
                horizon_override_requested
            ),
            "records_recurrent_gradient_horizon_override": (
                bool(horizon_override_requested)
                and bool(recurrent_horizon_override.get("applied", False))
            ),
            "recurrent_gradient_horizon_override_applied": bool(
                recurrent_horizon_override_applied
            ),
            "pre_learning_checkpoint_recurrent_horizon_matches": bool(
                pre_horizon_matches
            ),
            "learned_checkpoint_recurrent_horizon_matches": bool(
                learned_horizon_matches
            ),
            "records_actual_continual_learning": update_tokens > 0,
            "records_forgetting": "old_domain_forgetting" in evidence,
            "records_replay_retention": "general_replay_retention_delta" in evidence,
            "records_update_throughput": float(
                evidence.get("tokens_per_second", 0.0) or 0.0
            )
            > 0.0,
            "records_total_window_throughput": float(
                evidence.get("total_window_tokens_per_second", 0.0) or 0.0
            )
            > 0.0,
            "house_scale_524288_update_tokens_reached": update_tokens >= 524288,
            "learned_brain_checkpoint_restore_verified": bool(
                learned_checkpoint_report["restore_verified"]
            ),
            "post_learning_sustained_enabled": bool(cfg.run_post_learning_sustained),
            "post_learning_sustained_target_reached": bool(
                post_learning_sustained.get("success", False)
            ),
            "post_learning_sustained_8192_boundary_reached": int(
                post_learning_sustained.get("token_delta", 0) or 0
            )
            >= 8192,
            "post_learning_sustained_131072_boundary_reached": int(
                post_learning_sustained.get("token_delta", 0) or 0
            )
            >= 131072,
            "post_learning_sustained_524288_boundary_reached": int(
                post_learning_sustained.get("token_delta", 0) or 0
            )
            >= 524288,
            "status_read_mutation_absent": not bool(status_read_mutation),
            "external_llm_absent": not bool(report.get("external_llm_used", False)),
            "service_owned_cognition_absent": not bool(
                report.get("service_owned_cognition", False)
            ),
            "ready_for_runtime_claim_review": False,
            "promotes_runtime_claim": False,
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
        _restore_training_backend_policy(training_backend_policy)
        if exception is not None:
            report.setdefault(
                "promotion_gate",
                {
                    "surface": (
                        "marulho_language_brain_installed_continual_learning_gate.v1"
                    ),
                    "ready_for_runtime_claim_review": False,
                    "promotes_runtime_claim": False,
                },
            )
        write_json_report_with_readme(output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--promotion-review", type=Path, default=None)
    parser.add_argument("--language-checkpoint", type=Path, default=None)
    parser.add_argument("--language-checkpoint-sha256", type=str, default="")
    parser.add_argument("--artifact-base-dir", type=Path, default=Path.cwd())
    parser.add_argument("--pre-learning-brain-checkpoint", type=Path, default=None)
    parser.add_argument("--learned-brain-checkpoint", type=Path, default=None)
    parser.add_argument("--post-learning-sustained-output", type=Path, default=None)
    parser.add_argument("--operator-id", type=str, default="codex-operator")
    parser.add_argument("--approval-note", type=str, default="")
    parser.add_argument("--operator-approved", action="store_true")
    parser.add_argument("--old-corpus", type=Path, default=None)
    parser.add_argument("--new-corpus", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-old-eval-batches", type=int, default=22)
    parser.add_argument("--max-new-eval-batches", type=int, default=27)
    parser.add_argument("--max-new-batches", type=int, default=4)
    parser.add_argument("--max-replay-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--replay-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=8)
    parser.add_argument("--recurrent-gradient-horizon", type=int, default=None)
    parser.add_argument(
        "--dense-adamw-backend",
        choices=("default", "foreach", "fused"),
        default="default",
    )
    parser.add_argument("--forgetting-tolerance", type=float, default=100.0)
    parser.add_argument("--replay-retention-tolerance", type=float, default=100.0)
    parser.add_argument("--rollback-on-forgetting", action="store_true")
    parser.add_argument("--collect-training-telemetry", action="store_true")
    parser.add_argument("--sampled-vocab-ce-triton-training", action="store_true")
    parser.add_argument("--memory-slots-triton-training", action="store_true")
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument(
        "--cuda-float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--skip-post-learning-sustained", action="store_true")
    parser.add_argument("--sustained-target-tokens", type=int, default=8192)
    parser.add_argument("--sustained-tick-tokens", type=int, default=128)
    parser.add_argument("--sustained-quantum-tokens", type=int, default=16)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--sustained-prompt", type=str, default="MARULHO")
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = BrainInstalledContinualLearningEvidenceConfig(
        sequence_length=int(args.sequence_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        eval_fraction=float(args.eval_fraction),
        max_old_eval_batches=int(args.max_old_eval_batches),
        max_new_eval_batches=int(args.max_new_eval_batches),
        max_new_batches=int(args.max_new_batches),
        max_replay_batches=int(args.max_replay_batches),
        learning_rate=float(args.learning_rate),
        max_steps=int(args.max_steps),
        replay_loss_weight=float(args.replay_loss_weight),
        max_grad_norm=float(args.max_grad_norm),
        gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
        recurrent_gradient_horizon=(
            None
            if args.recurrent_gradient_horizon is None
            else max(0, int(args.recurrent_gradient_horizon))
        ),
        dense_adamw_backend=str(args.dense_adamw_backend),
        forgetting_tolerance=float(args.forgetting_tolerance),
        replay_retention_tolerance=float(args.replay_retention_tolerance),
        rollback_on_forgetting=bool(args.rollback_on_forgetting),
        collect_training_telemetry=bool(args.collect_training_telemetry),
        sampled_vocab_ce_triton_training=bool(
            args.sampled_vocab_ce_triton_training
        ),
        memory_slots_triton_training=bool(args.memory_slots_triton_training),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        cuda_float32_matmul_precision=str(args.cuda_float32_matmul_precision),
        run_post_learning_sustained=not bool(args.skip_post_learning_sustained),
        sustained_target_tokens=int(args.sustained_target_tokens),
        sustained_tick_tokens=int(args.sustained_tick_tokens),
        sustained_quantum_tokens=int(args.sustained_quantum_tokens),
        sustained_timeout_seconds=float(args.sustained_timeout_seconds),
        sustained_prompt=str(args.sustained_prompt),
        generation_repetition_penalty=max(
            1.0,
            float(args.generation_repetition_penalty),
        ),
        generation_no_repeat_ngram_size=max(
            0,
            int(args.generation_no_repeat_ngram_size),
        ),
        seed=int(args.seed),
        device=str(args.device),
    )
    report = build_language_brain_installed_continual_learning_evidence(
        output_path=args.output,
        promotion_review_path=args.promotion_review,
        language_checkpoint_path=args.language_checkpoint,
        language_checkpoint_sha256=str(args.language_checkpoint_sha256 or ""),
        operator_approved=bool(args.operator_approved),
        operator_id=args.operator_id,
        approval_note=args.approval_note,
        artifact_base_dir=args.artifact_base_dir,
        pre_learning_brain_checkpoint_path=args.pre_learning_brain_checkpoint,
        learned_brain_checkpoint_path=args.learned_brain_checkpoint,
        post_learning_sustained_output_path=args.post_learning_sustained_output,
        old_corpus_path=args.old_corpus,
        new_corpus_path=args.new_corpus,
        config=config,
    )
    return 0 if report.get("report_status") == "final" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
