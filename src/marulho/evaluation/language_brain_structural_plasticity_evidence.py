"""Evidence runner for structural mutation through an installed MarulhoBrain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.brain import MarulhoBrain
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import LanguageBatch, build_language_model_splits
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
)


SURFACE = "marulho_language_brain_installed_structural_plasticity_evidence.v1"
ARTIFACT_KIND = "marulho_language_brain_installed_structural_plasticity_evidence"

SUPPORTED_MUTATION_KINDS = (
    "route_bank_expansion",
    "memory_slot_expansion",
)


@dataclass(frozen=True)
class BrainInstalledStructuralPlasticityEvidenceConfig:
    mutation_kind: str = "route_bank_expansion"
    sequence_length: int = 64
    stride: int = 32
    batch_size: int = 16
    eval_fraction: float = 0.2
    max_eval_batches: int = 8
    max_eval_loss_delta: float = 100.0
    route_saturation_threshold: float = 0.5
    route_candidate_growth: int = 4
    max_route_candidate_count: int = 0
    memory_slot_growth: int = 4
    max_memory_slot_count: int = 0
    max_memory_slot_candidate_count: int = 8
    run_post_structure_sustained: bool = True
    sustained_target_tokens: int = 8192
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 600.0
    sustained_prompt: str = "MARULHO"
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3
    seed: int = 20260705


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: str | Path | None) -> tuple[str, str]:
    if path is None:
        return DEFAULT_CORPUS, "default_inline"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Structural-plasticity corpus is empty: {resolved}")
    return text, str(resolved)


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


def _language_state(brain: MarulhoBrain) -> Mapping[str, Any]:
    state = brain.export_state()
    language_state = state.get("language_model")
    if not isinstance(language_state, Mapping):
        raise RuntimeError("MARULHO language model runtime is not installed")
    return language_state


def _language_config(language_state: Mapping[str, Any]) -> dict[str, Any]:
    config = language_state.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("Installed language runtime has no checkpointed config")
    return dict(config)


def _language_tokenizer(language_state: Mapping[str, Any]) -> ByteLevelLanguageTokenizer:
    tokenizer_state = language_state.get("tokenizer")
    if not isinstance(tokenizer_state, Mapping):
        raise RuntimeError("Installed language runtime has no checkpointed tokenizer")
    return ByteLevelLanguageTokenizer.load_state_dict(tokenizer_state)


def _device_from_status(status: Mapping[str, Any]) -> torch.device:
    language = (
        status.get("language_model")
        if isinstance(status.get("language_model"), Mapping)
        else {}
    )
    device = str(language.get("device") or status.get("device") or "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Loaded brain reports CUDA language runtime but CUDA is unavailable")
    return resolved


def _structural_config(
    config: BrainInstalledStructuralPlasticityEvidenceConfig,
) -> LanguageStructuralPlasticityConfig:
    return LanguageStructuralPlasticityConfig(
        max_eval_loss_delta=float(config.max_eval_loss_delta),
        route_saturation_threshold=float(config.route_saturation_threshold),
        max_route_candidate_growth=max(1, int(config.route_candidate_growth)),
        max_route_candidate_count=max(0, int(config.max_route_candidate_count)),
        max_memory_slot_growth=max(1, int(config.memory_slot_growth)),
        max_memory_slot_count=max(0, int(config.max_memory_slot_count)),
        max_memory_slot_candidate_count=max(
            1,
            int(config.max_memory_slot_candidate_count),
        ),
    )


def _routing_evidence(
    *,
    mutation_kind: str,
    language_config: Mapping[str, Any],
    batch_token_count: int,
) -> dict[str, Any]:
    expert_count = max(0, int(language_config.get("expert_count", 0) or 0))
    active_expert_count = max(1, int(language_config.get("active_expert_count", 1) or 1))
    route_candidate_count = max(
        0,
        int(language_config.get("route_candidate_count", 0) or 0),
    )
    if str(mutation_kind) in {"memory_slot_expansion", "memory_slot"}:
        return {
            "surface": "marulho_language_memory_slots.v1",
            "memory_slot_pressure": True,
            "novel_concept_cluster": True,
            "replay_conflict": True,
            "candidate_rows_scored": max(1, int(batch_token_count)),
            "runs_all_columns": False,
        }
    return {
        "surface": "marulho_routed_language_experts.v1",
        "total_columns": expert_count,
        "active_columns": active_expert_count,
        "route_candidate_count": route_candidate_count,
        "output_candidate_count": active_expert_count,
        "candidate_rows_scored": max(1, int(batch_token_count)),
        "runs_all_columns": False,
        "route_bank_pressure": True,
    }


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
        "structural_transaction_count": int(
            language_model.get("structural_transaction_count", 0) or 0
        ),
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
        "surface": "marulho_brain_post_structure_sustained_generation_summary.v1",
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


def _transaction_summary(transaction: Mapping[str, Any]) -> dict[str, Any]:
    report = (
        transaction.get("report")
        if isinstance(transaction.get("report"), Mapping)
        else {}
    )
    mutation = (
        report.get("mutation") if isinstance(report.get("mutation"), Mapping) else {}
    )
    checkpoint = (
        report.get("checkpoint")
        if isinstance(report.get("checkpoint"), Mapping)
        else {}
    )
    rollback = (
        report.get("rollback_evidence")
        if isinstance(report.get("rollback_evidence"), Mapping)
        else {}
    )
    gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    return {
        "surface": "marulho_brain_installed_structural_transaction_summary.v1",
        "brain_surface": transaction.get("surface"),
        "training_surface": report.get("surface"),
        "status": report.get("status"),
        "trace_event": (transaction.get("trace") or {}).get("event")
        if isinstance(transaction.get("trace"), Mapping)
        else None,
        "applied": bool(report.get("applied", False)),
        "operator_approved": bool(report.get("operator_approved", False)),
        "proposal_kind": mutation.get("proposal_kind"),
        "source_expert_count": mutation.get("source_expert_count"),
        "target_expert_count": mutation.get("target_expert_count"),
        "source_route_candidate_count": mutation.get("source_route_candidate_count"),
        "target_route_candidate_count": mutation.get("target_route_candidate_count"),
        "route_bank_candidate_count_delta": mutation.get(
            "route_bank_candidate_count_delta"
        ),
        "source_memory_slot_count": mutation.get("source_memory_slot_count"),
        "target_memory_slot_count": mutation.get("target_memory_slot_count"),
        "memory_slot_count_delta": mutation.get("memory_slot_count_delta"),
        "target_memory_slot_candidate_count": mutation.get(
            "target_memory_slot_candidate_count"
        ),
        "target_active_memory_slot_count": mutation.get(
            "target_active_memory_slot_count"
        ),
        "checkpoint_restore_verified": bool(
            checkpoint.get("checkpoint_restore_verified", False)
        ),
        "rollback_verified": bool(rollback.get("rollback_verified", False)),
        "heldout_non_regression": bool(gate.get("heldout_non_regression", False)),
        "eligible_for_reviewed_structural_promotion": bool(
            gate.get("eligible_for_reviewed_structural_promotion", False)
        ),
    }


def _base_report(
    *,
    output_path: str | Path,
    brain_checkpoint_path: str | Path,
    pre_structural_brain_checkpoint_path: str | Path,
    post_structural_brain_checkpoint_path: str | Path,
    structural_baseline_checkpoint_path: str | Path,
    config: BrainInstalledStructuralPlasticityEvidenceConfig,
) -> dict[str, Any]:
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "report_status": "partial",
        "output_path": str(output_path),
        "brain_checkpoint_path": str(brain_checkpoint_path),
        "pre_structural_brain_checkpoint_path": str(
            pre_structural_brain_checkpoint_path
        ),
        "post_structural_brain_checkpoint_path": str(
            post_structural_brain_checkpoint_path
        ),
        "structural_baseline_checkpoint_path": str(
            structural_baseline_checkpoint_path
        ),
        "runtime_owner": "MarulhoBrain",
        "cuda_available": bool(torch.cuda.is_available()),
        "config": asdict(config),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "service_owned_cognition": False,
        "status_read_mutation": False,
        "promotes_runtime_claim": False,
    }


def build_language_brain_installed_structural_plasticity_evidence(
    *,
    output_path: str | Path,
    brain_checkpoint_path: str | Path,
    operator_approved: bool,
    pre_structural_brain_checkpoint_path: str | Path | None = None,
    post_structural_brain_checkpoint_path: str | Path | None = None,
    structural_baseline_checkpoint_path: str | Path | None = None,
    post_structure_sustained_output_path: str | Path | None = None,
    corpus_path: str | Path | None = None,
    config: BrainInstalledStructuralPlasticityEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Load an installed brain checkpoint, apply structural mutation, and verify."""

    cfg = config or BrainInstalledStructuralPlasticityEvidenceConfig()
    mutation_kind = str(cfg.mutation_kind)
    if mutation_kind not in SUPPORTED_MUTATION_KINDS:
        raise ValueError(f"Unsupported mutation kind: {mutation_kind}")

    output = Path(output_path)
    pre_checkpoint = Path(
        pre_structural_brain_checkpoint_path
        or output.with_name(f"{output.stem}-pre-structure-brain.pt")
    )
    post_checkpoint = Path(
        post_structural_brain_checkpoint_path
        or output.with_name(f"{output.stem}-post-structure-brain.pt")
    )
    baseline_checkpoint = Path(
        structural_baseline_checkpoint_path
        or output.with_name(f"{output.stem}-{mutation_kind.replace('_', '-')}-baseline.pt")
    )
    report = _base_report(
        output_path=output,
        brain_checkpoint_path=brain_checkpoint_path,
        pre_structural_brain_checkpoint_path=pre_checkpoint,
        post_structural_brain_checkpoint_path=post_checkpoint,
        structural_baseline_checkpoint_path=baseline_checkpoint,
        config=cfg,
    )
    exception: BaseException | None = None
    try:
        torch.manual_seed(int(cfg.seed))
        brain_checkpoint = Path(brain_checkpoint_path)
        if not brain_checkpoint.is_file():
            report.update(
                {
                    "status": "blocked_brain_installed_structural_plasticity_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_checkpoint_missing",
                }
            )
            return report

        brain = MarulhoBrain.load(brain_checkpoint)
        loaded_status = brain.status()
        loaded_language = (
            loaded_status.get("language_model")
            if isinstance(loaded_status.get("language_model"), Mapping)
            else {}
        )
        if not bool(loaded_language.get("available", False)):
            report.update(
                {
                    "status": "blocked_brain_installed_structural_plasticity_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_language_runtime_missing",
                }
            )
            return report

        pre_save = brain.save(pre_checkpoint)
        pre_restored = MarulhoBrain.load(pre_save["path"])
        before_status = pre_restored.status()
        after_status = pre_restored.status()
        status_read_mutation = _status_read_mutates(before_status, after_status)
        pre_checkpoint_report = _compact_checkpoint(
            surface="marulho_brain_pre_structure_installed_checkpoint.v1",
            save_report=pre_save,
            status=before_status,
        )
        language_state = _language_state(pre_restored)
        tokenizer = _language_tokenizer(language_state)
        language_config = _language_config(language_state)
        device = _device_from_status(before_status)
        tokenizer_hash_matches_installed = (
            tokenizer.vocabulary_hash() == loaded_language.get("tokenizer_hash")
        )

        corpus, corpus_source = _read_text(corpus_path)
        split = build_language_model_splits(
            [corpus],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=device,
        )
        eval_batches = _trim(split.eval, int(cfg.max_eval_batches))
        batch_token_count = sum(int(batch.target_ids.numel()) for batch in eval_batches)
        routing_evidence = _routing_evidence(
            mutation_kind=mutation_kind,
            language_config=language_config,
            batch_token_count=batch_token_count,
        )
        learning_evidence = {
            "surface": "marulho_brain_installed_structural_learning_context.v1",
            "source": "installed_brain_checkpoint_recent_learning_reports",
            "continual_learning_window_count": int(
                loaded_language.get("continual_learning_window_count", 0) or 0
            ),
            "last_continual_learning_available": loaded_language.get(
                "last_continual_learning"
            )
            is not None,
        }
        structural_config = _structural_config(cfg)
        proposal_status_before = pre_restored.status()
        proposal = pre_restored.propose_language_structure(
            routing_evidence=routing_evidence,
            learning_evidence=learning_evidence,
            config=structural_config,
            mutation_kind=mutation_kind,
        )
        proposal_status_after = pre_restored.status()
        proposal_mutated_status = _status_read_mutates(
            proposal_status_before,
            proposal_status_after,
        )
        transaction = pre_restored.apply_language_structure(
            proposal,
            eval_batches=eval_batches,
            checkpoint_path=baseline_checkpoint,
            operator_approved=bool(operator_approved),
            config=structural_config,
        )
        post_status = pre_restored.status()
        post_save = pre_restored.save(post_checkpoint)
        post_restored = MarulhoBrain.load(post_save["path"])
        post_restored_status = post_restored.status()
        post_checkpoint_report = _compact_checkpoint(
            surface="marulho_brain_post_structure_checkpoint.v1",
            save_report=post_save,
            status=post_restored_status,
        )
        transaction_summary = _transaction_summary(transaction)
        post_structure_sustained: dict[str, Any] = {
            "surface": "marulho_brain_post_structure_sustained_generation_summary.v1",
            "enabled": False,
            "reason": "post_structure_sustained_not_requested",
        }
        if bool(cfg.run_post_structure_sustained):
            sustained_output = Path(
                post_structure_sustained_output_path
                or output.with_name(f"{output.stem}-post-structure-sustained.json")
            )
            sustained_generation = post_restored.generate_sustained_language(
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
            post_structure_sustained = _compact_sustained_generation(
                sustained_generation,
                output_path=sustained_output,
            )

        structural_count_before = int(
            (
                before_status.get("language_model", {})
                if isinstance(before_status.get("language_model"), Mapping)
                else {}
            ).get("structural_transaction_count", 0)
            or 0
        )
        structural_count_after_restore = int(
            (
                post_restored_status.get("language_model", {})
                if isinstance(post_restored_status.get("language_model"), Mapping)
                else {}
            ).get("structural_transaction_count", 0)
            or 0
        )
        applied = bool(transaction_summary["applied"])
        success = (
            bool(pre_checkpoint_report["restore_verified"])
            and bool(post_checkpoint_report["restore_verified"])
            and not bool(status_read_mutation)
            and not bool(proposal_mutated_status)
            and bool(proposal.get("mutates_runtime_state") is False)
            and bool(tokenizer_hash_matches_installed)
            and transaction.get("surface")
            == "marulho_brain_language_structural_transaction.v1"
            and bool(applied)
            and transaction_summary["trace_event"] == "language_structure"
            and bool(transaction_summary["checkpoint_restore_verified"])
            and bool(transaction_summary["rollback_verified"])
            and structural_count_after_restore > structural_count_before
            and (
                not bool(cfg.run_post_structure_sustained)
                or bool(post_structure_sustained.get("success", False))
            )
        )
        report.update(
            {
                "status": (
                    "final"
                    if success
                    else "blocked_brain_installed_structural_plasticity_evidence"
                ),
                "report_status": "final" if success else "partial",
                "failure_reason": None if success else "evidence_gate_not_satisfied",
                "brain_checkpoint": {
                    "path": str(brain_checkpoint),
                    "sha256": _sha256_file(brain_checkpoint),
                    "active_language_path": loaded_status.get("active_language_path"),
                    "language_model": dict(loaded_language),
                },
                "pre_structural_brain_checkpoint": pre_checkpoint_report,
                "post_structural_brain_checkpoint": post_checkpoint_report,
                "status_read": {
                    "surface": "marulho_brain_status_read_check.v1",
                    "mutates_runtime_state": bool(status_read_mutation),
                    "before_token_count": int(before_status.get("token_count", 0) or 0),
                    "after_token_count": int(after_status.get("token_count", 0) or 0),
                    "active_language_path": before_status.get("active_language_path"),
                },
                "tokenizer": {
                    "surface": "marulho_brain_structural_tokenizer_check.v1",
                    "tokenizer_hash": tokenizer.vocabulary_hash(),
                    "tokenizer_hash_matches_installed_runtime": bool(
                        tokenizer_hash_matches_installed
                    ),
                    "used_for_batch_tokenization_only": True,
                },
                "corpus": {
                    "source": corpus_source,
                    "character_count": len(corpus),
                },
                "split": {
                    "report": split.report,
                    "used_eval_batches": len(eval_batches),
                    "used_eval_tokens": int(batch_token_count),
                },
                "structural_config": asdict(structural_config),
                "routing_evidence": routing_evidence,
                "learning_evidence": learning_evidence,
                "proposal": dict(proposal),
                "proposal_read_only": {
                    "surface": "marulho_brain_structural_proposal_read_only_check.v1",
                    "proposal_mutates_runtime_state_field": bool(
                        proposal.get("mutates_runtime_state", True)
                    ),
                    "status_changed_during_proposal": bool(proposal_mutated_status),
                    "mutates_runtime_state": bool(proposal_mutated_status),
                },
                "structural_transaction": dict(transaction),
                "structural_transaction_summary": transaction_summary,
                "post_structure_status": {
                    "surface": "marulho_brain_post_structure_status_summary.v1",
                    "active_language_path": post_status.get("active_language_path"),
                    "device": post_status.get("device"),
                    "language_model": dict(post_status.get("language_model") or {}),
                    "last_trace": dict(post_status.get("last_trace") or {}),
                },
                "post_structure_sustained_window": post_structure_sustained,
                "active_language_path": post_restored_status.get(
                    "active_language_path"
                ),
                "status_read_mutation": bool(status_read_mutation),
            }
        )
        report["promotion_gate"] = {
            "surface": "marulho_language_brain_installed_structural_gate.v1",
            "loaded_installed_brain_checkpoint": True,
            "batch_tokenizer_matches_installed_runtime": bool(
                tokenizer_hash_matches_installed
            ),
            "pre_structure_brain_checkpoint_restore_verified": bool(
                pre_checkpoint_report["restore_verified"]
            ),
            "proposal_runs_through_marulho_brain": True,
            "proposal_non_mutating": bool(
                proposal.get("mutates_runtime_state") is False
                and not bool(proposal_mutated_status)
            ),
            "structural_apply_runs_through_marulho_brain": transaction.get("surface")
            == "marulho_brain_language_structural_transaction.v1",
            "language_structure_trace_recorded": transaction_summary["trace_event"]
            == "language_structure",
            "records_checkpoint_backed_transaction": bool(
                transaction_summary["checkpoint_restore_verified"]
            ),
            "records_rollback_evidence": bool(transaction_summary["rollback_verified"]),
            "records_reviewed_structural_mutation": bool(applied),
            "post_structure_brain_checkpoint_restore_verified": bool(
                post_checkpoint_report["restore_verified"]
            ),
            "post_structure_status_restores_transaction": (
                structural_count_after_restore > structural_count_before
            ),
            "post_structure_sustained_enabled": bool(cfg.run_post_structure_sustained),
            "post_structure_sustained_target_reached": bool(
                post_structure_sustained.get("success", False)
            ),
            "post_structure_sustained_8192_boundary_reached": int(
                post_structure_sustained.get("token_delta", 0) or 0
            )
            >= 8192,
            "post_structure_sustained_131072_boundary_reached": int(
                post_structure_sustained.get("token_delta", 0) or 0
            )
            >= 131072,
            "post_structure_sustained_524288_boundary_reached": int(
                post_structure_sustained.get("token_delta", 0) or 0
            )
            >= 524288,
            "status_read_mutation_absent": not bool(status_read_mutation),
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
        if exception is not None:
            report.setdefault(
                "promotion_gate",
                {
                    "surface": "marulho_language_brain_installed_structural_gate.v1",
                    "ready_for_runtime_claim_review": False,
                    "promotes_runtime_claim": False,
                },
            )
        write_json_report_with_readme(output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain-checkpoint", type=Path, required=True)
    parser.add_argument("--pre-structural-brain-checkpoint", type=Path, default=None)
    parser.add_argument("--post-structural-brain-checkpoint", type=Path, default=None)
    parser.add_argument("--structural-baseline-checkpoint", type=Path, default=None)
    parser.add_argument("--post-structure-sustained-output", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--mutation-kind",
        choices=SUPPORTED_MUTATION_KINDS,
        default="route_bank_expansion",
    )
    parser.add_argument("--operator-approved", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-eval-batches", type=int, default=8)
    parser.add_argument("--max-eval-loss-delta", type=float, default=100.0)
    parser.add_argument("--route-saturation-threshold", type=float, default=0.5)
    parser.add_argument("--route-candidate-growth", type=int, default=4)
    parser.add_argument("--max-route-candidate-count", type=int, default=0)
    parser.add_argument("--memory-slot-growth", type=int, default=4)
    parser.add_argument("--max-memory-slot-count", type=int, default=0)
    parser.add_argument("--max-memory-slot-candidate-count", type=int, default=8)
    parser.add_argument("--skip-post-structure-sustained", action="store_true")
    parser.add_argument("--sustained-target-tokens", type=int, default=8192)
    parser.add_argument("--sustained-tick-tokens", type=int, default=128)
    parser.add_argument("--sustained-quantum-tokens", type=int, default=16)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--sustained-prompt", type=str, default="MARULHO")
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    config = BrainInstalledStructuralPlasticityEvidenceConfig(
        mutation_kind=str(args.mutation_kind),
        sequence_length=max(2, int(args.sequence_length)),
        stride=max(1, int(args.stride)),
        batch_size=max(1, int(args.batch_size)),
        eval_fraction=float(args.eval_fraction),
        max_eval_batches=max(1, int(args.max_eval_batches)),
        max_eval_loss_delta=float(args.max_eval_loss_delta),
        route_saturation_threshold=float(args.route_saturation_threshold),
        route_candidate_growth=max(1, int(args.route_candidate_growth)),
        max_route_candidate_count=max(0, int(args.max_route_candidate_count)),
        memory_slot_growth=max(1, int(args.memory_slot_growth)),
        max_memory_slot_count=max(0, int(args.max_memory_slot_count)),
        max_memory_slot_candidate_count=max(
            1,
            int(args.max_memory_slot_candidate_count),
        ),
        run_post_structure_sustained=not bool(args.skip_post_structure_sustained),
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
    )
    report = build_language_brain_installed_structural_plasticity_evidence(
        output_path=args.output,
        brain_checkpoint_path=args.brain_checkpoint,
        operator_approved=bool(args.operator_approved),
        pre_structural_brain_checkpoint_path=args.pre_structural_brain_checkpoint,
        post_structural_brain_checkpoint_path=args.post_structural_brain_checkpoint,
        structural_baseline_checkpoint_path=args.structural_baseline_checkpoint,
        post_structure_sustained_output_path=args.post_structure_sustained_output,
        corpus_path=args.corpus,
        config=config,
    )
    return 0 if report.get("report_status") == "final" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
