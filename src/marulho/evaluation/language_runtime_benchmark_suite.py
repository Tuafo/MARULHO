"""MARULHO LM-head benchmark-suite evidence aggregator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_grounding_support import (
    run_language_grounding_support_report,
)
from marulho.evaluation.language_generation_coherence import (
    ARTIFACT_KIND as GENERATION_COHERENCE_ARTIFACT_KIND,
    SURFACE as GENERATION_COHERENCE_SURFACE,
)
from marulho.evaluation.language_scale_ladder import (
    build_language_scale_ladder_report,
    estimate_language_model_parameters,
)
from marulho.evaluation.language_sustained_runtime_evidence import (
    run_language_sustained_runtime_evidence,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_checkpoint_evolution import (
    LanguageCheckpointEvolutionConfig,
    run_language_checkpoint_evolution,
)
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_column_split_proposal,
    build_language_structural_deep_sleep_proposal,
    build_language_structural_memory_slot_expansion_proposal,
    build_language_structural_merge_proposal,
    build_language_structural_prune_proposal,
    build_language_structural_plasticity_proposal,
    build_language_structural_retire_proposal,
    build_language_structural_route_bank_expansion_proposal,
    build_language_structural_synapse_bundle_proposal,
)


SURFACE = "marulho_language_runtime_benchmark_suite.v1"
ARTIFACT_KIND = "marulho_language_runtime_benchmark_suite"
SUSTAINED_SURFACE = "marulho_language_sustained_runtime_evidence.v1"
SUSTAINED_ARTIFACT_KIND = "marulho_language_sustained_runtime_evidence"
MEMORY_SLOT_RUNTIME_IMPACT_SURFACE = (
    "marulho_language_memory_slot_runtime_impact.v1"
)
MEMORY_SLOT_RUNTIME_IMPACT_ARTIFACT_KIND = (
    "marulho_language_memory_slot_runtime_impact"
)
MEMORY_SLOT_ARCHITECTURE_COST_SURFACE = (
    "marulho_language_continual_memory_slot_architecture_cost.v1"
)
CONTINUAL_LEARNING_EXPERIMENT_SURFACE = (
    "marulho_language_continual_learning_experiment.v1"
)
CONTINUAL_LEARNING_ARTIFACT_KIND = "marulho_language_continual_learning_window"
BRAIN_INSTALLED_CONTINUAL_LEARNING_SURFACE = (
    "marulho_language_brain_installed_continual_learning_evidence.v1"
)
BRAIN_INSTALLED_CONTINUAL_LEARNING_ARTIFACT_KIND = (
    "marulho_language_brain_installed_continual_learning_evidence"
)
STRUCTURAL_PLASTICITY_EXPERIMENT_SURFACE = (
    "marulho_language_structural_plasticity_experiment.v1"
)
STRUCTURAL_PLASTICITY_EXPERIMENT_ARTIFACT_KIND = (
    "marulho_language_structural_plasticity_experiment"
)
STRUCTURAL_PLASTICITY_TRANSACTION_SURFACE = (
    "marulho_language_structural_plasticity_transaction.v1"
)
STRUCTURAL_PLASTICITY_TRANSACTION_ARTIFACT_KIND = (
    "marulho_language_structural_plasticity_transaction"
)
QUALITY_REPLAY_SURFACE = "marulho_language_quality_replay_experiment.v1"
QUALITY_REPLAY_ARTIFACT_KIND = "marulho_language_quality_replay_experiment"
CHECKPOINT_EVOLUTION_EXPERIMENT_SURFACE = (
    "marulho_language_checkpoint_evolution_experiment.v1"
)
CHECKPOINT_EVOLUTION_EXPERIMENT_ARTIFACT_KIND = (
    "marulho_language_checkpoint_evolution_experiment"
)
KERNEL_SURFACE = "marulho_language_triton_kernel_report.v1"
KERNEL_ARTIFACT_KIND = "marulho_language_triton_kernel_report"
RMSNORM_KERNEL_NAME = "language_rmsnorm_forward"
PLIF_FORWARD_KERNEL_NAME = "language_plif_forward"
PLIF_SURROGATE_KERNEL_NAME = "language_plif_surrogate_backward"
SELECTIVE_SCAN_KERNEL_NAME = "language_selective_state_scan"
ELIGIBILITY_TRACE_KERNEL_NAME = "language_local_eligibility_trace_update"
ROUTE_TOPK_KERNEL_NAME = "language_route_vote_topk"
EXPERT_DISPATCH_KERNEL_NAME = "language_block_sparse_expert_dispatch"
SAMPLED_VOCAB_CE_KERNEL_NAME = "language_sampled_vocab_cross_entropy"
MEMORY_SLOT_RETRIEVAL_KERNEL_NAME = "language_memory_slot_retrieval"
SUPPORTED_GPU_KERNEL_NAMES = {
    RMSNORM_KERNEL_NAME,
    PLIF_FORWARD_KERNEL_NAME,
    PLIF_SURROGATE_KERNEL_NAME,
    SELECTIVE_SCAN_KERNEL_NAME,
    ELIGIBILITY_TRACE_KERNEL_NAME,
    ROUTE_TOPK_KERNEL_NAME,
    EXPERT_DISPATCH_KERNEL_NAME,
    SAMPLED_VOCAB_CE_KERNEL_NAME,
    MEMORY_SLOT_RETRIEVAL_KERNEL_NAME,
}


def _read_generation_coherence_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Generation coherence report is not an object: {report_path}")
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_quality_replay_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Quality replay report is not an object: {report_path}")
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_checkpoint_evolution_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint-evolution report is not an object: {report_path}"
        )
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _fixture_config(tokenizer: ByteLevelLanguageTokenizer) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=12,
        state_dim=20,
        expert_count=2,
        active_expert_count=1,
        route_candidate_count=2,
        expert_hidden_dim=32,
    )


def _new_model(tokenizer: ByteLevelLanguageTokenizer) -> MarulhoLanguageModel:
    return MarulhoLanguageModel(_fixture_config(tokenizer))


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _category(
    name: str,
    *,
    status: str,
    evidence: Mapping[str, Any] | None = None,
    missing: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "surface": "marulho_language_benchmark_category.v1",
        "name": name,
        "status": status,
        "passed": status in {"pass", "smoke_only"},
        "missing_evidence": list(missing),
        "evidence": dict(evidence or {}),
    }


def _tiny_brain_config() -> MarulhoConfig:
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
        device="cpu",
    )


def _read_sustained_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Sustained evidence report is not an object: {report_path}")
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_gpu_kernel_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"GPU kernel evidence report is not an object: {report_path}")
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_memory_slot_runtime_impact_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Memory-slot runtime impact report is not an object: {report_path}"
        )
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_memory_slot_architecture_cost_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Memory-slot architecture-cost report is not an object: {report_path}"
        )
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_brain_installed_continual_learning_report(
    path: str | Path,
) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Brain-installed continual-learning report is not an object: {report_path}"
        )
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _read_structural_plasticity_evidence_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    with report_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Structural-plasticity evidence report is not an object: {report_path}"
        )
    payload = dict(payload)
    payload.setdefault("path", str(report_path))
    return payload


def _valid_lm_sustained_report(report: Mapping[str, Any]) -> bool:
    return (
        report.get("artifact_kind") == SUSTAINED_ARTIFACT_KIND
        and report.get("surface") == SUSTAINED_SURFACE
        and report.get("report_status") == "final"
        and report.get("success") is True
        and report.get("active_language_path") == "marulho_lm_head"
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
    )


def _valid_language_memory_slot_runtime_impact_report(
    report: Mapping[str, Any],
) -> bool:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    return (
        report.get("artifact_kind") == MEMORY_SLOT_RUNTIME_IMPACT_ARTIFACT_KIND
        and report.get("surface") == MEMORY_SLOT_RUNTIME_IMPACT_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and promotion_gate.get("complete_runtime_impact_available") is True
        and promotion_gate.get("bounded_memory_slots_enabled") is True
        and promotion_gate.get("bounded_avoids_all_slot_scan") is True
        and promotion_gate.get("neutral_initialization_parity") is True
        and promotion_gate.get("trainable_neutral_initialization") is True
    )


def _valid_language_memory_slot_architecture_cost_report(
    report: Mapping[str, Any],
) -> bool:
    cost = (
        report.get("memory_slot_architecture_cost")
        if isinstance(report.get("memory_slot_architecture_cost"), Mapping)
        else {}
    )
    learning_evidence = (
        report.get("learning_evidence")
        if isinstance(report.get("learning_evidence"), Mapping)
        else {}
    )
    memory_slots = (
        learning_evidence.get("memory_slots")
        if isinstance(learning_evidence.get("memory_slots"), Mapping)
        else {}
    )
    training_backend = (
        report.get("training_memory_slot_backend_summary")
        if isinstance(report.get("training_memory_slot_backend_summary"), Mapping)
        else {}
    )
    return (
        report.get("artifact_kind") == CONTINUAL_LEARNING_ARTIFACT_KIND
        and report.get("surface") == "marulho_language_continual_learning_window.v1"
        and report.get("experiment_surface") == CONTINUAL_LEARNING_EXPERIMENT_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and report.get("status") == "accepted_online_update"
        and cost.get("surface") == MEMORY_SLOT_ARCHITECTURE_COST_SURFACE
        and cost.get("status") == "memory_slot_architecture_cost_measured"
        and cost.get("comparison_is_no_memory_baseline") is True
        and cost.get("comparable_update_throughput") is True
        and cost.get("comparable_total_window_throughput") is True
        and memory_slots.get("enabled") is True
        and memory_slots.get("bounded_memory_slot_path") is True
        and memory_slots.get("runs_all_slots") is False
        and training_backend.get("training_window_stats_recorded") is True
        and int(learning_evidence.get("update_token_count", 0) or 0) > 0
    )


def _valid_brain_installed_continual_learning_report(
    report: Mapping[str, Any],
) -> bool:
    gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    summary = (
        report.get("learning_summary")
        if isinstance(report.get("learning_summary"), Mapping)
        else {}
    )
    learned_checkpoint = (
        report.get("learned_brain_checkpoint")
        if isinstance(report.get("learned_brain_checkpoint"), Mapping)
        else {}
    )
    return (
        report.get("artifact_kind") == BRAIN_INSTALLED_CONTINUAL_LEARNING_ARTIFACT_KIND
        and report.get("surface") == BRAIN_INSTALLED_CONTINUAL_LEARNING_SURFACE
        and report.get("report_status") == "final"
        and report.get("status") == "final"
        and report.get("runtime_owner") == "MarulhoBrain"
        and report.get("active_language_path") == "marulho_lm_head"
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("service_owned_cognition") is False
        and report.get("status_read_mutation") is False
        and summary.get("brain_surface") == "marulho_brain_language_learning_window.v1"
        and summary.get("training_surface") == "marulho_language_continual_learning_window.v1"
        and summary.get("status") == "accepted_online_update"
        and summary.get("trace_event") == "language_learn"
        and summary.get("mutates_language_model_weights") is True
        and int(summary.get("update_token_count", 0) or 0) > 0
        and _finite_positive(summary.get("tokens_per_second"))
        and _finite_positive(summary.get("total_window_tokens_per_second"))
        and float(summary.get("final_parameter_delta_l2", 0.0) or 0.0) > 0.0
        and learned_checkpoint.get("restore_verified") is True
        and gate.get("installed_reviewed_checkpoint") is True
        and gate.get("batch_tokenizer_matches_installed_runtime") is True
        and gate.get("pre_learning_brain_checkpoint_restore_verified") is True
        and gate.get("learning_runs_through_marulho_brain") is True
        and gate.get("language_learn_trace_recorded") is True
        and gate.get("records_actual_continual_learning") is True
        and gate.get("records_forgetting") is True
        and gate.get("records_replay_retention") is True
        and gate.get("records_update_throughput") is True
        and gate.get("records_total_window_throughput") is True
        and gate.get("learned_brain_checkpoint_restore_verified") is True
        and gate.get("status_read_mutation_absent") is True
        and gate.get("external_llm_absent") is True
        and gate.get("service_owned_cognition_absent") is True
        and gate.get("promotes_runtime_claim") is False
    )


def _structural_transaction_bounded_route_bank(mutation: Mapping[str, Any]) -> bool:
    proposal_kind = str(mutation.get("proposal_kind") or "")
    if proposal_kind != "route_bank_expansion":
        return True
    target_candidates = int(mutation.get("target_route_candidate_count", 0) or 0)
    target_experts = int(mutation.get("target_expert_count", 0) or 0)
    return target_candidates > 0 and target_experts > 0 and target_candidates < target_experts


def _structural_transaction_bounded_memory_slots(mutation: Mapping[str, Any]) -> bool:
    proposal_kind = str(mutation.get("proposal_kind") or "")
    if proposal_kind != "memory_slot_expansion":
        return True
    target_slots = int(mutation.get("target_memory_slot_count", 0) or 0)
    target_candidates = int(mutation.get("target_memory_slot_candidate_count", 0) or 0)
    target_active = int(mutation.get("target_active_memory_slot_count", 0) or 0)
    return (
        target_slots > 0
        and target_candidates > 0
        and target_active > 0
        and target_active <= target_candidates < target_slots
    )


def _valid_language_structural_plasticity_transaction_report(
    report: Mapping[str, Any],
) -> bool:
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
    mutation = (
        report.get("mutation")
        if isinstance(report.get("mutation"), Mapping)
        else {}
    )
    proposal_kind = str(mutation.get("proposal_kind") or "")
    return (
        report.get("artifact_kind") == STRUCTURAL_PLASTICITY_TRANSACTION_ARTIFACT_KIND
        and report.get("surface") == STRUCTURAL_PLASTICITY_TRANSACTION_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and report.get("status") == "applied_structural_mutation"
        and report.get("applied") is True
        and report.get("operator_approved") is True
        and checkpoint.get("checkpoint_restore_verified") is True
        and rollback.get("rollback_verified") is True
        and gate.get("checkpoint_backed") is True
        and gate.get("heldout_non_regression") is True
        and gate.get("eligible_for_reviewed_structural_promotion") is True
        and proposal_kind
        and _structural_transaction_bounded_route_bank(mutation)
        and _structural_transaction_bounded_memory_slots(mutation)
    )


def _valid_language_gpu_kernel_report(report: Mapping[str, Any]) -> bool:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    return (
        report.get("artifact_kind") == KERNEL_ARTIFACT_KIND
        and report.get("surface") == KERNEL_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("kernel_name") in SUPPORTED_GPU_KERNEL_NAMES
        and report.get("parity_passed") is True
        and promotion_gate.get("kernel_parity_available") is True
    )


def _valid_language_generation_coherence_report(report: Mapping[str, Any]) -> bool:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    return (
        report.get("artifact_kind") == GENERATION_COHERENCE_ARTIFACT_KIND
        and report.get("surface") == GENERATION_COHERENCE_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and promotion_gate.get("generation_coherence_available") is True
        and promotion_gate.get("grounded_prompt_suite_available") is True
        and int(summary.get("case_count", 0) or 0) > 0
    )


def _generation_coherence_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    prompt_suite = (
        report.get("prompt_suite")
        if isinstance(report.get("prompt_suite"), Mapping)
        else {}
    )
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "checkpoint_path": report.get("checkpoint_path"),
        "case_count": int(summary.get("case_count", 0) or 0),
        "passed_case_count": int(summary.get("passed_case_count", 0) or 0),
        "case_pass_rate": float(summary.get("case_pass_rate", 0.0) or 0.0),
        "mean_prefix_match_chars": summary.get("mean_prefix_match_chars"),
        "mean_prefix_match_fraction": summary.get("mean_prefix_match_fraction"),
        "mean_printable_fraction": summary.get("mean_printable_fraction"),
        "mean_distinct_bigram_fraction": summary.get("mean_distinct_bigram_fraction"),
        "next_character_match_rate": summary.get("next_character_match_rate"),
        "review_kind": (
            prompt_suite.get("review_kind")
        ),
        "generation_decode_controls": dict(
            prompt_suite.get("generation_decode_controls")
            if isinstance(prompt_suite.get("generation_decode_controls"), Mapping)
            else {}
        ),
        "human_review_available": bool(promotion_gate.get("human_review_available")),
        "promotes_generation_quality_claim": bool(
            promotion_gate.get("promotes_generation_quality_claim")
        ),
    }


def _normalized_evidence_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.replace("\\", "/").lower()


def _language_generation_coherence_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_generation_coherence_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: float(
            (item.get("summary") or {}).get("case_pass_rate", 0.0)
            if isinstance(item.get("summary"), Mapping)
            else 0.0
        ),
        default=None,
    )
    missing = [] if best_report is not None else ["grounded_generation_coherence_report"]
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "generation_coherence_available": best_report is not None,
        "grounded_prompt_suite_available": best_report is not None,
        "best_report": (
            None
            if best_report is None
            else _generation_coherence_report_summary(best_report)
        ),
        "missing_evidence": missing,
        "promotes_runtime_claim": False,
    }


def _long_run_report_summaries(
    long_run_evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    keys = (
        "diagnostic_report",
        "long_gate_report",
        "house_scale_report",
        "controlled_decode_diagnostic_report",
        "controlled_decode_long_gate_report",
        "controlled_decode_house_scale_report",
    )
    summaries: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for key in keys:
        report = long_run_evidence.get(key)
        if not isinstance(report, Mapping):
            continue
        normalized = _normalized_evidence_path(report.get("checkpoint_path"))
        token_delta = _token_delta(report)
        identity = (normalized or str(report.get("path") or ""), token_delta)
        if identity in seen:
            continue
        seen.add(identity)
        summaries.append(dict(report))
    return tuple(summaries)


def _generation_long_run_alignment_evidence(
    generation_evidence: Mapping[str, Any],
    long_run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    best_report = (
        generation_evidence.get("best_report")
        if isinstance(generation_evidence.get("best_report"), Mapping)
        else {}
    )
    generation_checkpoint = best_report.get("checkpoint_path")
    generation_checkpoint_key = _normalized_evidence_path(generation_checkpoint)
    long_reports = _long_run_report_summaries(long_run_evidence)
    matching_reports = [
        report
        for report in long_reports
        if generation_checkpoint_key
        and _normalized_evidence_path(report.get("checkpoint_path"))
        == generation_checkpoint_key
    ]
    matching_long_reports = [
        report for report in matching_reports if _token_delta(report) >= 131072
    ]
    matching_house_reports = [
        report for report in matching_reports if _token_delta(report) >= 524288
    ]
    matching_controlled_reports = [
        report
        for report in matching_reports
        if (
            isinstance(report.get("generation_decode"), Mapping)
            and bool(report["generation_decode"].get("decode_controls_requested"))
        )
    ]
    matching_controlled_house_reports = [
        report
        for report in matching_controlled_reports
        if _token_delta(report) >= 524288
    ]
    evidence_available = bool(generation_checkpoint_key and matching_long_reports)
    controlled_house_required = bool(
        long_run_evidence.get("controlled_decode_house_scale_gate_reached")
    )
    missing: list[str] = []
    if not evidence_available:
        missing.append("same_checkpoint_generation_coherence_long_run")
    if controlled_house_required and not matching_controlled_house_reports:
        missing.append(
            "same_checkpoint_generation_coherence_controlled_decode_house_scale"
        )
    return {
        "surface": "marulho_language_generation_long_run_alignment.v1",
        "generation_checkpoint_path": generation_checkpoint,
        "long_run_checkpoint_paths": [
            report.get("checkpoint_path") for report in long_reports
        ],
        "matching_report_count": len(matching_reports),
        "same_checkpoint_long_run_available": evidence_available,
        "same_checkpoint_house_scale_available": bool(matching_house_reports),
        "same_checkpoint_controlled_decode_available": bool(matching_controlled_reports),
        "same_checkpoint_controlled_decode_house_scale_available": bool(
            matching_controlled_house_reports
        ),
        "controlled_decode_house_scale_required": controlled_house_required,
        "matching_reports": matching_reports,
        "missing_evidence": missing,
    }


def _quality_replay_selected_candidate(
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    selection = (
        report.get("candidate_selection")
        if isinstance(report.get("candidate_selection"), Mapping)
        else {}
    )
    selected_id = str(selection.get("selected_candidate_id") or "")
    candidates = selection.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("selected") is True:
            return dict(candidate)
    for candidate in candidates:
        if (
            isinstance(candidate, Mapping)
            and selected_id
            and str(candidate.get("candidate_id") or "") == selected_id
        ):
            return dict(candidate)
    return None


def _valid_language_quality_replay_report(report: Mapping[str, Any]) -> bool:
    selection = (
        report.get("candidate_selection")
        if isinstance(report.get("candidate_selection"), Mapping)
        else {}
    )
    lineage = (
        report.get("checkpoint_lineage")
        if isinstance(report.get("checkpoint_lineage"), Mapping)
        else {}
    )
    learning_report = (
        report.get("learning_evidence")
        if isinstance(report.get("learning_evidence"), Mapping)
        else {}
    )
    learning_gate = (
        learning_report.get("promotion_gate")
        if isinstance(learning_report.get("promotion_gate"), Mapping)
        else {}
    )
    rollback = (
        learning_report.get("rollback_evidence")
        if isinstance(learning_report.get("rollback_evidence"), Mapping)
        else {}
    )
    trained_delta = (
        report.get("generation_coherence_delta")
        if isinstance(report.get("generation_coherence_delta"), Mapping)
        else {}
    )
    heldout_delta = (
        report.get("heldout_generation_coherence_delta")
        if isinstance(report.get("heldout_generation_coherence_delta"), Mapping)
        else {}
    )
    after_gate = (
        (report.get("generation_coherence_after") or {}).get("promotion_gate")
        if isinstance(report.get("generation_coherence_after"), Mapping)
        and isinstance(
            (report.get("generation_coherence_after") or {}).get("promotion_gate"),
            Mapping,
        )
        else {}
    )
    heldout_after_gate = (
        (report.get("heldout_generation_coherence_after") or {}).get(
            "promotion_gate"
        )
        if isinstance(report.get("heldout_generation_coherence_after"), Mapping)
        and isinstance(
            (report.get("heldout_generation_coherence_after") or {}).get(
                "promotion_gate"
            ),
            Mapping,
        )
        else {}
    )
    review = (
        report.get("quality_generalization_review")
        if isinstance(report.get("quality_generalization_review"), Mapping)
        else {}
    )
    selected = _quality_replay_selected_candidate(report)
    selected_child_path = str(selection.get("selected_child_checkpoint_path") or "")
    selected_path_matches = bool(
        selected is not None
        and selected_child_path
        and _normalized_evidence_path(selected.get("child_checkpoint_path"))
        == _normalized_evidence_path(selected_child_path)
        and _normalized_evidence_path(report.get("child_checkpoint_path"))
        == _normalized_evidence_path(selected_child_path)
    )
    return (
        report.get("artifact_kind") == QUALITY_REPLAY_ARTIFACT_KIND
        and report.get("surface") == QUALITY_REPLAY_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and selected_path_matches
        and bool(selection.get("selected_candidate_id"))
        and int(selection.get("candidate_count", 0) or 0) >= 1
        and selection.get("mutates_parent_checkpoint") is False
        and selection.get("heldout_cases_used_for_replay_training") is False
        and selection.get("saves_child_checkpoint_per_candidate") is True
        and selection.get("runs_sustained_runtime_only_for_selected_child") is True
        and lineage.get("writes_child_checkpoint") is True
        and lineage.get("mutates_parent_checkpoint") is False
        and learning_report.get("status") == "accepted_online_update"
        and learning_report.get("external_llm_used") is False
        and learning_report.get("loads_external_checkpoint") is False
        and learning_report.get("active_language_path") == "marulho_lm_head"
        and learning_gate.get("eligible_for_online_learning_review") is True
        and learning_gate.get("rollback_available") is True
        and rollback.get("restore_verified") is True
        and after_gate.get("generation_coherence_available") is True
        and heldout_after_gate.get("generation_coherence_available") is True
        and review.get("heldout_prompt_coherence_available") is True
        and int(review.get("heldout_regressed_prompt_count", 0) or 0) == 0
        and int(trained_delta.get("regressed_prompt_count", 0) or 0) == 0
        and int(heldout_delta.get("regressed_prompt_count", 0) or 0) == 0
    )


def _quality_replay_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    selection = (
        report.get("candidate_selection")
        if isinstance(report.get("candidate_selection"), Mapping)
        else {}
    )
    learning_report = (
        report.get("learning_evidence")
        if isinstance(report.get("learning_evidence"), Mapping)
        else {}
    )
    learning_evidence = (
        learning_report.get("learning_evidence")
        if isinstance(learning_report.get("learning_evidence"), Mapping)
        else {}
    )
    trained_delta = (
        report.get("generation_coherence_delta")
        if isinstance(report.get("generation_coherence_delta"), Mapping)
        else {}
    )
    heldout_delta = (
        report.get("heldout_generation_coherence_delta")
        if isinstance(report.get("heldout_generation_coherence_delta"), Mapping)
        else {}
    )
    review = (
        report.get("quality_generalization_review")
        if isinstance(report.get("quality_generalization_review"), Mapping)
        else {}
    )
    selected = _quality_replay_selected_candidate(report) or {}
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "parent_checkpoint_path": report.get("parent_checkpoint_path"),
        "selected_child_checkpoint_path": selection.get(
            "selected_child_checkpoint_path"
        ),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "candidate_count": int(selection.get("candidate_count", 0) or 0),
        "selection_policy": selection.get("selection_policy"),
        "selected_learning_config": dict(
            selected.get("learning_config")
            if isinstance(selected.get("learning_config"), Mapping)
            else {}
        ),
        "selected_update_tokens_per_second": selected.get(
            "update_tokens_per_second",
            learning_evidence.get("tokens_per_second"),
        ),
        "selected_total_window_tokens_per_second": selected.get(
            "total_window_tokens_per_second",
            learning_evidence.get("total_window_tokens_per_second"),
        ),
        "selected_update_token_count": learning_evidence.get("update_token_count"),
        "selected_new_domain_loss_delta": learning_evidence.get(
            "new_domain_loss_delta"
        ),
        "selected_old_domain_forgetting": learning_evidence.get(
            "old_domain_forgetting"
        ),
        "selected_general_replay_retention_delta": learning_evidence.get(
            "general_replay_retention_delta"
        ),
        "trained_repaired_prompt_count": int(
            trained_delta.get("repaired_prompt_count", 0) or 0
        ),
        "trained_regressed_prompt_count": int(
            trained_delta.get("regressed_prompt_count", 0) or 0
        ),
        "trained_mean_prefix_match_chars_delta": trained_delta.get(
            "mean_prefix_match_chars_delta"
        ),
        "heldout_repaired_prompt_count": int(
            heldout_delta.get("repaired_prompt_count", 0) or 0
        ),
        "heldout_regressed_prompt_count": int(
            heldout_delta.get("regressed_prompt_count", 0) or 0
        ),
        "heldout_mean_prefix_match_chars_delta": heldout_delta.get(
            "mean_prefix_match_chars_delta"
        ),
        "heldout_case_pass_rate": review.get("heldout_case_pass_rate"),
        "heldout_case_count": int(review.get("heldout_case_count", 0) or 0),
        "heldout_prompt_coherence_available": bool(
            review.get("heldout_prompt_coherence_available")
        ),
        "promotes_generation_quality_claim": bool(
            review.get("promotes_generation_quality_claim", False)
        ),
        "promotes_runtime_claim": False,
    }


def _language_quality_replay_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_quality_replay_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: float(
            (
                (_quality_replay_selected_candidate(item) or {}).get(
                    "update_tokens_per_second",
                    0.0,
                )
                or 0.0
            )
        ),
        default=None,
    )
    supplied_invalid_reports = len(reports) > 0 and best_report is None
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "quality_replay_available": best_report is not None,
        "best_report": (
            None if best_report is None else _quality_replay_report_summary(best_report)
        ),
        "missing_evidence": (
            ["valid_language_quality_replay_report"]
            if supplied_invalid_reports
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_generation_quality_claim": False,
        "promotes_runtime_claim": False,
    }


def _valid_language_checkpoint_evolution_report(report: Mapping[str, Any]) -> bool:
    gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    lineage = (
        report.get("checkpoint_lineage")
        if isinstance(report.get("checkpoint_lineage"), Mapping)
        else {}
    )
    review = (
        report.get("evolution_review")
        if isinstance(report.get("evolution_review"), Mapping)
        else {}
    )
    runtime = (
        report.get("runtime_evidence")
        if isinstance(report.get("runtime_evidence"), Mapping)
        else {}
    )
    experiment_review = (
        report.get("experiment_review")
        if isinstance(report.get("experiment_review"), Mapping)
        else {}
    )
    return (
        report.get("artifact_kind") == CHECKPOINT_EVOLUTION_EXPERIMENT_ARTIFACT_KIND
        and report.get("surface") == CHECKPOINT_EVOLUTION_EXPERIMENT_SURFACE
        and report.get("owned_by_marulho") is True
        and report.get("external_llm_used") is False
        and report.get("loads_external_checkpoint") is False
        and report.get("active_language_path") == "marulho_lm_head"
        and gate.get("checkpoint_evolution_evidence_available") is True
        and gate.get("rollback_to_parent_verified") is True
        and gate.get("parent_runtime_unchanged") is True
        and gate.get("checkpoint_lineage_complete") is True
        and gate.get("child_checkpoint_available") is True
        and gate.get("long_run_evidence_required_for_parent_promotion") is True
        and gate.get("promotes_parent_promotion") is False
        and lineage.get("lineage_complete") is True
        and lineage.get("child_initial_matches_parent_state") is True
        and lineage.get("child_final_matches_child_runtime") is True
        and lineage.get("child_final_differs_from_parent_state") is True
        and review.get("parent_kept_installed") is True
        and review.get("isolated_child_training") is True
        and int(review.get("child_update_token_count", 0) or 0) > 0
        and runtime.get("checkpoint_storage_device") == "cpu"
        and experiment_review.get("records_checkpoint_lineage") is True
        and experiment_review.get("records_runtime_evidence") is True
        and experiment_review.get("records_child_learning_update") is True
    )


def _checkpoint_evolution_report_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = (
        report.get("checkpoint_lineage")
        if isinstance(report.get("checkpoint_lineage"), Mapping)
        else {}
    )
    review = (
        report.get("evolution_review")
        if isinstance(report.get("evolution_review"), Mapping)
        else {}
    )
    runtime = (
        report.get("runtime_evidence")
        if isinstance(report.get("runtime_evidence"), Mapping)
        else {}
    )
    gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    split = report.get("split") if isinstance(report.get("split"), Mapping) else {}
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "status": report.get("status"),
        "lineage_id": lineage.get("lineage_id"),
        "parent_checkpoint_path": lineage.get("parent_checkpoint_path"),
        "parent_checkpoint_sha256": lineage.get("parent_checkpoint_sha256"),
        "child_initial_checkpoint_path": lineage.get("child_initial_checkpoint_path"),
        "child_final_checkpoint_path": lineage.get("child_final_checkpoint_path"),
        "child_final_checkpoint_sha256": lineage.get(
            "child_final_checkpoint_sha256"
        ),
        "child_update_token_count": int(
            review.get("child_update_token_count", 0) or 0
        ),
        "child_optimizer_step_count": int(
            review.get("child_optimizer_step_count", 0) or 0
        ),
        "child_training_tokens_per_second": runtime.get(
            "child_training_tokens_per_second"
        ),
        "child_training_total_window_tokens_per_second": runtime.get(
            "child_training_total_window_tokens_per_second"
        ),
        "child_training_device": runtime.get("child_training_device"),
        "child_training_dense_adamw_backend": runtime.get(
            "child_training_dense_adamw_backend"
        ),
        "checkpoint_storage_device": runtime.get("checkpoint_storage_device"),
        "structural_growth_attempted": bool(
            review.get("structural_growth_attempted")
        ),
        "structural_transaction_applied": bool(
            review.get("structural_transaction_applied")
        ),
        "used_child_train_tokens": split.get("used_child_train_tokens"),
        "used_replay_tokens": split.get("used_replay_tokens"),
        "eligible_for_parent_promotion_review": bool(
            gate.get("eligible_for_parent_promotion_review")
        ),
        "long_run_evidence_required_for_parent_promotion": bool(
            gate.get("long_run_evidence_required_for_parent_promotion")
        ),
        "promotes_runtime_claim": False,
        "promotes_parent_promotion": False,
    }


def _language_checkpoint_evolution_saved_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_checkpoint_evolution_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: float(
            (
                (
                    item.get("runtime_evidence")
                    if isinstance(item.get("runtime_evidence"), Mapping)
                    else {}
                ).get("child_training_tokens_per_second", 0.0)
            )
            or 0.0
        ),
        default=None,
    )
    supplied_invalid_reports = len(reports) > 0 and best_report is None
    return {
        "surface": "marulho_language_checkpoint_evolution_saved_evidence.v1",
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "checkpoint_evolution_evidence_available": best_report is not None,
        "best_report": (
            None
            if best_report is None
            else _checkpoint_evolution_report_summary(best_report)
        ),
        "missing_evidence": (
            ["valid_language_checkpoint_evolution_experiment"]
            if supplied_invalid_reports
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_runtime_claim": False,
        "promotes_parent_promotion": False,
    }


def _quality_replay_long_run_alignment_evidence(
    quality_replay_evidence: Mapping[str, Any],
    long_run_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    best_report = (
        quality_replay_evidence.get("best_report")
        if isinstance(quality_replay_evidence.get("best_report"), Mapping)
        else {}
    )
    child_checkpoint = best_report.get("selected_child_checkpoint_path")
    child_checkpoint_key = _normalized_evidence_path(child_checkpoint)
    long_reports = _long_run_report_summaries(long_run_evidence)
    matching_reports = [
        report
        for report in long_reports
        if child_checkpoint_key
        and _normalized_evidence_path(report.get("checkpoint_path"))
        == child_checkpoint_key
    ]
    matching_long_reports = [
        report for report in matching_reports if _token_delta(report) >= 131072
    ]
    matching_house_reports = [
        report for report in matching_reports if _token_delta(report) >= 524288
    ]
    matching_controlled_reports = [
        report
        for report in matching_reports
        if (
            isinstance(report.get("generation_decode"), Mapping)
            and bool(report["generation_decode"].get("decode_controls_requested"))
        )
    ]
    matching_controlled_house_reports = [
        report
        for report in matching_controlled_reports
        if _token_delta(report) >= 524288
    ]
    evidence_available = bool(child_checkpoint_key and matching_long_reports)
    controlled_house_required = bool(
        long_run_evidence.get("controlled_decode_house_scale_gate_reached")
    )
    missing: list[str] = []
    if not evidence_available:
        missing.append("same_child_quality_replay_long_run")
    if controlled_house_required and not matching_controlled_house_reports:
        missing.append("same_child_quality_replay_controlled_decode_house_scale")
    return {
        "surface": "marulho_language_quality_replay_long_run_alignment.v1",
        "selected_child_checkpoint_path": child_checkpoint,
        "long_run_checkpoint_paths": [
            report.get("checkpoint_path") for report in long_reports
        ],
        "matching_report_count": len(matching_reports),
        "same_child_long_run_available": evidence_available,
        "same_child_house_scale_available": bool(matching_house_reports),
        "same_child_controlled_decode_available": bool(matching_controlled_reports),
        "same_child_controlled_decode_house_scale_available": bool(
            matching_controlled_house_reports
        ),
        "controlled_decode_house_scale_required": controlled_house_required,
        "matching_reports": matching_reports,
        "missing_evidence": missing,
    }


def _token_delta(report: Mapping[str, Any]) -> int:
    try:
        return int(report.get("token_delta", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _sustained_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    device_backend = (
        report.get("device_backend")
        if isinstance(report.get("device_backend"), Mapping)
        else {}
    )
    generation_decode = (
        report.get("generation_decode")
        if isinstance(report.get("generation_decode"), Mapping)
        else {}
    )
    execution_evidence = (
        report.get("execution_evidence")
        if isinstance(report.get("execution_evidence"), Mapping)
        else {}
    )
    decode_controls_requested = bool(
        generation_decode.get("decode_controls_requested")
        or execution_evidence.get("decode_controls_requested")
        or generation_decode.get("repetition_penalty_applied")
        or generation_decode.get("no_repeat_ngram_applied")
        or execution_evidence.get("repetition_penalty_applied")
        or execution_evidence.get("no_repeat_ngram_applied")
    )
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "report_status": report.get("report_status"),
        "success": bool(report.get("success")),
        "target_tokens": int(report.get("target_tokens", 0) or 0),
        "token_delta": _token_delta(report),
        "tokens_per_second": report.get("tokens_per_second"),
        "checkpoint_path": report.get("checkpoint_path"),
        "active_language_path": report.get("active_language_path"),
        "runtime_owner": report.get("runtime_owner"),
        "device": device_backend.get("device"),
        "backend": device_backend.get("backend"),
        "triton_kernel_used": bool(device_backend.get("triton_kernel_used")),
        "promoted_hot_path": bool(device_backend.get("promoted_hot_path")),
        "generation_decode": {
            "decode_controls_requested": decode_controls_requested,
            "decode_controls_backend": str(
                generation_decode.get(
                    "decode_controls_backend",
                    execution_evidence.get("decode_controls_backend", ""),
                )
                or ""
            ),
            "decode_controls_cpu_token_copy": bool(
                generation_decode.get(
                    "decode_controls_cpu_token_copy",
                    execution_evidence.get("decode_controls_cpu_token_copy", False),
                )
            ),
            "decode_controls_graph_compatible": bool(
                generation_decode.get(
                    "decode_controls_graph_compatible",
                    execution_evidence.get("decode_controls_graph_compatible", False),
                )
            ),
            "cuda_graph_decode_controls_used": bool(
                generation_decode.get(
                    "cuda_graph_decode_controls_used",
                    execution_evidence.get("cuda_graph_decode_controls_used", False),
                )
            ),
            "repetition_penalty": float(
                generation_decode.get(
                    "repetition_penalty",
                    execution_evidence.get("repetition_penalty", 1.0),
                )
                or 1.0
            ),
            "repetition_penalty_applied": bool(
                generation_decode.get(
                    "repetition_penalty_applied",
                    execution_evidence.get("repetition_penalty_applied", False),
                )
            ),
            "no_repeat_ngram_size": int(
                generation_decode.get(
                    "no_repeat_ngram_size",
                    execution_evidence.get("no_repeat_ngram_size", 0),
                )
                or 0
            ),
            "no_repeat_ngram_applied": bool(
                generation_decode.get(
                    "no_repeat_ngram_applied",
                    execution_evidence.get("no_repeat_ngram_applied", False),
                )
            ),
            "decode_control_fallback_count": int(
                generation_decode.get(
                    "decode_control_fallback_count",
                    execution_evidence.get("decode_control_fallback_count", 0),
                )
                or 0
            ),
            "repetition_penalty_adjusted_token_count": int(
                generation_decode.get(
                    "repetition_penalty_adjusted_token_count",
                    execution_evidence.get("repetition_penalty_adjusted_token_count", 0),
                )
                or 0
            ),
            "no_repeat_ngram_banned_token_count": int(
                generation_decode.get(
                    "no_repeat_ngram_banned_token_count",
                    execution_evidence.get("no_repeat_ngram_banned_token_count", 0),
                )
                or 0
            ),
        },
        "diagnostic_boundary_reached": bool(
            promotion_gate.get("diagnostic_boundary_reached")
        ),
        "long_run_gate_reached": bool(promotion_gate.get("long_run_gate_reached")),
        "house_scale_gate_reached": bool(promotion_gate.get("house_scale_gate_reached")),
        "promotes_runtime_claim": bool(promotion_gate.get("promotes_runtime_claim")),
        "promotes_hot_path": bool(promotion_gate.get("promotes_hot_path")),
    }


def _sustained_report_uses_decode_controls(report: Mapping[str, Any]) -> bool:
    generation_decode = (
        report.get("generation_decode")
        if isinstance(report.get("generation_decode"), Mapping)
        else {}
    )
    execution_evidence = (
        report.get("execution_evidence")
        if isinstance(report.get("execution_evidence"), Mapping)
        else {}
    )
    return bool(
        generation_decode.get("decode_controls_requested")
        or execution_evidence.get("decode_controls_requested")
        or generation_decode.get("repetition_penalty_applied")
        or generation_decode.get("no_repeat_ngram_applied")
        or execution_evidence.get("repetition_penalty_applied")
        or execution_evidence.get("no_repeat_ngram_applied")
    )


def _language_long_run_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [dict(report) for report in reports if _valid_lm_sustained_report(report)]
    diagnostic_reports = [
        report for report in valid_reports if _token_delta(report) >= 8192
    ]
    diagnostic_only_reports = [
        report for report in diagnostic_reports if _token_delta(report) < 131072
    ]
    long_gate_reports = [
        report for report in valid_reports if _token_delta(report) >= 131072
    ]
    long_gate_only_reports = [
        report for report in long_gate_reports if _token_delta(report) < 524288
    ]
    house_scale_reports = [
        report for report in valid_reports if _token_delta(report) >= 524288
    ]
    controlled_decode_reports = [
        report for report in valid_reports if _sustained_report_uses_decode_controls(report)
    ]
    controlled_diagnostic_reports = [
        report for report in controlled_decode_reports if _token_delta(report) >= 8192
    ]
    controlled_diagnostic_only_reports = [
        report for report in controlled_diagnostic_reports if _token_delta(report) < 131072
    ]
    controlled_long_gate_reports = [
        report for report in controlled_decode_reports if _token_delta(report) >= 131072
    ]
    controlled_long_gate_only_reports = [
        report for report in controlled_long_gate_reports if _token_delta(report) < 524288
    ]
    controlled_house_scale_reports = [
        report for report in controlled_decode_reports if _token_delta(report) >= 524288
    ]
    best_diagnostic = max(
        diagnostic_only_reports or diagnostic_reports,
        key=_token_delta,
        default=None,
    )
    best_long = max(long_gate_only_reports or long_gate_reports, key=_token_delta, default=None)
    best_house = max(house_scale_reports, key=_token_delta, default=None)
    best_controlled_diagnostic = max(
        controlled_diagnostic_only_reports or controlled_diagnostic_reports,
        key=_token_delta,
        default=None,
    )
    best_controlled_long = max(
        controlled_long_gate_only_reports or controlled_long_gate_reports,
        key=_token_delta,
        default=None,
    )
    best_controlled_house = max(
        controlled_house_scale_reports,
        key=_token_delta,
        default=None,
    )
    missing: list[str] = []
    if best_diagnostic is None:
        missing.append("8192_token_diagnostic_run")
    if best_long is None:
        missing.append("131072_token_long_run_gate")
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "diagnostic_report": (
            None if best_diagnostic is None else _sustained_report_summary(best_diagnostic)
        ),
        "long_gate_report": (
            None if best_long is None else _sustained_report_summary(best_long)
        ),
        "house_scale_report": (
            None if best_house is None else _sustained_report_summary(best_house)
        ),
        "controlled_decode_report_count": len(controlled_decode_reports),
        "controlled_decode_available": bool(controlled_decode_reports),
        "controlled_decode_diagnostic_report": (
            None
            if best_controlled_diagnostic is None
            else _sustained_report_summary(best_controlled_diagnostic)
        ),
        "controlled_decode_long_gate_report": (
            None
            if best_controlled_long is None
            else _sustained_report_summary(best_controlled_long)
        ),
        "controlled_decode_house_scale_report": (
            None
            if best_controlled_house is None
            else _sustained_report_summary(best_controlled_house)
        ),
        "diagnostic_boundary_reached": best_diagnostic is not None,
        "long_run_gate_reached": best_long is not None,
        "house_scale_gate_reached": best_house is not None,
        "controlled_decode_diagnostic_boundary_reached": (
            best_controlled_diagnostic is not None
        ),
        "controlled_decode_long_run_gate_reached": best_controlled_long is not None,
        "controlled_decode_house_scale_gate_reached": best_controlled_house is not None,
        "missing_evidence": missing,
        "controlled_decode_missing_evidence": (
            []
            if controlled_decode_reports
            else ["controlled_decode_sustained_run"]
        ),
        "promotes_runtime_claim": False,
        "promotes_hot_path": False,
    }


def _memory_slot_runtime_impact_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    comparison = (
        report.get("comparison") if isinstance(report.get("comparison"), Mapping) else {}
    )
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    arms = report.get("arms") if isinstance(report.get("arms"), Mapping) else {}
    bounded = arms.get("bounded_memory_slots_enabled")
    all_slot = arms.get("all_slot_memory_scan_contrast")
    bounded = bounded if isinstance(bounded, Mapping) else {}
    all_slot = all_slot if isinstance(all_slot, Mapping) else {}
    batch = report.get("batch") if isinstance(report.get("batch"), Mapping) else {}
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "model_vocab_size": int(report.get("model_vocab_size", 0) or 0),
        "tokens_per_forward": int(batch.get("tokens_per_forward", 0) or 0),
        "control_tokens_per_second": comparison.get("control_tokens_per_second"),
        "bounded_tokens_per_second": comparison.get("bounded_tokens_per_second"),
        "bounded_vs_control_tokens_per_second_ratio": comparison.get(
            "bounded_vs_control_tokens_per_second_ratio"
        ),
        "all_slot_tokens_per_second": comparison.get("all_slot_tokens_per_second"),
        "all_slot_vs_bounded_tokens_per_second_ratio": comparison.get(
            "all_slot_vs_bounded_tokens_per_second_ratio"
        ),
        "bounded_candidate_slot_count": int(
            bounded.get("candidate_slot_count", 0) or 0
        ),
        "bounded_active_slots_per_token": int(
            bounded.get("active_slots_per_token", 0) or 0
        ),
        "bounded_candidate_slots_scored": int(
            bounded.get("candidate_slots_scored", 0) or 0
        ),
        "bounded_runs_all_slots": bool(bounded.get("runs_all_slots", False)),
        "all_slot_candidate_slots_scored": int(
            all_slot.get("candidate_slots_scored", 0) or 0
        ),
        "all_slot_runs_all_slots": bool(all_slot.get("runs_all_slots", False)),
        "memory_gate_readback": bool(comparison.get("memory_gate_readback", False)),
        "bounded_memory_slot_nonzero_count": int(
            comparison.get("bounded_memory_slot_nonzero_count", 0) or 0
        ),
        "bounded_trainable_neutral_initialization": bool(
            comparison.get("bounded_trainable_neutral_initialization", False)
        ),
        "bounded_avoids_all_slot_scan": bool(
            promotion_gate.get("bounded_avoids_all_slot_scan")
        ),
        "neutral_initialization_parity": bool(
            promotion_gate.get("neutral_initialization_parity")
        ),
        "trainable_neutral_initialization": bool(
            promotion_gate.get("trainable_neutral_initialization")
        ),
        "promotes_hot_path": bool(promotion_gate.get("promotes_hot_path")),
        "promotes_runtime_claim": bool(promotion_gate.get("promotes_runtime_claim")),
    }


def _language_memory_slot_runtime_impact_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_memory_slot_runtime_impact_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: float(
            (item.get("comparison") or {}).get(
                "bounded_vs_control_tokens_per_second_ratio",
                0.0,
            )
            if isinstance(item.get("comparison"), Mapping)
            else 0.0
        ),
        default=None,
    )
    supplied_invalid_reports = len(reports) > 0 and best_report is None
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "memory_slot_runtime_impact_available": best_report is not None,
        "best_report": (
            None
            if best_report is None
            else _memory_slot_runtime_impact_summary(best_report)
        ),
        "missing_evidence": (
            ["valid_memory_slot_runtime_impact_report"]
            if supplied_invalid_reports
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_runtime_claim": False,
        "promotes_hot_path": False,
    }


def _memory_slot_architecture_cost_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    cost = (
        report.get("memory_slot_architecture_cost")
        if isinstance(report.get("memory_slot_architecture_cost"), Mapping)
        else {}
    )
    learning_evidence = (
        report.get("learning_evidence")
        if isinstance(report.get("learning_evidence"), Mapping)
        else {}
    )
    memory_slots = (
        learning_evidence.get("memory_slots")
        if isinstance(learning_evidence.get("memory_slots"), Mapping)
        else {}
    )
    training_backend = (
        report.get("training_memory_slot_backend_summary")
        if isinstance(report.get("training_memory_slot_backend_summary"), Mapping)
        else {}
    )
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "status": report.get("status"),
        "model_vocab_size": int(report.get("model_vocab_size", 0) or 0),
        "sampled_vocab_size": int(report.get("sampled_vocab_size", 0) or 0),
        "update_token_count": int(learning_evidence.get("update_token_count", 0) or 0),
        "current_update_tokens_per_second": cost.get(
            "current_update_tokens_per_second"
        ),
        "comparison_update_tokens_per_second": cost.get(
            "comparison_update_tokens_per_second"
        ),
        "delta_vs_no_memory_update_percent": cost.get(
            "delta_vs_no_memory_update_percent"
        ),
        "current_total_window_tokens_per_second": cost.get(
            "current_total_window_tokens_per_second"
        ),
        "comparison_total_window_tokens_per_second": cost.get(
            "comparison_total_window_tokens_per_second"
        ),
        "delta_vs_no_memory_total_window_percent": cost.get(
            "delta_vs_no_memory_total_window_percent"
        ),
        "delta_vs_no_memory_new_domain_loss_delta": cost.get(
            "delta_vs_no_memory_new_domain_loss_delta"
        ),
        "delta_vs_no_memory_old_domain_forgetting": cost.get(
            "delta_vs_no_memory_old_domain_forgetting"
        ),
        "delta_vs_no_memory_general_replay_retention_delta": cost.get(
            "delta_vs_no_memory_general_replay_retention_delta"
        ),
        "delta_vs_no_memory_after_mean_source_prefix_match_chars": cost.get(
            "delta_vs_no_memory_after_mean_source_prefix_match_chars"
        ),
        "candidate_slots_scored": int(memory_slots.get("candidate_slots_scored", 0) or 0),
        "candidate_id_source": memory_slots.get("candidate_id_source"),
        "runs_all_slots": bool(memory_slots.get("runs_all_slots", False)),
        "bounded_memory_slot_path": bool(
            memory_slots.get("bounded_memory_slot_path", False)
        ),
        "memory_slot_retrieval_backend": memory_slots.get(
            "memory_slot_retrieval_backend"
        ),
        "training_window_backend": training_backend.get(
            "memory_slot_retrieval_backend"
        ),
        "triton_training_autograd_used": bool(
            training_backend.get("triton_autograd_used", False)
        ),
        "comparison_report": cost.get("comparison_report"),
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }


def _language_memory_slot_architecture_cost_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_memory_slot_architecture_cost_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: int(
            (item.get("learning_evidence") or {}).get("update_token_count", 0)
            if isinstance(item.get("learning_evidence"), Mapping)
            else 0
        ),
        default=None,
    )
    supplied_invalid_reports = len(reports) > 0 and best_report is None
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "memory_slot_architecture_cost_available": best_report is not None,
        "best_report": (
            None
            if best_report is None
            else _memory_slot_architecture_cost_summary(best_report)
        ),
        "missing_evidence": (
            ["valid_memory_slot_architecture_cost_report"]
            if supplied_invalid_reports
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }


def _brain_installed_continual_learning_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    summary = (
        report.get("learning_summary")
        if isinstance(report.get("learning_summary"), Mapping)
        else {}
    )
    memory_slots = (
        summary.get("memory_slots")
        if isinstance(summary.get("memory_slots"), Mapping)
        else {}
    )
    pre_checkpoint = (
        report.get("pre_learning_brain_checkpoint")
        if isinstance(report.get("pre_learning_brain_checkpoint"), Mapping)
        else {}
    )
    learned_checkpoint = (
        report.get("learned_brain_checkpoint")
        if isinstance(report.get("learned_brain_checkpoint"), Mapping)
        else {}
    )
    sustained = (
        report.get("post_learning_sustained_window")
        if isinstance(report.get("post_learning_sustained_window"), Mapping)
        else {}
    )
    gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "runtime_owner": report.get("runtime_owner"),
        "active_language_path": report.get("active_language_path"),
        "brain_surface": summary.get("brain_surface"),
        "training_surface": summary.get("training_surface"),
        "learning_status": summary.get("status"),
        "update_token_count": int(summary.get("update_token_count", 0) or 0),
        "tokens_per_second": float(summary.get("tokens_per_second", 0.0) or 0.0),
        "total_window_tokens_per_second": float(
            summary.get("total_window_tokens_per_second", 0.0) or 0.0
        ),
        "new_domain_loss_delta": float(
            summary.get("new_domain_loss_delta", 0.0) or 0.0
        ),
        "old_domain_forgetting": float(
            summary.get("old_domain_forgetting", 0.0) or 0.0
        ),
        "general_replay_retention_delta": float(
            summary.get("general_replay_retention_delta", 0.0) or 0.0
        ),
        "final_parameter_delta_l2": float(
            summary.get("final_parameter_delta_l2", 0.0) or 0.0
        ),
        "device": summary.get("device"),
        "memory_slots_enabled": bool(memory_slots.get("enabled", False)),
        "memory_slot_candidate_slots_scored": int(
            memory_slots.get("candidate_slots_scored", 0) or 0
        ),
        "memory_slot_runs_all_slots": bool(memory_slots.get("runs_all_slots", False)),
        "memory_slot_bounded_path": bool(
            memory_slots.get("bounded_memory_slot_path", False)
        ),
        "memory_slot_retrieval_backend": memory_slots.get(
            "memory_slot_retrieval_backend"
        ),
        "pre_learning_checkpoint_path": pre_checkpoint.get("path"),
        "pre_learning_checkpoint_restore_verified": bool(
            pre_checkpoint.get("restore_verified", False)
        ),
        "learned_brain_checkpoint_path": learned_checkpoint.get("path"),
        "learned_brain_checkpoint_restore_verified": bool(
            learned_checkpoint.get("restore_verified", False)
        ),
        "post_learning_sustained_enabled": bool(sustained.get("enabled", False)),
        "post_learning_sustained_success": bool(sustained.get("success", False)),
        "post_learning_sustained_token_delta": int(
            sustained.get("token_delta", 0) or 0
        ),
        "post_learning_sustained_tokens_per_second": float(
            sustained.get("tokens_per_second", 0.0) or 0.0
        ),
        "post_learning_sustained_backend": sustained.get("backend"),
        "post_learning_sustained_triton_kernel_names": list(
            sustained.get("tracked_triton_kernel_used_names") or []
        ),
        "house_scale_update_tokens_reached": bool(
            gate.get("house_scale_524288_update_tokens_reached", False)
        ),
        "post_learning_sustained_524288_boundary_reached": bool(
            gate.get("post_learning_sustained_524288_boundary_reached", False)
        ),
        "status_read_mutation_absent": bool(
            gate.get("status_read_mutation_absent", False)
        ),
        "promotes_runtime_claim": False,
    }


def _language_brain_installed_continual_learning_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_brain_installed_continual_learning_report(report)
    ]
    best_report = max(
        valid_reports,
        key=lambda item: (
            int(
                (item.get("learning_summary") or {}).get("update_token_count", 0)
                if isinstance(item.get("learning_summary"), Mapping)
                else 0
            ),
            float(
                (item.get("learning_summary") or {}).get("tokens_per_second", 0.0)
                if isinstance(item.get("learning_summary"), Mapping)
                else 0.0
            ),
        ),
        default=None,
    )
    supplied_invalid_reports = len(reports) > 0 and best_report is None
    return {
        "surface": "marulho_language_brain_installed_continual_learning_saved_evidence.v1",
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "brain_installed_continual_learning_available": best_report is not None,
        "best_report": (
            None
            if best_report is None
            else _brain_installed_continual_learning_summary(best_report)
        ),
        "missing_evidence": (
            ["valid_brain_installed_continual_learning_report"]
            if supplied_invalid_reports
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_runtime_claim": False,
    }


def _structural_plasticity_transactions_from_report(
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if (
        report.get("artifact_kind") == STRUCTURAL_PLASTICITY_TRANSACTION_ARTIFACT_KIND
        and report.get("surface") == STRUCTURAL_PLASTICITY_TRANSACTION_SURFACE
    ):
        return [dict(report)]
    if (
        report.get("artifact_kind") != STRUCTURAL_PLASTICITY_EXPERIMENT_ARTIFACT_KIND
        or report.get("surface") != STRUCTURAL_PLASTICITY_EXPERIMENT_SURFACE
    ):
        return []
    transactions = report.get("transactions")
    if not isinstance(transactions, Sequence) or isinstance(
        transactions,
        (str, bytes),
    ):
        return []
    extracted: list[dict[str, Any]] = []
    for entry in transactions:
        if not isinstance(entry, Mapping):
            continue
        transaction = entry.get("transaction")
        if isinstance(transaction, Mapping):
            payload = dict(transaction)
            payload.setdefault("path", report.get("path"))
            payload.setdefault("experiment_path", report.get("path"))
            extracted.append(payload)
    return extracted


def _structural_plasticity_transaction_summary(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    mutation = (
        report.get("mutation")
        if isinstance(report.get("mutation"), Mapping)
        else {}
    )
    evaluation = (
        report.get("evaluation")
        if isinstance(report.get("evaluation"), Mapping)
        else {}
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
        "path": str(report.get("experiment_path") or report.get("path") or ""),
        "proposal_kind": mutation.get("proposal_kind"),
        "status": report.get("status"),
        "applied": bool(report.get("applied", False)),
        "operator_approved": bool(report.get("operator_approved", False)),
        "checkpoint_path": checkpoint.get("path"),
        "checkpoint_restore_verified": bool(
            checkpoint.get("checkpoint_restore_verified", False)
        ),
        "rollback_verified": bool(rollback.get("rollback_verified", False)),
        "heldout_loss_delta": evaluation.get("heldout_loss_delta"),
        "source_expert_count": mutation.get("source_expert_count"),
        "target_expert_count": mutation.get("target_expert_count"),
        "source_route_candidate_count": mutation.get("source_route_candidate_count"),
        "target_route_candidate_count": mutation.get("target_route_candidate_count"),
        "source_memory_slot_count": mutation.get("source_memory_slot_count"),
        "target_memory_slot_count": mutation.get("target_memory_slot_count"),
        "target_memory_slot_candidate_count": mutation.get(
            "target_memory_slot_candidate_count"
        ),
        "target_active_memory_slot_count": mutation.get(
            "target_active_memory_slot_count"
        ),
        "eligible_for_reviewed_structural_promotion": bool(
            gate.get("eligible_for_reviewed_structural_promotion", False)
        ),
    }


def _language_structural_plasticity_saved_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transactions: list[dict[str, Any]] = []
    for report in reports:
        transactions.extend(_structural_plasticity_transactions_from_report(report))
    valid_transactions = [
        transaction
        for transaction in transactions
        if _valid_language_structural_plasticity_transaction_report(transaction)
    ]
    supplied_invalid = len(reports) > 0 and (
        not transactions or len(valid_transactions) != len(transactions)
    )
    proposal_kinds = sorted(
        {
            str((transaction.get("mutation") or {}).get("proposal_kind"))
            for transaction in valid_transactions
            if isinstance(transaction.get("mutation"), Mapping)
            and (transaction.get("mutation") or {}).get("proposal_kind")
        }
    )
    return {
        "report_count": len(reports),
        "transaction_count": len(transactions),
        "valid_transaction_count": len(valid_transactions),
        "saved_structural_plasticity_evidence_available": bool(valid_transactions),
        "proposal_kinds": proposal_kinds,
        "transaction_summaries": [
            _structural_plasticity_transaction_summary(transaction)
            for transaction in valid_transactions
        ],
        "missing_evidence": (
            ["valid_structural_plasticity_transaction_report"]
            if supplied_invalid
            else []
        ),
        "required_for_runtime_promotion": False,
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }


def _gpu_kernel_report_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    promotion_gate = (
        report.get("promotion_gate")
        if isinstance(report.get("promotion_gate"), Mapping)
        else {}
    )
    benchmark_summary = (
        report.get("benchmark_summary")
        if isinstance(report.get("benchmark_summary"), Mapping)
        else {}
    )
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "kernel_name": report.get("kernel_name"),
        "parity_passed": bool(report.get("parity_passed")),
        "valid_shape_result_count": int(report.get("valid_shape_result_count", 0) or 0),
        "dtype_coverage": list(report.get("dtype_coverage") or []),
        "geometric_speedup_vs_torch": benchmark_summary.get(
            "geometric_speedup_vs_torch"
        ),
        "complete_runtime_impact_available": bool(
            promotion_gate.get("complete_runtime_impact_available")
        ),
        "promotes_hot_path": bool(promotion_gate.get("promotes_hot_path")),
    }


def _language_gpu_kernel_evidence(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_reports = [
        dict(report)
        for report in reports
        if _valid_language_gpu_kernel_report(report)
    ]
    rmsnorm_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == RMSNORM_KERNEL_NAME
        ),
        None,
    )
    plif_forward_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == PLIF_FORWARD_KERNEL_NAME
        ),
        None,
    )
    plif_surrogate_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == PLIF_SURROGATE_KERNEL_NAME
        ),
        None,
    )
    selective_scan_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == SELECTIVE_SCAN_KERNEL_NAME
        ),
        None,
    )
    eligibility_trace_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == ELIGIBILITY_TRACE_KERNEL_NAME
        ),
        None,
    )
    route_topk_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == ROUTE_TOPK_KERNEL_NAME
        ),
        None,
    )
    expert_dispatch_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == EXPERT_DISPATCH_KERNEL_NAME
        ),
        None,
    )
    sampled_vocab_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == SAMPLED_VOCAB_CE_KERNEL_NAME
        ),
        None,
    )
    memory_slot_retrieval_report = next(
        (
            report
            for report in valid_reports
            if report.get("kernel_name") == MEMORY_SLOT_RETRIEVAL_KERNEL_NAME
        ),
        None,
    )
    missing = []
    if rmsnorm_report is None:
        missing.append("rmsnorm_triton_parity")
    if plif_forward_report is None:
        missing.append("plif_triton_forward_parity")
    if plif_surrogate_report is None:
        missing.append("plif_triton_backward_surrogate_parity")
    if selective_scan_report is None:
        missing.append("selective_scan_triton_parity")
    if eligibility_trace_report is None:
        missing.append("local_eligibility_trace_update_parity")
    if route_topk_report is None:
        missing.append("route_vote_topk_parity")
    if expert_dispatch_report is None:
        missing.append("block_sparse_expert_dispatch_parity")
    if sampled_vocab_report is None:
        missing.append("sampled_vocab_cross_entropy_parity")
    if memory_slot_retrieval_report is None:
        missing.append("bounded_memory_slot_retrieval_parity")
    return {
        "report_count": len(reports),
        "valid_report_count": len(valid_reports),
        "covered_kernel_names": sorted(
            {
                str(report.get("kernel_name"))
                for report in valid_reports
                if report.get("kernel_name")
            }
        ),
        "rmsnorm_triton_parity": rmsnorm_report is not None,
        "rmsnorm_report": (
            None if rmsnorm_report is None else _gpu_kernel_report_summary(rmsnorm_report)
        ),
        "plif_triton_forward_parity": plif_forward_report is not None,
        "plif_forward_report": (
            None
            if plif_forward_report is None
            else _gpu_kernel_report_summary(plif_forward_report)
        ),
        "plif_triton_backward_surrogate_parity": plif_surrogate_report is not None,
        "plif_surrogate_report": (
            None
            if plif_surrogate_report is None
            else _gpu_kernel_report_summary(plif_surrogate_report)
        ),
        "selective_scan_triton_parity": selective_scan_report is not None,
        "selective_scan_report": (
            None
            if selective_scan_report is None
            else _gpu_kernel_report_summary(selective_scan_report)
        ),
        "local_eligibility_trace_update_parity": eligibility_trace_report is not None,
        "local_eligibility_trace_report": (
            None
            if eligibility_trace_report is None
            else _gpu_kernel_report_summary(eligibility_trace_report)
        ),
        "route_vote_topk_parity": route_topk_report is not None,
        "route_topk_report": (
            None
            if route_topk_report is None
            else _gpu_kernel_report_summary(route_topk_report)
        ),
        "block_sparse_expert_dispatch_parity": expert_dispatch_report is not None,
        "expert_dispatch_report": (
            None
            if expert_dispatch_report is None
            else _gpu_kernel_report_summary(expert_dispatch_report)
        ),
        "sampled_vocab_cross_entropy_parity": sampled_vocab_report is not None,
        "sampled_vocab_cross_entropy_report": (
            None
            if sampled_vocab_report is None
            else _gpu_kernel_report_summary(sampled_vocab_report)
        ),
        "bounded_memory_slot_retrieval_parity": (
            memory_slot_retrieval_report is not None
        ),
        "bounded_memory_slot_retrieval_report": (
            None
            if memory_slot_retrieval_report is None
            else _gpu_kernel_report_summary(memory_slot_retrieval_report)
        ),
        "lm_triton_kernel_used": bool(valid_reports),
        "pytorch_fallback_available": True,
        "missing_evidence": missing,
        "promotes_hot_path": False,
    }


def run_language_runtime_benchmark_suite(
    *,
    output_path: str | Path,
    sustained_target_tokens: int = 8,
    sustained_evidence_paths: Sequence[str | Path] = (),
    brain_installed_continual_learning_evidence_paths: Sequence[str | Path] = (),
    memory_slot_runtime_impact_evidence_paths: Sequence[str | Path] = (),
    memory_slot_architecture_cost_evidence_paths: Sequence[str | Path] = (),
    structural_plasticity_evidence_paths: Sequence[str | Path] = (),
    gpu_kernel_evidence_paths: Sequence[str | Path] = (),
    generation_coherence_evidence_paths: Sequence[str | Path] = (),
    quality_replay_evidence_paths: Sequence[str | Path] = (),
    checkpoint_evolution_evidence_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Run a compact benchmark-suite smoke pass and write an evidence report."""

    torch.manual_seed(20260703)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    old_split = build_language_model_splits(
        [
            "old domain runtime truth protects replay evidence. " * 8,
            "checkpointed language state keeps rollback reviewable. " * 8,
        ],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    new_split = build_language_model_splits(
        [
            "new domain continual language learning updates owned weights. " * 8,
            "growth pressure should stay checkpoint backed and bounded. " * 8,
        ],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )

    categories: list[dict[str, Any]] = []
    base_model = _new_model(tokenizer)
    loss_result = base_model.next_token_loss(
        old_split.train[0].input_ids,
        old_split.train[0].target_ids,
    )
    loss_value = float(loss_result["loss"].detach().cpu().item())
    categories.append(
        _category(
            "next_token_loss",
            status="pass" if _finite_positive(loss_value) else "fail",
            evidence={
                "loss_kind": loss_result["loss_kind"],
                "loss": loss_value,
                "external_llm_used": False,
            },
        )
    )

    eval_report = evaluate_language_model(base_model, old_split.eval)
    categories.append(
        _category(
            "heldout_perplexity",
            status=(
                "pass"
                if _finite_positive(eval_report["heldout_loss"])
                and _finite_positive(eval_report["heldout_perplexity"])
                else "fail"
            ),
            evidence={
                "heldout_loss": eval_report["heldout_loss"],
                "heldout_perplexity": eval_report["heldout_perplexity"],
                "eval_token_count": eval_report["eval_token_count"],
            },
        )
    )

    prompt = torch.tensor(tokenizer.encode("marulho", add_eos=False), dtype=torch.long)
    generation = base_model.generate(prompt, max_new_tokens=4, eos_id=tokenizer.eos_id)
    generation_coherence_reports = [
        _read_generation_coherence_report(path)
        for path in generation_coherence_evidence_paths
    ]
    generation_coherence_evidence = _language_generation_coherence_evidence(
        generation_coherence_reports
    )
    quality_replay_reports = [
        _read_quality_replay_report(path) for path in quality_replay_evidence_paths
    ]
    quality_replay_evidence = _language_quality_replay_evidence(
        quality_replay_reports
    )
    checkpoint_evolution_reports = [
        _read_checkpoint_evolution_report(path)
        for path in checkpoint_evolution_evidence_paths
    ]
    saved_checkpoint_evolution_evidence = (
        _language_checkpoint_evolution_saved_evidence(
            checkpoint_evolution_reports
        )
    )
    generation_coherence_missing = tuple(
        generation_coherence_evidence["missing_evidence"]
        + quality_replay_evidence["missing_evidence"]
    )
    generation_category = _category(
        "generation_coherence",
        status="pass" if not generation_coherence_missing else "smoke_only",
        evidence={
            "generated_token_count": int(generation["new_token_count"]),
            "active_language_path": generation["active_language_path"],
            "external_llm_used": generation["external_llm_used"],
            "review_kind": "token_stream_smoke_not_human_quality_review",
            **generation_coherence_evidence,
            "quality_replay_evidence": quality_replay_evidence,
        },
        missing=generation_coherence_missing,
    )
    categories.append(generation_category)

    grounding_report = run_language_grounding_support_report(
        base_model,
        tokenizer,
        prompt_text="runtime truth replay evidence",
        source_text=(
            "Runtime truth records replay evidence for MARULHO-owned language "
            "support. Source windows keep checkpointed evidence reviewable."
        ),
        required_terms=("runtime", "truth", "replay", "evidence"),
        max_new_tokens=0,
        output_path=output.parent / "language-suite-grounding-support.json",
    )
    grounding_gate = grounding_report["promotion_gate"]
    categories.append(
        _category(
            "grounding_support",
            status=(
                "pass"
                if grounding_gate["grounding_support_available"]
                else "fail"
            ),
            evidence={
                "grounding_surface": grounding_report["surface"],
                "grounding_support_report": str(
                    output.parent / "language-suite-grounding-support.json"
                ),
                "source_term_coverage": grounding_report["source_terms"][
                    "source_term_coverage"
                ],
                "matched_required_terms": grounding_report["source_terms"][
                    "matched_required_terms"
                ],
                "missing_required_terms": grounding_report["source_terms"][
                    "missing_required_terms"
                ],
                "source_term_coverage_gate_passed": grounding_gate[
                    "source_term_coverage_gate_passed"
                ],
                "active_language_path": grounding_report["generation"][
                    "active_language_path"
                ],
                "external_llm_used": grounding_report["generation"][
                    "external_llm_used"
                ],
                "review_kind": grounding_report["generation"]["review_kind"],
            },
            missing=(
                ()
                if grounding_gate["grounding_support_available"]
                else ("grounding_support_report", "source_term_coverage_gate")
            ),
        )
    )

    brain_installed_learning_reports = [
        _read_brain_installed_continual_learning_report(path)
        for path in brain_installed_continual_learning_evidence_paths
    ]
    brain_installed_learning_evidence = (
        _language_brain_installed_continual_learning_evidence(
            brain_installed_learning_reports
        )
    )
    brain_installed_learning_missing = tuple(
        brain_installed_learning_evidence["missing_evidence"]
    )
    best_brain_learning = (
        brain_installed_learning_evidence.get("best_report")
        if isinstance(brain_installed_learning_evidence.get("best_report"), Mapping)
        else {}
    )
    learning_model = _new_model(tokenizer)
    learning_report = run_language_continual_learning_window(
        learning_model,
        new_batches=new_split.train[:2],
        old_eval_batches=old_split.eval,
        new_eval_batches=new_split.train[:2],
        replay_batches=old_split.train[:1],
        config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=2,
            replay_loss_weight=0.25,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
            rollback_on_forgetting=False,
        ),
    )
    learning_evidence = learning_report["learning_evidence"]
    categories.append(
        _category(
            "continual_learning",
            status=(
                "fail"
                if brain_installed_learning_missing
                else (
                    "pass"
                    if learning_report["rollback_evidence"]["restore_verified"]
                    and learning_evidence["final_parameter_delta_l2"] > 0.0
                    else "fail"
                )
            ),
            evidence={
                "new_domain_loss_delta": learning_evidence["new_domain_loss_delta"],
                "final_parameter_delta_l2": learning_evidence["final_parameter_delta_l2"],
                "tokens_per_second": learning_evidence["tokens_per_second"],
                "rollback_available": learning_report["promotion_gate"]["rollback_available"],
                "brain_installed_continual_learning_evidence": (
                    brain_installed_learning_evidence
                ),
                "brain_installed_learning_available": bool(
                    brain_installed_learning_evidence[
                        "brain_installed_continual_learning_available"
                    ]
                ),
                "brain_installed_update_token_count": int(
                    best_brain_learning.get("update_token_count", 0) or 0
                ),
                "brain_installed_tokens_per_second": float(
                    best_brain_learning.get("tokens_per_second", 0.0) or 0.0
                ),
                "brain_installed_total_window_tokens_per_second": float(
                    best_brain_learning.get(
                        "total_window_tokens_per_second",
                        0.0,
                    )
                    or 0.0
                ),
                "brain_installed_learned_checkpoint_restore_verified": bool(
                    best_brain_learning.get(
                        "learned_brain_checkpoint_restore_verified",
                        False,
                    )
                ),
                "brain_installed_post_learning_sustained_tokens_per_second": float(
                    best_brain_learning.get(
                        "post_learning_sustained_tokens_per_second",
                        0.0,
                    )
                    or 0.0
                ),
            },
            missing=brain_installed_learning_missing,
        )
    )
    categories.append(
        _category(
            "forgetting",
            status=(
                "fail"
                if brain_installed_learning_missing
                else "pass"
                if "old_domain_forgetting" in learning_evidence
                else "fail"
            ),
            evidence={
                "old_domain_forgetting": learning_evidence["old_domain_forgetting"],
                "old_domain_forgetting_within_tolerance": learning_report[
                    "promotion_gate"
                ]["old_domain_forgetting_within_tolerance"],
                "brain_installed_old_domain_forgetting": (
                    best_brain_learning.get("old_domain_forgetting")
                ),
                "brain_installed_forgetting_measured": bool(
                    brain_installed_learning_evidence[
                        "brain_installed_continual_learning_available"
                    ]
                ),
                "brain_installed_continual_learning_evidence": (
                    brain_installed_learning_evidence
                ),
            },
            missing=brain_installed_learning_missing,
        )
    )
    categories.append(
        _category(
            "replay_recovery",
            status=(
                "fail"
                if brain_installed_learning_missing
                else (
                    "pass"
                    if "general_replay_retention_delta" in learning_evidence
                    else "fail"
                )
            ),
            evidence={
                "general_replay_retention_delta": learning_evidence[
                    "general_replay_retention_delta"
                ],
                "general_replay_retention_within_tolerance": learning_report[
                    "promotion_gate"
                ]["general_replay_retention_within_tolerance"],
                "brain_installed_general_replay_retention_delta": (
                    best_brain_learning.get("general_replay_retention_delta")
                ),
                "brain_installed_replay_retention_measured": bool(
                    brain_installed_learning_evidence[
                        "brain_installed_continual_learning_available"
                    ]
                ),
                "brain_installed_memory_slot_candidate_slots_scored": int(
                    best_brain_learning.get(
                        "memory_slot_candidate_slots_scored",
                        0,
                    )
                    or 0
                ),
                "brain_installed_memory_slot_runs_all_slots": bool(
                    best_brain_learning.get("memory_slot_runs_all_slots", False)
                ),
                "brain_installed_continual_learning_evidence": (
                    brain_installed_learning_evidence
                ),
            },
            missing=brain_installed_learning_missing,
        )
    )

    structural_growth_model = _new_model(tokenizer)
    routing_evidence = {
        "surface": "marulho_routed_language_experts.v1",
        "total_columns": 2,
        "active_columns": 2,
        "candidate_rows_scored": 20,
        "runs_all_columns": False,
    }
    proposal = build_language_structural_plasticity_proposal(
        structural_growth_model,
        routing_evidence=routing_evidence,
        config=LanguageStructuralPlasticityConfig(route_saturation_threshold=0.5),
    )
    structural_candidate, structural_report = apply_language_structural_plasticity_transaction(
        structural_growth_model,
        proposal,
        eval_batches=old_split.eval,
        checkpoint_path=output.parent / "language-suite-structure-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(max_eval_loss_delta=100.0),
    )
    structural_split_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
        )
    )
    split_proposal = build_language_structural_column_split_proposal(
        structural_split_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "split_candidate_expert_ids": [1],
            "expert_loads": [0.1, 0.95, 0.2, 0.3],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_added_experts=2,
            max_split_experts=1,
            split_load_threshold=0.8,
        ),
    )
    structural_split_candidate, structural_split_report = (
        apply_language_structural_plasticity_transaction(
            structural_split_model,
            split_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-column-split-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                max_added_experts=2,
                max_split_experts=1,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_prune_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
        )
    )
    prune_proposal = build_language_structural_prune_proposal(
        structural_prune_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "active_expert_ids": [0],
            "inactive_expert_ids": [2],
            "expert_utilities": [0.7, 0.2, 0.0],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_pruned_experts=1,
            prune_utility_threshold=0.05,
        ),
    )
    structural_pruned_candidate, structural_prune_report = (
        apply_language_structural_plasticity_transaction(
            structural_prune_model,
            prune_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-prune-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                min_expert_count=2,
                max_pruned_experts=1,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_retire_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
        )
    )
    retire_proposal = build_language_structural_retire_proposal(
        structural_retire_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "retire_candidate_expert_ids": [3],
            "dead_spike_expert_ids": [3],
            "expert_utilities": [0.8, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_retired_experts=1,
            prune_utility_threshold=0.05,
        ),
    )
    structural_retire_candidate, structural_retire_report = (
        apply_language_structural_plasticity_transaction(
            structural_retire_model,
            retire_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-retire-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                min_expert_count=2,
                max_retired_experts=1,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_merge_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
        )
    )
    merge_proposal = build_language_structural_merge_proposal(
        structural_merge_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "duplicate_expert_pairs": [[1, 2]],
            "expert_pair_similarities": {"1,2": 0.99},
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_merged_expert_pairs=1,
            merge_similarity_threshold=0.95,
        ),
    )
    structural_merged_candidate, structural_merge_report = (
        apply_language_structural_plasticity_transaction(
            structural_merge_model,
            merge_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-merge-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                min_expert_count=2,
                max_merged_expert_pairs=1,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_sleep_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=4,
            expert_hidden_dim=32,
        )
    )
    sleep_proposal = build_language_structural_deep_sleep_proposal(
        structural_sleep_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "stale_expert_ids": [3],
            "low_activation_expert_ids": [3],
            "expert_utilities": [0.7, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_deep_sleep_experts=1,
            deep_sleep_utility_threshold=0.10,
        ),
    )
    structural_sleep_candidate, structural_sleep_report = (
        apply_language_structural_plasticity_transaction(
            structural_sleep_model,
            sleep_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-deep-sleep-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                min_expert_count=2,
                max_deep_sleep_experts=1,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_route_bank_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=5,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
        )
    )
    route_bank_proposal = build_language_structural_route_bank_expansion_proposal(
        structural_route_bank_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 5,
            "active_columns": 2,
            "route_candidate_count": 2,
            "output_candidate_count": 1,
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
            "route_bank_pressure": True,
        },
        config=LanguageStructuralPlasticityConfig(
            route_saturation_threshold=0.5,
            max_route_candidate_growth=2,
        ),
    )
    structural_route_bank_candidate, structural_route_bank_report = (
        apply_language_structural_plasticity_transaction(
            structural_route_bank_model,
            route_bank_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-route-bank-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                max_route_candidate_growth=2,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_synapse_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
        )
    )
    synapse_bundle_proposal = build_language_structural_synapse_bundle_proposal(
        structural_synapse_model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "synapse_bundle_pressure": True,
            "high_surprise_expert_ids": [1],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_synapse_bundle_hidden_growth=8,
        ),
    )
    structural_synapse_candidate, structural_synapse_report = (
        apply_language_structural_plasticity_transaction(
            structural_synapse_model,
            synapse_bundle_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-synapse-bundle-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                max_synapse_bundle_hidden_growth=8,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    structural_memory_model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
            memory_slot_count=0,
            memory_slot_candidate_count=0,
            active_memory_slot_count=1,
        )
    )
    memory_slot_proposal = build_language_structural_memory_slot_expansion_proposal(
        structural_memory_model,
        routing_evidence={
            "surface": "marulho_language_memory_slots.v1",
            "memory_slot_pressure": True,
            "novel_concept_cluster": True,
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_memory_slot_growth=4,
            max_memory_slot_candidate_count=2,
        ),
    )
    structural_memory_candidate, structural_memory_report = (
        apply_language_structural_plasticity_transaction(
            structural_memory_model,
            memory_slot_proposal,
            eval_batches=old_split.eval,
            checkpoint_path=output.parent / "language-suite-memory-slot-baseline.pt",
            operator_approved=True,
            config=LanguageStructuralPlasticityConfig(
                max_memory_slot_growth=4,
                max_memory_slot_candidate_count=2,
                max_eval_loss_delta=100.0,
            ),
        )
    )
    sleep_eval = evaluate_language_model(structural_sleep_candidate, old_split.eval)
    sleep_routing = sleep_eval["spike_telemetry"]["routing"]
    route_bank_eval = evaluate_language_model(
        structural_route_bank_candidate,
        old_split.eval,
    )
    route_bank_routing = route_bank_eval["spike_telemetry"]["routing"]
    memory_eval = evaluate_language_model(structural_memory_candidate, old_split.eval)
    memory_slot_evidence = memory_eval["spike_telemetry"]["memory"]
    structural_plasticity_reports = [
        _read_structural_plasticity_evidence_report(path)
        for path in structural_plasticity_evidence_paths
    ]
    saved_structural_plasticity_evidence = (
        _language_structural_plasticity_saved_evidence(
            structural_plasticity_reports
        )
    )
    saved_structural_missing = tuple(
        saved_structural_plasticity_evidence["missing_evidence"]
    )
    categories.append(
        _category(
            "growth_prune_safety",
            status=(
                "fail"
                if saved_structural_missing
                else (
                "pass"
                if proposal["mutates_runtime_state"] is False
                and structural_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_report["rollback_evidence"]["rollback_verified"]
                and split_proposal["mutates_runtime_state"] is False
                and structural_split_report["checkpoint"][
                    "checkpoint_restore_verified"
                ]
                and structural_split_report["rollback_evidence"]["rollback_verified"]
                and prune_proposal["mutates_runtime_state"] is False
                and structural_prune_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_prune_report["rollback_evidence"]["rollback_verified"]
                and retire_proposal["mutates_runtime_state"] is False
                and structural_retire_report["checkpoint"][
                    "checkpoint_restore_verified"
                ]
                and structural_retire_report["rollback_evidence"]["rollback_verified"]
                and merge_proposal["mutates_runtime_state"] is False
                and structural_merge_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_merge_report["rollback_evidence"]["rollback_verified"]
                and sleep_proposal["mutates_runtime_state"] is False
                and structural_sleep_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_sleep_report["rollback_evidence"]["rollback_verified"]
                and route_bank_proposal["mutates_runtime_state"] is False
                and structural_route_bank_report["checkpoint"][
                    "checkpoint_restore_verified"
                ]
                and structural_route_bank_report["rollback_evidence"][
                    "rollback_verified"
                ]
                and not route_bank_routing["runs_all_columns"]
                and synapse_bundle_proposal["mutates_runtime_state"] is False
                and structural_synapse_report["checkpoint"][
                    "checkpoint_restore_verified"
                ]
                and structural_synapse_report["rollback_evidence"][
                    "rollback_verified"
                ]
                and memory_slot_proposal["mutates_runtime_state"] is False
                and structural_memory_report["checkpoint"][
                    "checkpoint_restore_verified"
                ]
                and structural_memory_report["rollback_evidence"][
                    "rollback_verified"
                ]
                and not memory_slot_evidence["runs_all_slots"]
                else "fail"
                )
            ),
            evidence={
                "proposal_mutates_runtime_state": proposal["mutates_runtime_state"],
                "growth_transaction_applied": structural_report["applied"],
                "growth_checkpoint_backed": structural_report["promotion_gate"]["checkpoint_backed"],
                "growth_rollback_verified": structural_report["rollback_evidence"]["rollback_verified"],
                "growth_target_expert_count": structural_candidate.config.expert_count,
                "column_split_proposal_mutates_runtime_state": split_proposal[
                    "mutates_runtime_state"
                ],
                "column_split_transaction_applied": structural_split_report[
                    "applied"
                ],
                "column_split_checkpoint_backed": structural_split_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "column_split_rollback_verified": structural_split_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "column_split_source_expert_count": structural_split_report[
                    "mutation"
                ]["source_expert_count"],
                "column_split_target_expert_count": structural_split_candidate.config.expert_count,
                "column_split_parent_child_pairs": structural_split_report[
                    "mutation"
                ]["parent_child_expert_pairs"],
                "prune_proposal_mutates_runtime_state": prune_proposal[
                    "mutates_runtime_state"
                ],
                "prune_transaction_applied": structural_prune_report["applied"],
                "prune_checkpoint_backed": structural_prune_report["promotion_gate"][
                    "checkpoint_backed"
                ],
                "prune_rollback_verified": structural_prune_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "prune_target_expert_count": structural_pruned_candidate.config.expert_count,
                "retire_proposal_mutates_runtime_state": retire_proposal[
                    "mutates_runtime_state"
                ],
                "retire_transaction_applied": structural_retire_report[
                    "applied"
                ],
                "retire_checkpoint_backed": structural_retire_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "retire_rollback_verified": structural_retire_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "retire_target_expert_count": structural_retire_candidate.config.expert_count,
                "retired_expert_ids": structural_retire_report["mutation"][
                    "retired_expert_ids"
                ],
                "merge_proposal_mutates_runtime_state": merge_proposal[
                    "mutates_runtime_state"
                ],
                "merge_transaction_applied": structural_merge_report["applied"],
                "merge_checkpoint_backed": structural_merge_report["promotion_gate"][
                    "checkpoint_backed"
                ],
                "merge_rollback_verified": structural_merge_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "merge_target_expert_count": structural_merged_candidate.config.expert_count,
                "deep_sleep_proposal_mutates_runtime_state": sleep_proposal[
                    "mutates_runtime_state"
                ],
                "deep_sleep_transaction_applied": structural_sleep_report["applied"],
                "deep_sleep_checkpoint_backed": structural_sleep_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "deep_sleep_rollback_verified": structural_sleep_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "deep_sleep_target_expert_count": (
                    structural_sleep_candidate.config.expert_count
                ),
                "deep_sleep_awake_expert_count": sleep_routing["awake_columns"],
                "deep_sleep_sleeping_expert_ids": sleep_routing[
                    "sleeping_expert_ids"
                ],
                "deep_sleep_candidate_rows_scored": sleep_routing[
                    "candidate_rows_scored"
                ],
                "deep_sleep_runs_all_columns": sleep_routing["runs_all_columns"],
                "route_bank_proposal_mutates_runtime_state": route_bank_proposal[
                    "mutates_runtime_state"
                ],
                "route_bank_transaction_applied": structural_route_bank_report[
                    "applied"
                ],
                "route_bank_checkpoint_backed": structural_route_bank_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "route_bank_rollback_verified": structural_route_bank_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "route_bank_source_candidate_count": structural_route_bank_report[
                    "mutation"
                ]["source_route_candidate_count"],
                "route_bank_target_candidate_count": structural_route_bank_report[
                    "mutation"
                ]["target_route_candidate_count"],
                "route_bank_candidate_rows_scored": route_bank_routing[
                    "candidate_rows_scored"
                ],
                "route_bank_runs_all_columns": route_bank_routing[
                    "runs_all_columns"
                ],
                "route_bank_candidate_id_source": route_bank_routing[
                    "candidate_id_source"
                ],
                "synapse_bundle_proposal_mutates_runtime_state": synapse_bundle_proposal[
                    "mutates_runtime_state"
                ],
                "synapse_bundle_transaction_applied": structural_synapse_report[
                    "applied"
                ],
                "synapse_bundle_checkpoint_backed": structural_synapse_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "synapse_bundle_rollback_verified": structural_synapse_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "synapse_bundle_source_hidden_dim": structural_synapse_report[
                    "mutation"
                ]["source_expert_hidden_dim"],
                "synapse_bundle_target_hidden_dim": structural_synapse_report[
                    "mutation"
                ]["target_expert_hidden_dim"],
                "synapse_bundle_hidden_growth": structural_synapse_report[
                    "mutation"
                ]["synapse_bundle_hidden_growth"],
                "memory_slot_proposal_mutates_runtime_state": memory_slot_proposal[
                    "mutates_runtime_state"
                ],
                "memory_slot_transaction_applied": structural_memory_report[
                    "applied"
                ],
                "memory_slot_checkpoint_backed": structural_memory_report[
                    "promotion_gate"
                ]["checkpoint_backed"],
                "memory_slot_rollback_verified": structural_memory_report[
                    "rollback_evidence"
                ]["rollback_verified"],
                "memory_slot_source_count": structural_memory_report["mutation"][
                    "source_memory_slot_count"
                ],
                "memory_slot_target_count": structural_memory_report["mutation"][
                    "target_memory_slot_count"
                ],
                "memory_slot_candidate_count": memory_slot_evidence[
                    "candidate_slot_count"
                ],
                "memory_slot_active_count": memory_slot_evidence[
                    "active_slots_per_token"
                ],
                "memory_slot_candidate_slots_scored": memory_slot_evidence[
                    "candidate_slots_scored"
                ],
                "memory_slot_runs_all_slots": memory_slot_evidence["runs_all_slots"],
                "memory_slot_candidate_id_source": memory_slot_evidence[
                    "candidate_id_source"
                ],
                "saved_structural_plasticity_evidence": (
                    saved_structural_plasticity_evidence
                ),
            },
            missing=saved_structural_missing,
        )
    )

    sustained_model = _new_model(tokenizer)
    sustained_report = run_language_sustained_runtime_evidence(
        sustained_model,
        tokenizer,
        output_path=output.parent / "language-suite-sustained-smoke.json",
        target_tokens=max(1, int(sustained_target_tokens)),
        prompt="marulho",
        tick_tokens=4,
        quantum_tokens=2,
        timeout_seconds=30.0,
        collect_environment=False,
    )
    sustained_evidence_reports = [
        _read_sustained_report(path) for path in sustained_evidence_paths
    ]
    sustained_long_run_evidence = _language_long_run_evidence(sustained_evidence_reports)
    long_run_missing = tuple(sustained_long_run_evidence["missing_evidence"])
    long_run_status = "pass" if not long_run_missing else "smoke_only"
    long_run_evidence = {
        "smoke_report_status": sustained_report["report_status"],
        "smoke_target_tokens": sustained_report["target_tokens"],
        "smoke_token_delta": sustained_report["token_delta"],
        "smoke_tokens_per_second": sustained_report["tokens_per_second"],
        "smoke_short_run_is_smoke_only": sustained_report["promotion_gate"][
            "short_run_is_smoke_only"
        ],
        **sustained_long_run_evidence,
    }
    categories.append(
        _category(
            "long_run_throughput",
            status=long_run_status,
            evidence=long_run_evidence,
            missing=long_run_missing,
        )
    )
    generation_long_run_alignment = _generation_long_run_alignment_evidence(
        generation_coherence_evidence,
        long_run_evidence,
    )
    generation_category["evidence"]["long_run_alignment"] = (
        generation_long_run_alignment
    )
    if (
        generation_coherence_evidence["generation_coherence_available"]
        and not long_run_missing
        and generation_long_run_alignment["missing_evidence"]
    ):
        generation_category["status"] = "smoke_only"
        generation_category["missing_evidence"].extend(
            generation_long_run_alignment["missing_evidence"]
        )
    if quality_replay_evidence["quality_replay_available"]:
        quality_replay_long_run_alignment = (
            _quality_replay_long_run_alignment_evidence(
                quality_replay_evidence,
                long_run_evidence,
            )
        )
        generation_category["evidence"]["quality_replay_long_run_alignment"] = (
            quality_replay_long_run_alignment
        )
        if (
            not long_run_missing
            and quality_replay_long_run_alignment["missing_evidence"]
        ):
            generation_category["status"] = "smoke_only"
            generation_category["missing_evidence"].extend(
                quality_replay_long_run_alignment["missing_evidence"]
            )
    else:
        generation_category["evidence"]["quality_replay_long_run_alignment"] = {
            "surface": "marulho_language_quality_replay_long_run_alignment.v1",
            "same_child_long_run_available": False,
            "same_child_house_scale_available": False,
            "same_child_controlled_decode_available": False,
            "same_child_controlled_decode_house_scale_available": False,
            "controlled_decode_house_scale_required": bool(
                long_run_evidence.get("controlled_decode_house_scale_gate_reached")
            ),
            "matching_report_count": 0,
            "missing_evidence": [],
        }
    quality_replay_long_run_alignment = generation_category["evidence"][
        "quality_replay_long_run_alignment"
    ]

    parameter_estimate = estimate_language_model_parameters(base_model.config)
    categories.append(
        _category(
            "active_compute",
            status="pass",
            evidence={
                "total_parameters": parameter_estimate["total_parameters"],
                "active_parameters_per_token_estimate": parameter_estimate[
                    "active_parameters_per_token_estimate"
                ],
                "active_parameter_fraction_estimate": parameter_estimate[
                    "active_parameter_fraction_estimate"
                ],
                "active_expert_count_per_token": parameter_estimate[
                    "active_expert_count_per_token"
                ],
            },
        )
    )

    memory_slot_runtime_impact_reports = [
        _read_memory_slot_runtime_impact_report(path)
        for path in memory_slot_runtime_impact_evidence_paths
    ]
    memory_slot_runtime_impact_evidence = (
        _language_memory_slot_runtime_impact_evidence(
            memory_slot_runtime_impact_reports
        )
    )
    memory_slot_runtime_impact_missing = tuple(
        memory_slot_runtime_impact_evidence["missing_evidence"]
    )
    categories.append(
        _category(
            "memory_slot_runtime_impact",
            status=(
                "fail"
                if memory_slot_runtime_impact_missing
                else (
                    "pass"
                    if memory_slot_runtime_impact_evidence[
                        "memory_slot_runtime_impact_available"
                    ]
                    else "smoke_only"
                )
            ),
            evidence=memory_slot_runtime_impact_evidence,
            missing=memory_slot_runtime_impact_missing,
        )
    )

    memory_slot_architecture_cost_reports = [
        _read_memory_slot_architecture_cost_report(path)
        for path in memory_slot_architecture_cost_evidence_paths
    ]
    memory_slot_architecture_cost_evidence = (
        _language_memory_slot_architecture_cost_evidence(
            memory_slot_architecture_cost_reports
        )
    )
    memory_slot_architecture_cost_missing = tuple(
        memory_slot_architecture_cost_evidence["missing_evidence"]
    )
    categories.append(
        _category(
            "memory_slot_architecture_cost",
            status=(
                "fail"
                if memory_slot_architecture_cost_missing
                else (
                    "pass"
                    if memory_slot_architecture_cost_evidence[
                        "memory_slot_architecture_cost_available"
                    ]
                    else "smoke_only"
                )
            ),
            evidence=memory_slot_architecture_cost_evidence,
            missing=memory_slot_architecture_cost_missing,
        )
    )

    gpu_kernel_reports = [
        _read_gpu_kernel_report(path) for path in gpu_kernel_evidence_paths
    ]
    gpu_kernel_evidence = _language_gpu_kernel_evidence(gpu_kernel_reports)
    gpu_kernel_missing = tuple(gpu_kernel_evidence["missing_evidence"])
    categories.append(
        _category(
            "gpu_kernel_correctness",
            status="pass" if not gpu_kernel_missing else "missing",
            evidence=gpu_kernel_evidence,
            missing=gpu_kernel_missing,
        )
    )

    checkpoint_path = save_language_model_checkpoint(
        output.parent / "language-suite-checkpoint.pt",
        base_model,
        tokenizer,
        metadata={"suite": "language_runtime_benchmark"},
    )
    restored_model, restored_tokenizer, restored_metadata = load_language_model_checkpoint(
        checkpoint_path
    )
    restored_eval = evaluate_language_model(restored_model, old_split.eval)
    categories.append(
        _category(
            "checkpoint_restore",
            status=(
                "fail"
                if brain_installed_learning_missing
                else (
                    "pass"
                    if restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
                    and restored_metadata.get("suite") == "language_runtime_benchmark"
                    and _finite_positive(restored_eval["heldout_loss"])
                    else "fail"
                )
            ),
            evidence={
                "checkpoint_path": str(checkpoint_path),
                "tokenizer_hash_restored": restored_tokenizer.vocabulary_hash(),
                "metadata_restored": dict(restored_metadata),
                "heldout_loss_after_restore": restored_eval["heldout_loss"],
                "brain_installed_continual_learning_evidence": (
                    brain_installed_learning_evidence
                ),
                "brain_installed_pre_learning_checkpoint_restore_verified": bool(
                    best_brain_learning.get(
                        "pre_learning_checkpoint_restore_verified",
                        False,
                    )
                ),
                "brain_installed_learned_checkpoint_restore_verified": bool(
                    best_brain_learning.get(
                        "learned_brain_checkpoint_restore_verified",
                        False,
                    )
                ),
                "brain_installed_learned_checkpoint_path": (
                    best_brain_learning.get("learned_brain_checkpoint_path")
                ),
            },
            missing=brain_installed_learning_missing,
        )
    )

    evolution_model = _new_model(tokenizer)
    _child, evolution_report = run_language_checkpoint_evolution(
        evolution_model,
        tokenizer,
        eval_batches=old_split.eval,
        child_train_batches=new_split.train[:2],
        child_new_eval_batches=new_split.train[:2],
        replay_batches=old_split.train[:1],
        checkpoint_dir=output.parent / "language-suite-evolution",
        config=LanguageCheckpointEvolutionConfig(
            max_child_loss_delta=100.0,
            max_old_domain_forgetting=100.0,
            require_child_learning=False,
            allow_structural_growth=False,
        ),
        learning_config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=1,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
            rollback_on_forgetting=False,
        ),
    )
    evolution_lineage = (
        evolution_report.get("checkpoint_lineage")
        if isinstance(evolution_report.get("checkpoint_lineage"), Mapping)
        else {}
    )
    evolution_review = (
        evolution_report.get("evolution_review")
        if isinstance(evolution_report.get("evolution_review"), Mapping)
        else {}
    )
    evolution_runtime = (
        evolution_report.get("runtime_evidence")
        if isinstance(evolution_report.get("runtime_evidence"), Mapping)
        else {}
    )
    saved_checkpoint_evolution_missing = tuple(
        saved_checkpoint_evolution_evidence["missing_evidence"]
    )
    categories.append(
        _category(
            "rollback",
            status=(
                "fail"
                if saved_checkpoint_evolution_missing
                else (
                    "pass"
                    if evolution_report["promotion_gate"][
                        "rollback_to_parent_verified"
                    ]
                    and evolution_report["promotion_gate"]["parent_runtime_unchanged"]
                    and evolution_report["promotion_gate"][
                        "checkpoint_lineage_complete"
                    ]
                    and evolution_lineage.get("lineage_complete") is True
                    and evolution_review.get("parent_kept_installed") is True
                    and evolution_review.get("isolated_child_training") is True
                    else "fail"
                )
            ),
            evidence={
                "rollback_to_parent_verified": evolution_report["promotion_gate"][
                    "rollback_to_parent_verified"
                ],
                "parent_runtime_unchanged": evolution_report["promotion_gate"][
                    "parent_runtime_unchanged"
                ],
                "lineage_id": evolution_report["lineage"]["lineage_id"],
                "checkpoint_lineage_complete": evolution_report["promotion_gate"][
                    "checkpoint_lineage_complete"
                ],
                "child_initial_matches_parent_state": evolution_lineage.get(
                    "child_initial_matches_parent_state"
                ),
                "child_final_matches_child_runtime": evolution_lineage.get(
                    "child_final_matches_child_runtime"
                ),
                "child_final_differs_from_parent_state": evolution_lineage.get(
                    "child_final_differs_from_parent_state"
                ),
                "parent_checkpoint_sha256": evolution_lineage.get(
                    "parent_checkpoint_sha256"
                ),
                "child_final_checkpoint_sha256": evolution_lineage.get(
                    "child_final_checkpoint_sha256"
                ),
                "parent_kept_installed": evolution_review.get("parent_kept_installed"),
                "isolated_child_training": evolution_review.get(
                    "isolated_child_training"
                ),
                "child_update_token_count": evolution_review.get(
                    "child_update_token_count"
                ),
                "operator_review_required": evolution_review.get(
                    "operator_review_required"
                ),
                "long_run_evidence_required_for_promotion": evolution_review.get(
                    "long_run_evidence_required_for_promotion"
                ),
                "child_training_device": evolution_runtime.get("child_training_device"),
                "child_training_dense_adamw_backend": evolution_runtime.get(
                    "child_training_dense_adamw_backend"
                ),
                "checkpoint_storage_device": evolution_runtime.get(
                    "checkpoint_storage_device"
                ),
                "saved_checkpoint_evolution_evidence": (
                    saved_checkpoint_evolution_evidence
                ),
            },
            missing=saved_checkpoint_evolution_missing,
        )
    )

    brain = MarulhoBrain.fresh(_tiny_brain_config())
    brain.install_language_model(
        _new_model(tokenizer),
        tokenizer,
        evaluation_report=eval_report,
    )
    before_status = brain.status()
    after_status = brain.status()
    generation_status = brain.generate(prompt="marulho", max_tokens=2)
    categories.append(
        _category(
            "service_contract",
            status=(
                "pass"
                if before_status["token_count"] == after_status["token_count"]
                and before_status["active_language_path"] == "marulho_lm_head"
                and generation_status["external_llm_used"] is False
                else "fail"
            ),
            evidence={
                "status_read_mutates_token_count": before_status["token_count"]
                != after_status["token_count"],
                "active_language_path": before_status["active_language_path"],
                "generation_external_llm_used": generation_status["external_llm_used"],
                "service_owner": "thin_brain_adapter",
            },
        )
    )

    scale_report = build_language_scale_ladder_report(
        output_path=output.parent / "language-suite-scale-ladder.json",
        smoke_model=_new_model(tokenizer),
        smoke_tokenizer=tokenizer,
        smoke_eval_batches=old_split.eval,
    )
    categories.append(
        _category(
            "scale_ladder",
            status="pass" if scale_report["entry_count"] >= 5 else "fail",
            evidence={
                "entry_count": scale_report["entry_count"],
                "large_ladders_trained": scale_report["promotion_gate"][
                    "large_ladders_trained"
                ],
                "frontier_competitiveness_claimed": scale_report["promotion_gate"][
                    "frontier_competitiveness_claimed"
                ],
            },
        )
    )

    missing_categories = [
        item for item in categories if item["status"] == "missing" or item["missing_evidence"]
    ]
    failed_categories = [item for item in categories if item["status"] == "fail"]
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "category_count": len(categories),
        "passed_or_smoke_category_count": len(
            [item for item in categories if item["passed"]]
        ),
        "missing_category_count": len(missing_categories),
        "failed_category_count": len(failed_categories),
        "categories": categories,
        "subreports": {
            "grounding_support": str(
                output.parent / "language-suite-grounding-support.json"
            ),
            "sustained_smoke": str(output.parent / "language-suite-sustained-smoke.json"),
            "sustained_evidence": [
                str(Path(path)) for path in sustained_evidence_paths
            ],
            "brain_installed_continual_learning_evidence": [
                str(Path(path))
                for path in brain_installed_continual_learning_evidence_paths
            ],
            "memory_slot_runtime_impact_evidence": [
                str(Path(path))
                for path in memory_slot_runtime_impact_evidence_paths
            ],
            "memory_slot_architecture_cost_evidence": [
                str(Path(path))
                for path in memory_slot_architecture_cost_evidence_paths
            ],
            "structural_plasticity_evidence": [
                str(Path(path)) for path in structural_plasticity_evidence_paths
            ],
            "gpu_kernel_evidence": [
                str(Path(path)) for path in gpu_kernel_evidence_paths
            ],
            "generation_coherence_evidence": [
                str(Path(path)) for path in generation_coherence_evidence_paths
            ],
            "quality_replay_evidence": [
                str(Path(path)) for path in quality_replay_evidence_paths
            ],
            "checkpoint_evolution_evidence": [
                str(Path(path)) for path in checkpoint_evolution_evidence_paths
            ],
            "scale_ladder": str(output.parent / "language-suite-scale-ladder.json"),
            "checkpoint": str(checkpoint_path),
            "checkpoint_evolution_dir": str(output.parent / "language-suite-evolution"),
        },
        "promotion_gate": {
            "status": (
                "blocked_missing_required_evidence"
                if missing_categories or failed_categories
                else "ready_for_review"
            ),
            "promotes_runtime_claim": False,
            "benchmark_suite_report_written": True,
            "requires_long_run_evidence": True,
            "requires_gpu_kernel_parity": True,
            "requires_grounding_support": True,
            "grounding_support_available": grounding_gate[
                "grounding_support_available"
            ],
            "generation_coherence_available": not generation_coherence_missing,
            "quality_replay_evidence_available": quality_replay_evidence[
                "quality_replay_available"
            ],
            "brain_installed_continual_learning_evidence_available": bool(
                brain_installed_learning_evidence[
                    "brain_installed_continual_learning_available"
                ]
            ),
            "checkpoint_evolution_evidence_available": (
                saved_checkpoint_evolution_evidence[
                    "checkpoint_evolution_evidence_available"
                ]
            ),
            "long_run_evidence_available": not long_run_missing,
            "controlled_decode_house_scale_evidence_available": bool(
                long_run_evidence.get("controlled_decode_house_scale_gate_reached")
            ),
            "generation_controlled_decode_house_scale_aligned": bool(
                generation_long_run_alignment.get(
                    "same_checkpoint_controlled_decode_house_scale_available"
                )
            ),
            "quality_replay_controlled_decode_house_scale_aligned": bool(
                isinstance(quality_replay_long_run_alignment, Mapping)
                and quality_replay_long_run_alignment.get(
                    "same_child_controlled_decode_house_scale_available"
                )
            ),
            "missing_required_category_names": [
                item["name"] for item in missing_categories
            ],
            "failed_category_names": [item["name"] for item in failed_categories],
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sustained-target-tokens", type=int, default=8)
    parser.add_argument(
        "--sustained-evidence",
        type=Path,
        action="append",
        default=[],
        help="Existing marulho_language_sustained_runtime_evidence JSON report.",
    )
    parser.add_argument(
        "--gpu-kernel-evidence",
        type=Path,
        action="append",
        default=[],
        help="Existing marulho_language_triton_kernel_report JSON report.",
    )
    parser.add_argument(
        "--memory-slot-runtime-impact-evidence",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing marulho_language_memory_slot_runtime_impact JSON report."
        ),
    )
    parser.add_argument(
        "--brain-installed-continual-learning-evidence",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing marulho_language_brain_installed_continual_learning_evidence "
            "JSON report."
        ),
    )
    parser.add_argument(
        "--memory-slot-architecture-cost-evidence",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing marulho_language_continual_learning_experiment JSON report "
            "with memory-slot architecture-cost evidence."
        ),
    )
    parser.add_argument(
        "--structural-plasticity-evidence",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing marulho_language_structural_plasticity_experiment or "
            "structural-plasticity transaction JSON report."
        ),
    )
    parser.add_argument(
        "--generation-coherence-evidence",
        type=Path,
        action="append",
        default=[],
        help="Existing marulho_language_generation_coherence_report JSON report.",
    )
    parser.add_argument(
        "--quality-replay-evidence",
        type=Path,
        action="append",
        default=[],
        help="Existing marulho_language_quality_replay_experiment JSON report.",
    )
    parser.add_argument(
        "--checkpoint-evolution-evidence",
        type=Path,
        action="append",
        default=[],
        help=(
            "Existing marulho_language_checkpoint_evolution_experiment JSON "
            "report."
        ),
    )
    args = parser.parse_args()
    run_language_runtime_benchmark_suite(
        output_path=args.output,
        sustained_target_tokens=args.sustained_target_tokens,
        sustained_evidence_paths=tuple(args.sustained_evidence),
        brain_installed_continual_learning_evidence_paths=tuple(
            args.brain_installed_continual_learning_evidence
        ),
        memory_slot_runtime_impact_evidence_paths=tuple(
            args.memory_slot_runtime_impact_evidence
        ),
        memory_slot_architecture_cost_evidence_paths=tuple(
            args.memory_slot_architecture_cost_evidence
        ),
        structural_plasticity_evidence_paths=tuple(
            args.structural_plasticity_evidence
        ),
        gpu_kernel_evidence_paths=tuple(args.gpu_kernel_evidence),
        generation_coherence_evidence_paths=tuple(args.generation_coherence_evidence),
        quality_replay_evidence_paths=tuple(args.quality_replay_evidence),
        checkpoint_evolution_evidence_paths=tuple(
            args.checkpoint_evolution_evidence
        ),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
