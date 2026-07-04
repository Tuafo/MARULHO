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
    build_language_structural_deep_sleep_proposal,
    build_language_structural_merge_proposal,
    build_language_structural_prune_proposal,
    build_language_structural_plasticity_proposal,
)


SURFACE = "marulho_language_runtime_benchmark_suite.v1"
ARTIFACT_KIND = "marulho_language_runtime_benchmark_suite"
SUSTAINED_SURFACE = "marulho_language_sustained_runtime_evidence.v1"
SUSTAINED_ARTIFACT_KIND = "marulho_language_sustained_runtime_evidence"
KERNEL_SURFACE = "marulho_language_triton_kernel_report.v1"
KERNEL_ARTIFACT_KIND = "marulho_language_triton_kernel_report"
RMSNORM_KERNEL_NAME = "language_rmsnorm_forward"
PLIF_FORWARD_KERNEL_NAME = "language_plif_forward"
PLIF_SURROGATE_KERNEL_NAME = "language_plif_surrogate_backward"
SELECTIVE_SCAN_KERNEL_NAME = "language_selective_state_scan"
EXPERT_DISPATCH_KERNEL_NAME = "language_block_sparse_expert_dispatch"
SAMPLED_VOCAB_CE_KERNEL_NAME = "language_sampled_vocab_cross_entropy"
SUPPORTED_GPU_KERNEL_NAMES = {
    RMSNORM_KERNEL_NAME,
    PLIF_FORWARD_KERNEL_NAME,
    PLIF_SURROGATE_KERNEL_NAME,
    SELECTIVE_SCAN_KERNEL_NAME,
    EXPERT_DISPATCH_KERNEL_NAME,
    SAMPLED_VOCAB_CE_KERNEL_NAME,
}


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
    return {
        "path": str(report.get("path") or report.get("output_path") or ""),
        "report_status": report.get("report_status"),
        "success": bool(report.get("success")),
        "target_tokens": int(report.get("target_tokens", 0) or 0),
        "token_delta": _token_delta(report),
        "tokens_per_second": report.get("tokens_per_second"),
        "active_language_path": report.get("active_language_path"),
        "runtime_owner": report.get("runtime_owner"),
        "device": device_backend.get("device"),
        "backend": device_backend.get("backend"),
        "triton_kernel_used": bool(device_backend.get("triton_kernel_used")),
        "promoted_hot_path": bool(device_backend.get("promoted_hot_path")),
        "diagnostic_boundary_reached": bool(
            promotion_gate.get("diagnostic_boundary_reached")
        ),
        "long_run_gate_reached": bool(promotion_gate.get("long_run_gate_reached")),
        "house_scale_gate_reached": bool(promotion_gate.get("house_scale_gate_reached")),
        "promotes_runtime_claim": bool(promotion_gate.get("promotes_runtime_claim")),
        "promotes_hot_path": bool(promotion_gate.get("promotes_hot_path")),
    }


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
    house_scale_reports = [
        report for report in valid_reports if _token_delta(report) >= 524288
    ]
    best_diagnostic = max(
        diagnostic_only_reports or diagnostic_reports,
        key=_token_delta,
        default=None,
    )
    best_long = max(long_gate_reports, key=_token_delta, default=None)
    best_house = max(house_scale_reports, key=_token_delta, default=None)
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
        "diagnostic_boundary_reached": best_diagnostic is not None,
        "long_run_gate_reached": best_long is not None,
        "house_scale_gate_reached": best_house is not None,
        "missing_evidence": missing,
        "promotes_runtime_claim": False,
        "promotes_hot_path": False,
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
    missing = []
    if rmsnorm_report is None:
        missing.append("rmsnorm_triton_parity")
    if plif_forward_report is None:
        missing.append("plif_triton_forward_parity")
    if plif_surrogate_report is None:
        missing.append("plif_triton_backward_surrogate_parity")
    if selective_scan_report is None:
        missing.append("selective_scan_triton_parity")
    if expert_dispatch_report is None:
        missing.append("block_sparse_expert_dispatch_parity")
    if sampled_vocab_report is None:
        missing.append("sampled_vocab_cross_entropy_parity")
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
    gpu_kernel_evidence_paths: Sequence[str | Path] = (),
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
    categories.append(
        _category(
            "generation_coherence",
            status="smoke_only",
            evidence={
                "generated_token_count": int(generation["new_token_count"]),
                "active_language_path": generation["active_language_path"],
                "external_llm_used": generation["external_llm_used"],
                "review_kind": "token_stream_smoke_not_human_quality_review",
            },
            missing=("human_generation_quality_review", "grounded_prompt_suite"),
        )
    )

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
                "pass"
                if learning_report["rollback_evidence"]["restore_verified"]
                and learning_evidence["final_parameter_delta_l2"] > 0.0
                else "fail"
            ),
            evidence={
                "new_domain_loss_delta": learning_evidence["new_domain_loss_delta"],
                "final_parameter_delta_l2": learning_evidence["final_parameter_delta_l2"],
                "tokens_per_second": learning_evidence["tokens_per_second"],
                "rollback_available": learning_report["promotion_gate"]["rollback_available"],
            },
        )
    )
    categories.append(
        _category(
            "forgetting",
            status="pass" if "old_domain_forgetting" in learning_evidence else "fail",
            evidence={
                "old_domain_forgetting": learning_evidence["old_domain_forgetting"],
                "old_domain_forgetting_within_tolerance": learning_report[
                    "promotion_gate"
                ]["old_domain_forgetting_within_tolerance"],
            },
        )
    )
    categories.append(
        _category(
            "replay_recovery",
            status=(
                "pass"
                if "general_replay_retention_delta" in learning_evidence
                else "fail"
            ),
            evidence={
                "general_replay_retention_delta": learning_evidence[
                    "general_replay_retention_delta"
                ],
                "general_replay_retention_within_tolerance": learning_report[
                    "promotion_gate"
                ]["general_replay_retention_within_tolerance"],
            },
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
    sleep_eval = evaluate_language_model(structural_sleep_candidate, old_split.eval)
    sleep_routing = sleep_eval["spike_telemetry"]["routing"]
    categories.append(
        _category(
            "growth_prune_safety",
            status=(
                "pass"
                if proposal["mutates_runtime_state"] is False
                and structural_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_report["rollback_evidence"]["rollback_verified"]
                and prune_proposal["mutates_runtime_state"] is False
                and structural_prune_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_prune_report["rollback_evidence"]["rollback_verified"]
                and merge_proposal["mutates_runtime_state"] is False
                and structural_merge_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_merge_report["rollback_evidence"]["rollback_verified"]
                and sleep_proposal["mutates_runtime_state"] is False
                and structural_sleep_report["checkpoint"]["checkpoint_restore_verified"]
                and structural_sleep_report["rollback_evidence"]["rollback_verified"]
                else "fail"
            ),
            evidence={
                "proposal_mutates_runtime_state": proposal["mutates_runtime_state"],
                "growth_transaction_applied": structural_report["applied"],
                "growth_checkpoint_backed": structural_report["promotion_gate"]["checkpoint_backed"],
                "growth_rollback_verified": structural_report["rollback_evidence"]["rollback_verified"],
                "growth_target_expert_count": structural_candidate.config.expert_count,
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
            },
            missing=(),
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
                "pass"
                if restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
                and restored_metadata.get("suite") == "language_runtime_benchmark"
                and _finite_positive(restored_eval["heldout_loss"])
                else "fail"
            ),
            evidence={
                "checkpoint_path": str(checkpoint_path),
                "tokenizer_hash_restored": restored_tokenizer.vocabulary_hash(),
                "metadata_restored": dict(restored_metadata),
                "heldout_loss_after_restore": restored_eval["heldout_loss"],
            },
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
    categories.append(
        _category(
            "rollback",
            status=(
                "pass"
                if evolution_report["promotion_gate"]["rollback_to_parent_verified"]
                and evolution_report["promotion_gate"]["parent_runtime_unchanged"]
                else "fail"
            ),
            evidence={
                "rollback_to_parent_verified": evolution_report["promotion_gate"][
                    "rollback_to_parent_verified"
                ],
                "parent_runtime_unchanged": evolution_report["promotion_gate"][
                    "parent_runtime_unchanged"
                ],
                "lineage_id": evolution_report["lineage"]["lineage_id"],
            },
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
            "gpu_kernel_evidence": [
                str(Path(path)) for path in gpu_kernel_evidence_paths
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
            "long_run_evidence_available": not long_run_missing,
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
    args = parser.parse_args()
    run_language_runtime_benchmark_suite(
        output_path=args.output,
        sustained_target_tokens=args.sustained_target_tokens,
        sustained_evidence_paths=tuple(args.sustained_evidence),
        gpu_kernel_evidence_paths=tuple(args.gpu_kernel_evidence),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
