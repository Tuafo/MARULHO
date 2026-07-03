"""MARULHO LM-head benchmark-suite evidence aggregator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

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


def run_language_runtime_benchmark_suite(
    *,
    output_path: str | Path,
    sustained_target_tokens: int = 8,
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
    categories.append(
        _category(
            "long_run_throughput",
            status="smoke_only",
            evidence={
                "report_status": sustained_report["report_status"],
                "target_tokens": sustained_report["target_tokens"],
                "token_delta": sustained_report["token_delta"],
                "tokens_per_second": sustained_report["tokens_per_second"],
                "short_run_is_smoke_only": sustained_report["promotion_gate"][
                    "short_run_is_smoke_only"
                ],
            },
            missing=("8192_token_diagnostic_run", "131072_token_long_run_gate"),
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

    categories.append(
        _category(
            "gpu_kernel_correctness",
            status="missing",
            evidence={
                "lm_triton_kernel_used": False,
                "pytorch_fallback_available": True,
            },
            missing=(
                "plif_triton_parity",
                "selective_scan_triton_parity",
                "block_sparse_expert_dispatch_parity",
                "sampled_vocab_cross_entropy_parity",
            ),
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
    args = parser.parse_args()
    run_language_runtime_benchmark_suite(
        output_path=args.output,
        sustained_target_tokens=args.sustained_target_tokens,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
