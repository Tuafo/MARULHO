from __future__ import annotations

import json

from marulho.evaluation.language_structural_plasticity_experiment import (
    ARTIFACT_KIND,
    SURFACE,
    LanguageStructuralPlasticityExperimentConfig,
    run_language_structural_plasticity_experiment,
)


def test_language_structural_plasticity_experiment_writes_saved_evidence(
    tmp_path,
) -> None:
    output = tmp_path / "structural-plasticity.json"

    report = run_language_structural_plasticity_experiment(
        output_path=output,
        proposal_kinds=("memory_slot_expansion", "route_bank_expansion"),
        config=LanguageStructuralPlasticityExperimentConfig(
            embedding_dim=12,
            state_dim=20,
            expert_count=5,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=32,
            memory_slot_growth=4,
            memory_slot_candidate_count=2,
            active_memory_slot_count=1,
            sequence_length=10,
            stride=10,
            batch_size=2,
            max_eval_batches=1,
            quality_prompt_case_count=2,
            quality_prompt_max_new_tokens=4,
            quality_prompt_min_new_tokens=1,
            quality_prompt_min_prefix_match_chars=0,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["promotion_gate"]["standalone_structural_evidence_available"] is False
    assert report["promotion_gate"]["all_proposals_non_mutating"] is True
    assert report["promotion_gate"]["all_transactions_checkpoint_backed"] is True
    assert report["promotion_gate"]["all_rollbacks_verified"] is True
    assert report["promotion_gate"]["all_quality_impacts_recorded"] is True
    assert (
        report["promotion_gate"]["all_prompt_coherence_regression_absent"]
        is True
    )
    assert (
        report["promotion_gate"][
            "all_prompt_loss_regression_without_pass_regression_absent"
        ]
        is False
    )
    assert (
        report["promotion_gate"]["requires_sustained_speed_delta_for_runtime_promotion"]
        is True
    )
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["proposal_kinds"] == [
        "memory_slot_expansion",
        "route_bank_expansion",
    ]
    assert report["status"] == "structural_plasticity_transaction_evidence_incomplete"
    assert report["valid_transaction_count"] == 2
    summaries = {
        item["proposal_kind"]: item for item in report["transaction_summaries"]
    }
    assert summaries["memory_slot_expansion"]["target_memory_slot_count"] == 4
    assert summaries["memory_slot_expansion"][
        "target_memory_slot_candidate_count"
    ] == 2
    assert summaries["memory_slot_expansion"]["quality_impact_recorded"] is True
    assert (
        summaries["memory_slot_expansion"]["prompt_coherence_regressed_prompt_count"]
        == 0
    )
    assert summaries["route_bank_expansion"]["target_route_candidate_count"] == 4
    assert summaries["route_bank_expansion"]["quality_impact_recorded"] is True
    assert (
        summaries["route_bank_expansion"][
            "prompt_pass_nonregressed_but_loss_regressed"
        ]
        is True
    )
    first_transaction = report["transactions"][0]
    quality = first_transaction["structural_quality_impact"]
    assert quality["surface"] == "marulho_language_structural_quality_impact.v1"
    assert quality["quality_impact_recorded"] is True
    assert quality["heldout"]["heldout_loss_delta"] <= 100.0
    assert quality["prompt_quality"]["source_continuation_loss_available"] is True
    assert quality["active_compute"]["active_parameters_per_token_delta"] >= 0
    assert quality["sustained_speed_delta"]["available"] is False
    assert quality["promotion_gate"]["prompt_continuation_loss_recorded"] is True
    assert first_transaction["generation_coherence_before"]["summary"][
        "source_continuation_loss_available"
    ] is True
    assert first_transaction["generation_coherence_after"]["summary"][
        "source_continuation_loss_available"
    ] is True
    assert (tmp_path / "README.md").exists()
