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
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["promotion_gate"]["standalone_structural_evidence_available"] is True
    assert report["promotion_gate"]["all_proposals_non_mutating"] is True
    assert report["promotion_gate"]["all_transactions_checkpoint_backed"] is True
    assert report["promotion_gate"]["all_rollbacks_verified"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["proposal_kinds"] == [
        "memory_slot_expansion",
        "route_bank_expansion",
    ]
    assert report["valid_transaction_count"] == 2
    summaries = {
        item["proposal_kind"]: item for item in report["transaction_summaries"]
    }
    assert summaries["memory_slot_expansion"]["target_memory_slot_count"] == 4
    assert summaries["memory_slot_expansion"][
        "target_memory_slot_candidate_count"
    ] == 2
    assert summaries["route_bank_expansion"]["target_route_candidate_count"] == 4
    assert (tmp_path / "README.md").exists()
