from __future__ import annotations

from marulho.evaluation.language_memory_slot_runtime_impact import (
    MemorySlotRuntimeImpactConfig,
    run_language_memory_slot_runtime_impact,
)


def test_language_memory_slot_runtime_impact_reports_forward_workload(tmp_path) -> None:
    output = tmp_path / "memory-slot-runtime-impact.json"

    report = run_language_memory_slot_runtime_impact(
        output_path=output,
        config=MemorySlotRuntimeImpactConfig(
            vocab_size=384,
            embedding_dim=12,
            state_dim=16,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=24,
            memory_slot_count=4,
            bounded_memory_slot_candidate_count=2,
            active_memory_slot_count=1,
            sequence_length=8,
            batch_size=2,
            warmup_steps=0,
            repeats=1,
            device="cpu",
        ),
    )

    assert output.exists()
    assert report["surface"] == "marulho_language_memory_slot_runtime_impact.v1"
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["model_vocab_size"] == 384
    assert report["generation_vocab_size"] == report["tokenizer_vocab_size"]
    assert report["batch"]["tokens_per_forward"] == 16
    assert report["review"]["complete_forward_runtime_impact"] is True
    assert report["review"]["not_kernel_microbench_only"] is True
    assert report["review"]["mutates_model_state"] is False
    assert report["review"]["gradient_training_unchanged"] is True
    assert report["promotion_gate"]["complete_runtime_impact_available"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["bounded_memory_slots_enabled"] is True
    assert report["promotion_gate"]["bounded_avoids_all_slot_scan"] is True
    assert report["promotion_gate"]["all_slot_scan_contrast_available"] is True
    assert report["promotion_gate"]["neutral_initialization_parity"] is True

    control = report["arms"]["memory_slots_disabled_control"]
    bounded = report["arms"]["bounded_memory_slots_enabled"]
    all_slot = report["arms"]["all_slot_memory_scan_contrast"]
    assert control["success"] is True
    assert bounded["success"] is True
    assert all_slot["success"] is True
    assert control["token_count"] == 16
    assert bounded["token_count"] == 16
    assert all_slot["token_count"] == 16
    assert control["memory_enabled"] is False
    assert bounded["memory_enabled"] is True
    assert bounded["total_slots"] == 4
    assert bounded["candidate_slot_count"] == 2
    assert bounded["active_slots_per_token"] == 1
    assert bounded["candidate_slots_scored"] == 32
    assert bounded["runs_all_slots"] is False
    assert bounded["memory_gate_readback"] is False
    assert all_slot["memory_enabled"] is True
    assert all_slot["candidate_slot_count"] == 4
    assert all_slot["candidate_slots_scored"] == 64
    assert all_slot["runs_all_slots"] is True

    comparison = report["comparison"]
    assert comparison["control_success"] is True
    assert comparison["bounded_success"] is True
    assert comparison["all_slot_success"] is True
    assert comparison["bounded_avoids_all_slot_scan"] is True
    assert comparison["all_slot_scan_contrast_available"] is True
    assert comparison["bounded_neutral_initialization_parity"]["passed"] is True
    assert comparison["all_slot_neutral_initialization_parity"]["passed"] is True
    assert comparison["memory_gate_readback"] is False
    assert comparison["evidence_status"] == "measured_bounded_memory_slot_forward_impact"
