from __future__ import annotations

from marulho.evaluation.language_eligibility_trace_runtime_impact import (
    EligibilityTraceRuntimeImpactConfig,
    run_language_eligibility_trace_runtime_impact,
)


def test_language_eligibility_trace_runtime_impact_reports_forward_workload(tmp_path) -> None:
    output = tmp_path / "eligibility-trace-runtime-impact.json"

    report = run_language_eligibility_trace_runtime_impact(
        output_path=output,
        config=EligibilityTraceRuntimeImpactConfig(
            vocab_size=384,
            embedding_dim=12,
            state_dim=16,
            expert_count=4,
            active_expert_count=2,
            route_candidate_count=2,
            expert_hidden_dim=24,
            sequence_length=8,
            batch_size=2,
            warmup_steps=0,
            repeats=1,
            device="cpu",
        ),
    )

    assert output.exists()
    assert report["surface"] == "marulho_language_eligibility_trace_runtime_impact.v1"
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
    assert report["review"]["one_token_streaming_policy_unchanged"] is True
    assert report["review"]["local_eligibility_trace_update_kernel_claimed"] is True
    assert report["review"]["full_state_selective_scan_fusion_claimed"] is False
    assert report["promotion_gate"]["complete_runtime_impact_available"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False

    baseline = report["arms"]["inline_plif_eligibility_baseline"]
    deferred = report["arms"]["deferred_sequence_scan_eligibility"]
    assert baseline["success"] is True
    assert deferred["success"] is True
    assert baseline["token_count"] == 16
    assert deferred["token_count"] == 16
    assert baseline["eligibility_trace_update_mode"] == "inline_plif_update"
    assert deferred["eligibility_trace_update_mode"] == (
        "deferred_sequence_scan_no_grad"
    )
    assert deferred["eligibility_trace_sequence_buffer_mode"] == "spike_sequence_buffer"
    assert baseline["route_selection_backend"] == "torch_route_topk"
    assert deferred["route_selection_backend"] == "torch_route_topk"
    assert baseline["expert_dispatch_backend"] == "torch_selected_expert_dispatch"
    assert deferred["expert_dispatch_backend"] == "torch_selected_expert_dispatch"

    comparison = report["comparison"]
    assert comparison["baseline_success"] is True
    assert comparison["deferred_success"] is True
    assert comparison["parity_passed"] is True
    assert comparison["baseline_eligibility_trace_update_mode"] == "inline_plif_update"
    assert comparison["deferred_eligibility_trace_update_mode"] == (
        "deferred_sequence_scan_no_grad"
    )
    assert comparison["evidence_status"] == (
        "measured_deferred_vs_inline_eligibility_trace_forward"
    )
