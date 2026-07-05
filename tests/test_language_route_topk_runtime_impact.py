from __future__ import annotations

from marulho.evaluation.language_route_topk_runtime_impact import (
    RouteTopKRuntimeImpactConfig,
    run_language_route_topk_runtime_impact,
)


def test_language_route_topk_runtime_impact_reports_forward_workload(tmp_path) -> None:
    output = tmp_path / "route-topk-runtime-impact.json"

    report = run_language_route_topk_runtime_impact(
        output_path=output,
        config=RouteTopKRuntimeImpactConfig(
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
    assert report["surface"] == "marulho_language_route_topk_runtime_impact.v1"
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

    fallback = report["arms"]["torch_route_topk_policy_fallback"]
    triton = report["arms"]["triton_route_topk_enabled"]
    assert fallback["success"] is True
    assert triton["success"] is True
    assert fallback["token_count"] == 16
    assert triton["token_count"] == 16
    assert fallback["route_selection_backend"] == "torch_route_topk"
    assert triton["route_selection_backend"] == "torch_route_topk"
    assert fallback["route_candidate_count"] == 2
    assert triton["active_expert_count_per_token"] == 2

    comparison = report["comparison"]
    assert comparison["fallback_success"] is True
    assert comparison["triton_success"] is True
    assert comparison["parity_passed"] is True
    assert comparison["triton_route_topk_kernel_used"] is False
    assert comparison["fallback_route_topk_torch_used"] is False
    assert comparison["evidence_status"] == "measured_without_route_topk_triton_use"
