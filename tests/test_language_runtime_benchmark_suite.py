from __future__ import annotations

import json

from marulho.evaluation.language_runtime_benchmark_suite import (
    SURFACE,
    run_language_runtime_benchmark_suite,
)


def test_language_runtime_benchmark_suite_writes_blocked_promotion_report(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=4,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    categories = {item["name"]: item for item in report["categories"]}

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert set(categories) == {
        "next_token_loss",
        "heldout_perplexity",
        "generation_coherence",
        "grounding_support",
        "continual_learning",
        "forgetting",
        "replay_recovery",
        "growth_prune_safety",
        "long_run_throughput",
        "active_compute",
        "gpu_kernel_correctness",
        "checkpoint_restore",
        "rollback",
        "service_contract",
        "scale_ladder",
    }
    assert categories["grounding_support"]["status"] == "missing"
    assert categories["gpu_kernel_correctness"]["status"] == "missing"
    assert categories["generation_coherence"]["status"] == "smoke_only"
    assert categories["growth_prune_safety"]["evidence"]["growth_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["prune_transaction_applied"] is True
    assert "prune_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert categories["long_run_throughput"]["status"] == "smoke_only"
    assert categories["long_run_throughput"]["evidence"]["token_delta"] == 4
    assert categories["service_contract"]["evidence"]["status_read_mutates_token_count"] is False
    assert categories["checkpoint_restore"]["status"] == "pass"
    assert report["promotion_gate"]["status"] == "blocked_missing_required_evidence"
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["requires_gpu_kernel_parity"] is True
    assert report["promotion_gate"]["requires_grounding_support"] is True
    assert report["promotion_gate"]["requires_long_run_evidence"] is True
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "language-suite-sustained-smoke.json").exists()
    assert (tmp_path / "language-suite-scale-ladder.json").exists()
    assert (tmp_path / "language-suite-checkpoint.pt").exists()
