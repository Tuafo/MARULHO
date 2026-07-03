from __future__ import annotations

import json

from marulho.evaluation.language_runtime_benchmark_suite import (
    KERNEL_ARTIFACT_KIND,
    KERNEL_SURFACE,
    RMSNORM_KERNEL_NAME,
    SURFACE,
    SUSTAINED_ARTIFACT_KIND,
    SUSTAINED_SURFACE,
    run_language_runtime_benchmark_suite,
)


def _write_sustained_report(path, *, token_delta: int) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": SUSTAINED_ARTIFACT_KIND,
                "surface": SUSTAINED_SURFACE,
                "report_status": "final",
                "success": True,
                "target_tokens": token_delta,
                "token_delta": token_delta,
                "tokens_per_second": 1234.5,
                "runtime_owner": "MarulhoLanguageModel",
                "active_language_path": "marulho_lm_head",
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "device_backend": {
                    "device": "cpu",
                    "backend": "torch_eager_cpu",
                    "triton_kernel_used": False,
                    "promoted_hot_path": False,
                },
                "promotion_gate": {
                    "diagnostic_boundary_reached": token_delta >= 8192,
                    "long_run_gate_reached": token_delta >= 131072,
                    "house_scale_gate_reached": token_delta >= 524288,
                    "promotes_runtime_claim": False,
                    "promotes_hot_path": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_gpu_kernel_report(path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_kind": KERNEL_ARTIFACT_KIND,
                "surface": KERNEL_SURFACE,
                "kernel_name": RMSNORM_KERNEL_NAME,
                "owned_by_marulho": True,
                "external_llm_used": False,
                "loads_external_checkpoint": False,
                "parity_passed": True,
                "valid_shape_result_count": 2,
                "dtype_coverage": ["float16", "float32"],
                "benchmark_summary": {
                    "geometric_speedup_vs_torch": 1.25,
                },
                "promotion_gate": {
                    "kernel_parity_available": True,
                    "complete_runtime_impact_available": False,
                    "promotes_hot_path": False,
                },
            }
        ),
        encoding="utf-8",
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
    assert categories["grounding_support"]["status"] == "pass"
    assert categories["grounding_support"]["evidence"]["source_term_coverage"] == 1.0
    assert categories["grounding_support"]["evidence"]["missing_required_terms"] == []
    assert categories["grounding_support"]["evidence"][
        "source_term_coverage_gate_passed"
    ] is True
    assert categories["gpu_kernel_correctness"]["status"] == "missing"
    assert categories["gpu_kernel_correctness"]["evidence"]["lm_triton_kernel_used"] is False
    assert categories["gpu_kernel_correctness"]["evidence"]["rmsnorm_triton_parity"] is False
    assert "rmsnorm_triton_parity" in categories["gpu_kernel_correctness"]["missing_evidence"]
    assert categories["generation_coherence"]["status"] == "smoke_only"
    assert categories["growth_prune_safety"]["evidence"]["growth_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["prune_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["merge_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["deep_sleep_transaction_applied"] is True
    assert categories["growth_prune_safety"]["evidence"]["deep_sleep_runs_all_columns"] is False
    assert "prune_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "merge_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert "deep_sleep_transaction" not in categories["growth_prune_safety"]["missing_evidence"]
    assert categories["growth_prune_safety"]["missing_evidence"] == []
    assert categories["long_run_throughput"]["status"] == "smoke_only"
    assert categories["long_run_throughput"]["evidence"]["smoke_token_delta"] == 4
    assert categories["long_run_throughput"]["evidence"][
        "diagnostic_boundary_reached"
    ] is False
    assert categories["long_run_throughput"]["evidence"]["long_run_gate_reached"] is False
    assert categories["service_contract"]["evidence"]["status_read_mutates_token_count"] is False
    assert categories["checkpoint_restore"]["status"] == "pass"
    assert report["promotion_gate"]["status"] == "blocked_missing_required_evidence"
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["requires_gpu_kernel_parity"] is True
    assert report["promotion_gate"]["requires_grounding_support"] is True
    assert report["promotion_gate"]["grounding_support_available"] is True
    assert report["promotion_gate"]["long_run_evidence_available"] is False
    assert report["promotion_gate"]["missing_required_category_names"] == [
        "generation_coherence",
        "long_run_throughput",
        "gpu_kernel_correctness",
    ]
    assert report["promotion_gate"]["requires_long_run_evidence"] is True
    assert (tmp_path / "README.md").exists()
    assert (tmp_path / "language-suite-grounding-support.json").exists()
    assert (tmp_path / "language-suite-sustained-smoke.json").exists()
    assert (tmp_path / "language-suite-scale-ladder.json").exists()
    assert (tmp_path / "language-suite-checkpoint.pt").exists()


def test_language_runtime_benchmark_suite_accepts_saved_lm_long_run_reports(
    tmp_path,
) -> None:
    output = tmp_path / "language-suite.json"
    diagnostic = tmp_path / "diagnostic-8192.json"
    long_gate = tmp_path / "long-gate-131072.json"
    gpu_kernel = tmp_path / "rmsnorm-triton.json"
    _write_sustained_report(diagnostic, token_delta=8192)
    _write_sustained_report(long_gate, token_delta=131072)
    _write_gpu_kernel_report(gpu_kernel)

    report = run_language_runtime_benchmark_suite(
        output_path=output,
        sustained_target_tokens=2,
        sustained_evidence_paths=(diagnostic, long_gate),
        gpu_kernel_evidence_paths=(gpu_kernel,),
    )
    categories = {item["name"]: item for item in report["categories"]}
    long_run = categories["long_run_throughput"]
    gpu_kernel_category = categories["gpu_kernel_correctness"]

    assert long_run["status"] == "pass"
    assert long_run["missing_evidence"] == []
    assert long_run["evidence"]["valid_report_count"] == 2
    assert long_run["evidence"]["diagnostic_boundary_reached"] is True
    assert long_run["evidence"]["long_run_gate_reached"] is True
    assert long_run["evidence"]["diagnostic_report"]["token_delta"] == 8192
    assert long_run["evidence"]["long_gate_report"]["token_delta"] == 131072
    assert long_run["evidence"]["promotes_runtime_claim"] is False
    assert long_run["evidence"]["promotes_hot_path"] is False
    assert gpu_kernel_category["status"] == "missing"
    assert gpu_kernel_category["evidence"]["lm_triton_kernel_used"] is True
    assert gpu_kernel_category["evidence"]["rmsnorm_triton_parity"] is True
    assert gpu_kernel_category["evidence"]["covered_kernel_names"] == [
        RMSNORM_KERNEL_NAME
    ]
    assert "rmsnorm_triton_parity" not in gpu_kernel_category["missing_evidence"]
    assert "plif_triton_parity" in gpu_kernel_category["missing_evidence"]
    assert report["promotion_gate"]["long_run_evidence_available"] is True
    assert report["promotion_gate"]["missing_required_category_names"] == [
        "generation_coherence",
        "gpu_kernel_correctness",
    ]
