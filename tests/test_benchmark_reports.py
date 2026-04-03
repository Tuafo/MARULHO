from __future__ import annotations

import json
import tempfile
import unittest
from csv import DictWriter
from pathlib import Path

from hecsn.service.benchmark_reports import load_benchmark_reports


def _write_summary(root: Path, report_name: str, payload: dict[str, object]) -> None:
    report_dir = root / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_metrics(root: Path, report_name: str, rows: list[dict[str, object]]) -> None:
    report_dir = root / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = DictWriter(
            handle,
            fieldnames=["token", "drift", "surprise", "recon_error", "pred_error", "drift_floor", "sleep_events_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _acquisition_summary(*, include_scout: bool = False) -> dict[str, object]:
    policy_results: dict[str, object] = {
        "active": {
            "final_mean_candidate_gap": 0.31,
            "final_mean_candidate_diagnostic_gap": 0.41,
            "final_mean_candidate_info_gain": 0.41,
            "final_candidate_gap_by_source": {"alpha": 0.31},
            "final_candidate_diagnostic_gap_by_source": {"alpha": 0.41},
            "final_candidate_info_gain_by_source": {"alpha": 0.41},
            "acquired_sources": ["alpha"],
            "acquisition_history": [],
        },
        "round_robin": {
            "final_mean_candidate_gap": 0.27,
            "final_mean_candidate_diagnostic_gap": 0.37,
            "final_mean_candidate_info_gain": 0.37,
            "final_candidate_gap_by_source": {"alpha": 0.27},
            "final_candidate_diagnostic_gap_by_source": {"alpha": 0.37},
            "final_candidate_info_gain_by_source": {"alpha": 0.37},
            "acquired_sources": ["alpha"],
        },
    }
    if include_scout:
        policy_results["scout_commit"] = {
            "final_mean_candidate_gap": 0.33,
            "final_mean_candidate_diagnostic_gap": 0.43,
            "final_mean_candidate_info_gain": 0.43,
            "final_candidate_gap_by_source": {"alpha": 0.33},
            "final_candidate_diagnostic_gap_by_source": {"alpha": 0.43},
            "final_candidate_info_gain_by_source": {"alpha": 0.43},
            "acquired_sources": ["alpha"],
            "acquisition_history": [],
        }
    return {
        "policy_results": policy_results,
        "runtime_scope": {
            "input_representation": "order_weighted_ascii",
            "supports_contextual_routing": True,
            "supports_binding_conjunction_memory": True,
        },
    }


class BenchmarkReportsTests(unittest.TestCase):
    def test_load_benchmark_reports_prefers_maintained_hf_allocation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_summary(root, "refactor_autonomy_acquisition_open_web_scout_projected_20260402", _acquisition_summary(include_scout=True))
            _write_summary(root, "refactor_autonomy_acquisition_hf_allocation_rng_20260401", _acquisition_summary())

            reports = load_benchmark_reports(reports_root=root)

            acquisition = next(item for item in reports["benchmarks"] if item["benchmark_id"] == "source_acquisition")
            self.assertEqual(acquisition["artifact_id"], "refactor_autonomy_acquisition_hf_allocation_rng_20260401")
            self.assertEqual(acquisition["artifact_label"], "HF projected active allocation")
            self.assertEqual(acquisition["history_policy"], "active")
            self.assertNotIn("scout_mean_candidate_gap", acquisition["summary"])
            self.assertNotIn("scout_lookahead_history", acquisition)

    def test_load_benchmark_reports_uses_latest_maintained_hf_report_when_multiple_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_summary(root, "refactor_autonomy_acquisition_hf_allocation_rng_20260401", _acquisition_summary())
            _write_summary(root, "refactor_autonomy_acquisition_hf_allocation_rng_20260402", _acquisition_summary())

            reports = load_benchmark_reports(reports_root=root)

            acquisition = next(item for item in reports["benchmarks"] if item["benchmark_id"] == "source_acquisition")
            self.assertEqual(acquisition["artifact_id"], "refactor_autonomy_acquisition_hf_allocation_rng_20260402")

    def test_load_benchmark_reports_falls_back_to_open_web_report_when_hf_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_summary(root, "refactor_autonomy_acquisition_open_web_scout_projected_20260402", _acquisition_summary(include_scout=True))

            reports = load_benchmark_reports(reports_root=root)
            acquisition = next(item for item in reports["benchmarks"] if item["benchmark_id"] == "source_acquisition")

            self.assertEqual(acquisition["artifact_id"], "refactor_autonomy_acquisition_open_web_scout_projected_20260402")
            self.assertEqual(acquisition["artifact_label"], "Open-web scout projected frontier")
            self.assertEqual(acquisition["history_policy"], "scout_commit")

    def test_load_benchmark_reports_supports_new_benchmark_artifact_names_and_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_summary(
                root,
                "refactor_mechanism_validation_smoke",
                {
                    "drift_mean": 0.1,
                    "surprise_mean": 0.2,
                    "sparsity_mean": 0.3,
                    "recon_error_mean": 0.4,
                    "winner_entropy_bits": 1.2,
                    "winner_max_share": 0.25,
                    "mechanism_validation_gate": {"pass": True},
                },
            )
            _write_metrics(
                root,
                "refactor_mechanism_validation_smoke",
                [
                    {
                        "token": 1,
                        "drift": 0.1,
                        "surprise": 0.2,
                        "recon_error": 0.3,
                        "pred_error": 0.4,
                        "drift_floor": 0.05,
                        "sleep_events_total": 0,
                    }
                ],
            )
            _write_summary(
                root,
                "refactor_memory_consolidation_smoke",
                {
                    "metrics": {
                        "task_a_recovery_delta": 0.12,
                        "task_a_overlap_after_b": 0.6,
                        "task_b_overlap_after_b": 0.5,
                        "task_a_overlap_after_consolidation": 0.7,
                        "task_b_overlap_after_consolidation": 0.65,
                        "task_a_relative_degradation_after_consolidation": 0.01,
                        "task_a_recon_after_a": 0.3,
                        "task_b_recon_before_b": 0.4,
                        "task_a_recon_after_b": 0.45,
                        "task_b_recon_after_b": 0.25,
                        "task_a_recon_after_consolidation": 0.28,
                        "task_b_recon_after_consolidation": 0.22,
                    },
                    "consolidation": {
                        "mean_capture_tag_before": 0.2,
                        "mean_capture_tag_after": 0.1,
                        "mean_prp_level_before": 0.3,
                        "mean_prp_level_after": 0.4,
                        "mean_capture_strength_before": 0.5,
                        "mean_capture_strength_after": 0.6,
                        "mean_consolidation_level_before": 0.7,
                        "mean_consolidation_level_after": 0.8,
                    },
                    "memory_consolidation_gate": {"pass": True},
                },
            )
            _write_summary(
                root,
                "refactor_contextual_routing_smoke",
                {
                    "training_diagnostics": {
                        "mean_dopamine": 0.1,
                        "mean_acetylcholine": 0.2,
                        "mean_norepinephrine": 0.3,
                        "mean_context_gain": 0.4,
                    },
                    "contextual_routing_metrics": {
                        "task_a_recon_error": 0.11,
                        "task_b_recon_error": 0.12,
                        "context_state_separation": 0.8,
                        "probe_winner_switch_rate": 0.3,
                        "probe_mean_assembly_distance": 0.2,
                        "bank_polysemy_accuracy": 0.9,
                        "bank_polysemy_signature_margin": 0.05,
                        "bank_polysemy_winner_sequence_difference_rate": 0.4,
                    },
                    "contextual_routing_gate": {"pass": True},
                },
            )
            _write_summary(
                root,
                "refactor_hierarchical_scale_smoke",
                {
                    "routing_metrics": {
                        "recall_at_k": 0.98,
                        "top1_recall": 0.95,
                        "mean_latency_ms": 2.0,
                        "p95_latency_ms": 3.0,
                    },
                    "training_diagnostics": {
                        "eval_recon_error": 0.2,
                        "throughput_chars_per_sec": 1200.0,
                    },
                    "sharding": {
                        "index_shard_sizes": [16, 16],
                        "primary_query_shard_counts": [10, 9],
                        "winner_shard_counts": [8, 7],
                        "index_shard_balance_ratio": 1.0,
                        "winner_shard_coverage": 1.0,
                    },
                    "hierarchical_scale_gate": {"pass": True},
                },
            )

            reports = load_benchmark_reports(reports_root=root)
            benchmarks = {item["benchmark_id"]: item for item in reports["benchmarks"]}

            self.assertIn("mechanism_validation", benchmarks)
            self.assertIn("memory_consolidation", benchmarks)
            self.assertIn("contextual_routing", benchmarks)
            self.assertIn("hierarchical_scale", benchmarks)
            self.assertTrue(benchmarks["memory_consolidation"]["summary"]["gate_pass"])
            self.assertTrue(benchmarks["contextual_routing"]["summary"]["gate_pass"])
            self.assertTrue(benchmarks["hierarchical_scale"]["summary"]["gate_pass"])
            self.assertTrue(
                str(benchmarks["mechanism_validation"]["summary_path"]).endswith(
                    "refactor_mechanism_validation_smoke/summary.json"
                )
            )


if __name__ == "__main__":
    unittest.main()
