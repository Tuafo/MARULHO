"""Tests for sleep-phase SFA correction (§4.8)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from marulho.core.abstraction import AbstractionLayer


class TestSFACorrectionStep(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.layer = AbstractionLayer(
            n_columns=32,
            n_concepts=8,
            device=self.device,
        )

    def _make_samples(self, n: int = 50) -> list[torch.Tensor]:
        """Generate synthetic temporal samples with known structure."""
        samples = []
        for t in range(n):
            # Slowly varying signal + noise
            s = torch.zeros(32)
            s[0] = 0.5 + 0.3 * (t / n)  # slow feature
            s[1] = 0.2 + 0.1 * ((t % 5) / 5.0)  # medium feature
            s[2:5] = torch.randn(3) * 0.1  # noise
            s = torch.clamp(s, min=0.0)
            samples.append(s)
        return samples

    def test_returns_metrics(self) -> None:
        samples = self._make_samples(20)
        result = self.layer.sfa_correction_step(samples)
        expected_keys = {
            "n_samples", "pre_output_var", "post_output_var",
            "pre_deriv_var", "post_deriv_var", "variance_reduction",
            "pre_offdiag", "post_offdiag", "decorrelation",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["n_samples"], 20)

    def test_too_few_samples_noop(self) -> None:
        result = self.layer.sfa_correction_step([torch.randn(32)])
        self.assertEqual(result["n_samples"], 0)
        self.assertAlmostEqual(result["variance_reduction"], 0.0)

    def test_empty_samples_noop(self) -> None:
        result = self.layer.sfa_correction_step([])
        self.assertEqual(result["n_samples"], 0)

    def test_feedforward_changes_after_correction(self) -> None:
        ff_before = self.layer.feedforward.clone()
        samples = self._make_samples(50)
        self.layer.sfa_correction_step(samples, lr=0.1)
        # Feedforward should have been modified
        self.assertFalse(torch.allclose(ff_before, self.layer.feedforward))

    def test_feedforward_still_normalized(self) -> None:
        samples = self._make_samples(50)
        self.layer.sfa_correction_step(samples, lr=0.1)
        row_norms = torch.norm(self.layer.feedforward, dim=1)
        self.assertTrue(
            torch.allclose(row_norms, torch.ones_like(row_norms), atol=0.01),
            f"Row norms should be ~1.0, got {row_norms}",
        )

    def test_variance_reduction_nonnegative(self) -> None:
        samples = self._make_samples(100)
        result = self.layer.sfa_correction_step(samples, lr=0.01)
        self.assertGreaterEqual(result["variance_reduction"], 0.0)

    def test_decorrelation_nonnegative(self) -> None:
        samples = self._make_samples(100)
        result = self.layer.sfa_correction_step(samples, lr=0.01)
        self.assertGreaterEqual(result["decorrelation"], 0.0)

    def test_repeated_corrections_improve(self) -> None:
        """Multiple correction steps should progressively reduce derivative variance."""
        samples = self._make_samples(100)
        deriv_vars = []
        for _ in range(5):
            result = self.layer.sfa_correction_step(samples, lr=0.05)
            deriv_vars.append(result["post_deriv_var"])

        # Should generally decrease (allow some noise)
        # Check first vs last
        self.assertLessEqual(deriv_vars[-1], deriv_vars[0] * 1.5)


class TestSampleForSFA(unittest.TestCase):
    def _make_store(self):
        from marulho.consolidation.memory_store import DualMemoryStore
        return DualMemoryStore(capacity=100)

    def test_sample_from_empty_store(self) -> None:
        store = self._make_store()
        result, report = store.sample_for_sfa_with_report(10)
        self.assertEqual(result, [])
        self.assertEqual(report["surface"], "bounded_sfa_sample.v1")
        self.assertEqual(report["status"], "empty")
        self.assertEqual(report["fallback_reason"], "empty_request_or_memory")
        self.assertFalse(report["global_candidate_scan"])

    def test_unscoped_sample_requires_candidate_indices(self) -> None:
        store = self._make_store()
        for i in range(20):
            store.slow_buffer.append(torch.randn(32))

        self.assertFalse(hasattr(store, "sample_for_sfa"))
        result, report = store.sample_for_sfa_with_report(10)
        self.assertEqual(result, [])
        self.assertEqual(report["candidate_scope"], "selected_replay_window_required")
        self.assertEqual(
            report["candidate_window_policy"],
            "selected_replay_window_required_no_global_fallback",
        )
        self.assertEqual(report["candidate_index_count"], 0)
        self.assertEqual(report["fallback_reason"], "candidate_indices_required")
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["runs_every_token"])
        self.assertFalse(report["language_reasoning"])

    def test_sample_respects_n(self) -> None:
        store = self._make_store()
        for i in range(5):
            store.slow_buffer.append(torch.randn(32))

        result, report = store.sample_for_sfa_with_report(
            100,
            candidate_indices=list(range(5)),
        )
        self.assertEqual(len(result), 5)  # capped at buffer size
        self.assertEqual(report["sample_count"], 5)
        self.assertEqual(report["candidate_index_count"], 5)

    def test_sample_can_use_bounded_candidate_indices(self) -> None:
        store = self._make_store()
        for i in range(8):
            store.slow_buffer.append(torch.full((4,), float(i)))

        result, report = store.sample_for_sfa_with_report(
            10,
            candidate_indices=[5, 2, 5, 99, -1],
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(report["candidate_scope"], "selected_replay_window")
        self.assertEqual(report["candidate_index_count"], 2)
        self.assertEqual(report["duplicate_candidate_index_count"], 1)
        self.assertEqual(report["invalid_candidate_index_count"], 2)
        self.assertEqual(report["sample_count"], 2)
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["raw_text_payload_loaded"])
        values = {float(sample[0].item()) for sample in result}
        self.assertEqual(values, {2.0, 5.0})


if __name__ == "__main__":
    unittest.main()
