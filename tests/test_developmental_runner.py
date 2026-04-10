"""Tests for developmental protocol runners (§7.2–§7.5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hecsn.config.model_config import HECSNConfig
from hecsn.training.developmental_runner import (
    StageResult,
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_45,
    run_full_developmental_protocol,
    _make_config_for_stage,
    _compute_grounding_confidence,
)


class TestStageResult(unittest.TestCase):
    def test_to_dict(self) -> None:
        r = StageResult(stage=1, passed=True, metrics={"x": 0.5}, tokens_processed=100)
        d = r.to_dict()
        self.assertEqual(d["stage"], 1)
        self.assertTrue(d["passed"])
        self.assertEqual(d["metrics"]["x"], 0.5)
        self.assertEqual(d["tokens_processed"], 100)


class TestConfigForStage(unittest.TestCase):
    def test_stage1_config(self) -> None:
        cfg = _make_config_for_stage(1)
        self.assertEqual(cfg.context_mode, "adaptive")
        self.assertEqual(cfg.plasticity_rule, "triplet")
        self.assertTrue(cfg.enable_cross_modal)

    def test_stage2_inherits_base(self) -> None:
        base = HECSNConfig()
        base.n_columns = 20
        cfg = _make_config_for_stage(2, base)
        self.assertEqual(cfg.n_columns, 20)
        self.assertTrue(cfg.enable_cross_modal)


class TestStage1(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result = run_stage_1(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 1)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("grounding_confidence", result.metrics)
        self.assertIn("visual_sparsity", result.metrics)
        self.assertIn("audio_sparsity", result.metrics)
        self.assertGreater(result.tokens_processed, 0)

    def test_visual_sparsity_nonzero(self) -> None:
        result = run_stage_1(n_tokens=200, seed=42)
        self.assertGreater(result.metrics["visual_sparsity"], 0.0)


class TestStage2(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result = run_stage_2(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 2)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("probe_accuracy", result.metrics)
        self.assertIn("filter_precision", result.metrics)
        self.assertGreater(result.tokens_processed, 0)


class TestStage3(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result = run_stage_3(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 3)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("probe_accuracy", result.metrics)
        self.assertIn("ungrounded_rate", result.metrics)
        self.assertIn("confirmation_rate", result.metrics)


class TestStage45(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result = run_stage_45(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 45)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("final_probe_accuracy", result.metrics)
        self.assertIn("accuracy_delta", result.metrics)


class TestFullProtocol(unittest.TestCase):
    def test_runs_all_stages(self) -> None:
        results = run_full_developmental_protocol(n_tokens_per_stage=100, seed=42)
        self.assertGreater(len(results), 0)
        # At least stage 1 should run
        self.assertEqual(results[0].stage, 1)

    def test_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dev_test"
            results = run_full_developmental_protocol(
                n_tokens_per_stage=100,
                seed=42,
                output_dir=out,
            )
            self.assertTrue(out.exists())
            summary_file = out / "developmental_summary.json"
            self.assertTrue(summary_file.exists())
            summary = json.loads(summary_file.read_text())
            self.assertIn("stages_completed", summary)
            self.assertIn("total_tokens", summary)


if __name__ == "__main__":
    unittest.main()
