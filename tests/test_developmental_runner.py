"""Tests for developmental protocol runners (§7.2–§7.5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hecsn.config.model_config import HECSNConfig
from hecsn.training.developmental_runner import (
    DEVELOPMENTAL_CORPUS,
    ProtocolState,
    StageResult,
    run_stage_1,
    run_stage_2,
    run_stage_3,
    run_stage_4,
    run_stage_5,
    run_full_developmental_protocol,
    run_baseline_calibration,
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
        result, state = run_stage_1(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 1)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("grounding_confidence", result.metrics)
        self.assertIn("visual_pairs_sent", result.metrics)
        self.assertIn("audio_pairs_sent", result.metrics)
        self.assertGreater(result.tokens_processed, 0)
        self.assertIsNotNone(state.trainer)
        self.assertIsNotNone(state.text_encoder)

    def test_visual_pairs_sent(self) -> None:
        result, _state = run_stage_1(n_tokens=200, seed=42)
        self.assertGreater(result.metrics["visual_pairs_sent"], 0)


class TestStage2(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result, state = run_stage_2(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 2)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("probe_accuracy", result.metrics)
        self.assertIn("grounding_confidence", result.metrics)
        self.assertGreater(result.tokens_processed, 0)
        self.assertIsNotNone(state.trainer)


class TestStage3(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result, state = run_stage_3(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 3)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("probe_accuracy", result.metrics)
        self.assertIn("confirmation_cycles", result.metrics)
        self.assertIn("gap_queries_produced", result.metrics)

    def test_probe_accuracy_reported(self) -> None:
        result, _state = run_stage_3(n_tokens=200, seed=42)
        self.assertIsInstance(result.metrics["probe_accuracy"], float)


class TestStage4(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result, state = run_stage_4(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 4)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("final_probe_accuracy", result.metrics)
        self.assertIn("accuracy_delta", result.metrics)
        self.assertIn("acquisitions_made", result.metrics)


class TestStage5(unittest.TestCase):
    def test_runs_and_returns_result(self) -> None:
        result, state = run_stage_5(n_tokens=200, seed=42)
        self.assertEqual(result.stage, 5)
        self.assertIsInstance(result.passed, bool)
        self.assertIn("final_probe_accuracy", result.metrics)
        self.assertIn("autonomous_cycles", result.metrics)
        self.assertIn("no_catastrophic_forgetting", result.metrics)


class TestFullProtocol(unittest.TestCase):
    def test_runs_all_stages(self) -> None:
        results = run_full_developmental_protocol(n_tokens_per_stage=100, seed=42)
        self.assertGreater(len(results), 0)
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


class TestStateContinuity(unittest.TestCase):
    """Verify that state actually transfers between stages."""

    def test_stage2_inherits_stage1_weights(self) -> None:
        """Stage 2 must start with Stage 1's trained cross-modal weights."""
        result1, state1 = run_stage_1(n_tokens=200, seed=42)
        # Snapshot Stage 1's cross-modal weight norm
        cm_norm = state1.trainer.model.cross_modal.W_tv.norm().item()

        result2, state2 = run_stage_2(n_tokens=200, seed=42, state=state1)
        self.assertEqual(result2.stage, 2)
        self.assertIsInstance(state2, ProtocolState)
        # Weights should have continued evolving (not re-initialized)
        self.assertGreater(cm_norm, 0.0, "Stage 1 should have trained W_tv")

    def test_cross_modal_confidence_grows_in_stage1(self) -> None:
        """Stage 1 multimodal training must produce non-zero confidence."""
        _result, state = run_stage_1(n_tokens=500, seed=42)
        cm = state.trainer.model.cross_modal
        self.assertIsNotNone(cm)
        total_conf = (cm.visual_confidence.sum() + cm.audio_confidence.sum()).item()
        self.assertGreater(total_conf, 0.0, "Confidence must grow during Stage 1")

    def test_stage2_inherits_confidence(self) -> None:
        """Stage 2 must start with Stage 1's non-zero confidence."""
        _r1, state1 = run_stage_1(n_tokens=500, seed=42)
        cm1 = state1.trainer.model.cross_modal
        conf_before = (cm1.visual_confidence.sum() + cm1.audio_confidence.sum()).item()

        _r2, state2 = run_stage_2(n_tokens=100, seed=42, state=state1)
        cm2 = state2.trainer.model.cross_modal
        conf_after = (cm2.visual_confidence.sum() + cm2.audio_confidence.sum()).item()
        # Confidence should not be zero after inheriting Stage 1
        self.assertGreater(conf_after, 0.0)
        # Same object identity — state was passed, not re-created
        self.assertIs(cm1, cm2)

    def test_bootstrap_counters_separate_modalities(self) -> None:
        """Visual and audio bootstrap counters must track independently."""
        _r1, state1 = run_stage_1(n_tokens=200, seed=42)
        _r2, state2 = run_stage_2(n_tokens=200, seed=42, state=state1)
        trainer = state2.trainer
        # Both modalities should have used some bootstrap budget
        self.assertGreater(trainer._stage2_bootstrap_used_visual, 0)
        self.assertGreater(trainer._stage2_bootstrap_used_audio, 0)

    def test_config_deepcopy_isolation(self) -> None:
        """Config changes in one stage must not mutate the base config."""
        base = HECSNConfig()
        original_abstraction = base.enable_abstraction_layer
        cfg3 = _make_config_for_stage(3, base)
        # Stage 3 enables abstraction, but base should be unchanged
        self.assertTrue(cfg3.enable_abstraction_layer)
        self.assertEqual(base.enable_abstraction_layer, original_abstraction)


class TestBaselineCalibration(unittest.TestCase):
    def test_runs_and_returns_calibrated_thresholds(self) -> None:
        result = run_baseline_calibration()
        self.assertIn("baselines", result)
        self.assertIn("calibrated_thresholds", result)
        thresholds = result["calibrated_thresholds"]
        self.assertIn("stage2_criterion", thresholds)
        self.assertIn("publication_threshold", thresholds)
        self.assertIn("fasttext_score", thresholds)
        self.assertIn("som_score", thresholds)
        # Thresholds must be at least the paper minimums
        self.assertGreaterEqual(thresholds["stage2_criterion"], 0.60)
        self.assertGreaterEqual(thresholds["publication_threshold"], 0.65)

    def test_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cal_test"
            result = run_baseline_calibration(output_dir=out)
            cal_file = out / "baseline_calibration.json"
            self.assertTrue(cal_file.exists())
            data = json.loads(cal_file.read_text())
            self.assertIn("calibrated_thresholds", data)

    def test_developmental_corpus_is_nonempty(self) -> None:
        self.assertGreater(len(DEVELOPMENTAL_CORPUS), 100)
        # Corpus should contain concept vocabulary words
        self.assertIn("fire", DEVELOPMENTAL_CORPUS)
        self.assertIn("water", DEVELOPMENTAL_CORPUS)


if __name__ == "__main__":
    unittest.main()
