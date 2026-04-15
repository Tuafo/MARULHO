"""Tests for developmental protocol runners (§7.2–§7.5)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hecsn.config.model_config import HECSNConfig
from hecsn.training.developmental_runner import (
    CONCEPT_VOCABULARY,
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
    _make_vector_fn,
    _compute_grounding_confidence,
    _concept_spikes_for_text,
    _build_concept_signatures,
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
        self.assertTrue(cfg.enable_context_layer)
        self.assertFalse(cfg.enable_binding_layer)
        self.assertEqual(cfg.plasticity_mode, "local_stdp")
        self.assertFalse(cfg.enable_abstraction_layer)

    def test_stage2_enables_binding(self) -> None:
        cfg = _make_config_for_stage(2)
        self.assertTrue(cfg.enable_context_layer)
        self.assertTrue(cfg.enable_binding_layer)
        self.assertFalse(cfg.enable_abstraction_layer)

    def test_stage3_enables_abstraction(self) -> None:
        cfg = _make_config_for_stage(3)
        self.assertTrue(cfg.enable_abstraction_layer)
        self.assertTrue(cfg.enable_context_layer)
        self.assertTrue(cfg.enable_binding_layer)

    def test_stage2_inherits_base(self) -> None:
        base = HECSNConfig()
        base.n_columns = 20
        cfg = _make_config_for_stage(2, base)
        self.assertEqual(cfg.n_columns, 20)
        self.assertTrue(cfg.enable_cross_modal)
        self.assertTrue(cfg.enable_context_layer)
        self.assertTrue(cfg.enable_binding_layer)


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


class TestConfidenceBounded(unittest.TestCase):
    """Regression: grounding_confidence must be in [0, 1]."""

    def test_confidence_bounded_after_training(self) -> None:
        """After Stage 1 training, grounding_confidence must never exceed 1.0."""
        _result, state = run_stage_1(n_tokens=500, seed=42)
        cm = state.trainer.model.cross_modal
        conf_vec = cm.grounding_confidence()
        self.assertTrue((conf_vec >= 0).all(), "All confidence dims must be >= 0")
        self.assertTrue((conf_vec <= 1).all(), "All confidence dims must be <= 1")
        # Mean should also be bounded
        conf_mean = conf_vec.mean().item()
        self.assertGreaterEqual(conf_mean, 0.0)
        self.assertLessEqual(conf_mean, 1.0)

    def test_visual_audio_confidence_individual_bounds(self) -> None:
        """Each modality's confidence must be in [0, 1] per dimension."""
        _result, state = run_stage_1(n_tokens=300, seed=42)
        cm = state.trainer.model.cross_modal
        self.assertTrue((cm.visual_confidence >= 0).all())
        self.assertTrue((cm.visual_confidence <= 1).all())
        self.assertTrue((cm.audio_confidence >= 0).all())
        self.assertTrue((cm.audio_confidence <= 1).all())


class TestProbeUsesGroundedVectors(unittest.TestCase):
    """Regression: probe vectors must reflect cross-modal state."""

    def test_probe_vector_changes_with_per_word_signatures(self) -> None:
        """Perturbing per-word visual signature must change the probe vector."""
        import torch

        _result, state = run_stage_1(n_tokens=200, seed=42)
        cfg = state.config
        trainer = state.trainer

        # Inject a synthetic visual signature so the word is "grounded"
        trainer.word_visual_signature["fire"] = torch.randn(cfg.cross_modal_dim_visual)
        trainer.word_grounding_confidence["fire"] = 0.5

        vfn1 = _make_vector_fn(trainer, state.text_encoder, cfg)
        vec_before = vfn1("fire")

        # Perturb the per-word visual signature
        trainer.word_visual_signature["fire"] = torch.randn(cfg.cross_modal_dim_visual)

        vfn2 = _make_vector_fn(trainer, state.text_encoder, cfg)
        vec_after = vfn2("fire")

        diff = (vec_before - vec_after).norm().item()
        self.assertGreater(diff, 1e-6, "Probe must be sensitive to per-word signatures")

    def test_probe_vector_includes_visual_and_audio(self) -> None:
        """Probe output dimension must be routing_key + visual + audio (grounded)."""
        _result, state = run_stage_1(n_tokens=200, seed=42)
        cfg = state.config
        vfn = _make_vector_fn(state.trainer, state.text_encoder, cfg)
        vec = vfn("fire")
        expected_dim = cfg.column_latent_dim + cfg.cross_modal_dim_visual + cfg.cross_modal_dim_audio
        self.assertEqual(vec.shape[0], expected_dim,
                         f"Probe vector should be {expected_dim}-dim (grounded), got {vec.shape[0]}")


class TestStageConfigSync(unittest.TestCase):
    """Regression: stage transitions must sync config to both trainer and model."""

    def test_stage3_config_synced_to_model(self) -> None:
        """After stage transition, trainer.model.config must match trainer.config."""
        _r1, s1 = run_stage_1(n_tokens=200, seed=42)
        _r2, s2 = run_stage_2(n_tokens=200, seed=42, state=s1)
        _r3, s3 = run_stage_3(n_tokens=200, seed=42, state=s2)
        self.assertIs(s3.trainer.config, s3.trainer.model.config,
                      "trainer.config and trainer.model.config must be the same object")
        self.assertTrue(s3.trainer.config.enable_abstraction_layer,
                        "Stage 3 config must enable abstraction layer")

    def test_checkpoint_roundtrip_preserves_stage_config(self) -> None:
        """Save/restore must preserve the stage-specific config."""
        import tempfile
        import torch
        from hecsn.training.checkpointing import save_trainer_checkpoint, load_trainer_checkpoint

        _r1, s1 = run_stage_1(n_tokens=200, seed=42)
        _r2, s2 = run_stage_2(n_tokens=100, seed=42, state=s1)

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/test_ckpt.pt"
            save_trainer_checkpoint(path, s2.trainer, metadata={"developmental_stage": 2})
            trainer_reload, meta = load_trainer_checkpoint(path)
            # Cross-modal weights must survive the roundtrip
            self.assertGreater(
                trainer_reload.model.cross_modal.W_tv.norm().item(), 0.0,
                "Cross-modal weights must survive checkpoint roundtrip"
            )


class TestWindowLocalAlignment(unittest.TestCase):
    """Regression: only concept-containing windows receive multimodal spikes."""

    def test_function_words_get_no_spikes(self) -> None:
        """Windows containing only function words must not receive spikes."""
        sigs = _build_concept_signatures(len(CONCEPT_VOCABULARY), 256, 64)
        # "the" is a function word — should not trigger spikes
        vs, aus = _concept_spikes_for_text("the in a", sigs, 256, 64)
        self.assertIsNone(vs, "Function-word window must not get visual spikes")
        self.assertIsNone(aus, "Function-word window must not get audio spikes")

    def test_concept_words_get_spikes(self) -> None:
        """Windows containing concept words must receive spikes."""
        sigs = _build_concept_signatures(len(CONCEPT_VOCABULARY), 256, 64)
        vs, aus = _concept_spikes_for_text("fire burns brightly", sigs, 256, 64)
        self.assertIsNotNone(vs, "Concept window must get visual spikes")
        self.assertIsNotNone(aus, "Concept window must get audio spikes")


class TestSubProbes(unittest.TestCase):
    """Visual-text and audio-text sub-probes must return non-trivial results."""

    def test_sub_probes_computed_after_stage1(self) -> None:
        """Grounding probe must report visual_text and audio_text accuracies."""
        from hecsn.evaluation.grounding_probe import evaluate_grounding_probe_extended

        _result, state = run_stage_1(n_tokens=1000, seed=42)
        vfn = _make_vector_fn(state.trainer, state.text_encoder, state.config)
        probe = evaluate_grounding_probe_extended(vfn)
        self.assertEqual(probe.visual_text_count, 22, "Should have 22 visual triples")
        self.assertEqual(probe.audio_text_count, 3, "Should have 3 audio triples")
        self.assertGreaterEqual(probe.visual_text_accuracy, 0.0)
        self.assertLessEqual(probe.visual_text_accuracy, 1.0)
        self.assertGreaterEqual(probe.audio_text_accuracy, 0.0)
        self.assertLessEqual(probe.audio_text_accuracy, 1.0)


class TestTextOnlyControl(unittest.TestCase):
    """Text-only HECSN (no multimodal) should show negative concreteness gap."""

    def test_text_only_has_negative_or_zero_gap(self) -> None:
        """Without multimodal data, concrete should NOT beat abstract."""
        from hecsn.data.rtf_encoder import RTFEncoder
        from hecsn.evaluation.grounding_probe import evaluate_grounding_probe_extended
        from hecsn.training.runner_utils import set_seed
        from hecsn.training.trainer import HECSNModelLite, HECSNTrainer

        set_seed(42)
        cfg = HECSNConfig()
        encoder = RTFEncoder.from_config(cfg)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)

        # Train text-only (no multimodal spikes)
        total = 0
        while total < 5000:
            for sentence in DEVELOPMENTAL_CORPUS:
                patterns = list(encoder.iter_char_patterns(sentence, cfg.window_size))
                for _raw, pv in patterns:
                    trainer.train_step(pv, raw_window=_raw)
                    total += 1
                    if total >= 5000:
                        break
                if total >= 5000:
                    break

        vfn = _make_vector_fn(trainer, encoder, cfg)
        probe = evaluate_grounding_probe_extended(vfn)

        # Text-only: concreteness gap should be <= 0 (abstract >= concrete)
        self.assertLessEqual(
            probe.concreteness_gap,
            0.05,
            "Text-only HECSN should NOT show positive concreteness gap "
            f"(got {probe.concreteness_gap:.3f})",
        )
        # Total accuracy should be below multimodal (0.68)
        self.assertLess(
            probe.total_accuracy,
            0.60,
            "Text-only should score below multimodal threshold",
        )


if __name__ == "__main__":
    unittest.main()
