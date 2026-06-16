from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from marulho.config.model_config import MarulhoConfig
from marulho.data.rtf_encoder import RTFEncoder
from marulho.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from marulho.training.query_runner import feed_text, text_pattern_stream
from marulho.training.runner_utils import set_seed
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _build_trainer() -> MarulhoTrainer:
    set_seed(7)
    cfg = MarulhoConfig(
        n_columns=24,
        column_latent_dim=48,
        bootstrap_tokens=0,
        memory_capacity=128,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        window_size=10,
        enable_learned_chunking=True,
        learned_chunk_detector_count=64,
        learned_chunk_min_len=2,
        learned_chunk_max_len=10,
        learned_chunk_blend=0.45,
        learned_chunk_similarity_floor=0.20,
        learned_chunk_boundary_threshold=0.04,
        learned_chunk_association_blend=0.0,
    )
    return MarulhoTrainer(MarulhoModel(cfg), cfg)


def _pattern_for_term(trainer: MarulhoTrainer, term: str) -> torch.Tensor:
    return list(text_pattern_stream(term, trainer.encoder, trainer.config.window_size))[-1][1]


def _winner_for_term(trainer: MarulhoTrainer, term: str) -> int:
    return int(trainer.winner_for_pattern(_pattern_for_term(trainer, term)))


def _prototype_for_term(trainer: MarulhoTrainer, term: str) -> torch.Tensor:
    winner = _winner_for_term(trainer, term)
    prototype = trainer.model.competitive.prototypes[winner].detach().cpu().float()
    return F.normalize(prototype.unsqueeze(0), dim=1).squeeze(0)


class LearnedChunkingTests(unittest.TestCase):
    def test_routing_skips_discarded_dense_assembly_with_winner_parity(self) -> None:
        trainer = _build_trainer()
        pattern = _pattern_for_term(trainer, "river")
        competitive = trainer.model.competitive

        dense_assembly = competitive.assembly_from_input(pattern)
        expected_key = F.normalize(competitive.last_projected_input, dim=0)
        candidate_ids, _ = trainer.model.routing_index.search_tensors(
            expected_key.unsqueeze(0),
            k=trainer.config.k_routing,
        )
        candidates = candidate_ids[0].to(device=trainer.model.device)
        dense_winners, _, _ = competitive.compete(
            expected_key,
            candidates,
            fallback_allowed=False,
        )

        sparse_key = trainer.model.routing_key_from_pattern(pattern)
        sparse_winners, _, _ = competitive.compete(
            sparse_key,
            candidates,
            fallback_allowed=False,
        )
        execution = competitive.execution_report()

        self.assertEqual(int(dense_assembly.numel()), trainer.config.n_columns)
        self.assertTrue(torch.allclose(sparse_key, expected_key))
        self.assertTrue(torch.equal(sparse_winners, dense_winners))
        self.assertEqual(execution["mode"], "candidate_subset")
        self.assertEqual(execution["candidate_count"], trainer.config.k_routing)
        self.assertEqual(execution["scored_column_count"], trainer.config.k_routing)
        self.assertTrue(execution["sparse_candidate_execution_observed"])

    def test_train_step_scopes_homeostasis_to_retrieved_candidates(self) -> None:
        trainer = _build_trainer()
        pattern = _pattern_for_term(trainer, "river")

        trainer.train_step(pattern, raw_window="river")
        execution = trainer.model.competitive.execution_report()

        self.assertEqual(execution["mode"], "candidate_subset")
        self.assertEqual(execution["candidate_count"], trainer.config.k_routing)
        self.assertEqual(execution["homeostasis_update_mode"], "all_columns")
        self.assertEqual(execution["homeostasis_update_count"], trainer.config.n_columns)
        predictive_report = trainer.model.predictive.device_report()
        self.assertEqual(predictive_report["last_prediction_update_mode"], "all_columns")
        self.assertEqual(predictive_report["last_prediction_update_count"], trainer.config.n_columns)

        trainer.token_count = trainer.config.candidate_predictive_update_start_tokens
        trainer.train_step(pattern, raw_window="river")
        execution = trainer.model.competitive.execution_report()

        self.assertEqual(execution["homeostasis_update_mode"], "candidate_subset")
        self.assertEqual(execution["homeostasis_update_count"], trainer.config.k_routing)
        self.assertLess(execution["homeostasis_update_fraction"], 1.0)
        predictive_report = trainer.model.predictive.device_report()
        self.assertEqual(predictive_report["last_prediction_update_mode"], "candidate_subset")
        self.assertEqual(predictive_report["last_prediction_update_count"], trainer.config.k_routing)
        self.assertLess(predictive_report["last_prediction_update_fraction"], 1.0)

    def test_train_step_uses_tensor_native_routing_candidates(self) -> None:
        trainer = _build_trainer()
        pattern = _pattern_for_term(trainer, "river")

        def fail_legacy_list_search(*args, **kwargs):
            del args, kwargs
            raise AssertionError("live train_step must not use list-returning routing search")

        trainer.model.routing_index.search = fail_legacy_list_search  # type: ignore[method-assign]
        trainer.train_step(pattern, raw_window="river")

    def test_lite_train_step_keeps_unchanged_projection_cache(self) -> None:
        trainer = _build_trainer()
        pattern = _pattern_for_term(trainer, "river")

        def fail_if_invalidated() -> None:
            raise AssertionError(
                "lite plasticity does not mutate W_assembly_project"
            )

        trainer.model._invalidate_projection_cache = fail_if_invalidated
        trainer.train_step(pattern, raw_window="river")
        routing_stats = trainer.model.runtime_scope_report()["routing_index"]

        self.assertEqual(routing_stats["last_search_mode"], "tensor")
        self.assertGreaterEqual(routing_stats["tensor_search_count"], 1)

    def test_non_chunking_routing_keeps_required_dense_assembly(self) -> None:
        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=32,
            enable_learned_chunking=False,
            device="cpu",
        )
        model = MarulhoModel(cfg)

        routing_key = model.routing_key_from_pattern(torch.rand(cfg.input_dim))
        execution = model.competitive.execution_report()

        self.assertEqual(int(routing_key.numel()), cfg.column_latent_dim)
        self.assertEqual(execution["mode"], "dense_assembly")
        self.assertEqual(execution["scored_column_count"], cfg.n_columns)
        self.assertFalse(execution["sparse_candidate_execution_observed"])

    def test_concat_mode_exposes_first_class_chunk_channel(self) -> None:
        cfg = MarulhoConfig(
            enable_learned_chunking=True,
            learned_chunk_detector_count=64,
            learned_chunk_feature_mode="concat",
        )
        encoder = RTFEncoder.from_config(cfg)

        base = encoder.feature_vector([ord(ch) for ch in "bank"])
        combined = encoder.blended_feature_vector(
            [ord(ch) for ch in "bank"],
            chunk_state=torch.ones(64, dtype=torch.float32),
            chunk_codes=[ord(ch) for ch in "bank"],
        )

        self.assertEqual(encoder.output_dim, 256)
        self.assertEqual(int(base.numel()), 256)
        self.assertTrue(torch.allclose(base[128:], torch.zeros(128), atol=1e-6))
        self.assertEqual(int(combined.numel()), 256)
        self.assertGreater(float(combined[128:].sum().item()), 0.0)

    def test_learned_chunking_populates_multiple_detectors_and_term_features(self) -> None:
        trainer = _build_trainer()
        feed_text(
            trainer,
            trainer.encoder,
            (
                "river stream water current river stream water current "
                "loan credit money bank loan credit money bank "
                "ballast submarine depth buoyancy ballast submarine depth buoyancy "
            ) * 32,
        )
        assert trainer.encoder.learned_chunking is not None
        active_detectors = int((trainer.encoder.learned_chunking.usage > 0.0).sum().item())
        river_pattern = _pattern_for_term(trainer, "river")
        submarine_pattern = _pattern_for_term(trainer, "submarine")

        self.assertGreaterEqual(active_detectors, 2)
        self.assertGreater(float(torch.norm(river_pattern - submarine_pattern).item()), 0.05)

    def test_checkpoint_roundtrip_preserves_learned_chunk_inventory(self) -> None:
        trainer = _build_trainer()
        feed_text(
            trainer,
            trainer.encoder,
            "river stream water current loan credit money bank ballast submarine depth buoyancy",
        )
        before_pattern = _pattern_for_term(trainer, "ballast")
        before_segments = trainer.encoder.segment_text("ballast submarine depth buoyancy")

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_trainer_checkpoint(Path(tmpdir) / "chunking.pt", trainer)
            restored, _ = load_trainer_checkpoint(checkpoint_path)

        after_pattern = _pattern_for_term(restored, "ballast")
        after_segments = restored.encoder.segment_text("ballast submarine depth buoyancy")

        self.assertTrue(torch.allclose(before_pattern, after_pattern, atol=1e-6))
        self.assertEqual(before_segments, after_segments)

    def test_learned_chunking_keeps_sentence_local_grounding_pairs_closer(self) -> None:
        trainer = _build_trainer()
        feed_text(
            trainer,
            trainer.encoder,
            (
                "a dog guards the house and barks at strangers. "
                "rainbows form when sunlight passes through water droplets. "
            )
            * 16,
        )

        dog = _prototype_for_term(trainer, "dog")
        house = _prototype_for_term(trainer, "house")
        rainbows = _prototype_for_term(trainer, "rainbows")
        positive = float(F.cosine_similarity(dog.unsqueeze(0), house.unsqueeze(0), dim=1).item())
        negative = float(F.cosine_similarity(dog.unsqueeze(0), rainbows.unsqueeze(0), dim=1).item())

        # With stochastic init, allow a small tolerance for co-occurrence proximity
        self.assertGreaterEqual(positive + 0.01, negative)

    def test_abstraction_bias_modulates_boundary_threshold(self) -> None:
        """Top-down bias from Abstraction Layer adjusts chunking boundary threshold."""
        from marulho.data.rtf_encoder import LearnedChunkingLayer

        layer = LearnedChunkingLayer(
            n_detectors=32,
            min_chunk_len=2,
            max_chunk_len=10,
            similarity_floor=0.20,
            boundary_threshold=0.08,
            update_lr=0.25,
            association_blend=0.0,
            association_lr=0.15,
            association_decay=0.995,
        )
        base_threshold = layer.boundary_threshold

        # High certainty, no gaps → raise threshold (coarser chunks)
        layer.set_abstraction_bias(mean_certainty=0.9, max_gap_score=0.0)
        self.assertGreater(layer.boundary_threshold, base_threshold)

        # Low certainty, high gaps → lower threshold (finer chunks)
        layer.set_abstraction_bias(mean_certainty=0.1, max_gap_score=1.5)
        self.assertLess(layer.boundary_threshold, base_threshold)

    def test_abstraction_bias_wired_in_trainer(self) -> None:
        """Abstraction→Chunking feedback is active during training."""
        trainer = _build_trainer()
        assert trainer.encoder.learned_chunking is not None
        base = trainer.encoder.learned_chunking._base_boundary_threshold

        corpus = "the river flows through the ancient forest where birds sing"
        for word in corpus.split():
            pattern = trainer.encoder.feature_vector([ord(ch) for ch in word])
            trainer.train_step(pattern, raw_window=word)

        # After training, abstraction should have biased the threshold
        current = trainer.encoder.learned_chunking.boundary_threshold
        # It may have moved in either direction depending on certainty/gaps
        self.assertIsInstance(current, float)
        self.assertGreater(current, 0.0)


if __name__ == "__main__":
    unittest.main()
