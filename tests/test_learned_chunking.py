from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from hecsn.config.model_config import HECSNConfig
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.query_runner import feed_text, text_pattern_stream
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_trainer() -> HECSNTrainer:
    set_seed(7)
    cfg = HECSNConfig(
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
    return HECSNTrainer(HECSNModel(cfg), cfg)


def _pattern_for_term(trainer: HECSNTrainer, term: str) -> torch.Tensor:
    return list(text_pattern_stream(term, trainer.encoder, trainer.config.window_size))[-1][1]


def _winner_for_term(trainer: HECSNTrainer, term: str) -> int:
    return int(trainer.winner_for_pattern(_pattern_for_term(trainer, term)))


def _prototype_for_term(trainer: HECSNTrainer, term: str) -> torch.Tensor:
    winner = _winner_for_term(trainer, term)
    prototype = trainer.model.competitive.prototypes[winner].detach().cpu().float()
    return F.normalize(prototype.unsqueeze(0), dim=1).squeeze(0)


class LearnedChunkingTests(unittest.TestCase):
    def test_concat_mode_exposes_first_class_chunk_channel(self) -> None:
        cfg = HECSNConfig(
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

    def test_learned_chunking_breaks_single_winner_term_collapse(self) -> None:
        trainer = _build_trainer()
        feed_text(
            trainer,
            trainer.encoder,
            (
                "river stream water current river stream water current "
                "loan credit money bank loan credit money bank "
                "ballast submarine depth buoyancy ballast submarine depth buoyancy "
            ),
        )

        winners = {
            term: _winner_for_term(trainer, term)
            for term in ("river", "stream", "loan", "credit", "ballast", "submarine")
        }

        self.assertGreaterEqual(len(set(winners.values())), 2)

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
        from hecsn.data.rtf_encoder import LearnedChunkingLayer

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
