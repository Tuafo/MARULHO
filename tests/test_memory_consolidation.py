from __future__ import annotations

import unittest

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.core.columns import CompetitiveColumnLayer
from hecsn.consolidation.memory_store import DualMemoryStore
from hecsn.training.runner_utils import set_seed
from hecsn.training.memory_consolidation_runner import build_memory_consolidation_gate
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


class MemoryConsolidationTests(unittest.TestCase):
    def test_memory_store_snapshot_preserves_text_contexts(self) -> None:
        store = DualMemoryStore(capacity=8)
        assembly = torch.tensor([1.0, 0.0], dtype=torch.float32)
        pattern = torch.tensor([0.0, 1.0], dtype=torch.float32)
        store.update(
            assembly,
            importance=1.0,
            token_count=12,
            bucket_id=1,
            input_pattern=pattern,
            raw_window="purrs safe.",
            text="a cat purrs when it feels safe.",
            capture_tag=0.4,
        )

        restored = DualMemoryStore(capacity=8)
        restored.restore(store.snapshot())

        replay_entry = restored.replay_entry(0, current_token=12)
        self.assertEqual(replay_entry["raw_window"], "purrs safe.")
        self.assertEqual(replay_entry["text"], "a cat purrs when it feels safe.")

    def test_sleep_replay_skips_global_activity_state_updates(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=2,
            column_dim=2,
            input_dim=2,
            dead_column_steps=1,
            device=torch.device("cpu"),
        )
        routing_key = torch.tensor([1.0, 0.0], dtype=torch.float32)
        winner = torch.tensor([0], dtype=torch.long)
        layer.last_input_pattern = routing_key.clone()
        layer.steps_since_win = torch.ones_like(layer.steps_since_win)
        thresholds_before = layer.thresholds.clone()
        win_rate_before = layer.win_rate_ema.clone()
        update_count_before = int(layer.update_count)

        layer.process(
            routing_key,
            winner,
            modulator=0.5,
            update_global_state=False,
        )

        self.assertEqual(int(layer.update_count), update_count_before)
        self.assertTrue(torch.equal(layer.steps_since_win, torch.ones_like(layer.steps_since_win)))
        self.assertTrue(torch.equal(layer.thresholds, thresholds_before))
        self.assertTrue(torch.equal(layer.win_rate_ema, win_rate_before))
        self.assertEqual(int(layer.last_revived_indices.numel()), 0)

    def test_deep_sleep_replay_preserves_recent_pattern_reconstruction(self) -> None:
        set_seed(7)
        cfg = HECSNConfig(
            n_columns=12,
            column_latent_dim=24,
            bootstrap_tokens=0,
            memory_capacity=96,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
            deep_sleep_replay_steps=24,
            deep_sleep_candidate_pool=24,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)

        def pattern(*indices: int) -> torch.Tensor:
            vec = torch.zeros(cfg.input_dim, dtype=torch.float32)
            for idx in indices:
                vec[idx] = 1.0
            return vec / vec.sum()

        task_a = [
            ("alpha memory signal", pattern(1, 2, 3)),
            ("alpha plastic trace", pattern(4, 5, 6)),
            ("alpha stable concept", pattern(7, 8, 9)),
        ]
        task_b = [
            ("beta routing context", pattern(20, 21, 22)),
            ("beta semantic drift", pattern(23, 24, 25)),
            ("beta retrieval anchor", pattern(26, 27, 28)),
        ]

        def mean_recon(items: list[tuple[str, torch.Tensor]]) -> float:
            return sum(trainer.reconstruction_error(pattern_vec) for _, pattern_vec in items) / len(items)

        for _ in range(18):
            for raw_window, pattern_vec in task_a:
                trainer.train_step(pattern_vec, raw_window=raw_window)
        task_a_after_a = mean_recon(task_a)

        trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
        trainer.capture_recent_memory_anchors(window_tokens=trainer.token_count, strength=8.0)
        trainer.run_sleep_maintenance(mode="deep", cycles=2)

        for _ in range(18):
            for raw_window, pattern_vec in task_b:
                trainer.train_step(pattern_vec, raw_window=raw_window)
        task_a_after_b = mean_recon(task_a)

        trainer.run_sleep_maintenance(mode="deep", cycles=4)
        task_a_after_consolidation = mean_recon(task_a)

        self.assertLessEqual(task_a_after_consolidation, task_a_after_b * 2.0)
        self.assertLessEqual(task_a_after_consolidation, task_a_after_a * 5.0)

    def test_capture_tags_recruit_prp_and_raise_replay_priority(self) -> None:
        store = DualMemoryStore(
            capacity=4,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
            capture_release=0.5,
            consolidation_rate=1.0,
            prp_synthesis_rate=0.5,
        )
        store.update(torch.tensor([1.0, 0.0]), token_count=0, importance=0.5, bucket_id=0, routing_key=torch.tensor([1.0, 0.0]))
        store.update(torch.tensor([0.0, 1.0]), token_count=2, importance=0.5, bucket_id=1, routing_key=torch.tensor([0.0, 1.0]))

        tagged = store.tag_recent_entries(current_token=2, window_tokens=1, strength=2.0)
        scores_before = store.replay_scores(current_token=2)
        tagged_entry = store.replay_entry(1, current_token=2)

        self.assertEqual(tagged, 1)
        self.assertGreater(tagged_entry["prp_level"], 0.0)
        self.assertGreater(tagged_entry["capture_strength"], 0.0)
        self.assertGreater(float(scores_before[1].item()), float(scores_before[0].item()))

        store.consolidate_replay([1], current_token=3, blend=0.5, protein_synthesis_level=1.25)
        consolidated_entry = store.replay_entry(1, current_token=3)

        self.assertGreater(consolidated_entry["consolidation_level"], 0.0)
        self.assertLess(consolidated_entry["capture_tag"], 2.0)
        self.assertEqual(store.slow_consolidation_events[1], 1)
        self.assertEqual(store.slow_replay_count[1], 1)

    def test_capture_tag_decay_uses_functional_minutes(self) -> None:
        store = DualMemoryStore(
            capacity=1,
            ema_alpha=0.1,
            functional_minute=10,
            capture_tag_decay=0.5,
            tag_duration_weak=1e12,
            tag_duration_strong=1e12,
            prp_tau_weak=1e12,
            prp_tau_strong=1e12,
        )
        store.update(
            torch.tensor([1.0, 0.0]),
            token_count=0,
            importance=1.0,
            bucket_id=0,
            routing_key=torch.tensor([1.0, 0.0]),
        )

        store.tag_recent_entries(current_token=0, window_tokens=1, strength=1.0)
        decayed_entry = store.replay_entry(0, current_token=10)

        self.assertAlmostEqual(decayed_entry["capture_tag"], 0.5, places=4)

    def test_fragility_priority_prefers_stale_unconsolidated_memories(self) -> None:
        store = DualMemoryStore(
            capacity=2,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
            tag_duration_weak=1e12,
            tag_duration_strong=1e12,
            prp_tau_weak=1e12,
            prp_tau_strong=1e12,
        )
        store.update(torch.tensor([1.0, 0.0]), token_count=0, importance=0.8, bucket_id=0, routing_key=torch.tensor([1.0, 0.0]))
        store.update(torch.tensor([0.0, 1.0]), token_count=0, importance=0.8, bucket_id=1, routing_key=torch.tensor([0.0, 1.0]))

        store.slow_consolidation_level[0] = 0.95
        store.slow_last_replay_token[0] = 35
        store.slow_replay_count[0] = 4
        store.slow_capture_tag[0] = 0.8
        store.slow_local_prp[0] = 0.8

        store.slow_consolidation_level[1] = 0.15
        store.slow_last_replay_token[1] = 0
        store.slow_replay_count[1] = 0
        store.slow_capture_tag[1] = 0.1
        store.slow_local_prp[1] = 0.1

        scores = store.maintenance_scores(current_token=40)

        self.assertGreater(float(scores[1].item()), float(scores[0].item()))
        self.assertEqual(store.sample_replay_indices(n=1, current_token=40, strategy="maintenance"), [1])

    def test_micro_sleep_refreshes_tags_without_weight_commit(self) -> None:
        set_seed(7)
        cfg = HECSNConfig(
            n_columns=10,
            column_latent_dim=20,
            bootstrap_tokens=0,
            memory_capacity=48,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[1:5] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="alpha memory trace")

        tagged = trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=2.0)
        self.assertGreater(tagged, 0)

        before_proto = trainer.model.competitive.prototypes.detach().clone()
        before_weights = trainer.model.competitive.input_weights.detach().clone()
        before_levels = list(trainer.model.memory_store.slow_consolidation_level)
        before_tags = list(trainer.model.memory_store.slow_capture_tag)

        updates = trainer.run_sleep_maintenance(mode="micro", cycles=1)

        self.assertGreater(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertTrue(torch.allclose(before_weights, trainer.model.competitive.input_weights))
        self.assertEqual(before_levels, trainer.model.memory_store.slow_consolidation_level)
        self.assertGreater(sum(trainer.model.memory_store.slow_replay_count), 0)
        self.assertLessEqual(
            max(trainer.model.memory_store.slow_capture_tag),
            max(before_tags),
        )

    def test_repair_sleep_reanchors_prototypes_without_consolidation(self) -> None:
        set_seed(7)
        cfg = HECSNConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        trainer.memory_warm_started = True
        trainer.token_count = 10

        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()
        routing_key = trainer.model.routing_key_from_pattern(pattern)
        assembly = trainer.model.competitive.assembly_from_input(pattern.to(trainer.model.device)).detach().cpu()
        trainer.model.memory_store.update(
            assembly,
            importance=1.0,
            token_count=trainer.token_count,
            bucket_id=0,
            input_pattern=pattern,
            routing_key=routing_key.detach().cpu(),
            raw_window="repair memory trace",
            text="repair memory trace",
            capture_tag=0.4,
        )

        disturbed = torch.roll(routing_key.detach().cpu(), shifts=1, dims=0)
        disturbed = torch.nn.functional.normalize(disturbed, dim=0).to(trainer.model.device)
        trainer.model.competitive.prototypes[0] = disturbed
        before_distance = float(torch.norm(trainer.model.competitive.prototypes[0] - routing_key.to(trainer.model.device)).item())
        before_weights = trainer.model.competitive.input_weights.detach().clone()
        before_levels = list(trainer.model.memory_store.slow_consolidation_level)

        updates = trainer._sleep_replay("repair")
        after_distance = float(torch.norm(trainer.model.competitive.prototypes[0] - routing_key.to(trainer.model.device)).item())

        self.assertEqual(updates, 1)
        self.assertLess(after_distance, before_distance)
        self.assertTrue(torch.allclose(before_weights, trainer.model.competitive.input_weights))
        self.assertEqual(before_levels, trainer.model.memory_store.slow_consolidation_level)

    def test_wake_learning_reports_consolidation_resistance(self) -> None:
        set_seed(7)
        cfg = HECSNConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[3:7] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(8):
            trainer.train_step(pattern, raw_window="stable consolidated trace")

        winner_id = int(trainer.last_winner)
        matched = False
        for idx, bucket_id in enumerate(trainer.model.memory_store.slow_bucket_ids):
            if bucket_id == winner_id:
                trainer.model.memory_store.slow_consolidation_level[idx] = 1.0
                matched = True
        self.assertTrue(matched)

        metrics = trainer.train_step(pattern, raw_window="stable consolidated trace")

        self.assertGreater(metrics["winner_consolidation_level"], 0.0)
        self.assertLess(metrics["effective_modulator"], metrics["surprise"])

    def test_memory_consolidation_gate_uses_absolute_threshold_at_numerical_floor(self) -> None:
        gate = build_memory_consolidation_gate(
            task_a_after_a=5e-7,
            task_a_after_b=2.95e-4,
            task_a_after_consolidation=2.91e-4,
            task_a_overlap_after_consolidation=0.90,
        )

        self.assertTrue(gate["uses_absolute_degradation_gate"])
        self.assertTrue(gate["task_a_degradation_ok"])
        self.assertTrue(gate["task_a_recovery_nonnegative"])
        self.assertTrue(gate["pass"])

    def test_nearest_prototype_distance_is_clamped_non_negative(self) -> None:
        layer = CompetitiveColumnLayer(
            n_columns=1,
            column_dim=2,
            input_dim=2,
            device=torch.device("cpu"),
        )
        routing_key = torch.tensor([1.0, 1.0], dtype=torch.float32)
        normalized = torch.nn.functional.normalize(routing_key, dim=0)

        with torch.no_grad():
            layer.prototypes[0] = normalized * 1.000001

        self.assertEqual(layer.nearest_prototype_distance(normalized), 0.0)

    def test_snapshot_roundtrip_preserves_prp_state_stack(self) -> None:
        store = DualMemoryStore(
            capacity=2,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
            consolidation_rate=1.0,
            prp_synthesis_rate=0.4,
        )
        store.update(torch.tensor([1.0, 0.0]), token_count=1, importance=0.8, bucket_id=0, routing_key=torch.tensor([1.0, 0.0]))
        store.tag_recent_entries(current_token=2, window_tokens=4, strength=1.5)
        store.consolidate_replay([0], current_token=3, blend=0.4, protein_synthesis_level=1.2)

        snapshot = store.snapshot()
        restored = DualMemoryStore(capacity=1)
        restored.restore(snapshot)

        self.assertEqual(restored.slow_capture_tag, store.slow_capture_tag)
        self.assertEqual(restored.slow_tag_is_strong, store.slow_tag_is_strong)
        self.assertEqual(restored.slow_replay_count, store.slow_replay_count)
        self.assertAlmostEqual(restored.global_prp_pool, store.global_prp_pool, places=6)
        self.assertAlmostEqual(restored.summary_stats()["mean_capture_strength"], store.summary_stats()["mean_capture_strength"], places=6)

if __name__ == "__main__":
    unittest.main()
