from __future__ import annotations

from array import array
import math
import unittest
from unittest.mock import patch

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.core.columns import CompetitiveColumnLayer
from marulho.consolidation.memory_store import DualMemoryStore
from marulho.training.runner_utils import set_seed
from marulho.training.memory_consolidation_runner import build_memory_consolidation_gate
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


class MemoryConsolidationTests(unittest.TestCase):
    def test_train_step_can_defer_due_sleep_maintenance_until_allowed(self) -> None:
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=1,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[1:5] = 1.0
        pattern = pattern / pattern.sum()
        trainer.token_count = cfg.deep_sleep_interval_tokens
        calls: list[str] = []

        def _fake_sleep_replay(mode: str) -> int:
            calls.append(mode)
            return 1

        with patch.object(trainer, "_sleep_replay", side_effect=_fake_sleep_replay):
            deferred_metrics = trainer.train_step(
                pattern,
                raw_window="alpha",
                allow_sleep_maintenance=False,
            )
            allowed_metrics = trainer.train_step(pattern, raw_window="beta")

        self.assertEqual(calls, ["deep"])
        self.assertEqual(deferred_metrics["sleep_triggered"], 0)
        self.assertEqual(deferred_metrics["sleep_maintenance_deferred"], 1)
        self.assertEqual(allowed_metrics["sleep_triggered"], 1)
        self.assertEqual(allowed_metrics["sleep_type"], "deep")

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

    def test_memory_store_device_report_marks_archival_storage_cpu(self) -> None:
        store = DualMemoryStore(capacity=8)
        store.update(
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            importance=1.0,
            token_count=1,
            bucket_id=2,
            input_pattern=torch.tensor([0.0, 1.0], dtype=torch.float32),
            routing_key=torch.tensor([1.0, 0.0], dtype=torch.float32),
        )

        report = store.device_report()

        self.assertEqual(report["storage_role"], "archival_replay_ledger")
        self.assertEqual(report["expected_storage_device"], "cpu")
        self.assertEqual(report["slow_buffer_devices"], {"cpu": 1})
        self.assertEqual(report["slow_input_pattern_devices"], {"cpu": 1})
        self.assertEqual(report["slow_routing_key_devices"], {"cpu": 1})
        self.assertEqual(report["fast_ema_device"], "cpu")
        self.assertTrue(report["all_archival_tensors_cpu"])
        self.assertEqual(report["stc_state_storage"], "zero_copy_array_buffer")
        self.assertTrue(report["stc_decay_zero_copy"])
        self.assertEqual(report["stc_state_bytes"], 17)
        self.assertEqual(report["hot_path"]["update_calls"], 1)
        self.assertEqual(report["hot_path"]["admission_count"], 1)
        self.assertEqual(report["hot_path"]["optional_payload_copy_count"], 2)
        self.assertEqual(report["hot_path"]["ripple_awake_bucket_scan_count"], 0)
        self.assertEqual(
            report["hot_path"]["last_ripple_awake_candidate_count"],
            0,
        )

    def test_awake_ripple_tagging_uses_awake_bucket_index(self) -> None:
        store = DualMemoryStore(capacity=8)
        for token, bucket_id in enumerate([1, 2, 1, 3], start=1):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                importance=0.8,
                token_count=token,
                bucket_id=bucket_id,
            )

        snapshot = store.snapshot()
        tagged = store.ripple_tag_awake(
            current_token=5,
            window_tokens=10,
            da_level=0.95,
            awake_bucket_ids=[1, 1],
        )

        self.assertEqual(tagged, 2)
        self.assertEqual(store.last_ripple_scan_mode, "awake_bucket_index")
        self.assertEqual(store.ripple_scalar_scan_count, 0)
        self.assertEqual(store.ripple_vector_scan_count, 0)
        self.assertEqual(store.ripple_awake_bucket_scan_count, 1)
        self.assertEqual(store.last_ripple_awake_bucket_count, 1)
        self.assertEqual(store.last_ripple_awake_candidate_count, 2)
        tagged_buckets = [
            bucket_id
            for bucket_id, strength in zip(
                store.slow_bucket_ids,
                store.slow_ripple_strength,
            )
            if float(strength) > 0.0
        ]
        self.assertEqual(tagged_buckets, [1, 1])

        restored = DualMemoryStore(capacity=8)
        restored.restore(snapshot)
        restored_tagged = restored.ripple_tag_awake(
            current_token=5,
            window_tokens=10,
            da_level=0.95,
            awake_bucket_ids=torch.tensor([3], dtype=torch.long),
        )
        self.assertEqual(restored_tagged, 1)
        self.assertEqual(restored.last_ripple_scan_mode, "awake_bucket_index")
        self.assertEqual(restored.last_ripple_awake_candidate_count, 1)
        self.assertEqual(restored.slow_bucket_ids[3], 3)
        self.assertGreater(float(restored.slow_ripple_strength[3]), 0.0)

    def test_awake_ripple_bucket_index_updates_on_reservoir_replacement(self) -> None:
        store = DualMemoryStore(capacity=1)
        store.update(
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            token_count=1,
            bucket_id=1,
        )

        with patch("torch.randint", return_value=torch.tensor([0])):
            store.update(
                torch.tensor([0.0, 1.0], dtype=torch.float32),
                token_count=2,
                bucket_id=2,
            )

        old_bucket_tagged = store.ripple_tag_awake(
            current_token=3,
            window_tokens=10,
            da_level=0.95,
            awake_bucket_ids=[1],
        )
        self.assertEqual(old_bucket_tagged, 0)
        self.assertEqual(store.last_ripple_awake_candidate_count, 0)

        new_bucket_tagged = store.ripple_tag_awake(
            current_token=3,
            window_tokens=10,
            da_level=0.95,
            awake_bucket_ids=[2],
        )
        self.assertEqual(new_bucket_tagged, 1)
        self.assertEqual(store.last_ripple_awake_candidate_count, 1)

    def test_memory_store_rejected_reservoir_sample_skips_optional_payload_copies(self) -> None:
        store = DualMemoryStore(capacity=1)
        store.update(
            torch.tensor([1.0, 0.0]),
            token_count=1,
            input_pattern=torch.tensor([0.0, 1.0]),
            routing_key=torch.tensor([1.0, 0.0]),
        )
        copies_before = store.optional_payload_copy_count

        with patch("torch.randint", return_value=torch.tensor([1])):
            admitted = store.update(
                torch.tensor([0.0, 1.0]),
                token_count=2,
                input_pattern=torch.tensor([1.0, 0.0]),
                routing_key=torch.tensor([0.0, 1.0]),
            )

        self.assertIsNone(admitted)
        self.assertEqual(store.optional_payload_copy_count, copies_before)
        self.assertEqual(store.optional_payload_copy_avoidance_count, 2)
        self.assertEqual(store.reservoir_rejection_count, 1)
        self.assertEqual(store.update_calls, 2)

    def test_stc_decay_uses_zero_copy_numeric_buffers_with_expected_values(self) -> None:
        store = DualMemoryStore(
            capacity=2,
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
            capture_tag=1.0,
        )
        tag_buffer = store.slow_capture_tag
        prp_buffer = store.slow_local_prp
        initial_prp = store.slow_local_prp[0]

        store._advance_state(10)

        self.assertIsInstance(store.slow_capture_tag, array)
        self.assertIsInstance(store.slow_tag_is_strong, array)
        self.assertIsInstance(store.slow_local_prp, array)
        self.assertIs(store.slow_capture_tag, tag_buffer)
        self.assertIs(store.slow_local_prp, prp_buffer)
        expected_tag = 0.5 * math.exp(
            -10.0 / store._tag_tau_tokens(True)
        )
        expected_prp = initial_prp * math.exp(
            -10.0 / store._prp_tau_tokens(True)
        )
        self.assertAlmostEqual(store.slow_capture_tag[0], expected_tag, places=12)
        self.assertAlmostEqual(store.slow_local_prp[0], expected_prp, places=12)

    def test_model_device_report_includes_memory_store_boundary(self) -> None:
        cfg = MarulhoConfig(n_columns=4, column_latent_dim=8, bootstrap_tokens=0, memory_capacity=8)
        model = MarulhoModel(cfg)

        report = model.subcortex_device_report()["memory_store"]

        self.assertEqual(report["storage_role"], "archival_replay_ledger")
        self.assertEqual(report["expected_storage_device"], "cpu")
        self.assertTrue(report["all_archival_tensors_cpu"])

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
        cfg = MarulhoConfig(
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
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)

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

    def test_bounded_replay_window_selection_scores_only_bucket_candidates(self) -> None:
        store = DualMemoryStore(
            capacity=8,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token, bucket_id in enumerate([1, 2, 1, 3], start=1):
            admitted = store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=bucket_id,
                capture_tag=1.0,
            )
            self.assertIsNotNone(admitted)
            store.slow_local_prp[int(admitted)] = 1.0

        report = store.select_replay_window(
            n=2,
            current_token=8,
            candidate_pool=4,
            strategy="consolidation",
            candidate_bucket_ids=[1, 1],
        )

        self.assertEqual(report["surface"], "bounded_replay_window_selection.v1")
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertTrue(report["bounded_by_bucket_index"])
        self.assertFalse(report["global_score_scan"])
        self.assertEqual(report["candidate_bucket_ids"], [1])
        self.assertEqual(report["candidate_index_count"], 2)
        self.assertEqual(report["score_count"], 2)
        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(
            {store.slow_bucket_ids[idx] for idx in report["selected_indices"]},
            {1},
        )
        self.assertEqual(
            store.summary_stats()["last_replay_selection_report"]["score_count"],
            2,
        )

        restored = DualMemoryStore(capacity=1)
        restored.restore(store.snapshot())
        restored_report = restored.summary_stats()["last_replay_selection_report"]
        self.assertEqual(restored_report["surface"], "bounded_replay_window_selection.v1")
        self.assertEqual(restored_report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(restored_report["selected_count"], 2)

    def test_global_replay_selection_retires_zero_pressure_window(self) -> None:
        store = DualMemoryStore(
            capacity=4,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        admitted = store.update(
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            token_count=1,
            importance=1.0,
            bucket_id=1,
            capture_tag=0.0,
        )
        self.assertIsNotNone(admitted)

        report = store.select_replay_window(
            n=1,
            current_token=8,
            candidate_pool=2,
            strategy="consolidation",
        )

        self.assertEqual(report["candidate_scope"], "global_slow_path_score_scan")
        self.assertTrue(report["global_score_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["selected_indices"], [])
        self.assertEqual(report["fallback_reason"], "no_positive_global_scores")
        self.assertEqual(store.sample_replay_indices(n=1, current_token=9), [])

    def test_bounded_replay_window_recall_uses_bucket_routing_keys(self) -> None:
        store = DualMemoryStore(
            capacity=8,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token, bucket_id, key in (
            (1, 1, torch.tensor([1.0, 0.0], dtype=torch.float32)),
            (2, 2, torch.tensor([0.0, 1.0], dtype=torch.float32)),
            (3, 1, torch.tensor([0.8, 0.2], dtype=torch.float32)),
        ):
            admitted = store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=bucket_id,
                input_pattern=key,
                routing_key=key,
                capture_tag=1.0,
            )
            self.assertIsNotNone(admitted)

        report = store.recall_replay_window(
            query_routing_key=torch.tensor([1.0, 0.0], dtype=torch.float32),
            query_input_pattern=torch.tensor([1.0, 0.0], dtype=torch.float32),
            current_token=8,
            candidate_bucket_ids=[1],
            max_candidates=4,
        )

        self.assertEqual(report["surface"], "bounded_replay_window_recall.v1")
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(report["score_device"], "cpu")
        self.assertEqual(report["archival_storage_device"], "cpu")
        self.assertEqual(report["routing_key_count"], 2)
        self.assertEqual(report["input_pattern_count"], 2)
        self.assertEqual(
            {store.slow_bucket_ids[idx] for idx in report["routing_key_indices"]},
            {1},
        )
        self.assertLess(report["best_distance"], 1e-5)
        self.assertLess(report["best_input_distance"], 1e-5)

        restored = DualMemoryStore(capacity=1)
        restored.restore(store.snapshot())
        restored_report = restored.summary_stats()["last_replay_recall_report"]
        self.assertEqual(restored_report["surface"], "bounded_replay_window_recall.v1")
        self.assertLess(restored_report["best_distance"], 1e-5)
        self.assertLess(restored_report["best_input_distance"], 1e-5)

    def test_deep_sleep_uses_anchor_bucket_replay_window_report(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
            deep_sleep_replay_steps=4,
            deep_sleep_candidate_pool=8,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="anchored alpha replay")
        trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=trainer.token_count,
            strength=2.0,
        )
        for idx in range(len(trainer.model.memory_store.slow_local_prp)):
            trainer.model.memory_store.slow_local_prp[idx] = 1.0

        updates = trainer.run_sleep_maintenance(mode="deep", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertGreater(anchored, 0)
        self.assertGreater(updates, 0)
        self.assertEqual(report["candidate_bucket_source"], "column_anchor_bucket_index")
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertTrue(report["bounded_by_bucket_index"])
        self.assertFalse(report["global_score_scan"])
        self.assertEqual(report["sleep_replay_applied_count"], updates)
        self.assertTrue(report["sleep_replay_mutates_runtime_state"])
        self.assertTrue(report["sleep_replay_applies_plasticity"])
        self.assertEqual(
            report["sleep_replay_commit_strategy"],
            "bounded_reconstruction_gated_candidate_repair",
        )
        self.assertEqual(report["sleep_replay_winner_source"], "bounded_route_candidates")
        self.assertFalse(report["sleep_replay_forced_stored_bucket_winner"])
        self.assertGreater(report["sleep_replay_candidate_column_union_count"], 0)
        self.assertGreater(report["sleep_replay_candidate_column_trial_count"], 0)
        self.assertGreaterEqual(
            report["sleep_replay_quality_before"],
            report["sleep_replay_quality_after"],
        )

    def test_deep_sleep_without_anchors_blocks_global_replay_mutation(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
            deep_sleep_replay_steps=4,
            deep_sleep_candidate_pool=8,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="unanchored alpha replay")
        trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
        before_proto = trainer.model.competitive.prototypes.detach().clone()

        updates = trainer.run_sleep_maintenance(mode="deep", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(report["candidate_bucket_ids"], [])
        self.assertFalse(report["global_score_scan"])
        self.assertTrue(report["unscoped_global_fallback_retired"])
        self.assertEqual(
            report["global_fallback_blocked_reason"],
            "no_anchor_bucket_scope_for_deep_replay",
        )
        self.assertFalse(report["sleep_replay_mutates_runtime_state"])
        self.assertFalse(report["sleep_replay_applies_plasticity"])

    def test_deep_sleep_anchor_zero_pressure_blocks_global_replay_mutation(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
            deep_sleep_replay_steps=4,
            deep_sleep_candidate_pool=8,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="zero-pressure alpha replay")
        trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=trainer.token_count,
            strength=2.0,
        )
        store = trainer.model.memory_store
        for idx in range(len(store.slow_buffer)):
            store.slow_capture_tag[idx] = 0.0
            store.slow_local_prp[idx] = 0.0
        before_proto = trainer.model.competitive.prototypes.detach().clone()

        updates = trainer.run_sleep_maintenance(mode="deep", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertGreater(anchored, 0)
        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertGreater(report["candidate_bucket_count"], 0)
        self.assertFalse(report["global_score_scan"])
        self.assertTrue(report["unscoped_global_fallback_retired"])
        self.assertEqual(
            report["global_fallback_blocked_reason"],
            "bucket_window_zero_positive_replay_pressure",
        )
        self.assertFalse(report["sleep_replay_mutates_runtime_state"])
        self.assertFalse(report["sleep_replay_applies_plasticity"])

    def test_micro_sleep_refreshes_tags_without_weight_commit(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=10,
            column_latent_dim=20,
            bootstrap_tokens=0,
            memory_capacity=48,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[1:5] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="alpha memory trace")

        tagged = trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=2.0)
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=trainer.token_count,
            strength=2.0,
        )
        self.assertGreater(tagged, 0)

        before_proto = trainer.model.competitive.prototypes.detach().clone()
        before_weights = trainer.model.competitive.input_weights.detach().clone()
        before_levels = list(trainer.model.memory_store.slow_consolidation_level)
        before_tags = list(trainer.model.memory_store.slow_capture_tag)

        updates = trainer.run_sleep_maintenance(mode="micro", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertGreater(anchored, 0)
        self.assertGreater(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertTrue(torch.allclose(before_weights, trainer.model.competitive.input_weights))
        self.assertEqual(before_levels, trainer.model.memory_store.slow_consolidation_level)
        self.assertGreater(sum(trainer.model.memory_store.slow_replay_count), 0)
        self.assertLessEqual(
            max(trainer.model.memory_store.slow_capture_tag),
            max(before_tags),
        )
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertFalse(report["global_score_scan"])
        self.assertEqual(
            report["sleep_replay_commit_strategy"],
            "bounded_micro_maintenance_refresh",
        )
        self.assertEqual(
            report["sleep_replay_winner_source"],
            "bucket_indexed_replay_window",
        )
        self.assertTrue(report["sleep_replay_bypasses_competitive_process"])
        self.assertTrue(report["sleep_replay_mutates_runtime_state"])
        self.assertFalse(report["sleep_replay_applies_plasticity"])

    def test_micro_sleep_without_anchors_blocks_global_maintenance_refresh(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=10,
            column_latent_dim=20,
            bootstrap_tokens=0,
            memory_capacity=48,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[1:5] = 1.0
        pattern = pattern / pattern.sum()

        for _ in range(6):
            trainer.train_step(pattern, raw_window="unanchored micro trace")

        tagged = trainer.tag_recent_memories(
            window_tokens=trainer.token_count,
            strength=2.0,
        )
        before_proto = trainer.model.competitive.prototypes.detach().clone()
        before_weights = trainer.model.competitive.input_weights.detach().clone()
        before_replay_count = list(trainer.model.memory_store.slow_replay_count)
        before_tags = list(trainer.model.memory_store.slow_capture_tag)

        updates = trainer.run_sleep_maintenance(mode="micro", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertGreater(tagged, 0)
        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertTrue(torch.allclose(before_weights, trainer.model.competitive.input_weights))
        self.assertEqual(before_replay_count, trainer.model.memory_store.slow_replay_count)
        self.assertEqual(before_tags, list(trainer.model.memory_store.slow_capture_tag))
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(report["candidate_bucket_ids"], [])
        self.assertFalse(report["global_score_scan"])
        self.assertTrue(report["unscoped_global_fallback_retired"])
        self.assertEqual(
            report["global_fallback_blocked_reason"],
            "no_anchor_bucket_scope_for_micro_replay",
        )
        self.assertFalse(report["sleep_replay_mutates_runtime_state"])
        self.assertFalse(report["sleep_replay_applies_plasticity"])

    def test_repair_sleep_reanchors_prototypes_without_consolidation(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=1,
            strength=2.0,
        )

        disturbed = torch.roll(routing_key.detach().cpu(), shifts=1, dims=0)
        disturbed = torch.nn.functional.normalize(disturbed, dim=0).to(trainer.model.device)
        trainer.model.competitive.prototypes[0] = disturbed
        before_distance = float(torch.norm(trainer.model.competitive.prototypes[0] - routing_key.to(trainer.model.device)).item())
        before_weights = trainer.model.competitive.input_weights.detach().clone()
        before_levels = list(trainer.model.memory_store.slow_consolidation_level)

        updates = trainer._sleep_replay("repair")
        after_distance = float(torch.norm(trainer.model.competitive.prototypes[0] - routing_key.to(trainer.model.device)).item())

        self.assertGreater(anchored, 0)
        self.assertEqual(updates, 1)
        self.assertLess(after_distance, before_distance)
        self.assertTrue(torch.allclose(before_weights, trainer.model.competitive.input_weights))
        self.assertEqual(before_levels, trainer.model.memory_store.slow_consolidation_level)
        report = trainer._last_sleep_replay_selection_report
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertFalse(report["global_score_scan"])
        self.assertEqual(report["sleep_replay_commit_strategy"], "bounded_repair_reanchor")
        self.assertEqual(
            report["sleep_replay_winner_source"],
            "stored_replay_bucket_with_anchor_scope",
        )

    def test_repair_sleep_without_anchors_blocks_global_repair_mutation(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        trainer.memory_warm_started = True
        trainer.token_count = 10

        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()
        routing_key = trainer.model.routing_key_from_pattern(pattern)
        assembly = trainer.model.competitive.assembly_from_input(
            pattern.to(trainer.model.device)
        ).detach().cpu()
        trainer.model.memory_store.update(
            assembly,
            importance=1.0,
            token_count=trainer.token_count,
            bucket_id=0,
            input_pattern=pattern,
            routing_key=routing_key.detach().cpu(),
            raw_window="unanchored repair memory trace",
            text="unanchored repair memory trace",
            capture_tag=0.4,
        )

        disturbed = torch.roll(routing_key.detach().cpu(), shifts=1, dims=0)
        disturbed = torch.nn.functional.normalize(disturbed, dim=0).to(
            trainer.model.device
        )
        trainer.model.competitive.prototypes[0] = disturbed
        before_proto = trainer.model.competitive.prototypes.detach().clone()

        updates = trainer._sleep_replay("repair")
        report = trainer._last_sleep_replay_selection_report

        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(report["candidate_bucket_ids"], [])
        self.assertFalse(report["global_score_scan"])
        self.assertTrue(report["unscoped_global_fallback_retired"])
        self.assertEqual(
            report["global_fallback_blocked_reason"],
            "no_anchor_bucket_scope_for_repair_replay",
        )
        self.assertFalse(report["sleep_replay_mutates_runtime_state"])
        self.assertFalse(report["sleep_replay_applies_plasticity"])

    def test_wake_learning_reports_consolidation_resistance(self) -> None:
        set_seed(7)
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=32,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        self.assertEqual(restored.update_calls, store.update_calls)
        self.assertEqual(restored.admission_count, store.admission_count)
        self.assertEqual(
            restored.optional_payload_copy_count,
            store.optional_payload_copy_count,
        )

    def test_stc_sensitivity_task_a_recall_robust_across_functional_minute(self) -> None:
        """§4.9 STC sensitivity: Task-A recall must be robust across 100x range of functional_minute."""
        from marulho.training.memory_consolidation_runner import (
            build_memory_consolidation_gate,
            collect_assemblies,
            mean_assembly_overlap,
            mean_reconstruction_error,
        )
        from marulho.training.runner_utils import set_seed

        task_a_windows = ("alpha memory signal", "alpha plastic trace", "alpha stable concept")
        task_b_windows = ("beta routing context", "beta semantic drift", "beta retrieval anchor")

        results: list[dict] = []
        for fm in (100, 500, 2000, 10000):
            set_seed(42)
            cfg = MarulhoConfig(
                n_columns=12,
                column_latent_dim=24,
                bootstrap_tokens=0,
                memory_capacity=96,
                eta_competitive=0.05,
                eta_decay=0.0,
                input_weight_blend=0.0,
                plasticity_mode="local_stdp",
                micro_sleep_interval_tokens=10**9,
                deep_sleep_interval_tokens=10**9,
                deep_sleep_replay_steps=24,
                deep_sleep_candidate_pool=24,
                enable_learned_chunking=False,
                functional_minute=fm,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            task_a = [trainer.encoder.feature_vector([ord(c) for c in w]).float() for w in task_a_windows]
            task_b = [trainer.encoder.feature_vector([ord(c) for c in w]).float() for w in task_b_windows]

            for _ in range(18):
                for i, w in enumerate(task_a_windows):
                    trainer.train_step(task_a[i], raw_window=w)
            a_after_a = mean_reconstruction_error(trainer, task_a)
            ref_asm = collect_assemblies(trainer, task_a)

            trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
            trainer.run_sleep_maintenance(mode="deep", cycles=2)

            for _ in range(18):
                for i, w in enumerate(task_b_windows):
                    trainer.train_step(task_b[i], raw_window=w)
            a_after_b = mean_reconstruction_error(trainer, task_a)

            trainer.run_sleep_maintenance(mode="deep", cycles=4)
            a_after_consol = mean_reconstruction_error(trainer, task_a)
            overlap = mean_assembly_overlap(ref_asm, collect_assemblies(trainer, task_a))

            gate = build_memory_consolidation_gate(
                task_a_after_a=a_after_a,
                task_a_after_b=a_after_b,
                task_a_after_consolidation=a_after_consol,
                task_a_overlap_after_consolidation=overlap,
            )
            results.append({"fm": fm, "gate_pass": gate["pass"], "overlap": overlap})

        for r in results:
            self.assertTrue(r["gate_pass"], f"Gate failed at functional_minute={r['fm']}")
            self.assertGreaterEqual(r["overlap"], 0.50, f"Overlap too low at fm={r['fm']}")

if __name__ == "__main__":
    unittest.main()
