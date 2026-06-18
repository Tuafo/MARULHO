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
from marulho.training.memory_consolidation_runner import (
    REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
    REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_POLICY,
    _bounded_replay_recall_evaluation,
    _collect_anchor_replay_queries,
    _run_reconstruction_guarded_sleep_maintenance,
    build_memory_consolidation_gate,
)
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


class _IndexOnlySequence:
    def __init__(self, values: list[object]) -> None:
        self._values = values
        self.iteration_attempts = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> object:
        return self._values[int(index)]

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_attempts += 1
        raise AssertionError("archive iteration is not allowed")


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

    def test_runtime_concept_memory_lookup_uses_explicit_indices_without_archive_iteration(self) -> None:
        store = DualMemoryStore(capacity=16)
        store.slow_buffer = [torch.zeros(2, dtype=torch.float32) for _ in range(16)]
        routing_keys = _IndexOnlySequence(
            [
                torch.tensor([1.0, 0.0], dtype=torch.float32)
                if index == 3
                else None
                for index in range(16)
            ]
        )
        stored_texts = _IndexOnlySequence(
            [
                "submarine ballast memory evidence"
                if index == 3
                else None
                for index in range(16)
            ]
        )
        stored_windows = _IndexOnlySequence(
            [
                "submarine ballast raw evidence"
                if index == 3
                else None
                for index in range(16)
            ]
        )
        store.slow_routing_keys = routing_keys  # type: ignore[assignment]
        store.slow_texts = stored_texts  # type: ignore[assignment]
        store.slow_raw_windows = stored_windows  # type: ignore[assignment]
        store.slow_importance = [1.0 for _ in range(16)]
        store.slow_capture_tag = array("d", [0.25 for _ in range(16)])
        store.slow_consolidation_level = [0.5 for _ in range(16)]

        resolved = store.resolve_runtime_concept_memory_matches(
            observations=[
                ("fallback one", {"memory_index": 3}),
                ("fallback two", {"memory_index": 3}),
                ("invalid", {}),
                ("missing", None),
                ("out", {"memory_index": 99}),
            ],
            max_observations=8,
        )
        report = resolved["report"]

        self.assertEqual(report["surface"], "bounded_runtime_concept_memory_lookup.v1")
        self.assertEqual(report["candidate_scope"], "train_step_memory_index_evidence")
        self.assertEqual(report["candidate_window_policy"], "explicit_train_step_memory_indices_only")
        self.assertEqual(report["match_indices"], [3, 3])
        self.assertEqual(report["candidate_index_count"], 2)
        self.assertEqual(report["unique_candidate_index_count"], 1)
        self.assertEqual(report["raw_text_payload_count"], 1)
        self.assertEqual(report["raw_text_payload_cache_hits"], 1)
        self.assertEqual(report["invalid_memory_index_count"], 1)
        self.assertEqual(report["invalid_observation_count"], 1)
        self.assertEqual(report["out_of_bounds_index_count"], 1)
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["runs_every_token"])
        self.assertFalse(report["language_reasoning"])
        self.assertEqual(report["archival_storage_device"], "cpu")
        self.assertEqual(resolved["result_slots"], [0, 1, None, None, None])
        self.assertEqual(len(resolved["matches"]), 2)
        self.assertEqual(routing_keys.iteration_attempts, 0)
        self.assertEqual(stored_texts.iteration_attempts, 0)
        self.assertEqual(stored_windows.iteration_attempts, 0)
        self.assertEqual(
            store.last_runtime_concept_memory_lookup_report["match_indices"],
            [3, 3],
        )

    def test_frontier_gap_collection_uses_bounded_recent_index(self) -> None:
        store = DualMemoryStore(capacity=64)
        assembly = torch.tensor([1.0, 0.0], dtype=torch.float32)
        for token in range(40):
            store.update(
                assembly,
                importance=1.0,
                token_count=token,
                bucket_id=token % 4,
                raw_window=f"frontier memory {token}",
                capture_tag=0.2,
            )

        report = store.collect_frontier_gap_indices(current_token=40, max_candidates=8)

        self.assertEqual(report["surface"], "bounded_frontier_gap_candidates.v1")
        self.assertEqual(report["status"], "collected")
        self.assertEqual(report["candidate_window_policy"], "recent_entry_index_candidate_window")
        self.assertEqual(report["candidate_index_count"], 8)
        self.assertEqual(report["candidate_indices"][0], 39)
        self.assertEqual(report["candidate_indices"][-1], 32)
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["raw_text_payload_loaded"])
        self.assertFalse(report["language_reasoning"])

        restored = DualMemoryStore(capacity=64)
        restored.restore(store.snapshot())
        self.assertEqual(
            restored.last_frontier_gap_collection_report["candidate_index_count"],
            8,
        )

    def test_replay_entry_can_exclude_text_payload_for_sleep_replay(self) -> None:
        store = DualMemoryStore(capacity=8)
        assembly = torch.tensor([1.0, 0.0], dtype=torch.float32)
        pattern = torch.tensor([0.0, 1.0], dtype=torch.float32)
        store.update(
            assembly,
            importance=1.0,
            token_count=12,
            bucket_id=1,
            input_pattern=pattern,
            routing_key=assembly,
            raw_window="bounded replay raw window",
            text="expanded replay text should stay out of sleep replay",
            metadata={"source": "unit"},
            capture_tag=0.4,
        )

        replay_entry = store.replay_entry(
            0,
            current_token=12,
            include_text_payload=False,
        )

        self.assertIsInstance(replay_entry["assembly"], torch.Tensor)
        self.assertIsInstance(replay_entry["input_pattern"], torch.Tensor)
        self.assertIsInstance(replay_entry["routing_key"], torch.Tensor)
        self.assertIsNone(replay_entry["raw_window"])
        self.assertIsNone(replay_entry["text"])
        self.assertIsNone(replay_entry["metadata"])

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
        report = store.last_awake_ripple_tag_report
        self.assertEqual(report["surface"], "bounded_awake_ripple_tag.v1")
        self.assertEqual(report["candidate_scope"], "awake_bucket_index_candidate_window")
        self.assertEqual(report["candidate_window_policy"], "recent_bucket_round_robin_candidate_pool")
        self.assertEqual(report["candidate_index_available_count"], 2)
        self.assertEqual(report["candidate_index_count"], 2)
        self.assertEqual(report["tagged_count"], 2)
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_every_token"])
        self.assertEqual(report["archival_storage_device"], "cpu")
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

    def test_awake_ripple_tagging_caps_awake_bucket_candidates(self) -> None:
        store = DualMemoryStore(capacity=16)
        for token in range(1, 11):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                importance=0.8,
                token_count=token,
                bucket_id=1,
            )
        old_strength = float(store.slow_ripple_strength[0])

        tagged = store.ripple_tag_awake(
            current_token=10,
            window_tokens=10,
            da_level=0.95,
            awake_bucket_ids=[1],
            max_candidate_entries=3,
        )
        report = store.last_awake_ripple_tag_report

        self.assertEqual(tagged, 3)
        self.assertEqual(report["surface"], "bounded_awake_ripple_tag.v1")
        self.assertEqual(report["candidate_window_limit"], 3)
        self.assertEqual(report["candidate_index_available_count"], 10)
        self.assertEqual(report["candidate_index_count"], 3)
        self.assertEqual(report["candidate_indices"], [9, 8, 7])
        self.assertEqual(report["tagged_count"], 3)
        self.assertEqual(report["scan_mode"], "awake_bucket_index")
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["diagnostic_global_candidate_scan"])
        self.assertFalse(report["runs_every_token"])
        self.assertEqual(report["archival_storage_device"], "cpu")
        self.assertGreater(float(store.slow_ripple_strength[9]), 0.0)
        self.assertEqual(float(store.slow_ripple_strength[0]), old_strength)

    def test_awake_ripple_unscoped_requires_awake_bucket_scope(self) -> None:
        store = DualMemoryStore(capacity=16)
        for token in range(1, 5):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                importance=0.8,
                token_count=token,
                bucket_id=token,
            )

        tagged = store.ripple_tag_awake(
            current_token=5,
            window_tokens=5,
            da_level=0.95,
        )
        report = store.last_awake_ripple_tag_report

        self.assertEqual(tagged, 0)
        self.assertEqual(store.last_ripple_scan_mode, "awake_bucket_scope_required")
        self.assertEqual(store.ripple_scalar_scan_count, 0)
        self.assertEqual(store.ripple_vector_scan_count, 0)
        self.assertEqual(report["candidate_scope"], "awake_bucket_scope_required")
        self.assertEqual(
            report["candidate_window_policy"],
            "awake_bucket_scope_required_no_global_fallback",
        )
        self.assertEqual(
            report["fallback_reason"],
            "awake_bucket_scope_required_for_ripple_tagging",
        )
        self.assertFalse(report["global_candidate_scan"])

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
        scores_before = store.replay_scores_for_indices([0, 1], current_token=2)
        tagged_entry = store.replay_entry(1, current_token=2)

        self.assertEqual(tagged, 1)
        self.assertGreater(tagged_entry["prp_level"], 0.0)
        self.assertGreater(tagged_entry["capture_strength"], 0.0)
        self.assertGreater(float(scores_before[1]), float(scores_before[0]))

        store.consolidate_replay([1], current_token=3, blend=0.5, protein_synthesis_level=1.25)
        consolidated_entry = store.replay_entry(1, current_token=3)

        self.assertGreater(consolidated_entry["consolidation_level"], 0.0)
        self.assertLess(consolidated_entry["capture_tag"], 2.0)
        self.assertEqual(store.slow_consolidation_events[1], 1)
        self.assertEqual(store.slow_replay_count[1], 1)

    def test_recent_memory_tagging_uses_capped_recency_index(self) -> None:
        store = DualMemoryStore(
            capacity=16,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token in range(1, 11):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=token,
                capture_tag=0.0,
            )
        old_tag_before = float(store.slow_capture_tag[0])

        tagged = store.tag_recent_entries(
            current_token=10,
            window_tokens=10,
            strength=2.0,
            max_recent_entries=3,
        )
        report = store.last_recent_memory_tag_report

        self.assertEqual(tagged, 3)
        self.assertEqual(report["surface"], "bounded_recent_memory_tag.v1")
        self.assertEqual(report["candidate_window_policy"], "recent_entry_index_reverse_window")
        self.assertEqual(report["candidate_window_limit"], 3)
        self.assertEqual(report["candidate_index_count"], 3)
        self.assertEqual(report["candidate_indices"], [9, 8, 7])
        self.assertTrue(report["candidate_index_available_count_is_lower_bound"])
        self.assertEqual(report["tagged_count"], 3)
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertGreater(store.slow_capture_tag[9], 0.0)
        self.assertLessEqual(float(store.slow_capture_tag[0]), old_tag_before)

    def test_recent_anchor_capture_uses_capped_recency_index(self) -> None:
        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            deep_sleep_candidate_pool=1,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        trainer._recent_replay_setup_limit = lambda: 3  # type: ignore[method-assign]
        for token in range(1, 11):
            trainer.model.memory_store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=token,
                capture_tag=0.0,
            )
        trainer.token_count = 10

        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=10,
            strength=4.0,
        )
        report = trainer.model.memory_store.last_anchor_capture_report

        self.assertEqual(anchored, 3)
        self.assertEqual(report["surface"], "bounded_recent_anchor_capture.v1")
        self.assertEqual(report["candidate_window_limit"], 3)
        self.assertEqual(report["candidate_index_count"], 3)
        self.assertEqual(report["candidate_indices"], [9, 8, 7])
        self.assertEqual(report["captured_entry_count"], 3)
        self.assertEqual(report["captured_anchor_count"], 3)
        self.assertEqual(report["candidate_bucket_ids"], [8, 9, 10])
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertTrue({8, 9, 10}.issubset(set(trainer.column_anchors)))
        self.assertNotIn(1, trainer.column_anchors)

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

        self.assertFalse(hasattr(store, "maintenance_scores"))
        self.assertFalse(hasattr(store, "consolidation_scores"))
        self.assertFalse(hasattr(store, "repair_scores"))
        self.assertFalse(hasattr(store, "fragility_scores"))
        report = store.select_replay_window(
            n=1,
            current_token=40,
            candidate_pool=2,
            strategy="maintenance",
            candidate_bucket_ids=[0, 1],
        )
        self.assertEqual(report["selected_indices"], [1])
        self.assertEqual(report["score_count"], 2)
        self.assertFalse(hasattr(store, "sample_replay_indices"))
        bounded_report = store.select_replay_window(
            n=1,
            current_token=40,
            strategy="maintenance",
            candidate_bucket_ids=[1],
        )
        self.assertEqual(
            bounded_report["selected_indices"],
            [1],
        )
        self.assertEqual(
            store.summary_stats()["last_replay_selection_report"]["candidate_scope"],
            "bucket_indexed_candidate_window",
        )

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
        self.assertFalse(report["raw_text_payload_loaded"])
        self.assertFalse(report["language_reasoning"])
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

    def test_bucket_replay_selection_caps_candidate_window_before_scoring(self) -> None:
        store = DualMemoryStore(
            capacity=16,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token in range(1, 6):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=1,
                capture_tag=1.0,
            )
        for token in range(6, 11):
            store.update(
                torch.tensor([float(token), 2.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=2,
                capture_tag=1.0,
            )
        store.slow_importance[0] = 1000.0

        report = store.select_replay_window(
            n=2,
            current_token=50,
            candidate_pool=4,
            strategy="repair",
            candidate_bucket_ids=[1, 2],
        )

        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(
            report["candidate_window_policy"],
            "recent_bucket_round_robin_candidate_pool",
        )
        self.assertEqual(report["candidate_window_limit"], 4)
        self.assertEqual(report["selection_budget"]["candidate_window_entries"], 4)
        self.assertEqual(report["candidate_index_available_count"], 10)
        self.assertEqual(report["candidate_index_count"], 4)
        self.assertEqual(report["score_count"], 4)
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertTrue(set(report["selected_indices"]).issubset({3, 4, 8, 9}))
        self.assertNotIn(0, report["selected_indices"])

    def test_replay_query_collection_uses_capped_bucket_window(self) -> None:
        store = DualMemoryStore(
            capacity=16,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token in range(1, 11):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=1,
                input_pattern=torch.tensor([float(token), 0.0], dtype=torch.float32),
                capture_tag=1.0,
            )

        report = store.collect_replay_query_indices(
            candidate_bucket_ids=[1],
            max_queries=3,
            scope="unit_query_collection",
        )

        self.assertEqual(report["surface"], "bounded_replay_query_collection.v1")
        self.assertEqual(report["status"], "collected")
        self.assertEqual(report["candidate_window_policy"], "recent_bucket_round_robin_candidate_pool")
        self.assertEqual(report["candidate_window_limit"], 3)
        self.assertEqual(report["candidate_index_available_count"], 10)
        self.assertEqual(report["candidate_index_count"], 3)
        self.assertEqual(report["query_indices"], [9, 8, 7])
        self.assertEqual(report["query_count"], 3)
        self.assertEqual(report["score_count"], 0)
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertEqual(
            store.summary_stats()["last_replay_query_collection_report"][
                "query_indices"
            ],
            [9, 8, 7],
        )

    def test_query_memory_match_collection_uses_capped_bucket_window(self) -> None:
        store = DualMemoryStore(
            capacity=16,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token in range(1, 11):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=1,
                input_pattern=torch.tensor([float(token), 0.0], dtype=torch.float32),
                capture_tag=1.0,
            )

        report = store.collect_query_memory_match_indices(
            candidate_bucket_ids=[1],
            max_candidates=4,
            scope="unit_query_memory_match",
        )
        priority_scores = store.replay_scores_for_indices(
            report["match_indices"],
            current_token=20,
        )

        self.assertEqual(report["surface"], "bounded_query_memory_match_candidates.v1")
        self.assertEqual(report["status"], "collected")
        self.assertEqual(
            report["candidate_window_policy"],
            "recent_bucket_round_robin_candidate_pool",
        )
        self.assertEqual(report["candidate_window_limit"], 4)
        self.assertEqual(report["candidate_index_available_count"], 10)
        self.assertEqual(report["candidate_index_count"], 4)
        self.assertEqual(report["match_indices"], [9, 8, 7, 6])
        self.assertEqual(report["score_count"], 0)
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["global_candidate_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertEqual(sorted(priority_scores), [6, 7, 8, 9])
        self.assertEqual(
            store.summary_stats()["last_query_memory_match_report"]["match_indices"],
            [9, 8, 7, 6],
        )

    def test_unscoped_replay_selection_requires_bucket_scope(self) -> None:
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

        self.assertEqual(report["candidate_scope"], "bucket_index_scope_required")
        self.assertEqual(
            report["candidate_window_policy"],
            "bucket_scope_required_no_global_fallback",
        )
        self.assertFalse(report["global_score_scan"])
        self.assertFalse(report["runs_live_tick"])
        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["selected_indices"], [])
        self.assertEqual(report["candidate_index_available_count"], 0)
        self.assertEqual(
            report["fallback_reason"],
            "candidate_bucket_scope_required_for_replay_window",
        )
        self.assertFalse(hasattr(store, "sample_replay_indices"))

    def test_unscoped_random_replay_selection_requires_bucket_scope(self) -> None:
        store = DualMemoryStore(
            capacity=4,
            ema_alpha=0.1,
            slow_mean_decay=1.0,
            capture_tag_decay=1.0,
        )
        for token in range(1, 4):
            store.update(
                torch.tensor([float(token), 1.0], dtype=torch.float32),
                token_count=token,
                importance=1.0,
                bucket_id=token,
                capture_tag=1.0,
            )

        report = store.select_replay_window(
            n=1,
            current_token=8,
            candidate_pool=2,
            strategy="random",
        )

        self.assertEqual(report["candidate_scope"], "bucket_index_scope_required")
        self.assertEqual(
            report["candidate_window_policy"],
            "bucket_scope_required_no_global_fallback",
        )
        self.assertEqual(report["selected_indices"], [])
        self.assertEqual(report["candidate_index_available_count"], 0)
        self.assertEqual(report["candidate_index_count"], 0)
        self.assertFalse(report["global_candidate_scan"])
        self.assertEqual(
            report["fallback_reason"],
            "candidate_bucket_scope_required_for_replay_window",
        )

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
        self.assertFalse(report["raw_text_payload_loaded"])
        self.assertFalse(report["language_reasoning"])
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

    def test_hf_recall_evaluation_reports_bounded_anchor_window(self) -> None:
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
            raw_window="hf-like anchor replay trace",
            text="hf-like anchor replay trace",
            capture_tag=1.0,
        )
        trainer.model.memory_store.slow_local_prp[0] = 1.0
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=1,
            strength=2.0,
        )

        queries, query_collection = _collect_anchor_replay_queries(
            trainer,
            max_queries=4,
        )
        report = _bounded_replay_recall_evaluation(
            trainer,
            queries,
            max_candidates=4,
            scope="unit_hf_anchor_replay",
            query_collection_report=query_collection,
        )

        self.assertGreater(anchored, 0)
        self.assertEqual(
            query_collection["surface"],
            "bounded_replay_query_collection.v1",
        )
        self.assertEqual(
            query_collection["candidate_scope"],
            "bucket_indexed_candidate_window",
        )
        self.assertEqual(query_collection["anchor_bucket_source_total_count"], 1)
        self.assertEqual(
            query_collection["anchor_bucket_window_limit"],
            REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
        )
        self.assertEqual(query_collection["anchor_bucket_window_count"], 1)
        self.assertEqual(
            query_collection["anchor_bucket_window_policy"],
            REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_POLICY,
        )
        self.assertFalse(query_collection["anchor_source_full_scan"])
        self.assertEqual(
            query_collection["source_window"]["surface"],
            "bounded_replay_query_anchor_bucket_source_window.v1",
        )
        self.assertEqual(query_collection["candidate_window_limit"], 4)
        self.assertEqual(query_collection["query_count"], 1)
        self.assertEqual(report["surface"], "bounded_replay_window_hf_recall.v1")
        self.assertEqual(
            report["candidate_bucket_ids"],
            query_collection["candidate_bucket_ids"],
        )
        self.assertEqual(
            report["anchor_bucket_source_window"]["surface"],
            "bounded_replay_query_anchor_bucket_source_window.v1",
        )
        self.assertEqual(report["candidate_scope"], "bucket_indexed_candidate_window")
        self.assertEqual(report["query_count"], 1)
        self.assertEqual(
            report["query_collection"]["surface"],
            "bounded_replay_query_collection.v1",
        )
        self.assertTrue(report["gate"]["pass"])
        self.assertFalse(report["runs_live_tick"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(report["score_device"], "cpu")
        self.assertLess(report["mean_input_pattern_distance"], 1e-5)
        self.assertEqual(
            report["reports"][0]["candidate_scope"],
            "bucket_indexed_candidate_window",
        )

    def test_hf_replay_query_collection_caps_anchor_bucket_source_window(self) -> None:
        set_seed(9)
        cfg = MarulhoConfig(
            n_columns=32,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=64,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        trainer.memory_warm_started = True
        trainer.token_count = 100

        for bucket in range(24):
            pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
            pattern[bucket % cfg.input_dim] = 1.0
            routing_key = trainer.model.routing_key_from_pattern(pattern)
            assembly = trainer.model.competitive.assembly_from_input(
                pattern.to(trainer.model.device)
            ).detach().cpu()
            trainer.model.memory_store.update(
                assembly,
                importance=1.0,
                token_count=bucket + 1,
                bucket_id=bucket,
                input_pattern=pattern,
                routing_key=routing_key.detach().cpu(),
                capture_tag=1.0,
            )
            trainer.column_anchors[bucket] = {
                "prototype": trainer.model.competitive.prototypes[bucket]
                .detach()
                .clone(),
                "input_weights": trainer.model.competitive.input_weights[bucket]
                .detach()
                .clone(),
                "strength": 2.0,
                "captured_at_token": bucket + 1,
                "captured_source_index": bucket,
                "capture_sequence": bucket,
            }

        queries, query_collection = _collect_anchor_replay_queries(
            trainer,
            max_queries=4,
        )
        source_window = query_collection["source_window"]

        expected_anchor_buckets = list(range(23, 7, -1))
        self.assertEqual(len(queries), 4)
        self.assertEqual(
            source_window["surface"],
            "bounded_replay_query_anchor_bucket_source_window.v1",
        )
        self.assertEqual(source_window["anchor_bucket_source_total_count"], 24)
        self.assertEqual(
            source_window["anchor_bucket_window_limit"],
            REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
        )
        self.assertEqual(source_window["anchor_bucket_window_count"], 16)
        self.assertEqual(source_window["anchor_bucket_ids"], expected_anchor_buckets)
        self.assertTrue(source_window["truncated_source_count"])
        self.assertFalse(source_window["anchor_source_full_scan"])
        self.assertFalse(source_window["global_candidate_scan"])
        self.assertEqual(query_collection["candidate_bucket_ids"], expected_anchor_buckets)
        self.assertEqual(query_collection["candidate_bucket_count"], 16)
        self.assertEqual(query_collection["candidate_index_available_count"], 16)
        self.assertEqual(query_collection["candidate_window_limit"], 4)
        self.assertEqual(query_collection["candidate_index_count"], 4)
        self.assertEqual(query_collection["query_indices"], [23, 22, 21, 20])
        self.assertEqual(query_collection["query_count"], 4)
        self.assertFalse(query_collection["global_candidate_scan"])
        self.assertFalse(query_collection["runs_live_tick"])

    def test_reconstruction_guard_rolls_back_harmful_replay_cycle(self) -> None:
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
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[2:6] = 1.0
        pattern = pattern / pattern.sum()
        before_proto = trainer.model.competitive.prototypes.detach().clone()

        def _harmful_replay(mode: str, cycles: int = 1) -> int:
            self.assertEqual(mode, "deep")
            self.assertEqual(cycles, 1)
            trainer.model.competitive.prototypes[0] = torch.roll(
                trainer.model.competitive.prototypes[0],
                shifts=1,
                dims=0,
            )
            trainer._last_sleep_replay_selection_report = {
                "surface": "bounded_replay_window_selection.v1",
                "sleep_replay_applied_count": 1,
                "sleep_replay_mutates_runtime_state": True,
                "sleep_replay_applies_plasticity": True,
            }
            return 1

        with (
            patch.object(
                trainer,
                "run_sleep_maintenance",
                side_effect=_harmful_replay,
            ),
            patch(
                "marulho.training.memory_consolidation_runner.mean_reconstruction_error",
                side_effect=[0.10, 0.20],
            ),
        ):
            updates, report = _run_reconstruction_guarded_sleep_maintenance(
                trainer,
                [pattern],
                mode="deep",
                cycles=1,
                quality_scope="unit_reconstruction",
            )

        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertEqual(report["surface"], "reconstruction_guarded_replay_consolidation.v1")
        self.assertEqual(report["attempted_update_count"], 1)
        self.assertEqual(report["accepted_update_count"], 0)
        self.assertEqual(report["rejected_cycle_count"], 1)
        self.assertEqual(
            report["cycle_reports"][0]["sleep_replay_rollback_reason"],
            "task_a_reconstruction_regression",
        )
        self.assertFalse(report["cycle_reports"][0]["sleep_replay_mutates_runtime_state"])
        self.assertEqual(
            trainer._last_sleep_replay_selection_report["sleep_replay_applied_count"],
            0,
        )

    def test_reconstruction_guard_rejects_regression_even_when_no_updates_reported(self) -> None:
        set_seed(8)
        cfg = MarulhoConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[4:8] = 1.0
        pattern = pattern / pattern.sum()
        before_proto = trainer.model.competitive.prototypes.detach().clone()

        def _hidden_mutation(mode: str, cycles: int = 1) -> int:
            self.assertEqual(mode, "deep")
            self.assertEqual(cycles, 1)
            trainer.model.competitive.prototypes[1] = torch.roll(
                trainer.model.competitive.prototypes[1],
                shifts=1,
                dims=0,
            )
            trainer._last_sleep_replay_selection_report = {
                "surface": "bounded_replay_window_selection.v1",
                "sleep_replay_applied_count": 0,
                "sleep_replay_mutates_runtime_state": True,
                "sleep_replay_applies_plasticity": True,
            }
            return 0

        with (
            patch.object(
                trainer,
                "run_sleep_maintenance",
                side_effect=_hidden_mutation,
            ),
            patch(
                "marulho.training.memory_consolidation_runner.mean_reconstruction_error",
                side_effect=[0.10, 0.20],
            ),
        ):
            updates, report = _run_reconstruction_guarded_sleep_maintenance(
                trainer,
                [pattern],
                mode="deep",
                cycles=1,
                quality_scope="unit_reconstruction",
            )

        self.assertEqual(updates, 0)
        self.assertTrue(torch.allclose(before_proto, trainer.model.competitive.prototypes))
        self.assertEqual(report["attempted_update_count"], 0)
        self.assertEqual(report["accepted_update_count"], 0)
        self.assertEqual(report["rejected_cycle_count"], 1)
        self.assertEqual(
            report["cycle_reports"][0]["sleep_replay_rollback_reason"],
            "task_a_reconstruction_regression_no_updates_reported",
        )
        self.assertFalse(report["cycle_reports"][0]["sleep_replay_mutates_runtime_state"])

    def test_reconstruction_guard_skips_repeated_rejected_selection(self) -> None:
        set_seed(9)
        cfg = MarulhoConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[6:10] = 1.0
        pattern = pattern / pattern.sum()
        call_count = 0

        def _harmful_replay(mode: str, cycles: int = 1) -> int:
            nonlocal call_count
            call_count += 1
            self.assertEqual(mode, "deep")
            self.assertEqual(cycles, 1)
            trainer.model.competitive.prototypes[2] = torch.roll(
                trainer.model.competitive.prototypes[2],
                shifts=1,
                dims=0,
            )
            trainer._last_sleep_replay_selection_report = {
                "surface": "bounded_replay_window_selection.v1",
                "scope": "deep_sleep_slow_path",
                "strategy": "consolidation",
                "candidate_scope": "bucket_indexed_candidate_window",
                "candidate_bucket_ids": [2],
                "selected_indices": [0, 1],
                "selected_count": 2,
                "score_count": 2,
                "sleep_replay_applied_count": 1,
                "sleep_replay_mutates_runtime_state": True,
                "sleep_replay_applies_plasticity": True,
            }
            return 1

        with (
            patch.object(
                trainer,
                "run_sleep_maintenance",
                side_effect=_harmful_replay,
            ),
            patch(
                "marulho.training.memory_consolidation_runner.mean_reconstruction_error",
                side_effect=[0.10, 0.20],
            ),
        ):
            updates, report = _run_reconstruction_guarded_sleep_maintenance(
                trainer,
                [pattern],
                mode="deep",
                cycles=3,
                quality_scope="unit_reconstruction",
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(updates, 0)
        self.assertEqual(report["cycle_count"], 3)
        self.assertEqual(report["attempted_update_count"], 1)
        self.assertEqual(report["rejected_cycle_count"], 1)
        self.assertEqual(report["skipped_repeated_rejection_cycle_count"], 2)
        self.assertEqual(
            report["cadence_strategy"],
            "skip_repeated_rejected_selection",
        )
        self.assertEqual(
            report["cycle_reports"][1]["sleep_replay_rollback_reason"],
            "repeated_rejected_selection_skipped",
        )
        self.assertFalse(report["cycle_reports"][1]["sleep_replay_mutates_runtime_state"])

    def test_reconstruction_guard_selects_nonregressing_repair_strength(self) -> None:
        set_seed(10)
        cfg = MarulhoConfig(
            n_columns=6,
            column_latent_dim=12,
            bootstrap_tokens=0,
            memory_capacity=16,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
        pattern[8:12] = 1.0
        pattern = pattern / pattern.sum()
        before_proto = trainer.model.competitive.prototypes.detach().clone()
        rejected_proto = torch.roll(before_proto[0], shifts=1, dims=0)
        accepted_proto = torch.roll(before_proto[0], shifts=2, dims=0)
        trial_strengths: list[float] = []

        def _strength_replay(
            mode: str,
            cycles: int = 1,
            *,
            deep_replay_repair_strength: float | None = None,
        ) -> int:
            self.assertEqual(mode, "deep")
            self.assertEqual(cycles, 1)
            strength = float(deep_replay_repair_strength or 1.0)
            trial_strengths.append(strength)
            trainer.model.competitive.prototypes[0] = (
                rejected_proto if strength == 1.0 else accepted_proto
            )
            trainer._last_sleep_replay_selection_report = {
                "surface": "bounded_replay_window_selection.v1",
                "scope": "deep_sleep_slow_path",
                "strategy": "consolidation",
                "candidate_scope": "bucket_indexed_candidate_window",
                "candidate_bucket_ids": [0],
                "selected_indices": [0],
                "selected_count": 1,
                "score_count": 1,
                "sleep_replay_applied_count": 1,
                "sleep_replay_mutates_runtime_state": True,
                "sleep_replay_applies_plasticity": True,
                "sleep_replay_repair_strength": strength,
            }
            return 1

        with (
            patch.object(
                trainer,
                "run_sleep_maintenance",
                side_effect=_strength_replay,
            ),
            patch(
                "marulho.training.memory_consolidation_runner.mean_reconstruction_error",
                side_effect=[0.10, 0.20, 0.08],
            ),
        ):
            updates, report = _run_reconstruction_guarded_sleep_maintenance(
                trainer,
                [pattern],
                mode="deep",
                cycles=1,
                quality_scope="unit_reconstruction",
                repair_strength_schedule=[1.0, 0.25],
            )

        self.assertEqual(trial_strengths, [1.0, 0.25])
        self.assertEqual(updates, 1)
        self.assertTrue(
            torch.allclose(trainer.model.competitive.prototypes[0], accepted_proto)
        )
        self.assertFalse(
            torch.allclose(trainer.model.competitive.prototypes[0], rejected_proto)
        )
        self.assertEqual(report["repair_strength_strategy"], "target_reconstruction_strength_search")
        self.assertEqual(report["repair_strength_schedule"], [1.0, 0.25])
        self.assertEqual(report["repair_strength_trial_budget"], 2)
        self.assertEqual(
            report["repair_strength_trial_budget_policy"],
            "explicit_schedule_length",
        )
        self.assertEqual(report["attempted_update_count"], 2)
        self.assertEqual(report["accepted_update_count"], 1)
        self.assertEqual(report["rejected_attempted_update_count"], 1)
        cycle_report = report["cycle_reports"][0]
        self.assertTrue(cycle_report["sleep_replay_commit_accepted"])
        self.assertEqual(cycle_report["sleep_replay_selected_repair_strength"], 0.25)
        self.assertEqual(cycle_report["sleep_replay_strength_trial_count"], 2)
        self.assertEqual(cycle_report["sleep_replay_strength_trial_budget"], 2)
        self.assertEqual(
            cycle_report["sleep_replay_strength_trial_budget_policy"],
            "explicit_schedule_length",
        )
        self.assertEqual(cycle_report["sleep_replay_attempted_applied_count"], 2)
        self.assertEqual(cycle_report["sleep_replay_effective_applied_count"], 1)
        self.assertFalse(
            cycle_report["sleep_replay_strength_trial_reports"][0][
                "sleep_replay_strength_trial_accepted"
            ]
        )
        self.assertTrue(
            cycle_report["sleep_replay_strength_trial_reports"][1][
                "sleep_replay_strength_trial_accepted"
            ]
        )

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
        self.assertFalse(report["sleep_replay_text_payload_loaded"])
        self.assertFalse(report["sleep_replay_language_reasoning"])
        self.assertEqual(
            report["sleep_replay_text_payload_policy"],
            "sleep_replay_uses_tensor_payloads_only",
        )
        self.assertEqual(
            report["sleep_replay_local_trace_source"],
            "stored_input_pattern_or_routing_key",
        )
        self.assertTrue(report["sleep_replay_sfa_full_memory_sample_retired"])
        self.assertEqual(report["sleep_replay_sfa_correction_scope"], "not_run")
        self.assertEqual(report["sleep_replay_winner_source"], "bounded_route_candidates")
        self.assertFalse(report["sleep_replay_forced_stored_bucket_winner"])
        self.assertGreater(report["sleep_replay_candidate_column_union_count"], 0)
        self.assertGreater(report["sleep_replay_candidate_column_trial_count"], 0)
        self.assertGreaterEqual(
            report["sleep_replay_quality_before"],
            report["sleep_replay_quality_after"],
        )

    def test_deep_sleep_sfa_correction_samples_selected_replay_window(self) -> None:
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
            enable_abstraction_layer=True,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        patterns = []
        for offset in (2, 5, 8):
            pattern = torch.zeros(cfg.input_dim, dtype=torch.float32)
            pattern[offset : offset + 3] = 1.0
            pattern = pattern / pattern.sum()
            patterns.append(pattern)

        for index in range(3):
            for pattern in patterns:
                trainer.train_step(pattern, raw_window=f"anchored sfa replay {index}")
        trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=trainer.token_count,
            strength=2.0,
        )
        for idx in range(len(trainer.model.memory_store.slow_local_prp)):
            trainer.model.memory_store.slow_local_prp[idx] = 1.0

        captured: dict[str, list[int]] = {}
        original_sample = trainer.model.memory_store.sample_for_sfa_with_report

        def _sample_for_sfa(*args, **kwargs):
            captured["candidate_indices"] = [
                int(index)
                for index in kwargs.get("candidate_indices") or []
            ]
            return original_sample(*args, **kwargs)

        with patch.object(
            trainer.model.memory_store,
            "sample_for_sfa_with_report",
            side_effect=_sample_for_sfa,
        ):
            updates = trainer.run_sleep_maintenance(mode="deep", cycles=1)
        report = trainer._last_sleep_replay_selection_report

        self.assertGreater(anchored, 0)
        self.assertGreater(updates, 0)
        self.assertIn("candidate_indices", captured)
        self.assertTrue(captured["candidate_indices"])
        self.assertEqual(
            report["sleep_replay_sfa_correction_scope"],
            "selected_replay_window",
        )
        self.assertTrue(report["sleep_replay_sfa_full_memory_sample_retired"])
        self.assertEqual(
            report["sleep_replay_sfa_candidate_index_count"],
            len(set(captured["candidate_indices"])),
        )
        self.assertLessEqual(
            report["sleep_replay_sfa_sample_count"],
            report["sleep_replay_sfa_candidate_index_count"],
        )
        sample_report = report["sleep_replay_sfa_sample_report"]
        self.assertEqual(sample_report["surface"], "bounded_sfa_sample.v1")
        self.assertEqual(sample_report["scope"], "deep_sleep_sfa_correction")
        self.assertEqual(sample_report["candidate_scope"], "selected_replay_window")
        self.assertEqual(
            sample_report["candidate_index_count"],
            report["sleep_replay_sfa_candidate_index_count"],
        )
        self.assertFalse(sample_report["global_candidate_scan"])
        self.assertFalse(sample_report["runs_live_tick"])
        self.assertFalse(sample_report["language_reasoning"])

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

        with patch.object(
            trainer.model.competitive,
            "assembly_from_input",
            side_effect=AssertionError(
                "repair replay must not build dense input assemblies when stored routing keys exist"
            ),
        ):
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
        self.assertFalse(report["sleep_replay_text_payload_loaded"])
        self.assertFalse(report["sleep_replay_language_reasoning"])
        self.assertEqual(
            report["sleep_replay_winner_source"],
            "stored_replay_bucket_with_anchor_scope",
        )
        self.assertTrue(report["sleep_replay_unconditional_dense_input_assembly_retired"])
        self.assertEqual(report["sleep_replay_dense_input_assembly_fallback_count"], 0)
        self.assertEqual(report["sleep_replay_bounded_input_prepare_count"], 1)
        self.assertEqual(report["sleep_replay_stored_routing_key_count"], 1)

    def test_repair_sleep_missing_routing_key_uses_stored_assembly_projection(self) -> None:
        set_seed(11)
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
        pattern[1:5] = 1.0
        pattern = pattern / pattern.sum()
        assembly = trainer.model.competitive.assembly_from_input(
            pattern.to(trainer.model.device)
        ).detach().cpu()
        target_key = torch.nn.functional.normalize(
            torch.mv(trainer.model._W_assembly_project_t, assembly.to(trainer.model.device)),
            dim=0,
        )
        trainer.model.memory_store.update(
            assembly,
            importance=1.0,
            token_count=trainer.token_count,
            bucket_id=0,
            input_pattern=pattern,
            routing_key=None,
            raw_window="legacy repair memory trace",
            text="legacy repair memory trace",
            capture_tag=0.4,
        )
        anchored = trainer.capture_recent_memory_anchors(
            window_tokens=1,
            strength=2.0,
        )

        disturbed = torch.roll(target_key.detach().cpu(), shifts=1, dims=0)
        disturbed = torch.nn.functional.normalize(disturbed, dim=0).to(
            trainer.model.device
        )
        trainer.model.competitive.prototypes[0] = disturbed
        before_distance = float(
            torch.norm(
                trainer.model.competitive.prototypes[0] - target_key.to(trainer.model.device)
            ).item()
        )

        with patch.object(
            trainer.model.competitive,
            "assembly_from_input",
            side_effect=AssertionError(
                "repair replay must not rebuild dense input assemblies for missing routing keys"
            ),
        ):
            updates = trainer._sleep_replay("repair")

        after_distance = float(
            torch.norm(
                trainer.model.competitive.prototypes[0] - target_key.to(trainer.model.device)
            ).item()
        )

        self.assertGreater(anchored, 0)
        self.assertEqual(updates, 1)
        self.assertLess(after_distance, before_distance)
        report = trainer._last_sleep_replay_selection_report
        self.assertEqual(report["sleep_replay_commit_strategy"], "bounded_repair_reanchor")
        self.assertTrue(report["sleep_replay_unconditional_dense_input_assembly_retired"])
        self.assertEqual(report["sleep_replay_dense_input_assembly_fallback_count"], 0)
        self.assertEqual(report["sleep_replay_bounded_input_prepare_count"], 0)
        self.assertEqual(report["sleep_replay_stored_routing_key_count"], 0)
        self.assertEqual(report["sleep_replay_missing_routing_key_count"], 1)
        self.assertEqual(
            report["sleep_replay_stored_assembly_projection_fallback_count"],
            1,
        )
        self.assertEqual(
            report["sleep_replay_local_trace_prepare_policy"],
            "stored_routing_key_then_stored_assembly_projection_no_dense_fallback",
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
            self.assertGreater(
                trainer.capture_recent_memory_anchors(
                    window_tokens=trainer.token_count,
                    strength=3.0,
                ),
                0,
            )
            trainer.run_sleep_maintenance(mode="deep", cycles=2)

            for _ in range(18):
                for i, w in enumerate(task_b_windows):
                    trainer.train_step(task_b[i], raw_window=w)
            a_after_b = mean_reconstruction_error(trainer, task_a)

            trainer.tag_recent_memories(window_tokens=trainer.token_count, strength=3.0)
            self.assertGreater(
                trainer.capture_recent_memory_anchors(
                    window_tokens=trainer.token_count,
                    strength=3.0,
                ),
                0,
            )
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
