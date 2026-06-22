from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from marulho.training.checkpointing import _checkpoint_load_device, load_trainer_checkpoint, save_trainer_checkpoint


class CheckpointDevicePlacementTests(unittest.TestCase):
    def test_checkpoint_load_device_prefers_runtime_env(self) -> None:
        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            self.assertEqual(_checkpoint_load_device(), torch.device("cpu"))

    def test_checkpoint_restore_uses_runtime_device_not_hardcoded_cpu(self) -> None:
        captured: dict[str, object] = {}

        def fake_load(path: Path, *, map_location: object) -> dict[str, object]:
            captured["path"] = path
            captured["map_location"] = map_location
            raise RuntimeError("stop before trainer construction")

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            with patch("marulho.training.checkpointing.torch.load", side_effect=fake_load):
                with self.assertRaisesRegex(RuntimeError, "stop before trainer construction"):
                    load_trainer_checkpoint(Path("checkpoint.pt"))

        self.assertEqual(captured["path"], Path("checkpoint.pt"))
        self.assertEqual(captured["map_location"], torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_checkpoint_restore_selects_cuda_when_available(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_checkpoint_load_device().type, "cuda")

    def test_checkpoint_save_failure_preserves_existing_file(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint.pt"
            target.write_bytes(b"previous")
            trainer = SimpleNamespace(
                config=object(),
                encoder=SimpleNamespace(state_dict=lambda: {}),
                token_count=0,
                is_bootstrap=True,
                sleep_events=0,
                micro_sleep_events=0,
                deep_sleep_events=0,
                last_micro_sleep_token=0,
                last_deep_sleep_token=0,
                current_window_min_drift=0.0,
                previous_window_min_drift=None,
                recent_drifts=[],
                current_rolling_drift_floor=None,
                previous_rolling_drift_floor=None,
                last_floor_check_token=0,
                memory_warm_started=False,
                last_winner=None,
                pending_emergency_deep_sleep=False,
                developmental_stage=0,
                _stage2_bootstrap_budget=0,
                _stage2_bootstrap_used_visual=0,
                _stage2_bootstrap_used_audio=0,
                column_anchors={},
            )

            with patch("marulho.training.checkpointing.asdict", return_value={}):
                with patch("marulho.training.checkpointing._model_snapshot", return_value={}):
                    with patch("marulho.training.checkpointing.torch.save", side_effect=RuntimeError("interrupted")):
                        with self.assertRaisesRegex(RuntimeError, "interrupted"):
                            save_trainer_checkpoint(target, trainer)

            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_checkpoint_roundtrip_preserves_predictive_failure_streak(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.model.predictive.prediction_failure_streak[:] = torch.arange(
                8,
                dtype=torch.int32,
                device=trainer.model.device,
            )
            checkpoint = save_trainer_checkpoint(Path(tmpdir) / "predictive.pt", trainer)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            self.assertTrue(
                torch.equal(
                    restored.model.predictive.prediction_failure_streak.cpu(),
                    torch.arange(8, dtype=torch.int32),
                )
            )

    def test_checkpoint_roundtrip_preserves_column_anchor_recency_metadata(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.column_anchors[3] = {
                "prototype": trainer.model.competitive.prototypes[3]
                .detach()
                .clone(),
                "input_weights": trainer.model.competitive.input_weights[3]
                .detach()
                .clone(),
                "strength": 2.5,
                "captured_at_token": 123,
                "captured_source_index": 7,
                "capture_sequence": 4,
            }
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "anchor-recency.pt",
                trainer,
            )

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            self.assertIn(3, restored.column_anchors)
            anchor = restored.column_anchors[3]
            self.assertEqual(anchor["captured_at_token"], 123)
            self.assertEqual(anchor["captured_source_index"], 7)
            self.assertEqual(anchor["capture_sequence"], 4)
            self.assertEqual(anchor["strength"], 2.5)
            self.assertEqual(str(anchor["prototype"].device), str(restored.model.device))

    def test_checkpoint_roundtrip_preserves_sleep_replay_selection_report(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            report = {
                "surface": "bounded_replay_window_selection.v1",
                "status": "selected",
                "scope": "deep_sleep_slow_path",
                "candidate_scope": "bucket_indexed_candidate_window",
                "bounded_by_bucket_index": True,
                "global_score_scan": False,
                "score_count": 2,
                "selected_count": 1,
                "selected_indices": [3],
                "sleep_replay_applied_count": 1,
                "sleep_replay_mutates_runtime_state": True,
                "sleep_replay_applies_plasticity": True,
                "sleep_replay_commit_strategy": "bounded_reconstruction_gated_candidate_repair",
                "sleep_replay_winner_source": "bounded_route_candidates",
                "sleep_replay_text_payload_loaded": False,
                "sleep_replay_language_reasoning": False,
                "sleep_replay_text_payload_policy": "sleep_replay_uses_tensor_payloads_only",
                "sleep_replay_local_trace_source": "stored_input_pattern_or_routing_key",
                "sleep_replay_sfa_correction_scope": "selected_replay_window",
                "sleep_replay_sfa_full_memory_sample_retired": True,
                "sleep_replay_sfa_candidate_index_count": 1,
                "sleep_replay_sfa_sample_count": 1,
                "sleep_replay_sfa_applied": False,
                "sleep_replay_quality_before": 0.25,
                "sleep_replay_quality_after": 0.10,
                "sleep_replay_candidate_column_union_count": 4,
                "runs_live_tick": False,
            }
            trainer._last_sleep_replay_selection_report = dict(report)
            trainer.model.memory_store.last_replay_selection_report = dict(report)
            sfa_sample_report = {
                "surface": "bounded_sfa_sample.v1",
                "status": "selected",
                "scope": "deep_sleep_sfa_correction",
                "candidate_scope": "selected_replay_window",
                "candidate_index_count": 2,
                "candidate_indices": [3, 5],
                "sample_indices": [5],
                "sample_count": 1,
                "global_candidate_scan": False,
                "runs_live_tick": False,
                "language_reasoning": False,
                "archival_storage_device": "cpu",
            }
            trainer.model.memory_store.last_sfa_sample_report = dict(sfa_sample_report)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "sleep-replay-selection.pt",
                trainer,
            )

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            restored_report = restored._last_sleep_replay_selection_report
            self.assertEqual(restored_report["surface"], "bounded_replay_window_selection.v1")
            self.assertEqual(restored_report["candidate_scope"], "bucket_indexed_candidate_window")
            self.assertEqual(restored_report["score_count"], 2)
            self.assertEqual(restored_report["selected_indices"], [3])
            self.assertTrue(restored_report["sleep_replay_mutates_runtime_state"])
            self.assertEqual(
                restored_report["sleep_replay_commit_strategy"],
                "bounded_reconstruction_gated_candidate_repair",
            )
            self.assertEqual(
                restored_report["sleep_replay_winner_source"],
                "bounded_route_candidates",
            )
            self.assertFalse(restored_report["sleep_replay_text_payload_loaded"])
            self.assertFalse(restored_report["sleep_replay_language_reasoning"])
            self.assertEqual(
                restored_report["sleep_replay_text_payload_policy"],
                "sleep_replay_uses_tensor_payloads_only",
            )
            self.assertEqual(
                restored_report["sleep_replay_sfa_correction_scope"],
                "selected_replay_window",
            )
            self.assertTrue(restored_report["sleep_replay_sfa_full_memory_sample_retired"])
            self.assertEqual(restored_report["sleep_replay_quality_after"], 0.10)
            self.assertEqual(
                restored_report["sleep_replay_candidate_column_union_count"],
                4,
            )
            self.assertEqual(
                restored.model.memory_store.last_replay_selection_report[
                    "selected_indices"
                ],
                [3],
            )
            self.assertEqual(
                restored.model.memory_store.last_sfa_sample_report["surface"],
                "bounded_sfa_sample.v1",
            )
            self.assertEqual(
                restored.model.memory_store.last_sfa_sample_report["sample_indices"],
                [5],
            )
            self.assertFalse(
                restored.model.memory_store.last_sfa_sample_report[
                    "global_candidate_scan"
                ]
            )

    def test_checkpoint_roundtrip_preserves_replay_window_recall_report(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            report = {
                "surface": "bounded_replay_window_recall.v1",
                "status": "recalled",
                "scope": "replay_recall_slow_path",
                "candidate_scope": "bucket_indexed_candidate_window",
                "selected_indices": [2, 5],
                "selected_count": 2,
                "routing_key_count": 2,
                "input_pattern_count": 2,
                "best_distance": 0.001,
                "best_input_distance": 0.0,
                "runs_live_tick": False,
                "mutates_runtime_state": False,
                "applies_plasticity": False,
            }
            trainer.model.memory_store.last_replay_recall_report = dict(report)
            query_report = {
                "surface": "bounded_replay_query_collection.v1",
                "status": "collected",
                "scope": "hf_task_a_anchor_query_collection",
                "candidate_scope": "bucket_indexed_candidate_window",
                "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
                "candidate_window_limit": 4,
                "candidate_index_available_count": 9,
                "candidate_index_count": 4,
                "query_indices": [8, 7, 6],
                "query_count": 3,
                "runs_live_tick": False,
                "mutates_runtime_state": False,
            }
            trainer.model.memory_store.last_replay_query_collection_report = dict(
                query_report
            )
            query_match_report = {
                "surface": "bounded_query_memory_match_candidates.v1",
                "status": "collected",
                "scope": "query_runner_memory_matches",
                "candidate_scope": "bucket_indexed_candidate_window",
                "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
                "candidate_window_limit": 5,
                "candidate_index_available_count": 12,
                "candidate_index_count": 5,
                "match_indices": [9, 8, 7, 6, 5],
                "score_count": 0,
                "runs_live_tick": False,
                "mutates_runtime_state": False,
            }
            trainer.model.memory_store.last_query_memory_match_report = dict(
                query_match_report
            )
            bank_memory_report = {
                "surface": "bounded_source_bank_memory_match.v1",
                "status": "matched",
                "scope": "source_bank_semantic_recall_slow_path",
                "candidate_scope": "source_bank_merged_probe_memory_recall_window",
                "candidate_window_policy": "merged_probe_bucket_indexed_candidate_window",
                "probe_count": 2,
                "scored_probe_count": 2,
                "candidate_index_count": 10,
                "unique_candidate_index_count": 10,
                "merged_probe_candidate_window": True,
                "per_probe_query_match_call_count": 0,
                "retired_per_probe_query_match_call_count": 2,
                "match_indices": [9, 8],
                "raw_text_payload_count": 2,
                "raw_text_payload_cache_hits": 0,
                "raw_text_payload_policy": "returned_merged_probe_matches_only",
                "runs_live_tick": False,
                "language_reasoning": False,
                "global_candidate_scan": False,
                "global_score_scan": False,
            }
            trainer.model.memory_store.last_bank_memory_match_report = dict(
                bank_memory_report
            )
            runtime_concept_lookup_report = {
                "surface": "bounded_runtime_concept_memory_lookup.v1",
                "status": "matched",
                "scope": "cadenced_runtime_concept_observation",
                "candidate_scope": "train_step_memory_index_evidence",
                "candidate_window_policy": "explicit_train_step_memory_indices_only",
                "candidate_window_limit": 4,
                "candidate_index_count": 2,
                "unique_candidate_index_count": 1,
                "match_indices": [4, 4],
                "raw_text_payload_count": 1,
                "raw_text_payload_cache_hits": 1,
                "runs_live_tick": True,
                "runs_every_token": False,
                "language_reasoning": False,
                "global_candidate_scan": False,
                "global_score_scan": False,
            }
            trainer.model.memory_store.last_runtime_concept_memory_lookup_report = dict(
                runtime_concept_lookup_report
            )
            awake_ripple_report = {
                "surface": "bounded_awake_ripple_tag.v1",
                "status": "tagged",
                "scope": "awake_ripple_tagging_cadenced_path",
                "candidate_scope": "awake_bucket_index_candidate_window",
                "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
                "candidate_window_limit": 3,
                "candidate_index_available_count": 10,
                "candidate_index_count": 3,
                "candidate_indices": [9, 8, 7],
                "tagged_count": 3,
                "runs_live_tick": True,
                "runs_every_token": False,
                "global_candidate_scan": False,
                "diagnostic_global_candidate_scan": False,
            }
            trainer.model.memory_store.last_awake_ripple_tag_report = dict(
                awake_ripple_report
            )
            recent_tag_report = {
                "surface": "bounded_recent_memory_tag.v1",
                "status": "tagged",
                "scope": "recent_memory_tagging_slow_path",
                "candidate_scope": "recent_entry_index_window",
                "candidate_window_policy": "recent_entry_index_reverse_window",
                "candidate_window_limit": 3,
                "candidate_index_count": 3,
                "candidate_indices": [9, 8, 7],
                "tagged_count": 3,
                "runs_live_tick": False,
                "mutates_runtime_state": True,
            }
            trainer.model.memory_store.last_recent_memory_tag_report = dict(
                recent_tag_report
            )
            anchor_capture_report = {
                "surface": "bounded_recent_anchor_capture.v1",
                "status": "captured",
                "scope": "recent_anchor_capture_slow_path",
                "candidate_scope": "bucketed_recent_entry_index_window",
                "candidate_window_policy": "recent_entry_index_reverse_window",
                "candidate_window_limit": 3,
                "candidate_index_count": 3,
                "candidate_indices": [9, 8, 7],
                "captured_anchor_count": 3,
                "candidate_bucket_ids": [7, 8, 9],
                "runs_live_tick": False,
                "mutates_runtime_state": True,
            }
            trainer.model.memory_store.last_anchor_capture_report = dict(
                anchor_capture_report
            )
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "replay-window-recall.pt",
                trainer,
            )

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            restored_report = restored.model.memory_store.last_replay_recall_report
            self.assertEqual(restored_report["surface"], "bounded_replay_window_recall.v1")
            self.assertEqual(restored_report["candidate_scope"], "bucket_indexed_candidate_window")
            self.assertEqual(restored_report["selected_indices"], [2, 5])
            self.assertEqual(restored_report["routing_key_count"], 2)
            self.assertEqual(restored_report["input_pattern_count"], 2)
            self.assertEqual(restored_report["best_input_distance"], 0.0)
            self.assertFalse(restored_report["runs_live_tick"])
            self.assertFalse(restored_report["mutates_runtime_state"])
            restored_query_report = (
                restored.model.memory_store.last_replay_query_collection_report
            )
            self.assertEqual(
                restored_query_report["surface"],
                "bounded_replay_query_collection.v1",
            )
            self.assertEqual(
                restored_query_report["candidate_scope"],
                "bucket_indexed_candidate_window",
            )
            self.assertEqual(restored_query_report["query_indices"], [8, 7, 6])
            self.assertFalse(restored_query_report["runs_live_tick"])
            restored_query_match_report = (
                restored.model.memory_store.last_query_memory_match_report
            )
            self.assertEqual(
                restored_query_match_report["surface"],
                "bounded_query_memory_match_candidates.v1",
            )
            self.assertEqual(
                restored_query_match_report["candidate_scope"],
                "bucket_indexed_candidate_window",
            )
            self.assertEqual(
                restored_query_match_report["match_indices"],
                [9, 8, 7, 6, 5],
            )
            self.assertFalse(restored_query_match_report["runs_live_tick"])
            restored_bank_report = (
                restored.model.memory_store.last_bank_memory_match_report
            )
            self.assertEqual(
                restored_bank_report["surface"],
                "bounded_source_bank_memory_match.v1",
            )
            self.assertEqual(
                restored_bank_report["candidate_scope"],
                "source_bank_merged_probe_memory_recall_window",
            )
            self.assertEqual(
                restored_bank_report["candidate_window_policy"],
                "merged_probe_bucket_indexed_candidate_window",
            )
            self.assertTrue(restored_bank_report["merged_probe_candidate_window"])
            self.assertEqual(restored_bank_report["per_probe_query_match_call_count"], 0)
            self.assertEqual(restored_bank_report["match_indices"], [9, 8])
            self.assertEqual(restored_bank_report["raw_text_payload_cache_hits"], 0)
            self.assertFalse(restored_bank_report["runs_live_tick"])
            self.assertFalse(restored_bank_report["language_reasoning"])
            restored_runtime_concept_lookup_report = (
                restored.model.memory_store.last_runtime_concept_memory_lookup_report
            )
            self.assertEqual(
                restored_runtime_concept_lookup_report["surface"],
                "bounded_runtime_concept_memory_lookup.v1",
            )
            self.assertEqual(
                restored_runtime_concept_lookup_report["candidate_scope"],
                "train_step_memory_index_evidence",
            )
            self.assertEqual(
                restored_runtime_concept_lookup_report["match_indices"],
                [4, 4],
            )
            self.assertEqual(
                restored_runtime_concept_lookup_report["raw_text_payload_cache_hits"],
                1,
            )
            self.assertFalse(
                restored_runtime_concept_lookup_report["runs_every_token"]
            )
            self.assertFalse(
                restored_runtime_concept_lookup_report["language_reasoning"]
            )
            restored_awake_report = (
                restored.model.memory_store.last_awake_ripple_tag_report
            )
            self.assertEqual(
                restored_awake_report["surface"],
                "bounded_awake_ripple_tag.v1",
            )
            self.assertEqual(
                restored_awake_report["candidate_scope"],
                "awake_bucket_index_candidate_window",
            )
            self.assertEqual(restored_awake_report["candidate_indices"], [9, 8, 7])
            self.assertFalse(restored_awake_report["runs_every_token"])
            self.assertFalse(restored_awake_report["global_candidate_scan"])
            restored_tag_report = (
                restored.model.memory_store.last_recent_memory_tag_report
            )
            self.assertEqual(
                restored_tag_report["surface"],
                "bounded_recent_memory_tag.v1",
            )
            self.assertEqual(restored_tag_report["candidate_indices"], [9, 8, 7])
            self.assertFalse(restored_tag_report["runs_live_tick"])
            restored_anchor_report = (
                restored.model.memory_store.last_anchor_capture_report
            )
            self.assertEqual(
                restored_anchor_report["surface"],
                "bounded_recent_anchor_capture.v1",
            )
            self.assertEqual(restored_anchor_report["candidate_indices"], [9, 8, 7])
            self.assertFalse(restored_anchor_report["runs_live_tick"])

    def test_checkpoint_roundtrip_preserves_column_metabolism_state(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.model.column_metabolism.estimated_cost[:] = torch.linspace(
                0.0,
                0.7,
                steps=8,
                device=trainer.model.device,
            )
            trainer.model.column_metabolism.memory_pressure[:] = torch.linspace(
                0.8,
                0.1,
                steps=8,
                device=trainer.model.device,
            )
            trainer.model.column_metabolism.usefulness[:] = torch.linspace(
                0.2,
                0.9,
                steps=8,
                device=trainer.model.device,
            )
            trainer.model.column_metabolism.last_memory_pressure_source = (
                "unit_test_cached_pressure"
            )
            trainer.model.column_metabolism.last_usefulness_source = (
                "unit_test_cached_usefulness"
            )
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "column_metabolism.pt",
                trainer,
            )

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            self.assertTrue(
                torch.allclose(
                    restored.model.column_metabolism.estimated_cost.cpu(),
                    torch.linspace(0.0, 0.7, steps=8),
                )
            )
            self.assertTrue(
                torch.allclose(
                    restored.model.column_metabolism.memory_pressure.cpu(),
                    torch.linspace(0.8, 0.1, steps=8),
                )
            )
            self.assertTrue(
                torch.allclose(
                    restored.model.column_metabolism.usefulness.cpu(),
                    torch.linspace(0.2, 0.9, steps=8),
                )
            )
            self.assertEqual(
                restored.model.column_metabolism.last_memory_pressure_source,
                "unit_test_cached_pressure",
            )
            self.assertEqual(
                restored.model.column_metabolism.last_usefulness_source,
                "unit_test_cached_usefulness",
            )

    def test_checkpoint_roundtrip_preserves_column_structural_review_queue(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.model.predictive.prediction_error[3] = 0.9
            trainer.model.predictive.confidence[3] = 0.2
            trainer.model.predictive.prediction_failure_streak[3] = 5
            trainer.model.column_structural_review_queue.record_candidates(
                torch.tensor([3], device=trainer.model.device),
                token_count=12,
                mode="awake_mask_tick",
                prediction_error=trainer.model.predictive.prediction_error,
                confidence=trainer.model.predictive.confidence,
                prediction_failure_streak=(
                    trainer.model.predictive.prediction_failure_streak
                ),
                estimated_cost=trainer.model.column_metabolism.estimated_cost,
                memory_pressure=trainer.model.column_metabolism.memory_pressure,
                usefulness=trainer.model.column_metabolism.usefulness,
                wake_reason="unit_awake",
                sleep_reason=None,
            )
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "structural-review.pt",
                trainer,
            )

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            report = restored.model.column_structural_review_queue.report()
            self.assertEqual(report["pending_count"], 1)
            self.assertEqual(report["growth_ticket_count"], 1)
            self.assertEqual(report["last_evaluated_column_count"], 1)
            self.assertFalse(report["runs_all_columns"])
            self.assertTrue(report["checkpoint_backed"])
            self.assertEqual(report["tickets_sample"][0]["column_id"], 3)
            self.assertEqual(len(report["checkpoint_baseline"]["queue_state_hash"]), 64)
            self.assertEqual(
                len(report["tickets_sample"][0]["candidate_evidence_hash"]),
                64,
            )
            self.assertFalse(report["no_mutation_proof"]["mutates_runtime_state"])

    def test_legacy_checkpoint_migrates_retired_slow_memory_archive_cadence(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
                slow_memory_archive_interval_tokens=8,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(Path(tmpdir) / "legacy.pt", trainer)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["cuda_graph_host_truth_sync_interval_tokens"] = 8
            payload["config"]["cuda_graph_native_burst_tokens"] = 16
            payload["config"]["cuda_graph_sequence_executor"] = "native_repeated_child_graph"
            payload["config"]["cuda_graph_sequence_loop_tokens"] = 8
            payload["metadata"].pop("hot_path_config_defaults_revision", None)
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertEqual(restored.config.slow_memory_archive_interval_tokens, 256)
            self.assertEqual(restored.config.cuda_graph_host_truth_sync_interval_tokens, 32)
            self.assertEqual(restored.config.cuda_graph_sequence_executor, "conditional_while")
            self.assertEqual(restored.config.cuda_graph_sequence_loop_tokens, 16)
            self.assertEqual(restored.config.cuda_graph_native_burst_tokens, 8)
            self.assertEqual(
                metadata["config_migrations"][-1]["reason"],
                "retired_host_truth_sync_interval_cadence",
            )
            self.assertEqual(
                [item["reason"] for item in metadata["config_migrations"][-5:]],
                [
                    "retired_native_burst_capacity_prototype",
                    "retired_sequence_loop_capacity_prototype",
                    "retired_sequence_executor_selector",
                    "retired_hot_path_memory_archive_cadence",
                    "retired_host_truth_sync_interval_cadence",
                ],
            )

    def test_legacy_checkpoint_migrates_retired_host_truth_interval(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
                cuda_graph_host_truth_sync_interval_tokens=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(Path(tmpdir) / "legacy_truth.pt", trainer)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["cuda_graph_sequence_executor"] = "native_repeated_child_graph"
            payload["config"].pop("cuda_graph_sequence_loop_tokens", None)
            payload["metadata"].pop("hot_path_config_defaults_revision", None)
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertEqual(restored.config.cuda_graph_host_truth_sync_interval_tokens, 32)
            self.assertEqual(restored.config.cuda_graph_sequence_executor, "conditional_while")
            self.assertEqual(
                [item["reason"] for item in metadata["config_migrations"][-2:]],
                [
                    "retired_sequence_executor_selector",
                    "retired_host_truth_sync_interval_cadence",
                ],
            )

    def test_legacy_checkpoint_migrates_retired_predictive_transition_modes(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        for mode in ("compiled", "fused_eager", "legacy"):
            with self.subTest(mode=mode), TemporaryDirectory() as tmpdir:
                cfg = MarulhoConfig(
                    n_columns=8,
                    column_latent_dim=4,
                    bootstrap_tokens=0,
                    memory_capacity=16,
                )
                trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
                checkpoint = save_trainer_checkpoint(
                    Path(tmpdir) / f"{mode}_predictive.pt",
                    trainer,
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                payload["config"]["predictive_dense_transition_mode"] = mode
                payload["metadata"].pop("hot_path_config_defaults_revision", None)
                torch.save(payload, checkpoint)

                with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                    restored, metadata = load_trainer_checkpoint(checkpoint)

                self.assertEqual(
                    restored.config.predictive_dense_transition_mode,
                    "inplace_triton",
                )
                self.assertIn(
                    {
                        "field": "predictive_dense_transition_mode",
                        "from": mode,
                        "to": "inplace_triton",
                        "reason": "promoted_inplace_triton_scheduler_boundary",
                    },
                    metadata["config_migrations"],
                )

    def test_revision_stamped_checkpoint_retired_predictive_transition_migrates(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "retired-predictive-transition.pt",
                trainer,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["predictive_dense_transition_mode"] = "fused_eager"
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertEqual(
                restored.config.predictive_dense_transition_mode,
                "inplace_triton",
            )
            self.assertIn(
                {
                    "field": "predictive_dense_transition_mode",
                    "from": "fused_eager",
                    "to": "inplace_triton",
                    "reason": "promoted_inplace_triton_scheduler_boundary",
                },
                metadata["config_migrations"],
            )

    def test_legacy_checkpoint_route_vote_default_promotes_cuda_graph_text(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        for old_value in ("missing", "tensor", "fused_triton_text"):
            with self.subTest(old_value=old_value), TemporaryDirectory() as tmpdir:
                cfg = MarulhoConfig(
                    n_columns=8,
                    column_latent_dim=4,
                    bootstrap_tokens=0,
                    memory_capacity=16,
                )
                trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
                checkpoint = save_trainer_checkpoint(
                    Path(tmpdir) / f"{old_value}_route_vote.pt",
                    trainer,
                )
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if old_value == "missing":
                    payload["config"].pop("predictive_route_vote_mode", None)
                else:
                    payload["config"]["predictive_route_vote_mode"] = old_value
                payload["metadata"].pop("hot_path_config_defaults_revision", None)
                torch.save(payload, checkpoint)

                with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                    restored, metadata = load_trainer_checkpoint(checkpoint)

                self.assertEqual(
                    restored.config.predictive_route_vote_mode,
                    "cuda_graph_text",
                )
                self.assertIn(
                    {
                        "field": "predictive_route_vote_mode",
                        "from": old_value,
                        "to": "cuda_graph_text",
                        "reason": (
                            "missing_route_vote_mode_promoted"
                            if old_value == "missing"
                            else "retired_route_vote_mode_selector"
                        ),
                    },
                    metadata["config_migrations"],
                )

    def test_revision_stamped_checkpoint_retired_route_vote_selector_migrates(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "retired-route-vote-selector.pt",
                trainer,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["predictive_route_vote_mode"] = "fused_triton_text"
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertEqual(
                restored.config.predictive_route_vote_mode,
                "cuda_graph_text",
            )
            self.assertIn(
                {
                    "field": "predictive_route_vote_mode",
                    "from": "fused_triton_text",
                    "to": "cuda_graph_text",
                    "reason": "retired_route_vote_mode_selector",
                },
                metadata["config_migrations"],
            )

    def test_revision_stamped_checkpoint_preserves_explicit_archive_cadence(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
                slow_memory_archive_interval_tokens=64,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(Path(tmpdir) / "current.pt", trainer)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertEqual(restored.config.slow_memory_archive_interval_tokens, 64)
            self.assertNotIn("config_migrations", metadata)

    def test_checkpoint_loader_drops_retired_shard_merge_switch(self) -> None:
        from dataclasses import fields
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "retired-merge-shards.pt",
                trainer,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["merge_torch_routing_shards"] = False
            payload["config"]["routing_index_mode"] = "faiss_hnsw"
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertNotIn(
                "merge_torch_routing_shards",
                {field.name for field in fields(restored.config)},
            )
            self.assertNotIn(
                "routing_index_mode",
                {field.name for field in fields(restored.config)},
            )
            self.assertIn(
                {
                    "field": "merge_torch_routing_shards",
                    "from": False,
                    "to": "merged_torch_route_cache_required",
                    "reason": "retired_non_promoted_sharded_route_cache_switch",
                },
                metadata["config_migrations"],
            )
            self.assertIn(
                {
                    "field": "routing_index_mode",
                    "from": "faiss_hnsw",
                    "to": "torch_topk_fixed_retrieval_surface",
                    "reason": "retired_routing_backend_config_surface",
                },
                metadata["config_migrations"],
            )

    def test_checkpoint_loader_drops_retired_route_candidate_bank_size_selector(self) -> None:
        from dataclasses import fields
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=16,
                column_latent_dim=4,
                bootstrap_tokens=0,
                k_routing=4,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "retired-route-bank-size.pt",
                trainer,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["config"]["route_candidate_bank_size"] = 8
            torch.save(payload, checkpoint)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, metadata = load_trainer_checkpoint(checkpoint)

            self.assertNotIn(
                "route_candidate_bank_size",
                {field.name for field in fields(restored.config)},
            )
            self.assertIn(
                {
                    "field": "route_candidate_bank_size",
                    "from": 8,
                    "to": "k_routing_promoted_route_bank",
                    "reason": "retired_route_candidate_bank_size_selector",
                },
                metadata["config_migrations"],
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_checkpoint_cuda_graph_capture_happens_after_state_restore(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=32,
                column_latent_dim=8,
                bootstrap_tokens=0,
                k_routing=5,
                memory_capacity=16,
                predictive_dense_transition_mode="inplace_triton",
                predictive_route_vote_mode="cuda_graph_text",
                plasticity_mode="lite",
                input_weight_blend=0.0,
                enable_context_layer=False,
                enable_binding_layer=False,
                enable_abstraction_layer=False,
                device="cuda",
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "cuda-graph.pt",
                trainer,
            )

            restored, _metadata = load_trainer_checkpoint(checkpoint)
            before = restored.column_transition_runtime_report()
            restored.train_step(
                torch.rand(cfg.input_dim, device="cuda"),
                raw_window="checkpoint graph activation",
                allow_sleep_maintenance=False,
            )
            after = restored.column_transition_runtime_report()

            self.assertEqual(before["route_vote_resolved_mode"], "cuda_graph_text")
            self.assertTrue(before["cuda_graph_route_transition"]["active"])
            self.assertTrue(before["cuda_graph_route_transition"]["capture_succeeded"])
            self.assertEqual(after["route_vote_execution_count"], 1)
            self.assertEqual(after["last_selection_mode"], "fused_route_vote_cuda")
            self.assertEqual(
                after["route_candidate_bank"]["last_reason"],
                "route_candidate_bank_seeded_from_exact_route",
            )
            self.assertEqual(after["route_candidate_bank"]["seed_count"], 1)
            self.assertEqual(after["route_candidate_bank"]["graph_bypass_count"], 1)
            self.assertTrue(after["route_vote_scoring"]["route_rows_run_all_columns"])
            self.assertEqual(
                after["route_vote_scoring"]["route_scoring_unbounded_reason"],
                "route_candidate_bank_not_ready_exact_seed",
            )
            self.assertEqual(
                after["cuda_graph_route_transition"]["replay_count"],
                0,
            )
            self.assertEqual(
                after["cuda_graph_route_transition"]["pre_route_replay_count"],
                0,
            )
            self.assertEqual(
                after["cuda_graph_route_transition"]["failure_count"],
                0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_checkpoint_restores_route_candidate_bank_before_first_tick(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=32,
                column_latent_dim=8,
                bootstrap_tokens=0,
                k_routing=5,
                memory_capacity=16,
                predictive_dense_transition_mode="inplace_triton",
                predictive_route_vote_mode="cuda_graph_text",
                plasticity_mode="lite",
                input_weight_blend=0.0,
                dead_column_steps=1000,
                candidate_homeostasis_start_tokens=0,
                candidate_predictive_update_start_tokens=0,
                candidate_deep_sleep_filter_start_tokens=10**9,
                candidate_memory_pressure_filter_start_tokens=10**9,
                enable_context_layer=False,
                enable_binding_layer=False,
                enable_abstraction_layer=False,
                cuda_graph_host_truth_sync_interval_tokens=1,
                device="cuda",
            )
            torch.manual_seed(20260616)
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.train_step(
                torch.rand(cfg.input_dim, device="cuda"),
                raw_window="checkpoint route bank seed",
                allow_sleep_maintenance=False,
            )
            seed_report = trainer.column_transition_runtime_report()
            self.assertTrue(seed_report["route_candidate_bank"]["ready"])
            self.assertEqual(seed_report["route_candidate_bank"]["seed_count"], 1)

            checkpoint = save_trainer_checkpoint(
                Path(tmpdir) / "route-bank-restored.pt",
                trainer,
            )
            restored, _metadata = load_trainer_checkpoint(checkpoint)
            before = restored.column_transition_runtime_report()
            self.assertTrue(before["route_candidate_bank"]["ready"])
            self.assertEqual(
                before["route_candidate_bank"]["checkpoint_restore_count"],
                1,
            )
            self.assertEqual(
                before["route_candidate_bank"]["restore_reason"],
                "route_candidate_bank_restored_from_checkpoint",
            )

            restored.train_step(
                torch.rand(cfg.input_dim, device="cuda"),
                raw_window="checkpoint route bank bounded first tick",
                allow_sleep_maintenance=False,
            )
            torch.cuda.synchronize()
            after = restored.column_transition_runtime_report()
            self.assertEqual(after["route_candidate_bank"]["seed_count"], 0)
            self.assertEqual(
                after["route_candidate_bank"]["graph_bypass_count"],
                0,
            )
            self.assertEqual(
                after["route_vote_scoring"]["route_input_rows_scored"],
                cfg.k_routing + 2,
            )
            self.assertFalse(after["route_vote_scoring"]["route_rows_run_all_columns"])
            self.assertTrue(after["route_vote_scoring"]["bounded_route_scoring"])
            self.assertIsNone(
                after["route_vote_scoring"]["route_scoring_unbounded_reason"]
            )
            self.assertGreaterEqual(
                after["cuda_graph_route_transition"]["replay_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
