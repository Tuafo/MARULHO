from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest
from threading import RLock
from unittest.mock import patch

from marulho.service.persistence import RuntimePersistence, RuntimePersistenceDependencies


@dataclass
class _FakeRuntimeState:
    state_revision: int = 9

    def __post_init__(self) -> None:
        self.clean_calls = 0

    def mark_clean(self) -> None:
        self.clean_calls += 1

    def mutation_summary(self) -> dict[str, int]:
        return {"dirty_state": False, "state_revision": self.state_revision}


class _FakeConceptStore:
    def state_dict(self) -> dict[str, object]:
        return {"concept_mode": "slow_feature_concept_memory", "concept_count": 1}


class _FakeTrainer:
    token_count = 17


class _FakePersistenceManager:
    def __init__(self, root: Path) -> None:
        self._lock = RLock()
        self._trace_dir = root / "traces"
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = root / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self._checkpoint_dir / "initial.pt"
        self._metadata = {"service_state": {"existing": True}}
        self._runtime_state = _FakeRuntimeState()
        self._trainer = _FakeTrainer()
        self._concept_store = _FakeConceptStore()
        self._snn_language_plasticity_state = {
            "sparse_transition_weights": {"1:2": 0.5},
            "synapse_provenance_by_key": {
                "1:2": {
                    "provenance_type": "replay_regeneration",
                    "permit_id": "permit-1",
                    "replay_artifact_id": "artifact-1",
                    "source_metadata_hash": "source-metadata-hash-1",
                    "emission_lineage": {
                        "emission_hash": "emission-hash-1",
                        "readout_evidence_hash": "readout-hash-1",
                        "prediction_hash": "prediction-hash-1",
                    },
                }
            },
            "homeostatic_maintenance": {
                "maintenance_count": 1,
                "recent_events": [{"event_index": 1, "pruned_synapse_count": 0}],
            },
        }
        self._snn_language_readout_ledger_state = {
            "events": [{"readout_evidence_hash": "readout-hash-1", "prediction_hash": "prediction-hash-1"}],
            "total_recorded_count": 1,
        }
        self._runtime_env = {}
        self._env_root = None
        self._action_root = root
        self._replay_sample_history = deque(maxlen=256)
        self._replay_regeneration_permits = deque(maxlen=64)
        self._snn_replay_evaluation_contexts = deque(maxlen=64)
        self._snn_replay_artifact_recording_review_tickets = deque(maxlen=64)
        self._snn_transition_memory_replay_artifacts = deque(maxlen=64)
        self._delayed_consequence_records = deque(maxlen=24)
        self._delayed_consequence_cooled_total = 0
        self._delayed_consequence_retired_total = 0
        self._delayed_consequence_compacted_total = 0
        self._delayed_consequence_split_total = 0
        self._delayed_consequence_remerged_total = 0
        self._brain_last_error = None
        self._brain_config = {"tick_tokens": 8}

    def _brain_persisted_state_locked(self) -> dict[str, object]:
        return {
            "replay_sample_history": [dict(item) for item in list(self._replay_sample_history)],
            "replay_regeneration_permits": [dict(item) for item in list(self._replay_regeneration_permits)],
            "snn_replay_evaluation_contexts": [
                dict(item) for item in list(self._snn_replay_evaluation_contexts)
            ],
            "snn_replay_artifact_recording_review_tickets": [
                dict(item) for item in list(self._snn_replay_artifact_recording_review_tickets)
            ],
            "snn_transition_memory_replay_artifacts": [
                dict(item) for item in list(self._snn_transition_memory_replay_artifacts)
            ],
            "last_event": {"type": "brain_event_recorded"},
            "recent_events": [{"type": "brain_event_recorded"}],
            "action_history": [],
            "runtime_episode_traces": [],
            "delayed_consequence_records": [],
            "delayed_consequence_cooled_total": self._delayed_consequence_cooled_total,
            "delayed_consequence_retired_total": self._delayed_consequence_retired_total,
            "delayed_consequence_compacted_total": self._delayed_consequence_compacted_total,
            "delayed_consequence_split_total": self._delayed_consequence_split_total,
            "delayed_consequence_remerged_total": self._delayed_consequence_remerged_total,
        }

    def _brain_runtime_snapshot_locked(self, **kwargs: object) -> dict[str, object]:
        return self._brain_persisted_state_locked()

    @staticmethod
    def _normalize_background_source_utility_state(value: object) -> dict[str, object]:
        return {} if value is None else dict(value)  # pragma: no cover - defensive

    @staticmethod
    def _normalize_action_record(value: object) -> dict[str, object] | None:
        return None

    @staticmethod
    def _normalize_replay_sample_record(value: object) -> dict[str, object] | None:
        return None

    @staticmethod
    def _normalize_delayed_consequence_record(value: object) -> dict[str, object] | None:
        return None

    def _rebuild_brain_sources_locked(self) -> None:
        return None

    def _request_brain_stop(self):
        return None

    @staticmethod
    def _join_brain_thread(thread: object) -> None:
        return None

    class _InteractionPipelineStub:
        def load_interaction_state(self, **kwargs: object) -> None:
            self.loaded = kwargs

    _interaction_pipeline = _InteractionPipelineStub()


def _runtime_persistence(
    manager: _FakePersistenceManager,
    *,
    trace_history_limit: int = 2,
    refresh_root_captures=lambda: None,
) -> RuntimePersistence:
    return RuntimePersistence(
        RuntimePersistenceDependencies(
            get_state=lambda name: getattr(manager, name),
            set_state=lambda name, value: setattr(manager, name, value),
            brain_persisted_state=manager._brain_persisted_state_locked,
            brain_runtime_snapshot=manager._brain_runtime_snapshot_locked,
            join_brain_thread=manager._join_brain_thread,
            lock=manager._lock,
            normalize_background_source_utility_state=manager._normalize_background_source_utility_state,
            normalize_delayed_consequence_record=manager._normalize_delayed_consequence_record,
            rebuild_brain_sources=manager._rebuild_brain_sources_locked,
            refresh_root_captures=refresh_root_captures,
            request_brain_stop=manager._request_brain_stop,
        ),
        trace_history_limit=trace_history_limit,
    )


class RuntimePersistenceTests(unittest.TestCase):
    def test_critical_period_learning_state_survives_checkpoint_normalization(
        self,
    ) -> None:
        developmental = {
            "learning_cycle_count": 2,
            "by_synapse": {
                "4:64": {
                    "critical_period_age_cycles": 2,
                    "critical_period_cycles_remaining": 62,
                    "active_cycle_count": 2,
                    "inactive_cycle_count": 0,
                    "current_maturation_state": "critical_period",
                    "critical_period_learning_application_hash": "a" * 64,
                }
            },
            "last_learning_cycle": {
                "newborn_neuron_critical_period_learning_event_hash": "b"
                * 64,
                "after_state_revision": 4,
            },
        }

        checkpoint_state = (
            RuntimePersistence._snn_language_plasticity_checkpoint_state(
                {
                    "sparse_transition_weights": {"4:64": 0.01625},
                    "thought_newborn_neuron_critical_period_learning": (
                        developmental
                    ),
                }
            )
        )

        self.assertEqual(
            checkpoint_state[
                "thought_newborn_neuron_critical_period_learning"
            ],
            developmental,
        )
        self.assertEqual(
            checkpoint_state["sparse_transition_weights"]["4:64"],
            0.01625,
        )

    def test_language_capacity_checkpoint_preserves_dynamic_mutation_truth(
        self,
    ) -> None:
        event = {
            "capacity_mutation_event_hash": "a" * 64,
            "target_neuron_capacity": 66,
        }

        capacity = RuntimePersistence._snn_language_capacity_checkpoint_state(
            {
                "language_capacity": {
                    "language_neuron_count": 66,
                    "sparse_edge_budget": 258,
                    "outgoing_fanout_budget": 16,
                    "dynamic_capacity_enabled": True,
                    "capacity_expansion_count": 1,
                    "resizes_network": True,
                    "adds_neurons": True,
                    "adds_layers": False,
                    "writes_checkpoint": True,
                    "last_capacity_mutation": event,
                }
            }
        )

        self.assertEqual(capacity["language_neuron_count"], 66)
        self.assertEqual(capacity["sparse_edge_budget"], 258)
        self.assertEqual(capacity["outgoing_fanout_budget"], 16)
        self.assertTrue(capacity["dynamic_capacity_enabled"])
        self.assertEqual(capacity["capacity_expansion_count"], 1)
        self.assertTrue(capacity["resizes_network"])
        self.assertTrue(capacity["adds_neurons"])
        self.assertFalse(capacity["adds_layers"])
        self.assertTrue(capacity["writes_checkpoint"])
        self.assertEqual(capacity["last_capacity_mutation"], event)

    def test_persist_trace_is_owned_by_runtime_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager, trace_history_limit=2)

            trace_path = persistence.persist_trace(
                {
                    "trace_id": "trace-1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "payload": {"path": Path("reports/runtime/trace.json")},
                }
            )

            history = persistence.recent_traces(limit=1)
            self.assertTrue(trace_path.exists())
            self.assertEqual(history[0]["trace_id"], "trace-1")
            self.assertEqual(Path(str(history[0]["trace_path"])).name, trace_path.name)

    def test_trace_history_setter_preserves_existing_deque_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager, trace_history_limit=2)

            history = persistence.trace_history
            persistence.trace_history = [
                {
                    "trace_id": "trace-1",
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]

            self.assertIs(persistence.trace_history, history)
            self.assertEqual(history[0]["trace_id"], "trace-1")

    def test_save_checkpoint_uses_runtime_state_and_service_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            manager._replay_sample_history.appendleft(
                {
                    "schema_version": 1,
                    "replay_sample_id": "replay-1",
                    "mode": "sample",
                    "selected_candidate_ids": ["candidate-1"],
                }
            )
            manager._replay_regeneration_permits.appendleft(
                {"permit_id": "permit-1", "evidence_hash": "hash-1"}
            )
            manager._snn_replay_evaluation_contexts.appendleft(
                {"replay_evaluation_context_id": "context-1", "evidence_hash": "context-hash-1"}
            )
            manager._snn_replay_artifact_recording_review_tickets.appendleft(
                {"review_ticket_id": "ticket-1", "evidence_hash": "ticket-hash-1"}
            )
            manager._snn_transition_memory_replay_artifacts.appendleft(
                {"replay_artifact_id": "artifact-1", "evidence_hash": "artifact-hash-1"}
            )
            persistence = _runtime_persistence(manager, trace_history_limit=2)

            captured: dict[str, object] = {}

            def _fake_save_trainer_checkpoint(path: Path, trainer: object, *, metadata: dict[str, object]) -> Path:
                captured["path"] = path
                captured["trainer"] = trainer
                captured["metadata"] = metadata
                path.write_bytes(b"checkpoint")
                return path

            with patch("marulho.service.persistence.save_trainer_checkpoint", side_effect=_fake_save_trainer_checkpoint):
                with patch("marulho.service.persistence.load_trainer_checkpoint", return_value=(manager._trainer, {"state_revision": 9})):
                    result = persistence.save_checkpoint(str(root / "service.pt"))

            self.assertTrue(manager._runtime_state.clean_calls >= 1)
            self.assertEqual(result["token_count"], 17)
            self.assertEqual(Path(result["path"]).name, "service.pt")
            metadata = captured["metadata"]
            assert isinstance(metadata, dict)
            service_state = metadata["service_state"]
            assert isinstance(service_state, dict)
            self.assertEqual(service_state["concept_store"]["concept_mode"], "slow_feature_concept_memory")
            self.assertEqual(service_state["terminus_runtime"]["replay_sample_history"][0]["replay_sample_id"], "replay-1")
            self.assertEqual(service_state["terminus_runtime"]["replay_regeneration_permits"][0]["permit_id"], "permit-1")
            self.assertEqual(
                service_state["terminus_runtime"]["snn_replay_evaluation_contexts"][0][
                    "replay_evaluation_context_id"
                ],
                "context-1",
            )
            self.assertEqual(
                service_state["terminus_runtime"]["snn_replay_artifact_recording_review_tickets"][0][
                    "review_ticket_id"
                ],
                "ticket-1",
            )
            self.assertEqual(
                service_state["terminus_runtime"]["snn_transition_memory_replay_artifacts"][0][
                    "replay_artifact_id"
                ],
                "artifact-1",
            )
            self.assertEqual(service_state["snn_language_plasticity"]["sparse_transition_weights"]["1:2"], 0.5)
            capacity_state = service_state["snn_language_plasticity"]["language_capacity"]
            self.assertEqual(capacity_state["surface"], "snn_language_capacity_state.v1")
            self.assertTrue(capacity_state["owned_by_marulho"])
            self.assertEqual(capacity_state["language_neuron_count"], 64)
            self.assertEqual(capacity_state["sparse_edge_budget"], 256)
            self.assertEqual(capacity_state["outgoing_fanout_budget"], 16)
            self.assertFalse(capacity_state["dynamic_capacity_enabled"])
            self.assertFalse(capacity_state["resizes_network"])
            self.assertFalse(capacity_state["adds_neurons"])
            self.assertFalse(capacity_state["adds_layers"])
            dense_layout = service_state["snn_language_plasticity"][
                "dense_readout_layout"
            ]
            self.assertEqual(
                dense_layout["surface"],
                "snn_language_dense_readout_layout_state.v1",
            )
            self.assertTrue(dense_layout["owned_by_marulho"])
            self.assertEqual(dense_layout["current_dense_readout_shape"], [64, 64])
            self.assertEqual(dense_layout["target_dense_readout_shape"], [64, 64])
            self.assertEqual(dense_layout["preserved_dense_window"], [64, 64])
            self.assertFalse(dense_layout["requires_cuda_relayout"])
            self.assertFalse(dense_layout["dense_resize_applied"])
            self.assertFalse(dense_layout["resizes_network"])
            self.assertEqual(
                service_state["snn_language_plasticity"]["synapse_provenance_by_key"]["1:2"][
                    "source_metadata_hash"
                ],
                "source-metadata-hash-1",
            )
            self.assertEqual(
                service_state["snn_language_plasticity"]["synapse_provenance_by_key"]["1:2"][
                    "emission_lineage"
                ]["emission_hash"],
                "emission-hash-1",
            )
            lineage_summary = service_state["snn_applied_replay_lineage_checkpoint_summary"]
            self.assertEqual(
                lineage_summary["surface"],
                "snn_applied_replay_lineage_checkpoint_summary.v1",
            )
            self.assertEqual(lineage_summary["applied_replay_lineage_count"], 1)
            self.assertEqual(lineage_summary["complete_applied_replay_lineage_count"], 1)
            self.assertEqual(lineage_summary["incomplete_applied_replay_lineage_count"], 0)
            self.assertTrue(lineage_summary["lineage_material_hash"])
            self.assertTrue(lineage_summary["raw_text_absent"])
            self.assertTrue(lineage_summary["operator_identity_absent"])
            self.assertEqual(
                service_state["snn_language_plasticity"]["homeostatic_maintenance"]["recent_events"][0]["event_index"],
                1,
            )
            self.assertEqual(
                service_state["snn_language_readout_ledger"]["events"][0]["readout_evidence_hash"],
                "readout-hash-1",
            )
            self.assertEqual(captured["trainer"], manager._trainer)
            manifest = result["current_checkpoint_manifest"]
            self.assertTrue(Path(manifest["checkpoint_path"]).is_file())
            self.assertIn("objects", Path(manifest["checkpoint_path"]).parts)

    def test_applied_replay_lineage_restore_validation_compares_saved_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager)
            summary = persistence._applied_replay_lineage_checkpoint_summary(
                manager._snn_language_plasticity_state
            )
            metadata = {
                "service_state": {
                    "snn_applied_replay_lineage_checkpoint_summary": summary,
                }
            }

            validation = persistence._applied_replay_lineage_restore_validation(metadata)
            tampered_metadata = deepcopy(metadata)
            tampered_metadata["service_state"][
                "snn_applied_replay_lineage_checkpoint_summary"
            ]["lineage_material_hash"] = "tampered"
            tampered = persistence._applied_replay_lineage_restore_validation(
                tampered_metadata
            )

            self.assertEqual(
                validation["surface"],
                "snn_applied_replay_lineage_restore_validation.v1",
            )
            self.assertTrue(validation["saved_summary_available"])
            self.assertTrue(validation["summary_counts_match_restored_state"])
            self.assertTrue(validation["summary_hash_matches_restored_state"])
            self.assertTrue(validation["summary_matches_restored_state"])
            self.assertEqual(
                validation["restored_summary"]["lineage_material_hash"],
                summary["lineage_material_hash"],
            )
            self.assertFalse(validation["runs_replay"])
            self.assertFalse(validation["applies_plasticity"])
            self.assertFalse(validation["issues_regeneration_permit"])
            self.assertFalse(tampered["summary_hash_matches_restored_state"])
            self.assertFalse(tampered["summary_matches_restored_state"])
            self.assertTrue(tampered["summary_counts_match_restored_state"])

    def test_failed_manifest_publication_preserves_previous_current_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager)
            previous = root / "previous.pt"
            next_path = root / "next.pt"
            previous.write_bytes(b"previous")
            next_path.write_bytes(b"next")
            with patch("marulho.service.persistence.load_trainer_checkpoint", return_value=(manager._trainer, {"state_revision": 9})):
                persistence.publish_current_checkpoint(previous, operation="previous")

                real_replace = __import__("os").replace

                def _fail_manifest_replace(source: object, target: object) -> None:
                    if Path(target).name == "marulho_current_checkpoint.json":
                        raise RuntimeError("interrupted")
                    real_replace(source, target)

                with patch("marulho.service.persistence.os.replace", side_effect=_fail_manifest_replace):
                    with self.assertRaisesRegex(RuntimeError, "interrupted"):
                        persistence.publish_current_checkpoint(next_path, operation="next")

            manifest = json.loads((root / "checkpoints" / "marulho_current_checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["current"]["operation"], "previous")

    def test_failed_refresh_does_not_publish_manifest_or_change_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            original_path = manager._checkpoint_path
            original_metadata = dict(manager._metadata)
            persistence = _runtime_persistence(
                manager,
                refresh_root_captures=lambda: (_ for _ in ()).throw(RuntimeError("refresh failed")),
            )
            next_path = root / "next.pt"
            next_path.write_bytes(b"next")

            with patch("marulho.service.persistence.load_trainer_checkpoint", return_value=(manager._trainer, {"state_revision": 9})):
                with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                    persistence.publish_current_checkpoint(next_path, operation="next")

            self.assertFalse((root / "checkpoints" / "marulho_current_checkpoint.json").exists())
            self.assertEqual(manager._checkpoint_path, original_path)
            self.assertEqual(manager._metadata, original_metadata)

    def test_resolver_falls_back_to_previous_descriptor_when_current_object_is_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager)
            previous = root / "previous.pt"
            next_path = root / "next.pt"
            previous.write_bytes(b"previous")
            next_path.write_bytes(b"next")
            with patch("marulho.service.persistence.load_trainer_checkpoint", return_value=(manager._trainer, {"state_revision": 9})):
                first = persistence.publish_current_checkpoint(previous, operation="previous")
                second = persistence.publish_current_checkpoint(next_path, operation="next")
                Path(second["checkpoint_path"]).write_bytes(b"corrupt")
                resolved = RuntimePersistence.resolve_current_checkpoint_path(root / "checkpoints" / "initial.pt")

            self.assertEqual(resolved, Path(first["checkpoint_path"]).resolve())

    def test_unpublished_snapshot_does_not_change_active_checkpoint_bookkeeping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _FakePersistenceManager(root)
            persistence = _runtime_persistence(manager)
            original_path = manager._checkpoint_path
            original_metadata = dict(manager._metadata)

            def _fake_save(path: Path, trainer: object, *, metadata: dict[str, object]) -> Path:
                path.write_bytes(b"checkpoint")
                return path

            with patch("marulho.service.persistence.save_trainer_checkpoint", side_effect=_fake_save):
                result = persistence.save_checkpoint(str(root / "rollback.pt"), publish=False)

            self.assertEqual(Path(result["path"]), root / "rollback.pt")
            self.assertEqual(manager._checkpoint_path, original_path)
            self.assertEqual(manager._metadata, original_metadata)
            self.assertEqual(manager._runtime_state.clean_calls, 0)


if __name__ == "__main__":
    unittest.main()
