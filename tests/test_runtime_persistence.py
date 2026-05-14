from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from threading import RLock
from unittest.mock import patch

from hecsn.service.persistence import RuntimePersistence, RuntimePersistenceDependencies


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
        self._runtime_env = {}
        self._env_root = None
        self._action_root = root
        self._replay_sample_history = deque(maxlen=256)
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

    def _replay_action_history_into_cortex_locked(self) -> None:
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


def _runtime_persistence(manager: _FakePersistenceManager, *, trace_history_limit: int = 2) -> RuntimePersistence:
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
            replay_action_history_into_cortex=manager._replay_action_history_into_cortex_locked,
            request_brain_stop=manager._request_brain_stop,
        ),
        trace_history_limit=trace_history_limit,
    )


class RuntimePersistenceTests(unittest.TestCase):
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
            persistence = _runtime_persistence(manager, trace_history_limit=2)

            captured: dict[str, object] = {}

            def _fake_save_trainer_checkpoint(path: Path, trainer: object, *, metadata: dict[str, object]) -> Path:
                captured["path"] = path
                captured["trainer"] = trainer
                captured["metadata"] = metadata
                return path

            with patch("hecsn.service.persistence.save_trainer_checkpoint", side_effect=_fake_save_trainer_checkpoint):
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
            self.assertEqual(captured["trainer"], manager._trainer)


if __name__ == "__main__":
    unittest.main()
