from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
import time
import unittest
from threading import RLock
from types import SimpleNamespace
from unittest.mock import patch
from typing import Any

from hecsn.config.model_config import HECSNConfig
from hecsn.service.cortex_controller import CortexController, CORTEX_CONTROLLER_STATE_FIELDS
from hecsn.service.cortex_runtime import CortexRuntimeMixin
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_manager(root: Path, *, test_case: str) -> HECSNServiceManager:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.dirty_state = False
        self.state_revision = 0

    def mutation_summary(self) -> dict[str, Any]:
        return {"dirty_state": self.dirty_state, "state_revision": self.state_revision}


class _FakeThoughtLoop:
    def __init__(self) -> None:
        self.is_running = False
        self.submitted_queries: list[str] = []
        self.sleep_requests: list[dict[str, Any]] = []
        self.injected_action_results: list[dict[str, Any]] = []
        self.start_calls = 0

    def submit_query(self, query: str) -> None:
        self.submitted_queries.append(query)

    def request_sleep(self, **kwargs: Any) -> dict[str, Any]:
        request = {
            "source": kwargs.get("source", "operator"),
            "reason": kwargs.get("reason", ""),
            "metadata": deepcopy(kwargs.get("metadata") or {}),
        }
        payload = {
            "accepted": True,
            "coalesced": False,
            "running": self.is_running,
            "request": request,
            "sleep_control": {"requests_submitted": len(self.sleep_requests) + 1},
        }
        self.sleep_requests.append(deepcopy(payload))
        return payload

    def inject_action_result(self, **kwargs: Any) -> None:
        self.injected_action_results.append(deepcopy(kwargs))

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "running": self.is_running,
            "thoughts_generated": len(self.submitted_queries),
            "dreams_generated": len(self.sleep_requests),
            "current_mode": "idle",
            "recent_thoughts": [{"thought": query} for query in self.submitted_queries],
        }

    def start(self) -> None:
        self.start_calls += 1
        self.is_running = True

    def stop(self, timeout: float = 5.0) -> None:
        self.is_running = False

    def request_stop(self) -> None:
        self.is_running = False


class _BuildableFakeThoughtLoop(_FakeThoughtLoop):
    def __init__(
        self,
        *,
        cortex: Any,
        memory: Any,
        curiosity_controller: Any = None,
        signal_provider: Any = None,
        narrative_state_path: str = "",
        on_thought: Any = None,
        on_sleep_summary: Any = None,
    ) -> None:
        super().__init__()
        self.cortex = cortex
        self.memory = memory
        self.curiosity_controller = curiosity_controller
        self.signal_provider = signal_provider
        self.narrative_state_path = narrative_state_path
        self.on_thought = on_thought
        self.on_sleep_summary = on_sleep_summary


class _FakeCortex:
    def __init__(self, model: str = "fake-model") -> None:
        self.model = model


class _FakeEmbedder:
    pass


class _FakeMemory:
    def __init__(self, capacity: int, embedder: Any) -> None:
        self.capacity = capacity
        self.embedder = embedder


class _FakeManager:
    def __init__(self, *, checkpoint_dir: Path) -> None:
        self._lock = RLock()
        self._runtime_state = _FakeRuntimeState()
        self._checkpoint_dir = checkpoint_dir
        self._brain_running = False
        self._geometric_curiosity = None
        self._action_history: deque[dict[str, Any]] = deque()
        self.recorded_events: list[dict[str, Any]] = []
        self.executed_actions: list[dict[str, Any]] = []
        self.relevant_records: list[dict[str, Any]] = []

    def _record_brain_event_locked(self, event: dict[str, Any]) -> None:
        self.recorded_events.append(deepcopy(event))

    def _action_history_memory_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        return {"action_id": record.get("action_id"), "recorded": True}

    def _action_query_terms(self, query_text: str) -> tuple[str, ...]:
        return tuple(part for part in str(query_text).split() if part)

    def _action_focus_query_text(self, query_text: str) -> str:
        return " ".join(str(query_text).split())

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        return ""

    def _query_web_url_candidate(self, query_text: str) -> str:
        return ""

    def _query_api_url_candidate(self, query_text: str) -> str:
        return ""

    def _api_request_record_matches_explicit_url(self, record: dict[str, Any], explicit_url: str) -> bool:
        return False

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: tuple[str, ...] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        return [deepcopy(item) for item in self.relevant_records[:limit]]

    def _action_record_to_response_episodes_locked(
        self,
        record: dict[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        return [deepcopy(record)]

    def execute_digital_action(
        self,
        action: dict[str, Any],
        *,
        trigger_reason: str | None = None,
        trigger_query_text: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "accepted": True,
            "result": {
                "action_id": "new-action",
                "action_type": action.get("action_type"),
                "inputs": deepcopy(action),
                "verification": {"status": "verified", "success": True, "confidence": 0.8, "contradiction": False},
                "episode_text": "executed action",
                "topics": ["cats"],
            },
        }
        self.executed_actions.append(
            {
                "action": deepcopy(action),
                "trigger_reason": trigger_reason,
                "trigger_query_text": trigger_query_text,
            }
        )
        return payload

    def _brain_runtime_snapshot_locked(self) -> dict[str, Any]:
        return {"running": self._brain_running}

    def _cortex_signal_state(self) -> dict[str, Any]:
        return {"prediction_error_mean": 0.0, "prediction_error_max": 0.0, "predictive_confidence_mean": 0.5, "predictive_confidence_min": 0.5, "recent_concepts": []}


class CortexControllerTests(unittest.TestCase):
    def test_runtime_mixin_alias_points_to_controller(self) -> None:
        self.assertIs(CortexRuntimeMixin, CortexController)

    def test_manager_routes_cortex_state_to_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_controller_state_ownership")
            try:
                for field_name in CORTEX_CONTROLLER_STATE_FIELDS:
                    with self.subTest(field_name=field_name):
                        self.assertNotIn(field_name, manager.__dict__)
                        self.assertIn(field_name, manager._cortex_controller.__dict__)
            finally:
                manager.close()

    def test_query_hint_sleep_and_thoughts_use_controller_state(self) -> None:
        fake_manager = _FakeManager(checkpoint_dir=Path("."))
        controller = CortexController(fake_manager)
        loop = _FakeThoughtLoop()

        controller._thought_loop = loop
        ask = controller.cortex_ask("  cats   chase   mice  ")
        self.assertTrue(ask["accepted"])
        self.assertEqual(loop.submitted_queries, ["  cats   chase   mice  "])
        self.assertEqual(controller._last_cortex_query_hint_text, "cats chase mice")
        self.assertNotIn("_last_cortex_query_hint_text", fake_manager.__dict__)

        sleep = controller.cortex_sleep("  rest   now  ")
        self.assertTrue(sleep["accepted"])
        self.assertEqual(loop.sleep_requests[0]["request"]["reason"], "rest now")
        self.assertEqual(fake_manager.recorded_events[-1]["type"], "cortex_sleep_requested")

        thoughts = controller.cortex_thoughts(limit=1)
        self.assertTrue(thoughts["enabled"])
        self.assertEqual(len(thoughts["thoughts"]), 1)
        snapshot = controller.cortex_snapshot()
        self.assertTrue(snapshot["enabled"])

    def test_build_thought_loop_injects_action_history(self) -> None:
        fake_manager = _FakeManager(checkpoint_dir=Path("C:/tmp"))
        controller = CortexController(fake_manager)
        injected_loop = _FakeThoughtLoop()

        def _create_cortex() -> _FakeCortex:
            return _FakeCortex()

        def _create_embedder(*, allow_fallback: bool = False) -> _FakeEmbedder:
            return _FakeEmbedder()

        controller._cortex_factory_refs = (
            _BuildableFakeThoughtLoop,
            _create_cortex,
            _create_embedder,
            _FakeMemory,
        )
        action_history = [
            {
                "episode_text": "workspace search episode",
                "topics": ["cats", "mice"],
                "verification": {"success": True, "confidence": 0.9, "contradiction": False},
                "action_id": "action-1",
            }
        ]

        with patch.object(controller, "_inject_action_record_into_loop", wraps=controller._inject_action_record_into_loop) as inject:
            built = controller._build_cortex_thought_loop(action_history)

        self.assertIsInstance(built, _BuildableFakeThoughtLoop)
        self.assertEqual(inject.call_count, 1)
        self.assertEqual(built.injected_action_results[0]["content"], "workspace search episode")
        self.assertEqual(built.injected_action_results[0]["topics"], ("cats", "mice"))
        self.assertEqual(built.injected_action_results[0]["metadata"]["action_id"], "action-1")

    def test_delayed_initialization_starts_loop_when_runtime_active(self) -> None:
        fake_manager = _FakeManager(checkpoint_dir=Path("C:/tmp"))
        fake_manager._brain_running = True
        controller = CortexController(fake_manager)
        loop = _FakeThoughtLoop()

        with patch.object(controller, "_build_cortex_thought_loop", return_value=loop) as build_loop:
            controller._start_cortex_initialization()
            self.assertTrue(controller._cortex_init_event.wait(timeout=1.0))

        self.assertTrue(build_loop.called)
        self.assertIs(controller._thought_loop_actual, loop)
        self.assertTrue(loop.is_running)
        self.assertTrue(controller._cortex_available)

    def test_action_intent_reuses_recent_verified_action(self) -> None:
        fake_manager = _FakeManager(checkpoint_dir=Path("C:/tmp"))
        controller = CortexController(fake_manager)
        fake_manager.relevant_records = [
            {
                "action_id": "action-1",
                "action_type": "workspace_search",
                "inputs": {"query_text": "cats chase mice"},
                "verification": {"status": "verified", "success": True, "confidence": 0.9, "contradiction": False},
            }
        ]
        controller._remember_cortex_query_hint_locked("cats chase mice")

        result = controller._handle_cortex_action_intent_locked(
            SimpleNamespace(action_intent="search", thought="cats chase mice", topics=["cats"]),
            action_intent="search",
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result["reused"])
        self.assertEqual(result["record"]["action_id"], "action-1")
        self.assertEqual(fake_manager.executed_actions, [])
        self.assertEqual(fake_manager.recorded_events[-1]["type"], "cortex_action_reused")


if __name__ == "__main__":
    unittest.main()
