from __future__ import annotations

from pathlib import Path
from threading import Event, RLock
import tempfile
import unittest
from unittest.mock import patch
from typing import Any, cast

from hecsn.config.model_config import HECSNConfig
from hecsn.service.manager import HECSNServiceManager
from hecsn.service.runtime_control import RuntimeControl
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
        self.events: list[dict[str, object]] = []

    def mutation_summary(self) -> dict[str, object]:
        return {"dirty_state": False, "state_revision": 0}

    def mark_mutated(self) -> None:
        return None


class _FakeManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runtime_state = _FakeRuntimeState()
        self.recorded_events: list[dict[str, object]] = []

    def _record_brain_event_locked(self, event: dict[str, object]) -> None:
        self.recorded_events.append(dict(event))

    def _interrupt_brain_sources_locked(self) -> None:
        return None

    def _interrupt_sensory_sources_locked(self) -> None:
        return None


class _FakeThread:
    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)


class RuntimeControlTests(unittest.TestCase):
    def test_manager_uses_explicit_runtime_control_seam(self) -> None:
        self.assertNotIn(RuntimeControl, HECSNServiceManager.__mro__)

    def test_manager_runtime_control_state_lives_on_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _build_manager(Path(tmpdir), test_case="runtime_control_state_ownership")
            try:
                self.assertNotIn("_brain_thread", manager.__dict__)
                self.assertNotIn("_brain_stop_event", manager.__dict__)
                self.assertNotIn("_ingestion_prewarm_thread", manager.__dict__)
                self.assertNotIn("_active_execution_requests", manager.__dict__)
                self.assertNotIn("_remote_warm_promotion_thread", manager.__dict__)
                self.assertIn("_brain_thread", manager._runtime_control.__dict__)
                self.assertIn("_brain_stop_event", manager._runtime_control.__dict__)
                self.assertIn("_ingestion_prewarm_thread", manager._runtime_control.__dict__)
                self.assertIn("_active_execution_requests", manager._runtime_control.__dict__)
                self.assertIn("_remote_warm_promotion_thread", manager._runtime_control.__dict__)
            finally:
                manager.close()

    def test_controller_owns_lifecycle_and_prewarm_state(self) -> None:
        controller = RuntimeControl(_FakeManager())
        brain_thread = _FakeThread()
        prewarm_thread = _FakeThread()
        promotion_thread = _FakeThread()
        controller_any = cast(Any, controller)

        controller_any._brain_thread = brain_thread
        controller_any._brain_stop_event = Event()
        controller_any._brain_running = True
        controller_any._brain_running_since = "2026-05-10T00:00:00+00:00"
        controller_any._ingestion_prewarm_thread = prewarm_thread
        controller_any._ingestion_prewarm_stop_event = Event()
        controller_any._remote_warm_promotion_thread = promotion_thread
        controller_any._remote_warm_promotion_stop_event = Event()
        controller_any._active_execution_requests = 2

        self.assertIs(controller_any._brain_thread, brain_thread)
        self.assertIs(controller_any._ingestion_prewarm_thread, prewarm_thread)
        self.assertIs(controller_any._remote_warm_promotion_thread, promotion_thread)
        self.assertEqual(controller_any._active_execution_requests, 2)
        self.assertTrue(controller_any._brain_running)
        self.assertIn("_brain_thread", controller.__dict__)
        self.assertIn("_ingestion_prewarm_thread", controller.__dict__)
        self.assertIn("_remote_warm_promotion_thread", controller.__dict__)
        self.assertIn("_active_execution_requests", controller.__dict__)
        self.assertIs(controller.dependencies, controller_any._dependencies)
        self.assertNotIn("_brain_thread", controller.dependencies.__dict__)
        self.assertNotIn("_ingestion_prewarm_thread", controller.dependencies.__dict__)
        self.assertNotIn("_remote_warm_promotion_thread", controller.dependencies.__dict__)
        self.assertNotIn("_active_execution_requests", controller.dependencies.__dict__)

    def test_controller_no_longer_uses_manager_bound_transition_base(self) -> None:
        source = Path("src/hecsn/service/runtime_control.py").read_text(encoding="utf-8")

        self.assertNotIn("ExplicitOwnerModule", source)
        self.assertNotIn("install_owner_forwarders", source)
        self.assertNotIn('"_manager"', source)
        self.assertNotIn("'_manager'", source)

    def test_request_and_finalize_brain_stop_transition_state(self) -> None:
        manager = _FakeManager()
        controller = RuntimeControl(manager)
        thread = _FakeThread()
        controller_any = cast(Any, controller)
        controller_any._brain_thread = thread
        controller_any._brain_stop_event = Event()
        controller_any._brain_running = True
        controller_any._brain_running_since = "2026-05-10T00:00:00+00:00"

        with patch("hecsn.service.runtime_control.time.perf_counter", return_value=101.0):
            requested = controller._request_brain_stop_locked(reason="manual")
        self.assertIs(requested, thread)
        self.assertTrue(cast(Event, controller_any._brain_stop_event).is_set())
        self.assertFalse(controller_any._brain_running)
        self.assertEqual(controller_any._brain_stop_requested_reason, "manual")
        self.assertGreaterEqual(len(manager.recorded_events), 1)
        self.assertEqual(manager.recorded_events[-1]["type"], "stop_requested")

        thread._alive = False
        with patch("hecsn.service.runtime_control.time.perf_counter", return_value=101.75):
            controller._finalize_brain_stop_locked(cast(Any, thread))

        self.assertIsNone(controller_any._brain_thread)
        self.assertIsNone(controller_any._brain_stop_event)
        self.assertIsNone(controller_any._brain_stop_requested_reason)
        self.assertIsNone(controller_any._brain_stop_requested_perf)
        self.assertIsNone(controller_any._brain_running_since)
        self.assertFalse(controller_any._brain_stop_timed_out)
        self.assertIsNotNone(controller_any._brain_last_stop_duration_ms)
        self.assertAlmostEqual(cast(float, controller_any._brain_last_stop_duration_ms), 750.0, places=3)
        self.assertEqual(manager.recorded_events[-1]["type"], "stopped")

    def test_tick_rejected_while_background_runtime_is_active(self) -> None:
        controller = RuntimeControl(_FakeManager())
        controller_any = cast(Any, controller)
        controller_any._brain_thread = _FakeThread()
        controller_any._brain_running = True

        with self.assertRaisesRegex(ValueError, "background runtime is active"):
            controller.terminus_tick()

    def test_join_brain_thread_records_timeout_state(self) -> None:
        manager = _FakeManager()
        controller = RuntimeControl(manager)
        controller_any = cast(Any, controller)
        controller_any._brain_stop_requested_reason = "manual"
        controller_any._brain_stop_requested_perf = 100.0
        thread = _FakeThread(alive=True)

        joined = controller._join_brain_thread(cast(Any, thread), timeout=0.0, raise_on_timeout=False)

        self.assertFalse(joined)
        self.assertTrue(controller._brain_stop_timed_out)
        self.assertEqual(controller._brain_last_stop_duration_ms, 0.0)
        self.assertEqual(manager.recorded_events[-1]["type"], "stop_timeout")


if __name__ == "__main__":
    unittest.main()
