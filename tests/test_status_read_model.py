"""Direct test surface for the Status Read Model seam.

These tests exercise the StatusReadModel through its own interface with
injected adapters, verifying snapshot payloads and cache/freshness semantics
without requiring the full Service Manager composition root.

Regression coverage for unchanged public behavior remains in
test_service_manager.py and test_service_api.py.
"""
from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any

from hecsn.config.model_config import HECSNConfig
from hecsn.service.runtime_state import RuntimeState
from hecsn.service.status_read_model import StatusReadModel
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer

def _build_config() -> HECSNConfig:
    return HECSNConfig(
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


def _build_brain_snapshot() -> dict[str, Any]:
    return {
        "configured": False,
        "running": False,
        "running_since": None,
        "last_error": None,
        "tick_count": 0,
        "background_tokens_processed": 0,
        "autonomy_tokens_processed": 0,
        "last_work_at": None,
        "source_bank": [],
        "cortex": {"enabled": False},
        "living_loop": {},
    }


def _build_animation_snapshot() -> dict[str, Any]:
    return {
        "n_columns": 4,
        "winner_id": None,
        "activations": [0.0, 0.0, 0.0, 0.0],
        "spike_counts": [0, 0, 0, 0],
        "cross_modal": None,
        "context_tau": None,
        "binding": None,
        "abstraction": None,
        "stdp": None,
        "memory_fill": 0.0,
    }


def _build_read_model(
    *,
    cortex_active: bool = False,
) -> tuple[StatusReadModel, HECSNTrainer, threading.RLock, RuntimeState]:
    cfg = _build_config()
    trainer = HECSNTrainer(HECSNModel(cfg), cfg)
    lock = threading.RLock()
    runtime_state = RuntimeState(lock=lock)
    brain_snapshot = _build_brain_snapshot()
    animation_snapshot = _build_animation_snapshot()
    model = StatusReadModel(
        lock=lock,
        runtime_state=runtime_state,
        trainer=trainer,
        trace_history=deque(maxlen=200),
        metadata={},
        checkpoint_path_str="/tmp/test.pt",
        trace_dir_str="/tmp/traces",
        concept_store_snapshot_fn=lambda: deepcopy({"top_concepts": [], "total_concepts": 0}),
        brain_runtime_snapshot_fn=lambda: deepcopy(brain_snapshot),
        cortex_active_fn=lambda: cortex_active,
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
    )
    return model, trainer, lock, runtime_state


class StatusReadModelConstructionTests(unittest.TestCase):
    """StatusReadModel can be constructed with injected dependencies."""

    def test_read_model_constructs_with_adapter(self) -> None:
        """StatusReadModel should accept a manager-like adapter at construction."""
        model, _, _, _ = _build_read_model()
        self.assertIsNotNone(model)


class StatusReadModelStatusTests(unittest.TestCase):
    """StatusReadModel.status() produces valid snapshots with correct payload keys."""

    def test_status_returns_checkpoint_path(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertEqual(result["checkpoint_path"], "/tmp/test.pt")

    def test_status_returns_token_count(self) -> None:
        model, trainer, _, _ = _build_read_model()
        result = model.status()
        self.assertEqual(result["token_count"], int(trainer.token_count))

    def test_status_returns_runtime_truth(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertIn("runtime_truth", result)
        truth = result["runtime_truth"]
        self.assertEqual(truth["schema_version"], 1)
        self.assertIn("verdict", truth)
        self.assertIn("recommended_action", truth)
        self.assertIn("evidence", truth)
        self.assertIn("memory_pressure", truth)

    def test_status_returns_memory_store(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertIn("memory_store", result)

    def test_status_returns_concept_store(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertIn("concept_store", result)

    def test_status_returns_terminus_runtime(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertIn("terminus_runtime", result)

    def test_status_includes_state_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        result = model.status()
        self.assertIn("state_revision", result)
        self.assertEqual(result["state_revision"], runtime_state.state_revision)

    def test_status_includes_dirty_state(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertIn("dirty_state", result)

    def test_status_runtime_truth_verdict_partial_when_unconfigured(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.status()
        self.assertEqual(result["runtime_truth"]["verdict"], "partial")
        self.assertEqual(
            result["runtime_truth"]["recommended_action"],
            "configure_terminus_sources",
        )


class StatusReadModelTerminusStatusTests(unittest.TestCase):
    """StatusReadModel.terminus_status() produces valid snapshots with correct payload keys."""

    def test_terminus_status_returns_terminus_runtime(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.terminus_status()
        self.assertIn("terminus_runtime", result)

    def test_terminus_status_returns_token_count(self) -> None:
        model, trainer, _, _ = _build_read_model()
        result = model.terminus_status()
        self.assertEqual(result["token_count"], int(trainer.token_count))

    def test_terminus_status_returns_runtime_truth(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.terminus_status()
        self.assertIn("runtime_truth", result)
        truth = result["runtime_truth"]
        self.assertEqual(truth["schema_version"], 1)

    def test_terminus_status_includes_state_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        result = model.terminus_status()
        self.assertIn("state_revision", result)
        self.assertEqual(result["state_revision"], runtime_state.state_revision)

    def test_terminus_status_includes_multimodal(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.terminus_status()
        self.assertIn("multimodal", result)

    def test_terminus_status_verdict_partial_when_unconfigured(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.terminus_status()
        self.assertEqual(result["runtime_truth"]["verdict"], "partial")
        self.assertEqual(
            result["runtime_truth"]["recommended_action"],
            "configure_terminus_sources",
        )


class StatusReadModelCacheTests(unittest.TestCase):
    """Cache and non-blocking semantics for status() and terminus_status()."""

    def test_status_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, status() returns cached data instead of blocking."""
        model, _, lock, _ = _build_read_model()
        # First call populates cache
        first = model.status()
        # Hold the lock in another thread and verify we get the cached result
        barrier = threading.Barrier(2, timeout=5.0)
        cached_result: dict[str, Any] | None = [None]

        def _hold_lock_and_read():
            with lock:
                barrier.wait()  # signal that we hold the lock
                time.sleep(0.3)  # hold it long enough for the other thread to time out

        def _read_status():
            barrier.wait()  # wait until the lock is held
            time.sleep(0.05)  # small delay to ensure lock contention
            cached_result[0] = model.status()

        holder = threading.Thread(target=_hold_lock_and_read, daemon=True)
        reader = threading.Thread(target=_read_status, daemon=True)
        holder.start()
        reader.start()
        holder.join(timeout=5.0)
        reader.join(timeout=5.0)
        self.assertIsNotNone(cached_result[0])
        self.assertEqual(cached_result[0]["checkpoint_path"], first["checkpoint_path"])

    def test_terminus_status_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, terminus_status() returns cached data instead of blocking."""
        model, _, lock, _ = _build_read_model()
        first = model.terminus_status()
        barrier = threading.Barrier(2, timeout=5.0)
        cached_result: dict[str, Any] | None = [None]

        def _hold_lock_and_read():
            with lock:
                barrier.wait()
                time.sleep(0.3)

        def _read_status():
            barrier.wait()
            time.sleep(0.05)
            cached_result[0] = model.terminus_status()

        holder = threading.Thread(target=_hold_lock_and_read, daemon=True)
        reader = threading.Thread(target=_read_status, daemon=True)
        holder.start()
        reader.start()
        holder.join(timeout=5.0)
        reader.join(timeout=5.0)
        self.assertIsNotNone(cached_result[0])
        self.assertEqual(cached_result[0]["token_count"], first["token_count"])


class StatusReadModelReadonlyTests(unittest.TestCase):
    """The StatusReadModel is read-only: it does not mutate runtime state."""

    def test_status_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        rev_before = runtime_state.state_revision
        model.status()
        model.terminus_status()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_status_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        self.assertFalse(runtime_state.dirty_state)
        model.status()
        model.terminus_status()
        self.assertFalse(runtime_state.dirty_state)


class StatusReadModelTelemetryTests(unittest.TestCase):
    """StatusReadModel.telemetry_snapshot() produces valid snapshots with correct payload keys."""

    def test_telemetry_snapshot_returns_checkpoint_path(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertEqual(result["checkpoint_path"], "/tmp/test.pt")

    def test_telemetry_snapshot_returns_token_count(self) -> None:
        model, trainer, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertEqual(result["token_count"], int(trainer.token_count))

    def test_telemetry_snapshot_returns_state_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("state_revision", result)
        self.assertEqual(result["state_revision"], runtime_state.state_revision)

    def test_telemetry_snapshot_returns_dirty_state(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("dirty_state", result)

    def test_telemetry_snapshot_returns_memory_fill_fraction(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("memory_fill_fraction", result)

    def test_telemetry_snapshot_returns_animation(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("animation", result)
        anim = result["animation"]
        self.assertIn("n_columns", anim)

    def test_telemetry_snapshot_returns_neurotransmitters(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        for key in ("dopamine", "serotonin", "acetylcholine", "norepinephrine"):
            self.assertIn(key, result)

    def test_telemetry_snapshot_returns_drift(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("drift", result)
        self.assertIn("drift_floor", result)

    def test_telemetry_snapshot_returns_terminus_runtime(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("terminus_runtime", result)

    def test_telemetry_snapshot_returns_replay_dataset_summary(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("replay_dataset_summary", result)

    def test_telemetry_snapshot_returns_sleep_events(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        for key in ("sleep_events", "micro_sleep_events", "deep_sleep_events"):
            self.assertIn(key, result)

    def test_telemetry_snapshot_returns_cross_modal_confidence(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("cross_modal_visual_confidence", result)
        self.assertIn("cross_modal_audio_confidence", result)

    def test_telemetry_snapshot_returns_grounding_confidence(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("grounding_confidence", result)

    def test_telemetry_snapshot_returns_trace_fields(self) -> None:
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        self.assertIn("trace_history_size", result)


class StatusReadModelTelemetryCacheTests(unittest.TestCase):
    """Telemetry revision-keyed cache reuse and lock-contention fallback."""

    def test_telemetry_snapshot_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, telemetry_snapshot() returns cached data."""
        model, _, lock, _ = _build_read_model()
        # First call populates cache
        first = model.telemetry_snapshot()
        # Hold the lock in another thread and verify we get the cached result
        barrier = threading.Barrier(2, timeout=5.0)
        cached_result: dict[str, Any] | None = [None]

        def _hold_lock_and_read():
            with lock:
                barrier.wait()  # signal that we hold the lock
                time.sleep(0.3)  # hold it long enough for the other thread to time out

        def _read_telemetry():
            barrier.wait()  # wait until the lock is held
            time.sleep(0.05)  # small delay to ensure lock contention
            cached_result[0] = model.telemetry_snapshot()

        holder = threading.Thread(target=_hold_lock_and_read, daemon=True)
        reader = threading.Thread(target=_read_telemetry, daemon=True)
        holder.start()
        reader.start()
        holder.join(timeout=5.0)
        reader.join(timeout=5.0)
        self.assertIsNotNone(cached_result[0])
        self.assertEqual(cached_result[0]["checkpoint_path"], first["checkpoint_path"])

    def test_telemetry_snapshot_reuses_cache_at_same_revision_when_cortex_inactive(self) -> None:
        """When cortex is inactive and revision is the same, telemetry returns the cached snapshot."""
        call_count = 0
        brain_snapshot = _build_brain_snapshot()

        def counting_brain_fn() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return deepcopy(brain_snapshot)

        cfg = _build_config()
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
        lock = threading.RLock()
        runtime_state = RuntimeState(lock=lock)
        animation_snapshot = _build_animation_snapshot()
        model = StatusReadModel(
            lock=lock,
            runtime_state=runtime_state,
            trainer=trainer,
            trace_history=deque(maxlen=200),
            metadata={},
            checkpoint_path_str="/tmp/test.pt",
            trace_dir_str="/tmp/traces",
            concept_store_snapshot_fn=lambda: deepcopy({"top_concepts": [], "total_concepts": 0}),
            brain_runtime_snapshot_fn=counting_brain_fn,
            cortex_active_fn=lambda: False,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        # First call populates cache
        first = model.telemetry_snapshot()
        first_call_count = call_count
        # Second call at the same revision should reuse cache
        second = model.telemetry_snapshot()
        self.assertIs(second, first)
        # The brain runtime snapshot function should not have been called again
        self.assertEqual(call_count, first_call_count)

    def test_telemetry_snapshot_rebuilds_on_revision_change_when_cortex_inactive(self) -> None:
        """When cortex is inactive but revision changes, telemetry rebuilds."""
        cfg = _build_config()
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
        lock = threading.RLock()
        runtime_state = RuntimeState(lock=lock)
        brain_snapshot = _build_brain_snapshot()
        animation_snapshot = _build_animation_snapshot()
        model = StatusReadModel(
            lock=lock,
            runtime_state=runtime_state,
            trainer=trainer,
            trace_history=deque(maxlen=200),
            metadata={},
            checkpoint_path_str="/tmp/test.pt",
            trace_dir_str="/tmp/traces",
            concept_store_snapshot_fn=lambda: deepcopy({"top_concepts": [], "total_concepts": 0}),
            brain_runtime_snapshot_fn=lambda: deepcopy(brain_snapshot),
            cortex_active_fn=lambda: False,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        # First call
        first = model.telemetry_snapshot()
        rev_before = first["state_revision"]
        # Advance the revision
        with lock:
            runtime_state.mark_mutated()
        # Second call should rebuild
        second = model.telemetry_snapshot()
        self.assertIsNot(second, first)
        self.assertNotEqual(second["state_revision"], rev_before)


class StatusReadModelTelemetryReadonlyTests(unittest.TestCase):
    """telemetry_snapshot() is read-only: it does not mutate runtime state."""

    def test_telemetry_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        rev_before = runtime_state.state_revision
        model.telemetry_snapshot()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_telemetry_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        self.assertFalse(runtime_state.dirty_state)
        model.telemetry_snapshot()
        self.assertFalse(runtime_state.dirty_state)


class ServiceManagerDelegationTests(unittest.TestCase):
    """Service Manager delegates status() and terminus_status() to StatusReadModel."""

    def test_manager_status_delegates_to_read_model(self) -> None:
        """The manager's status() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                self.assertIsNotNone(manager._status_read_model)
                self.assertIsInstance(manager._status_read_model, StatusReadModel)
                # Calling status() through the manager should work
                result = manager.status()
                self.assertIn("runtime_truth", result)
                self.assertIn("checkpoint_path", result)
            finally:
                manager.close()

    def test_manager_terminus_status_delegates_to_read_model(self) -> None:
        """The manager's terminus_status() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.terminus_status()
                self.assertIn("runtime_truth", result)
                self.assertIn("terminus_runtime", result)
                # Multimodal is now populated through the read model seam
                self.assertIn("multimodal", result)
                self.assertIn("enabled", result["multimodal"])
            finally:
                manager.close()

    def test_manager_terminus_status_multimodal_preserved_through_read_model(self) -> None:
        """terminus_status() preserves the full multimodal payload through the read model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "multimodal_preserved"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.terminus_status()
                multimodal = result["multimodal"]
                # The multimodal payload must have the same keys as the mixin's
                # _multimodal_runtime_summary_locked produces
                expected_keys = {
                    "enabled", "mode", "episodes_completed",
                    "focus_terms", "source_names",
                }
                self.assertTrue(
                    expected_keys.issubset(set(multimodal.keys())),
                    f"Missing keys: {expected_keys - set(multimodal.keys())}",
                )
            finally:
                manager.close()

    def test_manager_telemetry_snapshot_delegates_to_read_model(self) -> None:
        """The manager's telemetry_snapshot() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager

            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "telemetry_delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.telemetry_snapshot()
                self.assertIn("animation", result)
                self.assertIn("terminus_runtime", result)
                self.assertIn("token_count", result)
                self.assertIn("state_revision", result)
            finally:
                manager.close()
