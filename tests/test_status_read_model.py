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


def _build_read_model() -> tuple[StatusReadModel, HECSNTrainer, threading.RLock, RuntimeState]:
    cfg = _build_config()
    trainer = HECSNTrainer(HECSNModel(cfg), cfg)
    lock = threading.RLock()
    runtime_state = RuntimeState(lock=lock)
    brain_snapshot = _build_brain_snapshot()
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
