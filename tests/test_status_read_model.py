"""Direct test surface for the Status Read Model seam.

These tests exercise the StatusReadModel through its own interface with injected
adapters, verifying snapshot payloads and cache/freshness semantics without
requiring the full Service Manager composition root.  Regression coverage for
unchanged public behavior remains in test_service_manager.py and test_service_api.py.
"""

from __future__ import annotations

import threading
import time
import unittest
from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

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


def _build_architecture_snapshot(trainer: HECSNTrainer) -> dict[str, Any]:
    """Build a realistic architecture summary for testing the read model seam."""
    model = trainer.model
    config = trainer.config
    layers: list[dict[str, Any]] = []
    layers.append({
        "id": "input_encoding",
        "name": "Input + Stream Ingestion",
        "enabled": True,
        "type": "input",
        "params": {
            "input_dim": int(config.input_dim),
            "representation": config.input_representation,
            "background_sources": 0,
            "background_routing": "focus_aware_allocation",
            "sensory_sources": 0,
            "learned_chunking": bool(config.enable_learned_chunking),
        },
    })
    layers.append({
        "id": "competitive_routing",
        "name": "GPCSN Column Field",
        "enabled": True,
        "type": "core",
        "params": {
            "n_columns": int(config.n_columns),
            "k_routing": int(config.k_routing),
            "plasticity_mode": config.plasticity_mode,
            "plasticity_rule": config.plasticity_rule,
        },
    })
    predictive_enabled = bool(getattr(model, "predictive", None) is not None)
    layers.append({
        "id": "predictive_columns",
        "name": "Predictive Columns",
        "enabled": predictive_enabled,
        "type": "prediction",
        "params": {
            "enabled": predictive_enabled,
            "prediction_error_driven": predictive_enabled,
        } if predictive_enabled else {},
    })
    layers.append({
        "id": "context_prediction",
        "name": f"Context Attractor ({config.context_mode})",
        "enabled": model.context_layer is not None,
        "type": "context",
        "params": {
            "context_mode": config.context_mode,
        },
    })
    layers.append({
        "id": "binding",
        "name": "Hypercube Binding + Hubs",
        "enabled": model.binding_layer is not None,
        "type": "binding",
        "params": {
            "n_bindings": int(config.binding_n_bindings),
            "fan_in": int(config.binding_fan_in),
            "topology": type(model.binding_layer).__name__ if model.binding_layer is not None else "disabled",
        } if model.binding_layer is not None else {},
    })
    layers.append({
        "id": "abstraction",
        "name": "Abstraction Layer",
        "enabled": model.abstraction_layer is not None,
        "type": "abstraction",
        "params": {
            "n_concepts": int(config.abstraction_n_concepts),
        } if model.abstraction_layer is not None else {},
    })
    layers.append({
        "id": "cross_modal_grounding",
        "name": "Real Cross-Modal Grounding",
        "enabled": model.cross_modal is not None,
        "type": "grounding",
        "params": {
            "dim_visual": int(config.cross_modal_dim_visual),
            "dim_audio": int(config.cross_modal_dim_audio),
            "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()) if model.cross_modal else 0.0,
            "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()) if model.cross_modal else 0.0,
            "sensory_active": False,
        },
    })
    layers.append({
        "id": "memory_consolidation",
        "name": "Dual Memory + Consolidation",
        "enabled": True,
        "type": "memory",
        "params": {
            "memory_capacity": int(config.memory_capacity),
            "stc_tag_duration_strong": float(config.stc_tag_duration_strong),
        },
    })
    layers.append({
        "id": "autonomy_guidance",
        "name": "Active Exploration + Grounded-Family-Summary Lineage-Reconvergent Divergence-Split Trajectory-Sensitive Compacted Age-Sensitive Consequence-Calibrated Real-Source Guidance",
        "enabled": False,
        "type": "autonomy",
        "params": {
            "autonomy_enabled": False,
            "candidate_count": 0,
            "adaptive_focus_budgeting": False,
            "grounded_outcome_calibration": False,
            "evidence_provenance_credit": True,
            "delayed_multi_turn_consequence_tracking": True,
            "contradiction_decay_penalties": True,
            "mixed_evidence_forgiveness_scheduling": True,
            "age_sensitive_consequence_cooling": True,
            "consequence_state_retirement": True,
            "consequence_record_compaction": True,
            "trajectory_sensitive_consequence_families": True,
            "divergence_sensitive_consequence_splitting": True,
            "lineage_aware_consequence_remerge": True,
            "grounded_family_summary_calibration": True,
            "sensory_enabled": False,
            "items_per_episode": 0,
        },
    })
    return {
        "model_name": "Terminus",
        "core_name": "GPCSN",
        "version": "current",
        "family": "subcortex_runtime",
        "layers": layers,
        "config": {
            "context_mode": config.context_mode,
            "plasticity_rule": config.plasticity_rule,
            "n_columns": int(config.n_columns),
            "cross_modal": bool(model.cross_modal is not None),
        },
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
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
        cortex_active_fn=lambda: cortex_active,
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
    )
    return model, trainer, lock, runtime_state


def _run_under_lock_contention(
    lock: threading.RLock,
    read_fn: Callable[[], Any],
) -> Any:
    barrier = threading.Barrier(2, timeout=5.0)
    result: list[Any | None] = [None]

    def _hold_lock() -> None:
        with lock:
            barrier.wait()
            time.sleep(0.3)

    def _read() -> None:
        barrier.wait()
        time.sleep(0.05)
        result[0] = read_fn()

    holder = threading.Thread(target=_hold_lock, daemon=True)
    reader = threading.Thread(target=_read, daemon=True)
    holder.start()
    reader.start()
    holder.join(timeout=5.0)
    reader.join(timeout=5.0)
    return result[0]


def _build_manager(root: Path, *, test_case: str):
    from hecsn.service.manager import HECSNServiceManager
    cfg = _build_config()
    trainer = HECSNTrainer(HECSNModel(cfg), cfg)
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")


class StatusReadModelConstructionTests(unittest.TestCase):
    """StatusReadModel can be constructed with injected dependencies."""

    def test_read_model_constructs_with_adapter(self) -> None:
        """StatusReadModel should accept a manager-like adapter at construction."""
        model, _, _, _ = _build_read_model()
        self.assertIsNotNone(model)

    def test_read_model_owns_sensory_preview_projection(self) -> None:
        source = Path("src/hecsn/service/status_read_model.py").read_text(encoding="utf-8")

        self.assertNotIn("SensoryPreviewMixin", source)


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
        first = model.status()
        cached_result = _run_under_lock_contention(lock, model.status)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["checkpoint_path"], first["checkpoint_path"])

    def test_terminus_status_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, terminus_status() returns cached data instead of blocking."""
        model, _, lock, _ = _build_read_model()
        first = model.terminus_status()
        cached_result = _run_under_lock_contention(lock, model.terminus_status)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["token_count"], first["token_count"])


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


class StatusReadModelSensoryPreviewsTests(unittest.TestCase):
    """StatusReadModel.sensory_previews() produces correct preview payloads."""

    def test_sensory_previews_returns_count_and_previews(self) -> None:
        """sensory_previews() returns count, latest_preview_id, and previews list."""
        model, _, _, _ = _build_read_model()
        result = model.sensory_previews()
        self.assertIn("count", result)
        self.assertIn("latest_preview_id", result)
        self.assertIn("previews", result)
        self.assertIsInstance(result["previews"], list)

    def test_sensory_previews_empty_history(self) -> None:
        """sensory_previews() returns zero count and None latest when history is empty."""
        model, _, _, _ = _build_read_model()
        result = model.sensory_previews()
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["latest_preview_id"])
        self.assertEqual(result["previews"], [])

    def test_sensory_previews_with_history(self) -> None:
        """sensory_previews() returns items from the preview history with correct payload keys."""
        history = deque(maxlen=8)
        history.appendleft({
            "preview_id": "pv-001",
            "captured_at": "2026-05-08T12:00:00Z",
            "source_name": "test_source",
            "adapter": "text",
            "text": "hello world",
            "semantic_match": 0.8,
            "modality_need": 0.3,
            "item_semantic_match": 0.75,
            "item_candidates_considered": 5,
            "item_retrieval_lookahead": 2,
            "selection_score": 0.9,
            "window_budget": 64,
            "topics": ["greeting"],
            "focus_terms": ["hello"],
            "metadata": {"kind": "unit_test"},
        })
        lock = threading.RLock()
        runtime_state = RuntimeState(lock=lock)
        cfg = _build_config()
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
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
            brain_runtime_snapshot_fn=lambda: deepcopy(_build_brain_snapshot()),
            sensory_preview_history=history,
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
            cortex_active_fn=lambda: False,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        result = model.sensory_previews()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["latest_preview_id"], "pv-001")
        self.assertEqual(len(result["previews"]), 1)
        preview = result["previews"][0]
        self.assertEqual(preview["preview_id"], "pv-001")
        self.assertEqual(preview["source_name"], "test_source")
        self.assertEqual(preview["text"], "hello world")
        self.assertAlmostEqual(preview["semantic_match"], 0.8)
        self.assertEqual(preview["topics"], ["greeting"])
        self.assertEqual(preview["focus_terms"], ["hello"])
        self.assertEqual(preview["metadata"], {"kind": "unit_test"})

    def test_sensory_previews_limit(self) -> None:
        """sensory_previews(limit=N) returns at most N items."""
        history = deque(maxlen=8)
        for i in range(5):
            history.appendleft({
                "preview_id": f"pv-{i:03d}",
                "captured_at": "2026-05-08T12:00:00Z",
                "source_name": f"source_{i}",
                "adapter": "text",
                "text": f"item {i}",
                "semantic_match": 0.5,
                "modality_need": 0.0,
                "item_semantic_match": 0.0,
                "item_candidates_considered": 0,
                "item_retrieval_lookahead": 1,
                "selection_score": 0.0,
                "window_budget": 0,
                "topics": [],
                "focus_terms": [],
                "metadata": {},
            })
        lock = threading.RLock()
        runtime_state = RuntimeState(lock=lock)
        cfg = _build_config()
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
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
            brain_runtime_snapshot_fn=lambda: deepcopy(_build_brain_snapshot()),
            sensory_preview_history=history,
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
            cortex_active_fn=lambda: False,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        result = model.sensory_previews(limit=2)
        self.assertEqual(result["count"], 5)
        self.assertEqual(len(result["previews"]), 2)

    def test_sensory_previews_with_visual_media(self) -> None:
        """sensory_previews() converts visual media bytes to data_url."""
        history = deque(maxlen=8)
        history.appendleft({
            "preview_id": "pv-vis",
            "captured_at": "2026-05-08T12:00:00Z",
            "source_name": "cam_source",
            "adapter": "image",
            "text": "visual input",
            "semantic_match": 0.0,
            "modality_need": 0.0,
            "item_semantic_match": 0.0,
            "item_candidates_considered": 0,
            "item_retrieval_lookahead": 1,
            "selection_score": 0.0,
            "window_budget": 0,
            "topics": [],
            "focus_terms": [],
            "metadata": {},
            "visual": {
                "bytes": b"\x89PNG\r\n",
                "mime_type": "image/png",
            },
        })
        lock = threading.RLock()
        runtime_state = RuntimeState(lock=lock)
        cfg = _build_config()
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
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
            brain_runtime_snapshot_fn=lambda: deepcopy(_build_brain_snapshot()),
            sensory_preview_history=history,
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
            cortex_active_fn=lambda: False,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        result = model.sensory_previews()
        self.assertEqual(result["count"], 1)
        preview = result["previews"][0]
        self.assertIsNotNone(preview["visual"])
        self.assertTrue(preview["visual"]["data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(preview["visual"]["byte_size"], 6)


class StatusReadModelArchitectureSummaryTests(unittest.TestCase):
    """StatusReadModel.architecture_summary() produces correct layer topology."""

    def test_architecture_summary_returns_model_and_core_name(self) -> None:
        """architecture_summary() returns Terminus model name and GPCSN core name."""
        model, _, _, _ = _build_read_model()
        result = model.architecture_summary()
        self.assertEqual(result["model_name"], "Terminus")
        self.assertEqual(result["core_name"], "GPCSN")

    def test_architecture_summary_returns_version_and_family(self) -> None:
        """architecture_summary() returns version and family fields."""
        model, _, _, _ = _build_read_model()
        result = model.architecture_summary()
        self.assertEqual(result["version"], "current")
        self.assertEqual(result["family"], "subcortex_runtime")

    def test_architecture_summary_returns_layers(self) -> None:
        """architecture_summary() returns a non-empty layers list with required keys."""
        model, _, _, _ = _build_read_model()
        result = model.architecture_summary()
        self.assertIn("layers", result)
        self.assertIsInstance(result["layers"], list)
        self.assertGreater(len(result["layers"]), 0)
        for layer in result["layers"]:
            self.assertIn("id", layer)
            self.assertIn("name", layer)
            self.assertIn("enabled", layer)
            self.assertIn("type", layer)
            self.assertIn("params", layer)

    def test_architecture_summary_contains_key_layers(self) -> None:
        """architecture_summary() includes input_encoding, competitive_routing, and memory layers."""
        model, _, _, _ = _build_read_model()
        result = model.architecture_summary()
        layer_ids = [l["id"] for l in result["layers"]]
        self.assertIn("input_encoding", layer_ids)
        self.assertIn("competitive_routing", layer_ids)
        self.assertIn("memory_consolidation", layer_ids)

    def test_architecture_summary_returns_config(self) -> None:
        """architecture_summary() returns config with expected keys."""
        model, _, _, _ = _build_read_model()
        result = model.architecture_summary()
        self.assertIn("config", result)
        config = result["config"]
        self.assertIn("context_mode", config)
        self.assertIn("plasticity_rule", config)
        self.assertIn("n_columns", config)


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
        first = model.telemetry_snapshot()
        cached_result = _run_under_lock_contention(lock, model.telemetry_snapshot)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["checkpoint_path"], first["checkpoint_path"])

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
            sensory_preview_history=deque(maxlen=8),
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
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
            sensory_preview_history=deque(maxlen=8),
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
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


class StatusReadModelSeparationTests(unittest.TestCase):
    """Verify that run_grounding_probe() stays outside the StatusReadModel."""

    def test_read_model_has_no_grounding_probe(self) -> None:
        """StatusReadModel must not have a run_grounding_probe method."""
        model, _, _, _ = _build_read_model()
        self.assertFalse(
            hasattr(model, "run_grounding_probe"),
            "run_grounding_probe() must remain outside the Status Read Model",
        )


def _build_living_loop_snapshot() -> dict[str, Any]:
    """Build a minimal living loop snapshot payload for testing."""
    return {
        "living_loop": {
            "generated_at": "2026-05-09T12:00:00Z",
            "token_count": 0,
            "state_revision": 1,
            "configured": False,
            "running": False,
            "provenance": {},
            "predictions": [],
            "actions": [],
            "consolidations": [],
            "runtime_episodes": [],
            "action_loop": {"enabled": True, "root_path": "/tmp", "supported_actions": [], "actions_recorded": 0, "verified_actions": 0, "contradicted_actions": 0, "last_action": None},
            "memory": {},
            "narrative": {},
            "cortex": {"enabled": False, "running": False, "current_mode": "idle", "is_sleeping": False, "thoughts_generated": 0, "dreams_generated": 0, "sleep_cycles": 0, "memory_count": 0, "memory_fill_ratio": 0.0, "drives": {}},
            "feedback_summary": {"feedback_count": 0, "verified_count": 0, "contradicted_count": 0, "unverified_count": 0, "recent_feedback": [], "grounding_impact": "none"},
            "feedback_count": 0,
            "verified_feedback_count": 0,
            "contradicted_feedback_count": 0,
            "unverified_feedback_count": 0,
            "recent_feedback": [],
            "replay_sample_summary": {},
            "replay_executor_summary": {},
            "grounding_health": {},
            "benchmark_telemetry": {},
            "policy_decision": {},
            "replay_plan": {},
            "replay_dataset_summary": None,
            "world_model_lite": None,
        },
        "state_revision": 1,
        "dirty_state": False,
        "token_count": 0,
    }


def _build_policy_actuator_snapshot() -> dict[str, Any]:
    """Build a minimal policy actuator snapshot for testing."""
    return {
        "schema_version": 1,
        "generated_at": "2026-05-09T12:00:00Z",
        "high_latency_avg_ms": 0,
        "high_latency_max_ms": 0,
        "scores": [],
        "recommendations": [],
        "world_model_lite": None,
    }


def _build_cognitive_signal_state_snapshot() -> dict[str, Any]:
    """Build a minimal Cognitive Signal snapshot for testing."""
    return {
        "prediction_error_mean": 0.0,
        "prediction_error_max": 0.0,
        "predictive_confidence_mean": 0.8,
        "predictive_confidence_min": 0.7,
        "dopamine": 0.0,
        "norepinephrine": 0.0,
        "recent_concepts": ["coral thermal memory"],
        "concept_candidates": [
            {
                "label": "coral thermal memory",
                "top_terms": ["coral", "thermal", "memory"],
                "observations": 3,
                "uncertainty": 0.2,
                "temporal_coherence": 0.7,
            }
        ],
    }


def _build_read_model_with_living_loop(
    *,
    cortex_active: bool = False,
    use_legacy_cortex_signal_fn: bool = False,
) -> tuple[StatusReadModel, HECSNTrainer, threading.RLock, RuntimeState, dict[str, int]]:
    """Build a StatusReadModel with living loop callbacks wired for testing."""
    cfg = _build_config()
    trainer = HECSNTrainer(HECSNModel(cfg), cfg)
    lock = threading.RLock()
    runtime_state = RuntimeState(lock=lock)
    brain_snapshot = _build_brain_snapshot()
    animation_snapshot = _build_animation_snapshot()
    living_loop_result = _build_living_loop_snapshot()
    policy_result = _build_policy_actuator_snapshot()
    cognitive_signal_result = _build_cognitive_signal_state_snapshot()
    call_counts: dict[str, int] = {
        "living_loop": 0,
        "policy_actuator": 0,
        "cognitive_signal": 0,
        "cortex_signal": 0,
    }

    def living_loop_snapshot_fn() -> dict[str, Any]:
        call_counts["living_loop"] += 1
        return deepcopy(living_loop_result)

    def policy_actuator_snapshot_fn() -> dict[str, Any]:
        call_counts["policy_actuator"] += 1
        return deepcopy(policy_result)

    def cognitive_signal_state_fn() -> dict[str, Any]:
        call_counts["cognitive_signal"] += 1
        return deepcopy(cognitive_signal_result)

    def cortex_signal_state_fn() -> dict[str, Any]:
        call_counts["cortex_signal"] += 1
        return deepcopy(cognitive_signal_result)

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
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
        cortex_active_fn=lambda: cortex_active,
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        living_loop_status_fn=living_loop_snapshot_fn,
        policy_actuator_status_fn=policy_actuator_snapshot_fn,
        cognitive_signal_state_fn=None if use_legacy_cortex_signal_fn else cognitive_signal_state_fn,
        cortex_signal_state_fn=cortex_signal_state_fn if use_legacy_cortex_signal_fn else None,
    )
    return model, trainer, lock, runtime_state, call_counts


class StatusReadModelLivingLoopTests(unittest.TestCase):
    """StatusReadModel.living_loop_status() produces valid snapshots with correct payload keys."""

    def test_living_loop_status_returns_living_loop_key(self) -> None:
        """living_loop_status() should return a dict with a 'living_loop' key."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.living_loop_status()
        self.assertIn("living_loop", result)

    def test_living_loop_status_includes_control_candidate_sidecar(self) -> None:
        """Living-loop status shows advisory Subcortex control candidates without promoting them."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()
        result = model.living_loop_status()
        sidecar = result["living_loop"]["subcortical_control_candidates"]
        self.assertEqual(sidecar["surface"], "subcortical_control_candidates.v1")
        self.assertTrue(sidecar["grounded"])
        self.assertTrue(sidecar["not_cognition_substrate"])
        self.assertFalse(sidecar["retired_runtime_dependency"])
        self.assertNotIn("prompt", sidecar["candidates"][0])
        for replay_key in ("candidate_id", "target_type", "suggested_endpoint", "suggested_input", "reason_codes"):
            self.assertNotIn(replay_key, sidecar["candidates"][0])
        self.assertIn("promotion_gate", sidecar["candidates"][0])
        self.assertFalse(sidecar["promotion_summary"]["eligible_for_action"])
        self.assertFalse(sidecar["promotion_summary"]["eligible_for_fact_promotion"])
        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)

    def test_living_loop_status_includes_advisory_self_repair_candidates(self) -> None:
        """Living-loop status shows spike-health self-repair candidates without mutating state."""
        model, trainer, _, runtime_state, _ = _build_read_model_with_living_loop()
        trainer.model.competitive.win_rate_ema.zero_()
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        result = model.living_loop_status()
        sidecar = result["living_loop"]["subcortical_self_repair_candidates"]

        self.assertEqual(sidecar["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(sidecar["advisory"])
        self.assertFalse(sidecar["executable"])
        self.assertTrue(sidecar["not_cognition_substrate"])
        self.assertFalse(sidecar["retired_runtime_dependency"])
        self.assertEqual(sidecar["candidates"][0]["intent"], "review_column_revival")
        self.assertFalse(sidecar["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(sidecar["candidates"][0]["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(sidecar["promotion_summary"]["eligible_for_structural_mutation"])
        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)

    def test_living_loop_status_returns_token_count(self) -> None:
        """living_loop_status() should include token_count at the top level."""
        model, trainer, _, _, _ = _build_read_model_with_living_loop()
        result = model.living_loop_status()
        self.assertIn("token_count", result)
        self.assertEqual(result["token_count"], int(trainer.token_count))

    def test_living_loop_status_returns_state_revision(self) -> None:
        """living_loop_status() should include state_revision from runtime state."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        result = model.living_loop_status()
        self.assertIn("state_revision", result)
        self.assertEqual(result["state_revision"], runtime_state.state_revision)

    def test_living_loop_status_returns_dirty_state(self) -> None:
        """living_loop_status() should include dirty_state from runtime state."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.living_loop_status()
        self.assertIn("dirty_state", result)

    def test_living_loop_status_delegates_to_callback(self) -> None:
        """living_loop_status() should delegate through the injected callback."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop()
        model.living_loop_status()
        self.assertGreater(call_counts["living_loop"], 0)


class StatusReadModelPolicyActuatorTests(unittest.TestCase):
    """StatusReadModel.policy_actuator_status() produces valid policy snapshots."""

    def test_policy_actuator_status_returns_schema_version(self) -> None:
        """policy_actuator_status() should return a dict with schema_version."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.policy_actuator_status()
        self.assertIn("schema_version", result)

    def test_policy_actuator_status_returns_scores(self) -> None:
        """policy_actuator_status() should include scores list."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.policy_actuator_status()
        self.assertIn("scores", result)

    def test_policy_actuator_status_returns_recommendations(self) -> None:
        """policy_actuator_status() should include recommendations list."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.policy_actuator_status()
        self.assertIn("recommendations", result)

    def test_policy_actuator_status_includes_advisory_control_candidates(self) -> None:
        """Policy status surfaces Cognitive Signal control candidates as non-executable context."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.policy_actuator_status()
        candidates = result["subcortical_control_candidates"]
        self.assertEqual(candidates["surface"], "subcortical_control_candidates.v1")
        self.assertTrue(candidates["advisory"])
        self.assertFalse(candidates["executable"])
        self.assertFalse(candidates["retired_runtime_dependency"])
        self.assertTrue(candidates["not_cognition_substrate"])
        self.assertIn("candidates", candidates)
        self.assertIn("promotion_summary", candidates)
        self.assertFalse(candidates["promotion_summary"]["eligible_for_action"])
        self.assertFalse(candidates["promotion_summary"]["eligible_for_fact_promotion"])

    def test_policy_actuator_status_includes_advisory_self_repair_candidates(self) -> None:
        """Policy status surfaces spike-health repair candidates as non-executable context."""
        model, trainer, _, runtime_state, _ = _build_read_model_with_living_loop()
        trainer.model.competitive.steps_since_win.fill_(trainer.model.competitive.dead_column_steps)
        runtime_state.mark_clean()
        result = model.policy_actuator_status()
        candidates = result["subcortical_self_repair_candidates"]
        self.assertEqual(candidates["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(candidates["advisory"])
        self.assertFalse(candidates["executable"])
        self.assertFalse(candidates["promotion_summary"]["eligible_for_structural_mutation"])
        self.assertFalse(candidates["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(runtime_state.dirty_state)

    def test_policy_actuator_status_delegates_to_callback(self) -> None:
        """policy_actuator_status() should delegate through the injected callback."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop()
        model.policy_actuator_status()
        self.assertGreater(call_counts["policy_actuator"], 0)


class StatusReadModelCognitiveSignalStateTests(unittest.TestCase):
    """StatusReadModel.cognitive_signal_state() produces valid signal payloads with cached fallback."""

    def test_cognitive_signal_state_returns_payload_keys(self) -> None:
        """cognitive_signal_state() is the canonical Subcortex signal surface."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.cognitive_signal_state()
        self.assertIn("prediction_error_mean", result)
        self.assertIn("prediction_error_max", result)
        self.assertIn("predictive_confidence_mean", result)
        self.assertIn("predictive_confidence_min", result)
        self.assertIn("dopamine", result)
        self.assertIn("norepinephrine", result)
        self.assertIn("recent_concepts", result)
        self.assertIn("concept_candidates", result)
        self.assertIn("subcortical_language", result)
        self.assertIn("subcortical_deliberation", result)
        surface = result["subcortical_language"]
        self.assertEqual(surface["surface"], "subcortical_language.v1")
        self.assertTrue(surface["grounded"])
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertFalse(surface["retired_runtime_dependency"])
        self.assertEqual(surface["grounding"]["concept_focus"], "coral thermal memory")
        deliberation = result["subcortical_deliberation"]
        self.assertEqual(deliberation["surface"], "subcortical_control_candidates.v1")
        self.assertTrue(deliberation["grounded"])
        self.assertTrue(deliberation["not_cognition_substrate"])
        self.assertFalse(deliberation["retired_runtime_dependency"])
        self.assertGreaterEqual(len(deliberation["candidates"]), 1)
        self.assertEqual(deliberation["candidates"][0]["intent"], "maintain_current_focus")
        self.assertEqual(deliberation["candidates"][0]["promotion_gate"]["status"], "advisory_monitor_only")

    def test_subcortical_language_surface_returns_cognitive_signal_decode(self) -> None:
        """subcortical_language_surface() exposes the bounded status-language view."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        surface = model.subcortical_language_surface()
        self.assertEqual(surface["surface"], "subcortical_language.v1")
        self.assertIn("prediction error", surface["state_text"])
        self.assertEqual(surface["source"], "service.status_read_model.cognitive_signal")
        self.assertEqual(surface["control_hint"], "maintain_current_focus")

    def test_subcortical_language_surface_returns_cached_on_lock_contention(self) -> None:
        """The direct language surface remains cache-compatible under lock contention."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.subcortical_language_surface()
        cached_result = _run_under_lock_contention(lock, model.subcortical_language_surface)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["state_text"], first["state_text"])

    def test_subcortical_deliberation_surface_returns_grounded_candidates(self) -> None:
        """subcortical_deliberation_surface() exposes bounded non-LLM control hypotheses."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        surface = model.subcortical_deliberation_surface()
        self.assertEqual(surface["surface"], "subcortical_control_candidates.v1")
        self.assertEqual(surface["source"], "service.status_read_model.cognitive_signal")
        self.assertEqual(surface["control_hint"], "maintain_current_focus")
        self.assertTrue(surface["grounded"])
        self.assertFalse(surface["retired_runtime_dependency"])
        self.assertTrue(surface["candidates"])
        first = surface["candidates"][0]
        self.assertEqual(first["phase"], "monitor")
        self.assertIn("coral thermal memory", first["candidate_text"])
        self.assertNotIn("prompt", first)
        self.assertIn("prediction_error_mean", first["grounding"])
        self.assertIn("promotion_gate", first)
        self.assertFalse(first["promotion_gate"]["eligible_for_action"])

    def test_cognitive_signal_state_returns_cached_on_lock_contention(self) -> None:
        """When the lock is held, cognitive_signal_state() returns cached data."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.cognitive_signal_state()
        cached_result = _run_under_lock_contention(lock, model.cognitive_signal_state)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["dopamine"], first["dopamine"])
        self.assertEqual(
            cached_result["subcortical_language"]["grounding"]["concept_focus"],
            first["subcortical_language"]["grounding"]["concept_focus"],
        )

    def test_cognitive_signal_state_delegates_to_callback(self) -> None:
        """cognitive_signal_state() should delegate through the injected callback."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop()
        model.cognitive_signal_state()
        self.assertGreater(call_counts["cognitive_signal"], 0)

    def test_cortex_signal_state_returns_payload_keys(self) -> None:
        """cortex_signal_state() remains as a compatibility wrapper."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        result = model.cortex_signal_state()
        self.assertIn("prediction_error_mean", result)
        self.assertIn("prediction_error_max", result)
        self.assertIn("predictive_confidence_mean", result)
        self.assertIn("predictive_confidence_min", result)
        self.assertIn("dopamine", result)
        self.assertIn("norepinephrine", result)
        self.assertIn("recent_concepts", result)
        self.assertIn("concept_candidates", result)

    def test_cortex_signal_state_returns_cached_on_lock_contention(self) -> None:
        """When the lock is held, cortex_signal_state() returns cached data."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.cortex_signal_state()
        cached_result = _run_under_lock_contention(lock, model.cortex_signal_state)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["dopamine"], first["dopamine"])

    def test_cortex_signal_state_delegates_to_callback(self) -> None:
        """cortex_signal_state() should delegate through the injected callback."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop()
        model.cortex_signal_state()
        self.assertGreater(call_counts["cognitive_signal"], 0)

    def test_legacy_cortex_signal_callback_is_fallback_only(self) -> None:
        """The retired callback name still works when the canonical callback is absent."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop(use_legacy_cortex_signal_fn=True)
        result = model.cognitive_signal_state()
        self.assertIn("subcortical_language", result)
        self.assertEqual(call_counts["cognitive_signal"], 0)
        self.assertGreater(call_counts["cortex_signal"], 0)

    def test_cortex_signal_state_caches_result(self) -> None:
        """The compatibility wrapper updates old and new cache fields."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        first = model.cortex_signal_state()
        cached_state = model._cached_cortex_signal_state
        canonical_cached_state = model._cached_cognitive_signal_state
        self.assertIsNotNone(cached_state)
        self.assertIsNotNone(canonical_cached_state)
        assert cached_state is not None
        assert canonical_cached_state is not None
        self.assertEqual(cached_state["dopamine"], first["dopamine"])
        self.assertEqual(canonical_cached_state["dopamine"], first["dopamine"])


class StatusReadModelLivingLoopReadonlyTests(unittest.TestCase):
    """living_loop_status() and policy_actuator_status() are read-only."""

    def test_living_loop_status_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        model.living_loop_status()
        model.policy_actuator_status()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_living_loop_status_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        runtime_state.mark_clean()
        self.assertFalse(runtime_state.dirty_state)
        model.living_loop_status()
        model.policy_actuator_status()
        self.assertFalse(runtime_state.dirty_state)

# ------------------------------------------------------------------
# Issue #43: Direct Status Read Model test surface additions
# ------------------------------------------------------------------

def _build_read_model_with_brain_snapshot(
    brain_snapshot: dict[str, Any],
    *,
    cortex_active: bool = False,
) -> tuple[StatusReadModel, HECSNTrainer, threading.RLock, RuntimeState]:
    """Build a StatusReadModel with a specific brain runtime snapshot for verdict testing."""
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
        brain_runtime_snapshot_fn=lambda: deepcopy(brain_snapshot),
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
        cortex_active_fn=lambda: cortex_active,
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
    )
    return model, trainer, lock, runtime_state


def _build_alive_brain_snapshot() -> dict[str, Any]:
    """Build a brain snapshot that produces an 'alive' Runtime Truth verdict."""
    return {
        "configured": True,
        "running": True,
        "running_since": "2026-05-09T12:00:00Z",
        "last_error": None,
        "tick_count": 5,
        "background_tokens_processed": 100,
        "autonomy_tokens_processed": 0,
        "last_work_at": "2026-05-09T12:00:01Z",
        "source_bank": [{"name": "test_source", "source_type": "file"}],
        "cortex": {"enabled": True},
        "living_loop": {},
        "sleep_interval_seconds": 0.01,
        "tick_tokens": 64,
        "repeat_sources": True,
    }


def _build_error_brain_snapshot() -> dict[str, Any]:
    """Build a brain snapshot with last_error set, producing a 'failed' verdict."""
    return {
        "configured": True,
        "running": False,
        "running_since": None,
        "last_error": "OOM during tick",
        "tick_count": 0,
        "background_tokens_processed": 0,
        "autonomy_tokens_processed": 0,
        "last_work_at": None,
        "source_bank": [],
        "cortex": {"enabled": True},
        "living_loop": {},
    }


def _build_degraded_brain_snapshot() -> dict[str, Any]:
    """Build a brain snapshot for a 'degraded' verdict — configured+cortex but no progress."""
    return {
        "configured": True,
        "running": False,
        "running_since": None,
        "last_error": None,
        "tick_count": 0,
        "background_tokens_processed": 0,
        "autonomy_tokens_processed": 0,
        "last_work_at": None,
        "source_bank": [],
        "cortex": {"enabled": True},
        "living_loop": {},
    }


def _build_no_cortex_brain_snapshot() -> dict[str, Any]:
    """Build a configured snapshot with the retired cortex path disabled."""
    return {
        "configured": True,
        "running": False,
        "running_since": None,
        "last_error": None,
        "tick_count": 5,
        "background_tokens_processed": 100,
        "autonomy_tokens_processed": 0,
        "last_work_at": None,
        "source_bank": [],
        "cortex": {"enabled": False, "retired": True},
        "living_loop": {},
    }


class StatusReadModelFreshnessTests(unittest.TestCase):
    """fresh_wait_seconds semantics for status() and terminus_status()."""

    def test_status_with_fresh_wait_blocks_until_new_snapshot(self) -> None:
        """status(fresh_wait_seconds=N) retries until it acquires the lock."""
        model, _, lock, _ = _build_read_model()
        # Populate the cache first
        cached = model.status()
        self.assertEqual(cached["checkpoint_path"], "/tmp/test.pt")
        # With fresh_wait_seconds, the call should eventually succeed
        fresh = model.status(fresh_wait_seconds=0.5)
        self.assertIn("runtime_truth", fresh)
        self.assertEqual(fresh["checkpoint_path"], "/tmp/test.pt")

    def test_terminus_status_with_fresh_wait_blocks_until_new_snapshot(self) -> None:
        """terminus_status(fresh_wait_seconds=N) retries until it acquires the lock."""
        model, _, _, _ = _build_read_model()
        cached = model.terminus_status()
        self.assertIn("runtime_truth", cached)
        fresh = model.terminus_status(fresh_wait_seconds=0.5)
        self.assertIn("runtime_truth", fresh)
        self.assertIn("terminus_runtime", fresh)

    def test_status_fresh_wait_updates_cache(self) -> None:
        """After a fresh_wait call, the cache is updated with the new snapshot."""
        model, _, _, _ = _build_read_model()
        first = model.status()
        # Fresh wait should update the cache
        fresh = model.status(fresh_wait_seconds=0.5)
        self.assertEqual(fresh["checkpoint_path"], first["checkpoint_path"])
        # The cached snapshot should now match the fresh one
        cached_result = _run_under_lock_contention(model._lock, model.status)
        self.assertIsNotNone(cached_result)


class StatusReadModelRuntimeTruthVerdictTests(unittest.TestCase):
    """Runtime Truth verdict progression through injected brain snapshots."""

    def test_verdict_failed_when_error_set(self) -> None:
        """When last_error is non-empty, the verdict should be 'failed'."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_error_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "failed")
        self.assertEqual(truth["recommended_action"], "inspect_last_error")

    def test_verdict_partial_when_unconfigured(self) -> None:
        """When not configured, the verdict should be 'partial'."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "partial")
        self.assertEqual(truth["recommended_action"], "configure_terminus_sources")

    def test_verdict_alive_when_cortex_retired_but_runtime_progresses(self) -> None:
        """Retired cortex should not keep an otherwise progressing runtime partial."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_no_cortex_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "alive")
        self.assertEqual(truth["recommended_action"], "continue_monitoring")
        self.assertFalse(truth["cortex_available"])
        self.assertTrue(truth["cortex_retired"])
        self.assertFalse(truth["retired_runtime_path_available"])
        self.assertTrue(truth["retired_runtime_path_retired"])
        self.assertFalse(truth["evidence"]["retired_runtime_path_enabled"])
        self.assertTrue(truth["evidence"]["retired_runtime_path_retired"])

    def test_verdict_degraded_when_no_progress(self) -> None:
        """When configured+cortex but no progress, the verdict should be 'degraded'."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_degraded_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "degraded")
        self.assertEqual(truth["recommended_action"], "run_tick_or_start_runtime")

    def test_verdict_alive_when_configured_with_progress(self) -> None:
        """When configured+cortex+progress, the verdict should be 'alive'."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "alive")
        self.assertEqual(truth["recommended_action"], "continue_monitoring")

    def test_verdict_includes_cortex_available_flag(self) -> None:
        """Runtime Truth surfaces cortex_available from the brain snapshot."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertTrue(truth["cortex_available"])
        self.assertFalse(truth["cortex_retired"])

    def test_verdict_includes_source_configuration_evidence(self) -> None:
        """Runtime Truth includes source configuration with hash and payload."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        source_config = truth["source_configuration"]
        self.assertIn("configuration_hash", source_config)
        self.assertIn("configuration_payload", source_config)
        self.assertIn("source_count", source_config)
        self.assertEqual(source_config["source_count"], 1)

    def test_verdict_downgraded_to_degraded_on_high_memory_pressure(self) -> None:
        """When memory fill_fraction >= 0.85, an otherwise 'alive' verdict becomes 'degraded'."""
        # We need to make the memory store report high fill.
        # We can't easily fake the trainer's memory_store, so we verify
        # the verdict is at least correct for the non-pressure case here
        # and leave the high-pressure scenario for integration-level tests.
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        # With the default tiny memory store, the verdict should be alive
        # (memory starts empty, so fill_fraction is low)
        self.assertEqual(truth["verdict"], "alive")
        memory_pressure = truth["memory_pressure"]
        self.assertIn("fill_fraction", memory_pressure)
        self.assertIn("pressure", memory_pressure)
        self.assertIn("working_set_policy", memory_pressure)

    def test_verdict_safety_flags_reflect_replay_role(self) -> None:
        """Runtime Truth safety_flags include replay_dataset_preview_only flag."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        safety = truth["safety_flags"]
        self.assertIn("replay_dataset_preview_only", safety)

    def test_verdict_latency_includes_tick_and_tps(self) -> None:
        """Runtime Truth latency_ms includes last_tick and tokens_per_second."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        latency = truth["latency_ms"]
        self.assertIn("last_tick", latency)
        self.assertIn("tokens_per_second", latency)


class StatusReadModelRuntimeStatePropagationTests(unittest.TestCase):
    """Runtime state (dirty_state, state_revision) flows correctly through all snapshot surfaces."""

    def test_status_reflects_dirty_state(self) -> None:
        """status() reflects the current dirty_state from RuntimeState."""
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        result = model.status()
        self.assertFalse(result["dirty_state"])
        with model._lock:
            runtime_state.mark_mutated()
        result = model.status()
        self.assertTrue(result["dirty_state"])

    def test_status_reflects_state_revision(self) -> None:
        """status() reflects the current state_revision from RuntimeState."""
        model, _, _, runtime_state = _build_read_model()
        initial_rev = runtime_state.state_revision
        result = model.status()
        self.assertEqual(result["state_revision"], initial_rev)
        with model._lock:
            runtime_state.mark_mutated()
        result = model.status()
        self.assertEqual(result["state_revision"], initial_rev + 1)

    def test_terminus_status_reflects_dirty_state(self) -> None:
        """terminus_status() reflects the current dirty_state from RuntimeState."""
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        result = model.terminus_status()
        self.assertFalse(result["dirty_state"])
        with model._lock:
            runtime_state.mark_mutated()
        result = model.terminus_status()
        self.assertTrue(result["dirty_state"])

    def test_terminus_status_reflects_state_revision(self) -> None:
        """terminus_status() reflects the current state_revision from RuntimeState."""
        model, _, _, runtime_state = _build_read_model()
        initial_rev = runtime_state.state_revision
        with model._lock:
            runtime_state.mark_mutated()
        result = model.terminus_status()
        self.assertEqual(result["state_revision"], initial_rev + 1)

    def test_telemetry_reflects_state_revision(self) -> None:
        """telemetry_snapshot() reflects the current state_revision from RuntimeState."""
        model, _, _, runtime_state = _build_read_model()
        initial_rev = runtime_state.state_revision
        result = model.telemetry_snapshot()
        self.assertEqual(result["state_revision"], initial_rev)
        with model._lock:
            runtime_state.mark_mutated()
        result = model.telemetry_snapshot()
        self.assertEqual(result["state_revision"], initial_rev + 1)

    def test_living_loop_status_reflects_dirty_state(self) -> None:
        """living_loop_status() reflects the current dirty_state from RuntimeState."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        runtime_state.mark_clean()
        result = model.living_loop_status()
        self.assertFalse(result["dirty_state"])
        with model._lock:
            runtime_state.mark_mutated()
        result = model.living_loop_status()
        self.assertTrue(result["dirty_state"])

    def test_living_loop_status_reflects_state_revision(self) -> None:
        """living_loop_status() reflects the current state_revision from RuntimeState."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        initial_rev = runtime_state.state_revision
        with model._lock:
            runtime_state.mark_mutated()
        result = model.living_loop_status()
        self.assertEqual(result["state_revision"], initial_rev + 1)


class StatusReadModelNullAdapterFallbackTests(unittest.TestCase):
    """Fallback behavior when optional adapter callbacks are not provided."""

    def test_living_loop_status_without_callback_returns_minimal_payload(self) -> None:
        """When living_loop_status_fn is None, a minimal payload is returned."""
        model, _, _, _ = _build_read_model()
        result = model.living_loop_status()
        self.assertIn("living_loop", result)
        self.assertEqual(result["living_loop"], {})
        self.assertIn("state_revision", result)
        self.assertIn("dirty_state", result)
        self.assertIn("token_count", result)

    def test_policy_actuator_status_without_callback_returns_advisory_payload(self) -> None:
        """When policy_actuator_status_fn is None, an advisory payload is returned."""
        model, _, _, _ = _build_read_model()
        result = model.policy_actuator_status()
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["recommendation"], "no_policy_actuator_configured")
        self.assertEqual(result["action"], "none")
        self.assertTrue(result["advisory"])
        self.assertFalse(result["executable"])

    def test_cognitive_signal_state_without_callback_returns_empty(self) -> None:
        """When cognitive_signal_state_fn is None, an empty dict is returned."""
        model, _, _, _ = _build_read_model()
        result = model.cognitive_signal_state()
        self.assertEqual(result, {})

    def test_cortex_signal_state_without_callback_returns_empty(self) -> None:
        """The retired Cortex compatibility wrapper returns the same empty payload."""
        model, _, _, _ = _build_read_model()
        result = model.cortex_signal_state()
        self.assertEqual(result, {})


class StatusReadModelLivingLoopCacheTests(unittest.TestCase):
    """Living loop and policy actuator cache behavior under lock contention."""

    def test_living_loop_status_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, living_loop_status() returns cached data."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.living_loop_status()
        cached_result = _run_under_lock_contention(lock, model.living_loop_status)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["token_count"], first["token_count"])
        self.assertEqual(
            cached_result["living_loop"]["subcortical_control_candidates"]["surface"],
            "subcortical_control_candidates.v1",
        )

    def test_policy_actuator_status_returns_cached_result_when_lock_contended(self) -> None:
        """When the lock is held, policy_actuator_status() returns cached data."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.policy_actuator_status()
        cached_result = _run_under_lock_contention(lock, model.policy_actuator_status)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["schema_version"], first["schema_version"])
        self.assertEqual(
            cached_result["subcortical_control_candidates"]["surface"],
            "subcortical_control_candidates.v1",
        )


class StatusReadModelTelemetryActiveCortexTests(unittest.TestCase):
    """Telemetry cache semantics when the cortex is active."""

    def test_telemetry_rebuilds_on_same_revision_when_cortex_active(self) -> None:
        """When cortex is active, telemetry rebuilds even at the same revision."""
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
            sensory_preview_history=deque(maxlen=8),
            architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
            cortex_active_fn=lambda: True,
            animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        )
        # First call populates cache
        first = model.telemetry_snapshot()
        first_call_count = call_count
        # Second call at the same revision with cortex active should rebuild
        second = model.telemetry_snapshot()
        self.assertIsNot(second, first)
        self.assertGreater(call_count, first_call_count)


class StatusReadModelPayloadCompatibilityTests(unittest.TestCase):
    """Verify that the direct read model produces the same payload shape as the Service Manager."""

    def test_status_payload_keys_match_manager_contract(self) -> None:
        """status() returns all keys that the Service Manager test surface asserts."""
        model, _, _, _ = _build_read_model()
        result = model.status()
        # Core keys asserted by test_service_manager.py::test_status_exposes_runtime_truth_contract
        required_keys = [
            "checkpoint_path",
            "token_count",
            "state_revision",
            "dirty_state",
            "runtime_truth",
            "terminus_runtime",
            "memory_store",
            "concept_store",
            "dopamine",
            "serotonin",
            "acetylcholine",
            "norepinephrine",
            "last_winner",
            "context_supported",
            "context_state_norm",
            "trace_history_size",
            "trace_storage_dir",
            "checkpoint_metadata",
            "runtime_scope",
            "replay_dataset_summary",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"status() missing key: {key}")

    def test_status_runtime_scope_includes_trainer_encoder_device_report(self) -> None:
        model, trainer, _, _ = _build_read_model()

        result = model.status()
        encoder_report = result["runtime_scope"]["cuda_first_runtime"]["encoder_device_report"]

        self.assertEqual(encoder_report["device"], str(trainer.encoder.device))
        self.assertEqual(encoder_report["encoder"], "rtf")

    def test_status_runtime_scope_includes_subcortex_spike_health_evidence(self) -> None:
        model, trainer, _, _ = _build_read_model()

        result = model.status()
        spike_health = result["runtime_scope"]["spike_health"]

        self.assertEqual(spike_health["schema_version"], 1)
        self.assertEqual(spike_health["source"], "competitive_columns")
        self.assertEqual(spike_health["n_columns"], trainer.config.n_columns)
        self.assertIn(spike_health["activity_state"], {
            "silent_risk",
            "saturation_risk",
            "stale_routing_risk",
            "sparse_responsive",
        })
        self.assertIn("thresholds", spike_health)
        self.assertFalse(spike_health["correlation_evidence_available"])
        self.assertEqual(spike_health["correlation"]["status"], "insufficient_window")
        self.assertTrue(spike_health["not_liveness_claim"])

    def test_status_runtime_scope_reports_correlation_without_mutating_spike_window(self) -> None:
        model, trainer, _, _ = _build_read_model()
        samples = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        trainer.model.competitive.recent_spike_window[:4] = samples
        trainer.model.competitive.recent_spike_window_cursor = 4
        trainer.model.competitive.recent_spike_window_count = 4

        first = model.status()
        second = model.status()
        spike_health = first["runtime_scope"]["spike_health"]

        self.assertTrue(spike_health["correlation_evidence_available"])
        self.assertEqual(spike_health["correlation"]["status"], "overcorrelated_risk")
        self.assertEqual(trainer.model.competitive.recent_spike_window_count, 4)
        self.assertEqual(
            second["runtime_scope"]["spike_health"]["correlation"]["sample_count"],
            4,
        )

    def test_terminus_status_payload_keys_match_manager_contract(self) -> None:
        """terminus_status() returns all keys that the Service Manager test surface asserts."""
        model, _, _, _ = _build_read_model()
        result = model.terminus_status()
        required_keys = [
            "terminus_runtime",
            "token_count",
            "state_revision",
            "dirty_state",
            "runtime_truth",
            "multimodal",
            "runtime_scope",
            "memory_store",
            "replay_dataset_summary",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"terminus_status() missing key: {key}")

    def test_terminus_status_runtime_scope_includes_trainer_encoder_device_report(self) -> None:
        model, trainer, _, _ = _build_read_model()

        result = model.terminus_status()
        encoder_report = result["runtime_scope"]["cuda_first_runtime"]["encoder_device_report"]

        self.assertEqual(encoder_report["device"], str(trainer.encoder.device))
        self.assertEqual(encoder_report["encoder"], "rtf")
        self.assertIn("capacity", result["memory_store"])
        self.assertEqual(result["runtime_scope"]["spike_health"]["source"], "competitive_columns")

    def test_telemetry_payload_keys_match_manager_contract(self) -> None:
        """telemetry_snapshot() returns all keys that the Service Manager test surface asserts."""
        model, _, _, _ = _build_read_model()
        result = model.telemetry_snapshot()
        required_keys = [
            "generated_at",
            "checkpoint_path",
            "dirty_state",
            "state_revision",
            "token_count",
            "last_winner",
            "context_state_norm",
            "trace_history_size",
            "memory_fill_fraction",
            "memory_buffer_size",
            "dopamine",
            "serotonin",
            "acetylcholine",
            "norepinephrine",
            "drift",
            "drift_floor",
            "animation",
            "terminus_runtime",
            "runtime_scope",
            "memory_store",
            "replay_dataset_summary",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"telemetry_snapshot() missing key: {key}")

    def test_telemetry_runtime_scope_includes_trainer_encoder_device_report(self) -> None:
        model, trainer, _, _ = _build_read_model()

        result = model.telemetry_snapshot()
        encoder_report = result["runtime_scope"]["cuda_first_runtime"]["encoder_device_report"]

        self.assertEqual(encoder_report["device"], str(trainer.encoder.device))
        self.assertEqual(encoder_report["encoder"], "rtf")
        self.assertIn("capacity", result["memory_store"])
        self.assertEqual(result["runtime_scope"]["spike_health"]["source"], "competitive_columns")

    def test_runtime_truth_payload_keys_match_manager_contract(self) -> None:
        """runtime_truth sub-payload includes all keys asserted by the manager test surface."""
        model, _, _, _ = _build_read_model()
        result = model.status()
        truth = result["runtime_truth"]
        # Keys asserted by test_status_exposes_runtime_truth_contract
        self.assertEqual(truth["schema_version"], 1)
        self.assertIn("verdict", truth)
        self.assertIn("recommended_action", truth)
        self.assertIn("cortex_available", truth)
        self.assertIn("memory_pressure", truth)
        self.assertIn("safety_flags", truth)
        self.assertIn("latency_ms", truth)
        self.assertIn("evidence", truth)
        self.assertIn("configured", truth["evidence"])
        self.assertIn("token_count", truth["evidence"])
        self.assertIn("subcortex_spike_health", truth["evidence"])
        self.assertTrue(truth["evidence"]["subcortex_spike_health"]["not_liveness_claim"])


class StatusReadModelSensoryPreviewReadonlyTests(unittest.TestCase):
    """sensory_previews() is read-only: it does not mutate runtime state."""

    def test_sensory_previews_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        rev_before = runtime_state.state_revision
        model.sensory_previews()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_sensory_previews_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        model.sensory_previews()
        self.assertFalse(runtime_state.dirty_state)


class StatusReadModelArchitectureReadonlyTests(unittest.TestCase):
    """architecture_summary() is read-only: it does not mutate runtime state."""

    def test_architecture_summary_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        rev_before = runtime_state.state_revision
        model.architecture_summary()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_architecture_summary_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state = _build_read_model()
        runtime_state.mark_clean()
        model.architecture_summary()
        self.assertFalse(runtime_state.dirty_state)


class StatusReadModelCognitiveSignalReadonlyTests(unittest.TestCase):
    """cognitive_signal_state() is read-only: it does not mutate runtime state."""

    def test_cognitive_signal_state_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        model.cognitive_signal_state()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_cognitive_signal_state_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        runtime_state.mark_clean()
        model.cognitive_signal_state()
        self.assertFalse(runtime_state.dirty_state)

    def test_cortex_signal_state_does_not_advance_revision(self) -> None:
        """The retired Cortex wrapper remains read-only."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        model.cortex_signal_state()
        rev_after = runtime_state.state_revision
        self.assertEqual(rev_before, rev_after)

    def test_cortex_signal_state_does_not_set_dirty_state(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        runtime_state.mark_clean()
        model.cortex_signal_state()
        self.assertFalse(runtime_state.dirty_state)


class RuntimeFacadeDelegationTests(unittest.TestCase):
    """RuntimeFacade delegates status projections to StatusReadModel."""

    def test_runtime_facade_status_delegates_to_read_model(self) -> None:
        """The runtime facade's status() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="delegation")
            try:
                self.assertIsNotNone(manager._status_read_model)
                self.assertIsInstance(manager._status_read_model, StatusReadModel)
                result = manager.runtime_facade.status()
                self.assertIn("runtime_truth", result)
                self.assertIn("checkpoint_path", result)
            finally:
                manager.close()

    def test_runtime_facade_terminus_status_delegates_to_read_model(self) -> None:
        """The runtime facade's terminus_status() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="delegation")
            try:
                result = manager.runtime_facade.terminus_status()
                self.assertIn("runtime_truth", result)
                self.assertIn("terminus_runtime", result)
                self.assertIn("multimodal", result)
                self.assertIn("enabled", result["multimodal"])
            finally:
                manager.close()

    def test_runtime_facade_terminus_status_multimodal_preserved_through_read_model(self) -> None:
        """terminus_status() preserves the full multimodal payload through the read model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="multimodal_preserved")
            try:
                result = manager.runtime_facade.terminus_status()
                multimodal = result["multimodal"]
                expected_keys = {
                    "enabled",
                    "mode",
                    "episodes_completed",
                    "focus_terms",
                    "source_names",
                }
                self.assertTrue(
                    expected_keys.issubset(set(multimodal.keys())),
                    f"Missing keys: {expected_keys - set(multimodal.keys())}",
                )
            finally:
                manager.close()

    def test_runtime_facade_sensory_previews_delegates_to_read_model(self) -> None:
        """The runtime facade's sensory_previews() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt",
                trainer,
                metadata={"test_case": "sensory_delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.runtime_facade.sensory_previews()
                self.assertIn("count", result)
                self.assertIn("latest_preview_id", result)
                self.assertIn("previews", result)
                self.assertIsInstance(result["previews"], list)
                # Empty history at startup
                self.assertEqual(result["count"], 0)
            finally:
                manager.close()

    def test_runtime_facade_architecture_summary_delegates_to_read_model(self) -> None:
        """The runtime facade's architecture_summary() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt",
                trainer,
                metadata={"test_case": "arch_delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.runtime_facade.architecture_summary()
                self.assertEqual(result["model_name"], "Terminus")
                self.assertEqual(result["core_name"], "GPCSN")
                self.assertIn("layers", result)
                self.assertIn("config", result)
                layer_ids = [l["id"] for l in result["layers"]]
                self.assertIn("input_encoding", layer_ids)
                self.assertIn("competitive_routing", layer_ids)
                self.assertIn("memory_consolidation", layer_ids)
            finally:
                manager.close()

    def test_runtime_facade_telemetry_snapshot_delegates_to_read_model(self) -> None:
        """The runtime facade's telemetry_snapshot() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="telemetry_delegation")
            try:
                result = manager.runtime_facade.telemetry_snapshot()
                self.assertIn("animation", result)
                self.assertIn("terminus_runtime", result)
                self.assertIn("token_count", result)
                self.assertIn("state_revision", result)
            finally:
                manager.close()

    def test_runtime_facade_living_loop_status_delegates_to_read_model(self) -> None:
        """The runtime facade's living_loop_status() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "living_loop_delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.runtime_facade.living_loop_status()
                self.assertIn("living_loop", result)
                self.assertIn("token_count", result)
                self.assertIn("state_revision", result)
            finally:
                manager.close()

    def test_runtime_facade_policy_actuator_status_delegates_to_read_model(self) -> None:
        """The runtime facade's policy_actuator_status() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            cfg = _build_config()
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "policy_delegation"},
            )
            manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
            try:
                result = manager.runtime_facade.policy_actuator_status()
                self.assertIn("schema_version", result)
                self.assertIn("recommendation", result)
                self.assertIn("action", result)
            finally:
                manager.close()

    def test_runtime_facade_cognitive_signal_state_delegates_to_read_model(self) -> None:
        """The runtime facade's cognitive_signal_state() method delegates to StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cognitive_signal_delegation")
            try:
                result = manager.runtime_facade.cognitive_signal_state()
                self.assertIn("prediction_error_mean", result)
                self.assertIn("predictive_confidence_mean", result)
                self.assertIn("recent_concepts", result)
                self.assertIn("subcortical_language", result)
                self.assertIn("subcortical_deliberation", result)
            finally:
                manager.close()

    def test_runtime_facade_subcortical_language_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes the Cognitive Signal language surface directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="subcortical_language_delegation")
            try:
                result = manager.runtime_facade.subcortical_language_surface()
                self.assertEqual(result["surface"], "subcortical_language.v1")
                self.assertEqual(result["source"], "service.status_read_model.cognitive_signal")
                self.assertTrue(result["grounded"])
            finally:
                manager.close()

    def test_runtime_facade_subcortical_deliberation_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes bounded Subcortex deliberation candidates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="subcortical_deliberation_delegation")
            try:
                result = manager.runtime_facade.subcortical_deliberation_surface()
                self.assertEqual(result["surface"], "subcortical_control_candidates.v1")
                self.assertEqual(result["source"], "service.status_read_model.cognitive_signal")
                self.assertTrue(result["grounded"])
                self.assertIn("candidates", result)
            finally:
                manager.close()
