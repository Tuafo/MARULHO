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
        "id": "nim_cortex",
        "name": "NIM Mind Layer",
        "enabled": False,
        "type": "cortex",
        "params": {},
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
        "family": "hybrid_snn_llm",
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
        self.assertEqual(result["family"], "hybrid_snn_llm")

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


class ServiceManagerDelegationTests(unittest.TestCase):
    """Service Manager delegates status() and terminus_status() to StatusReadModel."""

    def test_manager_status_delegates_to_read_model(self) -> None:
        """The manager's status() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="delegation")
            try:
                self.assertIsNotNone(manager._status_read_model)
                self.assertIsInstance(manager._status_read_model, StatusReadModel)
                result = manager.status()
                self.assertIn("runtime_truth", result)
                self.assertIn("checkpoint_path", result)
            finally:
                manager.close()

    def test_manager_terminus_status_delegates_to_read_model(self) -> None:
        """The manager's terminus_status() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="delegation")
            try:
                result = manager.terminus_status()
                self.assertIn("runtime_truth", result)
                self.assertIn("terminus_runtime", result)
                self.assertIn("multimodal", result)
                self.assertIn("enabled", result["multimodal"])
            finally:
                manager.close()

    def test_manager_terminus_status_multimodal_preserved_through_read_model(self) -> None:
        """terminus_status() preserves the full multimodal payload through the read model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="multimodal_preserved")
            try:
                result = manager.terminus_status()
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

    def test_manager_sensory_previews_delegates_to_read_model(self) -> None:
        """The manager's sensory_previews() method delegates to its StatusReadModel."""
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
                result = manager.sensory_previews()
                self.assertIn("count", result)
                self.assertIn("latest_preview_id", result)
                self.assertIn("previews", result)
                self.assertIsInstance(result["previews"], list)
                # Empty history at startup
                self.assertEqual(result["count"], 0)
            finally:
                manager.close()

    def test_manager_architecture_summary_delegates_to_read_model(self) -> None:
        """The manager's architecture_summary() method delegates to its StatusReadModel."""
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
                result = manager.architecture_summary()
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

    def test_manager_telemetry_snapshot_delegates_to_read_model(self) -> None:
        """The manager's telemetry_snapshot() method delegates to its StatusReadModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="telemetry_delegation")
            try:
                result = manager.telemetry_snapshot()
                self.assertIn("animation", result)
                self.assertIn("terminus_runtime", result)
                self.assertIn("token_count", result)
                self.assertIn("state_revision", result)
            finally:
                manager.close()
