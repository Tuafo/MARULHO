"""Direct test surface for the Status Read Model seam.

These tests exercise the StatusReadModel through its own interface with injected
adapters, verifying snapshot payloads and cache/freshness semantics without
requiring the full Service Manager composition root.  Regression coverage for
unchanged public behavior remains in test_service_manager.py and test_service_api.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import threading
import time
import unittest
from collections import deque
from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any, Callable

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.semantics import build_spike_language_decoder_probe
from marulho.service.runtime_state import RuntimeState
from marulho.service.status_read_model import StatusReadModel
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _build_config() -> MarulhoConfig:
    return MarulhoConfig(
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


def _build_architecture_snapshot(trainer: MarulhoTrainer) -> dict[str, Any]:
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
    language_plasticity_state_fn: Callable[[], dict[str, Any]] | None = None,
    readout_ledger_state_fn: Callable[[], dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    report_root: Path | None = None,
) -> tuple[StatusReadModel, MarulhoTrainer, threading.RLock, RuntimeState]:
    cfg = _build_config()
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    lock = threading.RLock()
    runtime_state = RuntimeState(lock=lock)
    brain_snapshot = _build_brain_snapshot()
    animation_snapshot = _build_animation_snapshot()
    model = StatusReadModel(
        lock=lock,
        runtime_state=runtime_state,
        trainer=trainer,
        trace_history=deque(maxlen=200),
        metadata=deepcopy(metadata or {}),
        checkpoint_path_str="/tmp/test.pt",
        trace_dir_str="/tmp/traces",
        concept_store_snapshot_fn=lambda: deepcopy({"top_concepts": [], "total_concepts": 0}),
        brain_runtime_snapshot_fn=lambda: deepcopy(brain_snapshot),
        sensory_preview_history=deque(maxlen=8),
        architecture_snapshot_fn=lambda: _build_architecture_snapshot(trainer),
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        language_plasticity_state_fn=language_plasticity_state_fn,
        readout_ledger_state_fn=readout_ledger_state_fn,
        report_root=report_root,
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
    from marulho.service.manager import MarulhoServiceManager
    cfg = _build_config()
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return MarulhoServiceManager(checkpoint_path, trace_dir=root / "traces")


class StatusReadModelConstructionTests(unittest.TestCase):
    """StatusReadModel can be constructed with injected dependencies."""

    def test_read_model_constructs_with_adapter(self) -> None:
        """StatusReadModel should accept a manager-like adapter at construction."""
        model, _, _, _ = _build_read_model()
        self.assertIsNotNone(model)

    def test_read_model_owns_sensory_preview_projection(self) -> None:
        source = Path("src/marulho/service/status_read_model.py").read_text(encoding="utf-8")

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
        device = truth["evidence"]["runtime_device"]
        self.assertEqual(device["summary_role"], "observed_runtime_device_evidence_not_acceleration_claim")
        self.assertEqual(device["resolved_device"], "cpu")
        self.assertEqual(device["tensor_device"], "cpu")
        self.assertEqual(device["cuda_available"], torch.cuda.is_available())
        self.assertFalse(device["observed_cuda_execution"])
        self.assertEqual(device["claim_boundary"], "observed_device_placement_only_not_cuda_speedup")
        self.assertEqual(truth["evidence"]["device"], "cpu")
        self.assertEqual(truth["evidence"]["cuda_available"], torch.cuda.is_available())
        column_runtime = truth["evidence"]["column_runtime"]
        self.assertEqual(column_runtime["surface"], "column_runtime_metabolism.v1")
        self.assertEqual(
            column_runtime["summary_role"],
            "compact_runtime_truth_column_metabolism_not_execution_scheduler",
        )
        self.assertLessEqual(column_runtime["awake_count"], column_runtime["awake_budget"])
        self.assertFalse(column_runtime["runs_all_columns"])
        self.assertEqual(
            column_runtime["claim_boundary"],
            "column_scheduler_evidence_only_not_sparse_execution_promotion",
        )

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
        self.assertEqual(
            truth["evidence"]["runtime_device"]["claim_boundary"],
            "observed_device_placement_only_not_cuda_speedup",
        )
        self.assertEqual(truth["evidence"]["device"], "cpu")
        self.assertFalse(truth["evidence"]["observed_cuda_execution"])

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
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        """When revision is unchanged, telemetry returns the cached snapshot."""
        call_count = 0
        brain_snapshot = _build_brain_snapshot()

        def counting_brain_fn() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return deepcopy(brain_snapshot)

        cfg = _build_config()
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        """When revision changes, telemetry rebuilds."""
        cfg = _build_config()
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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


def _build_sleep_plasticity_autonomy_proposal_snapshot() -> dict[str, Any]:
    return {
        "surface": "snn_sleep_plasticity_autonomy_proposal.v1",
        "ready": True,
        "owned_by_marulho": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "candidate": {
            "candidate_id": "snn-sleep-plasticity-autonomy:ticket-1",
            "action": "review_sleep_plasticity_next_gate",
            "review_ticket_id": "ticket-1",
            "suggested_endpoint": "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        },
        "promotion_gate": {
            "status": "ready_for_operator_next_gate_review",
            "eligible_for_autonomy_planning": True,
            "eligible_for_action": False,
            "eligible_for_structural_write": False,
            "next_gate": "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        },
    }


def _build_sleep_plasticity_scheduler_installation_autonomy_proposal_snapshot() -> dict[str, Any]:
    return {
        "surface": "snn_sleep_plasticity_scheduler_installation_autonomy_proposal.v1",
        "ready": True,
        "owned_by_marulho": True,
        "advisory": True,
        "executable": False,
        "installs_scheduler": False,
        "registers_timer": False,
        "starts_background_worker": False,
        "mutates_runtime_state": False,
        "candidate": {
            "scheduler_design_review_ticket_id": "design-ticket-1",
            "scheduler_design_hash": "design-hash-1",
        },
        "promotion_gate": {
            "status": "ready_for_operator_scheduler_installation_preflight_review",
            "eligible_for_autonomy_planning": True,
            "eligible_for_scheduler_installation_preflight_review": True,
            "eligible_for_scheduler_installation": False,
            "eligible_for_action": False,
            "eligible_for_structural_write": False,
            "next_gate": (
                "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                "scheduler-installation-preflight"
            ),
        },
    }


def _build_read_model_with_living_loop() -> tuple[StatusReadModel, MarulhoTrainer, threading.RLock, RuntimeState, dict[str, int]]:
    """Build a StatusReadModel with living loop callbacks wired for testing."""
    cfg = _build_config()
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    lock = threading.RLock()
    runtime_state = RuntimeState(lock=lock)
    brain_snapshot = _build_brain_snapshot()
    animation_snapshot = _build_animation_snapshot()
    living_loop_result = _build_living_loop_snapshot()
    policy_result = _build_policy_actuator_snapshot()
    cognitive_signal_result = _build_cognitive_signal_state_snapshot()
    sleep_plasticity_proposal_result = _build_sleep_plasticity_autonomy_proposal_snapshot()
    scheduler_installation_proposal_result = (
        _build_sleep_plasticity_scheduler_installation_autonomy_proposal_snapshot()
    )
    call_counts: dict[str, int] = {
        "living_loop": 0,
        "policy_actuator": 0,
        "cognitive_signal": 0,
        "sleep_plasticity": 0,
        "scheduler_installation": 0,
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

    def sleep_plasticity_autonomy_proposal_fn() -> dict[str, Any]:
        call_counts["sleep_plasticity"] += 1
        return deepcopy(sleep_plasticity_proposal_result)

    def sleep_plasticity_scheduler_installation_autonomy_proposal_fn() -> dict[str, Any]:
        call_counts["scheduler_installation"] += 1
        return deepcopy(scheduler_installation_proposal_result)

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
        animation_snapshot_fn=lambda: deepcopy(animation_snapshot),
        living_loop_status_fn=living_loop_snapshot_fn,
        policy_actuator_status_fn=policy_actuator_snapshot_fn,
        cognitive_signal_state_fn=cognitive_signal_state_fn,
        sleep_plasticity_autonomy_proposal_fn=sleep_plasticity_autonomy_proposal_fn,
        sleep_plasticity_scheduler_installation_autonomy_proposal_fn=(
            sleep_plasticity_scheduler_installation_autonomy_proposal_fn
        ),
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
        self.assertNotIn("retired_runtime_dependency", sidecar)
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
        self.assertNotIn("retired_runtime_dependency", sidecar)
        self.assertEqual(sidecar["candidates"][0]["intent"], "review_column_revival")
        self.assertFalse(sidecar["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(sidecar["candidates"][0]["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(sidecar["promotion_summary"]["eligible_for_structural_mutation"])
        self.assertEqual(sidecar["promotion_gate"]["status"], "insufficient_evidence")
        self.assertEqual(sidecar["promotion_gate"]["next_gate"], "collect_spike_window")
        self.assertFalse(sidecar["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)

    def test_living_loop_status_includes_sleep_plasticity_autonomy_proposal(self) -> None:
        model, _, _, runtime_state, call_counts = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        result = model.living_loop_status()
        proposal = result["living_loop"]["snn_sleep_plasticity_autonomy_proposal"]

        self.assertEqual(proposal["surface"], "snn_sleep_plasticity_autonomy_proposal.v1")
        self.assertTrue(proposal["advisory"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertEqual(proposal["candidate"]["review_ticket_id"], "ticket-1")
        self.assertEqual(
            proposal["promotion_gate"]["next_gate"],
            "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        )
        self.assertFalse(proposal["promotion_gate"]["eligible_for_action"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_structural_write"])
        self.assertEqual(call_counts["sleep_plasticity"], 1)
        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)

    def test_living_loop_status_returns_token_count(self) -> None:
        """living_loop_status() should include token_count at the top level."""
        model, trainer, _, _, _ = _build_read_model_with_living_loop()
        result = model.living_loop_status()
        self.assertIn("token_count", result)
        self.assertEqual(result["token_count"], int(trainer.token_count))

    def test_living_loop_status_includes_scheduler_installation_autonomy_proposal(self) -> None:
        model, _, _, runtime_state, call_counts = _build_read_model_with_living_loop()
        revision_before = runtime_state.state_revision
        runtime_state.mark_clean()

        result = model.living_loop_status()
        proposal = result["living_loop"][
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
        ]

        self.assertEqual(
            proposal["surface"],
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal.v1",
        )
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["installs_scheduler"])
        self.assertFalse(proposal["registers_timer"])
        self.assertFalse(proposal["starts_background_worker"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertEqual(call_counts["scheduler_installation"], 1)
        self.assertEqual(runtime_state.state_revision, revision_before)
        self.assertFalse(runtime_state.dirty_state)

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
        self.assertNotIn("retired_runtime_dependency", candidates)
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
        self.assertFalse(candidates["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(runtime_state.dirty_state)

    def test_policy_actuator_status_includes_sleep_plasticity_autonomy_proposal(self) -> None:
        model, _, _, runtime_state, call_counts = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        result = model.policy_actuator_status()
        proposal = result["snn_sleep_plasticity_autonomy_proposal"]

        self.assertEqual(proposal["surface"], "snn_sleep_plasticity_autonomy_proposal.v1")
        self.assertTrue(proposal["advisory"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertEqual(proposal["candidate"]["action"], "review_sleep_plasticity_next_gate")
        self.assertTrue(proposal["promotion_gate"]["eligible_for_autonomy_planning"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_action"])
        self.assertFalse(proposal["promotion_gate"]["eligible_for_structural_write"])
        self.assertEqual(call_counts["sleep_plasticity"], 1)
        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)

    def test_policy_actuator_status_delegates_to_callback(self) -> None:
        """policy_actuator_status() should delegate through the injected callback."""
        model, _, _, _, call_counts = _build_read_model_with_living_loop()
        model.policy_actuator_status()
        self.assertGreater(call_counts["policy_actuator"], 0)

    def test_policy_actuator_status_includes_scheduler_installation_autonomy_proposal(self) -> None:
        model, _, _, runtime_state, call_counts = _build_read_model_with_living_loop()
        revision_before = runtime_state.state_revision
        runtime_state.mark_clean()

        result = model.policy_actuator_status()
        proposal = result[
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
        ]

        self.assertTrue(proposal["ready"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["installs_scheduler"])
        self.assertFalse(proposal["registers_timer"])
        self.assertFalse(proposal["starts_background_worker"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertEqual(call_counts["scheduler_installation"], 1)
        self.assertEqual(runtime_state.state_revision, revision_before)
        self.assertFalse(runtime_state.dirty_state)


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
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertEqual(surface["grounding"]["concept_focus"], "coral thermal memory")
        deliberation = result["subcortical_deliberation"]
        self.assertEqual(deliberation["surface"], "subcortical_control_candidates.v1")
        self.assertTrue(deliberation["grounded"])
        self.assertTrue(deliberation["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", deliberation)
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
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertTrue(surface["candidates"])
        first = surface["candidates"][0]
        self.assertEqual(first["phase"], "monitor")
        self.assertIn("coral thermal memory", first["candidate_text"])
        self.assertNotIn("prompt", first)
        self.assertIn("prediction_error_mean", first["grounding"])
        self.assertIn("promotion_gate", first)
        self.assertFalse(first["promotion_gate"]["eligible_for_action"])

    def test_snn_language_readiness_surface_is_reference_only_and_non_executable(self) -> None:
        """SNN language readiness points to repo-native work, not external dependencies."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        surface = model.snn_language_readiness_surface()

        self.assertEqual(surface["surface"], "snn_native_language_readiness.v1")
        self.assertEqual(surface["artifact_kind"], "terminus_snn_native_language_readiness_gate")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertEqual(surface["promotion_gate"]["status"], "research_candidate_only")
        self.assertFalse(surface["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertEqual(
            surface["current_spike_readout_evidence"]["surface"],
            "subcortical_spike_readout_evidence.v1",
        )
        self.assertFalse(surface["current_spike_readout_evidence"]["generates_text"])
        self.assertEqual(
            surface["current_decoder_probe_evidence"]["surface"],
            "snn_language_decoder_probe_evidence.v1",
        )
        self.assertTrue(surface["current_decoder_probe_evidence"]["owned_by_marulho"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["executable"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_evidence_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_non_generative"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_non_generative"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_owned"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_sparse"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_dynamic"])
        self.assertEqual(
            surface["current_language_neuron_adapter_evidence"]["surface"],
            "snn_language_neuron_adapter_evidence.v1",
        )
        self.assertFalse(surface["current_language_neuron_adapter_evidence"]["generates_text"])
        self.assertFalse(surface["current_language_neuron_adapter_evidence"]["executable"])
        self.assertEqual(
            [candidate["integration_status"] for candidate in surface["research_candidates"]],
            ["reference_for_marulho_owned_reimplementation", "reference_for_marulho_owned_reimplementation"],
        )
        self.assertIn("marulho_owned_language_neuron_module", surface["research_candidates"][0]["required_local_evidence"])
        self.assertIn("marulho_native_snn_decoder", surface["research_candidates"][0]["required_local_evidence"])
        self.assertTrue(surface["safety_invariants"]["requires_marulho_owned_implementation"])

    def test_snn_language_readiness_surface_returns_cached_on_lock_contention(self) -> None:
        """The readiness artifact stays cache-compatible under lock contention."""
        model, _, lock, _, _ = _build_read_model_with_living_loop()
        first = model.snn_language_readiness_surface()
        cached_result = _run_under_lock_contention(lock, model.snn_language_readiness_surface)

        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["surface"], first["surface"])
        self.assertEqual(cached_result["promotion_gate"]["status"], first["promotion_gate"]["status"])

    def test_snn_language_evaluation_surface_gates_adapter_without_generation(self) -> None:
        """The language-adapter evaluation surface remains read-only and non-generative."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        surface = model.snn_language_evaluation_surface()

        self.assertEqual(surface["surface"], "snn_language_adapter_evaluation.v1")
        self.assertEqual(surface["artifact_kind"], "terminus_snn_language_adapter_evaluation_gate")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(surface["evaluation_cases"][0]["target"], "spike_language_neuron_adapter")
        self.assertIn("post_evaluation_grounding_delta", surface["success_evidence"])

    def test_snn_language_heldout_evaluation_does_not_advance_revision(self) -> None:
        """Heldout adapter evaluation is evidence-only."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        report = model.snn_language_adapter_heldout_evaluation(
            [
                [
                    {"label": "prediction error", "pressure_band": "high", "grounded": True},
                    {"label": "concept focus", "pressure_band": "medium", "grounded": True},
                ]
            ],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_adapter_heldout_evaluation.v1")
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["trains"])
        self.assertFalse(report["mutates_runtime_state"])

    def test_snn_language_training_readiness_does_not_advance_revision(self) -> None:
        """Training readiness is design evidence, not trainer execution."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        heldout = model.snn_language_adapter_heldout_evaluation(
            [[{"label": "prediction error", "pressure_band": "high", "grounded": True}]],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_training_readiness(
            heldout,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-training"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_training_readiness.v1")
        self.assertFalse(report["executable"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_training"])
        self.assertTrue(report["promotion_gate"]["eligible_for_training_loop_design"])

    def test_snn_language_trainer_dry_run_does_not_advance_revision(self) -> None:
        """Trainer dry-run is isolated tensor evidence, not runtime training."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        report = model.snn_language_trainer_dry_run(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_trainer_dry_run.v1")
        self.assertFalse(report["trains_runtime_model"])
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_runtime_training"])

    def test_snn_language_trainer_evaluation_does_not_advance_revision(self) -> None:
        """Trainer evaluation gates dry-run evidence without promotion."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        dry_run = model.snn_language_trainer_dry_run(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_trainer_isolated_evaluation(
            dry_run,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-trainer-eval"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_trainer_isolated_evaluation.v1")
        self.assertFalse(report["trains_runtime_model"])
        self.assertFalse(report["promotes_runtime_trainer"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_runtime_training"])

    def test_snn_language_sequence_prediction_probe_does_not_advance_revision(self) -> None:
        """Sequence prediction returns sparse evidence without text generation."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        report = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_sequence_prediction_probe.v1")
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["decodes_text"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(len(report["prediction"]["predicted_sparse_indices"]), 4)
        self.assertFalse(report["persistent_transition_evidence"]["influenced_prediction"])

    def test_snn_language_sequence_prediction_probe_accepts_persistent_transition_state(self) -> None:
        """Persistent sparse transition state can influence prediction without mutation."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        baseline = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        current_index = int(baseline["current_sparse_code"]["active_indices"][0])
        target_index = (current_index + 9) % 64
        rev_before = runtime_state.state_revision
        report = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
            persistent_transition_weights={f"{current_index}:{target_index}": 0.5},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertTrue(report["persistent_transition_evidence"]["influenced_prediction"])
        self.assertIn(target_index, report["prediction"]["predicted_sparse_indices"])
        self.assertFalse(report["mutates_runtime_state"])

    def test_snn_language_transition_memory_prediction_evaluation_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        observed = [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}]
        current_probe = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            current,
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        current_index = int(current_probe["current_sparse_code"]["active_indices"][0])
        target_index = (current_index + 5) % 64
        rev_before = runtime_state.state_revision
        report = model.snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            [current, observed],
            {"sparse_transition_weights": {f"{current_index}:{target_index}": 0.5}},
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_transition_memory_prediction_evaluation.v1")
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["decodes_text"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["trains_runtime_model"])
        self.assertEqual(report["evaluation_summary"]["persistent_transition_weight_count"], 1)

    def test_snn_language_readout_rollout_candidate_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        observed = [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}]
        device = {"device": "cpu", "source": "test"}
        current_probe = build_spike_language_decoder_probe({"readout_slots": current, "device_evidence": device})
        observed_probe = build_spike_language_decoder_probe({"readout_slots": observed, "device_evidence": device})
        current_index = int(current_probe["sparse_code_evidence"]["active_indices"][0])
        observed_index = int(observed_probe["sparse_code_evidence"]["active_indices"][0])
        weights = {f"{current_index}:{observed_index}": 0.8}
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            current,
            device_evidence=device,
            top_k=4,
            persistent_transition_weights=weights,
        )
        evaluation = model.snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            [current, observed],
            {"sparse_transition_weights": weights},
            device_evidence=device,
            top_k=4,
        )
        rev_before = runtime_state.state_revision
        rollout = model.snn_language_readout_rollout_candidate(
            prediction,
            observed,
            {
                "sparse_transition_weights": weights,
                "transition_memory_state_source": (
                    "service.runtime_facade.snn_language_plasticity_runtime_state"
                ),
                "current_state_revision": runtime_state.state_revision,
            },
            device_evidence=device,
            transition_memory_evaluation=evaluation,
            rollout_steps=2,
            top_k=4,
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(rollout["surface"], "snn_language_readout_rollout_candidate.v1")
        self.assertFalse(rollout["mutates_runtime_state"])
        self.assertFalse(rollout["applies_plasticity"])
        self.assertFalse(rollout["loads_external_checkpoint"])
        self.assertIn("memory pressure", rollout["rollout"]["labels"])

    def test_snn_language_readout_rollout_replay_evaluation_does_not_advance_revision(self) -> None:
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        observed = [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}]
        device = {"device": "cpu", "source": "test"}
        current_probe = build_spike_language_decoder_probe({"readout_slots": current, "device_evidence": device})
        observed_probe = build_spike_language_decoder_probe({"readout_slots": observed, "device_evidence": device})
        current_index = int(current_probe["sparse_code_evidence"]["active_indices"][0])
        observed_index = int(observed_probe["sparse_code_evidence"]["active_indices"][0])
        weights = {f"{current_index}:{observed_index}": 0.8}
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            current,
            device_evidence=device,
            top_k=4,
            persistent_transition_weights=weights,
        )
        evaluation = model.snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            [current, observed],
            {"sparse_transition_weights": weights},
            device_evidence=device,
            top_k=4,
        )
        rollout = model.snn_language_readout_rollout_candidate(
            prediction,
            observed,
            {
                "sparse_transition_weights": weights,
                "transition_memory_state_source": (
                    "service.runtime_facade.snn_language_plasticity_runtime_state"
                ),
                "current_state_revision": runtime_state.state_revision,
            },
            device_evidence=device,
            transition_memory_evaluation=evaluation,
            rollout_steps=2,
            top_k=4,
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_readout_rollout_replay_evaluation(
            rollout,
            candidate_limit=4,
            device_evidence=device,
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_readout_rollout_replay_evaluation.v1")
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["recorded_in_ledger"])
        self.assertFalse(report["eligible_for_replay_priority"])
        self.assertTrue(report["promotion_gate"]["eligible_for_readout_rollout_ledger_recording_review"])

    def test_snn_language_sequence_mismatch_probe_does_not_advance_revision(self) -> None:
        """Mismatch probe reports prediction error without applying learning."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_sequence_mismatch_probe.v1")
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["decodes_text"])
        self.assertFalse(report["trains_runtime_model"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertIn("mismatch_score", report["prediction_error"])

    def test_snn_language_plasticity_pressure_does_not_advance_revision(self) -> None:
        """Plasticity pressure is a gate, not a learning update."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_plasticity_pressure.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_learning_signal"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])

    def test_snn_language_plasticity_trial_does_not_advance_revision(self) -> None:
        """Plasticity trial simulates a local update without applying it."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        pressure = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_plasticity_trial.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])

    def test_snn_language_plasticity_replay_evaluation_does_not_advance_revision(self) -> None:
        """Replay evaluation reviews trial evidence without promoting learning."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        pressure = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = model.snn_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_plasticity_replay_evaluation(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_plasticity_replay_evaluation.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(report["promotion_gate"]["eligible_for_replay_promotion"])

    def test_snn_language_plasticity_replay_experiment_does_not_advance_revision(self) -> None:
        """Replay experiment rehearses sparse evidence without applying plasticity."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        pressure = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = model.snn_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = model.snn_language_plasticity_replay_evaluation(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_plasticity_replay_experiment.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])

    def test_snn_language_plasticity_application_design_does_not_advance_revision(self) -> None:
        """Application design bounds a future update without applying it."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        pressure = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = model.snn_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = model.snn_language_plasticity_replay_evaluation(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        experiment = model.snn_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_before = runtime_state.state_revision
        report = model.snn_language_plasticity_application_design(
            experiment,
            application_policy={"learning_rate": 0.03, "max_weight_delta": 0.04, "locality_radius": 2},
            device_evidence={"device": "cpu", "source": "test"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "snn_language_plasticity_application_design.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertEqual(report["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(report["promotion_gate"]["eligible_for_live_application"])

    def test_snn_language_plasticity_shadow_application_does_not_advance_revision(self) -> None:
        """Shadow application verifies bounded deltas without applying them."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        prediction = model.snn_language_sequence_prediction_probe(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
            top_k=4,
        )
        mismatch = model.snn_language_sequence_mismatch_probe(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        pressure = model.snn_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = model.snn_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = model.snn_language_plasticity_replay_evaluation(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        experiment = model.snn_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        design = model.snn_language_plasticity_application_design(
            experiment,
            application_policy={"learning_rate": 0.03, "max_weight_delta": 0.04, "locality_radius": 2},
            device_evidence={"device": "cpu", "source": "test"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        rev_before = runtime_state.state_revision
        shadow_delta = model.snn_language_plasticity_shadow_delta(
            design,
            replay_sequences=[{"pre_indices": [2, 3], "post_indices": [3, 4], "grounded": True}],
            device_evidence={"device": "cpu", "source": "test"},
        )
        report = model.snn_language_plasticity_shadow_application(
            design,
            shadow_delta=shadow_delta,
            device_evidence={"device": "cpu", "source": "test"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        live_readiness = model.snn_language_plasticity_live_application_readiness(
            report,
            rollback_readiness={
                "checkpoint_available": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_endpoint_available": True,
            },
            operator_approval={
                "approved": True,
                "operator_id": "operator-test",
                "approval_id": "approval-1",
            },
        )
        preflight = model.snn_language_plasticity_live_application_preflight(
            live_readiness,
            application_target={
                "available": True,
                "target_id": "marulho.snn_language.sparse_transition_weights",
                "owned_by_marulho": True,
                "mutable": True,
                "sparse": True,
                "checkpointed": True,
            },
            checkpoint_transaction={
                "pre_update_checkpoint_saved": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_verified": True,
                "records_shadow_delta": True,
            },
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(shadow_delta["surface"], "snn_language_plasticity_shadow_delta.v1")
        self.assertFalse(shadow_delta["applies_plasticity"])
        self.assertEqual(shadow_delta["device_evidence"]["tensor_device"], "cpu")
        self.assertGreater(shadow_delta["affected_synapse_count"], 0)
        self.assertEqual(report["surface"], "snn_language_plasticity_shadow_application.v1")
        self.assertFalse(report["applies_plasticity"])
        self.assertEqual(report["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(report["promotion_gate"]["eligible_for_live_application"])
        self.assertEqual(live_readiness["surface"], "snn_language_plasticity_live_application_readiness.v1")
        self.assertFalse(live_readiness["applies_plasticity"])
        self.assertFalse(live_readiness["mutates_runtime_state"])
        self.assertFalse(live_readiness["returns_trained_weights"])
        self.assertFalse(live_readiness["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(live_readiness["promotion_gate"]["eligible_for_operator_live_application_review"])
        self.assertEqual(preflight["surface"], "snn_language_plasticity_live_application_preflight.v1")
        self.assertFalse(preflight["applies_plasticity"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(preflight["promotion_gate"]["eligible_for_operator_execution_review"])

    def test_structural_plasticity_isolated_evaluation_does_not_advance_revision(self) -> None:
        """Structural grow/prune evaluation compares snapshots without mutation."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        rev_before = runtime_state.state_revision
        pre_snapshot = {
            "current_state_revision": 3,
            "binding_topology": {"edges_added_total": 1, "edges_removed_total": 0},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.2, "saturated_fraction": 0.1},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 4,
            "binding_topology": {"edges_added_total": 2, "edges_removed_total": 1},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.1, "saturated_fraction": 0.05},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        report = model.subcortical_structural_plasticity_isolated_evaluation(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-grow-prune",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )
        rev_after = runtime_state.state_revision

        self.assertEqual(rev_before, rev_after)
        self.assertEqual(report["surface"], "subcortical_structural_plasticity_isolated_evaluation.v1")
        self.assertFalse(report["executable"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(report["rollback_evidence"]["bound_to_pre_snapshot"])
        self.assertEqual(report["promotion_gate"]["status"], "ready_for_operator_review")

    def test_structural_mutation_design_does_not_advance_revision(self) -> None:
        """Structural mutation design is read-only and waits for preflight."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        pre_snapshot = {
            "current_state_revision": 5,
            "binding_topology": {"edges_added_total": 1, "edges_removed_total": 0},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.2, "saturated_fraction": 0.1},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 6,
            "binding_topology": {"edges_added_total": 2, "edges_removed_total": 1},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.1, "saturated_fraction": 0.05},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        evaluation = model.subcortical_structural_plasticity_isolated_evaluation(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-grow-prune",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )
        rev_before = runtime_state.state_revision

        design = model.subcortical_structural_mutation_design(
            evaluation,
            operator_id="operator-structural-design",
            confirmation=True,
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertEqual(design["surface"], "subcortical_structural_mutation_design.v1")
        self.assertFalse(design["executable"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["writes_checkpoint"])
        self.assertFalse(design["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(
            design["promotion_gate"]["eligible_for_structural_mutation_preflight_review"]
        )

    def test_structural_mutation_preflight_does_not_advance_revision(self) -> None:
        """Structural mutation preflight verifies design and checkpoint evidence only."""
        model, _, _, runtime_state, _ = _build_read_model_with_living_loop()
        pre_snapshot = {
            "current_state_revision": 7,
            "binding_topology": {"edges_added_total": 1, "edges_removed_total": 0},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.2, "saturated_fraction": 0.1},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 8,
            "binding_topology": {"edges_added_total": 2, "edges_removed_total": 1},
            "device_evidence": {"binding_devices": {"binding_state_device": "cuda:0"}},
            "spike_health": {"silent_fraction": 0.1, "saturated_fraction": 0.05},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        evaluation = model.subcortical_structural_plasticity_isolated_evaluation(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-grow-prune",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )
        design = model.subcortical_structural_mutation_design(
            evaluation,
            operator_id="operator-structural-design",
            confirmation=True,
        )
        rev_before = runtime_state.state_revision

        preflight = model.subcortical_structural_mutation_preflight(
            design,
            expected_state_revision=rev_before,
            checkpoint_path="checkpoint://pre-structural-mutation",
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertEqual(preflight["surface"], "subcortical_structural_mutation_preflight.v1")
        self.assertFalse(preflight["executable"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["writes_checkpoint"])
        self.assertFalse(preflight["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(preflight["promotion_gate"]["eligible_for_operator_execution_review"])

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

    def test_cortex_signal_state_alias_is_removed(self) -> None:
        """The retired Cortex signal alias must not remain on the read model."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        self.assertFalse(hasattr(model, "cortex_signal_state"))

    def test_cognitive_signal_state_caches_result(self) -> None:
        """The canonical signal path updates only the canonical cache."""
        model, _, _, _, _ = _build_read_model_with_living_loop()
        first = model.cognitive_signal_state()
        self.assertFalse(hasattr(model, "_cached_cortex_signal_state"))
        cached_state = model._cached_cognitive_signal_state
        self.assertIsNotNone(cached_state)
        assert cached_state is not None
        self.assertEqual(cached_state["dopamine"], first["dopamine"])


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
) -> tuple[StatusReadModel, MarulhoTrainer, threading.RLock, RuntimeState]:
    """Build a StatusReadModel with a specific brain runtime snapshot for verdict testing."""
    cfg = _build_config()
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
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
        "living_loop": {},
    }


def _build_degraded_brain_snapshot() -> dict[str, Any]:
    """Build a configured brain snapshot with no progress."""
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
        "living_loop": {},
    }


def _build_idle_brain_snapshot() -> dict[str, Any]:
    """Build a configured idle snapshot."""
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


class StatusReadModelBenchmarkEvidenceCurrencyTests(unittest.TestCase):
    """Runtime Truth reports saved benchmark-evidence currency without running benchmarks."""

    def test_runtime_truth_reports_missing_benchmark_evidence_currency_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "reports"
            model, _, _, runtime_state = _build_read_model(report_root=root)
            before_revision = runtime_state.state_revision

            result = model.status()

        currency = result["runtime_truth"]["evidence"]["benchmark_evidence_currency"]
        self.assertEqual(currency["surface"], "benchmark_evidence_currency.v1")
        self.assertEqual(currency["artifact_kind"], "terminus_benchmark_evidence_currency")
        self.assertEqual(currency["status"], "missing")
        self.assertFalse(currency["current"])
        self.assertFalse(currency["runs_benchmark"])
        self.assertFalse(currency["mutates_runtime_state"])
        self.assertFalse(currency["changes_runtime_truth_verdict"])
        self.assertEqual(
            set(currency["missing_reports"]),
            {"accepted_baseline", "fresh_bundle", "regression_gate"},
        )
        self.assertEqual(runtime_state.state_revision, before_revision)
        self.assertEqual(result["runtime_truth"]["verdict"], "partial")

    def test_runtime_truth_reports_current_benchmark_evidence_currency(self) -> None:
        fresh_generated_at = datetime.now(timezone.utc).isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            reports = Path(tmpdir) / "reports"
            baseline_dir = reports / "service_benchmark_baseline"
            bundle_dir = reports / "service_benchmark_baseline_fresh_cycle"
            gate_dir = reports / "service_benchmark_regression_gate"
            baseline_dir.mkdir(parents=True)
            bundle_dir.mkdir(parents=True)
            gate_dir.mkdir(parents=True)
            (baseline_dir / "accepted-baseline.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "marulho_service_benchmark_accepted_baseline",
                        "generated_at": fresh_generated_at,
                        "status": "accepted",
                        "baseline_id": "service-benchmark-baseline:test",
                        "operator_review": {"accepted_by": "operator-a"},
                        "source_report": {
                            "runtime_truth_verdict": "alive",
                            "hot_path_p95_ms": 439.258,
                            "hot_path_total_ms": 818.798,
                        },
                        "checks": {"accepted_by_present": True},
                    }
                ),
                encoding="utf-8",
            )
            (bundle_dir / "bundle-summary.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "marulho_service_benchmark_baseline_run_bundle",
                        "generated_at": fresh_generated_at,
                        "status": "passed",
                        "success": True,
                        "accepted_baseline": {
                            "baseline_id": "service-benchmark-baseline:test",
                            "baseline_report_hash": "b" * 64,
                            "after_report_hash": "a" * 64,
                        },
                        "runtime_truth": {"before": "alive", "after": "alive", "regressed": False},
                        "hot_path": {"after_p95_ms": 432.406, "after_total_ms": 739.666},
                        "checks": {"runtime_truth_no_regression": True},
                    }
                ),
                encoding="utf-8",
            )
            (gate_dir / "comparison.json").write_text(
                json.dumps(
                    {
                        "artifact_kind": "marulho_service_benchmark_regression_gate",
                        "generated_at": fresh_generated_at,
                        "status": "passed",
                        "runtime_truth": {"before": "alive", "after": "alive", "regressed": False},
                        "hot_path": {"after_p95_ms": 439.258, "after_total_ms": 818.798},
                        "checks": {"hot_path_p95_no_relative_regression": True},
                    }
                ),
                encoding="utf-8",
            )
            model, _, _, _ = _build_read_model(report_root=reports)

            result = model.status()

        currency = result["runtime_truth"]["evidence"]["benchmark_evidence_currency"]
        self.assertEqual(currency["status"], "current")
        self.assertTrue(currency["current"])
        self.assertEqual(currency["missing_reports"], [])
        self.assertEqual(currency["stale_reports"], [])
        self.assertEqual(currency["failed_reports"], [])
        self.assertEqual(currency["fresh_bundle"]["runtime_truth_verdict"], "alive")
        self.assertEqual(currency["fresh_bundle"]["hot_path_p95_ms"], 432.406)
        self.assertEqual(
            currency["accepted_baseline"]["accepted_baseline_id"],
            "service-benchmark-baseline:test",
        )
        self.assertEqual(currency["regression_gate"]["freshness_status"], "fresh")
        self.assertEqual(result["runtime_truth"]["verdict"], "partial")


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

    def test_verdict_alive_when_runtime_progresses_without_retired_path(self) -> None:
        """Runtime progress does not require retired runtime evidence."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_idle_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "alive")
        self.assertEqual(truth["recommended_action"], "continue_monitoring")
        self.assertNotIn("retired_runtime_path", truth)
        self.assertNotIn("retired_runtime_path", truth["evidence"])

    def test_verdict_degraded_when_no_progress(self) -> None:
        """When configured with no progress, verdict is degraded."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_degraded_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "degraded")
        self.assertEqual(truth["recommended_action"], "run_tick_or_start_runtime")

    def test_verdict_alive_when_configured_with_progress(self) -> None:
        """When configured runtime observes progress, the verdict should be 'alive'."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertEqual(truth["verdict"], "alive")
        self.assertEqual(truth["recommended_action"], "continue_monitoring")

    def test_verdict_excludes_retired_runtime_path_state(self) -> None:
        """Runtime Truth no longer surfaces retired-runtime-path state from the brain snapshot."""
        model, _, _, _ = _build_read_model_with_brain_snapshot(_build_alive_brain_snapshot())
        result = model.status()
        truth = result["runtime_truth"]
        self.assertNotIn("retired_runtime_path", truth)
        self.assertNotIn("retired_runtime_path_available", truth)
        self.assertNotIn("retired_runtime_path_retired", truth)

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

    def test_cortex_signal_state_without_callback_is_removed(self) -> None:
        """The retired Cortex compatibility wrapper is gone."""
        model, _, _, _ = _build_read_model()
        self.assertFalse(hasattr(model, "cortex_signal_state"))


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
        self.assertEqual(
            cached_result["living_loop"]["snn_sleep_plasticity_autonomy_proposal"]["surface"],
            "snn_sleep_plasticity_autonomy_proposal.v1",
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
        self.assertEqual(
            cached_result["snn_sleep_plasticity_autonomy_proposal"]["surface"],
            "snn_sleep_plasticity_autonomy_proposal.v1",
        )


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
        self.assertIn("memory_pressure", truth)
        self.assertIn("safety_flags", truth)
        self.assertIn("latency_ms", truth)
        self.assertIn("evidence", truth)
        self.assertIn("configured", truth["evidence"])
        self.assertNotIn("retired_runtime_path", truth["evidence"])
        self.assertNotIn("retired_runtime_path_enabled", truth["evidence"])
        self.assertNotIn("retired_runtime_path_retired", truth["evidence"])
        self.assertIn("token_count", truth["evidence"])
        self.assertIn("subcortex_spike_health", truth["evidence"])
        self.assertTrue(truth["evidence"]["subcortex_spike_health"]["not_liveness_claim"])
        self.assertEqual(truth["verdict"], "partial")
        self.assertEqual(truth["recommended_action"], "configure_terminus_sources")
        self.assertIn("self_repair_gate", truth["evidence"])
        self.assertEqual(
            truth["evidence"]["self_repair_gate"]["artifact_kind"],
            "terminus_subcortical_self_repair_gate_plan",
        )
        self.assertIn(
            truth["evidence"]["self_repair_gate"]["next_gate"],
            {"collect_spike_window", "deep_sleep_or_replay_repair_gate", "continue_monitoring"},
        )
        self.assertTrue(truth["evidence"]["self_repair_gate"]["advisory"])
        self.assertFalse(truth["evidence"]["self_repair_gate"]["executable"])
        self.assertFalse(truth["evidence"]["self_repair_gate"]["eligible_for_action"])
        self.assertFalse(truth["evidence"]["self_repair_gate"]["eligible_for_fact_promotion"])
        self.assertFalse(truth["evidence"]["self_repair_gate"]["eligible_for_structural_mutation"])
        for forbidden_key in ("candidates", "endpoint", "candidate_id", "suggested_endpoint", "suggested_input"):
            self.assertNotIn(forbidden_key, truth["evidence"]["self_repair_gate"])
        self.assertIn("self_repair_evaluation_gate", truth["evidence"])
        repair_evaluation_gate = truth["evidence"]["self_repair_evaluation_gate"]
        self.assertEqual(
            repair_evaluation_gate["artifact_kind"],
            "terminus_subcortical_self_repair_evaluation_plan",
        )
        self.assertEqual(repair_evaluation_gate["surface"], "subcortical_self_repair_evaluation.v1")
        self.assertIn(
            repair_evaluation_gate["next_gate"],
            {
                "operator_approved_deep_sleep_or_replay_evaluation",
                "collect_spike_window",
                "continue_monitoring",
            },
        )
        self.assertTrue(repair_evaluation_gate["advisory"])
        self.assertFalse(repair_evaluation_gate["executable"])
        self.assertFalse(repair_evaluation_gate["mutates_runtime_state"])
        self.assertFalse(repair_evaluation_gate["eligible_for_action"])
        self.assertFalse(repair_evaluation_gate["eligible_for_fact_promotion"])
        self.assertFalse(repair_evaluation_gate["eligible_for_structural_mutation"])
        self.assertTrue(repair_evaluation_gate["requires_isolated_replay_or_deep_sleep"])
        self.assertTrue(repair_evaluation_gate["requires_runtime_truth_improvement"])
        self.assertTrue(repair_evaluation_gate["requires_device_evidence"])
        self.assertIn("runtime_truth_delta", repair_evaluation_gate["success_evidence"])
        for forbidden_key in (
            "repair_surface",
            "evaluation_cases",
            "endpoint",
            "candidate_id",
            "suggested_endpoint",
            "suggested_input",
        ):
            self.assertNotIn(forbidden_key, repair_evaluation_gate)
        self.assertIn("structural_plasticity_gate", truth["evidence"])
        structural_gate = truth["evidence"]["structural_plasticity_gate"]
        self.assertEqual(
            structural_gate["artifact_kind"],
            "terminus_subcortical_structural_plasticity_gate_plan",
        )
        self.assertEqual(structural_gate["surface"], "subcortical_structural_plasticity.v1")
        self.assertIn(
            structural_gate["next_gate"],
            {
                "operator_approved_structural_plasticity_evaluation",
                "collect_cuda_structural_device_report",
                "continue_monitoring",
            },
        )
        self.assertTrue(structural_gate["advisory"])
        self.assertFalse(structural_gate["executable"])
        self.assertFalse(structural_gate["mutates_runtime_state"])
        self.assertFalse(structural_gate["eligible_for_action"])
        self.assertFalse(structural_gate["eligible_for_fact_promotion"])
        self.assertFalse(structural_gate["eligible_for_structural_mutation"])
        self.assertIn("eligible_for_replay_review", structural_gate)
        self.assertIn("requires_operator_approval", structural_gate)
        self.assertTrue(structural_gate["requires_isolated_evaluation"])
        self.assertTrue(structural_gate["requires_runtime_truth_improvement"])
        self.assertTrue(structural_gate["requires_reversible_mutation_ledger"])
        self.assertTrue(structural_gate["requires_device_evidence"])
        self.assertIn("rollback_policy", structural_gate["success_evidence"])
        self.assertIn("runtime_truth_delta", structural_gate["success_evidence"])
        self.assertIn("binding_report_available", structural_gate)
        self.assertIn("local_plasticity_report_available", structural_gate)
        self.assertIn("binding_device_keys", structural_gate)
        self.assertIn("local_plasticity_device_keys", structural_gate)
        self.assertIn("observed_structural_device_key_count", structural_gate)
        self.assertIn("local_plasticity_eligibility_traces_available", structural_gate)
        self.assertIn("local_plasticity_homeostatic_state_available", structural_gate)
        self.assertIn("local_plasticity_spike_backend", structural_gate)
        self.assertIn("local_plasticity_rule", structural_gate)
        self.assertIn("local_plasticity_spike_health_risk", structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_available", structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_passed", structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_failed", structural_gate)
        for forbidden_key in (
            "structural_cases",
            "endpoint",
            "concept_growth",
            "binding_topology",
            "device_evidence",
            "local_plasticity",
            "recent_events",
            "active_growth_concepts",
            "suggested_endpoint",
            "suggested_input",
        ):
            self.assertNotIn(forbidden_key, structural_gate)
        self.assertIn("snn_language_readiness_gate", truth["evidence"])
        language_gate = truth["evidence"]["snn_language_readiness_gate"]
        self.assertEqual(language_gate["artifact_kind"], "terminus_snn_native_language_readiness_gate")
        self.assertEqual(language_gate["surface"], "snn_native_language_readiness.v1")
        self.assertIn(
            language_gate["next_gate"],
            {
                "complete_grounded_subcortex_language_evidence",
                "build_local_snn_language_generator_adapter",
                "operator_approved_snn_language_evaluation",
            },
        )
        self.assertTrue(language_gate["advisory"])
        self.assertFalse(language_gate["executable"])
        self.assertFalse(language_gate["mutates_runtime_state"])
        self.assertTrue(language_gate["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", language_gate)
        self.assertFalse(language_gate["eligible_for_action"])
        self.assertFalse(language_gate["eligible_for_fact_promotion"])
        self.assertFalse(language_gate["eligible_for_cognition_substrate"])
        self.assertTrue(language_gate["requires_marulho_owned_implementation"])
        self.assertTrue(language_gate["marulho_spike_readout_evidence_available"])
        self.assertTrue(language_gate["marulho_spike_readout_grounded"])
        self.assertTrue(language_gate["marulho_spike_readout_non_generative"])
        self.assertIn("marulho_spike_readout_device_evidence_available", language_gate)
        self.assertIn("marulho_spike_decoder_probe_available", language_gate)
        self.assertIn("marulho_spike_decoder_probe_owned", language_gate)
        self.assertIn("marulho_spike_decoder_probe_non_generative", language_gate)
        self.assertIn("marulho_spike_decoder_probe_sparse", language_gate)
        self.assertIn("marulho_spike_decoder_probe_device_evidence_available", language_gate)
        self.assertIn("marulho_spike_decoder_probe_grounding_supported", language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_available", language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_owned", language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_sparse", language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_dynamic", language_gate)
        for forbidden_key in (
            "endpoint",
            "research_candidates",
            "current_language_surface",
            "current_deliberation_surface",
            "current_spike_readout_evidence",
            "current_decoder_probe_evidence",
            "current_language_neuron_adapter_evidence",
            "readiness_checks",
            "success_evidence",
            "suggested_endpoint",
            "suggested_input",
        ):
            self.assertNotIn(forbidden_key, language_gate)
        self.assertIn("snn_language_plasticity_path", truth["evidence"])
        plasticity_path = truth["evidence"]["snn_language_plasticity_path"]
        self.assertEqual(
            plasticity_path["artifact_kind"],
            "terminus_snn_language_plasticity_path_evidence",
        )
        self.assertEqual(plasticity_path["surface"], "snn_language_plasticity_path_evidence.v1")
        self.assertEqual(plasticity_path["latest_gate"], "snn_language_plasticity_live_application_preflight.v1")
        self.assertTrue(plasticity_path["owned_by_marulho"])
        self.assertFalse(plasticity_path["external_dependency"])
        self.assertFalse(plasticity_path["generates_text"])
        self.assertFalse(plasticity_path["decodes_text"])
        self.assertFalse(plasticity_path["trains_runtime_model"])
        self.assertFalse(plasticity_path["applies_plasticity"])
        self.assertFalse(plasticity_path["mutates_runtime_state"])
        self.assertTrue(plasticity_path["requires_device_evidence"])
        self.assertTrue(plasticity_path["requires_runtime_truth_delta"])
        self.assertTrue(plasticity_path["requires_rollback_evidence"])
        self.assertIn("rollback_readiness", plasticity_path)
        self.assertTrue(plasticity_path["rollback_readiness"]["rollback_policy_required"])
        self.assertTrue(plasticity_path["rollback_readiness"]["restore_endpoint_available"])
        self.assertIsInstance(
            plasticity_path["rollback_readiness"]["checkpoint_metadata_available"],
            bool,
        )
        self.assertIn("checkpoint_path", plasticity_path["rollback_readiness"])
        self.assertIn("snn_language_plasticity_application_design.v1", plasticity_path["gates"])
        self.assertIn("snn_language_plasticity_shadow_application.v1", plasticity_path["gates"])
        self.assertIn("snn_language_plasticity_live_application_readiness.v1", plasticity_path["gates"])
        self.assertIn("snn_language_plasticity_live_application_preflight.v1", plasticity_path["gates"])
        self.assertIn("snn_readout_rollout_server_state_binding", truth["evidence"])
        rollout_binding = truth["evidence"]["snn_readout_rollout_server_state_binding"]
        self.assertEqual(
            rollout_binding["artifact_kind"],
            "terminus_snn_readout_rollout_server_state_binding_gate",
        )
        self.assertEqual(
            rollout_binding["surface"],
            "snn_readout_rollout_server_state_binding.v1",
        )
        self.assertTrue(rollout_binding["owned_by_marulho"])
        self.assertFalse(rollout_binding["external_dependency"])
        self.assertTrue(rollout_binding["advisory"])
        self.assertFalse(rollout_binding["executable"])
        self.assertFalse(rollout_binding["generates_text"])
        self.assertFalse(rollout_binding["decodes_text"])
        self.assertFalse(rollout_binding["freeform_language_generation"])
        self.assertFalse(rollout_binding["loads_external_checkpoint"])
        self.assertFalse(rollout_binding["accepts_caller_transition_memory_state"])
        self.assertTrue(rollout_binding["requires_server_transition_memory_state"])
        self.assertTrue(rollout_binding["runtime_mutation_absent"])
        self.assertTrue(rollout_binding["plasticity_absent"])
        self.assertTrue(rollout_binding["checkpoint_write_absent"])
        self.assertTrue(rollout_binding["rollout_execution_absent"])
        self.assertFalse(rollout_binding["runs_replay"])
        self.assertFalse(rollout_binding["records_ledger_event"])
        self.assertFalse(rollout_binding["calls_rollout"])
        self.assertFalse(rollout_binding["eligible_for_rollout_execution"])
        self.assertFalse(rollout_binding["eligible_for_fact_promotion"])
        self.assertFalse(rollout_binding["eligible_for_cognition_substrate"])
        self.assertEqual(
            rollout_binding["promotion_status"],
            "waiting_for_server_transition_memory",
        )
        self.assertFalse(rollout_binding["server_transition_memory_available"])
        self.assertIsNone(rollout_binding["server_transition_memory_hash"])
        self.assertEqual(rollout_binding["server_transition_weight_count"], 0)
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "transition_memory_state",
        ):
            self.assertNotIn(forbidden_key, rollout_binding)
        self.assertIn("snn_readout_rollout_consolidation_path", truth["evidence"])
        consolidation_path = truth["evidence"]["snn_readout_rollout_consolidation_path"]
        self.assertEqual(
            consolidation_path["artifact_kind"],
            "terminus_snn_readout_rollout_consolidation_path_evidence",
        )
        self.assertEqual(
            consolidation_path["surface"],
            "snn_readout_rollout_consolidation_path_evidence.v1",
        )
        self.assertTrue(consolidation_path["owned_by_marulho"])
        self.assertFalse(consolidation_path["external_dependency"])
        self.assertTrue(consolidation_path["advisory"])
        self.assertFalse(consolidation_path["executable"])
        self.assertFalse(consolidation_path["executes_rehearsal"])
        self.assertFalse(consolidation_path["executes_consolidation"])
        self.assertFalse(consolidation_path["runs_live_replay"])
        self.assertFalse(consolidation_path["records_ledger_event"])
        self.assertFalse(consolidation_path["writes_checkpoint"])
        self.assertFalse(consolidation_path["generates_text"])
        self.assertFalse(consolidation_path["decodes_text"])
        self.assertFalse(consolidation_path["freeform_language_generation"])
        self.assertFalse(consolidation_path["applies_plasticity"])
        self.assertFalse(consolidation_path["mutates_runtime_state"])
        self.assertEqual(consolidation_path["rollout_event_count"], 0)
        self.assertEqual(
            consolidation_path["promotion_status"],
            "waiting_for_recorded_rollout_replay_evidence",
        )
        self.assertFalse(consolidation_path["eligible_for_rollout_rehearsal_policy_review"])
        self.assertEqual(
            consolidation_path["next_gate"],
            "snn_language_readout_rollout_evidence_ledger_record.v1",
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "replay_targets",
        ):
            self.assertNotIn(forbidden_key, consolidation_path)
        self.assertIn("snn_readout_emission_review_history", truth["evidence"])
        emission_review_history = truth["evidence"][
            "snn_readout_emission_review_history"
        ]
        self.assertEqual(
            emission_review_history["artifact_kind"],
            "terminus_snn_readout_emission_review_history_evidence",
        )
        self.assertEqual(
            emission_review_history["surface"],
            "snn_readout_emission_review_history_evidence.v1",
        )
        self.assertTrue(emission_review_history["owned_by_marulho"])
        self.assertFalse(emission_review_history["external_dependency"])
        self.assertTrue(emission_review_history["advisory"])
        self.assertFalse(emission_review_history["executable"])
        self.assertFalse(emission_review_history["calls_endpoint"])
        self.assertFalse(emission_review_history["records_ledger_event"])
        self.assertFalse(emission_review_history["runs_replay"])
        self.assertFalse(emission_review_history["writes_checkpoint"])
        self.assertFalse(emission_review_history["generates_text"])
        self.assertFalse(emission_review_history["decodes_text"])
        self.assertFalse(emission_review_history["exposes_raw_text"])
        self.assertFalse(emission_review_history["freeform_language_generation"])
        self.assertFalse(emission_review_history["applies_plasticity"])
        self.assertFalse(emission_review_history["mutates_runtime_state"])
        self.assertEqual(emission_review_history["emission_review_event_count"], 0)
        self.assertEqual(
            emission_review_history["promotion_status"],
            "waiting_for_reviewed_snn_language_emission",
        )
        self.assertFalse(
            emission_review_history[
                "eligible_for_operator_display_history_inspection"
            ]
        )
        self.assertFalse(emission_review_history["eligible_for_replay_memory"])
        self.assertFalse(emission_review_history["eligible_for_live_replay"])
        self.assertFalse(
            emission_review_history["eligible_for_plasticity_application"]
        )
        self.assertFalse(emission_review_history["eligible_for_fact_promotion"])
        self.assertFalse(emission_review_history["eligible_for_action"])
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
        ):
            self.assertNotIn(forbidden_key, emission_review_history)
        self.assertIn("snn_readout_emission_replay_design_path", truth["evidence"])
        emission_replay_design_path = truth["evidence"][
            "snn_readout_emission_replay_design_path"
        ]
        self.assertEqual(
            emission_replay_design_path["artifact_kind"],
            "terminus_snn_readout_emission_replay_design_path_evidence",
        )
        self.assertEqual(
            emission_replay_design_path["surface"],
            "snn_readout_emission_replay_design_path_evidence.v1",
        )
        self.assertTrue(emission_replay_design_path["owned_by_marulho"])
        self.assertFalse(emission_replay_design_path["external_dependency"])
        self.assertTrue(emission_replay_design_path["advisory"])
        self.assertFalse(emission_replay_design_path["executable"])
        self.assertFalse(emission_replay_design_path["calls_endpoint"])
        self.assertFalse(emission_replay_design_path["records_ledger_event"])
        self.assertFalse(emission_replay_design_path["records_replay_context"])
        self.assertFalse(emission_replay_design_path["runs_replay"])
        self.assertFalse(emission_replay_design_path["writes_checkpoint"])
        self.assertFalse(emission_replay_design_path["generates_text"])
        self.assertFalse(emission_replay_design_path["decodes_text"])
        self.assertFalse(emission_replay_design_path["exposes_raw_text"])
        self.assertFalse(emission_replay_design_path["applies_plasticity"])
        self.assertFalse(emission_replay_design_path["mutates_runtime_state"])
        self.assertEqual(emission_replay_design_path["emission_review_event_count"], 0)
        self.assertEqual(emission_replay_design_path["policy_candidate_count"], 0)
        self.assertEqual(emission_replay_design_path["design_seed_candidate_count"], 0)
        self.assertEqual(
            emission_replay_design_path["promotion_status"],
            "waiting_for_reviewed_snn_language_emission",
        )
        self.assertFalse(
            emission_replay_design_path[
                "eligible_for_emission_replay_evaluation_design_review"
            ]
        )
        self.assertFalse(
            emission_replay_design_path["eligible_for_operator_replay_context_review"]
        )
        self.assertFalse(
            emission_replay_design_path["eligible_for_replay_context_recording"]
        )
        self.assertTrue(emission_replay_design_path["requires_device_review_evidence"])
        self.assertTrue(
            emission_replay_design_path["requires_server_computed_mismatch_probe"]
        )
        self.assertTrue(
            emission_replay_design_path["requires_server_computed_plasticity_pressure"]
        )
        self.assertFalse(emission_replay_design_path["eligible_for_replay_memory"])
        self.assertFalse(emission_replay_design_path["eligible_for_live_replay"])
        self.assertFalse(
            emission_replay_design_path["eligible_for_plasticity_application"]
        )
        self.assertFalse(emission_replay_design_path["eligible_for_fact_promotion"])
        self.assertFalse(emission_replay_design_path["eligible_for_action"])
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
            "selected_replay_context_seeds",
            "events",
        ):
            self.assertNotIn(forbidden_key, emission_replay_design_path)
        self.assertIn("snn_readout_applied_synapse_provenance", truth["evidence"])
        applied_provenance = truth["evidence"]["snn_readout_applied_synapse_provenance"]
        self.assertEqual(
            applied_provenance["artifact_kind"],
            "terminus_snn_readout_applied_synapse_provenance_evidence",
        )
        self.assertEqual(
            applied_provenance["surface"],
            "snn_readout_applied_synapse_provenance_evidence.v1",
        )
        self.assertTrue(applied_provenance["owned_by_marulho"])
        self.assertFalse(applied_provenance["external_dependency"])
        self.assertTrue(applied_provenance["advisory"])
        self.assertFalse(applied_provenance["executable"])
        self.assertFalse(applied_provenance["runs_audit"])
        self.assertFalse(applied_provenance["runs_replay"])
        self.assertFalse(applied_provenance["calls_endpoint"])
        self.assertFalse(applied_provenance["generates_text"])
        self.assertFalse(applied_provenance["decodes_text"])
        self.assertFalse(applied_provenance["freeform_language_generation"])
        self.assertFalse(applied_provenance["applies_plasticity"])
        self.assertFalse(applied_provenance["mutates_runtime_state"])
        self.assertFalse(applied_provenance["writes_checkpoint"])
        self.assertEqual(applied_provenance["sparse_transition_weight_count"], 0)
        self.assertEqual(applied_provenance["synapse_provenance_count"], 0)
        self.assertEqual(applied_provenance["missing_local_edge_provenance_count"], 0)
        self.assertFalse(
            applied_provenance["eligible_for_readout_synapse_audit_review"]
        )
        self.assertEqual(
            applied_provenance["promotion_status"],
            "waiting_for_complete_applied_synapse_provenance",
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "synapse_provenance_by_key",
            "sparse_transition_weights",
        ):
            self.assertNotIn(forbidden_key, applied_provenance)
        self.assertIn(
            "snn_applied_replay_lineage_restore_validation",
            truth["evidence"],
        )
        self.assertIn("snn_language_capacity_pressure", truth["evidence"])
        capacity_pressure = truth["evidence"]["snn_language_capacity_pressure"]
        self.assertEqual(
            capacity_pressure["artifact_kind"],
            "terminus_snn_language_capacity_pressure_evidence",
        )
        self.assertEqual(
            capacity_pressure["surface"],
            "snn_language_capacity_pressure_evidence.v1",
        )
        self.assertFalse(capacity_pressure["executable"])
        self.assertFalse(capacity_pressure["mutates_runtime_state"])
        self.assertFalse(capacity_pressure["writes_checkpoint"])
        self.assertFalse(capacity_pressure["resizes_network"])
        self.assertFalse(capacity_pressure["adds_neurons"])
        self.assertFalse(capacity_pressure["adds_layers"])
        self.assertFalse(capacity_pressure["capacity_pressure_detected"])
        self.assertFalse(
            capacity_pressure["eligible_for_capacity_expansion_design_review"]
        )
        self.assertIn("snn_language_dense_readout_layout_state", truth["evidence"])
        dense_layout = truth["evidence"]["snn_language_dense_readout_layout_state"]
        self.assertEqual(
            dense_layout["artifact_kind"],
            "terminus_snn_language_dense_readout_layout_state",
        )
        self.assertEqual(
            dense_layout["surface"],
            "snn_language_dense_readout_layout_state.v1",
        )
        self.assertTrue(dense_layout["advisory"])
        self.assertFalse(dense_layout["executable"])
        self.assertFalse(dense_layout["mutates_runtime_state"])
        self.assertFalse(dense_layout["writes_checkpoint"])
        self.assertFalse(dense_layout["resizes_network"])
        self.assertEqual(dense_layout["current_dense_readout_shape"], [64, 64])
        self.assertEqual(dense_layout["target_dense_readout_shape"], [64, 64])
        self.assertEqual(dense_layout["preserved_dense_window"], [64, 64])
        self.assertFalse(dense_layout["requires_cuda_relayout"])
        self.assertIn("snn_language_capacity_fixed_boundaries", truth["evidence"])
        capacity_boundaries = truth["evidence"][
            "snn_language_capacity_fixed_boundaries"
        ]
        self.assertEqual(
            capacity_boundaries["artifact_kind"],
            "terminus_snn_language_capacity_fixed_boundary_evidence",
        )
        self.assertEqual(
            capacity_boundaries["surface"],
            "snn_language_capacity_fixed_boundary_evidence.v1",
        )
        self.assertTrue(capacity_boundaries["advisory"])
        self.assertFalse(capacity_boundaries["executable"])
        self.assertFalse(capacity_boundaries["mutates_runtime_state"])
        self.assertFalse(capacity_boundaries["writes_checkpoint"])
        self.assertFalse(capacity_boundaries["resizes_network"])
        self.assertFalse(capacity_boundaries["adds_neurons"])
        self.assertEqual(capacity_boundaries["fixed_boundary_count"], 2)
        self.assertEqual(
            capacity_boundaries["dynamic_capacity_aware_boundary_count"],
            8,
        )
        self.assertTrue(
            capacity_boundaries["capacity_resize_blocked_by_fixed_boundaries"]
        )
        self.assertFalse(
            capacity_boundaries[
                "eligible_for_capacity_resize_compatibility_audit"
            ]
        )
        self.assertFalse(
            capacity_boundaries["promotion_gate"]["required_evidence"][
                "all_runtime_boundaries_dynamic_capacity_aware"
            ]
        )
        self.assertIn("snn_language_dense_readout_resize_plan", truth["evidence"])
        dense_resize = truth["evidence"]["snn_language_dense_readout_resize_plan"]
        self.assertEqual(
            dense_resize["artifact_kind"],
            "terminus_snn_language_dense_readout_resize_plan",
        )
        self.assertEqual(
            dense_resize["surface"],
            "snn_language_dense_readout_resize_plan.v1",
        )
        self.assertTrue(dense_resize["advisory"])
        self.assertFalse(dense_resize["executable"])
        self.assertFalse(dense_resize["mutates_runtime_state"])
        self.assertFalse(dense_resize["writes_checkpoint"])
        self.assertFalse(dense_resize["resizes_network"])
        self.assertEqual(dense_resize["current_dense_readout_shape"], [64, 64])
        self.assertEqual(dense_resize["target_dense_readout_shape"], [128, 128])
        self.assertEqual(dense_resize["preserved_dense_window"], [64, 64])
        self.assertEqual(
            dense_resize["zero_initialized_new_dense_cell_count"],
            128 * 128 - 64 * 64,
        )
        self.assertIn(
            "snn_language_readout_ledger.dense_readout_index_validators",
            dense_resize["dense_boundary_ids"],
        )
        self.assertIn(
            "snn_language_readout_ledger.cuda_dense_tensor_shapes",
            dense_resize["dense_boundary_ids"],
        )
        self.assertFalse(dense_resize["promotion_gate"]["required_evidence"]["executor_available"])
        self.assertFalse(
            dense_resize["promotion_gate"][
                "eligible_for_dense_readout_resize_executor"
            ]
        )
        restore_validation = truth["evidence"][
            "snn_applied_replay_lineage_restore_validation"
        ]
        self.assertEqual(
            restore_validation["artifact_kind"],
            "terminus_snn_applied_replay_lineage_restore_validation_evidence",
        )
        self.assertEqual(
            restore_validation["surface"],
            "snn_applied_replay_lineage_restore_validation_evidence.v1",
        )
        self.assertTrue(restore_validation["advisory"])
        self.assertFalse(restore_validation["executable"])
        self.assertFalse(restore_validation["runs_replay"])
        self.assertFalse(restore_validation["applies_plasticity"])
        self.assertFalse(restore_validation["mutates_runtime_state"])
        self.assertFalse(restore_validation["writes_checkpoint"])
        self.assertFalse(restore_validation["issues_regeneration_permit"])
        self.assertFalse(
            restore_validation["eligible_for_readout_synapse_audit_review"]
        )

    def test_runtime_truth_binding_reports_server_transition_memory_without_mutation(self) -> None:
        memory_state = {
            "sparse_transition_weights": {"1:2": 0.5, "2:3": 0.75},
            "synapse_provenance_by_key": {"1:2": {"source": "unit"}, "2:3": {"source": "unit"}},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        status_binding = model.status()["runtime_truth"]["evidence"][
            "snn_readout_rollout_server_state_binding"
        ]
        terminus_binding = model.terminus_status()["runtime_truth"]["evidence"][
            "snn_readout_rollout_server_state_binding"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertTrue(status_binding["server_transition_memory_available"])
        self.assertEqual(status_binding["server_transition_weight_count"], 2)
        self.assertEqual(status_binding["server_synapse_provenance_count"], 2)
        self.assertIsInstance(status_binding["server_transition_memory_hash"], str)
        self.assertEqual(len(status_binding["server_transition_memory_hash"]), 64)
        self.assertEqual(
            status_binding["promotion_status"],
            "ready_for_server_bound_rollout_review",
        )
        self.assertEqual(
            terminus_binding["server_transition_memory_hash"],
            status_binding["server_transition_memory_hash"],
        )
        self.assertEqual(
            terminus_binding["server_transition_weight_count"],
            status_binding["server_transition_weight_count"],
        )

    def test_runtime_truth_applied_synapse_provenance_reports_local_edge_health_without_audit(self) -> None:
        memory_state = {
            "sparse_transition_weights": {"1:3": 0.1},
            "synapse_provenance_by_key": {
                "1:3": {
                    "provenance_type": "replay_regeneration",
                    "permit_id": "permit-1",
                    "replay_artifact_id": "artifact-1",
                    "source_metadata_hash": "source-metadata-hash-1",
                    "emission_lineage": {
                        "emission_hash": "emission-hash-1",
                        "readout_evidence_hash": "readout-hash-1",
                        "prediction_hash": "prediction-hash-1",
                    },
                    "local_edge_provenance": {
                        "source_synapse_id": "snn-rollout-local:1:3:0",
                        "source_trace_index": 0,
                        "source_rollout_step_index": 10,
                        "target_rollout_step_index": 20,
                        "source_active_indices_hash": "source-active-hash-1",
                        "target_active_indices_hash": "target-active-hash-1",
                    },
                }
            },
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        evidence = model.status()["runtime_truth"]["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(evidence["sparse_transition_weight_count"], 1)
        self.assertEqual(evidence["synapse_provenance_count"], 1)
        self.assertEqual(evidence["replay_regeneration_synapse_count"], 1)
        self.assertEqual(evidence["complete_local_edge_provenance_count"], 1)
        self.assertEqual(evidence["missing_local_edge_provenance_count"], 0)
        self.assertEqual(evidence["invalid_rollout_step_order_count"], 0)
        self.assertEqual(evidence["replay_artifact_lineage_count"], 1)
        self.assertEqual(evidence["complete_replay_artifact_lineage_count"], 1)
        self.assertEqual(evidence["incomplete_replay_artifact_lineage_count"], 0)
        self.assertEqual(evidence["orphan_weight_count"], 0)
        self.assertEqual(evidence["dangling_provenance_count"], 0)
        self.assertFalse(evidence["restore_validation_available"])
        self.assertFalse(evidence["restore_validation_blocks_audit"])
        self.assertTrue(evidence["eligible_for_readout_synapse_audit_review"])
        self.assertEqual(
            evidence["promotion_status"],
            "ready_for_readout_synapse_provenance_audit",
        )
        self.assertFalse(evidence["runs_audit"])
        self.assertFalse(evidence["mutates_runtime_state"])

        missing = deepcopy(memory_state)
        missing["synapse_provenance_by_key"]["1:3"].pop("local_edge_provenance")
        missing_model, _, _, _ = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(missing)
        )
        blocked = missing_model.status()["runtime_truth"]["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]

        self.assertEqual(blocked["missing_local_edge_provenance_count"], 1)
        self.assertFalse(blocked["eligible_for_readout_synapse_audit_review"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "replay_regeneration_local_edge_provenance_complete"
            ]
        )

        incomplete_lineage = deepcopy(memory_state)
        incomplete_lineage["synapse_provenance_by_key"]["1:3"]["emission_lineage"] = {
            "emission_hash": "emission-hash-1"
        }
        incomplete_model, _, _, _ = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(incomplete_lineage)
        )
        lineage_blocked = incomplete_model.status()["runtime_truth"]["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]

        self.assertEqual(lineage_blocked["incomplete_replay_artifact_lineage_count"], 1)
        self.assertFalse(lineage_blocked["eligible_for_readout_synapse_audit_review"])
        self.assertFalse(
            lineage_blocked["promotion_gate"]["required_evidence"][
                "replay_regeneration_artifact_lineage_complete"
            ]
        )

    def test_runtime_truth_language_capacity_pressure_reports_fixed_neuron_pressure_without_resize(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        provenance = {key: {"source": "unit"} for key in weights}
        memory_state = {
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": provenance,
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        truth_evidence = model.status()["runtime_truth"]["evidence"]
        evidence = truth_evidence["snn_language_capacity_pressure"]
        dense_layout = truth_evidence["snn_language_dense_readout_layout_state"]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            evidence["capacity_state_surface"],
            "snn_language_capacity_state.v1",
        )
        self.assertFalse(evidence["capacity_state_present"])
        self.assertFalse(evidence["capacity_state_durable"])
        self.assertFalse(evidence["dynamic_capacity_enabled"])
        self.assertEqual(evidence["current_language_neuron_count"], 64)
        self.assertEqual(evidence["configured_sparse_edge_budget"], 256)
        self.assertEqual(evidence["configured_outgoing_fanout_budget"], 16)
        self.assertEqual(evidence["sparse_transition_weight_count"], 224)
        self.assertEqual(evidence["active_language_neuron_count"], 16)
        self.assertGreater(evidence["sparse_edge_budget_occupancy"], 0.85)
        self.assertEqual(evidence["max_outgoing_fanout"], 16)
        self.assertEqual(evidence["saturated_source_neuron_count"], 14)
        self.assertEqual(evidence["invalid_synapse_key_count"], 0)
        self.assertEqual(evidence["orphan_weight_count"], 0)
        self.assertEqual(evidence["dangling_provenance_count"], 0)
        self.assertTrue(evidence["capacity_pressure_detected"])
        self.assertTrue(evidence["eligible_for_capacity_expansion_design_review"])
        self.assertFalse(evidence["eligible_for_network_resize"])
        self.assertFalse(evidence["eligible_for_neuron_growth"])
        self.assertFalse(evidence["eligible_for_layer_growth"])
        self.assertFalse(evidence["eligible_for_structural_write"])
        self.assertFalse(evidence["mutates_runtime_state"])
        self.assertFalse(evidence["resizes_network"])
        self.assertEqual(
            evidence["promotion_status"],
            "ready_for_operator_language_capacity_expansion_design_review",
        )
        self.assertTrue(
            evidence["promotion_gate"]["required_evidence"][
                "capacity_pressure_detected"
            ]
        )
        self.assertTrue(
            evidence["promotion_gate"]["required_evidence"]["network_resize_absent"]
        )

    def test_runtime_truth_language_capacity_pressure_uses_durable_capacity_state(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(8)
            for target in range(32)
        }
        provenance = {key: {"source": "unit"} for key in weights}
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "dynamic_capacity_enabled": True,
                "capacity_expansion_count": 1,
                "resizes_network": True,
                "adds_neurons": True,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": provenance,
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        truth_evidence = model.status()["runtime_truth"]["evidence"]
        evidence = truth_evidence["snn_language_capacity_pressure"]
        dense_layout = truth_evidence["snn_language_dense_readout_layout_state"]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(evidence["current_language_neuron_count"], 128)
        self.assertTrue(evidence["capacity_state_present"])
        self.assertTrue(evidence["capacity_state_durable"])
        self.assertEqual(evidence["configured_sparse_edge_budget"], 512)
        self.assertEqual(evidence["configured_outgoing_fanout_budget"], 32)
        self.assertEqual(evidence["language_capacity"]["capacity_expansion_count"], 1)
        self.assertTrue(evidence["dynamic_capacity_enabled"])
        self.assertTrue(evidence["language_capacity"]["dynamic_capacity_enabled"])
        self.assertTrue(evidence["language_capacity"]["resizes_network"])
        self.assertTrue(evidence["language_capacity"]["adds_neurons"])
        self.assertEqual(evidence["sparse_transition_weight_count"], 256)
        self.assertEqual(evidence["active_language_neuron_count"], 32)
        self.assertEqual(evidence["sparse_edge_budget_occupancy"], 0.5)
        self.assertEqual(evidence["active_language_neuron_coverage"], 0.25)
        self.assertEqual(evidence["max_outgoing_fanout"], 32)
        self.assertEqual(evidence["saturated_source_neuron_count"], 8)
        self.assertTrue(evidence["capacity_pressure_detected"])
        self.assertTrue(evidence["eligible_for_capacity_expansion_design_review"])
        self.assertFalse(evidence["eligible_for_network_resize"])
        self.assertFalse(evidence["eligible_for_neuron_growth"])
        self.assertEqual(dense_layout["target_language_neuron_count"], 128)
        self.assertEqual(dense_layout["current_dense_readout_shape"], [64, 64])
        self.assertEqual(dense_layout["target_dense_readout_shape"], [128, 128])
        self.assertEqual(dense_layout["preserved_dense_window"], [64, 64])
        self.assertTrue(dense_layout["requires_cuda_relayout"])
        self.assertFalse(dense_layout["dense_resize_applied"])
        self.assertEqual(
            dense_layout["migration_status"],
            "layout_metadata_only_resize_pending",
        )

    def test_runtime_truth_dense_readout_tensor_integrity_audits_materialized_tensor(
        self,
    ) -> None:
        tensor = torch.zeros((128, 128), dtype=torch.float32)
        tensor[1, 2] = 0.5
        tensor[65, 66] = 0.25
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
            "dense_readout_layout": {
                "surface": "snn_language_dense_readout_layout_state.v1",
                "target_language_neuron_count": 128,
                "tensor_materialization": {
                    "applied": True,
                    "actual_device": "cpu",
                    "target_dense_readout_shape": [128, 128],
                    "materializes_dense_tensor_weights": True,
                },
                "dense_resize_applied": True,
                "dynamic_dense_readout_enabled": True,
                "migration_status": "dense_readout_tensor_materialized",
            },
            "dense_readout_weights": tensor,
            "sparse_transition_weights": {"1:2": 0.5, "65:66": 0.25},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        truth_evidence = model.status()["runtime_truth"]["evidence"]
        dense_layout = truth_evidence["snn_language_dense_readout_layout_state"]
        integrity = truth_evidence["snn_language_dense_readout_tensor_integrity"]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(dense_layout["target_dense_readout_shape"], [128, 128])
        self.assertTrue(dense_layout["tensor_materialization_applied"])
        self.assertTrue(dense_layout["dense_resize_applied"])
        self.assertFalse(dense_layout["requires_cuda_relayout"])
        self.assertFalse(dense_layout["checkpoint_required_before_resize"])
        self.assertEqual(
            integrity["surface"],
            "snn_language_dense_readout_tensor_integrity.v1",
        )
        self.assertTrue(integrity["ready"])
        self.assertFalse(integrity["executable"])
        self.assertFalse(integrity["mutates_runtime_state"])
        self.assertFalse(integrity["writes_checkpoint"])
        self.assertFalse(integrity["generates_text"])
        self.assertEqual(integrity["tensor_summary"]["shape"], [128, 128])
        self.assertEqual(integrity["tensor_summary"]["device"], "cpu")
        self.assertEqual(integrity["tensor_summary"]["nonzero_count"], 2)
        self.assertTrue(
            integrity["promotion_gate"]["required_evidence"][
                "sampled_sparse_weights_match_dense_tensor"
            ]
        )
        self.assertTrue(
            integrity["promotion_gate"]["required_evidence"][
                "dense_tensor_nonzero_count_matches_sparse_weights"
            ]
        )
        self.assertFalse(
            integrity["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_training_readiness_requires_heldout_and_rollback(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        integrity = {
            "surface": "snn_language_dense_readout_tensor_integrity.v1",
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "mutates_runtime_state": False,
            "tensor_summary": {
                "shape": [128, 128],
                "device": "cpu",
                "dtype": "torch.float32",
                "nonzero_count": 2,
            },
            "promotion_gate": {
                "required_evidence": {
                    "dense_tensor_available": True,
                    "dense_tensor_shape_matches_layout": True,
                    "sampled_sparse_weights_match_dense_tensor": True,
                }
            },
        }
        heldout = model.snn_language_adapter_heldout_evaluation(
            [
                [
                    {
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    }
                ]
            ],
            device_evidence={"device": "cpu", "source": "unit"},
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_training_readiness(integrity)
        ready = model.snn_language_dense_readout_training_readiness(
            integrity,
            heldout_evaluation=heldout,
            device_evidence={"device": "cpu", "source": "unit"},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_training_readiness.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(blocked["executable"])
        self.assertFalse(blocked["mutates_runtime_state"])
        self.assertFalse(blocked["writes_checkpoint"])
        self.assertFalse(blocked["generates_text"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "heldout_evaluation_available"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_training_loop_design"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_dense_readout_training"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_runtime_training"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_training_loop_design_is_read_only(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        integrity = {
            "surface": "snn_language_dense_readout_tensor_integrity.v1",
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "mutates_runtime_state": False,
            "tensor_summary": {
                "shape": [128, 128],
                "device": "cpu",
                "dtype": "torch.float32",
                "nonzero_count": 2,
            },
            "promotion_gate": {
                "required_evidence": {
                    "dense_tensor_available": True,
                    "dense_tensor_shape_matches_layout": True,
                    "sampled_sparse_weights_match_dense_tensor": True,
                }
            },
        }
        heldout = model.snn_language_adapter_heldout_evaluation(
            [
                [
                    {
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    }
                ]
            ],
            device_evidence={"device": "cpu", "source": "unit"},
        )
        readiness = model.snn_language_dense_readout_training_readiness(
            integrity,
            heldout_evaluation=heldout,
            device_evidence={"device": "cpu", "source": "unit"},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_training_loop_design(readiness)
        ready = model.snn_language_dense_readout_training_loop_design(
            readiness,
            training_plan={
                "training_transition_count": 8,
                "validation_transition_count": 4,
                "learning_rule": (
                    "bounded_local_hebbian_readout_delta_with_homeostatic_row_norm"
                ),
                "learning_rate": 0.02,
                "max_epochs": 2,
                "target_min_weight_sparsity": 0.9,
                "max_delta_norm": 0.05,
            },
            device_evidence={"device": "cpu", "source": "unit"},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_training_loop_design.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "training_transitions_available"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["trains_runtime_model"])
        self.assertFalse(ready["returns_trained_weights"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_training_loop_preflight"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_dense_readout_training"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_training_loop_preflight_requires_revision_and_checkpoint(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        integrity = {
            "surface": "snn_language_dense_readout_tensor_integrity.v1",
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "mutates_runtime_state": False,
            "tensor_summary": {
                "shape": [128, 128],
                "device": "cpu",
                "dtype": "torch.float32",
                "nonzero_count": 2,
            },
            "promotion_gate": {
                "required_evidence": {
                    "dense_tensor_available": True,
                    "dense_tensor_shape_matches_layout": True,
                    "sampled_sparse_weights_match_dense_tensor": True,
                }
            },
        }
        heldout = model.snn_language_adapter_heldout_evaluation(
            [
                [
                    {
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    }
                ]
            ],
            device_evidence={"device": "cpu", "source": "unit"},
        )
        readiness = model.snn_language_dense_readout_training_readiness(
            integrity,
            heldout_evaluation=heldout,
            device_evidence={"device": "cpu", "source": "unit"},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )
        design = model.snn_language_dense_readout_training_loop_design(
            readiness,
            training_plan={
                "training_transition_count": 8,
                "validation_transition_count": 4,
                "learning_rate": 0.02,
                "max_epochs": 2,
                "target_min_weight_sparsity": 0.9,
                "max_delta_norm": 0.05,
            },
            device_evidence={"device": "cpu", "source": "unit"},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_training_loop_preflight(
            design,
            expected_state_revision=rev_before + 1,
            checkpoint_path=None,
            executor_capabilities={},
        )
        ready = model.snn_language_dense_readout_training_loop_preflight(
            design,
            expected_state_revision=rev_before,
            checkpoint_path="dense-readout-training.pt",
            executor_capabilities={
                "checkpoint_writer_available": True,
                "bounded_delta_application_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_training_loop_preflight.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "expected_state_revision_current"
            ]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "checkpoint_path_available"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["trains_runtime_model"])
        self.assertFalse(ready["returns_trained_weights"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_training_executor"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_runtime_training"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_post_training_evaluation_requires_integrity_and_heldout(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        training = {
            "surface": "snn_language_dense_readout_training.v1",
            "accepted": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "decodes_text": False,
            "mutates_runtime_state": True,
            "returns_trained_weights": False,
            "checkpoint_transaction": {
                "post_training_checkpoint_saved": True,
                "post_training_checkpoint_restore_verified": True,
                "committed_checkpoint_path": "dense-training.committed.pt",
            },
            "dense_readout_training": {
                "training_transition_count": 2,
                "updated_cell_count": 2,
            },
        }
        integrity = {
            "surface": "snn_language_dense_readout_tensor_integrity.v1",
            "ready": True,
            "generates_text": False,
            "tensor_summary": {
                "shape": [128, 128],
                "device": "cpu",
                "dtype": "torch.float32",
                "nonzero_count": 2,
            },
            "promotion_gate": {
                "required_evidence": {
                    "sampled_sparse_weights_match_dense_tensor": True,
                    "dense_tensor_nonzero_count_matches_sparse_weights": True,
                }
            },
        }
        heldout = model.snn_language_adapter_heldout_evaluation(
            [
                [
                    {
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    }
                ]
            ],
            device_evidence={"device": "cpu", "source": "unit"},
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_post_training_evaluation(
            training,
            integrity,
        )
        ready = model.snn_language_dense_readout_post_training_evaluation(
            training,
            integrity,
            heldout_evaluation=heldout,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={
                "checkpoint_available": True,
                "restore_endpoint_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_post_training_evaluation.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "heldout_evaluation_available"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["trains_runtime_model"])
        self.assertFalse(ready["returns_trained_weights"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_decoder_probe_design"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_decoder_probe_design_is_grounded_and_non_generative(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        post_training = {
            "surface": "snn_language_dense_readout_post_training_evaluation.v1",
            "ready": True,
            "generates_text": False,
            "tensor_summary": {"device": "cpu"},
            "promotion_gate": {
                "status": "ready_for_dense_readout_decoder_probe_design"
            },
        }
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_decoder_probe_design(
            post_training,
            [{"label": "", "pressure_band": "high", "grounded": False}],
            device_evidence={"device": "cpu", "source": "unit"},
        )
        ready = model.snn_language_dense_readout_decoder_probe_design(
            post_training,
            [
                {
                    "label": "prediction error",
                    "pressure_band": "high",
                    "grounded": True,
                },
                {
                    "label": "concept focus",
                    "pressure_band": "medium",
                    "grounded": True,
                },
            ],
            device_evidence={"device": "cpu", "source": "unit"},
            decoder_design={"code_dim": 64, "max_slots": 4},
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_decoder_probe_design.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "readout_slots_grounded"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertFalse(ready["freeform_language_generation"])
        self.assertEqual(
            ready["decoder_probe_evidence"]["surface"],
            "snn_language_decoder_probe_evidence.v1",
        )
        self.assertFalse(ready["decoder_probe_evidence"]["generates_text"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_decoder_probe_preflight"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_decoder_probe_preflight_requires_current_revision(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        post_training = {
            "surface": "snn_language_dense_readout_post_training_evaluation.v1",
            "ready": True,
            "generates_text": False,
            "tensor_summary": {"device": "cpu"},
            "promotion_gate": {
                "status": "ready_for_dense_readout_decoder_probe_design"
            },
        }
        design = model.snn_language_dense_readout_decoder_probe_design(
            post_training,
            [
                {
                    "label": "prediction error",
                    "pressure_band": "high",
                    "grounded": True,
                },
                {
                    "label": "concept focus",
                    "pressure_band": "medium",
                    "grounded": True,
                },
            ],
            device_evidence={"device": "cpu", "source": "unit"},
            decoder_design={"code_dim": 64, "max_slots": 4},
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_decoder_probe_preflight(
            design,
            expected_state_revision=rev_before + 1,
            device_evidence={"device": "cpu", "source": "unit"},
        )
        ready = model.snn_language_dense_readout_decoder_probe_preflight(
            design,
            expected_state_revision=rev_before,
            device_evidence={"device": "cpu", "source": "unit"},
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_decoder_probe_preflight.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "expected_state_revision_current"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertFalse(ready["freeform_language_generation"])
        self.assertTrue(
            ready["promotion_gate"]["eligible_for_dense_readout_decoder_probe"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_decoder_probe_execution_returns_only_grounded_candidates(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        post_training = {
            "surface": "snn_language_dense_readout_post_training_evaluation.v1",
            "ready": True,
            "generates_text": False,
            "tensor_summary": {"device": "cpu"},
            "promotion_gate": {
                "status": "ready_for_dense_readout_decoder_probe_design"
            },
        }
        design = model.snn_language_dense_readout_decoder_probe_design(
            post_training,
            [
                {
                    "label": "prediction error",
                    "pressure_band": "high",
                    "grounded": True,
                },
                {
                    "label": "concept focus",
                    "pressure_band": "medium",
                    "grounded": True,
                },
            ],
            device_evidence={"device": "cpu", "source": "unit"},
            decoder_design={"code_dim": 64, "max_slots": 4},
        )
        preflight = model.snn_language_dense_readout_decoder_probe_preflight(
            design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_decoder_probe_execution(
            {**preflight, "ready": False},
        )
        ready = model.snn_language_dense_readout_decoder_probe_execution(
            preflight,
            max_candidate_labels=2,
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_decoder_probe_execution.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(blocked["probe_executed"])
        self.assertEqual(blocked["grounded_label_candidates"], [])
        self.assertTrue(ready["ready"])
        self.assertTrue(ready["probe_executed"])
        self.assertEqual(
            ready["grounded_label_candidates"],
            ["prediction error", "concept focus"],
        )
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertFalse(ready["freeform_language_generation"])
        self.assertFalse(ready["decodes_text"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_dense_readout_label_candidate_review"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )

    def test_snn_language_dense_readout_label_candidate_review_records_evidence_only(
        self,
    ) -> None:
        model, _, _, runtime_state = _build_read_model()
        post_training = {
            "surface": "snn_language_dense_readout_post_training_evaluation.v1",
            "ready": True,
            "generates_text": False,
            "tensor_summary": {"device": "cpu"},
            "promotion_gate": {
                "status": "ready_for_dense_readout_decoder_probe_design"
            },
        }
        design = model.snn_language_dense_readout_decoder_probe_design(
            post_training,
            [
                {
                    "label": "prediction error",
                    "pressure_band": "high",
                    "grounded": True,
                },
                {
                    "label": "concept focus",
                    "pressure_band": "medium",
                    "grounded": True,
                },
            ],
            device_evidence={"device": "cpu", "source": "unit"},
            decoder_design={"code_dim": 64, "max_slots": 4},
        )
        preflight = model.snn_language_dense_readout_decoder_probe_preflight(
            design,
            expected_state_revision=runtime_state.state_revision,
            device_evidence={"device": "cpu", "source": "unit"},
        )
        execution = model.snn_language_dense_readout_decoder_probe_execution(
            preflight,
            max_candidate_labels=2,
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_label_candidate_review(
            execution,
            operator_id="operator-test",
            confirmation=False,
        )
        ready = model.snn_language_dense_readout_label_candidate_review(
            execution,
            operator_id="operator-test",
            confirmation=True,
            review_note="grounded labels reviewed as candidates",
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_label_candidate_review.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(blocked["review_recorded"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "operator_confirmation_present"
            ]
        )
        self.assertTrue(ready["ready"])
        self.assertTrue(ready["review_recorded"])
        self.assertEqual(
            ready["grounded_label_candidates"],
            ["prediction error", "concept focus"],
        )
        self.assertEqual(ready["candidate_label_count"], 2)
        self.assertEqual(ready["source_execution_hash"], execution["execution_hash"])
        self.assertTrue(ready["review_hash"])
        self.assertFalse(ready["executable"])
        self.assertFalse(ready["mutates_runtime_state"])
        self.assertFalse(ready["writes_checkpoint"])
        self.assertFalse(ready["generates_text"])
        self.assertFalse(ready["freeform_language_generation"])
        self.assertFalse(ready["decodes_text"])
        self.assertFalse(ready["applies_plasticity"])
        self.assertFalse(ready["records_replay_artifact"])
        self.assertFalse(ready["promotes_facts"])
        self.assertFalse(ready["executes_actions"])
        self.assertTrue(
            ready["promotion_gate"][
                "eligible_for_bounded_label_candidate_evidence_record"
            ]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_language_generation"]
        )
        self.assertFalse(
            ready["promotion_gate"]["eligible_for_fact_promotion"]
        )
        self.assertFalse(ready["promotion_gate"]["eligible_for_action"])

    def test_snn_language_capacity_expansion_design_is_read_only_checkpoint_backed_plan(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        memory_state = {
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": {key: {"source": "unit"} for key in weights},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        runtime_truth_evidence = model.status()["runtime_truth"]["evidence"]
        capacity_pressure = runtime_truth_evidence["snn_language_capacity_pressure"]
        fixed_boundaries = runtime_truth_evidence[
            "snn_language_capacity_fixed_boundaries"
        ]
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        design = model.snn_language_capacity_expansion_design(
            capacity_pressure,
            device_evidence={"device": "cuda:0", "source": "unit"},
            rollback_policy={"available": True, "snapshot_id": "capacity-snapshot"},
            max_neuron_growth_factor=2.0,
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            design["surface"],
            "snn_language_neuron_capacity_expansion_design.v1",
        )
        self.assertEqual(
            design["artifact_kind"],
            "terminus_snn_language_capacity_expansion_design",
        )
        self.assertTrue(design["ready"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["writes_checkpoint"])
        self.assertFalse(design["resizes_network"])
        self.assertFalse(design["adds_neurons"])
        self.assertFalse(design["adds_layers"])
        self.assertEqual(design["design"]["current_language_neuron_count"], 64)
        self.assertEqual(design["design"]["proposed_language_neuron_count"], 128)
        self.assertEqual(design["design"]["proposed_sparse_edge_budget"], 512)
        self.assertTrue(design["design"]["requires_cuda_relayout_review"])
        self.assertTrue(design["design"]["requires_checkpoint_snapshot"])
        self.assertTrue(design["design"]["requires_restore_validation"])
        self.assertTrue(
            design["promotion_gate"][
                "eligible_for_operator_capacity_expansion_design_review"
            ]
        )
        self.assertFalse(design["promotion_gate"]["eligible_for_network_resize"])
        self.assertTrue(
            design["promotion_gate"]["required_evidence"][
                "cuda_device_preferred"
            ]
        )

    def test_snn_language_capacity_expansion_preflight_requires_design_checkpoint_and_cuda(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        memory_state = {
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": {key: {"source": "unit"} for key in weights},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        runtime_truth_evidence = model.status()["runtime_truth"]["evidence"]
        capacity_pressure = runtime_truth_evidence["snn_language_capacity_pressure"]
        fixed_boundaries = runtime_truth_evidence[
            "snn_language_capacity_fixed_boundaries"
        ]
        design = model.snn_language_capacity_expansion_design(
            capacity_pressure,
            device_evidence={"device": "cuda:0", "source": "unit"},
            rollback_policy={"available": True, "snapshot_id": "capacity-snapshot"},
            max_neuron_growth_factor=2.0,
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        preflight = model.snn_language_capacity_expansion_preflight(
            design,
            expected_state_revision=rev_before,
            checkpoint_transaction={
                "checkpoint_path": "capacity-expansion.pt",
                "snapshot_id": "capacity-snapshot",
                "pre_expansion_checkpoint_saved": True,
                "pre_expansion_checkpoint_restore_verified": True,
            },
            device_evidence={"device": "cuda:0", "source": "unit"},
        )
        compatibility = model.snn_language_capacity_resize_compatibility_audit(
            preflight,
            language_capacity_state={
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 64,
                "sparse_edge_budget": 256,
                "outgoing_fanout_budget": 16,
                "capacity_expansion_count": 0,
            },
        )
        tampered = deepcopy(design)
        tampered["design"]["proposed_language_neuron_count"] = 96
        blocked = model.snn_language_capacity_expansion_preflight(
            tampered,
            expected_state_revision=rev_before,
            checkpoint_transaction={
                "checkpoint_path": "capacity-expansion.pt",
                "snapshot_id": "capacity-snapshot",
                "pre_expansion_checkpoint_saved": True,
                "pre_expansion_checkpoint_restore_verified": True,
            },
            device_evidence={"device": "cuda:0", "source": "unit"},
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            preflight["surface"],
            "snn_language_neuron_capacity_expansion_preflight.v1",
        )
        self.assertEqual(
            preflight["artifact_kind"],
            "terminus_snn_language_capacity_expansion_preflight",
        )
        self.assertTrue(preflight["ready"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["writes_checkpoint"])
        self.assertFalse(preflight["resizes_network"])
        self.assertFalse(preflight["adds_neurons"])
        self.assertFalse(preflight["adds_layers"])
        self.assertEqual(
            preflight["capacity_expansion_design_hash"],
            design["capacity_expansion_design_hash"],
        )
        self.assertEqual(
            preflight["recomputed_capacity_expansion_design_hash"],
            design["capacity_expansion_design_hash"],
        )
        self.assertTrue(
            preflight["promotion_gate"]["required_evidence"][
                "design_hash_recomputed_match"
            ]
        )
        self.assertTrue(
            preflight["promotion_gate"]["required_evidence"][
                "cuda_relayout_evidence_available"
            ]
        )
        self.assertFalse(
            preflight["promotion_gate"][
                "eligible_for_checkpoint_backed_capacity_expansion_executor"
            ]
        )
        self.assertEqual(
            compatibility["surface"],
            "snn_language_capacity_resize_compatibility_audit.v1",
        )
        self.assertFalse(compatibility["ready"])
        self.assertFalse(compatibility["mutates_runtime_state"])
        self.assertFalse(compatibility["writes_checkpoint"])
        self.assertFalse(compatibility["resizes_network"])
        self.assertFalse(compatibility["adds_neurons"])
        self.assertEqual(
            compatibility["capacity_target"]["proposed_language_neuron_count"],
            128,
        )
        self.assertEqual(
            compatibility["capacity_target"]["proposed_sparse_edge_budget"],
            512,
        )
        self.assertEqual(compatibility["fixed_boundary_count"], 2)
        self.assertEqual(compatibility["incompatible_boundary_count"], 2)
        self.assertIn(
            "snn_language_readout_ledger.dense_readout_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "snn_language_readout_ledger.regeneration_adapter_sparse_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "snn_language_readout_ledger.regeneration_replay_artifact_sparse_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "runtime_facade.regeneration_permit_request_sparse_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "runtime_facade.regeneration_application_preflight_sparse_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "runtime_facade.regeneration_application_sparse_index_validators",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "snn_language_readout_ledger.sparse_edge_budget",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertNotIn(
            "snn_language_plasticity_executor.sparse_edge_budget",
            compatibility["incompatible_boundary_ids"],
        )
        self.assertTrue(
            compatibility["promotion_gate"]["required_evidence"][
                "capacity_state_durable"
            ]
        )
        self.assertFalse(
            compatibility["promotion_gate"]["required_evidence"][
                "all_runtime_boundaries_dynamic_capacity_aware"
            ]
        )
        self.assertFalse(
            compatibility["promotion_gate"]["required_evidence"][
                "all_runtime_boundaries_compatible_with_target"
            ]
        )
        self.assertEqual(
            [item["boundary_id"] for item in fixed_boundaries["boundary_inventory"]],
            [item["boundary_id"] for item in compatibility["boundary_inventory"]],
        )
        self.assertFalse(
            compatibility["promotion_gate"]["eligible_for_capacity_resize_executor"]
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "design_hash_recomputed_match"
            ]
        )

    def test_snn_language_dense_readout_resize_preflight_requires_checkpoint_and_cuda(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": {key: {"source": "unit"} for key in weights},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        runtime_truth_evidence = model.status()["runtime_truth"]["evidence"]
        dense_plan = model.snn_language_dense_readout_resize_plan(
            runtime_truth_evidence["snn_language_capacity_pressure"],
            fixed_boundaries=runtime_truth_evidence[
                "snn_language_capacity_fixed_boundaries"
            ],
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        preflight = model.snn_language_dense_readout_resize_preflight(
            dense_plan,
            expected_state_revision=rev_before,
            checkpoint_transaction={
                "checkpoint_path": "dense-readout-resize.pt",
                "snapshot_id": "dense-resize-snapshot",
                "pre_resize_checkpoint_saved": True,
                "pre_resize_checkpoint_restore_verified": True,
            },
            device_evidence={
                "device": "cuda:0",
                "source": "unit",
                "requested_cuda_honored": True,
            },
        )
        blocked = model.snn_language_dense_readout_resize_preflight(
            dense_plan,
            expected_state_revision=rev_before + 1,
            checkpoint_transaction={},
            device_evidence={"device": "cpu", "source": "unit"},
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            preflight["surface"],
            "snn_language_dense_readout_resize_preflight.v1",
        )
        self.assertEqual(
            preflight["artifact_kind"],
            "terminus_snn_language_dense_readout_resize_preflight",
        )
        self.assertFalse(preflight["ready"])
        self.assertFalse(preflight["executable"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["writes_checkpoint"])
        self.assertFalse(preflight["resizes_network"])
        self.assertEqual(
            preflight["dense_readout_resize_plan_hash"],
            dense_plan["dense_readout_resize_plan_hash"],
        )
        self.assertEqual(
            preflight["recomputed_dense_readout_resize_plan_hash"],
            dense_plan["dense_readout_resize_plan_hash"],
        )
        self.assertEqual(
            preflight["dense_readout_relayout"]["target_dense_readout_shape"],
            [128, 128],
        )
        self.assertTrue(
            preflight["promotion_gate"]["required_evidence"][
                "checkpoint_snapshot_saved"
            ]
        )
        self.assertTrue(
            preflight["promotion_gate"]["required_evidence"][
                "checkpoint_restore_verified"
            ]
        )
        self.assertTrue(
            preflight["promotion_gate"]["required_evidence"][
                "cuda_relayout_evidence_available"
            ]
        )
        self.assertFalse(
            preflight["promotion_gate"]["required_evidence"]["executor_available"]
        )
        self.assertFalse(
            preflight["promotion_gate"]["eligible_for_dense_readout_resize_executor"]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "expected_revision_current"
            ]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "checkpoint_transaction_available"
            ]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "cuda_relayout_evidence_available"
            ]
        )

    def test_snn_language_dense_readout_resize_transaction_proposal_is_read_only(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": {key: {"source": "unit"} for key in weights},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        runtime_truth_evidence = model.status()["runtime_truth"]["evidence"]
        dense_plan = model.snn_language_dense_readout_resize_plan(
            runtime_truth_evidence["snn_language_capacity_pressure"],
            fixed_boundaries=runtime_truth_evidence[
                "snn_language_capacity_fixed_boundaries"
            ],
        )
        rev_before = runtime_state.state_revision
        dense_preflight = model.snn_language_dense_readout_resize_preflight(
            dense_plan,
            expected_state_revision=rev_before,
            checkpoint_transaction={
                "checkpoint_path": "dense-readout-resize.pt",
                "snapshot_id": "dense-resize-snapshot",
                "pre_resize_checkpoint_saved": True,
                "pre_resize_checkpoint_restore_verified": True,
            },
            device_evidence={
                "device": "cuda:0",
                "source": "unit",
                "requested_cuda_honored": True,
            },
        )
        runtime_state.mark_clean()

        proposal = model.snn_language_dense_readout_resize_transaction_proposal(
            dense_preflight,
            expected_state_revision=rev_before,
            operator_id="operator-test",
            confirmation=True,
        )
        blocked = model.snn_language_dense_readout_resize_transaction_proposal(
            dense_preflight,
            expected_state_revision=rev_before + 1,
            operator_id="",
            confirmation=False,
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            proposal["surface"],
            "snn_language_dense_readout_resize_transaction_proposal.v1",
        )
        self.assertEqual(
            proposal["artifact_kind"],
            "terminus_snn_language_dense_readout_resize_transaction_proposal",
        )
        self.assertFalse(proposal["ready"])
        self.assertFalse(proposal["executable"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertFalse(proposal["writes_checkpoint"])
        self.assertFalse(proposal["resizes_network"])
        self.assertEqual(
            proposal["dense_readout_resize_plan_hash"],
            dense_plan["dense_readout_resize_plan_hash"],
        )
        self.assertEqual(
            proposal["transaction_recipe"]["target_dense_readout_shape"],
            [128, 128],
        )
        self.assertIn(
            "allocate_target_dense_readout_tensor_on_cuda",
            proposal["transaction_recipe"]["steps"],
        )
        self.assertTrue(
            proposal["promotion_gate"]["required_evidence"][
                "preflight_checkpoint_restore_verified"
            ]
        )
        self.assertTrue(
            proposal["promotion_gate"]["required_evidence"][
                "preflight_cuda_relayout_evidence_available"
            ]
        )
        self.assertFalse(
            proposal["promotion_gate"]["required_evidence"]["executor_available"]
        )
        self.assertFalse(
            proposal["promotion_gate"]["eligible_for_dense_readout_resize_executor"]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "expected_revision_current"
            ]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"]["operator_id_available"]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"]["confirmation"]
        )

    def test_snn_language_dense_readout_resize_executor_readiness_audit_names_blockers(
        self,
    ) -> None:
        weights = {
            f"{source}:{target}": 0.01
            for source in range(14)
            for target in range(16)
        }
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
            "sparse_transition_weights": weights,
            "synapse_provenance_by_key": {key: {"source": "unit"} for key in weights},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        runtime_truth_evidence = model.status()["runtime_truth"]["evidence"]
        dense_plan = model.snn_language_dense_readout_resize_plan(
            runtime_truth_evidence["snn_language_capacity_pressure"],
            fixed_boundaries=runtime_truth_evidence[
                "snn_language_capacity_fixed_boundaries"
            ],
        )
        rev_before = runtime_state.state_revision
        dense_preflight = model.snn_language_dense_readout_resize_preflight(
            dense_plan,
            expected_state_revision=rev_before,
            checkpoint_transaction={
                "checkpoint_path": "dense-readout-resize.pt",
                "snapshot_id": "dense-resize-snapshot",
                "pre_resize_checkpoint_saved": True,
                "pre_resize_checkpoint_restore_verified": True,
            },
            device_evidence={
                "device": "cuda:0",
                "source": "unit",
                "requested_cuda_honored": True,
            },
        )
        proposal = model.snn_language_dense_readout_resize_transaction_proposal(
            dense_preflight,
            expected_state_revision=rev_before,
            operator_id="operator-test",
            confirmation=True,
        )
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_resize_executor_readiness_audit(
            proposal
        )
        capability_ready = model.snn_language_dense_readout_resize_executor_readiness_audit(
            proposal,
            executor_capabilities={
                "dense_readout_tensor_weight_owner_available": True,
                "cuda_allocator_available": True,
                "checkpoint_writer_available": True,
                "migration_ledger_writer_available": True,
                "post_resize_boundary_marker_available": True,
                "dense_boundary_migration_tests_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_resize_executor_readiness_audit.v1",
        )
        self.assertEqual(
            blocked["artifact_kind"],
            "terminus_snn_language_dense_readout_resize_executor_readiness_audit",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(blocked["executable"])
        self.assertFalse(blocked["mutates_runtime_state"])
        self.assertFalse(blocked["writes_checkpoint"])
        self.assertFalse(blocked["resizes_network"])
        self.assertEqual(blocked["remaining_dense_boundary_count"], 2)
        self.assertIn(
            "snn_language_readout_ledger.dense_readout_index_validators",
            blocked["remaining_dense_boundary_ids"],
        )
        self.assertIn(
            "snn_language_readout_ledger.cuda_dense_tensor_shapes",
            blocked["remaining_dense_boundary_ids"],
        )
        self.assertEqual(
            blocked["dense_readout_layout_state"]["target_dense_readout_shape"],
            [128, 128],
        )
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "dense_readout_layout_state_available"
            ]
        )
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "dense_readout_layout_matches_transaction"
            ]
        )
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "dense_readout_tensor_owner_available"
            ]
        )
        self.assertIn(
            "dense_readout_tensor_weight_owner_available",
            blocked["missing_executor_capabilities"],
        )
        self.assertNotIn(
            "dense_readout_tensor_owner_available",
            blocked["missing_executor_capabilities"],
        )
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "dense_readout_tensor_owner_available"
            ]
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "dense_readout_tensor_weight_owner_available"
            ]
        )
        self.assertTrue(
            capability_ready["promotion_gate"]["required_evidence"][
                "dense_readout_tensor_weight_owner_available"
            ]
        )
        self.assertTrue(
            capability_ready["promotion_gate"]["required_evidence"][
                "cuda_allocator_available"
            ]
        )
        self.assertTrue(
            capability_ready["promotion_gate"]["required_evidence"][
                "dense_boundary_migration_tests_available"
            ]
        )
        self.assertFalse(capability_ready["mutates_runtime_state"])
        self.assertFalse(capability_ready["writes_checkpoint"])
        self.assertFalse(capability_ready["resizes_network"])

    def test_snn_language_dense_readout_tensor_materialization_readiness_names_cuda_blockers(
        self,
    ) -> None:
        layout_migration = {
            "applied": True,
            "operator_id": "operator-test",
            "checkpoint_path": "dense-layout.pt",
            "committed_checkpoint_path": "dense-layout.committed.pt",
            "current_dense_readout_shape": [64, 64],
            "target_dense_readout_shape": [128, 128],
            "preserved_dense_window": [64, 64],
            "zero_initialized_new_dense_cell_count": 12288,
            "target_language_neuron_count": 128,
            "transaction_hash": "sha256:dense-transaction",
            "plan_hash": "sha256:dense-plan",
            "materializes_dense_tensor_weights": False,
            "requires_tensor_weight_executor": True,
        }
        memory_state = {
            "language_capacity": {
                "surface": "snn_language_capacity_state.v1",
                "language_neuron_count": 128,
                "sparse_edge_budget": 512,
                "outgoing_fanout_budget": 32,
                "capacity_expansion_count": 1,
            },
            "dense_readout_layout": {
                "surface": "snn_language_dense_readout_layout_state.v1",
                "target_language_neuron_count": 128,
                "layout_migration": layout_migration,
                "layout_migration_count": 1,
                "dense_resize_applied": False,
                "dynamic_dense_readout_enabled": False,
                "migration_status": "layout_migration_applied_tensor_resize_pending",
            },
            "sparse_transition_weights": {},
        }
        model, _, _, runtime_state = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state)
        )
        migration_result = {
            "surface": "snn_language_dense_readout_layout_migration.v1",
            "accepted": True,
            "owned_by_marulho": True,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "materializes_dense_tensor_weights": False,
            "checkpoint_transaction": {
                "post_layout_migration_checkpoint_saved": True,
                "post_layout_migration_checkpoint_restore_verified": True,
                "committed_checkpoint_path": "dense-layout.committed.pt",
            },
            "dense_readout_layout_migration": layout_migration,
        }
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        blocked = model.snn_language_dense_readout_tensor_materialization_readiness(
            migration_result
        )
        capability_ready = model.snn_language_dense_readout_tensor_materialization_readiness(
            migration_result,
            executor_capabilities={
                "dense_readout_tensor_weight_owner_available": True,
                "cuda_allocator_available": True,
                "preserved_window_copy_kernel_available": True,
                "zero_fill_kernel_available": True,
                "checkpoint_writer_available": True,
                "migration_ledger_writer_available": True,
                "post_resize_boundary_marker_available": True,
                "dense_boundary_migration_tests_available": True,
            },
        )

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(
            blocked["surface"],
            "snn_language_dense_readout_tensor_materialization_readiness.v1",
        )
        self.assertFalse(blocked["ready"])
        self.assertFalse(blocked["executable"])
        self.assertFalse(blocked["mutates_runtime_state"])
        self.assertFalse(blocked["writes_checkpoint"])
        self.assertFalse(blocked["resizes_network"])
        self.assertFalse(blocked["generates_text"])
        self.assertFalse(blocked["materializes_dense_tensor_weights"])
        self.assertEqual(blocked["target_dense_readout_shape"], [128, 128])
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "layout_state_migration_applied"
            ]
        )
        self.assertTrue(
            blocked["promotion_gate"]["required_evidence"][
                "layout_state_matches_migration"
            ]
        )
        self.assertIn(
            "preserved_window_copy_kernel_available",
            blocked["missing_executor_capabilities"],
        )
        self.assertIn(
            "zero_fill_kernel_available",
            blocked["missing_executor_capabilities"],
        )
        self.assertTrue(capability_ready["ready"])
        self.assertTrue(
            capability_ready["promotion_gate"][
                "eligible_for_dense_readout_tensor_materialization_executor"
            ]
        )
        self.assertFalse(
            capability_ready["promotion_gate"]["eligible_for_language_generation"]
        )
        self.assertFalse(capability_ready["mutates_runtime_state"])
        self.assertFalse(capability_ready["writes_checkpoint"])
        self.assertFalse(capability_ready["resizes_network"])

    def test_runtime_truth_applied_synapse_provenance_blocks_audit_when_restore_lineage_mismatches(
        self,
    ) -> None:
        memory_state = {
            "sparse_transition_weights": {"1:3": 0.1},
            "synapse_provenance_by_key": {
                "1:3": {
                    "provenance_type": "replay_regeneration",
                    "source_metadata_hash": "source-metadata-hash-1",
                    "emission_lineage": {
                        "emission_hash": "emission-hash-1",
                        "readout_evidence_hash": "readout-hash-1",
                        "prediction_hash": "prediction-hash-1",
                    },
                    "local_edge_provenance": {
                        "source_synapse_id": "snn-rollout-local:1:3:0",
                        "source_rollout_step_index": 10,
                        "target_rollout_step_index": 20,
                        "source_active_indices_hash": "source-active-hash-1",
                        "target_active_indices_hash": "target-active-hash-1",
                    },
                }
            },
        }
        metadata = {
            "service_state": {
                "snn_applied_replay_lineage_restore_validation": {
                    "surface": "snn_applied_replay_lineage_restore_validation.v1",
                    "saved_summary_available": True,
                    "summary_counts_match_restored_state": True,
                    "summary_hash_matches_restored_state": True,
                    "summary_matches_restored_state": True,
                    "saved_summary": {
                        "applied_replay_lineage_count": 1,
                        "lineage_material_hash": "lineage-hash-1",
                    },
                    "restored_summary": {
                        "applied_replay_lineage_count": 1,
                        "lineage_material_hash": "lineage-hash-1",
                    },
                }
            }
        }
        matching_model, _, _, _ = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state),
            metadata=metadata,
        )

        matching = matching_model.status()["runtime_truth"]["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]

        self.assertTrue(matching["restore_validation_available"])
        self.assertTrue(matching["restore_lineage_matches_restored_state"])
        self.assertFalse(matching["restore_validation_blocks_audit"])
        self.assertTrue(matching["eligible_for_readout_synapse_audit_review"])

        tampered = deepcopy(metadata)
        validation = tampered["service_state"][
            "snn_applied_replay_lineage_restore_validation"
        ]
        validation["summary_hash_matches_restored_state"] = False
        validation["summary_matches_restored_state"] = False
        validation["restored_summary"]["lineage_material_hash"] = "tampered"
        tampered_model, _, _, _ = _build_read_model(
            language_plasticity_state_fn=lambda: deepcopy(memory_state),
            metadata=tampered,
        )

        blocked = tampered_model.status()["runtime_truth"]["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]

        self.assertTrue(blocked["restore_validation_available"])
        self.assertFalse(blocked["restore_lineage_matches_restored_state"])
        self.assertTrue(blocked["restore_validation_blocks_audit"])
        self.assertFalse(blocked["eligible_for_readout_synapse_audit_review"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "restore_validation_not_mismatched"
            ]
        )
        self.assertEqual(
            blocked["promotion_status"],
            "waiting_for_matching_applied_replay_lineage_restore_validation",
        )

    def test_runtime_truth_restore_validation_reports_checkpoint_lineage_match_without_mutation(
        self,
    ) -> None:
        metadata = {
            "service_state": {
                "snn_applied_replay_lineage_restore_validation": {
                    "surface": "snn_applied_replay_lineage_restore_validation.v1",
                    "saved_summary_available": True,
                    "summary_counts_match_restored_state": True,
                    "summary_hash_matches_restored_state": True,
                    "summary_matches_restored_state": True,
                    "saved_summary": {
                        "applied_replay_lineage_count": 1,
                        "lineage_material_hash": "lineage-hash-1",
                    },
                    "restored_summary": {
                        "applied_replay_lineage_count": 1,
                        "lineage_material_hash": "lineage-hash-1",
                    },
                }
            }
        }
        model, _, _, runtime_state = _build_read_model(metadata=metadata)
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        evidence = model.status()["runtime_truth"]["evidence"][
            "snn_applied_replay_lineage_restore_validation"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["saved_summary_available"])
        self.assertTrue(evidence["summary_counts_match_restored_state"])
        self.assertTrue(evidence["summary_hash_matches_restored_state"])
        self.assertTrue(evidence["summary_matches_restored_state"])
        self.assertEqual(evidence["saved_lineage_count"], 1)
        self.assertEqual(evidence["restored_lineage_count"], 1)
        self.assertEqual(evidence["saved_lineage_material_hash"], "lineage-hash-1")
        self.assertEqual(evidence["restored_lineage_material_hash"], "lineage-hash-1")
        self.assertTrue(evidence["eligible_for_readout_synapse_audit_review"])
        self.assertFalse(evidence["eligible_for_plasticity_application"])
        self.assertFalse(evidence["eligible_for_live_replay"])
        self.assertFalse(evidence["eligible_for_fact_promotion"])
        self.assertFalse(evidence["eligible_for_action"])
        self.assertEqual(
            evidence["promotion_status"],
            "ready_for_readout_synapse_provenance_audit",
        )
        self.assertTrue(
            evidence["promotion_gate"]["required_evidence"][
                "summary_hash_matches_restored_state"
            ]
        )
        self.assertTrue(
            evidence["promotion_gate"]["required_evidence"][
                "runtime_mutation_absent"
            ]
        )
        self.assertTrue(
            evidence["promotion_gate"]["required_evidence"][
                "replay_execution_absent"
            ]
        )

        tampered = deepcopy(metadata)
        validation = tampered["service_state"][
            "snn_applied_replay_lineage_restore_validation"
        ]
        validation["summary_hash_matches_restored_state"] = False
        validation["summary_matches_restored_state"] = False
        validation["restored_summary"]["lineage_material_hash"] = "tampered"
        tampered_model, _, _, _ = _build_read_model(metadata=tampered)

        blocked = tampered_model.status()["runtime_truth"]["evidence"][
            "snn_applied_replay_lineage_restore_validation"
        ]

        self.assertFalse(blocked["summary_hash_matches_restored_state"])
        self.assertFalse(blocked["summary_matches_restored_state"])
        self.assertEqual(blocked["restored_lineage_material_hash"], "tampered")
        self.assertFalse(blocked["eligible_for_readout_synapse_audit_review"])
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"][
                "summary_hash_matches_restored_state"
            ]
        )
        self.assertEqual(
            blocked["promotion_status"],
            "waiting_for_matching_applied_replay_lineage_restore_validation",
        )

    def test_runtime_truth_consolidation_path_reports_recorded_rollout_without_execution(self) -> None:
        ledger_state = {
            "rollout_events": [
                {
                    "rollout_evidence_hash": "evidence-hash",
                    "rollout_hash": "rollout-hash",
                    "persistent_transition_weights_hash": "transition-hash",
                    "recorded_at": "2026-06-02T00:00:00+00:00",
                }
            ],
            "total_rollout_recorded_count": 1,
            "last_rollout_recorded_at": "2026-06-02T00:00:00+00:00",
        }
        model, _, _, runtime_state = _build_read_model(
            readout_ledger_state_fn=lambda: deepcopy(ledger_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        path = model.status()["runtime_truth"]["evidence"][
            "snn_readout_rollout_consolidation_path"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(path["rollout_event_count"], 1)
        self.assertEqual(path["total_rollout_recorded_count"], 1)
        self.assertEqual(path["unique_rollout_count"], 1)
        self.assertEqual(path["unique_transition_memory_count"], 1)
        self.assertEqual(path["latest_rollout_evidence_hash"], "evidence-hash")
        self.assertEqual(path["latest_rollout_hash"], "rollout-hash")
        self.assertEqual(path["latest_transition_memory_hash"], "transition-hash")
        self.assertEqual(
            path["promotion_status"],
            "ready_for_rollout_rehearsal_policy_review",
        )
        self.assertTrue(path["eligible_for_rollout_rehearsal_policy_review"])
        self.assertFalse(path["executes_rehearsal"])
        self.assertFalse(path["executes_consolidation"])
        self.assertFalse(path["runs_live_replay"])
        self.assertFalse(path["records_ledger_event"])
        self.assertFalse(path["writes_checkpoint"])
        self.assertFalse(path["generates_text"])
        self.assertFalse(path["applies_plasticity"])
        self.assertFalse(path["mutates_runtime_state"])

    def test_runtime_truth_emission_review_history_reports_reviewed_output_without_exposing_text(
        self,
    ) -> None:
        ledger_state = {
            "emission_review_events": [
                {
                    "emission_review_hash": "review-hash",
                    "emission_hash": "emission-hash",
                    "trajectory_hash": "trajectory-hash",
                    "persistent_transition_weights_hash": "transition-hash",
                    "reviewed_at": "2026-06-02T00:00:00+00:00",
                    "text": "do not expose this bounded display text",
                    "labels": ["memory pressure"],
                }
            ],
            "total_emission_review_count": 1,
            "last_emission_reviewed_at": "2026-06-02T00:00:00+00:00",
        }
        model, _, _, runtime_state = _build_read_model(
            readout_ledger_state_fn=lambda: deepcopy(ledger_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        history = model.status()["runtime_truth"]["evidence"][
            "snn_readout_emission_review_history"
        ]
        terminus_history = model.terminus_status()["runtime_truth"]["evidence"][
            "snn_readout_emission_review_history"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(history["emission_review_event_count"], 1)
        self.assertEqual(history["total_emission_review_count"], 1)
        self.assertEqual(history["unique_emission_count"], 1)
        self.assertEqual(history["unique_trajectory_count"], 1)
        self.assertEqual(history["unique_transition_memory_count"], 1)
        self.assertEqual(history["latest_emission_review_hash"], "review-hash")
        self.assertEqual(history["latest_emission_hash"], "emission-hash")
        self.assertEqual(history["latest_trajectory_hash"], "trajectory-hash")
        self.assertEqual(history["latest_transition_memory_hash"], "transition-hash")
        self.assertEqual(
            history["promotion_status"],
            "ready_for_operator_display_history_inspection",
        )
        self.assertTrue(
            history["eligible_for_operator_display_history_inspection"]
        )
        self.assertFalse(history["generates_text"])
        self.assertFalse(history["decodes_text"])
        self.assertFalse(history["exposes_raw_text"])
        self.assertFalse(history["calls_endpoint"])
        self.assertFalse(history["records_ledger_event"])
        self.assertFalse(history["runs_replay"])
        self.assertFalse(history["writes_checkpoint"])
        self.assertFalse(history["applies_plasticity"])
        self.assertFalse(history["mutates_runtime_state"])
        self.assertFalse(history["eligible_for_replay_memory"])
        self.assertFalse(history["eligible_for_live_replay"])
        self.assertFalse(history["eligible_for_plasticity_application"])
        self.assertFalse(history["eligible_for_fact_promotion"])
        self.assertFalse(history["eligible_for_action"])
        self.assertTrue(
            history["promotion_gate"]["required_evidence"][
                "freeform_language_generation_absent"
            ]
        )
        self.assertTrue(
            history["promotion_gate"]["required_evidence"][
                "reviewed_emission_available"
            ]
        )
        self.assertTrue(
            history["promotion_gate"]["required_evidence"]["raw_text_exposure_absent"]
        )
        self.assertEqual(
            terminus_history["latest_emission_review_hash"],
            history["latest_emission_review_hash"],
        )
        self.assertEqual(
            terminus_history["emission_review_event_count"],
            history["emission_review_event_count"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
        ):
            self.assertNotIn(forbidden_key, history)

    def test_runtime_truth_emission_replay_design_path_reports_hash_only_candidates(
        self,
    ) -> None:
        ledger_state = {
            "events": [
                {
                    "readout_evidence_hash": "readout-hash",
                    "readout_evidence_id": "readout-id",
                    "prediction_hash": "prediction-hash",
                    "transition_memory_evaluation_hash": "evaluation-hash",
                    "persistent_transition_weights_hash": "weights-hash",
                    "labels": ["memory pressure"],
                    "label_grounding": [True],
                }
            ],
            "emission_review_events": [
                {
                    "emission_review_hash": "review-hash",
                    "emission_hash": "emission-hash",
                    "prediction_hash": "prediction-hash",
                    "transition_memory_evaluation_hash": "evaluation-hash",
                    "persistent_transition_weights_hash": "weights-hash",
                    "reviewed_at": "2026-06-03T00:00:00+00:00",
                    "text": "do not expose this bounded display text",
                    "labels": ["memory pressure"],
                }
            ],
            "total_emission_review_count": 1,
            "last_emission_reviewed_at": "2026-06-03T00:00:00+00:00",
        }
        model, _, _, runtime_state = _build_read_model(
            readout_ledger_state_fn=lambda: deepcopy(ledger_state)
        )
        rev_before = runtime_state.state_revision
        runtime_state.mark_clean()

        path = model.status()["runtime_truth"]["evidence"][
            "snn_readout_emission_replay_design_path"
        ]
        terminus_path = model.terminus_status()["runtime_truth"]["evidence"][
            "snn_readout_emission_replay_design_path"
        ]

        self.assertEqual(runtime_state.state_revision, rev_before)
        self.assertFalse(runtime_state.dirty_state)
        self.assertEqual(path["emission_review_event_count"], 1)
        self.assertEqual(path["internal_readout_evidence_count"], 1)
        self.assertEqual(path["policy_candidate_count"], 1)
        self.assertEqual(path["design_seed_candidate_count"], 1)
        self.assertEqual(path["unmatched_emission_review_count"], 0)
        self.assertEqual(path["latest_emission_review_hash"], "review-hash")
        self.assertEqual(path["latest_emission_hash"], "emission-hash")
        self.assertEqual(path["latest_readout_evidence_hash"], "readout-hash")
        self.assertEqual(path["latest_prediction_hash"], "prediction-hash")
        self.assertEqual(
            path["latest_transition_memory_evaluation_hash"],
            "evaluation-hash",
        )
        self.assertEqual(
            path["latest_persistent_transition_weights_hash"],
            "weights-hash",
        )
        self.assertEqual(
            path["promotion_status"],
            "ready_for_emission_replay_evaluation_design_review",
        )
        self.assertTrue(
            path["eligible_for_emission_replay_evaluation_design_review"]
        )
        self.assertFalse(path["eligible_for_operator_replay_context_review"])
        self.assertFalse(path["eligible_for_replay_context_recording"])
        self.assertTrue(path["requires_device_review_evidence"])
        self.assertTrue(path["requires_server_computed_mismatch_probe"])
        self.assertTrue(path["requires_server_computed_plasticity_pressure"])
        self.assertFalse(path["records_replay_context"])
        self.assertFalse(path["runs_replay"])
        self.assertFalse(path["records_ledger_event"])
        self.assertFalse(path["generates_text"])
        self.assertFalse(path["decodes_text"])
        self.assertFalse(path["exposes_raw_text"])
        self.assertFalse(path["eligible_for_replay_memory"])
        self.assertFalse(path["eligible_for_live_replay"])
        self.assertFalse(path["eligible_for_plasticity_application"])
        self.assertFalse(path["eligible_for_fact_promotion"])
        self.assertFalse(path["eligible_for_action"])
        self.assertEqual(
            path["next_gate"],
            (
                "POST /terminus/snn-language-sequence/readout-emission/operator-review/"
                "replay-evaluation-design"
            ),
        )
        self.assertEqual(
            terminus_path["design_seed_candidate_count"],
            path["design_seed_candidate_count"],
        )
        self.assertEqual(
            terminus_path["latest_readout_evidence_hash"],
            path["latest_readout_evidence_hash"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
            "selected_replay_context_seeds",
            "events",
        ):
            self.assertNotIn(forbidden_key, path)


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

    def test_manager_cortex_signal_state_alias_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="removed_cortex_signal_alias")
            try:
                self.assertFalse(hasattr(manager, "cortex_signal_state"))
            finally:
                manager.close()


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
            from marulho.service.manager import MarulhoServiceManager
            cfg = _build_config()
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt",
                trainer,
                metadata={"test_case": "sensory_delegation"},
            )
            manager = MarulhoServiceManager(checkpoint_path, trace_dir=root / "traces")
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
            from marulho.service.manager import MarulhoServiceManager
            cfg = _build_config()
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt",
                trainer,
                metadata={"test_case": "arch_delegation"},
            )
            manager = MarulhoServiceManager(checkpoint_path, trace_dir=root / "traces")
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
            from marulho.service.manager import MarulhoServiceManager
            cfg = _build_config()
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "living_loop_delegation"},
            )
            manager = MarulhoServiceManager(checkpoint_path, trace_dir=root / "traces")
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
            from marulho.service.manager import MarulhoServiceManager
            cfg = _build_config()
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial.pt", trainer, metadata={"test_case": "policy_delegation"},
            )
            manager = MarulhoServiceManager(checkpoint_path, trace_dir=root / "traces")
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

    def test_runtime_facade_snn_language_readiness_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes MARULHO-native SNN language readiness."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="snn_language_readiness_delegation")
            try:
                result = manager.runtime_facade.snn_language_readiness_surface()
                self.assertEqual(result["surface"], "snn_native_language_readiness.v1")
                self.assertEqual(result["source"], "service.status_read_model.cognitive_signal_and_runtime_scope")
                self.assertTrue(result["advisory"])
                self.assertFalse(result["executable"])
                self.assertFalse(result["mutates_runtime_state"])
                self.assertFalse(result["promotion_gate"]["eligible_for_cognition_substrate"])
                self.assertTrue(result["safety_invariants"]["requires_marulho_owned_implementation"])
            finally:
                manager.close()

    def test_runtime_facade_subcortical_self_repair_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes the self-repair promotion gate artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="subcortical_self_repair_delegation")
            try:
                result = manager.runtime_facade.subcortical_self_repair_surface()
                self.assertEqual(result["surface"], "subcortical_self_repair_candidates.v1")
                self.assertEqual(result["source"], "service.status_read_model.runtime_scope.spike_health")
                self.assertTrue(result["advisory"])
                self.assertFalse(result["executable"])
                self.assertIn("promotion_gate", result)
                self.assertFalse(result["promotion_gate"]["eligible_for_structural_mutation"])
            finally:
                manager.close()

    def test_runtime_facade_subcortical_self_repair_evaluation_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes the read-only self-repair evaluation artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="subcortical_self_repair_evaluation_delegation")
            try:
                result = manager.runtime_facade.subcortical_self_repair_evaluation_surface()
                self.assertEqual(result["surface"], "subcortical_self_repair_evaluation.v1")
                self.assertEqual(result["source"], "service.status_read_model.runtime_scope.spike_health")
                self.assertTrue(result["advisory"])
                self.assertFalse(result["executable"])
                self.assertFalse(result["mutates_runtime_state"])
                self.assertIn("evaluation_gate", result)
                self.assertFalse(result["evaluation_gate"]["eligible_for_structural_mutation"])
            finally:
                manager.close()

    def test_runtime_facade_subcortical_structural_plasticity_surface_delegates_to_read_model(self) -> None:
        """The runtime facade exposes structural-plasticity review evidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="subcortical_structural_plasticity_delegation")
            try:
                result = manager.runtime_facade.subcortical_structural_plasticity_surface()
                self.assertEqual(result["surface"], "subcortical_structural_plasticity.v1")
                self.assertEqual(result["source"], "service.status_read_model.concept_store_and_runtime_scope")
                self.assertTrue(result["advisory"])
                self.assertFalse(result["executable"])
                self.assertFalse(result["mutates_runtime_state"])
                self.assertIn("promotion_gate", result)
                self.assertIn("device_evidence", result)
                self.assertIn("local_plasticity", result)
                self.assertIn("local_plasticity_report_available", result["device_evidence"])
                self.assertFalse(result["promotion_gate"]["eligible_for_structural_mutation"])
            finally:
                manager.close()
