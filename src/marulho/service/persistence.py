from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from marulho.config.runtime_env import load_runtime_env
from marulho.reporting.io import write_json_file
from marulho.semantics import ConceptStore, GeometricCuriosityController
from marulho.service.applied_replay_lineage import (
    applied_replay_lineage_checkpoint_summary,
)
from marulho.service.snn_language_readout_ledger import (
    normalize_snn_language_readout_ledger_state,
)
from marulho.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint

DEFAULT_DELAYED_CONSEQUENCE_RECORDS = 24
DEFAULT_CHECKPOINT_LOCK_TIMEOUT_SECONDS = 0.25
CURRENT_CHECKPOINT_MANIFEST = "marulho_current_checkpoint.json"
_SNN_LANGUAGE_CAPACITY_SURFACE = "snn_language_capacity_state.v1"
_SNN_LANGUAGE_DENSE_READOUT_LAYOUT_SURFACE = "snn_language_dense_readout_layout_state.v1"
_SNN_LANGUAGE_NEURON_COUNT = 64
_SNN_LANGUAGE_SPARSE_EDGE_BUDGET = 256
_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET = 16


_FORWARDED_STATE_NAMES = frozenset({
    "_action_root",
    "_brain_config",
    "_brain_runtime",
    "_brain_last_acquisition_summary",
    "_brain_last_acquisition_token_count",
    "_brain_last_error",
    "_brain_source_utility",
    "_checkpoint_dir",
    "_checkpoint_path",
    "_concept_store",
    "_delayed_consequence_compacted_total",
    "_delayed_consequence_cooled_total",
    "_delayed_consequence_records",
    "_delayed_consequence_remerged_total",
    "_delayed_consequence_retired_total",
    "_delayed_consequence_split_total",
    "_delayed_consequence",
    "_encoder",
    "_env_root",
    "_geometric_curiosity",
    "_interaction_pipeline",
    "_metadata",
    "_runtime_config",
    "_runtime_env",
    "_runtime_state",
    "_snn_language_plasticity_state",
    "_snn_language_readout_ledger_state",
    "_trace_dir",
    "_trainer",
})


@dataclass(frozen=True)
class RuntimePersistenceDependencies:
    action_executor: Callable[[], Any]
    get_state: Callable[[str], Any]
    replay_controller: Callable[[], Any]
    set_state: Callable[[str, Any], None]
    brain_persisted_state: Callable[[], Mapping[str, Any]]
    marulho_brain_state: Callable[[], Mapping[str, Any]]
    brain_runtime_snapshot: Callable[..., Mapping[str, Any]]
    join_brain_thread: Callable[..., Any]
    lock: Any
    normalize_background_source_utility_state: Callable[[Any], dict[str, Any]]
    normalize_delayed_consequence_record: Callable[[Any], dict[str, Any] | None]
    rebuild_brain_sources: Callable[[], None]
    refresh_root_captures: Callable[[], None]
    request_brain_stop: Callable[..., Any]


class RuntimePersistence:
    """Checkpoint, trace-history, and JSON-safe persistence helpers."""

    def __init__(
        self,
        dependencies: RuntimePersistenceDependencies,
        *,
        trace_history_limit: int = 200,
        trace_history: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        object.__setattr__(self, "_dependencies", dependencies)
        object.__setattr__(self, "_checkpoint_root", Path(self._checkpoint_dir).resolve())
        self._trace_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(trace_history_limit)))
        self._cached_checkpoint_list: list[dict[str, Any]] = []
        self._cached_trace_lists: dict[int, list[dict[str, Any]]] = {}
        self.load_persisted_traces(trace_history or [])

    def __getattr__(self, name: str) -> Any:
        if name in _FORWARDED_STATE_NAMES:
            return self._dependencies.get_state(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {
            "_dependencies",
            "_checkpoint_root",
            "_trace_history",
            "_cached_checkpoint_list",
            "_cached_trace_lists",
        }:
            object.__setattr__(self, name, value)
            return
        if name in _FORWARDED_STATE_NAMES:
            self._dependencies.set_state(name, value)
            return
        object.__setattr__(self, name, value)

    @property
    def _lock(self) -> Any:
        return self._dependencies.lock

    def _brain_persisted_state_locked(self) -> Mapping[str, Any]:
        return self._dependencies.brain_persisted_state()

    def _brain_runtime_snapshot_locked(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._dependencies.brain_runtime_snapshot(**kwargs)

    def _join_brain_thread(self, *args: Any, **kwargs: Any) -> Any:
        return self._dependencies.join_brain_thread(*args, **kwargs)

    def _normalize_background_source_utility_state(self, value: Any) -> dict[str, Any]:
        return self._dependencies.normalize_background_source_utility_state(value)

    def _normalize_delayed_consequence_record(self, value: Any) -> dict[str, Any] | None:
        return self._dependencies.normalize_delayed_consequence_record(value)

    def _rebuild_brain_sources_locked(self) -> None:
        self._dependencies.rebuild_brain_sources()

    def _request_brain_stop(self, *args: Any, **kwargs: Any) -> Any:
        return self._dependencies.request_brain_stop(*args, **kwargs)

    @property
    def trace_history(self) -> deque[dict[str, Any]]:
        return self._trace_history

    @trace_history.setter
    def trace_history(self, value: Sequence[Mapping[str, Any]]) -> None:
        self.load_persisted_traces(value)

    def checkpoint_list(self) -> list[dict[str, Any]]:
        acquired = self._lock.acquire(timeout=0.05)
        if not acquired:
            return deepcopy(self._cached_checkpoint_list)
        try:
            if not self._checkpoint_dir.exists():
                self._cached_checkpoint_list = []
                return []
            records: list[dict[str, Any]] = []
            for path in sorted(self._checkpoint_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
                stat = path.stat()
                records.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    }
                )
            self._cached_checkpoint_list = deepcopy(records)
            return records
        finally:
            self._lock.release()

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        count = max(1, int(limit))
        acquired = self._lock.acquire(timeout=0.05)
        if not acquired:
            return deepcopy(self._cached_trace_lists.get(count, []))
        try:
            traces = [deepcopy(trace) for trace in list(self._trace_history)[:count]]
            self._cached_trace_lists[count] = deepcopy(traces)
            return traces
        finally:
            self._lock.release()

    def persist_trace(self, trace: dict[str, Any]) -> Path:
        with self._lock:
            return self._persist_trace_locked(trace)

    def load_persisted_traces(self, trace_history: Sequence[Mapping[str, Any]]) -> None:
        normalized = [
            deepcopy(dict(item))
            for item in trace_history
            if isinstance(item, Mapping)
        ]
        self._trace_history.clear()
        self._trace_history.extend(normalized)
        self._cached_trace_lists.clear()

    def save_checkpoint(self, path: str | None = None, *, publish: bool = True) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=DEFAULT_CHECKPOINT_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            raise TimeoutError(
                "Runtime is busy with an active mutation; checkpoint save was not started. "
                "Retry after the current tick or stop Terminus first."
            )
        try:
            runtime_snapshot = self._brain_runtime_snapshot_locked()
            execution = runtime_snapshot.get("execution")
            execution_busy = bool(
                isinstance(execution, Mapping)
                and (
                    execution.get("tick_in_progress")
                    or int(execution.get("active_execution_requests", 0) or 0) > 0
                )
            )
            if bool(runtime_snapshot.get("running")) or execution_busy:
                raise TimeoutError(
                    "Terminus is running; checkpoint save was not started. "
                    "Stop Terminus first so the checkpoint is coherent and does not stall Runtime Truth."
                )
            target = self._resolve_save_path(path)
            metadata = deepcopy(self._metadata)
            service_state = dict(metadata.get("service_state", {}))
            service_state["concept_store"] = self._concept_store.state_dict()
            service_state["terminus_runtime"] = self._brain_persisted_state_locked()
            service_state["snn_language_plasticity"] = self._snn_language_plasticity_checkpoint_state(
                self._snn_language_plasticity_state
            )
            service_state["snn_applied_replay_lineage_checkpoint_summary"] = (
                self._applied_replay_lineage_checkpoint_summary(
                    service_state["snn_language_plasticity"]
                )
            )
            service_state["snn_language_readout_ledger"] = (
                normalize_snn_language_readout_ledger_state(
                    self._snn_language_readout_ledger_state
                )
            )
            metadata.update(
                {
                    "brain_state": dict(self._dependencies.marulho_brain_state()),
                    "saved_by": "marulho.service",
                    "state_revision": int(self._runtime_state.state_revision),
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "service_state": service_state,
                }
            )
            saved_path = save_trainer_checkpoint(target, self._trainer, metadata=metadata)
            manifest = self.publish_current_checkpoint(saved_path, operation="manual_save") if publish else None
            return {
                "path": str(saved_path),
                "current_checkpoint_manifest": manifest,
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }
        finally:
            self._lock.release()

    @classmethod
    def _snn_language_plasticity_checkpoint_state(
        cls,
        plasticity_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        state = deepcopy(dict(plasticity_state or {}))
        state["language_capacity"] = cls._snn_language_capacity_checkpoint_state(
            state
        )
        state["dense_readout_layout"] = cls._snn_language_dense_readout_layout_checkpoint_state(
            state,
            state["language_capacity"],
        )
        return state

    @classmethod
    def _snn_language_capacity_checkpoint_state(
        cls,
        plasticity_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = (
            plasticity_state.get("language_capacity")
            if isinstance(plasticity_state.get("language_capacity"), Mapping)
            else {}
        )
        return {
            "surface": _SNN_LANGUAGE_CAPACITY_SURFACE,
            "owned_by_marulho": True,
            "external_dependency": False,
            "language_neuron_count": cls._positive_capacity_int(
                raw.get("language_neuron_count"),
                default=_SNN_LANGUAGE_NEURON_COUNT,
                minimum=_SNN_LANGUAGE_NEURON_COUNT,
            ),
            "sparse_edge_budget": cls._positive_capacity_int(
                raw.get("sparse_edge_budget"),
                default=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
                minimum=_SNN_LANGUAGE_SPARSE_EDGE_BUDGET,
            ),
            "outgoing_fanout_budget": cls._positive_capacity_int(
                raw.get("outgoing_fanout_budget"),
                default=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
                minimum=_SNN_LANGUAGE_OUTGOING_FANOUT_BUDGET,
            ),
            "dynamic_capacity_enabled": bool(
                raw.get("dynamic_capacity_enabled")
            ),
            "capacity_expansion_count": cls._positive_capacity_int(
                raw.get("capacity_expansion_count"),
                default=0,
                minimum=0,
            ),
            "resizes_network": bool(raw.get("resizes_network")),
            "adds_neurons": bool(raw.get("adds_neurons")),
            "adds_layers": bool(raw.get("adds_layers")),
            "writes_checkpoint": bool(raw.get("writes_checkpoint")),
            "last_capacity_mutation": deepcopy(
                raw.get("last_capacity_mutation")
            ),
        }

    @classmethod
    def _snn_language_dense_readout_layout_checkpoint_state(
        cls,
        plasticity_state: Mapping[str, Any],
        capacity_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = (
            plasticity_state.get("dense_readout_layout")
            if isinstance(plasticity_state.get("dense_readout_layout"), Mapping)
            else {}
        )
        target_neurons = cls._positive_capacity_int(
            raw.get("target_language_neuron_count"),
            default=int(
                capacity_state.get("language_neuron_count", _SNN_LANGUAGE_NEURON_COUNT)
            ),
            minimum=_SNN_LANGUAGE_NEURON_COUNT,
        )
        layout_migration = (
            raw.get("layout_migration")
            if isinstance(raw.get("layout_migration"), Mapping)
            else {}
        )
        tensor_materialization = (
            raw.get("tensor_materialization")
            if isinstance(raw.get("tensor_materialization"), Mapping)
            else {}
        )
        current_shape = [_SNN_LANGUAGE_NEURON_COUNT, _SNN_LANGUAGE_NEURON_COUNT]
        target_shape = [target_neurons, target_neurons]
        dense_resize_applied = bool(raw.get("dense_resize_applied"))
        layout_migration_applied = bool(layout_migration.get("applied"))
        tensor_materialization_applied = bool(tensor_materialization.get("applied"))
        return {
            "surface": _SNN_LANGUAGE_DENSE_READOUT_LAYOUT_SURFACE,
            "raw_surface": str(raw.get("surface") or "") if raw else None,
            "present": bool(raw),
            "owned_by_marulho": True,
            "external_dependency": False,
            "current_dense_readout_shape": current_shape,
            "target_dense_readout_shape": target_shape,
            "preserved_dense_window": current_shape,
            "zero_initialized_new_dense_cell_count": max(
                0,
                int(target_neurons * target_neurons)
                - int(_SNN_LANGUAGE_NEURON_COUNT * _SNN_LANGUAGE_NEURON_COUNT),
            ),
            "target_language_neuron_count": target_neurons,
            "requires_cuda_relayout": target_neurons > _SNN_LANGUAGE_NEURON_COUNT
            and not tensor_materialization_applied,
            "checkpoint_required_before_resize": not dense_resize_applied,
            "layout_migration_applied": layout_migration_applied,
            "tensor_materialization_applied": tensor_materialization_applied,
            "dense_resize_applied": dense_resize_applied,
            "dynamic_dense_readout_enabled": dense_resize_applied,
            "migration_status": "layout_metadata_only_resize_pending"
            if target_neurons > _SNN_LANGUAGE_NEURON_COUNT
            and not layout_migration_applied
            else str(
                raw.get(
                    "migration_status",
                    "dense_readout_tensor_materialized"
                    if tensor_materialization_applied
                    else "layout_migration_applied_tensor_resize_pending"
                    if layout_migration_applied
                    else "fixed_dense_layout",
                )
            ),
            "layout_migration": deepcopy(dict(layout_migration)),
            "tensor_materialization": deepcopy(dict(tensor_materialization)),
            "resizes_network": False,
            "writes_checkpoint": False,
        }

    @staticmethod
    def _positive_capacity_int(
        value: Any,
        *,
        default: int,
        minimum: int,
    ) -> int:
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            normalized = int(default)
        return max(int(minimum), normalized)

    @classmethod
    def _applied_replay_lineage_checkpoint_summary(
        cls,
        plasticity_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        return applied_replay_lineage_checkpoint_summary(
            plasticity_state,
            source="runtime_persistence.save_checkpoint",
        )

    def publish_current_checkpoint(self, path: str | Path, *, operation: str) -> dict[str, Any]:
        """Atomically publish the verified checkpoint selected for crash recovery."""

        with self._lock:
            source_path = Path(path).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"Cannot publish missing checkpoint: {source_path}")
            object_dir = self._checkpoint_root / "objects"
            object_dir.mkdir(parents=True, exist_ok=True)
            object_path = object_dir / (
                f"revision-{int(self._runtime_state.state_revision)}-{str(operation)}-{uuid4().hex}.pt"
            )
            self._copy_atomic(source_path, object_path)
            descriptor = self._checkpoint_descriptor(object_path, operation=operation)
            _trainer, metadata = load_trainer_checkpoint(object_path)
            if int(metadata.get("state_revision", -1)) != int(descriptor["state_revision"]):
                raise ValueError("Published checkpoint revision does not match runtime revision.")
            manifest_path = self.current_checkpoint_manifest_path(self._checkpoint_root)
            previous = None
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(existing.get("current"), Mapping):
                    previous = dict(existing["current"])
            except Exception:
                previous = None
            payload = {
                "schema_version": 1,
                "artifact_kind": "marulho_current_checkpoint_manifest",
                "current": descriptor,
                "previous": previous,
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
            previous_checkpoint_path = self._checkpoint_path
            previous_checkpoint_dir = self._checkpoint_dir
            previous_metadata = deepcopy(self._metadata)
            self._checkpoint_path = object_path
            self._checkpoint_dir = self._checkpoint_root
            self._metadata = dict(metadata)
            try:
                self._dependencies.refresh_root_captures()
                self._write_atomic_json(manifest_path, payload)
            except Exception:
                self._checkpoint_path = previous_checkpoint_path
                self._checkpoint_dir = previous_checkpoint_dir
                self._metadata = previous_metadata
                self._dependencies.refresh_root_captures()
                raise
            self._runtime_state.mark_clean()
            return {"path": str(manifest_path), "checkpoint_path": str(object_path), **payload}

    @staticmethod
    def checkpoint_root_for_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        parent = path.parent if path.parent != Path("") else Path("checkpoints")
        if parent.name == "objects" and (parent.parent / CURRENT_CHECKPOINT_MANIFEST).is_file():
            return parent.parent
        return parent

    @classmethod
    def resolve_current_checkpoint_path(cls, fallback: str | Path) -> Path:
        fallback_path = Path(fallback)
        manifest_path = cls.current_checkpoint_manifest_path(fallback_path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return fallback_path
        if payload.get("schema_version") != 1 or payload.get("artifact_kind") != "marulho_current_checkpoint_manifest":
            return fallback_path
        root = manifest_path.parent.resolve()
        for key in ("current", "previous"):
            descriptor = payload.get(key)
            if isinstance(descriptor, Mapping):
                candidate = cls._validated_descriptor_path(root, descriptor)
                if candidate is not None:
                    return candidate
        return fallback_path

    @staticmethod
    def current_checkpoint_manifest_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        parent = path if path.is_dir() else (path.parent if path.parent != Path("") else Path("."))
        return parent / CURRENT_CHECKPOINT_MANIFEST

    def _checkpoint_descriptor(self, path: Path, *, operation: str) -> dict[str, Any]:
        return {
            "relative_path": path.resolve().relative_to(self._checkpoint_root).as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": self._sha256_file(path),
            "state_revision": int(self._runtime_state.state_revision),
            "operation": str(operation),
        }

    @classmethod
    def _validated_descriptor_path(cls, root: Path, descriptor: Mapping[str, Any]) -> Path | None:
        try:
            candidate = (root / str(descriptor.get("relative_path") or "")).resolve()
            candidate.relative_to(root)
            if not candidate.is_file():
                return None
            if int(descriptor.get("size_bytes", -1)) != int(candidate.stat().st_size):
                return None
            if str(descriptor.get("sha256") or "") != cls._sha256_file(candidate):
                return None
            _trainer, metadata = load_trainer_checkpoint(candidate)
            if int(metadata.get("state_revision", -1)) != int(descriptor.get("state_revision", -2)):
                return None
            return candidate
        except Exception:
            return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _copy_atomic(source: Path, target: Path) -> None:
        temporary_path = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with source.open("rb") as read_handle, temporary_path.open("wb") as write_handle:
                shutil.copyfileobj(read_handle, write_handle)
                write_handle.flush()
                os.fsync(write_handle.fileno())
            os.replace(temporary_path, target)
            RuntimePersistence._sync_parent_directory(target)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        encoded = json.dumps(dict(payload), sort_keys=True, indent=2).encode("utf-8")
        try:
            with temporary_path.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            RuntimePersistence._sync_parent_directory(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _sync_parent_directory(path: Path) -> None:
        """Persist rename metadata when the host exposes fsync-able directories."""

        try:
            descriptor = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            return
        finally:
            os.close(descriptor)

    def restore_checkpoint(self, path: str | Path) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        with self._lock:
            selected_path = Path(path)
            recovery_path = self._checkpoint_root / f".marulho_operator_restore_recovery_{uuid4().hex}.pt"
            staged_path = self._checkpoint_root / f".marulho_operator_restore_committed_{uuid4().hex}.pt"
            previous_checkpoint_path = self._checkpoint_path
            previous_checkpoint_dir = self._checkpoint_dir
            previous_metadata = deepcopy(self._metadata)
            previous_runtime_env = self._runtime_env
            previous_action_root = self._action_root
            previous_runtime_state = self._runtime_state.snapshot()
            self.save_checkpoint(str(recovery_path), publish=False)
            try:
                restored_metadata = self._hydrate_checkpoint_locked(selected_path)
                restore_lineage_validation = self._applied_replay_lineage_restore_validation(
                    restored_metadata
                )
                self._runtime_state.commit_restored_revision(
                    int(restored_metadata.get("state_revision", 0) or 0)
                )
                staged = self.save_checkpoint(str(staged_path), publish=False)
                manifest = self.publish_current_checkpoint(
                    Path(str(staged["path"])),
                    operation="operator_restore",
                )
                return {
                    "path": str(manifest["checkpoint_path"]),
                    "restored_from_path": str(selected_path),
                    "current_checkpoint_manifest": manifest,
                    "applied_replay_lineage_restore_validation": restore_lineage_validation,
                    **self._runtime_state.mutation_summary(),
                    "token_count": int(self._trainer.token_count),
                }
            except Exception:
                self._hydrate_checkpoint_locked(recovery_path)
                self._checkpoint_path = previous_checkpoint_path
                self._checkpoint_dir = previous_checkpoint_dir
                self._metadata = previous_metadata
                self._runtime_env = previous_runtime_env
                self._action_root = previous_action_root
                self._runtime_state.state_revision = int(previous_runtime_state["state_revision"])
                self._runtime_state.dirty_state = bool(previous_runtime_state["dirty_state"])
                self._runtime_state.restore_event_history(
                    last_event=previous_runtime_state.get("last_event"),
                    recent_events=previous_runtime_state.get("recent_events"),
                )
                self._dependencies.refresh_root_captures()
                raise
            finally:
                for temporary_path in (recovery_path, staged_path):
                    if temporary_path.exists():
                        temporary_path.unlink()

    def _hydrate_checkpoint_locked(self, checkpoint_path: Path) -> dict[str, Any]:
        trainer, metadata = load_trainer_checkpoint(checkpoint_path)
        self._trainer = trainer
        self._metadata = dict(metadata)
        self._encoder = self._trainer.encoder
        self._checkpoint_path = checkpoint_path
        self._checkpoint_dir = self.checkpoint_root_for_path(checkpoint_path)
        self._runtime_env = load_runtime_env(anchor_paths=(self._env_root, self._checkpoint_dir))
        self._action_root = (self._env_root or self._checkpoint_dir).resolve()
        service_state = dict(self._metadata.get("service_state", {}))
        terminus_state = dict(service_state.get("terminus_runtime", service_state.get("brain_runtime")) or {})
        concept_state = service_state.get("concept_store")
        self._snn_language_plasticity_state = dict(service_state.get("snn_language_plasticity") or {})
        service_state["snn_applied_replay_lineage_restore_validation"] = (
            self._applied_replay_lineage_restore_validation(self._metadata)
        )
        self._metadata["service_state"] = service_state
        self._snn_language_readout_ledger_state = (
            normalize_snn_language_readout_ledger_state(
                service_state.get("snn_language_readout_ledger")
            )
        )
        self._concept_store = ConceptStore.from_state_dict(concept_state)
        geometric_curiosity_state = cast(
            dict[str, Any] | None,
            terminus_state.get("geometric_curiosity"),
        )
        self._geometric_curiosity = GeometricCuriosityController.from_state_dict(
            self._trainer.model.abstraction_layer,
            geometric_curiosity_state,
        )
        self._brain_config = self._runtime_config._normalize_brain_config(terminus_state)
        self._brain_source_utility = self._normalize_background_source_utility_state(
            terminus_state.get("background_source_utility")
        )
        self._brain_last_error = None
        self._dependencies.action_executor().history = list(
            terminus_state.get("action_history") or []
        )
        replay_controller = self._dependencies.replay_controller()
        replay_controller.regeneration_permits = (
            terminus_state.get("replay_regeneration_permits") or []
        )
        replay_controller.snn_replay_evaluation_contexts = (
            terminus_state.get("snn_replay_evaluation_contexts") or []
        )
        replay_controller.snn_replay_artifact_recording_review_tickets = (
            terminus_state.get("snn_replay_artifact_recording_review_tickets") or []
        )
        replay_controller.snn_sleep_plasticity_review_tickets = (
            terminus_state.get("snn_sleep_plasticity_review_tickets") or []
        )
        replay_controller.snn_sleep_plasticity_scheduler_design_review_tickets = (
            terminus_state.get("snn_sleep_plasticity_scheduler_design_review_tickets")
            or []
        )
        replay_controller.snn_sleep_plasticity_review_scheduler_installations = (
            terminus_state.get("snn_sleep_plasticity_review_scheduler_installations")
            or []
        )
        replay_controller.snn_transition_memory_replay_artifacts = (
            terminus_state.get("snn_transition_memory_replay_artifacts") or []
        )
        self._delayed_consequence_records = deque(
            (
                item
                for item in (
                    self._normalize_delayed_consequence_record(raw_item)
                    for raw_item in list(terminus_state.get("delayed_consequence_records") or [])
                )
                if item is not None
            ),
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )
        self._delayed_consequence_cooled_total = max(0, int(terminus_state.get("delayed_consequence_cooled_total", 0) or 0))
        self._delayed_consequence_retired_total = max(0, int(terminus_state.get("delayed_consequence_retired_total", 0) or 0))
        self._delayed_consequence_compacted_total = max(0, int(terminus_state.get("delayed_consequence_compacted_total", 0) or 0))
        self._delayed_consequence_split_total = max(0, int(terminus_state.get("delayed_consequence_split_total", 0) or 0))
        self._delayed_consequence_remerged_total = max(0, int(terminus_state.get("delayed_consequence_remerged_total", 0) or 0))
        self._delayed_consequence.restore_state(terminus_state)
        self._brain_last_acquisition_summary = None
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        self._dependencies.refresh_root_captures()
        self._brain_runtime.restore_runtime_state(terminus_state)
        self._rebuild_brain_sources_locked()
        self._interaction_pipeline.load_interaction_state(
            recent_query_gaps=list(terminus_state.get("recent_query_gaps") or []),
            runtime_episode_traces=list(terminus_state.get("runtime_episode_traces") or []),
        )
        self._runtime_state.restore_event_history(
            last_event=terminus_state.get("last_event"),
            recent_events=terminus_state.get("recent_events"),
        )
        return self._metadata

    def _applied_replay_lineage_restore_validation(
        self,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        service_state = (
            metadata.get("service_state")
            if isinstance(metadata.get("service_state"), Mapping)
            else {}
        )
        saved_summary = (
            service_state.get("snn_applied_replay_lineage_checkpoint_summary")
            if isinstance(
                service_state.get("snn_applied_replay_lineage_checkpoint_summary"),
                Mapping,
            )
            else {}
        )
        restored_summary = applied_replay_lineage_checkpoint_summary(
            self._snn_language_plasticity_state,
            source="runtime_persistence.restore_checkpoint",
        )
        summary_available = bool(saved_summary)
        def _summary_int(summary: Mapping[str, Any], key: str, default: int) -> int:
            value = summary.get(key)
            return default if value is None else int(value)

        counts_match = bool(
            summary_available
            and _summary_int(saved_summary, "applied_replay_lineage_count", -1)
            == _summary_int(restored_summary, "applied_replay_lineage_count", -2)
            and _summary_int(saved_summary, "complete_applied_replay_lineage_count", -1)
            == _summary_int(restored_summary, "complete_applied_replay_lineage_count", -2)
            and _summary_int(saved_summary, "incomplete_applied_replay_lineage_count", -1)
            == _summary_int(restored_summary, "incomplete_applied_replay_lineage_count", -2)
        )
        hash_matches = bool(
            summary_available
            and saved_summary.get("lineage_material_hash")
            == restored_summary.get("lineage_material_hash")
        )
        return {
            "surface": "snn_applied_replay_lineage_restore_validation.v1",
            "source": "runtime_persistence.restore_checkpoint",
            "owned_by_marulho": True,
            "saved_summary_available": summary_available,
            "summary_counts_match_restored_state": counts_match,
            "summary_hash_matches_restored_state": hash_matches,
            "summary_matches_restored_state": bool(counts_match and hash_matches),
            "saved_summary": deepcopy(dict(saved_summary)),
            "restored_summary": restored_summary,
            "summary_source_available": bool(
                restored_summary.get("summary_source_available")
            ),
            "summary_policy": restored_summary.get("summary_policy"),
            "full_provenance_scan": False,
            "source_record_scan_count": 0,
            "archival_metadata_device": "cpu",
            "gpu_used": False,
            "runs_replay": False,
            "applies_plasticity": False,
            "issues_regeneration_permit": False,
            "raw_text_absent": True,
            "operator_identity_absent": True,
        }

    def _service_state_snapshot(self) -> dict[str, Any]:
        last_trace = self._trace_history[0] if self._trace_history else None
        return {
            "checkpoint_path": str(self._checkpoint_path),
            **self._runtime_state.mutation_summary(),
            "token_count": int(self._trainer.token_count),
            "last_trace_id": None if last_trace is None else str(last_trace.get("trace_id")),
            "concept_count": int(self._concept_store.snapshot().get("concept_count", 0)),
            "terminus_runtime": self._brain_runtime_snapshot_locked(),
        }

    def _resolve_save_path(self, path: str | None) -> Path:
        if path:
            return Path(path)

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return self._checkpoint_dir / f"marulho_service_{stamp}_{uuid4().hex}.pt"

    def _persist_trace_locked(self, trace: dict[str, Any]) -> Path:
        payload = self._json_safe(trace)
        created_at = str(payload.get("created_at", datetime.now(timezone.utc).isoformat()))
        timestamp = created_at.replace(":", "").replace("-", "")
        trace_id = str(payload.get("trace_id", uuid4()))
        trace_path = self._trace_dir / f"{timestamp}_{trace_id}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(trace_path, payload)
        payload["trace_path"] = str(trace_path)
        self._trace_history.appendleft(deepcopy(payload))
        return trace_path

    def _load_persisted_traces_locked(self) -> None:
        trace_files = sorted(self._trace_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in trace_files[: self._trace_history.maxlen]:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            payload["trace_path"] = str(path)
            self._trace_history.append(payload)

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, deque)):
            return [self._json_safe(item) for item in value]
        return str(value)

    def _record_brain_event_locked(self, event: dict[str, Any]) -> None:
        self._runtime_state.record_event(event)
