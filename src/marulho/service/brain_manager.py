from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

from marulho.brain import MarulhoBrain
from marulho.reporting import (
    build_current_language_evidence_projection,
    build_evidence_report_inventory,
)


CURRENT_CHECKPOINT_MANIFEST = "marulho_current_checkpoint.json"


class MarulhoBrainRuntimeFacade:
    """Small operator facade for the active /brain service contract."""

    def __init__(self, manager: "MarulhoBrainServiceManager") -> None:
        self._manager = manager

    def checkpoint_list(self) -> list[dict[str, Any]]:
        return self._manager.checkpoint_list()

    def brain_status(self) -> dict[str, Any]:
        return self._manager.brain.status()

    def brain_feed(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.feed(**kwargs)

    def brain_tick(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.tick(**kwargs)

    def brain_generate(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.generate(**kwargs)

    def brain_replay(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.replay(**kwargs)

    def brain_grow_prune(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.grow_prune(**kwargs)

    def brain_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._manager.brain.trace_history(limit=limit)

    def brain_start(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.start(**kwargs)

    def brain_stop(self, **kwargs: Any) -> dict[str, Any]:
        return self._manager.brain.stop(**kwargs)

    def save_brain_checkpoint(self, path: str | None = None) -> dict[str, Any]:
        return self._manager.save_brain_checkpoint(path)

    def restore_checkpoint(self, path: str) -> dict[str, Any]:
        return self._manager.restore_checkpoint(path)

    def evidence_report_inventory(self, limit: int = 20) -> dict[str, Any]:
        return self._manager.evidence_report_inventory(limit=limit)

    def current_language_evidence(self) -> dict[str, Any]:
        return self._manager.current_language_evidence()


class MarulhoBrainServiceManager:
    """FastAPI composition root over the single MarulhoBrain runtime."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        trace_history_limit: int = 200,
        trace_dir: str | Path | None = None,
        env_root: str | Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._trace_history_limit = max(1, int(trace_history_limit))
        self._trace_dir = None if trace_dir is None else Path(trace_dir)
        if self._trace_dir is not None:
            self._trace_dir.mkdir(parents=True, exist_ok=True)
        self._env_root = None if env_root is None else Path(env_root)
        self._checkpoint_path = self.resolve_current_checkpoint_path(checkpoint_path)
        self._checkpoint_dir = self.checkpoint_root_for_path(self._checkpoint_path)
        self._brain = MarulhoBrain.load(
            self._checkpoint_path,
            trace_limit=self._trace_history_limit,
        )
        self._runtime_facade = MarulhoBrainRuntimeFacade(self)

    @property
    def brain(self) -> MarulhoBrain:
        return self._brain

    @property
    def runtime_facade(self) -> MarulhoBrainRuntimeFacade:
        return self._runtime_facade

    def close(self) -> None:
        self._brain.stop(timeout_seconds=2.0)

    def checkpoint_list(self) -> list[dict[str, Any]]:
        with self._lock:
            roots = [self._checkpoint_dir]
            object_dir = self._checkpoint_dir / "objects"
            if object_dir.is_dir():
                roots.append(object_dir)
            paths: dict[Path, None] = {}
            for root in roots:
                if root.is_dir():
                    for path in root.glob("*.pt"):
                        paths[path.resolve()] = None
            records: list[dict[str, Any]] = []
            for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
                stat = path.stat()
                records.append(
                    {
                        "path": str(path),
                        "name": path.name,
                        "size_bytes": int(stat.st_size),
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime,
                            tz=timezone.utc,
                        ).isoformat(),
                    }
                )
            return records

    def save_brain_checkpoint(self, path: str | None = None) -> dict[str, Any]:
        with self._lock:
            saved = self._brain.save(path)
            self._checkpoint_path = Path(str(saved["path"]))
            self._checkpoint_dir = self.checkpoint_root_for_path(self._checkpoint_path)
            return dict(saved)

    def restore_checkpoint(self, path: str | Path) -> dict[str, Any]:
        selected_path = self.resolve_current_checkpoint_path(path)
        with self._lock:
            self._brain.stop(timeout_seconds=2.0)
            self._brain = MarulhoBrain.load(
                selected_path,
                trace_limit=self._trace_history_limit,
            )
            self._checkpoint_path = selected_path
            self._checkpoint_dir = self.checkpoint_root_for_path(selected_path)
            status = self._brain.status()
            return {
                "surface": "marulho_brain_checkpoint_restore.v1",
                "path": str(selected_path),
                "restored_from_path": str(path),
                "token_count": int(status.get("token_count", 0) or 0),
                "readout": deepcopy(status.get("readout", {})),
                "trace": self._brain.trace(),
            }

    def evidence_report_inventory(self, limit: int = 20) -> dict[str, Any]:
        reports_root = (
            self._env_root / "reports"
            if self._env_root is not None
            else Path("reports")
        )
        return build_evidence_report_inventory(reports_root, limit=limit)

    def current_language_evidence(self) -> dict[str, Any]:
        reports_root = (
            self._env_root / "reports"
            if self._env_root is not None
            else Path("reports")
        )
        return build_current_language_evidence_projection(reports_root)

    @staticmethod
    def checkpoint_root_for_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        if path.is_dir():
            return path
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
        if (
            payload.get("schema_version") != 1
            or payload.get("artifact_kind") != "marulho_current_checkpoint_manifest"
        ):
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

    @classmethod
    def _validated_descriptor_path(
        cls,
        root: Path,
        descriptor: Mapping[str, Any],
    ) -> Path | None:
        try:
            candidate = (root / str(descriptor.get("relative_path") or "")).resolve()
            candidate.relative_to(root)
            if not candidate.is_file():
                return None
            if int(descriptor.get("size_bytes", -1)) != int(candidate.stat().st_size):
                return None
            if str(descriptor.get("sha256") or "") != cls._sha256_file(candidate):
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
