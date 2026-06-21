from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import islice
import hashlib
import json
import random
import time
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from marulho.service.living_loop_replay import (
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    build_replay_plan,
    replay_candidate_safety_flags,
)
from marulho.service.snn_language_plasticity_executor import (
    bounded_application_synapse_window,
)

DEFAULT_REPLAY_SAMPLE_HISTORY = 256
DEFAULT_REPLAY_REGENERATION_PERMITS = 64
DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS = 64
DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS = 64
DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS = 64
DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS = 64
DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS = 16
DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS = 64
SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT = 16
SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT = 16
SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT = 16
SNN_REPLAY_PRIORITY_READOUT_TARGET_LIMIT = 16
SNN_REPLAY_PRIORITY_SOURCE_WINDOW_POLICY = "bounded_recent_context_readout_target_window_v1"
SNN_REPLAY_PROVENANCE_SOURCE_RECORD_LIMIT = 4
SNN_REPLAY_PROVENANCE_SOURCE_WINDOW_POLICY = "indexed_context_ticket_artifact_permit_window_v1"
SNN_READOUT_KNOWN_EVIDENCE_SOURCE_WINDOW_SURFACE = (
    "bounded_snn_readout_known_evidence_hash_source_window.v1"
)
SNN_REPLAY_PRIORITY_SOURCE_WINDOW_FALSE_FLAGS = (
    "gpu_used",
    "global_candidate_scan",
    "global_score_scan",
    "runs_live_tick",
    "runs_live_replay",
    "records_replay_artifact",
    "raw_text_payload_loaded",
    "language_reasoning",
    "mutates_runtime_state",
    "applies_plasticity",
)
SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_FALSE_FLAGS = (
    "gpu_used",
    "global_candidate_scan",
    "global_score_scan",
    "raw_text_payload_loaded",
    "language_reasoning",
    "runs_live_tick",
    "runs_every_token",
    "mutates_runtime_state",
    "applies_plasticity",
)
SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT = 32
MAX_REPLAY_SAMPLE_LIMIT = 20
MAX_RUNTIME_TRACE_EXPORT_LIMIT = 50
REPLAY_SAMPLE_SUMMARY_SOURCE_WINDOW_LIMIT = 64
SNN_SLEEP_PLASTICITY_REVIEW_GATES = {
    "review_rollout_regeneration_application": (
        "rollout_regeneration_application_review",
        "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application",
    ),
    "review_rollout_regeneration_application_preflight": (
        "rollout_regeneration_application_preflight_review",
        "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application-preflight",
    ),
    "review_rollout_regeneration_permit_request": (
        "rollout_regeneration_permit_request_review",
        "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request",
    ),
    "review_transition_memory_homeostatic_maintenance": (
        "transition_memory_homeostatic_maintenance_review",
        "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
    ),
    "review_replay_artifact_recording_or_rollout_regeneration": (
        "replay_artifact_recording_or_rollout_regeneration_review",
        "/terminus/snn-language-sequence/transition-memory-replay-artifact/proposal",
    ),
}


def _known_readout_evidence_source_window_bounded(
    report: Mapping[str, Any],
) -> bool:
    try:
        source_window_count = int(report.get("source_window_count", -1))
        source_window_limit = int(report.get("source_window_limit", -1))
    except (TypeError, ValueError):
        return False
    explicit_false_flags = (
        "global_candidate_scan",
        "global_score_scan",
        "raw_text_payload_loaded",
        "language_reasoning",
        "runs_live_tick",
        "runs_every_token",
        "mutates_runtime_state",
        "applies_plasticity",
        "gpu_used",
    )
    return (
        report.get("surface") == SNN_READOUT_KNOWN_EVIDENCE_SOURCE_WINDOW_SURFACE
        and source_window_limit > 0
        and 0 <= source_window_count <= source_window_limit
        and all(report.get(flag) is False for flag in explicit_false_flags)
        and report.get("archival_storage_device") == "cpu"
    )


def _snn_replay_priority_source_window_bounded(
    source_window: Mapping[str, Any],
) -> bool:
    try:
        recent_count = int(source_window.get("recent_context_window_count", -1))
        recent_limit = int(source_window.get("recent_context_window_limit", -1))
        target_request_count = int(
            source_window.get("readout_target_request_count", -1)
        )
        target_context_count = int(source_window.get("readout_target_context_count", -1))
        target_limit = int(source_window.get("readout_target_window_limit", -1))
        source_context_count = int(source_window.get("source_context_count", -1))
        verified_context_count = int(source_window.get("verified_context_count", -1))
    except (TypeError, ValueError):
        return False
    return (
        source_window.get("surface") == "bounded_snn_replay_priority_source_window.v1"
        and 0 < recent_limit <= SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT
        and 0 <= recent_count <= recent_limit
        and 0 < target_limit <= SNN_REPLAY_PRIORITY_READOUT_TARGET_LIMIT
        and 0 <= target_request_count <= target_limit
        and 0 <= target_context_count <= target_request_count
        and 0 <= verified_context_count <= source_context_count
        and source_context_count <= recent_count + target_context_count
        and all(
            source_window.get(flag) is False
            for flag in SNN_REPLAY_PRIORITY_SOURCE_WINDOW_FALSE_FLAGS
        )
        and source_window.get("archival_storage_device") == "cpu"
        and source_window.get("score_device") == "cpu"
    )


def _snn_readout_replay_priority_source_window_bounded(
    source_window: Mapping[str, Any],
) -> bool:
    try:
        source_event_window_count = int(
            source_window.get("source_event_window_count", -1)
        )
        source_event_window_limit = int(
            source_window.get("source_event_window_limit", -1)
        )
        source_event_retention_count = int(
            source_window.get("source_event_retention_count", -1)
        )
        candidate_count_before_rank = int(
            source_window.get("candidate_count_before_rank", -1)
        )
        candidate_count_returned = int(
            source_window.get("candidate_count_returned", -1)
        )
    except (TypeError, ValueError):
        return False
    return (
        source_window.get("surface")
        == "bounded_snn_readout_replay_priority_source_window.v1"
        and 0 < source_event_window_limit <= SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_LIMIT
        and 0 <= source_event_window_count <= source_event_window_limit
        and source_event_window_count <= source_event_retention_count
        and 0 <= candidate_count_before_rank <= source_event_window_count
        and 0 <= candidate_count_returned <= candidate_count_before_rank
        and all(
            source_window.get(flag) is False
            for flag in SNN_READOUT_REPLAY_PRIORITY_SOURCE_WINDOW_FALSE_FLAGS
        )
        and source_window.get("archival_storage_device") == "cpu"
        and source_window.get("score_device") == "cpu"
    )


class _IndexedDeque(deque):
    """Deque with a side index for current retained control-plane records."""

    def __init__(
        self,
        *,
        maxlen: int,
        index: dict[str, dict[str, Any]],
        key: Callable[[Mapping[str, Any]], str],
    ) -> None:
        super().__init__(maxlen=maxlen)
        self._index = index
        self._key = key

    def _item_key(self, item: Any) -> str:
        if not isinstance(item, Mapping):
            return ""
        return str(self._key(item) or "").strip()

    def _drop_item(self, item: Any) -> None:
        key = self._item_key(item)
        if key and self._index.get(key) is item:
            self._index.pop(key, None)

    def _index_item(self, item: Any, *, overwrite: bool) -> None:
        key = self._item_key(item)
        if not key or not isinstance(item, dict):
            return
        if overwrite or key not in self._index:
            self._index[key] = item

    def rebuild_index(self) -> None:
        self._index.clear()
        for item in self:
            self._index_item(item, overwrite=False)

    def appendleft(self, item: Any) -> None:  # type: ignore[override]
        if self.maxlen is not None and len(self) == self.maxlen and len(self) > 0:
            self._drop_item(self[-1])
        super().appendleft(item)
        self._index_item(self[0], overwrite=True)

    def append(self, item: Any) -> None:  # type: ignore[override]
        if self.maxlen is not None and len(self) == self.maxlen and len(self) > 0:
            self._drop_item(self[0])
        super().append(item)
        self._index_item(self[-1], overwrite=False)

    def extend(self, items: Any) -> None:  # type: ignore[override]
        for item in items:
            self.append(item)

    def extendleft(self, items: Any) -> None:  # type: ignore[override]
        for item in items:
            self.appendleft(item)

    def clear(self) -> None:  # type: ignore[override]
        super().clear()
        self._index.clear()

    def pop(self) -> Any:  # type: ignore[override]
        item = super().pop()
        self._drop_item(item)
        return item

    def popleft(self) -> Any:  # type: ignore[override]
        item = super().popleft()
        self._drop_item(item)
        return item

    def __setitem__(self, index: Any, item: Any) -> None:
        super().__setitem__(index, item)
        self.rebuild_index()

    def __delitem__(self, index: Any) -> None:
        super().__delitem__(index)
        self.rebuild_index()

    def insert(self, index: int, item: Any) -> None:  # type: ignore[override]
        super().insert(index, item)
        self.rebuild_index()

    def remove(self, value: Any) -> None:  # type: ignore[override]
        super().remove(value)
        self.rebuild_index()


@dataclass(frozen=True)
class ReplayControllerDependencies:
    action_history: Callable[[], Sequence[Mapping[str, Any]]]
    living_loop_snapshot: Callable[..., Mapping[str, Any]]
    lock: Any
    normalize_action_text: Callable[[Any], str]
    normalize_feedback_text: Callable[..., str]
    replay_plan_summary: Callable[[Any], Mapping[str, Any]]
    runtime_feedback_summary: Callable[[], Mapping[str, Any]]
    runtime_state: Any
    runtime_trace_export_safe_value: Callable[[Any], Any]
    trainer: Callable[[], Any]


class ReplayController:
    """Advisory replay planning and operator-gated replay sampling helpers."""

    def __init__(
        self,
        dependencies: ReplayControllerDependencies,
        *,
        replay_sample_history: Sequence[Mapping[str, Any]] | None = None,
        regeneration_permits: Sequence[Mapping[str, Any]] | None = None,
        snn_replay_evaluation_contexts: Sequence[Mapping[str, Any]] | None = None,
        snn_replay_artifact_recording_review_tickets: Sequence[Mapping[str, Any]] | None = None,
        snn_sleep_plasticity_review_tickets: Sequence[Mapping[str, Any]] | None = None,
        snn_sleep_plasticity_scheduler_design_review_tickets: Sequence[Mapping[str, Any]] | None = None,
        snn_sleep_plasticity_review_scheduler_installations: Sequence[Mapping[str, Any]] | None = None,
        snn_transition_memory_replay_artifacts: Sequence[Mapping[str, Any]] | None = None,
        history_maxlen: int = DEFAULT_REPLAY_SAMPLE_HISTORY,
    ) -> None:
        self._dependencies = dependencies
        self._history_maxlen = max(1, int(history_maxlen))
        self._replay_sample_history: deque[dict[str, Any]] = deque(maxlen=self._history_maxlen)
        self._regeneration_permit_index: dict[str, dict[str, Any]] = {}
        self._regeneration_permits: deque[dict[str, Any]] = _IndexedDeque(
            maxlen=DEFAULT_REPLAY_REGENERATION_PERMITS,
            index=self._regeneration_permit_index,
            key=self._regeneration_permit_id,
        )
        self._snn_replay_evaluation_contexts: deque[dict[str, Any]] = deque(
            maxlen=DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS
        )
        self._snn_replay_evaluation_context_index: dict[str, dict[str, Any]] = {}
        self._snn_replay_artifact_recording_review_ticket_index: dict[str, dict[str, Any]] = {}
        self._snn_replay_artifact_recording_review_tickets: deque[dict[str, Any]] = _IndexedDeque(
            maxlen=DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS,
            index=self._snn_replay_artifact_recording_review_ticket_index,
            key=self._snn_replay_artifact_recording_review_ticket_id,
        )
        self._snn_sleep_plasticity_review_ticket_index: dict[str, dict[str, Any]] = {}
        self._snn_sleep_plasticity_review_tickets: deque[dict[str, Any]] = _IndexedDeque(
            maxlen=DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS,
            index=self._snn_sleep_plasticity_review_ticket_index,
            key=self._snn_sleep_plasticity_review_ticket_id,
        )
        self._snn_sleep_plasticity_scheduler_design_review_ticket_index: dict[str, dict[str, Any]] = {}
        self._snn_sleep_plasticity_scheduler_design_review_tickets: deque[dict[str, Any]] = _IndexedDeque(
            maxlen=DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS,
            index=self._snn_sleep_plasticity_scheduler_design_review_ticket_index,
            key=self._snn_sleep_plasticity_scheduler_design_review_ticket_id,
        )
        self._snn_sleep_plasticity_review_scheduler_installations: deque[dict[str, Any]] = deque(
            maxlen=DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS
        )
        self._snn_transition_memory_replay_artifact_index: dict[str, dict[str, Any]] = {}
        self._snn_transition_memory_replay_artifacts: deque[dict[str, Any]] = _IndexedDeque(
            maxlen=DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS,
            index=self._snn_transition_memory_replay_artifact_index,
            key=self._snn_transition_memory_replay_artifact_id,
        )
        self.load_replay_sample_history(replay_sample_history or [])
        self.load_regeneration_permits(regeneration_permits or [])
        self.load_snn_replay_evaluation_contexts(snn_replay_evaluation_contexts or [])
        self.load_snn_replay_artifact_recording_review_tickets(
            snn_replay_artifact_recording_review_tickets or []
        )
        self.load_snn_sleep_plasticity_review_tickets(snn_sleep_plasticity_review_tickets or [])
        self.load_snn_sleep_plasticity_scheduler_design_review_tickets(
            snn_sleep_plasticity_scheduler_design_review_tickets or []
        )
        self.load_snn_sleep_plasticity_review_scheduler_installations(
            snn_sleep_plasticity_review_scheduler_installations or []
        )
        self.load_snn_transition_memory_replay_artifacts(snn_transition_memory_replay_artifacts or [])

    @property
    def _action_history(self) -> Sequence[Mapping[str, Any]]:
        return self._dependencies.action_history()

    @property
    def _lock(self) -> Any:
        return self._dependencies.lock

    @property
    def _runtime_state(self) -> Any:
        return self._dependencies.runtime_state

    @property
    def _trainer(self) -> Any:
        return self._dependencies.trainer()

    def _living_loop_snapshot_locked(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._dependencies.living_loop_snapshot(**kwargs)

    def _normalize_action_text(self, value: Any) -> str:
        return self._dependencies.normalize_action_text(value)

    def _normalize_feedback_text(self, value: Any, **kwargs: Any) -> str:
        return self._dependencies.normalize_feedback_text(value, **kwargs)

    def _replay_plan_summary(self, replay_plan: Any) -> Mapping[str, Any]:
        return self._dependencies.replay_plan_summary(replay_plan)

    def _runtime_feedback_summary_locked(self) -> Mapping[str, Any]:
        return self._dependencies.runtime_feedback_summary()

    def _runtime_trace_export_safe_value(self, value: Any) -> Any:
        return self._dependencies.runtime_trace_export_safe_value(value)

    @staticmethod
    def _snn_replay_context_emission_lineage(
        source_metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(source_metadata or {})
        lineage_keys = (
            "source",
            "surface",
            "design_hash",
            "seed_hash",
            "emission_review_hash",
            "emission_hash",
            "readout_evidence_hash",
            "prediction_hash",
        )
        return {
            key: metadata[key]
            for key in lineage_keys
            if metadata.get(key) not in (None, "")
        }

    @staticmethod
    def _snn_replay_context_id(context: Mapping[str, Any] | None) -> str:
        if not isinstance(context, Mapping):
            return ""
        return str(context.get("replay_evaluation_context_id") or "").strip()

    @staticmethod
    def _regeneration_permit_id(permit: Mapping[str, Any] | None) -> str:
        if not isinstance(permit, Mapping):
            return ""
        return str(permit.get("permit_id") or "").strip()

    @staticmethod
    def _snn_replay_artifact_recording_review_ticket_id(ticket: Mapping[str, Any] | None) -> str:
        if not isinstance(ticket, Mapping):
            return ""
        return str(ticket.get("review_ticket_id") or "").strip()

    @staticmethod
    def _snn_sleep_plasticity_review_ticket_id(ticket: Mapping[str, Any] | None) -> str:
        if not isinstance(ticket, Mapping):
            return ""
        return str(ticket.get("review_ticket_id") or "").strip()

    @staticmethod
    def _snn_sleep_plasticity_scheduler_design_review_ticket_id(
        ticket: Mapping[str, Any] | None,
    ) -> str:
        if not isinstance(ticket, Mapping):
            return ""
        return str(ticket.get("scheduler_design_review_ticket_id") or "").strip()

    @staticmethod
    def _snn_transition_memory_replay_artifact_id(artifact: Mapping[str, Any] | None) -> str:
        if not isinstance(artifact, Mapping):
            return ""
        return str(artifact.get("replay_artifact_id") or "").strip()

    def _normalize_evaluated_snn_transition_memory_replay_artifact(
        self,
        artifact: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Keep only evaluated, internal-ledger-backed artifacts in retention."""

        if not isinstance(artifact, Mapping):
            return None
        normalized = dict(artifact)
        source_window = (
            dict(normalized.get("source_window"))
            if isinstance(normalized.get("source_window"), Mapping)
            else {}
        )
        readout_evidence_source_window = (
            dict(normalized.get("readout_evidence_source_window"))
            if isinstance(normalized.get("readout_evidence_source_window"), Mapping)
            else {}
        )
        replay_priority_source_window = (
            dict(normalized.get("replay_priority_source_window"))
            if isinstance(normalized.get("replay_priority_source_window"), Mapping)
            else {}
        )
        required_text_fields = (
            "replay_artifact_id",
            "evidence_hash",
            "artifact_proposal_hash",
            "replay_evaluation_context_id",
            "replay_evaluation_context_hash",
            "review_ticket_id",
            "review_ticket_hash",
            "source_window_hash",
            "readout_evidence_source_window_hash",
            "replay_priority_source_window_hash",
        )
        if (
            normalized.get("surface") != "snn_transition_memory_replay_artifact.v1"
            or normalized.get("artifact_kind")
            != "terminus_snn_transition_memory_replay_artifact"
            or normalized.get("internal_ledger_backed") is not True
            or not all(str(normalized.get(field) or "") for field in required_text_fields)
            or not list(normalized.get("readout_evidence_hashes") or [])
            or not source_window
            or not readout_evidence_source_window
            or not replay_priority_source_window
            or str(normalized.get("source_window_hash") or "")
            != self._sha256_json(source_window)
            or str(normalized.get("readout_evidence_source_window_hash") or "")
            != self._sha256_json(readout_evidence_source_window)
            or str(normalized.get("replay_priority_source_window_hash") or "")
            != self._sha256_json(replay_priority_source_window)
            or not _known_readout_evidence_source_window_bounded(
                readout_evidence_source_window
            )
            or not _snn_readout_replay_priority_source_window_bounded(
                replay_priority_source_window
            )
        ):
            return None
        return normalized

    def _rebuild_snn_replay_evaluation_context_index_locked(self) -> None:
        self._snn_replay_evaluation_context_index.clear()
        for context in self._snn_replay_evaluation_contexts:
            context_id = self._snn_replay_context_id(context)
            if context_id and context_id not in self._snn_replay_evaluation_context_index:
                self._snn_replay_evaluation_context_index[context_id] = context

    def _snn_replay_provenance_source_window_locked(
        self,
        *,
        replay_evaluation_context_id: str | None = None,
        review_ticket_id: str | None = None,
        replay_artifact_id: str | None = None,
        permit_id: str | None = None,
    ) -> dict[str, Any]:
        context_id = str(replay_evaluation_context_id or "").strip()
        ticket_id = str(review_ticket_id or "").strip()
        artifact_id = str(replay_artifact_id or "").strip()
        regeneration_permit_id = str(permit_id or "").strip()
        requested = {
            "replay_evaluation_context": bool(context_id),
            "review_ticket": bool(ticket_id),
            "replay_artifact": bool(artifact_id),
            "regeneration_permit": bool(regeneration_permit_id),
        }
        hits = {
            "replay_evaluation_context": bool(
                context_id and context_id in self._snn_replay_evaluation_context_index
            ),
            "review_ticket": bool(
                ticket_id and ticket_id in self._snn_replay_artifact_recording_review_ticket_index
            ),
            "replay_artifact": bool(
                artifact_id and artifact_id in self._snn_transition_memory_replay_artifact_index
            ),
            "regeneration_permit": bool(
                regeneration_permit_id and regeneration_permit_id in self._regeneration_permit_index
            ),
        }
        source_record_count = sum(1 for enabled in requested.values() if enabled)
        return {
            "surface": "bounded_snn_replay_artifact_provenance_source_window.v1",
            "policy": SNN_REPLAY_PROVENANCE_SOURCE_WINDOW_POLICY,
            "source_record_limit": SNN_REPLAY_PROVENANCE_SOURCE_RECORD_LIMIT,
            "source_record_count": int(source_record_count),
            "index_lookup_count": int(source_record_count),
            "index_hit_count": int(sum(1 for enabled in hits.values() if enabled)),
            "lookup_requested": requested,
            "lookup_hit": hits,
            "replay_evaluation_context_retention_count": int(
                len(self._snn_replay_evaluation_contexts)
            ),
            "review_ticket_retention_count": int(
                len(self._snn_replay_artifact_recording_review_tickets)
            ),
            "replay_artifact_retention_count": int(
                len(self._snn_transition_memory_replay_artifacts)
            ),
            "regeneration_permit_retention_count": int(len(self._regeneration_permits)),
            "context_lookup_policy": "replay_evaluation_context_id_index",
            "review_ticket_lookup_policy": "review_ticket_id_index",
            "artifact_lookup_policy": "replay_artifact_id_index",
            "permit_lookup_policy": "permit_id_index",
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_live_replay": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "score_device": "cpu",
            "gpu_used": False,
        }

    def _snn_replay_provenance_source_window(
        self,
        *,
        replay_evaluation_context_id: str | None = None,
        review_ticket_id: str | None = None,
        replay_artifact_id: str | None = None,
        permit_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            return self._snn_replay_provenance_source_window_locked(
                replay_evaluation_context_id=replay_evaluation_context_id,
                review_ticket_id=review_ticket_id,
                replay_artifact_id=replay_artifact_id,
                permit_id=permit_id,
            )

    def _verified_snn_replay_evaluation_context_payload_locked(
        self,
        context: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(context, Mapping):
            return None
        metadata = (
            context.get("source_metadata")
            if isinstance(context.get("source_metadata"), Mapping)
            else {}
        )
        material = {
            "recorded_state_revision": int(context.get("recorded_state_revision", -1)),
            "mismatch_hash": context.get("mismatch_hash"),
            "pressure_hash": context.get("pressure_hash"),
            "source_metadata_hash": context.get("source_metadata_hash"),
        }
        return (
            deepcopy(dict(context))
            if (
                context.get("ready")
                and context.get("owned_by_marulho")
                and int(context.get("recorded_state_revision", -1))
                == int(self._runtime_state.state_revision)
                and str(context.get("evidence_hash") or "") == self._sha256_json(material)
                and str(context.get("mismatch_hash") or "")
                == self._sha256_json(dict(context.get("mismatch_report") or {}))
                and str(context.get("pressure_hash") or "")
                == self._sha256_json(dict(context.get("pressure_report") or {}))
                and (
                    context.get("source_metadata_hash") is None
                    or str(context.get("source_metadata_hash") or "")
                    == self._sha256_json(dict(metadata))
                )
            )
            else None
        )

    @property
    def history(self) -> deque[dict[str, Any]]:
        return self._replay_sample_history

    @history.setter
    def history(self, replay_sample_history: Sequence[Mapping[str, Any]]) -> None:
        self.load_replay_sample_history(replay_sample_history)

    def load_replay_sample_history(self, replay_sample_history: Sequence[Mapping[str, Any]]) -> None:
        normalized = [
            item
            for item in (self._normalize_replay_sample_record(raw_item) for raw_item in replay_sample_history)
            if item is not None
        ]
        self._replay_sample_history.clear()
        self._replay_sample_history.extend(normalized)

    @property
    def regeneration_permits(self) -> deque[dict[str, Any]]:
        return self._regeneration_permits

    @regeneration_permits.setter
    def regeneration_permits(self, permits: Sequence[Mapping[str, Any]]) -> None:
        self.load_regeneration_permits(permits)

    def load_regeneration_permits(self, permits: Sequence[Mapping[str, Any]]) -> None:
        normalized = [dict(item) for item in permits if isinstance(item, Mapping)]
        self._regeneration_permits.clear()
        self._regeneration_permits.extend(normalized[:DEFAULT_REPLAY_REGENERATION_PERMITS])

    @property
    def snn_replay_evaluation_contexts(self) -> deque[dict[str, Any]]:
        return self._snn_replay_evaluation_contexts

    @snn_replay_evaluation_contexts.setter
    def snn_replay_evaluation_contexts(self, contexts: Sequence[Mapping[str, Any]]) -> None:
        self.load_snn_replay_evaluation_contexts(contexts)

    def load_snn_replay_evaluation_contexts(self, contexts: Sequence[Mapping[str, Any]]) -> None:
        normalized = [dict(item) for item in contexts if isinstance(item, Mapping)]
        self._snn_replay_evaluation_contexts.clear()
        self._snn_replay_evaluation_contexts.extend(
            normalized[:DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS]
        )
        self._rebuild_snn_replay_evaluation_context_index_locked()

    @property
    def snn_replay_artifact_recording_review_tickets(self) -> deque[dict[str, Any]]:
        return self._snn_replay_artifact_recording_review_tickets

    @snn_replay_artifact_recording_review_tickets.setter
    def snn_replay_artifact_recording_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        self.load_snn_replay_artifact_recording_review_tickets(tickets)

    def load_snn_replay_artifact_recording_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [dict(item) for item in tickets if isinstance(item, Mapping)]
        self._snn_replay_artifact_recording_review_tickets.clear()
        self._snn_replay_artifact_recording_review_tickets.extend(
            normalized[:DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS]
        )

    @property
    def snn_sleep_plasticity_review_tickets(self) -> deque[dict[str, Any]]:
        return self._snn_sleep_plasticity_review_tickets

    @snn_sleep_plasticity_review_tickets.setter
    def snn_sleep_plasticity_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        self.load_snn_sleep_plasticity_review_tickets(tickets)

    def load_snn_sleep_plasticity_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [dict(item) for item in tickets if isinstance(item, Mapping)]
        self._snn_sleep_plasticity_review_tickets.clear()
        self._snn_sleep_plasticity_review_tickets.extend(
            normalized[:DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS]
        )

    @property
    def snn_sleep_plasticity_scheduler_design_review_tickets(self) -> deque[dict[str, Any]]:
        return self._snn_sleep_plasticity_scheduler_design_review_tickets

    @snn_sleep_plasticity_scheduler_design_review_tickets.setter
    def snn_sleep_plasticity_scheduler_design_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        self.load_snn_sleep_plasticity_scheduler_design_review_tickets(tickets)

    def load_snn_sleep_plasticity_scheduler_design_review_tickets(
        self,
        tickets: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [dict(item) for item in tickets if isinstance(item, Mapping)]
        self._snn_sleep_plasticity_scheduler_design_review_tickets.clear()
        self._snn_sleep_plasticity_scheduler_design_review_tickets.extend(
            normalized[:DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS]
        )

    @property
    def snn_sleep_plasticity_review_scheduler_installations(self) -> deque[dict[str, Any]]:
        return self._snn_sleep_plasticity_review_scheduler_installations

    @snn_sleep_plasticity_review_scheduler_installations.setter
    def snn_sleep_plasticity_review_scheduler_installations(
        self,
        installations: Sequence[Mapping[str, Any]],
    ) -> None:
        self.load_snn_sleep_plasticity_review_scheduler_installations(installations)

    def load_snn_sleep_plasticity_review_scheduler_installations(
        self,
        installations: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [dict(item) for item in installations if isinstance(item, Mapping)]
        self._snn_sleep_plasticity_review_scheduler_installations.clear()
        self._snn_sleep_plasticity_review_scheduler_installations.extend(
            normalized[:DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_SCHEDULER_INSTALLATIONS]
        )

    def record_snn_replay_evaluation_context(
        self,
        *,
        mismatch_report: Mapping[str, Any],
        pressure_report: Mapping[str, Any],
        source_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record server-recomputed mismatch and pressure evidence for replay review."""

        mismatch = dict(mismatch_report)
        pressure = dict(pressure_report)
        metadata = dict(source_metadata or {})
        error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
        pressure_gate = (
            pressure.get("promotion_gate")
            if isinstance(pressure.get("promotion_gate"), Mapping)
            else {}
        )
        if (
            mismatch.get("surface") != "snn_language_sequence_mismatch_probe.v1"
            or not mismatch.get("available")
            or not mismatch.get("owned_by_marulho")
            or float(error.get("mismatch_score", 0.0) or 0.0) < 0.66
        ):
            raise ValueError("SNN replay evaluation context requires server-held high mismatch evidence.")
        if (
            pressure.get("surface") != "snn_language_plasticity_pressure.v1"
            or not pressure.get("available")
            or not pressure.get("owned_by_marulho")
            or str(pressure_gate.get("status") or "") != "ready_for_operator_review"
        ):
            raise ValueError("SNN replay evaluation context requires server-held plasticity pressure evidence.")
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            material = {
                "recorded_state_revision": recorded_revision,
                "mismatch_hash": self._sha256_json(mismatch),
                "pressure_hash": self._sha256_json(pressure),
                "source_metadata_hash": (
                    self._sha256_json(metadata) if metadata else None
                ),
            }
            evidence_hash = self._sha256_json(material)
            context = {
                "artifact_kind": "terminus_snn_replay_evaluation_context",
                "surface": "snn_replay_evaluation_context.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_replay_evaluation_context",
                "replay_evaluation_context_id": f"snn-replay-evaluation-{evidence_hash[:16]}-{uuid4()}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "source_metadata": deepcopy(metadata),
                "mismatch_report": mismatch,
                "pressure_report": pressure,
            }
            self._snn_replay_evaluation_contexts.appendleft(deepcopy(context))
            self._rebuild_snn_replay_evaluation_context_index_locked()
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(context)

    def verified_snn_replay_evaluation_context(self, context_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._verified_snn_replay_evaluation_context_payload_locked(
                self._snn_replay_evaluation_context_index.get(str(context_id).strip())
            )

    def snn_replay_consolidation_priority_queue(
        self,
        *,
        readout_replay_priority_report: Mapping[str, Any] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Rank verified SNN replay contexts for operator consolidation review."""

        started = time.perf_counter()
        report = dict(readout_replay_priority_report or {})
        readout_candidates = [
            dict(item)
            for item in list(report.get("candidates") or [])
            if isinstance(item, Mapping)
        ]
        readout_scores = [
            float(item.get("priority_score", 0.0) or 0.0)
            for item in readout_candidates
        ]
        grounded_readout_count = sum(
            1 for item in readout_candidates if bool(item.get("all_labels_grounded"))
        )
        readout_support = min(1.0, (max(readout_scores) if readout_scores else 0.0) / 100.0)
        grounded_support = (
            grounded_readout_count / max(1, len(readout_candidates))
            if readout_candidates
            else 0.0
        )
        readout_context_scores: dict[str, float] = {}
        readout_target_ids: list[str] = []
        for item, score in zip(readout_candidates, readout_scores):
            context_id = str(item.get("replay_evaluation_context_id") or "").strip()
            if not context_id:
                continue
            readout_context_scores[context_id] = max(
                float(score),
                float(readout_context_scores.get(context_id, 0.0)),
            )
            if (
                context_id not in readout_target_ids
                and len(readout_target_ids) < SNN_REPLAY_PRIORITY_READOUT_TARGET_LIMIT
            ):
                readout_target_ids.append(context_id)
        requested = max(0, min(int(limit), 32))
        with self._lock:
            context_retention_count = int(len(self._snn_replay_evaluation_contexts))
            recent_source_contexts = [
                item
                for item in islice(
                    self._snn_replay_evaluation_contexts,
                    SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
                )
                if isinstance(item, Mapping)
            ]
            source_slots: list[dict[str, Any]] = []
            seen_context_ids: set[str] = set()
            for source_rank, item in enumerate(recent_source_contexts):
                context_id = self._snn_replay_context_id(item)
                if not context_id or context_id in seen_context_ids:
                    continue
                seen_context_ids.add(context_id)
                source_slots.append(
                    {
                        "replay_evaluation_context_id": context_id,
                        "source": "recent_context_window",
                        "source_rank": int(source_rank),
                    }
                )
            readout_target_context_count = 0
            for context_id in readout_target_ids:
                if context_id in seen_context_ids:
                    continue
                if context_id not in self._snn_replay_evaluation_context_index:
                    continue
                seen_context_ids.add(context_id)
                readout_target_context_count += 1
                source_slots.append(
                    {
                        "replay_evaluation_context_id": context_id,
                        "source": "readout_priority_target_context_id",
                        "source_rank": SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT
                        + readout_target_context_count
                        - 1,
                    }
                )
            verified_contexts: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for slot in source_slots:
                context = self._verified_snn_replay_evaluation_context_payload_locked(
                    self._snn_replay_evaluation_context_index.get(
                        str(slot.get("replay_evaluation_context_id") or "")
                    )
                )
                if context is not None:
                    verified_contexts.append((slot, context))
            recent_denominator = max(1, min(context_retention_count, SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT) - 1)
            candidates: list[dict[str, Any]] = []
            for slot, context in verified_contexts:
                mismatch = (
                    context.get("mismatch_report")
                    if isinstance(context.get("mismatch_report"), Mapping)
                    else {}
                )
                pressure = (
                    context.get("pressure_report")
                    if isinstance(context.get("pressure_report"), Mapping)
                    else {}
                )
                error = (
                    mismatch.get("prediction_error")
                    if isinstance(mismatch.get("prediction_error"), Mapping)
                    else {}
                )
                pressure_payload = (
                    pressure.get("plasticity_pressure")
                    if isinstance(pressure.get("plasticity_pressure"), Mapping)
                    else {}
                )
                source_metadata = (
                    context.get("source_metadata")
                    if isinstance(context.get("source_metadata"), Mapping)
                    else {}
                )
                emission_lineage = self._snn_replay_context_emission_lineage(source_metadata)
                mismatch_score = max(0.0, min(1.0, float(error.get("mismatch_score", 0.0) or 0.0)))
                pressure_score = max(
                    0.0,
                    min(1.0, float(pressure_payload.get("pressure_score", mismatch_score) or 0.0)),
                )
                source_kind = str(slot.get("source") or "")
                source_rank = int(slot.get("source_rank", 0) or 0)
                recency = (
                    1.0 - min(1.0, source_rank / recent_denominator)
                    if source_kind == "recent_context_window"
                    else 0.0
                )
                context_id = str(context.get("replay_evaluation_context_id") or "")
                context_readout_support = min(
                    1.0,
                    float(readout_context_scores.get(context_id, 0.0)) / 100.0,
                )
                effective_readout_support = (
                    context_readout_support
                    if readout_context_scores
                    else readout_support
                )
                score = 100.0 * (
                    0.35 * mismatch_score
                    + 0.25 * pressure_score
                    + 0.20 * effective_readout_support
                    + 0.20 * recency
                )
                candidates.append(
                    {
                        "candidate_id": (
                            "snn-replay-consolidation-queue:"
                            f"{str(context.get('replay_evaluation_context_id') or '')[:32]}"
                        ),
                        "replay_evaluation_context_id": context.get("replay_evaluation_context_id"),
                        "replay_evaluation_context_hash": context.get("evidence_hash"),
                        "recorded_state_revision": int(context.get("recorded_state_revision", -1)),
                        "mismatch_hash": context.get("mismatch_hash"),
                        "pressure_hash": context.get("pressure_hash"),
                        "source_metadata_hash": context.get("source_metadata_hash"),
                        "emission_lineage": emission_lineage,
                        "emission_lineage_available": bool(emission_lineage),
                        "priority_score": float(score),
                        "priority_components": {
                            "prediction_error": float(mismatch_score),
                            "plasticity_pressure": float(pressure_score),
                            "readout_support": float(effective_readout_support),
                            "recency": float(recency),
                        },
                        "source_window": {
                            "source": source_kind,
                            "source_rank": source_rank,
                            "policy": SNN_REPLAY_PRIORITY_SOURCE_WINDOW_POLICY,
                        },
                        "reason_codes": [
                            code
                            for code, active in (
                                ("high_prediction_error", mismatch_score >= 0.66),
                                ("high_plasticity_pressure", pressure_score >= 0.66),
                                (
                                    "grounded_readout_candidates_available",
                                    grounded_support > 0.0,
                                ),
                                (
                                    "readout_target_context_id",
                                    context_readout_support > 0.0,
                                ),
                                ("recent_context", recency >= 0.5),
                            )
                            if active
                        ],
                        "suggested_review_action": (
                            "operator_review_snn_transition_memory_replay_artifact_proposal"
                        ),
                        "advisory": True,
                        "executable": False,
                        "generates_text": False,
                        "decodes_text": False,
                        "trains_runtime_model": False,
                        "applies_plasticity": False,
                        "mutates_runtime_state": False,
                        "eligible_for_action": False,
                        "eligible_for_fact_promotion": False,
                        "eligible_for_live_replay": False,
                        "eligible_for_artifact_recording": False,
                        "eligible_for_structural_write": False,
                    }
                )
            candidates.sort(
                key=lambda item: (
                    -float(item["priority_score"]),
                    str(item.get("replay_evaluation_context_id") or ""),
                )
            )
            selected = [
                {**candidate, "rank": rank}
                for rank, candidate in enumerate(candidates[:requested], start=1)
            ] if requested > 0 else []
            ready = bool(selected) and grounded_support > 0.0
            latency_ms = (time.perf_counter() - started) * 1000.0
            source_window = {
                "surface": "bounded_snn_replay_priority_source_window.v1",
                "policy": SNN_REPLAY_PRIORITY_SOURCE_WINDOW_POLICY,
                "status": "collected" if source_slots else "empty",
                "context_retention_count": context_retention_count,
                "context_retention_limit": DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
                "recent_context_window_limit": SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
                "recent_context_window_count": int(len(recent_source_contexts)),
                "readout_target_window_limit": SNN_REPLAY_PRIORITY_READOUT_TARGET_LIMIT,
                "readout_target_request_count": int(len(readout_target_ids)),
                "readout_target_context_count": int(readout_target_context_count),
                "source_context_count": int(len(source_slots)),
                "verified_context_count": int(len(verified_contexts)),
                "source_context_count_is_lower_bound": bool(
                    context_retention_count > len(recent_source_contexts)
                ),
                "truncated_context_count": int(
                    max(0, context_retention_count - len(recent_source_contexts))
                ),
                "candidate_window_policy": SNN_REPLAY_PRIORITY_SOURCE_WINDOW_POLICY,
                "candidate_scope": "recent_context_index_plus_readout_target_ids",
                "context_lookup": "replay_evaluation_context_id_index",
                "score_device": "cpu",
                "archival_storage_device": "cpu",
                "gpu_used": False,
                "global_candidate_scan": False,
                "global_score_scan": False,
                "runs_live_tick": False,
                "runs_live_replay": False,
                "records_replay_artifact": False,
                "raw_text_payload_loaded": False,
                "language_reasoning": False,
                "mutates_runtime_state": False,
                "applies_plasticity": False,
                "latency_ms": float(latency_ms),
                "selection_budget": {
                    "context_retention_entries": DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
                    "recent_context_window_entries": SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
                    "readout_target_window_entries": SNN_REPLAY_PRIORITY_READOUT_TARGET_LIMIT,
                    "selected_budget_entries": int(requested),
                },
            }
            return {
                "artifact_kind": "terminus_snn_replay_consolidation_priority_queue",
                "surface": "snn_replay_consolidation_priority_queue.v1",
                "available": bool(selected),
                "ready": ready,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_replay_consolidation_priority_queue",
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "advisory": True,
                "executable": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_action": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_structural_write": False,
                "candidate_count": len(selected),
                "source_window": source_window,
                "context_source_window_policy": SNN_REPLAY_PRIORITY_SOURCE_WINDOW_POLICY,
                "context_source_window_limit": SNN_REPLAY_PRIORITY_CONTEXT_WINDOW_LIMIT,
                "context_source_window_count": int(len(source_slots)),
                "verified_context_count": int(len(verified_contexts)),
                "context_retention_count": context_retention_count,
                "context_retention_limit": DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS,
                "global_candidate_scan": False,
                "global_score_scan": False,
                "runs_live_tick": False,
                "runs_live_replay": False,
                "score_device": "cpu",
                "archival_storage_device": "cpu",
                "gpu_used": False,
                "latency_ms": float(latency_ms),
                "priority_rules_version": "snn-replay-consolidation-deterministic-v1",
                "priority_weights": {
                    "prediction_error": 0.35,
                    "plasticity_pressure": 0.25,
                    "readout_support": 0.20,
                    "recency": 0.20,
                },
                "readout_priority_summary": {
                    "surface": report.get("surface"),
                    "candidate_count": len(readout_candidates),
                    "grounded_candidate_count": grounded_readout_count,
                    "max_priority_score": max(readout_scores) if readout_scores else 0.0,
                },
                "candidates": selected,
                "promotion_gate": {
                    "status": "ready_for_operator_consolidation_review"
                    if ready
                    else "collect_replay_context_and_grounded_readout_evidence",
                    "eligible_for_operator_consolidation_review": ready,
                    "eligible_for_artifact_recording": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_structural_write": False,
                    "eligible_for_action": False,
                    "eligible_for_fact_promotion": False,
                    "requires_operator_approval": ready,
                    "next_gate": (
                        "operator_review_snn_transition_memory_replay_artifact_proposal"
                        if ready
                        else "record_replay_context_and_grounded_readout_evidence"
                    ),
                    "required_evidence": {
                        "server_held_replay_context_available": bool(selected),
                        "current_revision_contexts_verified": (
                            len(verified_contexts) == len(source_slots)
                            if source_slots
                            else False
                        ),
                        "bounded_source_window": True,
                        "global_candidate_scan_absent": True,
                        "raw_text_reasoning_absent": True,
                        "grounded_readout_candidates_available": grounded_support > 0.0,
                        "runtime_mutation_absent": True,
                        "artifact_recording_absent": True,
                    },
                },
            }

    def snn_replay_artifact_recording_policy_proposal(
        self,
        *,
        consolidation_priority_queue: Mapping[str, Any],
        policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propose the next replay-artifact recording review without recording it."""

        queue = dict(consolidation_priority_queue)
        gate = queue.get("promotion_gate") if isinstance(queue.get("promotion_gate"), Mapping) else {}
        policy_payload = dict(policy or {})
        min_priority_score = max(
            0.0,
            min(float(policy_payload.get("min_priority_score", 66.0) or 0.0), 100.0),
        )
        max_candidates = max(1, min(int(policy_payload.get("max_candidates", 1) or 1), 8))
        candidates = [
            dict(item)
            for item in list(queue.get("candidates") or [])
            if isinstance(item, Mapping)
        ][:max_candidates]
        selected = [
            item
            for item in candidates
            if float(item.get("priority_score", 0.0) or 0.0) >= min_priority_score
        ]
        top = selected[0] if selected else {}
        required = {
            "priority_queue_surface_available": queue.get("surface")
            == "snn_replay_consolidation_priority_queue.v1",
            "priority_queue_owned_by_marulho": bool(queue.get("owned_by_marulho")),
            "priority_queue_non_executable": not bool(queue.get("executable")),
            "priority_queue_non_mutating": not bool(queue.get("mutates_runtime_state")),
            "priority_queue_gate_ready": bool(gate.get("eligible_for_operator_consolidation_review")),
            "candidate_available": bool(selected),
            "candidate_above_policy_threshold": bool(selected),
            "candidate_not_action": not bool(top.get("eligible_for_action")) if top else False,
            "candidate_not_fact_promotion": not bool(top.get("eligible_for_fact_promotion")) if top else False,
            "candidate_not_live_replay": not bool(top.get("eligible_for_live_replay")) if top else False,
            "candidate_not_artifact_recording": not bool(top.get("eligible_for_artifact_recording")) if top else False,
            "candidate_not_structural_write": not bool(top.get("eligible_for_structural_write")) if top else False,
        }
        ready = all(required.values())
        recommended_context_id = str(top.get("replay_evaluation_context_id") or "") if top else ""
        recommended_context = (
            self.verified_snn_replay_evaluation_context(recommended_context_id)
            if recommended_context_id
            else None
        )
        if ready and recommended_context is None:
            required["candidate_context_verified_current_revision"] = False
            ready = False
        else:
            required["candidate_context_verified_current_revision"] = bool(recommended_context)
        recommended_source_metadata_hash = top.get("source_metadata_hash") if top else None
        recommended_emission_lineage = (
            dict(top.get("emission_lineage"))
            if top and isinstance(top.get("emission_lineage"), Mapping)
            else {}
        )
        context_lineage = (
            self._snn_replay_context_emission_lineage(
                recommended_context.get("source_metadata")
                if isinstance(recommended_context, Mapping)
                and isinstance(recommended_context.get("source_metadata"), Mapping)
                else {}
            )
            if recommended_context
            else {}
        )
        required["candidate_lineage_matches_verified_context"] = (
            bool(recommended_context)
            and recommended_source_metadata_hash
            == recommended_context.get("source_metadata_hash")
            and recommended_emission_lineage == context_lineage
        )
        ready = ready and bool(required["candidate_lineage_matches_verified_context"])
        return {
            "artifact_kind": "terminus_snn_replay_artifact_recording_policy_proposal",
            "surface": "snn_replay_artifact_recording_policy_proposal.v1",
            "available": bool(candidates),
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_replay_artifact_recording_policy_proposal",
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "advisory": True,
            "executable": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_write": False,
            "policy": {
                "policy_version": "snn-replay-artifact-recording-policy-v1",
                "min_priority_score": float(min_priority_score),
                "max_candidates": int(max_candidates),
                "requires_operator_review": True,
            },
            "recommended": ready,
            "recommended_review": {
                "review_action": "operator_review_snn_transition_memory_replay_artifact_recording",
                "replay_evaluation_context_id": recommended_context_id or None,
                "replay_evaluation_context_hash": top.get("replay_evaluation_context_hash") if top else None,
                "source_metadata_hash": recommended_source_metadata_hash,
                "emission_lineage": recommended_emission_lineage,
                "priority_score": float(top.get("priority_score", 0.0) or 0.0) if top else 0.0,
                "reason_codes": [str(value) for value in list(top.get("reason_codes") or [])],
            },
            "candidate_count": len(selected),
            "promotion_gate": {
                "status": "ready_for_operator_artifact_recording_review"
                if ready
                else "collect_policy_ready_replay_consolidation_candidate",
                "eligible_for_operator_artifact_recording_review": ready,
                "eligible_for_artifact_recording": False,
                "eligible_for_live_replay": False,
                "eligible_for_structural_write": False,
                "eligible_for_action": False,
                "eligible_for_fact_promotion": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_transition_memory_replay_artifact_recording"
                if ready
                else "collect_replay_consolidation_priority_evidence",
                "required_evidence": required,
            },
        }

    def record_snn_replay_artifact_recording_review_ticket(
        self,
        *,
        policy_proposal: Mapping[str, Any],
        operator_id: str,
        confirmation: bool,
        due_cycle_review_proposal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record server-held policy intent for later artifact-recording review."""

        proposal = dict(policy_proposal)
        review = (
            proposal.get("recommended_review")
            if isinstance(proposal.get("recommended_review"), Mapping)
            else {}
        )
        gate = proposal.get("promotion_gate") if isinstance(proposal.get("promotion_gate"), Mapping) else {}
        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        context_id = str(review.get("replay_evaluation_context_id") or "")
        context = self.verified_snn_replay_evaluation_context(context_id)
        due_cycle_proposal = dict(due_cycle_review_proposal or {})
        due_cycle_target = (
            due_cycle_proposal.get("review_target")
            if isinstance(due_cycle_proposal.get("review_target"), Mapping)
            else {}
        )
        due_cycle_provenance = (
            due_cycle_proposal.get("provenance_evidence")
            if isinstance(due_cycle_proposal.get("provenance_evidence"), Mapping)
            else {}
        )
        if not confirmation:
            raise ValueError("SNN replay artifact recording review ticket confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN replay artifact recording review ticket operator_id is required.")
        if proposal.get("surface") != "snn_replay_artifact_recording_policy_proposal.v1":
            raise ValueError("SNN replay artifact recording review ticket requires policy proposal surface.")
        if not proposal.get("owned_by_marulho") or not proposal.get("ready") or not proposal.get("recommended"):
            raise ValueError("SNN replay artifact recording review ticket requires ready MARULHO policy proposal.")
        if not bool(gate.get("eligible_for_operator_artifact_recording_review")):
            raise ValueError("SNN replay artifact recording review ticket requires operator review gate.")
        if context is None or str(review.get("replay_evaluation_context_hash") or "") != str(
            context.get("evidence_hash") or ""
        ):
            raise ValueError("SNN replay artifact recording review ticket requires a verified replay context.")
        context_lineage = self._snn_replay_context_emission_lineage(
            context.get("source_metadata")
            if isinstance(context.get("source_metadata"), Mapping)
            else {}
        )
        review_lineage = (
            dict(review.get("emission_lineage"))
            if isinstance(review.get("emission_lineage"), Mapping)
            else {}
        )
        if str(review.get("source_metadata_hash") or "") != str(context.get("source_metadata_hash") or ""):
            raise ValueError(
                "SNN replay artifact recording review ticket requires replay context source lineage."
            )
        if review_lineage != context_lineage:
            raise ValueError(
                "SNN replay artifact recording review ticket requires replay context source lineage."
            )
        if due_cycle_review_proposal is not None and (
            due_cycle_proposal.get("surface")
            != "snn_due_cycle_replay_artifact_recording_review_proposal.v1"
            or not due_cycle_proposal.get("ready")
            or not due_cycle_proposal.get("owned_by_marulho")
            or due_cycle_proposal.get("executable") is not False
            or due_cycle_proposal.get("records_replay_artifact") is not False
            or due_cycle_proposal.get("runs_live_replay") is not False
            or due_cycle_proposal.get("applies_plasticity") is not False
            or due_cycle_proposal.get("mutates_runtime_state") is not False
            or str(due_cycle_target.get("replay_evaluation_context_id") or "")
            != context_id
            or str(due_cycle_target.get("replay_evaluation_context_hash") or "")
            != str(context.get("evidence_hash") or "")
            or not due_cycle_provenance.get(
                "due_cycle_replay_artifact_recording_review_proposal_hash"
            )
            or not due_cycle_target.get("scheduler_installation_id")
            or not due_cycle_target.get("scheduler_installation_evidence_hash")
            or not due_cycle_target.get("acknowledged_review_due_at")
        ):
            raise ValueError(
                "SNN replay artifact recording review ticket requires a ready due-cycle review proposal."
            )
        source_window = self._snn_replay_provenance_source_window(
            replay_evaluation_context_id=context_id,
        )
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            material = {
                "recorded_state_revision": recorded_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "policy_proposal_hash": self._sha256_json(proposal),
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "source_metadata_hash": context.get("source_metadata_hash"),
                "emission_lineage": context_lineage,
                "source_window_hash": self._sha256_json(source_window),
            }
            if due_cycle_review_proposal is not None:
                material["due_cycle_review_proposal_hash"] = due_cycle_provenance[
                    "due_cycle_replay_artifact_recording_review_proposal_hash"
                ]
                material["due_cycle_selection_proposal_hash"] = due_cycle_provenance[
                    "source_due_cycle_bounded_replay_selection_proposal_hash"
                ]
                material["scheduler_installation_id"] = due_cycle_target[
                    "scheduler_installation_id"
                ]
                material["scheduler_installation_evidence_hash"] = due_cycle_target[
                    "scheduler_installation_evidence_hash"
                ]
                material["acknowledged_review_due_at"] = due_cycle_target[
                    "acknowledged_review_due_at"
                ]
            evidence_hash = self._sha256_json(material)
            ticket = {
                "artifact_kind": "terminus_snn_replay_artifact_recording_review_ticket",
                "surface": "snn_replay_artifact_recording_review_ticket.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_replay_artifact_recording_review_ticket",
                "review_ticket_id": f"snn-replay-artifact-review-{evidence_hash[:16]}-{uuid4()}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "source_window": source_window,
                "policy_surface": proposal.get("surface"),
                "review_action": review.get("review_action"),
            }
            self._snn_replay_artifact_recording_review_tickets.appendleft(deepcopy(ticket))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(ticket)

    def verified_snn_replay_artifact_recording_review_ticket(
        self,
        review_ticket_id: str,
        *,
        replay_evaluation_context_id: str | None = None,
        operator_id: str | None = None,
    ) -> dict[str, Any] | None:
        expected_operator_id = (
            self._normalize_feedback_text(operator_id, max_chars=160)
            if operator_id is not None
            else None
        )
        with self._lock:
            ticket = self._snn_replay_artifact_recording_review_ticket_index.get(
                str(review_ticket_id or "").strip()
            )
            if ticket is None:
                return None
            ticket = dict(ticket)
            material = {
                "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                "operator_id": ticket.get("operator_id"),
                "confirmation": bool(ticket.get("confirmation")),
                "policy_proposal_hash": ticket.get("policy_proposal_hash"),
                "replay_evaluation_context_id": ticket.get("replay_evaluation_context_id"),
                "replay_evaluation_context_hash": ticket.get("replay_evaluation_context_hash"),
                "source_metadata_hash": ticket.get("source_metadata_hash"),
                "emission_lineage": (
                    dict(ticket.get("emission_lineage"))
                    if isinstance(ticket.get("emission_lineage"), Mapping)
                    else {}
                ),
            }
            if ticket.get("source_window_hash"):
                material["source_window_hash"] = ticket.get("source_window_hash")
            if ticket.get("due_cycle_review_proposal_hash"):
                material["due_cycle_review_proposal_hash"] = ticket.get(
                    "due_cycle_review_proposal_hash"
                )
                material["due_cycle_selection_proposal_hash"] = ticket.get(
                    "due_cycle_selection_proposal_hash"
                )
                material["scheduler_installation_id"] = ticket.get(
                    "scheduler_installation_id"
                )
                material["scheduler_installation_evidence_hash"] = ticket.get(
                    "scheduler_installation_evidence_hash"
                )
                material["acknowledged_review_due_at"] = ticket.get(
                    "acknowledged_review_due_at"
                )
            context = self.verified_snn_replay_evaluation_context(
                str(ticket.get("replay_evaluation_context_id") or "")
            )
            expected_context_id = str(replay_evaluation_context_id or ticket.get("replay_evaluation_context_id") or "")
            return (
                deepcopy(ticket)
                if (
                    ticket.get("ready")
                    and ticket.get("owned_by_marulho")
                    and ticket.get("confirmation") is True
                    and (
                        expected_operator_id is None
                        or str(ticket.get("operator_id") or "") == expected_operator_id
                    )
                    and int(ticket.get("recorded_state_revision", -1)) == int(self._runtime_state.state_revision)
                    and str(ticket.get("replay_evaluation_context_id") or "") == expected_context_id
                    and context is not None
                    and str(ticket.get("replay_evaluation_context_hash") or "")
                    == str(context.get("evidence_hash") or "")
                    and str(ticket.get("evidence_hash") or "") == self._sha256_json(material)
                )
                else None
            )

    def record_snn_sleep_plasticity_review_ticket(
        self,
        *,
        sleep_policy: Mapping[str, Any],
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Record operator intent to review a sleep-policy recommendation."""

        policy = dict(sleep_policy)
        recommendation = (
            policy.get("recommendation")
            if isinstance(policy.get("recommendation"), Mapping)
            else {}
        )
        transition_memory = (
            policy.get("transition_memory")
            if isinstance(policy.get("transition_memory"), Mapping)
            else {}
        )
        replay_evidence = (
            policy.get("replay_evidence")
            if isinstance(policy.get("replay_evidence"), Mapping)
            else {}
        )
        rollout_evidence = (
            policy.get("rollout_regeneration_evidence")
            if isinstance(policy.get("rollout_regeneration_evidence"), Mapping)
            else {}
        )
        readout_evidence = (
            policy.get("readout_ledger_evidence")
            if isinstance(policy.get("readout_ledger_evidence"), Mapping)
            else {}
        )
        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        action = str(recommendation.get("action") or "")
        suggested_endpoint = str(recommendation.get("suggested_endpoint") or "")
        gate = SNN_SLEEP_PLASTICITY_REVIEW_GATES.get(action)
        if not confirmation:
            raise ValueError("SNN sleep plasticity review ticket confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN sleep plasticity review ticket operator_id is required.")
        if policy.get("surface") != "snn_language_transition_memory_sleep_policy.v1":
            raise ValueError("SNN sleep plasticity review ticket requires sleep policy surface.")
        if not policy.get("owned_by_marulho") or policy.get("mutates_runtime_state"):
            raise ValueError("SNN sleep plasticity review ticket requires non-mutating MARULHO policy.")
        if (
            not bool(recommendation.get("recommended"))
            or bool(recommendation.get("executable"))
            or gate is None
            or suggested_endpoint != gate[1]
        ):
            raise ValueError("SNN sleep plasticity review ticket requires a non-executable recommendation.")
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            material = {
                "recorded_state_revision": recorded_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "sleep_policy_hash": self._sha256_json(policy),
                "recommended_action": action,
                "review_gate_key": gate[0],
                "suggested_endpoint": gate[1],
                "reason_codes": [
                    str(value)
                    for value in list(recommendation.get("reason_codes") or [])
                    if str(value)
                ][:32],
                "transition_memory_hash": self._sha256_json(transition_memory),
                "replay_evidence_hash": self._sha256_json(replay_evidence),
                "rollout_regeneration_evidence_hash": self._sha256_json(rollout_evidence),
                "readout_ledger_evidence_hash": self._sha256_json(readout_evidence),
            }
            evidence_hash = self._sha256_json(material)
            ticket = {
                "artifact_kind": "terminus_snn_sleep_plasticity_review_ticket",
                "surface": "snn_sleep_plasticity_review_ticket.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_review_ticket",
                "review_ticket_id": f"snn-sleep-plasticity-review-{evidence_hash[:16]}-{uuid4()}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "sleep_policy_surface": policy.get("surface"),
                "requires_operator_confirmation": True,
                "executable": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "writes_checkpoint": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
            }
            self._snn_sleep_plasticity_review_tickets.appendleft(deepcopy(ticket))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(ticket)

    def verified_snn_sleep_plasticity_review_ticket(
        self,
        review_ticket_id: str,
        *,
        operator_id: str | None = None,
    ) -> dict[str, Any] | None:
        expected_operator_id = (
            self._normalize_feedback_text(operator_id, max_chars=160)
            if operator_id is not None
            else None
        )
        with self._lock:
            ticket = self._snn_sleep_plasticity_review_ticket_index.get(
                str(review_ticket_id or "").strip()
            )
            if ticket is None:
                return None
            ticket = dict(ticket)
            material = {
                "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                "operator_id": ticket.get("operator_id"),
                "confirmation": bool(ticket.get("confirmation")),
                "sleep_policy_hash": ticket.get("sleep_policy_hash"),
                "recommended_action": ticket.get("recommended_action"),
                "review_gate_key": ticket.get("review_gate_key"),
                "suggested_endpoint": ticket.get("suggested_endpoint"),
                "reason_codes": list(ticket.get("reason_codes") or []),
                "transition_memory_hash": ticket.get("transition_memory_hash"),
                "replay_evidence_hash": ticket.get("replay_evidence_hash"),
                "rollout_regeneration_evidence_hash": ticket.get("rollout_regeneration_evidence_hash"),
                "readout_ledger_evidence_hash": ticket.get("readout_ledger_evidence_hash"),
            }
            return (
                deepcopy(ticket)
                if (
                    ticket.get("artifact_kind") == "terminus_snn_sleep_plasticity_review_ticket"
                    and ticket.get("surface") == "snn_sleep_plasticity_review_ticket.v1"
                    and ticket.get("source") == "replay_controller.snn_sleep_plasticity_review_ticket"
                    and ticket.get("ready")
                    and ticket.get("owned_by_marulho")
                    and ticket.get("confirmation") is True
                    and (
                        expected_operator_id is None
                        or str(ticket.get("operator_id") or "") == expected_operator_id
                    )
                    and int(ticket.get("recorded_state_revision", -1)) == int(self._runtime_state.state_revision)
                    and str(ticket.get("evidence_hash") or "") == self._sha256_json(material)
                    and SNN_SLEEP_PLASTICITY_REVIEW_GATES.get(
                        str(ticket.get("recommended_action") or "")
                    )
                    == (
                        str(ticket.get("review_gate_key") or ""),
                        str(ticket.get("suggested_endpoint") or ""),
                    )
                    and not bool(ticket.get("executable"))
                    and not bool(ticket.get("mutates_runtime_state"))
                    and not bool(ticket.get("applies_plasticity"))
                    and not bool(ticket.get("writes_checkpoint"))
                    and not bool(ticket.get("records_replay_artifact"))
                    and not bool(ticket.get("issues_regeneration_permit"))
                )
                else None
            )

    def snn_sleep_plasticity_review_ticket_queue(self, *, limit: int = 20) -> dict[str, Any]:
        """Expose durable sleep-policy review tickets without executing them."""

        with self._lock:
            started = time.perf_counter()
            requested = max(1, min(DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS, int(limit)))
            source_limit = min(
                requested,
                SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
            )
            retained_count = int(len(self._snn_sleep_plasticity_review_tickets))
            source_window_inspected_count = int(min(retained_count, source_limit))
            current_revision = int(self._runtime_state.state_revision)
            tickets: list[dict[str, Any]] = []
            action_counts: Counter[str] = Counter()
            verified_count = 0
            stale_count = 0
            tampered_count = 0
            for source_rank, raw_ticket in enumerate(
                islice(self._snn_sleep_plasticity_review_tickets, source_limit)
            ):
                if not isinstance(raw_ticket, Mapping):
                    continue
                ticket = dict(raw_ticket)
                material = {
                    "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                    "operator_id": ticket.get("operator_id"),
                    "confirmation": bool(ticket.get("confirmation")),
                    "sleep_policy_hash": ticket.get("sleep_policy_hash"),
                    "recommended_action": ticket.get("recommended_action"),
                    "review_gate_key": ticket.get("review_gate_key"),
                    "suggested_endpoint": ticket.get("suggested_endpoint"),
                    "reason_codes": list(ticket.get("reason_codes") or []),
                    "transition_memory_hash": ticket.get("transition_memory_hash"),
                    "replay_evidence_hash": ticket.get("replay_evidence_hash"),
                    "rollout_regeneration_evidence_hash": ticket.get("rollout_regeneration_evidence_hash"),
                    "readout_ledger_evidence_hash": ticket.get("readout_ledger_evidence_hash"),
                }
                hash_verified = str(ticket.get("evidence_hash") or "") == self._sha256_json(material)
                revision_current = int(ticket.get("recorded_state_revision", -1)) == current_revision
                identity_verified = bool(
                    ticket.get("artifact_kind") == "terminus_snn_sleep_plasticity_review_ticket"
                    and ticket.get("surface") == "snn_sleep_plasticity_review_ticket.v1"
                    and ticket.get("source") == "replay_controller.snn_sleep_plasticity_review_ticket"
                    and SNN_SLEEP_PLASTICITY_REVIEW_GATES.get(
                        str(ticket.get("recommended_action") or "")
                    )
                    == (
                        str(ticket.get("review_gate_key") or ""),
                        str(ticket.get("suggested_endpoint") or ""),
                    )
                )
                non_executing = (
                    not bool(ticket.get("executable"))
                    and not bool(ticket.get("mutates_runtime_state"))
                    and not bool(ticket.get("applies_plasticity"))
                    and not bool(ticket.get("writes_checkpoint"))
                    and not bool(ticket.get("records_replay_artifact"))
                    and not bool(ticket.get("issues_regeneration_permit"))
                )
                verified = bool(
                    ticket.get("ready")
                    and ticket.get("owned_by_marulho")
                    and ticket.get("confirmation") is True
                    and hash_verified
                    and revision_current
                    and identity_verified
                    and non_executing
                )
                if verified:
                    verified_count += 1
                    action_counts[str(ticket.get("recommended_action") or "unknown")] += 1
                elif not revision_current:
                    stale_count += 1
                elif not hash_verified or not identity_verified or not non_executing:
                    tampered_count += 1
                tickets.append(
                    {
                        "review_ticket_id": ticket.get("review_ticket_id"),
                        "surface": ticket.get("surface"),
                        "recorded_at": ticket.get("recorded_at"),
                        "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                        "operator_id": ticket.get("operator_id"),
                        "recommended_action": ticket.get("recommended_action"),
                        "review_gate_key": ticket.get("review_gate_key"),
                        "suggested_endpoint": ticket.get("suggested_endpoint"),
                        "reason_codes": list(ticket.get("reason_codes") or []),
                        "evidence_hash": ticket.get("evidence_hash"),
                        "sleep_policy_hash": ticket.get("sleep_policy_hash"),
                        "verified": verified,
                        "hash_verified": hash_verified,
                        "identity_verified": identity_verified,
                        "revision_current": revision_current,
                        "non_executing": non_executing,
                        "source_rank": int(source_rank),
                        "executable": False,
                        "mutates_runtime_state": False,
                        "applies_plasticity": False,
                    }
                )
            selected = tickets[:requested]
            latest_verified = next((deepcopy(item) for item in selected if item["verified"]), None)
            next_gate = (
                latest_verified.get("suggested_endpoint")
                if isinstance(latest_verified, Mapping)
                else None
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            source_window = {
                "surface": "bounded_snn_sleep_plasticity_review_ticket_queue_source_window.v1",
                "policy": "recent_sleep_plasticity_review_ticket_window",
                "source": "replay_controller.snn_sleep_plasticity_review_tickets",
                "selection_criteria": [
                    "newest_review_tickets_first",
                    "current_revision_hash_verified_non_executing_tickets",
                    "stop_at_source_window_before_scheduler_proposal",
                ],
                "retained_count": retained_count,
                "retention_limit": DEFAULT_SNN_SLEEP_PLASTICITY_REVIEW_TICKETS,
                "requested_limit": int(requested),
                "source_window_limit": SNN_SLEEP_PLASTICITY_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
                "source_window_inspected_count": source_window_inspected_count,
                "source_window_count": int(len(tickets)),
                "source_truncated_count": int(
                    max(0, retained_count - source_window_inspected_count)
                ),
                "count_is_source_window": True,
                "latest_verified_scope": "source_window_only",
                "global_candidate_scan": False,
                "global_score_scan": False,
                "raw_replay_text_payload_loaded": False,
                "language_reasoning": False,
                "runs_live_tick": False,
                "runs_every_token": False,
                "mutates_runtime_state": False,
                "applies_plasticity": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "archival_storage_device": "cpu",
                "source_window_selection_device": "cpu",
                "score_device": "cpu",
                "gpu_used": False,
                "latency_ms": float(latency_ms),
            }
            return {
                "artifact_kind": "terminus_snn_sleep_plasticity_review_ticket_queue",
                "surface": "snn_sleep_plasticity_review_ticket_queue.v1",
                "available": True,
                "ready": verified_count > 0,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_review_ticket_queue",
                "endpoint": "/terminus/snn-language-sequence/plasticity-sleep-policy/review-tickets",
                "count": int(len(tickets)),
                "retained_count": retained_count,
                "limit": int(requested),
                "source_window": source_window,
                "current_state_revision": current_revision,
                "verified_count": int(verified_count),
                "stale_count": int(stale_count),
                "tampered_count": int(tampered_count),
                "pending_action_counts": dict(action_counts),
                "latest_verified_ticket": latest_verified,
                "next_gate": next_gate,
                "advisory": True,
                "executable": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "tickets": selected,
            }

    def snn_sleep_plasticity_autonomy_proposal(self, *, limit: int = 20) -> dict[str, Any]:
        """Build a non-executing autonomy candidate from verified sleep-policy tickets."""

        queue = self.snn_sleep_plasticity_review_ticket_queue(limit=limit)
        latest = (
            queue.get("latest_verified_ticket")
            if isinstance(queue.get("latest_verified_ticket"), Mapping)
            else None
        )
        ready = latest is not None and bool(queue.get("ready"))
        recommended_action = str(latest.get("recommended_action") or "") if latest else ""
        suggested_endpoint = str(latest.get("suggested_endpoint") or "") if latest else ""
        reason_codes = [
            str(value)
            for value in list(latest.get("reason_codes") or [])
            if str(value)
        ] if latest else []
        required_evidence = {
            "verified_sleep_plasticity_review_ticket": ready,
            "ticket_hash_verified": bool(latest.get("hash_verified")) if latest else False,
            "ticket_revision_current": bool(latest.get("revision_current")) if latest else False,
            "ticket_non_executing": bool(latest.get("non_executing")) if latest else False,
            "suggested_endpoint_present": bool(suggested_endpoint),
        }
        action = (
            "review_sleep_plasticity_next_gate"
            if ready
            else "collect_sleep_plasticity_review_ticket"
        )
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_autonomy_proposal",
            "surface": "snn_sleep_plasticity_autonomy_proposal.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_autonomy_proposal",
            "advisory": True,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_structural_write": False,
            "candidate": {
                "candidate_id": (
                    "snn-sleep-plasticity-autonomy:"
                    f"{str(latest.get('review_ticket_id') or '')[:48]}"
                ) if latest else None,
                "action": action,
                "review_ticket_id": latest.get("review_ticket_id") if latest else None,
                "review_ticket_hash": latest.get("evidence_hash") if latest else None,
                "recommended_action": recommended_action or None,
                "suggested_endpoint": suggested_endpoint or None,
                "reason_codes": reason_codes,
                "state_revision": int(queue.get("current_state_revision", -1)),
                "priority_score": 1.0 if ready else 0.0,
            },
            "promotion_gate": {
                "status": "ready_for_operator_next_gate_review" if ready else "collect_sleep_plasticity_review_ticket",
                "eligible_for_autonomy_planning": ready,
                "eligible_for_action": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity": False,
                "next_gate": suggested_endpoint or "/terminus/snn-language-sequence/plasticity-sleep-policy",
                "required_evidence": required_evidence,
            },
            "ticket_queue": {
                "surface": queue.get("surface"),
                "count": int(queue.get("count", 0) or 0),
                "verified_count": int(queue.get("verified_count", 0) or 0),
                "stale_count": int(queue.get("stale_count", 0) or 0),
                "tampered_count": int(queue.get("tampered_count", 0) or 0),
                "pending_action_counts": dict(queue.get("pending_action_counts") or {}),
            },
        }

    def snn_sleep_plasticity_scheduler_experiment(
        self,
        *,
        limit: int = 20,
        cycles: int = 4,
    ) -> dict[str, Any]:
        """Measure proposal stability before any sleep/plasticity scheduler exists."""

        bounded_cycles = max(1, min(16, int(cycles)))
        observations: list[dict[str, Any]] = []
        baseline_hash = ""
        for cycle in range(1, bounded_cycles + 1):
            proposal = self.snn_sleep_plasticity_autonomy_proposal(limit=limit)
            candidate = (
                proposal.get("candidate")
                if isinstance(proposal.get("candidate"), Mapping)
                else {}
            )
            gate = (
                proposal.get("promotion_gate")
                if isinstance(proposal.get("promotion_gate"), Mapping)
                else {}
            )
            material = {
                "ready": bool(proposal.get("ready")),
                "review_ticket_id": candidate.get("review_ticket_id"),
                "review_ticket_hash": candidate.get("review_ticket_hash"),
                "recommended_action": candidate.get("recommended_action"),
                "suggested_endpoint": candidate.get("suggested_endpoint"),
                "state_revision": int(candidate.get("state_revision", -1)),
                "promotion_status": gate.get("status"),
                "next_gate": gate.get("next_gate"),
                "eligible_for_autonomy_planning": bool(gate.get("eligible_for_autonomy_planning")),
            }
            observation_hash = self._sha256_json(material)
            if cycle == 1:
                baseline_hash = observation_hash
            observations.append(
                {
                    "cycle": cycle,
                    "observation_hash": observation_hash,
                    "stable_against_first_cycle": observation_hash == baseline_hash,
                    **material,
                }
            )
        latest = observations[-1]
        ready = bool(
            latest["ready"]
            and latest["eligible_for_autonomy_planning"]
            and latest["review_ticket_id"]
            and all(item["stable_against_first_cycle"] for item in observations)
        )
        provenance_material = {
            "cycle_count": bounded_cycles,
            "observations": observations,
            "ready": ready,
        }
        scheduler_experiment_hash = self._sha256_json(provenance_material)
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_experiment",
            "surface": "snn_sleep_plasticity_scheduler_experiment.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_scheduler_experiment",
            "advisory": True,
            "isolated": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "returns_trained_weights": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "executes_suggested_endpoint": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_structural_write": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "scheduler_experiment_id": (
                    f"snn-sleep-plasticity-scheduler-experiment-{scheduler_experiment_hash[:16]}"
                ),
                "scheduler_experiment_hash": scheduler_experiment_hash,
                "hash_algorithm": "sha256",
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_hash_stability_experiment",
            },
            "experiment_summary": {
                "cycle_count": bounded_cycles,
                "stable_cycle_count": sum(
                    1 for item in observations if item["stable_against_first_cycle"]
                ),
                "proposal_stable": all(
                    item["stable_against_first_cycle"] for item in observations
                ),
                "review_ticket_id": latest["review_ticket_id"],
                "review_ticket_hash": latest["review_ticket_hash"],
                "next_gate": latest["next_gate"],
                "bound_state_revision": latest["state_revision"],
            },
            "ephemeral_experiment": {
                "observations": observations,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "plasticity_applied": False,
                "scheduler_installed": False,
                "suggested_endpoint_called": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_scheduler_design_review"
                if ready
                else "collect_verified_stable_sleep_plasticity_autonomy_proposal",
                "eligible_for_operator_scheduler_design_review": ready,
                "eligible_for_scheduler_installation": False,
                "eligible_for_action": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity": False,
                "next_gate": "operator_review_snn_sleep_plasticity_scheduler_design"
                if ready
                else "/terminus/snn-language-sequence/plasticity-sleep-policy",
                "required_evidence": {
                    "verified_sleep_plasticity_autonomy_proposal": bool(latest["ready"]),
                    "proposal_stable_across_cycles": all(
                        item["stable_against_first_cycle"] for item in observations
                    ),
                    "review_ticket_present": bool(latest["review_ticket_id"]),
                    "autonomy_planning_only": bool(latest["eligible_for_autonomy_planning"]),
                    "scheduler_not_installed": True,
                    "suggested_endpoint_not_called": True,
                },
            },
        }

    def snn_sleep_plasticity_scheduler_design(
        self,
        *,
        limit: int = 20,
        cycles: int = 4,
        min_stable_cycles: int = 3,
        max_review_interval_seconds: float = 300.0,
    ) -> dict[str, Any]:
        """Describe a bounded operator-reviewed scheduler without installing one."""

        bounded_min_stable_cycles = max(1, min(16, int(min_stable_cycles)))
        bounded_review_interval_seconds = max(
            1.0,
            min(3600.0, float(max_review_interval_seconds)),
        )
        experiment = self.snn_sleep_plasticity_scheduler_experiment(
            limit=limit,
            cycles=cycles,
        )
        summary = (
            experiment.get("experiment_summary")
            if isinstance(experiment.get("experiment_summary"), Mapping)
            else {}
        )
        experiment_provenance = (
            experiment.get("provenance_evidence")
            if isinstance(experiment.get("provenance_evidence"), Mapping)
            else {}
        )
        stable_cycle_count = int(summary.get("stable_cycle_count", 0) or 0)
        ready = bool(
            experiment.get("ready")
            and summary.get("proposal_stable")
            and summary.get("review_ticket_id")
            and stable_cycle_count >= bounded_min_stable_cycles
        )
        scheduler_design = {
            "scheduler_mode": "operator_review_only",
            "source_scheduler_experiment_id": experiment_provenance.get(
                "scheduler_experiment_id"
            ),
            "source_scheduler_experiment_hash": experiment_provenance.get(
                "scheduler_experiment_hash"
            ),
            "review_ticket_id": summary.get("review_ticket_id"),
            "review_ticket_hash": summary.get("review_ticket_hash"),
            "bound_state_revision": int(summary.get("bound_state_revision", -1)),
            "reviewed_next_gate": summary.get("next_gate"),
            "observed_cycle_count": int(summary.get("cycle_count", 0) or 0),
            "min_stable_cycles": bounded_min_stable_cycles,
            "observed_stable_cycles": stable_cycle_count,
            "max_review_interval_seconds": bounded_review_interval_seconds,
            "requires_current_revision": True,
            "requires_operator_confirmation": True,
            "automatic_endpoint_execution": False,
            "automatic_plasticity": False,
        }
        provenance_material = {
            "ready": ready,
            "scheduler_design": scheduler_design,
        }
        scheduler_design_hash = self._sha256_json(provenance_material)
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_design",
            "surface": "snn_sleep_plasticity_scheduler_design.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_scheduler_design",
            "advisory": True,
            "isolated": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "executable": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "returns_trained_weights": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "executes_suggested_endpoint": False,
            "installs_scheduler": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_structural_write": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "scheduler_design_id": (
                    f"snn-sleep-plasticity-scheduler-design-{scheduler_design_hash[:16]}"
                ),
                "scheduler_design_hash": scheduler_design_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
                "source_scheduler_experiment_surface": experiment.get("surface"),
                "source_scheduler_experiment_hash": experiment_provenance.get(
                    "scheduler_experiment_hash"
                ),
                "review_ticket_hash": summary.get("review_ticket_hash"),
                "bound_state_revision": int(summary.get("bound_state_revision", -1)),
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_scheduler_design",
            },
            "scheduler_design": scheduler_design,
            "safety_contract": {
                "scheduler_installation_allowed": False,
                "suggested_endpoint_execution_allowed": False,
                "replay_recording_allowed": False,
                "live_replay_allowed": False,
                "permit_issuance_allowed": False,
                "checkpoint_write_allowed": False,
                "growth_allowed": False,
                "pruning_allowed": False,
                "plasticity_allowed": False,
                "runtime_mutation_allowed": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_scheduler_design_review"
                if ready
                else "collect_verified_stable_sleep_plasticity_scheduler_experiment",
                "eligible_for_operator_scheduler_design_review": ready,
                "eligible_for_scheduler_installation": False,
                "eligible_for_action": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity": False,
                "next_gate": "operator_review_snn_sleep_plasticity_scheduler_design"
                if ready
                else "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment",
                "required_evidence": {
                    "verified_stable_scheduler_experiment": bool(experiment.get("ready")),
                    "minimum_stable_cycles_observed": (
                        stable_cycle_count >= bounded_min_stable_cycles
                    ),
                    "review_ticket_present": bool(summary.get("review_ticket_id")),
                    "scheduler_not_installed": True,
                    "suggested_endpoint_not_called": True,
                    "automatic_plasticity_disabled": True,
                },
            },
        }

    def record_snn_sleep_plasticity_scheduler_design_review_ticket(
        self,
        *,
        limit: int = 20,
        cycles: int = 4,
        min_stable_cycles: int = 3,
        max_review_interval_seconds: float = 300.0,
        expected_state_revision: int,
        scheduler_design_hash: str,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Record operator review of a current controller-recomputed scheduler design."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        if not confirmation:
            raise ValueError("SNN sleep plasticity scheduler design review confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN sleep plasticity scheduler design review operator_id is required.")
        if int(expected_state_revision) != int(self._runtime_state.state_revision):
            raise ValueError("SNN sleep plasticity scheduler design review requires current state revision.")
        current = self.snn_sleep_plasticity_scheduler_design(
            limit=limit,
            cycles=cycles,
            min_stable_cycles=min_stable_cycles,
            max_review_interval_seconds=max_review_interval_seconds,
        )
        current_design = (
            current.get("scheduler_design")
            if isinstance(current.get("scheduler_design"), Mapping)
            else {}
        )
        current_provenance = (
            current.get("provenance_evidence")
            if isinstance(current.get("provenance_evidence"), Mapping)
            else {}
        )
        if (
            not current.get("ready")
            or str(scheduler_design_hash or "")
            != str(current_provenance.get("scheduler_design_hash") or "")
        ):
            raise ValueError("SNN sleep plasticity scheduler design review requires current controller evidence.")
        with self._lock:
            material = {
                "recorded_state_revision": int(self._runtime_state.state_revision),
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "design_parameters": {
                    "limit": int(limit),
                    "cycles": int(cycles),
                    "min_stable_cycles": int(min_stable_cycles),
                    "max_review_interval_seconds": float(max_review_interval_seconds),
                },
                "scheduler_design_id": current_provenance.get("scheduler_design_id"),
                "scheduler_design_hash": current_provenance.get("scheduler_design_hash"),
                "source_scheduler_experiment_hash": current_provenance.get(
                    "source_scheduler_experiment_hash"
                ),
                "review_ticket_id": current_design.get("review_ticket_id"),
                "review_ticket_hash": current_design.get("review_ticket_hash"),
                "reviewed_next_gate": current_design.get("reviewed_next_gate"),
            }
            evidence_hash = self._sha256_json(material)
            ticket = {
                "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_design_review_ticket",
                "surface": "snn_sleep_plasticity_scheduler_design_review_ticket.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_scheduler_design_review_ticket",
                "scheduler_design_review_ticket_id": (
                    f"snn-sleep-plasticity-scheduler-design-review-{evidence_hash[:16]}-{uuid4()}"
                ),
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "requires_operator_confirmation": True,
                "advisory": True,
                "executable": False,
                "installs_scheduler": False,
                "executes_suggested_endpoint": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "applies_plasticity": False,
                "mutates_transition_memory": False,
                "mutates_runtime_state": False,
                "eligible_for_scheduler_installation": False,
            }
            self._snn_sleep_plasticity_scheduler_design_review_tickets.appendleft(
                deepcopy(ticket)
            )
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(ticket)

    def verified_snn_sleep_plasticity_scheduler_design_review_ticket(
        self,
        scheduler_design_review_ticket_id: str,
        *,
        operator_id: str | None = None,
    ) -> dict[str, Any] | None:
        expected_operator_id = (
            self._normalize_feedback_text(operator_id, max_chars=160)
            if operator_id is not None
            else None
        )
        with self._lock:
            ticket = self._snn_sleep_plasticity_scheduler_design_review_ticket_index.get(
                str(scheduler_design_review_ticket_id or "").strip()
            )
            if ticket is None:
                return None
            ticket = dict(ticket)
            try:
                design_parameters = dict(ticket.get("design_parameters") or {})
                current = self.snn_sleep_plasticity_scheduler_design(
                    limit=int(design_parameters.get("limit", 0) or 0),
                    cycles=int(design_parameters.get("cycles", 0) or 0),
                    min_stable_cycles=int(design_parameters.get("min_stable_cycles", 0) or 0),
                    max_review_interval_seconds=float(
                        design_parameters.get("max_review_interval_seconds", 0.0) or 0.0
                    ),
                )
            except (TypeError, ValueError):
                return None
            current_design = (
                current.get("scheduler_design")
                if isinstance(current.get("scheduler_design"), Mapping)
                else {}
            )
            current_provenance = (
                current.get("provenance_evidence")
                if isinstance(current.get("provenance_evidence"), Mapping)
                else {}
            )
            material = {
                "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                "operator_id": ticket.get("operator_id"),
                "confirmation": bool(ticket.get("confirmation")),
                "design_parameters": dict(ticket.get("design_parameters") or {}),
                "scheduler_design_id": ticket.get("scheduler_design_id"),
                "scheduler_design_hash": ticket.get("scheduler_design_hash"),
                "source_scheduler_experiment_hash": ticket.get(
                    "source_scheduler_experiment_hash"
                ),
                "review_ticket_id": ticket.get("review_ticket_id"),
                "review_ticket_hash": ticket.get("review_ticket_hash"),
                "reviewed_next_gate": ticket.get("reviewed_next_gate"),
            }
            return (
                deepcopy(ticket)
                if (
                    ticket.get("artifact_kind")
                    == "terminus_snn_sleep_plasticity_scheduler_design_review_ticket"
                    and ticket.get("surface")
                    == "snn_sleep_plasticity_scheduler_design_review_ticket.v1"
                    and ticket.get("source")
                    == "replay_controller.snn_sleep_plasticity_scheduler_design_review_ticket"
                    and ticket.get("ready")
                    and ticket.get("owned_by_marulho")
                    and ticket.get("confirmation") is True
                    and (
                        expected_operator_id is None
                        or str(ticket.get("operator_id") or "") == expected_operator_id
                    )
                    and int(ticket.get("recorded_state_revision", -1))
                    == int(self._runtime_state.state_revision)
                    and str(ticket.get("evidence_hash") or "") == self._sha256_json(material)
                    and current.get("ready")
                    and str(ticket.get("scheduler_design_id") or "")
                    == str(current_provenance.get("scheduler_design_id") or "")
                    and str(ticket.get("scheduler_design_hash") or "")
                    == str(current_provenance.get("scheduler_design_hash") or "")
                    and str(ticket.get("source_scheduler_experiment_hash") or "")
                    == str(current_provenance.get("source_scheduler_experiment_hash") or "")
                    and str(ticket.get("review_ticket_id") or "")
                    == str(current_design.get("review_ticket_id") or "")
                    and str(ticket.get("review_ticket_hash") or "")
                    == str(current_design.get("review_ticket_hash") or "")
                    and str(ticket.get("reviewed_next_gate") or "")
                    == str(current_design.get("reviewed_next_gate") or "")
                    and self.verified_snn_sleep_plasticity_review_ticket(
                        str(ticket.get("review_ticket_id") or "")
                    )
                    is not None
                    and not bool(ticket.get("executable"))
                    and not bool(ticket.get("installs_scheduler"))
                    and not bool(ticket.get("executes_suggested_endpoint"))
                    and not bool(ticket.get("records_replay_artifact"))
                    and not bool(ticket.get("issues_regeneration_permit"))
                    and not bool(ticket.get("writes_checkpoint"))
                    and not bool(ticket.get("applies_plasticity"))
                    and not bool(ticket.get("mutates_transition_memory"))
                    and not bool(ticket.get("mutates_runtime_state"))
                    and not bool(ticket.get("eligible_for_scheduler_installation"))
                )
                else None
            )

    def snn_sleep_plasticity_scheduler_design_review_ticket_queue(
        self,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Expose verified accepted scheduler designs without installing a scheduler."""

        with self._lock:
            started = time.perf_counter()
            requested = max(
                1,
                min(DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS, int(limit)),
            )
            source_limit = min(
                requested,
                SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT,
            )
            retained_count = int(
                len(self._snn_sleep_plasticity_scheduler_design_review_tickets)
            )
            source_window_inspected_count = int(min(retained_count, source_limit))
            tickets: list[dict[str, Any]] = []
            verified_count = 0
            stale_count = 0
            tampered_count = 0
            latest_verified: dict[str, Any] | None = None
            current_revision = int(self._runtime_state.state_revision)
            for source_rank, raw_ticket in enumerate(
                islice(
                    self._snn_sleep_plasticity_scheduler_design_review_tickets,
                    source_limit,
                )
            ):
                if not isinstance(raw_ticket, Mapping):
                    continue
                ticket = dict(raw_ticket)
                revision_current = (
                    int(ticket.get("recorded_state_revision", -1)) == current_revision
                )
                non_executing = not any(
                    bool(ticket.get(field))
                    for field in (
                        "executable",
                        "installs_scheduler",
                        "executes_suggested_endpoint",
                        "records_replay_artifact",
                        "issues_regeneration_permit",
                        "writes_checkpoint",
                        "applies_plasticity",
                        "mutates_transition_memory",
                        "mutates_runtime_state",
                        "eligible_for_scheduler_installation",
                    )
                )
                try:
                    verified_ticket = (
                        self.verified_snn_sleep_plasticity_scheduler_design_review_ticket(
                            str(ticket.get("scheduler_design_review_ticket_id") or "")
                        )
                    )
                except (TypeError, ValueError):
                    verified_ticket = None
                verified = verified_ticket is not None
                if verified:
                    verified_count += 1
                elif not revision_current:
                    stale_count += 1
                else:
                    tampered_count += 1
                projection = {
                        "scheduler_design_review_ticket_id": ticket.get(
                            "scheduler_design_review_ticket_id"
                        ),
                        "scheduler_design_review_ticket_hash": ticket.get("evidence_hash"),
                        "recorded_at": ticket.get("recorded_at"),
                        "bound_state_revision": int(
                            ticket.get("recorded_state_revision", -1)
                        ),
                        "operator_id": ticket.get("operator_id"),
                        "scheduler_design_id": ticket.get("scheduler_design_id"),
                        "scheduler_design_hash": ticket.get("scheduler_design_hash"),
                        "source_scheduler_experiment_hash": ticket.get(
                            "source_scheduler_experiment_hash"
                        ),
                        "review_ticket_id": ticket.get("review_ticket_id"),
                        "review_ticket_hash": ticket.get("review_ticket_hash"),
                        "scheduler_review_parameters": (
                            dict(ticket.get("design_parameters") or {})
                            if isinstance(ticket.get("design_parameters"), Mapping)
                            else {}
                        ),
                        "reviewed_sleep_plasticity_endpoint": ticket.get(
                            "reviewed_next_gate"
                        ),
                        "verified": verified,
                        "revision_current": revision_current,
                        "non_executing": non_executing,
                        "source_rank": int(source_rank),
                        "executable": False,
                        "installs_scheduler": False,
                        "executes_suggested_endpoint": False,
                        "mutates_runtime_state": False,
                        "applies_plasticity": False,
                    }
                tickets.append(projection)
                if verified and latest_verified is None:
                    latest_verified = deepcopy(projection)
            selected = tickets[:requested]
            latency_ms = (time.perf_counter() - started) * 1000.0
            source_window = {
                "surface": (
                    "bounded_snn_sleep_plasticity_scheduler_design_review_ticket_"
                    "queue_source_window.v1"
                ),
                "policy": "recent_sleep_plasticity_scheduler_design_review_ticket_window",
                "source": (
                    "replay_controller."
                    "snn_sleep_plasticity_scheduler_design_review_tickets"
                ),
                "selection_criteria": [
                    "newest_scheduler_design_review_tickets_first",
                    "current_revision_verified_design_tickets",
                    "stop_at_source_window_before_installation_proposal",
                ],
                "retained_count": retained_count,
                "retention_limit": DEFAULT_SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKETS,
                "requested_limit": int(requested),
                "source_window_limit": (
                    SNN_SLEEP_PLASTICITY_SCHEDULER_DESIGN_REVIEW_TICKET_QUEUE_SOURCE_WINDOW_LIMIT
                ),
                "source_window_inspected_count": source_window_inspected_count,
                "source_window_count": int(len(tickets)),
                "source_truncated_count": int(
                    max(0, retained_count - source_window_inspected_count)
                ),
                "count_is_source_window": True,
                "latest_verified_scope": "source_window_only",
                "global_candidate_scan": False,
                "global_score_scan": False,
                "raw_replay_text_payload_loaded": False,
                "language_reasoning": False,
                "runs_live_tick": False,
                "runs_every_token": False,
                "mutates_runtime_state": False,
                "applies_plasticity": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "installs_scheduler": False,
                "executes_suggested_endpoint": False,
                "archival_storage_device": "cpu",
                "source_window_selection_device": "cpu",
                "score_device": "cpu",
                "gpu_used": False,
                "latency_ms": float(latency_ms),
            }
            return {
                "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_design_review_ticket_queue",
                "surface": "snn_sleep_plasticity_scheduler_design_review_ticket_queue.v1",
                "available": True,
                "ready": verified_count > 0,
                "owned_by_marulho": True,
                "source": (
                    "replay_controller."
                    "snn_sleep_plasticity_scheduler_design_review_ticket_queue"
                ),
                "count": len(tickets),
                "retained_count": retained_count,
                "limit": requested,
                "source_window": source_window,
                "current_state_revision": current_revision,
                "verified_count": verified_count,
                "stale_count": stale_count,
                "tampered_count": tampered_count,
                "latest_verified_ticket": latest_verified,
                "advisory": True,
                "executable": False,
                "installs_scheduler": False,
                "executes_suggested_endpoint": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "eligible_for_scheduler_installation": False,
                "tickets": selected,
            }

    def snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
        self,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Propose installation-preflight review from accepted scheduler designs."""

        queue = self.snn_sleep_plasticity_scheduler_design_review_ticket_queue(limit=limit)
        latest = (
            queue.get("latest_verified_ticket")
            if isinstance(queue.get("latest_verified_ticket"), Mapping)
            else None
        )
        ready = latest is not None and bool(queue.get("ready"))
        ticket_id = (
            str(latest.get("scheduler_design_review_ticket_id") or "")
            if latest
            else ""
        )
        candidate = {
            "candidate_id": (
                f"snn-sleep-plasticity-scheduler-installation:{ticket_id[:64]}"
                if ticket_id
                else None
            ),
            "action": (
                "review_snn_sleep_plasticity_scheduler_installation_preflight"
                if ready
                else "collect_verified_scheduler_design_review_ticket"
            ),
            "scheduler_design_review_ticket_id": ticket_id or None,
            "scheduler_design_review_ticket_hash": (
                latest.get("scheduler_design_review_ticket_hash") if latest else None
            ),
            "scheduler_design_id": latest.get("scheduler_design_id") if latest else None,
            "scheduler_design_hash": latest.get("scheduler_design_hash") if latest else None,
            "source_scheduler_experiment_hash": (
                latest.get("source_scheduler_experiment_hash") if latest else None
            ),
            "review_ticket_id": latest.get("review_ticket_id") if latest else None,
            "review_ticket_hash": latest.get("review_ticket_hash") if latest else None,
            "reviewed_sleep_plasticity_endpoint": (
                latest.get("reviewed_sleep_plasticity_endpoint") if latest else None
            ),
            "scheduler_review_parameters": (
                dict(latest.get("scheduler_review_parameters") or {}) if latest else {}
            ),
            "bound_state_revision": int(queue.get("current_state_revision", -1)),
            "priority_score": 1.0 if ready else 0.0,
        }
        provenance_material = {
            "ready": ready,
            "current_state_revision": int(queue.get("current_state_revision", -1)),
            "candidate": candidate,
        }
        proposal_hash = self._sha256_json(provenance_material)
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_installation_autonomy_proposal",
            "surface": "snn_sleep_plasticity_scheduler_installation_autonomy_proposal.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": (
                "replay_controller."
                "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
            ),
            "advisory": True,
            "isolated": True,
            "executable": False,
            "installs_scheduler": False,
            "registers_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_scheduler_installation": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_structural_write": False,
            "eligible_for_growth": False,
            "eligible_for_pruning": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "scheduler_installation_autonomy_proposal_id": (
                    "snn-sleep-plasticity-scheduler-installation-autonomy-"
                    f"{proposal_hash[:16]}"
                ),
                "scheduler_installation_autonomy_proposal_hash": proposal_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_scheduler_installation_autonomy_proposal",
            },
            "safety_contract": {
                "scheduler_installation_allowed": False,
                "timer_registration_allowed": False,
                "background_worker_start_allowed": False,
                "suggested_endpoint_execution_allowed": False,
                "replay_recording_allowed": False,
                "permit_issuance_allowed": False,
                "checkpoint_write_allowed": False,
                "plasticity_allowed": False,
                "transition_memory_mutation_allowed": False,
                "runtime_mutation_allowed": False,
            },
            "candidate": candidate,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_scheduler_installation_preflight_review"
                    if ready
                    else "collect_verified_scheduler_design_review_ticket"
                ),
                "eligible_for_autonomy_planning": ready,
                "eligible_for_scheduler_installation_preflight_review": ready,
                "eligible_for_scheduler_installation": False,
                "eligible_for_action": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity": False,
                "next_gate": (
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "scheduler-installation-preflight"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "scheduler-design/review-tickets"
                ),
                "required_evidence": {
                    "verified_scheduler_design_review_ticket": ready,
                    "scheduler_not_installed": True,
                    "timer_not_registered": True,
                    "background_worker_not_started": True,
                    "suggested_endpoint_not_called": True,
                    "automatic_plasticity_disabled": True,
                },
            },
        }

    def snn_sleep_plasticity_scheduler_installation_preflight(
        self,
        *,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Verify scheduler-installation intent without creating a scheduler."""

        proposal = self.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
            limit=limit
        )
        candidate = (
            proposal.get("candidate")
            if isinstance(proposal.get("candidate"), Mapping)
            else {}
        )
        proposal_provenance = (
            proposal.get("provenance_evidence")
            if isinstance(proposal.get("provenance_evidence"), Mapping)
            else {}
        )
        ready = bool(
            proposal.get("ready")
            and candidate.get("scheduler_design_review_ticket_id")
            and candidate.get("scheduler_design_review_ticket_hash")
            and candidate.get("scheduler_design_id")
            and candidate.get("scheduler_design_hash")
            and candidate.get("source_scheduler_experiment_hash")
            and candidate.get("review_ticket_id")
            and candidate.get("review_ticket_hash")
            and candidate.get("reviewed_sleep_plasticity_endpoint")
            and all(
                key in dict(candidate.get("scheduler_review_parameters") or {})
                for key in (
                    "limit",
                    "cycles",
                    "min_stable_cycles",
                    "max_review_interval_seconds",
                )
            )
            and int(candidate.get("bound_state_revision", -1))
            == int(self._runtime_state.state_revision)
            and not any(
                bool(proposal.get(field))
                for field in (
                    "executable",
                    "installs_scheduler",
                    "registers_timer",
                    "starts_background_worker",
                    "executes_suggested_endpoint",
                    "records_replay_artifact",
                    "issues_regeneration_permit",
                    "writes_checkpoint",
                    "applies_plasticity",
                    "mutates_transition_memory",
                    "mutates_runtime_state",
                    "eligible_for_scheduler_installation",
                )
            )
        )
        installation_review_preflight = {
            "scheduler_mode": "operator_review_only",
            "scheduler_design_review_ticket_id": candidate.get(
                "scheduler_design_review_ticket_id"
            ),
            "scheduler_design_review_ticket_hash": candidate.get(
                "scheduler_design_review_ticket_hash"
            ),
            "scheduler_design_id": candidate.get("scheduler_design_id"),
            "scheduler_design_hash": candidate.get("scheduler_design_hash"),
            "source_scheduler_experiment_hash": candidate.get(
                "source_scheduler_experiment_hash"
            ),
            "review_ticket_id": candidate.get("review_ticket_id"),
            "review_ticket_hash": candidate.get("review_ticket_hash"),
            "reviewed_sleep_plasticity_endpoint": candidate.get(
                "reviewed_sleep_plasticity_endpoint"
            ),
            "scheduler_review_parameters": dict(
                candidate.get("scheduler_review_parameters") or {}
            ),
            "bound_state_revision": int(candidate.get("bound_state_revision", -1)),
            "requires_operator_confirmation": True,
            "automatic_endpoint_execution": False,
            "automatic_plasticity": False,
        }
        provenance_material = {
            "ready": ready,
            "source_scheduler_installation_autonomy_proposal_hash": (
                proposal_provenance.get(
                    "scheduler_installation_autonomy_proposal_hash"
                )
            ),
            "installation_review_preflight": installation_review_preflight,
        }
        preflight_hash = self._sha256_json(provenance_material)
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_scheduler_installation_preflight",
            "surface": "snn_sleep_plasticity_scheduler_installation_preflight.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": (
                "replay_controller."
                "snn_sleep_plasticity_scheduler_installation_preflight"
            ),
            "advisory": True,
            "isolated": True,
            "executable": False,
            "installs_scheduler": False,
            "registers_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_scheduler_installation": False,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_structural_write": False,
            "eligible_for_growth": False,
            "eligible_for_pruning": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "scheduler_installation_preflight_id": (
                    f"snn-sleep-plasticity-scheduler-installation-preflight-{preflight_hash[:16]}"
                ),
                "scheduler_installation_preflight_hash": preflight_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
                "source_scheduler_installation_autonomy_proposal_hash": (
                    proposal_provenance.get(
                        "scheduler_installation_autonomy_proposal_hash"
                    )
                ),
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_scheduler_installation_preflight",
            },
            "safety_contract": {
                "scheduler_installation_allowed": False,
                "timer_registration_allowed": False,
                "background_worker_start_allowed": False,
                "suggested_endpoint_execution_allowed": False,
                "replay_recording_allowed": False,
                "permit_issuance_allowed": False,
                "checkpoint_write_allowed": False,
                "plasticity_allowed": False,
                "transition_memory_mutation_allowed": False,
                "runtime_mutation_allowed": False,
            },
            "installation_review_preflight": installation_review_preflight,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_scheduler_installation_review"
                    if ready
                    else "collect_verified_scheduler_installation_autonomy_proposal"
                ),
                "eligible_for_operator_scheduler_installation_review": ready,
                "eligible_for_scheduler_installation": False,
                "eligible_for_action": False,
                "eligible_for_structural_write": False,
                "eligible_for_plasticity": False,
                "next_gate": (
                    "operator_review_snn_sleep_plasticity_scheduler_installation"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "scheduler-design/review-tickets"
                ),
                "required_evidence": {
                    "verified_scheduler_installation_autonomy_proposal": bool(
                        proposal.get("ready")
                    ),
                    "scheduler_design_review_ticket_present": bool(
                        candidate.get("scheduler_design_review_ticket_id")
                    ),
                    "scheduler_not_installed": True,
                    "timer_not_registered": True,
                    "background_worker_not_started": True,
                    "suggested_endpoint_not_called": True,
                    "automatic_plasticity_disabled": True,
                },
            },
        }

    def install_snn_sleep_plasticity_review_scheduler(
        self,
        *,
        limit: int = 20,
        expected_state_revision: int,
        scheduler_installation_preflight_hash: str,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Install a passive review scheduler configuration without running work."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        if not confirmation:
            raise ValueError("SNN sleep plasticity review scheduler installation confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN sleep plasticity review scheduler installation operator_id is required.")
        if int(expected_state_revision) != int(self._runtime_state.state_revision):
            raise ValueError("SNN sleep plasticity review scheduler installation requires current state revision.")
        preflight = self.snn_sleep_plasticity_scheduler_installation_preflight(limit=limit)
        preflight_provenance = (
            preflight.get("provenance_evidence")
            if isinstance(preflight.get("provenance_evidence"), Mapping)
            else {}
        )
        installation_review = (
            preflight.get("installation_review_preflight")
            if isinstance(preflight.get("installation_review_preflight"), Mapping)
            else {}
        )
        expected_hash = str(
            preflight_provenance.get("scheduler_installation_preflight_hash") or ""
        )
        if (
            not preflight.get("ready")
            or str(scheduler_installation_preflight_hash or "") != expected_hash
        ):
            raise ValueError("SNN sleep plasticity review scheduler installation requires current preflight evidence.")
        review_parameters = dict(
            installation_review.get("scheduler_review_parameters") or {}
        )
        review_interval_seconds = float(
            review_parameters.get("max_review_interval_seconds", 0.0) or 0.0
        )
        if review_interval_seconds < 60.0 or review_interval_seconds > 3600.0:
            raise ValueError("SNN sleep plasticity review scheduler installation requires bounded review cadence.")
        installed_at = datetime.now(timezone.utc)
        next_review_due_at = installed_at + timedelta(seconds=review_interval_seconds)
        with self._lock:
            material = {
                "installed_state_revision": int(self._runtime_state.state_revision),
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "scheduler_installation_preflight_hash": expected_hash,
                "scheduler_design_review_ticket_id": installation_review.get(
                    "scheduler_design_review_ticket_id"
                ),
                "scheduler_design_review_ticket_hash": installation_review.get(
                    "scheduler_design_review_ticket_hash"
                ),
                "scheduler_design_hash": installation_review.get("scheduler_design_hash"),
                "reviewed_sleep_plasticity_endpoint": installation_review.get(
                    "reviewed_sleep_plasticity_endpoint"
                ),
                "scheduler_review_parameters": review_parameters,
                "installed_at": installed_at.isoformat(),
                "next_review_due_at": next_review_due_at.isoformat(),
                "acknowledged_cycle_count": 0,
                "last_cycle_acknowledgment": None,
                "scheduler_configuration_revision": 0,
                "previous_scheduler_configuration_evidence_hash": None,
            }
            evidence_hash = self._sha256_json(material)
            installation = {
                "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_installation",
                "surface": "snn_sleep_plasticity_review_scheduler_installation.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_review_scheduler_installation",
                "scheduler_installation_id": (
                    f"snn-sleep-plasticity-review-scheduler-{evidence_hash[:16]}-{uuid4()}"
                ),
                "evidence_hash": evidence_hash,
                **material,
                "scheduler_installed": True,
                "scheduler_mode": "review_only",
                "review_due": False,
                "advisory": True,
                "executable": False,
                "registers_os_timer": False,
                "starts_background_worker": False,
                "executes_suggested_endpoint": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "applies_plasticity": False,
                "mutates_transition_memory": False,
                "mutates_runtime_state": False,
            }
            self._snn_sleep_plasticity_review_scheduler_installations.appendleft(
                deepcopy(installation)
            )
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(installation)

    def snn_sleep_plasticity_review_scheduler_runtime(
        self,
        *,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Expose the passive review scheduler and due-state without executing work."""

        with self._lock:
            installation = (
                dict(self._snn_sleep_plasticity_review_scheduler_installations[0])
                if self._snn_sleep_plasticity_review_scheduler_installations
                else None
            )
            if installation is None:
                return {
                    "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_runtime",
                    "surface": "snn_sleep_plasticity_review_scheduler_runtime.v1",
                    "available": True,
                    "ready": False,
                    "owned_by_marulho": True,
                    "source": "replay_controller.snn_sleep_plasticity_review_scheduler_runtime",
                    "scheduler_installed": False,
                    "scheduler_mode": "review_only",
                    "review_due": False,
                    "executable": False,
                    "registers_os_timer": False,
                    "starts_background_worker": False,
                    "executes_suggested_endpoint": False,
                    "applies_plasticity": False,
                    "mutates_runtime_state": False,
                }
            material = {
                "installed_state_revision": int(
                    installation.get("installed_state_revision", -1)
                ),
                "operator_id": installation.get("operator_id"),
                "confirmation": bool(installation.get("confirmation")),
                "scheduler_installation_preflight_hash": installation.get(
                    "scheduler_installation_preflight_hash"
                ),
                "scheduler_design_review_ticket_id": installation.get(
                    "scheduler_design_review_ticket_id"
                ),
                "scheduler_design_review_ticket_hash": installation.get(
                    "scheduler_design_review_ticket_hash"
                ),
                "scheduler_design_hash": installation.get("scheduler_design_hash"),
                "reviewed_sleep_plasticity_endpoint": installation.get(
                    "reviewed_sleep_plasticity_endpoint"
                ),
                "scheduler_review_parameters": dict(
                    installation.get("scheduler_review_parameters") or {}
                ),
                "installed_at": installation.get("installed_at"),
                "next_review_due_at": installation.get("next_review_due_at"),
                "acknowledged_cycle_count": int(
                    installation.get("acknowledged_cycle_count", 0) or 0
                ),
                "last_cycle_acknowledgment": (
                    dict(installation.get("last_cycle_acknowledgment") or {})
                    if installation.get("last_cycle_acknowledgment")
                    else None
                ),
                "scheduler_configuration_revision": int(
                    installation.get("scheduler_configuration_revision", 0) or 0
                ),
                "previous_scheduler_configuration_evidence_hash": installation.get(
                    "previous_scheduler_configuration_evidence_hash"
                ),
            }
            try:
                next_review_due_at = datetime.fromisoformat(
                    str(installation.get("next_review_due_at") or "")
                )
            except ValueError:
                next_review_due_at = datetime.min.replace(tzinfo=timezone.utc)
            current_preflight = self.snn_sleep_plasticity_scheduler_installation_preflight(
                limit=int(
                    (
                        installation.get("scheduler_review_parameters")
                        if isinstance(
                            installation.get("scheduler_review_parameters"),
                            Mapping,
                        )
                        else {}
                    ).get("limit", 20)
                    or 20
                )
            )
            current_preflight_provenance = (
                current_preflight.get("provenance_evidence")
                if isinstance(current_preflight.get("provenance_evidence"), Mapping)
                else {}
            )
            verified = bool(
                installation.get("artifact_kind")
                == "terminus_snn_sleep_plasticity_review_scheduler_installation"
                and installation.get("surface")
                == "snn_sleep_plasticity_review_scheduler_installation.v1"
                and installation.get("owned_by_marulho")
                and installation.get("confirmation") is True
                and int(installation.get("installed_state_revision", -1))
                == int(self._runtime_state.state_revision)
                and str(installation.get("evidence_hash") or "")
                == self._sha256_json(material)
                and current_preflight.get("ready")
                and str(installation.get("scheduler_installation_preflight_hash") or "")
                == str(
                    current_preflight_provenance.get(
                        "scheduler_installation_preflight_hash"
                    )
                    or ""
                )
                and installation.get("scheduler_mode") == "review_only"
                and installation.get("scheduler_installed") is True
                and not bool(installation.get("registers_os_timer"))
                and not bool(installation.get("starts_background_worker"))
                and not bool(installation.get("executes_suggested_endpoint"))
                and not bool(installation.get("records_replay_artifact"))
                and not bool(installation.get("issues_regeneration_permit"))
                and not bool(installation.get("writes_checkpoint"))
                and not bool(installation.get("applies_plasticity"))
                and not bool(installation.get("mutates_transition_memory"))
                and not bool(installation.get("mutates_runtime_state"))
            )
            effective_observed_at = observed_at or datetime.now(timezone.utc)
            review_due = bool(verified and effective_observed_at >= next_review_due_at)
            return {
                "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_runtime",
                "surface": "snn_sleep_plasticity_review_scheduler_runtime.v1",
                "available": True,
                "ready": verified,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_review_scheduler_runtime",
                "scheduler_installation_id": installation.get("scheduler_installation_id"),
                "scheduler_installation_evidence_hash": installation.get(
                    "evidence_hash"
                ),
                "scheduler_installed": verified,
                "scheduler_mode": "review_only",
                "installed_state_revision": installation.get("installed_state_revision"),
                "installed_at": installation.get("installed_at"),
                "next_review_due_at": installation.get("next_review_due_at"),
                "acknowledged_cycle_count": int(
                    installation.get("acknowledged_cycle_count", 0) or 0
                ),
                "last_cycle_acknowledgment": deepcopy(
                    installation.get("last_cycle_acknowledgment")
                ),
                "scheduler_configuration_revision": int(
                    installation.get("scheduler_configuration_revision", 0) or 0
                ),
                "previous_scheduler_configuration_evidence_hash": installation.get(
                    "previous_scheduler_configuration_evidence_hash"
                ),
                "observed_at": effective_observed_at.isoformat(),
                "review_due": review_due,
                "reviewed_sleep_plasticity_endpoint": installation.get(
                    "reviewed_sleep_plasticity_endpoint"
                ),
                "advisory": True,
                "executable": False,
                "registers_os_timer": False,
                "starts_background_worker": False,
                "executes_suggested_endpoint": False,
                "records_replay_artifact": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "applies_plasticity": False,
                "mutates_transition_memory": False,
                "mutates_runtime_state": False,
                "promotion_gate": {
                    "status": (
                        "ready_for_operator_review_cycle_inspection"
                        if review_due
                        else "waiting_for_review_cadence"
                    ),
                    "eligible_for_operator_review_cycle_inspection": review_due,
                    "eligible_for_endpoint_execution": False,
                    "eligible_for_plasticity": False,
                },
            }

    def snn_sleep_plasticity_review_scheduler_cycle_inspection(
        self,
        *,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Inspect a due passive scheduler cycle without executing its reviewed gate."""

        scheduler_runtime = self.snn_sleep_plasticity_review_scheduler_runtime(
            observed_at=observed_at
        )
        ready = bool(
            scheduler_runtime.get("ready")
            and scheduler_runtime.get("scheduler_installed")
            and scheduler_runtime.get("review_due")
            and scheduler_runtime.get("reviewed_sleep_plasticity_endpoint")
            and not scheduler_runtime.get("registers_os_timer")
            and not scheduler_runtime.get("starts_background_worker")
            and not scheduler_runtime.get("executes_suggested_endpoint")
            and not scheduler_runtime.get("applies_plasticity")
            and not scheduler_runtime.get("mutates_runtime_state")
        )
        cycle_inspection = {
            "scheduler_installation_id": scheduler_runtime.get(
                "scheduler_installation_id"
            ),
            "scheduler_installation_evidence_hash": scheduler_runtime.get(
                "scheduler_installation_evidence_hash"
            ),
            "scheduler_mode": scheduler_runtime.get("scheduler_mode"),
            "installed_state_revision": scheduler_runtime.get(
                "installed_state_revision"
            ),
            "observed_at": scheduler_runtime.get("observed_at"),
            "next_review_due_at": scheduler_runtime.get("next_review_due_at"),
            "review_due": bool(scheduler_runtime.get("review_due")),
            "reviewed_sleep_plasticity_endpoint": scheduler_runtime.get(
                "reviewed_sleep_plasticity_endpoint"
            ),
        }
        inspection_hash = self._sha256_json(
            {
                "ready": ready,
                "cycle_inspection": cycle_inspection,
            }
        )
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_cycle_inspection",
            "surface": "snn_sleep_plasticity_review_scheduler_cycle_inspection.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_review_scheduler_cycle_inspection",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_endpoint_execution": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "review_scheduler_cycle_inspection_id": (
                    f"snn-sleep-plasticity-review-scheduler-cycle-{inspection_hash[:16]}"
                ),
                "review_scheduler_cycle_inspection_hash": inspection_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_review_scheduler_cycle_inspection",
            },
            "cycle_inspection": cycle_inspection,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_reviewed_sleep_plasticity_gate_inspection"
                    if ready
                    else "waiting_for_review_cadence"
                ),
                "eligible_for_operator_reviewed_gate_inspection": ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_plasticity": False,
                "next_gate": (
                    "operator_inspect_reviewed_sleep_plasticity_gate"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler"
                ),
            },
        }

    def snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
        self,
        *,
        scheduler_installation_id: str,
        scheduler_installation_evidence_hash: str,
        review_ticket_id: str,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Verify one due-cycle cadence advancement without mutating scheduler state."""

        inspection = self.snn_sleep_plasticity_review_scheduler_cycle_inspection(
            observed_at=observed_at
        )
        cycle = inspection.get("cycle_inspection") if isinstance(
            inspection.get("cycle_inspection"), Mapping
        ) else {}
        ticket = self.verified_snn_replay_artifact_recording_review_ticket(
            review_ticket_id
        )
        ticket_already_consumed = any(
            str(
                (
                    item.get("last_cycle_acknowledgment")
                    if isinstance(item.get("last_cycle_acknowledgment"), Mapping)
                    else {}
                ).get("acknowledged_due_cycle_review_ticket_id")
                or ""
            )
            == str(review_ticket_id or "")
            for item in self._snn_sleep_plasticity_review_scheduler_installations
            if isinstance(item, Mapping)
        )
        required = {
            "due_cycle_inspection_ready": bool(inspection.get("ready")),
            "scheduler_installation_id_matches": bool(scheduler_installation_id)
            and str(scheduler_installation_id)
            == str(cycle.get("scheduler_installation_id") or ""),
            "scheduler_installation_evidence_hash_matches": bool(
                scheduler_installation_evidence_hash
            )
            and str(scheduler_installation_evidence_hash)
            == str(cycle.get("scheduler_installation_evidence_hash") or ""),
            "due_cycle_review_ticket_verified": ticket is not None,
            "due_cycle_review_ticket_lineage_present": bool(
                ticket and ticket.get("due_cycle_review_proposal_hash")
            ),
            "due_cycle_review_ticket_scheduler_matches": bool(ticket)
            and str(ticket.get("scheduler_installation_id") or "")
            == str(scheduler_installation_id or "")
            and str(ticket.get("scheduler_installation_evidence_hash") or "")
            == str(scheduler_installation_evidence_hash or ""),
            "due_cycle_review_ticket_deadline_matches": bool(ticket)
            and str(ticket.get("acknowledged_review_due_at") or "")
            == str(cycle.get("next_review_due_at") or ""),
            "due_cycle_review_ticket_not_consumed": not ticket_already_consumed,
            "endpoint_execution_blocked": inspection.get(
                "executes_suggested_endpoint"
            )
            is False,
            "artifact_recording_blocked": inspection.get("records_replay_artifact")
            is False,
            "checkpoint_write_blocked": inspection.get("writes_checkpoint") is False,
            "plasticity_blocked": inspection.get("applies_plasticity") is False,
            "transition_memory_mutation_blocked": inspection.get(
                "mutates_transition_memory"
            )
            is False,
            "runtime_mutation_blocked": inspection.get("mutates_runtime_state") is False,
        }
        ready = all(required.values())
        preflight_hash = self._sha256_json(
            {
                "ready": ready,
                "scheduler_installation_id": scheduler_installation_id,
                "scheduler_installation_evidence_hash": scheduler_installation_evidence_hash,
                "review_ticket_id": review_ticket_id,
                "required_evidence": required,
            }
        )
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight",
            "surface": "snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "provenance_evidence": {
                "cycle_acknowledgment_preflight_id": (
                    f"snn-sleep-plasticity-cycle-ack-preflight-{preflight_hash[:16]}"
                ),
                "cycle_acknowledgment_preflight_hash": preflight_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
            },
            "promotion_gate": {
                "status": (
                    "ready_for_operator_cycle_acknowledgment"
                    if ready
                    else "waiting_for_due_cycle_acknowledgment_evidence"
                ),
                "eligible_for_operator_cycle_acknowledgment": ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_plasticity": False,
                "required_evidence": required,
            },
        }

    def acknowledge_snn_sleep_plasticity_review_scheduler_cycle(
        self,
        *,
        expected_state_revision: int,
        scheduler_installation_id: str,
        scheduler_installation_evidence_hash: str,
        review_ticket_id: str,
        operator_id: str,
        confirmation: bool,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Advance one passive review cadence after operator inspection."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        if not confirmation:
            raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment operator_id is required.")
        if int(expected_state_revision) != int(self._runtime_state.state_revision):
            raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires current state revision.")
        preflight = self.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
            scheduler_installation_id=scheduler_installation_id,
            scheduler_installation_evidence_hash=scheduler_installation_evidence_hash,
            review_ticket_id=review_ticket_id,
            observed_at=observed_at,
        )
        if not preflight.get("ready"):
            raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires current preflight evidence.")
        inspection = self.snn_sleep_plasticity_review_scheduler_cycle_inspection(
            observed_at=observed_at
        )
        cycle = (
            inspection.get("cycle_inspection")
            if isinstance(inspection.get("cycle_inspection"), Mapping)
            else {}
        )
        provenance = inspection.get("provenance_evidence") if isinstance(
            inspection.get("provenance_evidence"), Mapping
        ) else {}
        expected_inspection_hash = str(provenance.get("review_scheduler_cycle_inspection_hash") or "")
        ticket = self.verified_snn_replay_artifact_recording_review_ticket(
            review_ticket_id
        )
        if (
            not inspection.get("ready")
            or str(scheduler_installation_id or "")
            != str(cycle.get("scheduler_installation_id") or "")
            or str(scheduler_installation_evidence_hash or "")
            != str(cycle.get("scheduler_installation_evidence_hash") or "")
            or ticket is None
            or not ticket.get("due_cycle_review_proposal_hash")
            or str(ticket.get("scheduler_installation_id") or "")
            != str(scheduler_installation_id or "")
            or str(ticket.get("scheduler_installation_evidence_hash") or "")
            != str(scheduler_installation_evidence_hash or "")
            or str(ticket.get("acknowledged_review_due_at") or "")
            != str(cycle.get("next_review_due_at") or "")
            or inspection.get("executes_suggested_endpoint") is not False
            or inspection.get("records_replay_artifact") is not False
            or inspection.get("writes_checkpoint") is not False
            or inspection.get("applies_plasticity") is not False
            or inspection.get("mutates_transition_memory") is not False
            or inspection.get("mutates_runtime_state") is not False
        ):
            raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires current due-cycle inspection evidence.")
        acknowledged_at = observed_at or datetime.now(timezone.utc)
        with self._lock:
            if not self._snn_sleep_plasticity_review_scheduler_installations:
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires an installed scheduler.")
            installation = dict(
                self._snn_sleep_plasticity_review_scheduler_installations[0]
            )
            if str(installation.get("scheduler_installation_id") or "") != str(
                scheduler_installation_id or ""
            ):
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires the current scheduler installation.")
            if str(installation.get("evidence_hash") or "") != str(
                scheduler_installation_evidence_hash or ""
            ):
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires the current scheduler evidence hash.")
            if any(
                str(
                    (
                        item.get("last_cycle_acknowledgment")
                        if isinstance(item.get("last_cycle_acknowledgment"), Mapping)
                        else {}
                    ).get("acknowledged_due_cycle_review_ticket_id")
                    or ""
                )
                == str(review_ticket_id or "")
                for item in self._snn_sleep_plasticity_review_scheduler_installations
                if isinstance(item, Mapping)
            ):
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment ticket was already consumed.")
            review_parameters = dict(
                installation.get("scheduler_review_parameters") or {}
            )
            review_interval_seconds = float(
                review_parameters.get("max_review_interval_seconds", 0.0) or 0.0
            )
            if review_interval_seconds < 60.0 or review_interval_seconds > 3600.0:
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires bounded review cadence.")
            try:
                previous_next_review_due_at = datetime.fromisoformat(
                    str(installation.get("next_review_due_at") or "")
                )
            except ValueError as exc:
                raise ValueError("SNN sleep plasticity review scheduler cycle acknowledgment requires a valid due timestamp.") from exc
            next_review_due_at = previous_next_review_due_at + timedelta(
                seconds=review_interval_seconds
            )
            acknowledged_cycle_count = int(
                installation.get("acknowledged_cycle_count", 0) or 0
            ) + 1
            acknowledgment_material = {
                "scheduler_installation_id": installation.get(
                    "scheduler_installation_id"
                ),
                "acknowledged_cycle_count": acknowledged_cycle_count,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "acknowledged_at": acknowledged_at.isoformat(),
                "review_scheduler_cycle_inspection_hash": expected_inspection_hash,
                "acknowledged_due_cycle_review_ticket_id": ticket.get(
                    "review_ticket_id"
                ),
                "acknowledged_due_cycle_review_ticket_hash": ticket.get(
                    "evidence_hash"
                ),
                "previous_next_review_due_at": previous_next_review_due_at.isoformat(),
                "next_review_due_at": next_review_due_at.isoformat(),
            }
            acknowledgment_hash = self._sha256_json(acknowledgment_material)
            acknowledgment = {
                "cycle_acknowledgment_id": (
                    f"snn-sleep-plasticity-review-cycle-ack-{acknowledgment_hash[:16]}"
                ),
                "cycle_acknowledgment_hash": acknowledgment_hash,
                **acknowledgment_material,
            }
            successor = deepcopy(installation)
            successor["next_review_due_at"] = next_review_due_at.isoformat()
            successor["acknowledged_cycle_count"] = acknowledged_cycle_count
            successor["last_cycle_acknowledgment"] = acknowledgment
            successor["scheduler_configuration_revision"] = int(
                installation.get("scheduler_configuration_revision", 0) or 0
            ) + 1
            successor["previous_scheduler_configuration_evidence_hash"] = (
                installation.get("evidence_hash")
            )
            installation_material = {
                "installed_state_revision": int(
                    successor.get("installed_state_revision", -1)
                ),
                "operator_id": successor.get("operator_id"),
                "confirmation": bool(successor.get("confirmation")),
                "scheduler_installation_preflight_hash": successor.get(
                    "scheduler_installation_preflight_hash"
                ),
                "scheduler_design_review_ticket_id": successor.get(
                    "scheduler_design_review_ticket_id"
                ),
                "scheduler_design_review_ticket_hash": successor.get(
                    "scheduler_design_review_ticket_hash"
                ),
                "scheduler_design_hash": successor.get("scheduler_design_hash"),
                "reviewed_sleep_plasticity_endpoint": successor.get(
                    "reviewed_sleep_plasticity_endpoint"
                ),
                "scheduler_review_parameters": review_parameters,
                "installed_at": successor.get("installed_at"),
                "next_review_due_at": successor.get("next_review_due_at"),
                "acknowledged_cycle_count": acknowledged_cycle_count,
                "last_cycle_acknowledgment": acknowledgment,
                "scheduler_configuration_revision": successor[
                    "scheduler_configuration_revision"
                ],
                "previous_scheduler_configuration_evidence_hash": successor[
                    "previous_scheduler_configuration_evidence_hash"
                ],
            }
            successor["evidence_hash"] = self._sha256_json(installation_material)
            self._snn_sleep_plasticity_review_scheduler_installations.appendleft(
                deepcopy(successor)
            )
            self._runtime_state.mark_dirty_without_revision()
            return {
                "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_cycle_acknowledgment",
                "surface": "snn_sleep_plasticity_review_scheduler_cycle_acknowledgment.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment",
                **acknowledgment,
                "scheduler_cadence_advanced": True,
                "persists_scheduler_cadence_state": True,
                "mutates_scheduler_cadence_state": True,
                "registers_os_timer": False,
                "starts_background_worker": False,
                "executes_suggested_endpoint": False,
                "records_replay_artifact": False,
                "runs_live_replay": False,
                "issues_regeneration_permit": False,
                "writes_checkpoint": False,
                "applies_plasticity": False,
                "mutates_transition_memory": False,
                "mutates_runtime_state": False,
                "promotion_gate": {
                    "status": "scheduler_review_cadence_advanced",
                    "eligible_for_endpoint_execution": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_artifact_recording": False,
                    "eligible_for_plasticity": False,
                    "next_gate": "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler",
                },
            }

    def snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(
        self,
        *,
        observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Expose a due reviewed-gate inspection candidate without executing it."""

        inspection = self.snn_sleep_plasticity_review_scheduler_cycle_inspection(
            observed_at=observed_at
        )
        cycle = (
            inspection.get("cycle_inspection")
            if isinstance(inspection.get("cycle_inspection"), Mapping)
            else {}
        )
        inspection_provenance = (
            inspection.get("provenance_evidence")
            if isinstance(inspection.get("provenance_evidence"), Mapping)
            else {}
        )
        ready = bool(
            inspection.get("ready")
            and cycle.get("scheduler_installation_id")
            and cycle.get("review_due")
            and cycle.get("reviewed_sleep_plasticity_endpoint")
            and not inspection.get("registers_os_timer")
            and not inspection.get("starts_background_worker")
            and not inspection.get("executes_suggested_endpoint")
            and not inspection.get("records_replay_artifact")
            and not inspection.get("writes_checkpoint")
            and not inspection.get("applies_plasticity")
            and not inspection.get("mutates_transition_memory")
            and not inspection.get("mutates_runtime_state")
        )
        candidate = {
            "candidate_id": (
                "snn-sleep-plasticity-review-cycle:"
                f"{str(cycle.get('scheduler_installation_id') or '')[:64]}"
                if cycle.get("scheduler_installation_id")
                else None
            ),
            "action": (
                "inspect_reviewed_sleep_plasticity_gate"
                if ready
                else "wait_for_review_scheduler_cadence"
            ),
            "scheduler_installation_id": cycle.get("scheduler_installation_id"),
            "scheduler_installation_evidence_hash": cycle.get(
                "scheduler_installation_evidence_hash"
            ),
            "reviewed_sleep_plasticity_endpoint": cycle.get(
                "reviewed_sleep_plasticity_endpoint"
            ),
            "observed_at": cycle.get("observed_at"),
            "next_review_due_at": cycle.get("next_review_due_at"),
            "review_due": bool(cycle.get("review_due")),
            "endpoint_execution_allowed": False,
            "priority_score": 1.0 if ready else 0.0,
        }
        proposal_hash = self._sha256_json(
            {
                "ready": ready,
                "source_review_scheduler_cycle_inspection_hash": (
                    inspection_provenance.get(
                        "review_scheduler_cycle_inspection_hash"
                    )
                ),
                "candidate": candidate,
            }
        )
        return {
            "artifact_kind": "terminus_snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal",
            "surface": "snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_endpoint_execution": False,
            "eligible_for_plasticity": False,
            "provenance_evidence": {
                "review_scheduler_cycle_autonomy_proposal_id": (
                    f"snn-sleep-plasticity-review-cycle-autonomy-{proposal_hash[:16]}"
                ),
                "review_scheduler_cycle_autonomy_proposal_hash": proposal_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
                "source_review_scheduler_cycle_inspection_hash": (
                    inspection_provenance.get(
                        "review_scheduler_cycle_inspection_hash"
                    )
                ),
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_review_scheduler_cycle_autonomy_proposal",
            },
            "candidate": candidate,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_reviewed_sleep_plasticity_gate_inspection"
                    if ready
                    else "waiting_for_review_cadence"
                ),
                "eligible_for_autonomy_planning": ready,
                "eligible_for_operator_reviewed_gate_inspection": ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_plasticity": False,
                "next_gate": (
                    "operator_inspect_reviewed_sleep_plasticity_gate"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler"
                ),
            },
        }

    def snn_due_cycle_bounded_replay_selection_proposal(
        self,
        *,
        consolidation_priority_queue: Mapping[str, Any],
        observed_at: datetime | None = None,
        max_candidates: int = 1,
    ) -> dict[str, Any]:
        """Nominate current replay contexts for due sleep review without replaying them."""

        cycle_proposal = (
            self.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal(
                observed_at=observed_at
            )
        )
        cycle_candidate = (
            cycle_proposal.get("candidate")
            if isinstance(cycle_proposal.get("candidate"), Mapping)
            else {}
        )
        cycle_provenance = (
            cycle_proposal.get("provenance_evidence")
            if isinstance(cycle_proposal.get("provenance_evidence"), Mapping)
            else {}
        )
        queue = dict(consolidation_priority_queue)
        queue_gate = (
            queue.get("promotion_gate")
            if isinstance(queue.get("promotion_gate"), Mapping)
            else {}
        )
        queue_source_window = (
            queue.get("source_window")
            if isinstance(queue.get("source_window"), Mapping)
            else {}
        )
        requested = max(1, min(int(max_candidates), 8))
        queue_candidates = [
            dict(item)
            for item in list(queue.get("candidates") or [])
            if isinstance(item, Mapping)
        ][:requested]
        selected: list[dict[str, Any]] = []
        for candidate in queue_candidates:
            context_id = str(candidate.get("replay_evaluation_context_id") or "")
            context = (
                self.verified_snn_replay_evaluation_context(context_id)
                if context_id
                else None
            )
            if (
                context is not None
                and str(candidate.get("replay_evaluation_context_hash") or "")
                == str(context.get("evidence_hash") or "")
                and candidate.get("eligible_for_live_replay") is False
                and candidate.get("eligible_for_artifact_recording") is False
                and candidate.get("eligible_for_structural_write") is False
            ):
                selected.append(candidate)
        required = {
            "cycle_proposal_surface_available": cycle_proposal.get("surface")
            == "snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal.v1",
            "cycle_proposal_ready": bool(cycle_proposal.get("ready")),
            "cycle_proposal_owned_by_marulho": bool(cycle_proposal.get("owned_by_marulho")),
            "cycle_proposal_advisory": bool(cycle_proposal.get("advisory")),
            "cycle_proposal_isolated": bool(cycle_proposal.get("isolated")),
            "cycle_proposal_non_executable": cycle_proposal.get("executable") is False,
            "cycle_proposal_timer_registration_blocked": cycle_proposal.get(
                "registers_os_timer"
            )
            is False,
            "cycle_proposal_background_worker_blocked": cycle_proposal.get(
                "starts_background_worker"
            )
            is False,
            "cycle_proposal_endpoint_execution_blocked": cycle_proposal.get(
                "executes_suggested_endpoint"
            )
            is False,
            "cycle_proposal_replay_recording_blocked": cycle_proposal.get(
                "records_replay_artifact"
            )
            is False,
            "cycle_proposal_plasticity_blocked": cycle_proposal.get(
                "applies_plasticity"
            )
            is False,
            "cycle_proposal_checkpoint_write_blocked": cycle_proposal.get(
                "writes_checkpoint"
            )
            is False,
            "cycle_proposal_regeneration_permit_blocked": cycle_proposal.get(
                "issues_regeneration_permit"
            )
            is False,
            "cycle_proposal_transition_memory_mutation_blocked": cycle_proposal.get(
                "mutates_transition_memory"
            )
            is False,
            "cycle_proposal_runtime_mutation_blocked": cycle_proposal.get(
                "mutates_runtime_state"
            )
            is False,
            "priority_queue_surface_available": queue.get("surface")
            == "snn_replay_consolidation_priority_queue.v1",
            "priority_queue_owned_by_marulho": bool(queue.get("owned_by_marulho")),
            "priority_queue_advisory": bool(queue.get("advisory")),
            "priority_queue_gate_ready": bool(
                queue_gate.get("eligible_for_operator_consolidation_review")
            ),
            "priority_queue_non_executable": queue.get("executable") is False,
            "priority_queue_live_replay_blocked": queue.get("eligible_for_live_replay")
            is False,
            "priority_queue_artifact_recording_blocked": queue.get(
                "eligible_for_artifact_recording"
            )
            is False,
            "priority_queue_structural_write_blocked": queue.get(
                "eligible_for_structural_write"
            )
            is False,
            "priority_queue_source_window_bounded": (
                _snn_replay_priority_source_window_bounded(queue_source_window)
            ),
            "current_revision_candidate_available": bool(selected),
        }
        ready = all(required.values())
        nominated = selected if ready else []
        selection = {
            "reviewed_sleep_plasticity_endpoint": cycle_candidate.get(
                "reviewed_sleep_plasticity_endpoint"
            ),
            "scheduler_installation_id": cycle_candidate.get(
                "scheduler_installation_id"
            ),
            "scheduler_installation_evidence_hash": cycle_candidate.get(
                "scheduler_installation_evidence_hash"
            ),
            "acknowledged_review_due_at": cycle_candidate.get("next_review_due_at"),
            "review_due": bool(cycle_candidate.get("review_due")),
            "max_candidates": requested,
            "candidate_count": len(nominated),
            "queue_source_window_policy": queue_source_window.get("policy"),
            "queue_source_context_count": int(
                queue_source_window.get("source_context_count", 0) or 0
            ),
            "queue_verified_context_count": int(
                queue_source_window.get("verified_context_count", 0) or 0
            ),
            "queue_global_candidate_scan": bool(
                queue_source_window.get("global_candidate_scan")
            ),
            "candidates": nominated,
        }
        proposal_hash = self._sha256_json(
            {
                "ready": ready,
                "source_review_scheduler_cycle_autonomy_proposal_hash": (
                    cycle_provenance.get(
                        "review_scheduler_cycle_autonomy_proposal_hash"
                    )
                ),
                "source_consolidation_priority_queue_hash": self._sha256_json(queue),
                "selection": selection,
            }
        )
        return {
            "artifact_kind": "terminus_snn_due_cycle_bounded_replay_selection_proposal",
            "surface": "snn_due_cycle_bounded_replay_selection_proposal.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_due_cycle_bounded_replay_selection_proposal",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_endpoint_execution": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_plasticity": False,
            "eligible_for_structural_write": False,
            "provenance_evidence": {
                "due_cycle_bounded_replay_selection_proposal_id": (
                    f"snn-due-cycle-bounded-replay-selection-{proposal_hash[:16]}"
                ),
                "due_cycle_bounded_replay_selection_proposal_hash": proposal_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
                "source_review_scheduler_cycle_autonomy_proposal_hash": (
                    cycle_provenance.get(
                        "review_scheduler_cycle_autonomy_proposal_hash"
                    )
                ),
                "source_consolidation_priority_queue_hash": self._sha256_json(queue),
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_due_cycle_bounded_replay_selection_proposal",
            },
            "selection": selection,
            "source_window": dict(queue_source_window),
            "promotion_gate": {
                "status": (
                    "ready_for_operator_sleep_replay_selection_inspection"
                    if ready
                    else "waiting_for_due_cycle_and_priority_evidence"
                ),
                "eligible_for_operator_sleep_replay_selection_inspection": ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_plasticity": False,
                "eligible_for_structural_write": False,
                "requires_operator_approval": ready,
                "next_gate": (
                    "operator_inspect_sleep_replay_selection"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/cycle-autonomy-proposal"
                ),
                "required_evidence": {
                    **required,
                    "priority_queue_source_window_bounded": bool(
                        required["priority_queue_source_window_bounded"]
                    ),
                },
            },
        }

    def snn_due_cycle_replay_artifact_recording_review_proposal(
        self,
        *,
        due_cycle_selection_proposal: Mapping[str, Any],
        artifact_recording_policy_proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind due-cycle context selection to artifact-recording review policy."""

        selection_proposal = dict(due_cycle_selection_proposal)
        selection = (
            selection_proposal.get("selection")
            if isinstance(selection_proposal.get("selection"), Mapping)
            else {}
        )
        selection_provenance = (
            selection_proposal.get("provenance_evidence")
            if isinstance(selection_proposal.get("provenance_evidence"), Mapping)
            else {}
        )
        selected_candidates = [
            dict(item)
            for item in list(selection.get("candidates") or [])
            if isinstance(item, Mapping)
        ]
        selected = selected_candidates[0] if len(selected_candidates) == 1 else {}
        policy_proposal = dict(artifact_recording_policy_proposal)
        policy_review = (
            policy_proposal.get("recommended_review")
            if isinstance(policy_proposal.get("recommended_review"), Mapping)
            else {}
        )
        policy_gate = (
            policy_proposal.get("promotion_gate")
            if isinstance(policy_proposal.get("promotion_gate"), Mapping)
            else {}
        )
        context_id = str(selected.get("replay_evaluation_context_id") or "")
        context = (
            self.verified_snn_replay_evaluation_context(context_id)
            if context_id
            else None
        )
        required = {
            "selection_surface_available": selection_proposal.get("surface")
            == "snn_due_cycle_bounded_replay_selection_proposal.v1",
            "selection_ready": bool(selection_proposal.get("ready")),
            "selection_owned_by_marulho": bool(selection_proposal.get("owned_by_marulho")),
            "selection_advisory": bool(selection_proposal.get("advisory")),
            "selection_isolated": bool(selection_proposal.get("isolated")),
            "selection_non_executable": selection_proposal.get("executable") is False,
            "selection_timer_registration_blocked": selection_proposal.get(
                "registers_os_timer"
            )
            is False,
            "selection_background_worker_blocked": selection_proposal.get(
                "starts_background_worker"
            )
            is False,
            "selection_endpoint_execution_blocked": selection_proposal.get(
                "executes_suggested_endpoint"
            )
            is False,
            "selection_artifact_recording_blocked": selection_proposal.get(
                "records_replay_artifact"
            )
            is False,
            "selection_live_replay_blocked": selection_proposal.get("runs_live_replay")
            is False,
            "selection_regeneration_permit_blocked": selection_proposal.get(
                "issues_regeneration_permit"
            )
            is False,
            "selection_checkpoint_write_blocked": selection_proposal.get(
                "writes_checkpoint"
            )
            is False,
            "selection_plasticity_blocked": selection_proposal.get("applies_plasticity")
            is False,
            "selection_transition_memory_mutation_blocked": selection_proposal.get(
                "mutates_transition_memory"
            )
            is False,
            "selection_runtime_mutation_blocked": selection_proposal.get(
                "mutates_runtime_state"
            )
            is False,
            "exactly_one_nominated_context": len(selected_candidates) == 1,
            "policy_surface_available": policy_proposal.get("surface")
            == "snn_replay_artifact_recording_policy_proposal.v1",
            "policy_ready": bool(policy_proposal.get("ready")),
            "policy_owned_by_marulho": bool(policy_proposal.get("owned_by_marulho")),
            "policy_advisory": bool(policy_proposal.get("advisory")),
            "policy_non_executable": policy_proposal.get("executable") is False,
            "policy_artifact_recording_blocked": policy_proposal.get(
                "eligible_for_artifact_recording"
            )
            is False,
            "policy_live_replay_blocked": policy_proposal.get("eligible_for_live_replay")
            is False,
            "policy_structural_write_blocked": policy_proposal.get(
                "eligible_for_structural_write"
            )
            is False,
            "policy_operator_review_gate_ready": bool(
                policy_gate.get("eligible_for_operator_artifact_recording_review")
            ),
            "selection_policy_context_id_match": bool(context_id)
            and context_id
            == str(policy_review.get("replay_evaluation_context_id") or ""),
            "selection_policy_context_hash_match": bool(
                selected.get("replay_evaluation_context_hash")
            )
            and str(selected.get("replay_evaluation_context_hash") or "")
            == str(policy_review.get("replay_evaluation_context_hash") or ""),
            "context_verified_current_revision": context is not None,
        }
        ready = all(required.values())
        review_target = {
            "review_action": (
                "operator_review_due_cycle_replay_artifact_recording"
                if ready
                else "wait_for_due_cycle_replay_artifact_recording_evidence"
            ),
            "replay_evaluation_context_id": context_id if ready else None,
            "replay_evaluation_context_hash": (
                selected.get("replay_evaluation_context_hash") if ready else None
            ),
            "recorded_state_revision": (
                int(context.get("recorded_state_revision", -1))
                if ready and context is not None
                else None
            ),
            "priority_score": (
                float(selected.get("priority_score", 0.0) or 0.0) if ready else 0.0
            ),
            "reason_codes": (
                [str(value) for value in list(selected.get("reason_codes") or [])]
                if ready
                else []
            ),
            "scheduler_installation_id": (
                selection.get("scheduler_installation_id") if ready else None
            ),
            "scheduler_installation_evidence_hash": (
                selection.get("scheduler_installation_evidence_hash")
                if ready
                else None
            ),
            "acknowledged_review_due_at": (
                selection.get("acknowledged_review_due_at") if ready else None
            ),
        }
        source_window = self._snn_replay_provenance_source_window(
            replay_evaluation_context_id=context_id if context_id else None,
        )
        proposal_hash = self._sha256_json(
            {
                "ready": ready,
                "source_due_cycle_bounded_replay_selection_proposal_hash": (
                    selection_provenance.get(
                        "due_cycle_bounded_replay_selection_proposal_hash"
                    )
                ),
                "source_artifact_recording_policy_proposal_hash": self._sha256_json(
                    policy_proposal
                ),
                "review_target": review_target,
                "source_window": source_window,
            }
        )
        return {
            "artifact_kind": "terminus_snn_due_cycle_replay_artifact_recording_review_proposal",
            "surface": "snn_due_cycle_replay_artifact_recording_review_proposal.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_due_cycle_replay_artifact_recording_review_proposal",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "eligible_for_endpoint_execution": False,
            "eligible_for_live_replay": False,
            "eligible_for_artifact_recording": False,
            "eligible_for_plasticity": False,
            "eligible_for_structural_write": False,
            "provenance_evidence": {
                "due_cycle_replay_artifact_recording_review_proposal_id": (
                    f"snn-due-cycle-replay-artifact-review-{proposal_hash[:16]}"
                ),
                "due_cycle_replay_artifact_recording_review_proposal_hash": proposal_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
                "source_due_cycle_bounded_replay_selection_proposal_hash": (
                    selection_provenance.get(
                        "due_cycle_bounded_replay_selection_proposal_hash"
                    )
                ),
                "source_artifact_recording_policy_proposal_hash": self._sha256_json(
                    policy_proposal
                ),
            },
            "device_evidence": {
                "tensor_execution_required": False,
                "cuda_applicable": False,
                "reason": "control_plane_due_cycle_replay_artifact_recording_review_proposal",
            },
            "review_target": review_target,
            "source_window": source_window,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_due_cycle_replay_artifact_recording_review"
                    if ready
                    else "waiting_for_due_cycle_replay_artifact_recording_evidence"
                ),
                "eligible_for_operator_due_cycle_replay_artifact_recording_review": ready,
                "eligible_for_artifact_recording": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity": False,
                "eligible_for_structural_write": False,
                "requires_operator_approval": ready,
                "next_gate": (
                    "operator_record_existing_replay_artifact_recording_review_ticket"
                    if ready
                    else "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/due-cycle-bounded-replay-selection-proposal"
                ),
                "required_evidence": required,
            },
        }

    def snn_sleep_phase_separation_proposal(
        self,
        *,
        due_cycle_selection_proposal: Mapping[str, Any],
        cycle_acknowledgment_preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Separate NREM-like replay nomination from REM-like stabilization review."""

        selection_proposal = dict(due_cycle_selection_proposal)
        selection = selection_proposal.get("selection") if isinstance(
            selection_proposal.get("selection"), Mapping
        ) else {}
        selection_gate = selection_proposal.get("promotion_gate") if isinstance(
            selection_proposal.get("promotion_gate"), Mapping
        ) else {}
        selected_candidates = [
            dict(item)
            for item in list(selection.get("candidates") or [])
            if isinstance(item, Mapping)
        ]
        selected = selected_candidates[0] if len(selected_candidates) == 1 else {}
        selected_context_id = str(selected.get("replay_evaluation_context_id") or "")
        selected_context = (
            self.verified_snn_replay_evaluation_context(selected_context_id)
            if selected_context_id
            else None
        )
        selection_provenance = selection_proposal.get("provenance_evidence") if isinstance(
            selection_proposal.get("provenance_evidence"), Mapping
        ) else {}
        acknowledgment_preflight = dict(cycle_acknowledgment_preflight or {})
        acknowledgment_gate = acknowledgment_preflight.get("promotion_gate") if isinstance(
            acknowledgment_preflight.get("promotion_gate"), Mapping
        ) else {}
        nrem_ready = bool(
            selection_proposal.get("surface")
            == "snn_due_cycle_bounded_replay_selection_proposal.v1"
            and selection_proposal.get("ready")
            and selection_proposal.get("owned_by_marulho")
            and selection_proposal.get("advisory")
            and selection_proposal.get("isolated")
            and selection_proposal.get("executable") is False
            and selection_proposal.get("executes_suggested_endpoint") is False
            and selection_proposal.get("records_replay_artifact") is False
            and selection_proposal.get("runs_live_replay") is False
            and selection_proposal.get("writes_checkpoint") is False
            and selection_proposal.get("applies_plasticity") is False
            and selection_proposal.get("mutates_transition_memory") is False
            and selection_proposal.get("mutates_runtime_state") is False
            and int(selection.get("candidate_count", 0) or 0) == 1
            and len(selected_candidates) == 1
            and selected_context is not None
            and str(selected.get("replay_evaluation_context_hash") or "")
            == str(selected_context.get("evidence_hash") or "")
            and bool(selection.get("scheduler_installation_id"))
            and bool(selection.get("scheduler_installation_evidence_hash"))
            and bool(selection.get("acknowledged_review_due_at"))
        )
        rem_ready = bool(
            acknowledgment_preflight.get("surface")
            == "snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight.v1"
            and acknowledgment_preflight.get("ready")
            and acknowledgment_preflight.get("owned_by_marulho")
            and acknowledgment_preflight.get("executable") is False
            and acknowledgment_preflight.get("records_replay_artifact") is False
            and acknowledgment_preflight.get("runs_live_replay") is False
            and acknowledgment_preflight.get("applies_plasticity") is False
            and acknowledgment_preflight.get("mutates_runtime_state") is False
        )
        proposal_hash = self._sha256_json(
            {
                "nrem_ready": nrem_ready,
                "rem_ready": rem_ready,
                "source_due_cycle_bounded_replay_selection_proposal_hash": (
                    selection_provenance.get(
                        "due_cycle_bounded_replay_selection_proposal_hash"
                    )
                ),
                "selected_context_id": selected_context_id or None,
                "selected_context_hash": selected.get(
                    "replay_evaluation_context_hash"
                ),
                "scheduler_installation_id": selection.get(
                    "scheduler_installation_id"
                ),
                "scheduler_installation_evidence_hash": selection.get(
                    "scheduler_installation_evidence_hash"
                ),
                "acknowledged_review_due_at": selection.get(
                    "acknowledged_review_due_at"
                ),
                "source_cycle_acknowledgment_preflight_hash": (
                    (acknowledgment_preflight.get("provenance_evidence") or {}).get(
                        "cycle_acknowledgment_preflight_hash"
                    )
                    if isinstance(acknowledgment_preflight.get("provenance_evidence"), Mapping)
                    else None
                ),
            }
        )
        return {
            "artifact_kind": "terminus_snn_sleep_phase_separation_proposal",
            "surface": "snn_sleep_phase_separation_proposal.v1",
            "available": True,
            "ready": nrem_ready or rem_ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_sleep_phase_separation_proposal",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "provenance_evidence": {
                "sleep_phase_separation_proposal_id": (
                    f"snn-sleep-phase-separation-{proposal_hash[:16]}"
                ),
                "sleep_phase_separation_proposal_hash": proposal_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
            },
            "nrem_like_replay_nomination": {
                "phase": "nrem_like_replay_nomination",
                "ready": nrem_ready,
                "candidate_count": int(selection.get("candidate_count", 0) or 0),
                "scheduler_installation_id": selection.get("scheduler_installation_id"),
                "acknowledged_review_due_at": selection.get(
                    "acknowledged_review_due_at"
                ),
                "next_gate": selection_gate.get("next_gate"),
                "eligible_for_operator_replay_context_inspection": nrem_ready,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_plasticity": False,
            },
            "rem_like_stabilization_review": {
                "phase": "rem_like_stabilization_review",
                "ready": rem_ready,
                "next_gate": acknowledgment_gate.get("next_gate"),
                "eligible_for_operator_cycle_acknowledgment": bool(
                    acknowledgment_gate.get("eligible_for_operator_cycle_acknowledgment")
                ),
                "eligible_for_homeostatic_stabilization_review": rem_ready,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_plasticity": False,
                "status": (
                    "ready_for_rem_like_stabilization_review"
                    if rem_ready
                    else "waiting_for_due_cycle_review_ticket"
                ),
            },
            "promotion_gate": {
                "status": (
                    "ready_for_phase_specific_sleep_review"
                    if nrem_ready or rem_ready
                    else "waiting_for_due_cycle_phase_evidence"
                ),
                "eligible_for_nrem_like_replay_nomination_review": nrem_ready,
                "eligible_for_rem_like_stabilization_review": rem_ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_live_replay": False,
                "eligible_for_artifact_recording": False,
                "eligible_for_plasticity": False,
            },
        }

    def snn_rem_like_homeostatic_stabilization_preflight(
        self,
        *,
        sleep_phase_separation_proposal: Mapping[str, Any],
        transition_memory_state: Mapping[str, Any],
        maintenance_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare REM-like stabilization review without applying maintenance."""

        phase = dict(sleep_phase_separation_proposal)
        rem_review = (
            phase.get("rem_like_stabilization_review")
            if isinstance(phase.get("rem_like_stabilization_review"), Mapping)
            else {}
        )
        nrem_review = (
            phase.get("nrem_like_replay_nomination")
            if isinstance(phase.get("nrem_like_replay_nomination"), Mapping)
            else {}
        )
        state = dict(transition_memory_state)
        policy = dict(maintenance_policy or {})
        sparse_transition_weight_count = max(
            0,
            int(state.get("sparse_transition_weight_count", 0) or 0),
        )
        homeostatic_maintenance_count = max(
            0,
            int(state.get("homeostatic_maintenance_count", 0) or 0),
        )
        regeneration_count = max(0, int(state.get("regeneration_count", 0) or 0))
        try:
            decay_factor = float(policy.get("decay_factor", 0.98))
            prune_below = float(policy.get("prune_below", 0.005))
            max_outgoing_row_mass = float(policy.get("max_outgoing_row_mass", 1.0))
        except (TypeError, ValueError):
            decay_factor = float("nan")
            prune_below = float("nan")
            max_outgoing_row_mass = float("nan")
        maintenance_parameters_bounded = (
            0.0 < decay_factor <= 1.0
            and 0.0 <= prune_below <= 0.25
            and 0.0 < max_outgoing_row_mass <= 4.0
        )
        post_growth_maintenance_due = bool(
            sparse_transition_weight_count > 0
            and regeneration_count > homeostatic_maintenance_count
        )
        required = {
            "phase_surface_available": phase.get("surface")
            == "snn_sleep_phase_separation_proposal.v1",
            "phase_owned_by_marulho": bool(phase.get("owned_by_marulho")),
            "phase_advisory": bool(phase.get("advisory")),
            "phase_isolated": bool(phase.get("isolated")),
            "phase_non_executable": phase.get("executable") is False,
            "rem_like_stabilization_review_ready": bool(rem_review.get("ready")),
            "nrem_like_replay_nomination_reviewed": bool(nrem_review.get("ready")),
            "transition_memory_present": sparse_transition_weight_count > 0,
            "post_growth_homeostatic_maintenance_due": post_growth_maintenance_due,
            "maintenance_parameters_bounded": maintenance_parameters_bounded,
            "endpoint_execution_blocked": phase.get("executes_suggested_endpoint") is False,
            "live_replay_blocked": phase.get("runs_live_replay") is False,
            "artifact_recording_blocked": phase.get("records_replay_artifact") is False,
            "checkpoint_write_blocked": phase.get("writes_checkpoint") is False,
            "plasticity_blocked": phase.get("applies_plasticity") is False,
            "transition_memory_mutation_blocked": phase.get("mutates_transition_memory")
            is False,
            "runtime_mutation_blocked": phase.get("mutates_runtime_state") is False,
        }
        ready = all(required.values())
        review_plan = {
            "review_action": (
                "operator_review_transition_memory_homeostatic_maintenance"
                if ready
                else "wait_for_rem_like_stabilization_evidence"
            ),
            "suggested_endpoint": (
                "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance"
                if ready
                else None
            ),
            "expected_state_revision": int(self._runtime_state.state_revision),
            "decay_factor": decay_factor if maintenance_parameters_bounded else None,
            "prune_below": prune_below if maintenance_parameters_bounded else None,
            "max_outgoing_row_mass": (
                max_outgoing_row_mass if maintenance_parameters_bounded else None
            ),
            "requires_operator_confirmation": True,
        }
        preflight_hash = self._sha256_json(
            {
                "ready": ready,
                "phase_hash": (
                    (phase.get("provenance_evidence") or {}).get(
                        "sleep_phase_separation_proposal_hash"
                    )
                    if isinstance(phase.get("provenance_evidence"), Mapping)
                    else None
                ),
                "transition_memory": {
                    "sparse_transition_weight_count": sparse_transition_weight_count,
                    "homeostatic_maintenance_count": homeostatic_maintenance_count,
                    "regeneration_count": regeneration_count,
                },
                "runtime_state_revision": int(self._runtime_state.state_revision),
                "endpoint_identity": (
                    "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance"
                ),
                "review_plan": review_plan,
            }
        )
        return {
            "artifact_kind": "terminus_snn_rem_like_homeostatic_stabilization_preflight",
            "surface": "snn_rem_like_homeostatic_stabilization_preflight.v1",
            "available": True,
            "ready": ready,
            "owned_by_marulho": True,
            "source": "replay_controller.snn_rem_like_homeostatic_stabilization_preflight",
            "advisory": True,
            "isolated": True,
            "executable": False,
            "registers_os_timer": False,
            "starts_background_worker": False,
            "executes_suggested_endpoint": False,
            "records_replay_artifact": False,
            "runs_live_replay": False,
            "issues_regeneration_permit": False,
            "writes_checkpoint": False,
            "applies_plasticity": False,
            "mutates_transition_memory": False,
            "mutates_runtime_state": False,
            "provenance_evidence": {
                "rem_like_homeostatic_stabilization_preflight_id": (
                    f"snn-rem-like-homeostatic-stabilization-{preflight_hash[:16]}"
                ),
                "rem_like_homeostatic_stabilization_preflight_hash": preflight_hash,
                "hash_algorithm": "sha256",
                "canonicalization": "json-sort-keys-compact-v1",
            },
            "transition_memory_review": {
                "sparse_transition_weight_count": sparse_transition_weight_count,
                "homeostatic_maintenance_count": homeostatic_maintenance_count,
                "regeneration_count": regeneration_count,
                "post_growth_homeostatic_maintenance_due": post_growth_maintenance_due,
            },
            "review_plan": review_plan,
            "promotion_gate": {
                "status": (
                    "ready_for_operator_homeostatic_maintenance_review"
                    if ready
                    else "waiting_for_rem_like_stabilization_evidence"
                ),
                "eligible_for_operator_homeostatic_maintenance_review": ready,
                "eligible_for_endpoint_execution": False,
                "eligible_for_checkpoint_write": False,
                "eligible_for_plasticity": False,
                "required_evidence": required,
            },
        }

    @property
    def snn_transition_memory_replay_artifacts(self) -> deque[dict[str, Any]]:
        return self._snn_transition_memory_replay_artifacts

    @snn_transition_memory_replay_artifacts.setter
    def snn_transition_memory_replay_artifacts(self, artifacts: Sequence[Mapping[str, Any]]) -> None:
        self.load_snn_transition_memory_replay_artifacts(artifacts)

    def load_snn_transition_memory_replay_artifacts(
        self,
        artifacts: Sequence[Mapping[str, Any]],
    ) -> None:
        normalized = [
            item
            for item in (
                self._normalize_evaluated_snn_transition_memory_replay_artifact(raw_item)
                for raw_item in artifacts
                if isinstance(raw_item, Mapping)
            )
            if item is not None
        ]
        self._snn_transition_memory_replay_artifacts.clear()
        self._snn_transition_memory_replay_artifacts.extend(
            normalized[:DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS]
        )

    def _record_evaluated_snn_transition_memory_replay_artifact(
        self,
        *,
        mismatch_report: Mapping[str, Any],
        pressure_report: Mapping[str, Any],
        replay_window: Sequence[Mapping[str, Any]],
        operator_id: str,
        confirmation: bool,
        artifact_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the evaluated internal-ledger-backed artifact only."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        mismatch = dict(mismatch_report)
        pressure = dict(pressure_report)
        window = [dict(item) for item in replay_window if isinstance(item, Mapping)]
        metadata = dict(artifact_metadata or {})
        source_window = (
            dict(metadata.get("source_window"))
            if isinstance(metadata.get("source_window"), Mapping)
            else {}
        )
        readout_evidence_source_window = (
            dict(metadata.get("readout_evidence_source_window"))
            if isinstance(metadata.get("readout_evidence_source_window"), Mapping)
            else {}
        )
        replay_priority_source_window = (
            dict(metadata.get("replay_priority_source_window"))
            if isinstance(metadata.get("replay_priority_source_window"), Mapping)
            else {}
        )
        error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
        mismatch_score = max(0.0, min(1.0, float(error.get("mismatch_score", 0.0) or 0.0)))
        pressure_score = max(0.0, min(1.0, float(pressure.get("pressure_score", mismatch_score) or 0.0)))
        if not confirmation:
            raise ValueError("SNN transition-memory replay artifact confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN transition-memory replay artifact operator_id is required.")
        if not mismatch.get("available") or mismatch_score < 0.66:
            raise ValueError("SNN transition-memory replay artifact requires high mismatch evidence.")
        if not pressure.get("available"):
            raise ValueError("SNN transition-memory replay artifact requires plasticity pressure evidence.")
        if not window or not all(bool(item.get("grounded")) for item in window):
            raise ValueError("SNN transition-memory replay artifact requires a grounded replay window.")
        if (
            metadata.get("internal_ledger_backed") is not True
            or not str(metadata.get("artifact_proposal_hash") or "")
            or not str(metadata.get("replay_evaluation_context_id") or "")
            or not str(metadata.get("replay_evaluation_context_hash") or "")
            or not str(metadata.get("review_ticket_id") or "")
            or not str(metadata.get("review_ticket_hash") or "")
            or not source_window
            or not _known_readout_evidence_source_window_bounded(
                readout_evidence_source_window
            )
            or not _snn_readout_replay_priority_source_window_bounded(
                replay_priority_source_window
            )
        ):
            raise ValueError(
                "SNN transition-memory replay artifacts require evaluated internal-ledger source-window evidence."
            )
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            readout_evidence_hashes = [
                str(value)
                for value in list(metadata.get("readout_evidence_hashes") or [])
                if str(value)
            ][:64]
            if not readout_evidence_hashes:
                readout_evidence_hashes = [
                    str(item.get("readout_evidence_hash") or "")
                    for item in window
                    if str(item.get("readout_evidence_hash") or "")
                ][:64]
            material = {
                "recorded_state_revision": recorded_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "mismatch_hash": self._sha256_json(mismatch),
                "mismatch_score": mismatch_score,
                "pressure_hash": self._sha256_json(pressure),
                "pressure_score": pressure_score,
                "replay_window_hash": self._sha256_json(window),
                "replay_window_size": len(window),
                "internal_ledger_backed": bool(metadata.get("internal_ledger_backed")),
                "artifact_proposal_hash": metadata.get("artifact_proposal_hash"),
                "replay_evaluation_context_id": metadata.get("replay_evaluation_context_id"),
                "replay_evaluation_context_hash": metadata.get("replay_evaluation_context_hash"),
                "review_ticket_id": metadata.get("review_ticket_id"),
                "review_ticket_hash": metadata.get("review_ticket_hash"),
                "readout_evidence_hashes": readout_evidence_hashes,
            }
            source_metadata_hash = metadata.get("source_metadata_hash")
            emission_lineage = (
                dict(metadata.get("emission_lineage"))
                if isinstance(metadata.get("emission_lineage"), Mapping)
                else {}
            )
            if source_metadata_hash or emission_lineage:
                material["source_metadata_hash"] = source_metadata_hash
                material["emission_lineage"] = emission_lineage
            material["source_window_hash"] = self._sha256_json(source_window)
            material["readout_evidence_source_window_hash"] = self._sha256_json(
                readout_evidence_source_window
            )
            material["replay_priority_source_window_hash"] = self._sha256_json(
                replay_priority_source_window
            )
            evidence_hash = self._sha256_json(material)
            artifact = {
                "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
                "surface": "snn_transition_memory_replay_artifact.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.snn_transition_memory_replay_artifact",
                "replay_artifact_id": f"snn-transition-replay-{evidence_hash[:16]}-{uuid4()}",
                "replay_window_id": f"replay-window-{material['replay_window_hash'][:16]}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "artifact_proposal_surface": metadata.get("artifact_proposal_surface"),
                "artifact_proposal_source": metadata.get("artifact_proposal_source"),
                "raw_caller_window_recording_retired": True,
            }
            artifact["source_window"] = source_window
            artifact["readout_evidence_source_window"] = (
                readout_evidence_source_window
            )
            artifact["replay_priority_source_window"] = replay_priority_source_window
            self._snn_transition_memory_replay_artifacts.appendleft(deepcopy(artifact))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(artifact)

    def record_evaluated_snn_transition_memory_replay_artifact(
        self,
        *,
        artifact_proposal: Mapping[str, Any],
        known_readout_evidence_hashes: Sequence[str],
        known_readout_evidence_source_window: Mapping[str, Any],
        replay_evaluation_context_id: str,
        review_ticket_id: str,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Record an internal-ledger-backed SNN replay context after review."""

        proposal = dict(artifact_proposal)
        gate = proposal.get("promotion_gate") if isinstance(proposal.get("promotion_gate"), Mapping) else {}
        replay_window = [
            dict(item)
            for item in list(proposal.get("replay_window") or [])
            if isinstance(item, Mapping)
        ]
        known_hashes = {str(value) for value in known_readout_evidence_hashes if str(value)}
        known_source_window = (
            dict(known_readout_evidence_source_window)
            if isinstance(known_readout_evidence_source_window, Mapping)
            else {}
        )
        replay_priority_source_window = (
            dict(proposal.get("replay_priority_source_window"))
            if isinstance(proposal.get("replay_priority_source_window"), Mapping)
            else {}
        )
        replay_priority_source_window_hash = str(
            proposal.get("replay_priority_source_window_hash") or ""
        )
        context = self.verified_snn_replay_evaluation_context(replay_evaluation_context_id)
        ticket = self.verified_snn_replay_artifact_recording_review_ticket(
            review_ticket_id,
            replay_evaluation_context_id=replay_evaluation_context_id,
            operator_id=operator_id,
        )
        if proposal.get("surface") != "snn_transition_memory_replay_artifact_proposal.v1":
            raise ValueError("Evaluated SNN replay artifact proposal surface is required.")
        if not proposal.get("owned_by_marulho") or not proposal.get("ready"):
            raise ValueError("Evaluated SNN replay artifact proposal must be MARULHO-owned and ready.")
        if str(gate.get("status") or "") != "ready_for_operator_recording_review":
            raise ValueError("Evaluated SNN replay artifact proposal gate must be ready.")
        if (
            context is None
            or str(proposal.get("replay_evaluation_context_id") or "")
            != str(context.get("replay_evaluation_context_id") or "")
            or str(proposal.get("replay_evaluation_context_hash") or "")
            != str(context.get("evidence_hash") or "")
        ):
            raise ValueError("Evaluated SNN replay artifact proposal requires a verified server-held evaluation context.")
        if ticket is None:
            raise ValueError("Evaluated SNN replay artifact proposal requires a verified review ticket.")
        context_lineage = self._snn_replay_context_emission_lineage(
            context.get("source_metadata")
            if isinstance(context.get("source_metadata"), Mapping)
            else {}
        )
        proposal_lineage = (
            dict(proposal.get("emission_lineage"))
            if isinstance(proposal.get("emission_lineage"), Mapping)
            else {}
        )
        ticket_lineage = (
            dict(ticket.get("emission_lineage"))
            if isinstance(ticket.get("emission_lineage"), Mapping)
            else {}
        )
        if str(proposal.get("source_metadata_hash") or "") != str(
            context.get("source_metadata_hash") or ""
        ) or str(ticket.get("source_metadata_hash") or "") != str(
            context.get("source_metadata_hash") or ""
        ):
            raise ValueError(
                "Evaluated SNN replay artifact proposal requires verified replay context source lineage."
            )
        if proposal_lineage != context_lineage or ticket_lineage != context_lineage:
            raise ValueError(
                "Evaluated SNN replay artifact proposal requires verified replay context source lineage."
            )
        if not replay_window or not all(
            bool(item.get("grounded"))
            and str(item.get("readout_evidence_hash") or "") in known_hashes
            for item in replay_window
        ):
            raise ValueError("Evaluated SNN replay artifact proposal must use current internal-ledger evidence.")
        if not _known_readout_evidence_source_window_bounded(known_source_window):
            raise ValueError(
                "Evaluated SNN replay artifact proposal requires bounded current internal-ledger evidence source window."
            )
        if not _snn_readout_replay_priority_source_window_bounded(
            replay_priority_source_window
        ):
            raise ValueError(
                "Evaluated SNN replay artifact proposal requires bounded replay-priority source window."
            )
        if replay_priority_source_window_hash != self._sha256_json(
            replay_priority_source_window
        ):
            raise ValueError(
                "Evaluated SNN replay artifact proposal requires matching replay-priority source-window hash."
            )
        source_window = self._snn_replay_provenance_source_window(
            replay_evaluation_context_id=str(context["replay_evaluation_context_id"]),
            review_ticket_id=str(ticket["review_ticket_id"]),
        )
        artifact = self._record_evaluated_snn_transition_memory_replay_artifact(
            mismatch_report=(
                proposal.get("mismatch_report")
                if isinstance(proposal.get("mismatch_report"), Mapping)
                else {}
            ),
            pressure_report=(
                proposal.get("pressure_report")
                if isinstance(proposal.get("pressure_report"), Mapping)
                else {}
            ),
            replay_window=replay_window,
            operator_id=operator_id,
            confirmation=confirmation,
            artifact_metadata={
                "internal_ledger_backed": True,
                "artifact_proposal_hash": self._sha256_json(proposal),
                "artifact_proposal_surface": proposal.get("surface"),
                "artifact_proposal_source": proposal.get("source"),
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
                "review_ticket_id": ticket["review_ticket_id"],
                "review_ticket_hash": ticket["evidence_hash"],
                "source_metadata_hash": context.get("source_metadata_hash"),
                "emission_lineage": context_lineage,
                "readout_evidence_hashes": [
                    str(item.get("readout_evidence_hash") or "")
                    for item in replay_window
                    if str(item.get("readout_evidence_hash") or "")
                ],
                "source_window": source_window,
                "readout_evidence_source_window": known_source_window,
                "replay_priority_source_window": replay_priority_source_window,
            },
        )
        return deepcopy(artifact)

    def issue_regeneration_permit(
        self,
        *,
        replay_artifact_id: str,
        regeneration_design: Mapping[str, Any],
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Issue durable replay provenance for a later bounded structural write."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        design = self._normalize_regeneration_design(regeneration_design)
        if not confirmation:
            raise ValueError("Regeneration permit confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("Regeneration permit operator_id is required.")
        if not design["candidate_synapses"]:
            raise ValueError("Regeneration permit requires a bounded reviewed regeneration design.")
        with self._lock:
            issued_revision = int(self._runtime_state.state_revision)
            artifact = self._verified_snn_transition_memory_replay_artifact(
                replay_artifact_id,
                operator_id=normalized_operator_id,
                expected_revision=issued_revision,
            )
            if artifact is None:
                raise ValueError("Regeneration permit requires a verified server-owned SNN replay artifact.")
            source_window = self._snn_replay_provenance_source_window_locked(
                replay_evaluation_context_id=str(
                    artifact.get("replay_evaluation_context_id") or ""
                ),
                review_ticket_id=str(artifact.get("review_ticket_id") or ""),
                replay_artifact_id=str(artifact.get("replay_artifact_id") or ""),
            )
            material = {
                "issued_state_revision": issued_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "mismatch_hash": artifact["mismatch_hash"],
                "mismatch_score": float(artifact.get("mismatch_score", 0.0) or 0.0),
                "pressure_hash": artifact["pressure_hash"],
                "pressure_score": float(artifact.get("pressure_score", 0.0) or 0.0),
                "replay_artifact_id": artifact["replay_artifact_id"],
                "replay_artifact_hash": artifact["evidence_hash"],
                "replay_window_hash": artifact["replay_window_hash"],
                "replay_window_size": artifact["replay_window_size"],
                "readout_evidence_hashes": list(artifact.get("readout_evidence_hashes") or []),
                "regeneration_design_hash": self._sha256_json(design),
                "regeneration_design_candidate_count": len(design["candidate_synapses"]),
                "source_window_hash": self._sha256_json(source_window),
            }
            if artifact.get("source_metadata_hash") or artifact.get("emission_lineage"):
                material["source_metadata_hash"] = artifact.get("source_metadata_hash")
                material["emission_lineage"] = (
                    dict(artifact.get("emission_lineage"))
                    if isinstance(artifact.get("emission_lineage"), Mapping)
                    else {}
                )
            evidence_hash = self._sha256_json(material)
            permit = {
                "artifact_kind": "terminus_snn_language_transition_memory_regeneration_permit",
                "surface": "snn_language_transition_memory_regeneration_permit.v1",
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.regeneration_permit",
                "permit_id": f"replay-regeneration-{evidence_hash[:16]}-{uuid4()}",
                "replay_window_id": f"replay-window-{material['replay_window_hash'][:16]}",
                "evidence_hash": evidence_hash,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "issued_state_revision": issued_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                **material,
                "source_window": source_window,
            }
            self._regeneration_permits.appendleft(deepcopy(permit))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(permit)

    def verify_regeneration_permit(self, proposal: Mapping[str, Any]) -> bool:
        replay = proposal.get("replay_evidence") if isinstance(proposal.get("replay_evidence"), Mapping) else {}
        permit_id = str(replay.get("permit_id") or "")
        with self._lock:
            permit = self._regeneration_permit_index.get(permit_id.strip())
            if permit is None:
                return False
            permit = dict(permit)
            material = {
                "issued_state_revision": int(permit.get("issued_state_revision", -1)),
                "operator_id": permit.get("operator_id"),
                "confirmation": bool(permit.get("confirmation")),
                "mismatch_hash": permit.get("mismatch_hash"),
                "mismatch_score": float(permit.get("mismatch_score", 0.0) or 0.0),
                "pressure_hash": permit.get("pressure_hash"),
                "pressure_score": float(permit.get("pressure_score", 0.0) or 0.0),
                "replay_window_hash": permit.get("replay_window_hash"),
                "replay_window_size": int(permit.get("replay_window_size", 0) or 0),
                "readout_evidence_hashes": list(permit.get("readout_evidence_hashes") or []),
                "replay_artifact_id": permit.get("replay_artifact_id"),
                "replay_artifact_hash": permit.get("replay_artifact_hash"),
                "regeneration_design_hash": permit.get("regeneration_design_hash"),
                "regeneration_design_candidate_count": int(
                    permit.get("regeneration_design_candidate_count", 0) or 0
                ),
            }
            if permit.get("source_window_hash"):
                material["source_window_hash"] = permit.get("source_window_hash")
            if permit.get("source_metadata_hash") or permit.get("emission_lineage"):
                material["source_metadata_hash"] = permit.get("source_metadata_hash")
                material["emission_lineage"] = (
                    dict(permit.get("emission_lineage"))
                    if isinstance(permit.get("emission_lineage"), Mapping)
                    else {}
                )
            try:
                proposal_design = self._normalize_regeneration_design(
                    proposal.get("regeneration_design")
                    if isinstance(proposal.get("regeneration_design"), Mapping)
                    else {}
                )
            except (TypeError, ValueError):
                return False
            return bool(
                permit.get("ready")
                and permit.get("owned_by_marulho")
                and permit.get("confirmation") is True
                and int(permit.get("issued_state_revision", -1)) == int(self._runtime_state.state_revision)
                and str(permit.get("evidence_hash") or "") == self._sha256_json(material)
                and self._verified_snn_transition_memory_replay_artifact(
                    str(permit.get("replay_artifact_id") or ""),
                    mismatch_hash=str(permit.get("mismatch_hash") or ""),
                    pressure_hash=str(permit.get("pressure_hash") or ""),
                    operator_id=str(permit.get("operator_id") or ""),
                    expected_revision=int(permit.get("issued_state_revision", -1)),
                )
                is not None
                and str(permit.get("regeneration_design_hash") or "")
                == self._sha256_json(proposal_design)
                and int(permit.get("regeneration_design_candidate_count", 0) or 0)
                == len(proposal_design["candidate_synapses"])
                and dict(replay) == permit
            )

    def _verified_snn_transition_memory_replay_artifact(
        self,
        replay_artifact_id: str,
        *,
        mismatch: Mapping[str, Any] | None = None,
        pressure: Mapping[str, Any] | None = None,
        mismatch_hash: str | None = None,
        pressure_hash: str | None = None,
        operator_id: str,
        expected_revision: int,
    ) -> dict[str, Any] | None:
        artifact = self._snn_transition_memory_replay_artifact_index.get(
            str(replay_artifact_id or "").strip()
        )
        if artifact is None:
            return None
        artifact = dict(artifact)
        source_window = (
            dict(artifact.get("source_window"))
            if isinstance(artifact.get("source_window"), Mapping)
            else {}
        )
        readout_evidence_source_window = (
            dict(artifact.get("readout_evidence_source_window"))
            if isinstance(artifact.get("readout_evidence_source_window"), Mapping)
            else {}
        )
        replay_priority_source_window = (
            dict(artifact.get("replay_priority_source_window"))
            if isinstance(artifact.get("replay_priority_source_window"), Mapping)
            else {}
        )
        material = {
            "recorded_state_revision": int(artifact.get("recorded_state_revision", -1)),
            "operator_id": artifact.get("operator_id"),
            "confirmation": bool(artifact.get("confirmation")),
            "mismatch_hash": artifact.get("mismatch_hash"),
            "mismatch_score": float(artifact.get("mismatch_score", 0.0) or 0.0),
            "pressure_hash": artifact.get("pressure_hash"),
            "pressure_score": float(artifact.get("pressure_score", 0.0) or 0.0),
            "replay_window_hash": artifact.get("replay_window_hash"),
            "replay_window_size": int(artifact.get("replay_window_size", 0) or 0),
            "internal_ledger_backed": bool(artifact.get("internal_ledger_backed")),
            "artifact_proposal_hash": artifact.get("artifact_proposal_hash"),
            "replay_evaluation_context_id": artifact.get("replay_evaluation_context_id"),
            "replay_evaluation_context_hash": artifact.get("replay_evaluation_context_hash"),
            "review_ticket_id": artifact.get("review_ticket_id"),
            "review_ticket_hash": artifact.get("review_ticket_hash"),
            "readout_evidence_hashes": list(artifact.get("readout_evidence_hashes") or []),
        }
        if artifact.get("source_window_hash"):
            material["source_window_hash"] = artifact.get("source_window_hash")
        if artifact.get("readout_evidence_source_window_hash"):
            material["readout_evidence_source_window_hash"] = artifact.get(
                "readout_evidence_source_window_hash"
            )
        if artifact.get("replay_priority_source_window_hash"):
            material["replay_priority_source_window_hash"] = artifact.get(
                "replay_priority_source_window_hash"
            )
        if artifact.get("source_metadata_hash") or artifact.get("emission_lineage"):
            material["source_metadata_hash"] = artifact.get("source_metadata_hash")
            material["emission_lineage"] = (
                dict(artifact.get("emission_lineage"))
                if isinstance(artifact.get("emission_lineage"), Mapping)
                else {}
            )
        expected_mismatch_hash = mismatch_hash or (
            self._sha256_json(dict(mismatch)) if mismatch is not None else str(artifact.get("mismatch_hash") or "")
        )
        expected_pressure_hash = pressure_hash or (
            self._sha256_json(dict(pressure)) if pressure is not None else str(artifact.get("pressure_hash") or "")
        )
        context = self.verified_snn_replay_evaluation_context(
            str(artifact.get("replay_evaluation_context_id") or "")
        )
        ticket = self.verified_snn_replay_artifact_recording_review_ticket(
            str(artifact.get("review_ticket_id") or ""),
            replay_evaluation_context_id=str(artifact.get("replay_evaluation_context_id") or ""),
            operator_id=operator_id,
        )
        context_lineage = (
            self._snn_replay_context_emission_lineage(
                context.get("source_metadata")
                if isinstance(context, Mapping)
                and isinstance(context.get("source_metadata"), Mapping)
                else {}
            )
            if context is not None
            else {}
        )
        artifact_lineage = (
            dict(artifact.get("emission_lineage"))
            if isinstance(artifact.get("emission_lineage"), Mapping)
            else {}
        )
        return artifact if bool(
            artifact.get("ready")
            and artifact.get("owned_by_marulho")
            and artifact.get("confirmation") is True
            and artifact.get("internal_ledger_backed") is True
            and bool(str(artifact.get("artifact_proposal_hash") or ""))
            and bool(list(artifact.get("readout_evidence_hashes") or []))
            and str(artifact.get("source_window_hash") or "")
            == self._sha256_json(source_window)
            and str(artifact.get("readout_evidence_source_window_hash") or "")
            == self._sha256_json(readout_evidence_source_window)
            and str(artifact.get("replay_priority_source_window_hash") or "")
            == self._sha256_json(replay_priority_source_window)
            and _known_readout_evidence_source_window_bounded(
                readout_evidence_source_window
            )
            and _snn_readout_replay_priority_source_window_bounded(
                replay_priority_source_window
            )
            and context is not None
            and ticket is not None
            and str(artifact.get("review_ticket_hash") or "") == str(ticket.get("evidence_hash") or "")
            and str(artifact.get("replay_evaluation_context_hash") or "")
            == str(context.get("evidence_hash") or "")
            and str(artifact.get("source_metadata_hash") or "")
            == str(context.get("source_metadata_hash") or "")
            and str(artifact.get("source_metadata_hash") or "")
            == str(ticket.get("source_metadata_hash") or "")
            and artifact_lineage == context_lineage
            and artifact_lineage
            == (
                dict(ticket.get("emission_lineage"))
                if isinstance(ticket.get("emission_lineage"), Mapping)
                else {}
            )
            and str(artifact.get("mismatch_hash") or "") == str(context.get("mismatch_hash") or "")
            and str(artifact.get("pressure_hash") or "") == str(context.get("pressure_hash") or "")
            and int(artifact.get("recorded_state_revision", -1)) == int(expected_revision)
            and str(artifact.get("operator_id") or "") == str(operator_id or "")
            and str(artifact.get("mismatch_hash") or "") == expected_mismatch_hash
            and str(artifact.get("pressure_hash") or "") == expected_pressure_hash
            and str(artifact.get("evidence_hash") or "") == self._sha256_json(material)
        ) else None

    @staticmethod
    def _normalize_regeneration_design(value: Mapping[str, Any]) -> dict[str, Any]:
        design = dict(value)
        radius = int(design.get("locality_radius", 0) or 0)
        initial_weight = float(design.get("initial_weight", 0.0) or 0.0)
        max_new_synapses = int(design.get("max_new_synapses", 0) or 0)
        mismatch_score = float(design.get("mismatch_score", 0.0) or 0.0)
        candidate_source_window_surface = (
            "bounded_snn_replay_controller_regeneration_design_candidate_window.v1"
        )
        raw_candidates, candidate_source_window = bounded_application_synapse_window(
            design.get("candidate_synapses"),
            source="service.replay_runtime.regeneration_design_candidate_synapses",
            surface=candidate_source_window_surface,
            field_name="regeneration_design.candidate_synapses",
        )
        if bool(candidate_source_window.get("source_payload_truncated")):
            raise ValueError("Regeneration design candidate count must be bounded.")
        if int(candidate_source_window.get("source_mapping_count", 0) or 0) != int(
            candidate_source_window.get("source_window_count", 0) or 0
        ):
            raise ValueError("Regeneration design candidates must be mappings.")
        candidates = []
        for item in raw_candidates:
            pre_index = int(item.get("pre_index", -1))
            post_index = int(item.get("post_index", -1))
            weight = float(item.get("initial_weight", 0.0) or 0.0)
            distance = abs(post_index - pre_index)
            if not 0 <= pre_index < 64 or not 0 <= post_index < 64:
                raise ValueError("Regeneration design indices must be canonical language-neuron indices.")
            if not 0.0 < weight <= 0.25:
                raise ValueError("Regeneration design weight must be bounded.")
            if distance > radius:
                raise ValueError("Regeneration design candidate must stay inside locality radius.")
            candidates.append(
                {
                    "pre_index": pre_index,
                    "post_index": post_index,
                    "synapse": f"{pre_index}:{post_index}",
                    "initial_weight": weight,
                    "locality_distance": distance,
                }
            )
        candidates.sort(key=lambda item: (item["pre_index"], item["post_index"]))
        if not 1 <= radius <= 8:
            raise ValueError("Regeneration design locality radius must be bounded.")
        if not 0.0 < initial_weight <= 0.25:
            raise ValueError("Regeneration design initial weight must be bounded.")
        if not 1 <= max_new_synapses <= 32 or len(candidates) > max_new_synapses:
            raise ValueError("Regeneration design candidate count must be bounded.")
        return {
            "locality_radius": radius,
            "initial_weight": initial_weight,
            "max_new_synapses": max_new_synapses,
            "mismatch_score": mismatch_score,
            "candidate_count": len(candidates),
            "candidate_synapses": candidates,
            "candidate_source_window": dict(candidate_source_window),
        }

    @staticmethod
    def _sha256_json(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def replay_plan_status(self, *, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            living_loop = self._living_loop_snapshot_locked()
            return build_replay_plan(living_loop, limit=limit).to_payload()

    def replay_sample(
        self,
        *,
        mode: str = "sample",
        candidate_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        operator_id: str,
        operator_note: str | None = None,
        confirmation: bool = False,
        limit: int | None = None,
        count: int | None = None,
        alpha: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_action_text(mode).lower()
        if normalized_mode not in {"dry_run", "sample"}:
            raise ValueError(f"Unsupported replay sample mode: {normalized_mode or '<empty>'}")
        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        if not normalized_operator_id:
            raise ValueError("Replay sample operator_id is required.")
        if not confirmation:
            raise ValueError("Replay sample confirmation=true is required for operator-gated audit sampling.")
        requested_candidate_id = self._normalize_feedback_text(candidate_id or "", max_chars=160) or None
        guard_target_type = self._normalize_action_text(target_type or "").lower() or None
        guard_target_id = self._normalize_feedback_text(target_id or "", max_chars=160) or None
        try:
            requested_count = int(count if count is not None else (limit if limit is not None else 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("Replay sample count/limit must be numeric.") from exc
        requested_count = max(1, min(MAX_REPLAY_SAMPLE_LIMIT, requested_count))
        try:
            normalized_alpha = max(0.0, min(4.0, float(alpha)))
        except (TypeError, ValueError) as exc:
            raise ValueError("Replay sample alpha must be numeric.") from exc

        with self._lock:
            before = self._replay_sample_state_counts_locked()
            living_loop = self._living_loop_snapshot_locked()
            plan = build_replay_plan(living_loop, limit=MAX_RUNTIME_TRACE_EXPORT_LIMIT).to_payload()
            candidates = [dict(item) for item in plan.get("candidates", []) if isinstance(item, Mapping)]
            if requested_candidate_id:
                selected = [candidate for candidate in candidates if str(candidate.get("candidate_id", "")) == requested_candidate_id]
                if not selected:
                    raise ValueError(f"Replay candidate_id is stale or invalid: {requested_candidate_id}")
            else:
                selected = self._sample_replay_candidates(
                    candidates,
                    count=requested_count,
                    alpha=normalized_alpha,
                    seed=seed,
                )
            if not selected:
                raise ValueError("Replay sample found no current replay-plan candidates.")
            for candidate in selected:
                candidate_target_type = self._normalize_action_text(candidate.get("target_type", "")).lower()
                candidate_target_id = self._normalize_feedback_text(candidate.get("target_id", ""), max_chars=160)
                if guard_target_type and candidate_target_type != guard_target_type:
                    raise ValueError(
                        f"Replay target_type guard mismatch for {candidate.get('candidate_id')}: "
                        f"{candidate_target_type or '<empty>'} != {guard_target_type}"
                    )
                if guard_target_id and candidate_target_id != guard_target_id:
                    raise ValueError(
                        f"Replay target_id guard mismatch for {candidate.get('candidate_id')}: "
                        f"{candidate_target_id or '<empty>'} != {guard_target_id}"
                    )
            selected_candidates = [self._replay_sample_candidate_payload(candidate) for candidate in selected]
            created_at = datetime.now(timezone.utc).isoformat()
            replay_sample_id = f"replay-{normalized_mode}-{uuid4()}"
            self._runtime_state.mark_dirty_without_revision()
            after = self._replay_sample_state_counts_locked()
            safety_flags = {
                "audit_only": True,
                "operator_confirmed": True,
                "training_started": False,
                "sleep_started": False,
                "memory_verification_promoted": False,
                "feedback_posted": False,
                "digital_action_executed": False,
                "external_calls_made": False,
                "memory_mutated": False,
                "state_revision_mutated": after["state_revision"] != before["state_revision"],
                "token_count_mutated": after["token_count"] != before["token_count"],
                "action_history_mutated": after["action_history_count"] != before["action_history_count"],
                "feedback_mutated": after["feedback_count"] != before["feedback_count"],
                "not_promoted": True,
            }
            status = "recorded"
            reason = (
                "operator-gated replay sample recorded without training, memory promotion, feedback posting, "
                "digital action execution, sleep, or external calls"
            )
            record = {
                "schema_version": 1,
                "replay_sample_id": replay_sample_id,
                "created_at": created_at,
                "mode": normalized_mode,
                "status": status,
                "reason": reason,
                "endpoint": "/terminus/replay-sample",
                "operator_id": normalized_operator_id,
                "operator_note": self._normalize_feedback_text(operator_note or "", max_chars=2000),
                "requested_candidate_id": requested_candidate_id,
                "target_type": guard_target_type,
                "target_id": guard_target_id,
                "requested_count": int(requested_count),
                "alpha": float(normalized_alpha),
                "seed": seed,
                "candidate_ids": [str(candidate.get("candidate_id", "")) for candidate in candidates if str(candidate.get("candidate_id", ""))],
                "selected_candidate_ids": [
                    str(candidate.get("candidate_id", ""))
                    for candidate in selected
                    if str(candidate.get("candidate_id", ""))
                ],
                "selected_candidates": selected_candidates,
                "safety_checks": {
                    "passed": True,
                    "candidate_revalidation": "passed",
                    "target_guard": "passed" if (guard_target_type or guard_target_id) else "not_requested",
                    "operator_confirmation": "passed",
                    "bounded_count": requested_count <= MAX_REPLAY_SAMPLE_LIMIT,
                    "max_count": MAX_REPLAY_SAMPLE_LIMIT,
                    "boundaries": list(REPLAY_SAMPLE_SAFETY_BOUNDARIES),
                },
                "safety_flags": safety_flags,
                "before": before,
                "after": after,
                "plan_summary": self._replay_plan_summary(plan),
            }
            normalized_record = self._normalize_replay_sample_record(record) or record
            self._replay_sample_history.appendleft(normalized_record)
            return deepcopy(normalized_record)

    def replay_sample_history(self, *, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            count = max(1, min(DEFAULT_REPLAY_SAMPLE_HISTORY, int(limit)))
            history = [deepcopy(item) for item in islice(self._replay_sample_history, count)]
            return {
                "schema_version": 1,
                "endpoint": "/terminus/replay-sample/history",
                "count": int(len(self._replay_sample_history)),
                "limit": int(count),
                "history": history,
            }

    def _replay_sample_summary_locked(self) -> dict[str, Any]:
        retained_count = int(len(self._replay_sample_history))
        source_limit = max(1, min(DEFAULT_REPLAY_SAMPLE_HISTORY, int(REPLAY_SAMPLE_SUMMARY_SOURCE_WINDOW_LIMIT)))
        records: list[dict[str, Any]] = []
        source_window_count = 0
        for item in islice(self._replay_sample_history, source_limit):
            source_window_count += 1
            if isinstance(item, Mapping):
                records.append(dict(item))
        source_window = {
            "surface": "bounded_replay_sample_summary_source_window.v1",
            "policy": "recent_replay_sample_summary_window",
            "window_policy": "recent_replay_sample_summary_window",
            "source": "replay_controller.replay_sample_history",
            "selection_criteria": [
                "newest_replay_sample_records_first",
                "bounded_summary_before_status_or_export",
                "latest_item_exact_inside_window",
            ],
            "source_window_limit": int(source_limit),
            "source_window_count": int(source_window_count),
            "source_record_count": retained_count,
            "source_record_count_known": True,
            "source_payload_truncated": bool(retained_count > source_window_count),
            "source_truncated_count": max(0, retained_count - source_window_count),
            "summary_record_count": int(len(records)),
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_replay_text_payload_loaded": False,
            "language_reasoning": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "summary_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "memory_budget": {
                "max_replay_sample_records": int(source_limit),
                "archival_storage_device": "cpu",
            },
        }
        mode_counts: Counter[str] = Counter({"dry_run": 0, "sample": 0})
        status_counts: Counter[str] = Counter()
        selected_count = 0
        for record in records:
            mode = self._normalize_action_text(record.get("mode", "sample")).lower() or "sample"
            if mode not in {"dry_run", "sample"}:
                mode = "sample"
            status = self._normalize_feedback_text(record.get("status", "recorded"), max_chars=80) or "recorded"
            mode_counts[mode] += 1
            status_counts[status] += 1
            selected_ids = record.get("selected_candidate_ids")
            if isinstance(selected_ids, Sequence) and not isinstance(selected_ids, (str, bytes)):
                selected_count += len(selected_ids)
            else:
                selected_candidates = record.get("selected_candidates")
                if isinstance(selected_candidates, Sequence) and not isinstance(selected_candidates, (str, bytes)):
                    selected_count += len(selected_candidates)
        latest_item: dict[str, Any] | None = None
        latest_safety_flags: dict[str, Any] = {
            "audit_only": True,
            "operator_confirmed": False,
            "training_started": False,
            "sleep_started": False,
            "memory_verification_promoted": False,
            "feedback_posted": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "memory_mutated": False,
            "state_revision_mutated": False,
            "token_count_mutated": False,
            "action_history_mutated": False,
            "feedback_mutated": False,
            "not_promoted": True,
        }
        latest_selected_count = 0
        if records:
            latest = self._normalize_replay_sample_record(records[0]) or records[0]
            selected_ids = latest.get("selected_candidate_ids")
            latest_selected_count = (
                len(selected_ids)
                if isinstance(selected_ids, Sequence) and not isinstance(selected_ids, (str, bytes))
                else 0
            )
            if not latest_selected_count:
                selected_candidates = latest.get("selected_candidates")
                latest_selected_count = (
                    len(selected_candidates)
                    if isinstance(selected_candidates, Sequence) and not isinstance(selected_candidates, (str, bytes))
                    else 0
                )
            latest_safety_flags.update(
                dict(latest.get("safety_flags", {})) if isinstance(latest.get("safety_flags"), Mapping) else {}
            )
            latest_item = {
                "schema_version": latest.get("schema_version", 1),
                "replay_sample_id": latest.get("replay_sample_id"),
                "created_at": latest.get("created_at"),
                "mode": latest.get("mode"),
                "status": latest.get("status"),
                "reason": latest.get("reason"),
                "endpoint": latest.get("endpoint", "/terminus/replay-sample"),
                "operator_id": latest.get("operator_id"),
                "requested_candidate_id": latest.get("requested_candidate_id"),
                "target_type": latest.get("target_type"),
                "target_id": latest.get("target_id"),
                "requested_count": latest.get("requested_count"),
                "selected_count": latest_selected_count,
                "selected_candidate_ids": list(latest.get("selected_candidate_ids") or [])[:MAX_REPLAY_SAMPLE_LIMIT],
                "safety_checks": dict(latest.get("safety_checks", {})) if isinstance(latest.get("safety_checks"), Mapping) else {},
                "safety_flags": dict(latest_safety_flags),
                "plan_summary": self._replay_plan_summary(latest.get("plan_summary")),
            }
        summary = {
            "schema_version": 1,
            "endpoint": "/terminus/replay-sample",
            "history_endpoint": "/terminus/replay-sample/history",
            "count": retained_count,
            "history_count": retained_count,
            "source_window_count": int(source_window_count),
            "source_window_complete": bool(retained_count <= source_window_count),
            "summary_count_scope": "bounded_source_window",
            "selected_count_scope": "bounded_source_window",
            "selected_count": int(selected_count),
            "latest_selected_count": int(latest_selected_count),
            "mode_counts": dict(mode_counts),
            "status_counts": dict(status_counts),
            "latest_history_item": latest_item,
            "source_window": source_window,
            "safety_flags": dict(latest_safety_flags),
            "safety_boundaries": list(REPLAY_SAMPLE_SAFETY_BOUNDARIES),
            "audit_only": True,
            "advisory": True,
            "executable": False,
        }
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(summary))

    def _replay_sample_state_counts_locked(self) -> dict[str, int]:
        feedback_summary = self._runtime_feedback_summary_locked()
        return {
            "token_count": int(self._trainer.token_count),
            "state_revision": int(self._runtime_state.state_revision),
            "action_history_count": int(len(self._action_history)),
            "feedback_count": int(feedback_summary.get("feedback_count", 0) or 0),
        }

    def _sample_replay_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        count: int,
        alpha: float,
        seed: int | None,
    ) -> list[dict[str, Any]]:
        available = [dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)]
        selected: list[dict[str, Any]] = []
        if not available:
            return selected
        rng = random.Random(seed)
        requested = max(1, min(MAX_REPLAY_SAMPLE_LIMIT, int(count), len(available)))
        normalized_alpha = max(0.0, min(4.0, float(alpha)))
        seen_target_types: set[str] = set()
        epsilon = 1.0e-6
        while available and len(selected) < requested:
            unseen_types = {
                self._normalize_action_text(candidate.get("target_type", "")).lower()
                for candidate in available
            } - seen_target_types
            weights: list[float] = []
            for candidate in available:
                try:
                    priority_score = max(0.0, float(candidate.get("priority_score", 0.0) or 0.0))
                except (TypeError, ValueError):
                    priority_score = 0.0
                weight = (epsilon + priority_score) ** normalized_alpha
                candidate_type = self._normalize_action_text(candidate.get("target_type", "")).lower()
                if unseen_types and candidate_type in seen_target_types:
                    weight *= 0.35
                weights.append(max(epsilon, weight))
            total = sum(weights)
            threshold = rng.random() * total
            cumulative = 0.0
            chosen_index = len(available) - 1
            for index, weight in enumerate(weights):
                cumulative += weight
                if threshold <= cumulative:
                    chosen_index = index
                    break
            chosen = available.pop(chosen_index)
            selected.append(chosen)
            chosen_type = self._normalize_action_text(chosen.get("target_type", "")).lower()
            if chosen_type:
                seen_target_types.add(chosen_type)
        return selected

    def _replay_sample_candidate_payload(self, candidate: Mapping[str, Any]) -> dict[str, Any]:
        safe_candidate = self._runtime_trace_export_safe_value(dict(candidate))
        payload = dict(safe_candidate) if isinstance(safe_candidate, Mapping) else {}
        payload["safety"] = replay_candidate_safety_flags(payload)
        return payload

    def _normalize_replay_sample_record(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return None
        safe = self._runtime_trace_export_safe_value(dict(raw))
        data = dict(safe) if isinstance(safe, Mapping) else {}
        if not data:
            return None

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        def _counts(value: Any) -> dict[str, int]:
            mapping = value if isinstance(value, Mapping) else {}
            return {
                "token_count": _safe_int(mapping.get("token_count")),
                "state_revision": _safe_int(mapping.get("state_revision")),
                "action_history_count": _safe_int(mapping.get("action_history_count")),
                "feedback_count": _safe_int(mapping.get("feedback_count")),
            }

        mode = self._normalize_action_text(data.get("mode", "sample")).lower()
        if mode not in {"dry_run", "sample"}:
            mode = "sample"
        selected_candidates = [
            dict(item)
            for item in data.get("selected_candidates", [])
            if isinstance(item, Mapping)
        ] if isinstance(data.get("selected_candidates", []), Sequence) and not isinstance(data.get("selected_candidates", []), (str, bytes)) else []
        selected_ids = [
            self._normalize_feedback_text(item, max_chars=160)
            for item in data.get("selected_candidate_ids", [])
            if self._normalize_feedback_text(item, max_chars=160)
        ] if isinstance(data.get("selected_candidate_ids", []), Sequence) and not isinstance(data.get("selected_candidate_ids", []), (str, bytes)) else []
        candidate_ids = [
            self._normalize_feedback_text(item, max_chars=160)
            for item in data.get("candidate_ids", [])
            if self._normalize_feedback_text(item, max_chars=160)
        ] if isinstance(data.get("candidate_ids", []), Sequence) and not isinstance(data.get("candidate_ids", []), (str, bytes)) else []
        replay_sample_id = self._normalize_feedback_text(data.get("replay_sample_id", ""), max_chars=160) or f"replay-{mode}-{uuid4()}"
        try:
            alpha = max(0.0, min(4.0, float(data.get("alpha", 1.0))))
        except (TypeError, ValueError):
            alpha = 1.0
        seed_raw: Any = data.get("seed")
        seed_value: int | None
        if seed_raw is None:
            seed_value = None
        else:
            try:
                seed_value = int(seed_raw)
            except (TypeError, ValueError):
                seed_value = None
        return {
            "schema_version": 1,
            "replay_sample_id": replay_sample_id,
            "created_at": self._normalize_feedback_text(data.get("created_at", ""), max_chars=80) or datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "status": self._normalize_feedback_text(data.get("status", "recorded"), max_chars=80) or "recorded",
            "reason": self._normalize_feedback_text(data.get("reason", ""), max_chars=2000),
            "endpoint": self._normalize_feedback_text(data.get("endpoint", "/terminus/replay-sample"), max_chars=120) or "/terminus/replay-sample",
            "operator_id": self._normalize_feedback_text(data.get("operator_id", ""), max_chars=160),
            "operator_note": self._normalize_feedback_text(data.get("operator_note", ""), max_chars=2000),
            "requested_candidate_id": self._normalize_feedback_text(data.get("requested_candidate_id", ""), max_chars=160) or None,
            "target_type": self._normalize_feedback_text(data.get("target_type", ""), max_chars=64) or None,
            "target_id": self._normalize_feedback_text(data.get("target_id", ""), max_chars=160) or None,
            "requested_count": max(1, min(MAX_REPLAY_SAMPLE_LIMIT, _safe_int(data.get("requested_count", 1)) or 1)),
            "alpha": alpha,
            "seed": seed_value,
            "candidate_ids": candidate_ids,
            "selected_candidate_ids": selected_ids,
            "selected_candidates": selected_candidates,
            "safety_checks": dict(data.get("safety_checks", {})) if isinstance(data.get("safety_checks"), Mapping) else {},
            "safety_flags": dict(data.get("safety_flags", {})) if isinstance(data.get("safety_flags"), Mapping) else {},
            "before": _counts(data.get("before")),
            "after": _counts(data.get("after")),
            "plan_summary": dict(data.get("plan_summary", {})) if isinstance(data.get("plan_summary"), Mapping) else {},
        }
