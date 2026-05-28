from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import random
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from hecsn.service.living_loop_replay import (
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    build_replay_plan,
    replay_candidate_safety_flags,
)

DEFAULT_REPLAY_SAMPLE_HISTORY = 256
MAX_REPLAY_SAMPLE_LIMIT = 20
MAX_RUNTIME_TRACE_EXPORT_LIMIT = 50


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
        history_maxlen: int = DEFAULT_REPLAY_SAMPLE_HISTORY,
    ) -> None:
        self._dependencies = dependencies
        self._history_maxlen = max(1, int(history_maxlen))
        self._replay_sample_history: deque[dict[str, Any]] = deque(maxlen=self._history_maxlen)
        self.load_replay_sample_history(replay_sample_history or [])

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
        if normalized_mode not in {"dry_run", "sample", "execute"}:
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
                "operator-gated audit execution recorded without training, memory promotion, feedback posting, "
                "digital action execution, sleep, or external calls"
                if normalized_mode == "execute"
                else "operator-gated replay sample recorded without training, memory promotion, feedback posting, digital action execution, sleep, or external calls"
            )
            record = {
                "schema_version": 1,
                "replay_sample_id": replay_sample_id,
                "execution_id": replay_sample_id if normalized_mode == "execute" else None,
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
            history = [deepcopy(item) for item in list(self._replay_sample_history)[:count]]
            return {
                "schema_version": 1,
                "endpoint": "/terminus/replay-sample/history",
                "count": int(len(self._replay_sample_history)),
                "limit": int(count),
                "history": history,
            }

    def _replay_sample_summary_locked(self) -> dict[str, Any]:
        records = [
            dict(item)
            for item in list(self._replay_sample_history)
            if isinstance(item, Mapping)
        ]
        mode_counts: Counter[str] = Counter({"dry_run": 0, "sample": 0, "execute": 0})
        status_counts: Counter[str] = Counter()
        selected_count = 0
        for record in records:
            mode = self._normalize_action_text(record.get("mode", "sample")).lower() or "sample"
            if mode not in {"dry_run", "sample", "execute"}:
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
                "execution_id": latest.get("execution_id"),
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
            "execution_endpoint": "/terminus/replay-execute",
            "history_endpoint": "/terminus/replay-sample/history",
            "execution_history_endpoint": "/terminus/replay-execute/history",
            "count": int(len(records)),
            "history_count": int(len(records)),
            "selected_count": int(selected_count),
            "latest_selected_count": int(latest_selected_count),
            "mode_counts": dict(mode_counts),
            "status_counts": dict(status_counts),
            "latest_history_item": latest_item,
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
        if mode not in {"dry_run", "sample", "execute"}:
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
            "execution_id": self._normalize_feedback_text(data.get("execution_id", ""), max_chars=160) or None,
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
