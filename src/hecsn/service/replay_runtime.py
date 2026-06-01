from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import random
from typing import Any, Callable, Mapping, Sequence, cast
from uuid import uuid4

from hecsn.service.living_loop_replay import (
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    build_replay_plan,
    replay_candidate_safety_flags,
)

DEFAULT_REPLAY_SAMPLE_HISTORY = 256
DEFAULT_REPLAY_REGENERATION_PERMITS = 64
DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS = 64
DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS = 64
DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS = 64
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
        regeneration_permits: Sequence[Mapping[str, Any]] | None = None,
        snn_replay_evaluation_contexts: Sequence[Mapping[str, Any]] | None = None,
        snn_replay_artifact_recording_review_tickets: Sequence[Mapping[str, Any]] | None = None,
        snn_transition_memory_replay_artifacts: Sequence[Mapping[str, Any]] | None = None,
        history_maxlen: int = DEFAULT_REPLAY_SAMPLE_HISTORY,
    ) -> None:
        self._dependencies = dependencies
        self._history_maxlen = max(1, int(history_maxlen))
        self._replay_sample_history: deque[dict[str, Any]] = deque(maxlen=self._history_maxlen)
        self._regeneration_permits: deque[dict[str, Any]] = deque(maxlen=DEFAULT_REPLAY_REGENERATION_PERMITS)
        self._snn_replay_evaluation_contexts: deque[dict[str, Any]] = deque(
            maxlen=DEFAULT_SNN_REPLAY_EVALUATION_CONTEXTS
        )
        self._snn_replay_artifact_recording_review_tickets: deque[dict[str, Any]] = deque(
            maxlen=DEFAULT_SNN_REPLAY_ARTIFACT_RECORDING_REVIEW_TICKETS
        )
        self._snn_transition_memory_replay_artifacts: deque[dict[str, Any]] = deque(
            maxlen=DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS
        )
        self.load_replay_sample_history(replay_sample_history or [])
        self.load_regeneration_permits(regeneration_permits or [])
        self.load_snn_replay_evaluation_contexts(snn_replay_evaluation_contexts or [])
        self.load_snn_replay_artifact_recording_review_tickets(
            snn_replay_artifact_recording_review_tickets or []
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

    def record_snn_replay_evaluation_context(
        self,
        *,
        mismatch_report: Mapping[str, Any],
        pressure_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record server-recomputed mismatch and pressure evidence for replay review."""

        mismatch = dict(mismatch_report)
        pressure = dict(pressure_report)
        error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
        pressure_gate = (
            pressure.get("promotion_gate")
            if isinstance(pressure.get("promotion_gate"), Mapping)
            else {}
        )
        if (
            mismatch.get("surface") != "snn_language_sequence_mismatch_probe.v1"
            or not mismatch.get("available")
            or not mismatch.get("owned_by_hecsn")
            or float(error.get("mismatch_score", 0.0) or 0.0) < 0.66
        ):
            raise ValueError("SNN replay evaluation context requires server-held high mismatch evidence.")
        if (
            pressure.get("surface") != "snn_language_plasticity_pressure.v1"
            or not pressure.get("available")
            or not pressure.get("owned_by_hecsn")
            or str(pressure_gate.get("status") or "") != "ready_for_operator_review"
        ):
            raise ValueError("SNN replay evaluation context requires server-held plasticity pressure evidence.")
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            material = {
                "recorded_state_revision": recorded_revision,
                "mismatch_hash": self._sha256_json(mismatch),
                "pressure_hash": self._sha256_json(pressure),
            }
            evidence_hash = self._sha256_json(material)
            context = {
                "artifact_kind": "terminus_snn_replay_evaluation_context",
                "surface": "snn_replay_evaluation_context.v1",
                "available": True,
                "ready": True,
                "owned_by_hecsn": True,
                "source": "replay_controller.snn_replay_evaluation_context",
                "replay_evaluation_context_id": f"snn-replay-evaluation-{evidence_hash[:16]}-{uuid4()}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "mismatch_report": mismatch,
                "pressure_report": pressure,
            }
            self._snn_replay_evaluation_contexts.appendleft(deepcopy(context))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(context)

    def verified_snn_replay_evaluation_context(self, context_id: str) -> dict[str, Any] | None:
        with self._lock:
            context = next(
                (
                    dict(item)
                    for item in self._snn_replay_evaluation_contexts
                    if str(item.get("replay_evaluation_context_id") or "") == str(context_id)
                ),
                None,
            )
            if context is None:
                return None
            material = {
                "recorded_state_revision": int(context.get("recorded_state_revision", -1)),
                "mismatch_hash": context.get("mismatch_hash"),
                "pressure_hash": context.get("pressure_hash"),
            }
            return (
                deepcopy(context)
                if (
                    context.get("ready")
                    and context.get("owned_by_hecsn")
                    and int(context.get("recorded_state_revision", -1)) == int(self._runtime_state.state_revision)
                    and str(context.get("evidence_hash") or "") == self._sha256_json(material)
                    and str(context.get("mismatch_hash") or "")
                    == self._sha256_json(dict(context.get("mismatch_report") or {}))
                    and str(context.get("pressure_hash") or "")
                    == self._sha256_json(dict(context.get("pressure_report") or {}))
                )
                else None
            )

    def snn_replay_consolidation_priority_queue(
        self,
        *,
        readout_replay_priority_report: Mapping[str, Any] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Rank verified SNN replay contexts for operator consolidation review."""

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
        requested = max(0, min(int(limit), 32))
        with self._lock:
            contexts = [
                context
                for context in (
                    self.verified_snn_replay_evaluation_context(
                        str(item.get("replay_evaluation_context_id") or "")
                    )
                    for item in list(self._snn_replay_evaluation_contexts)
                    if isinstance(item, Mapping)
                )
                if context is not None
            ]
            total = max(1, len(contexts))
            candidates: list[dict[str, Any]] = []
            for index, context in enumerate(contexts):
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
                mismatch_score = max(0.0, min(1.0, float(error.get("mismatch_score", 0.0) or 0.0)))
                pressure_score = max(
                    0.0,
                    min(1.0, float(pressure_payload.get("pressure_score", mismatch_score) or 0.0)),
                )
                recency = 1.0 - min(1.0, index / max(1, total - 1)) if total > 1 else 1.0
                score = 100.0 * (
                    0.35 * mismatch_score
                    + 0.25 * pressure_score
                    + 0.20 * readout_support
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
                        "priority_score": float(score),
                        "priority_components": {
                            "prediction_error": float(mismatch_score),
                            "plasticity_pressure": float(pressure_score),
                            "readout_support": float(readout_support),
                            "recency": float(recency),
                        },
                        "reason_codes": [
                            code
                            for code, active in (
                                ("high_prediction_error", mismatch_score >= 0.66),
                                ("high_plasticity_pressure", pressure_score >= 0.66),
                                ("grounded_readout_candidates_available", grounded_support > 0.0),
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
            return {
                "artifact_kind": "terminus_snn_replay_consolidation_priority_queue",
                "surface": "snn_replay_consolidation_priority_queue.v1",
                "available": bool(selected),
                "ready": ready,
                "owned_by_hecsn": True,
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
                        "current_revision_contexts_verified": len(selected) == len(contexts)
                        if contexts
                        else False,
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
            "priority_queue_owned_by_hecsn": bool(queue.get("owned_by_hecsn")),
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
        return {
            "artifact_kind": "terminus_snn_replay_artifact_recording_policy_proposal",
            "surface": "snn_replay_artifact_recording_policy_proposal.v1",
            "available": bool(candidates),
            "ready": ready,
            "owned_by_hecsn": True,
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
        if not confirmation:
            raise ValueError("SNN replay artifact recording review ticket confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN replay artifact recording review ticket operator_id is required.")
        if proposal.get("surface") != "snn_replay_artifact_recording_policy_proposal.v1":
            raise ValueError("SNN replay artifact recording review ticket requires policy proposal surface.")
        if not proposal.get("owned_by_hecsn") or not proposal.get("ready") or not proposal.get("recommended"):
            raise ValueError("SNN replay artifact recording review ticket requires ready HECSN policy proposal.")
        if not bool(gate.get("eligible_for_operator_artifact_recording_review")):
            raise ValueError("SNN replay artifact recording review ticket requires operator review gate.")
        if context is None or str(review.get("replay_evaluation_context_hash") or "") != str(
            context.get("evidence_hash") or ""
        ):
            raise ValueError("SNN replay artifact recording review ticket requires a verified replay context.")
        with self._lock:
            recorded_revision = int(self._runtime_state.state_revision)
            material = {
                "recorded_state_revision": recorded_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "policy_proposal_hash": self._sha256_json(proposal),
                "replay_evaluation_context_id": context["replay_evaluation_context_id"],
                "replay_evaluation_context_hash": context["evidence_hash"],
            }
            evidence_hash = self._sha256_json(material)
            ticket = {
                "artifact_kind": "terminus_snn_replay_artifact_recording_review_ticket",
                "surface": "snn_replay_artifact_recording_review_ticket.v1",
                "available": True,
                "ready": True,
                "owned_by_hecsn": True,
                "source": "replay_controller.snn_replay_artifact_recording_review_ticket",
                "review_ticket_id": f"snn-replay-artifact-review-{evidence_hash[:16]}-{uuid4()}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
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
            ticket = next(
                (
                    dict(item)
                    for item in self._snn_replay_artifact_recording_review_tickets
                    if str(item.get("review_ticket_id") or "") == str(review_ticket_id or "")
                ),
                None,
            )
            if ticket is None:
                return None
            material = {
                "recorded_state_revision": int(ticket.get("recorded_state_revision", -1)),
                "operator_id": ticket.get("operator_id"),
                "confirmation": bool(ticket.get("confirmation")),
                "policy_proposal_hash": ticket.get("policy_proposal_hash"),
                "replay_evaluation_context_id": ticket.get("replay_evaluation_context_id"),
                "replay_evaluation_context_hash": ticket.get("replay_evaluation_context_hash"),
            }
            context = self.verified_snn_replay_evaluation_context(
                str(ticket.get("replay_evaluation_context_id") or "")
            )
            expected_context_id = str(replay_evaluation_context_id or ticket.get("replay_evaluation_context_id") or "")
            return (
                deepcopy(ticket)
                if (
                    ticket.get("ready")
                    and ticket.get("owned_by_hecsn")
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
        normalized = [dict(item) for item in artifacts if isinstance(item, Mapping)]
        self._snn_transition_memory_replay_artifacts.clear()
        self._snn_transition_memory_replay_artifacts.extend(
            normalized[:DEFAULT_SNN_TRANSITION_MEMORY_REPLAY_ARTIFACTS]
        )

    def record_snn_transition_memory_replay_artifact(
        self,
        *,
        mismatch_report: Mapping[str, Any],
        pressure_report: Mapping[str, Any],
        replay_window: Sequence[Mapping[str, Any]],
        operator_id: str,
        confirmation: bool,
        artifact_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a durable server-owned SNN replay context for structural review."""

        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        mismatch = dict(mismatch_report)
        pressure = dict(pressure_report)
        window = [dict(item) for item in replay_window if isinstance(item, Mapping)]
        metadata = dict(artifact_metadata or {})
        error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
        if not confirmation:
            raise ValueError("SNN transition-memory replay artifact confirmation=true is required.")
        if not normalized_operator_id:
            raise ValueError("SNN transition-memory replay artifact operator_id is required.")
        if not mismatch.get("available") or float(error.get("mismatch_score", 0.0) or 0.0) < 0.66:
            raise ValueError("SNN transition-memory replay artifact requires high mismatch evidence.")
        if not pressure.get("available"):
            raise ValueError("SNN transition-memory replay artifact requires plasticity pressure evidence.")
        if not window or not all(bool(item.get("grounded")) for item in window):
            raise ValueError("SNN transition-memory replay artifact requires a grounded replay window.")
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
                "pressure_hash": self._sha256_json(pressure),
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
            evidence_hash = self._sha256_json(material)
            artifact = {
                "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
                "surface": "snn_transition_memory_replay_artifact.v1",
                "available": True,
                "ready": True,
                "owned_by_hecsn": True,
                "source": "replay_controller.snn_transition_memory_replay_artifact",
                "replay_artifact_id": f"snn-transition-replay-{evidence_hash[:16]}-{uuid4()}",
                "replay_window_id": f"replay-window-{material['replay_window_hash'][:16]}",
                "evidence_hash": evidence_hash,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **material,
                "artifact_proposal_surface": metadata.get("artifact_proposal_surface"),
                "artifact_proposal_source": metadata.get("artifact_proposal_source"),
            }
            self._snn_transition_memory_replay_artifacts.appendleft(deepcopy(artifact))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(artifact)

    def record_evaluated_snn_transition_memory_replay_artifact(
        self,
        *,
        artifact_proposal: Mapping[str, Any],
        known_readout_evidence_hashes: Sequence[str],
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
        context = self.verified_snn_replay_evaluation_context(replay_evaluation_context_id)
        ticket = self.verified_snn_replay_artifact_recording_review_ticket(
            review_ticket_id,
            replay_evaluation_context_id=replay_evaluation_context_id,
            operator_id=operator_id,
        )
        if proposal.get("surface") != "snn_transition_memory_replay_artifact_proposal.v1":
            raise ValueError("Evaluated SNN replay artifact proposal surface is required.")
        if not proposal.get("owned_by_hecsn") or not proposal.get("ready"):
            raise ValueError("Evaluated SNN replay artifact proposal must be HECSN-owned and ready.")
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
        if not replay_window or not all(
            bool(item.get("grounded"))
            and str(item.get("readout_evidence_hash") or "") in known_hashes
            for item in replay_window
        ):
            raise ValueError("Evaluated SNN replay artifact proposal must use current internal-ledger evidence.")
        artifact = self.record_snn_transition_memory_replay_artifact(
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
                "readout_evidence_hashes": [
                    str(item.get("readout_evidence_hash") or "")
                    for item in replay_window
                    if str(item.get("readout_evidence_hash") or "")
                ],
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
            material = {
                "issued_state_revision": issued_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                "mismatch_hash": artifact["mismatch_hash"],
                "pressure_hash": artifact["pressure_hash"],
                "replay_artifact_id": artifact["replay_artifact_id"],
                "replay_artifact_hash": artifact["evidence_hash"],
                "replay_window_hash": artifact["replay_window_hash"],
                "replay_window_size": artifact["replay_window_size"],
                "readout_evidence_hashes": list(artifact.get("readout_evidence_hashes") or []),
                "regeneration_design_hash": self._sha256_json(design),
                "regeneration_design_candidate_count": len(design["candidate_synapses"]),
            }
            evidence_hash = self._sha256_json(material)
            permit = {
                "artifact_kind": "terminus_snn_language_transition_memory_regeneration_permit",
                "surface": "snn_language_transition_memory_regeneration_permit.v1",
                "available": True,
                "ready": True,
                "owned_by_hecsn": True,
                "source": "replay_controller.regeneration_permit",
                "permit_id": f"replay-regeneration-{evidence_hash[:16]}-{uuid4()}",
                "replay_window_id": f"replay-window-{material['replay_window_hash'][:16]}",
                "evidence_hash": evidence_hash,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "issued_state_revision": issued_revision,
                "operator_id": normalized_operator_id,
                "confirmation": True,
                **material,
            }
            self._regeneration_permits.appendleft(deepcopy(permit))
            self._runtime_state.mark_dirty_without_revision()
            return deepcopy(permit)

    def verify_regeneration_permit(self, proposal: Mapping[str, Any]) -> bool:
        replay = proposal.get("replay_evidence") if isinstance(proposal.get("replay_evidence"), Mapping) else {}
        permit_id = str(replay.get("permit_id") or "")
        with self._lock:
            permit = next(
                (dict(item) for item in self._regeneration_permits if str(item.get("permit_id") or "") == permit_id),
                None,
            )
            if permit is None:
                return False
            material = {
                "issued_state_revision": int(permit.get("issued_state_revision", -1)),
                "operator_id": permit.get("operator_id"),
                "confirmation": bool(permit.get("confirmation")),
                "mismatch_hash": permit.get("mismatch_hash"),
                "pressure_hash": permit.get("pressure_hash"),
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
                and permit.get("owned_by_hecsn")
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
        artifact = next(
            (
                dict(item)
                for item in self._snn_transition_memory_replay_artifacts
                if str(item.get("replay_artifact_id") or "") == str(replay_artifact_id or "")
            ),
            None,
        )
        if artifact is None:
            return None
        material = {
            "recorded_state_revision": int(artifact.get("recorded_state_revision", -1)),
            "operator_id": artifact.get("operator_id"),
            "confirmation": bool(artifact.get("confirmation")),
            "mismatch_hash": artifact.get("mismatch_hash"),
            "pressure_hash": artifact.get("pressure_hash"),
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
        return artifact if bool(
            artifact.get("ready")
            and artifact.get("owned_by_hecsn")
            and artifact.get("confirmation") is True
            and artifact.get("internal_ledger_backed") is True
            and bool(str(artifact.get("artifact_proposal_hash") or ""))
            and bool(list(artifact.get("readout_evidence_hashes") or []))
            and context is not None
            and ticket is not None
            and str(artifact.get("review_ticket_hash") or "") == str(ticket.get("evidence_hash") or "")
            and str(artifact.get("replay_evaluation_context_hash") or "")
            == str(context.get("evidence_hash") or "")
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
        candidates = []
        for item in list(design.get("candidate_synapses") or []):
            if not isinstance(item, Mapping):
                raise ValueError("Regeneration design candidates must be mappings.")
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
