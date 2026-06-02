from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
import json
from typing import Any, Callable, Mapping, Sequence

import torch

from hecsn.service.runtime_state import RuntimeState


DEFAULT_SNN_LANGUAGE_READOUT_LEDGER_LIMIT = 128
_LANGUAGE_NEURON_COUNT = 64
_MAX_READOUT_SYNAPSE_ABS_WEIGHT = 1.0
_MAX_STRUCTURAL_EDGES_PER_EVENT = 32
_MAX_OUTGOING_FANOUT = 16
_MAX_SPARSE_TRANSITION_EDGES = 256
_MAX_OUTGOING_ROW_MASS = 1.0


class SNNLanguageReadoutEvidenceLedger:
    """Append-only evidence ledger for provenance-bound SNN language readout drafts."""

    def __init__(
        self,
        *,
        lock: Any,
        runtime_state: RuntimeState,
        ledger_state: Callable[[], dict[str, Any]],
        limit: int = DEFAULT_SNN_LANGUAGE_READOUT_LEDGER_LIMIT,
    ) -> None:
        self._lock = lock
        self._runtime_state = runtime_state
        self._ledger_state = ledger_state
        self._limit = max(1, int(limit))

    def record_readout_draft(
        self,
        *,
        readout_draft: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Record one review-ready readout draft as bounded replay evidence."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            required_evidence = {
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
            }
            draft = dict(readout_draft)
            gate = draft.get("promotion_gate") if isinstance(draft.get("promotion_gate"), Mapping) else {}
            evaluation = (
                draft.get("transition_memory_evaluation_evidence")
                if isinstance(draft.get("transition_memory_evaluation_evidence"), Mapping)
                else {}
            )
            labels = (
                draft.get("draft", {}).get("labels")
                if isinstance(draft.get("draft"), Mapping)
                else []
            )
            required_evidence.update(
                {
                    "readout_draft_surface": draft.get("surface") == "snn_language_readout_draft.v1",
                    "bounded_readout_generation_ready": bool(
                        gate.get("eligible_for_bounded_readout_generation")
                    ),
                    "freeform_language_generation_absent": not bool(
                        draft.get("freeform_language_generation")
                    ),
                    "cognition_substrate_absent": not bool(gate.get("eligible_for_cognition_substrate")),
                    "provenance_match": bool(evaluation.get("provenance_match")),
                    "prediction_hash_available": bool(str(evaluation.get("prediction_hash") or "")),
                    "transition_memory_evaluation_hash_available": bool(
                        str(evaluation.get("transition_memory_evaluation_hash") or "")
                    ),
                    "runtime_mutation_absent_in_draft": not bool(draft.get("mutates_runtime_state")),
                    "labels_available": bool(labels),
                }
            )
            accepted = all(required_evidence.values())
            if not accepted:
                return self._blocked(before_revision, required_evidence)

            state = self._normalized_state()
            event = self._ledger_event(
                draft=draft,
                operator_id=str(operator_id).strip(),
                state_revision=before_revision,
            )
            existing_hashes = {str(item.get("readout_evidence_hash") or "") for item in state["events"]}
            duplicate = event["readout_evidence_hash"] in existing_hashes
            if not duplicate:
                state["events"].appendleft(deepcopy(event))
                state["total_recorded_count"] = int(state.get("total_recorded_count", 0) or 0) + 1
                state["last_recorded_at"] = event["recorded_at"]
                self._store_state(state)
                self._runtime_state.mark_dirty_without_revision()
            return {
                "artifact_kind": "terminus_snn_language_readout_evidence_ledger_record",
                "surface": "snn_language_readout_evidence_ledger_record.v1",
                "accepted": True,
                "duplicate": duplicate,
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": not duplicate,
                "requires_operator_approval": True,
                "operator_id": str(operator_id).strip(),
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
                "recorded_event": event,
                "ledger_summary": self.snapshot(limit=0)["summary"],
                "promotion_gate": {
                    "status": "recorded" if not duplicate else "duplicate_already_recorded",
                    "eligible_for_replay_memory": True,
                    "eligible_for_freeform_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "required_evidence": required_evidence,
                },
            }

    def record_readout_rollout_replay_evaluation(
        self,
        *,
        readout_rollout_replay_evaluation: Mapping[str, Any],
        expected_state_revision: int,
        operator_id: str,
        confirmation: bool,
    ) -> dict[str, Any]:
        """Record one review-ready rollout replay evaluation without replay priority."""

        with self._lock:
            before_revision = int(self._runtime_state.state_revision)
            required_evidence = {
                "expected_revision_current": int(expected_state_revision) == before_revision,
                "confirmation": bool(confirmation),
                "operator_id_available": bool(str(operator_id or "").strip()),
            }
            evaluation = dict(readout_rollout_replay_evaluation)
            gate = (
                evaluation.get("promotion_gate")
                if isinstance(evaluation.get("promotion_gate"), Mapping)
                else {}
            )
            provenance = (
                evaluation.get("provenance_evidence")
                if isinstance(evaluation.get("provenance_evidence"), Mapping)
                else {}
            )
            replay_evaluation = (
                evaluation.get("replay_evaluation")
                if isinstance(evaluation.get("replay_evaluation"), Mapping)
                else {}
            )
            observed_device = (
                evaluation.get("device_evidence")
                if isinstance(evaluation.get("device_evidence"), Mapping)
                else {}
            )
            targets = [
                dict(item)
                for item in list(replay_evaluation.get("replay_targets") or [])
                if isinstance(item, Mapping)
            ][:32]
            required_evidence.update(
                {
                    "rollout_replay_evaluation_surface": evaluation.get("surface")
                    == "snn_language_readout_rollout_replay_evaluation.v1",
                    "rollout_recording_review_ready": bool(
                        gate.get("eligible_for_readout_rollout_ledger_recording_review")
                    ),
                    "external_dependency_absent": not bool(evaluation.get("external_dependency")),
                    "external_checkpoint_absent": not bool(evaluation.get("loads_external_checkpoint")),
                    "generation_absent": not bool(evaluation.get("generates_text")),
                    "freeform_language_generation_absent": not bool(
                        evaluation.get("freeform_language_generation")
                    ),
                    "text_decoding_absent": not bool(evaluation.get("decodes_text")),
                    "runtime_mutation_absent": not bool(evaluation.get("mutates_runtime_state")),
                    "training_absent": not bool(evaluation.get("trains_runtime_model")),
                    "plasticity_absent": not bool(evaluation.get("applies_plasticity")),
                    "not_already_recorded": not bool(evaluation.get("recorded_in_ledger")),
                    "replay_priority_absent": not bool(evaluation.get("eligible_for_replay_priority")),
                    "rollout_replay_evaluation_hash_available": bool(
                        str(provenance.get("rollout_replay_evaluation_hash") or "")
                    ),
                    "rollout_hash_available": bool(str(provenance.get("rollout_hash") or "")),
                    "rollout_id_available": bool(str(provenance.get("rollout_id") or "")),
                    "prediction_hash_available": bool(str(provenance.get("prediction_hash") or "")),
                    "current_sparse_code_hash_available": bool(
                        str(provenance.get("current_sparse_code_hash") or "")
                    ),
                    "persistent_transition_weights_hash_available": bool(
                        str(provenance.get("persistent_transition_weights_hash") or "")
                    ),
                    "server_transition_memory_hash_available": bool(
                        str(provenance.get("server_transition_memory_hash") or "")
                    ),
                    "server_transition_memory_hash_match": bool(
                        provenance.get("server_transition_memory_hash_match")
                    ),
                    "server_transition_memory_state_source_bound": (
                        str(provenance.get("transition_memory_state_source") or "")
                        == "service.runtime_facade.snn_language_plasticity_runtime_state"
                    ),
                    "transition_memory_evaluation_hash_available": bool(
                        str(provenance.get("transition_memory_evaluation_hash") or "")
                    ),
                    "grounded_replay_targets_available": bool(targets)
                    and all(bool(item.get("grounded")) for item in targets),
                    "replay_target_hashes_valid": bool(targets)
                    and all(
                        str(item.get("active_indices_hash") or "")
                        == self._sha256_json(
                            [
                                int(value)
                                for value in list(item.get("predicted_sparse_indices") or [])
                                if isinstance(value, int)
                            ][:16]
                        )
                        for item in targets
                    ),
                    "observed_tensor_device_available": bool(
                        str(observed_device.get("tensor_device") or "")
                    ),
                }
            )
            accepted = all(required_evidence.values())
            if not accepted:
                return self._blocked_rollout_record(before_revision, required_evidence)

            state = self._normalized_state()
            event = self._rollout_ledger_event(
                evaluation=evaluation,
                operator_id=str(operator_id).strip(),
                state_revision=before_revision,
                targets=targets,
            )
            existing_hashes = {
                str(item.get("rollout_evidence_hash") or "")
                for item in state["rollout_events"]
            }
            duplicate = event["rollout_evidence_hash"] in existing_hashes
            if not duplicate:
                state["rollout_events"].appendleft(deepcopy(event))
                state["total_rollout_recorded_count"] = int(
                    state.get("total_rollout_recorded_count", 0) or 0
                ) + 1
                state["last_rollout_recorded_at"] = event["recorded_at"]
                self._store_state(state)
                self._runtime_state.mark_dirty_without_revision()
            return {
                "artifact_kind": "terminus_snn_language_readout_rollout_evidence_ledger_record",
                "surface": "snn_language_readout_rollout_evidence_ledger_record.v1",
                "accepted": True,
                "duplicate": duplicate,
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": not duplicate,
                "requires_operator_approval": True,
                "operator_id": str(operator_id).strip(),
                "before": {"state_revision": before_revision},
                "after": self._runtime_state.mutation_summary(),
                "recorded_event": event,
                "ledger_summary": self.snapshot(limit=0)["summary"],
                "promotion_gate": {
                    "status": "recorded" if not duplicate else "duplicate_already_recorded",
                    "eligible_for_rollout_replay_memory": True,
                    "eligible_for_replay_priority": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_freeform_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "required_evidence": required_evidence,
                },
            }

    def snapshot(self, *, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            state = self._normalized_state()
            count = max(0, int(limit))
            events = list(state["events"])[:count] if count > 0 else []
            rollout_events = list(state["rollout_events"])[:count] if count > 0 else []
            prediction_hashes = {
                str(item.get("prediction_hash") or "")
                for item in state["events"]
                if str(item.get("prediction_hash") or "")
            }
            transition_memory_hashes = {
                str(item.get("persistent_transition_weights_hash") or "")
                for item in state["events"]
                if str(item.get("persistent_transition_weights_hash") or "")
            }
            return {
                "artifact_kind": "terminus_snn_language_readout_evidence_ledger",
                "surface": "snn_language_readout_evidence_ledger.v1",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "mutates_runtime_state": False,
                "summary": {
                    "event_count": len(state["events"]),
                    "rollout_event_count": len(state["rollout_events"]),
                    "total_recorded_count": int(state.get("total_recorded_count", 0) or 0),
                    "total_rollout_recorded_count": int(
                        state.get("total_rollout_recorded_count", 0) or 0
                    ),
                    "unique_prediction_count": len(prediction_hashes),
                    "unique_transition_memory_count": len(transition_memory_hashes),
                    "last_recorded_at": state.get("last_recorded_at"),
                    "last_rollout_recorded_at": state.get("last_rollout_recorded_at"),
                },
                "events": [deepcopy(item) for item in events],
                "rollout_events": [deepcopy(item) for item in rollout_events],
            }

    def replay_priority(self, *, limit: int = 12) -> dict[str, Any]:
        """Rank readout evidence for future isolated SNN replay review."""

        with self._lock:
            state = self._normalized_state()
            events = [deepcopy(item) for item in list(state["events"])]
            label_counts: dict[str, int] = {}
            transition_counts: dict[str, int] = {}
            for event in events:
                label_key = "|".join(str(value) for value in list(event.get("labels") or []))
                transition_key = str(event.get("persistent_transition_weights_hash") or "")
                if label_key:
                    label_counts[label_key] = label_counts.get(label_key, 0) + 1
                if transition_key:
                    transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            candidates: list[dict[str, Any]] = []
            total = max(1, len(events))
            for index, event in enumerate(events):
                labels = [str(value) for value in list(event.get("labels") or [])]
                label_grounding = [bool(value) for value in list(event.get("label_grounding") or [])]
                label_key = "|".join(labels)
                transition_key = str(event.get("persistent_transition_weights_hash") or "")
                recency = 1.0 - min(1.0, index / max(1, total - 1)) if total > 1 else 1.0
                repetition = min(1.0, label_counts.get(label_key, 0) / 3.0) if label_key else 0.0
                transition_reuse = min(1.0, transition_counts.get(transition_key, 0) / 3.0) if transition_key else 0.0
                provenance = 1.0 if event.get("prediction_hash") and event.get("transition_memory_evaluation_hash") else 0.0
                score = 100.0 * (
                    0.45 * provenance
                    + 0.25 * repetition
                    + 0.20 * recency
                    + 0.10 * transition_reuse
                )
                candidates.append(
                    {
                        "candidate_id": f"snn-readout-replay:{str(event.get('readout_evidence_hash') or '')[:16]}",
                        "readout_evidence_hash": event.get("readout_evidence_hash"),
                        "source_readout_evidence_id": event.get("readout_evidence_id"),
                        "prediction_hash": event.get("prediction_hash"),
                        "transition_memory_evaluation_hash": event.get("transition_memory_evaluation_hash"),
                        "persistent_transition_weights_hash": event.get("persistent_transition_weights_hash"),
                        "state_revision": int(event.get("state_revision", 0) or 0),
                        "labels": labels,
                        "label_grounding": label_grounding,
                        "all_labels_grounded": bool(labels)
                        and len(label_grounding) == len(labels)
                        and all(label_grounding),
                        "priority_score": float(score),
                        "priority_components": {
                            "provenance": float(provenance),
                            "label_repetition": float(repetition),
                            "recency": float(recency),
                            "transition_memory_reuse": float(transition_reuse),
                        },
                        "reason_codes": [
                            code
                            for code, active in (
                                ("provenance_bound_readout", provenance > 0.0),
                                ("repeated_label_evidence", repetition >= (2.0 / 3.0)),
                                ("recent_readout_evidence", recency >= 0.5),
                                ("transition_memory_reuse", transition_reuse >= (2.0 / 3.0)),
                            )
                            if active
                        ],
                        "suggested_rehearsal_action": "operator_review_isolated_snn_readout_replay",
                        "executable": False,
                        "advisory": True,
                        "applies_plasticity": False,
                        "trains_runtime_model": False,
                        "mutates_runtime_state": False,
                        "generates_text": False,
                        "decodes_text": False,
                        "eligible_for_fact_promotion": False,
                        "eligible_for_action": False,
                        "eligible_for_cognition_substrate": False,
                    }
                )
            candidates.sort(
                key=lambda item: (
                    -float(item["priority_score"]),
                    -len(list(item.get("labels") or [])),
                    str(item.get("readout_evidence_hash") or ""),
                )
            )
            count = max(0, int(limit))
            selected = [
                {**candidate, "rank": rank}
                for rank, candidate in enumerate(candidates[:count], start=1)
            ] if count > 0 else []
            ready = bool(selected)
            return {
                "artifact_kind": "terminus_snn_language_readout_replay_priority",
                "surface": "snn_language_readout_replay_priority.v1",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "advisory": True,
                "executable": False,
                "candidate_count": len(selected),
                "priority_rules_version": "readout-ledger-deterministic-v1",
                "priority_weights": {
                    "provenance": 0.45,
                    "label_repetition": 0.25,
                    "recency": 0.20,
                    "transition_memory_reuse": 0.10,
                },
                "ledger_summary": self.snapshot(limit=0)["summary"],
                "candidates": selected,
                "promotion_gate": {
                    "status": "ready_for_operator_replay_review" if ready else "collect_readout_evidence",
                    "eligible_for_operator_replay_review": ready,
                    "eligible_for_live_replay": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_freeform_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "requires_operator_approval": ready,
                    "next_gate": "operator_review_isolated_snn_readout_replay"
                    if ready
                    else "record_provenance_matched_readout_evidence",
                    "required_evidence": {
                        "readout_evidence_available": bool(events),
                        "provenance_hashes_available": all(
                            bool(candidate.get("priority_components", {}).get("provenance"))
                            for candidate in selected
                        )
                        if selected
                        else False,
                        "runtime_mutation_absent": True,
                        "freeform_language_generation_absent": True,
                    },
                },
            }

    def rollout_rehearsal_promotion_policy(
        self,
        *,
        candidate_limit: int = 8,
    ) -> dict[str, Any]:
        """Rank recorded rollout evidence for isolated rehearsal review only."""

        with self._lock:
            state = self._normalized_state()
            events = [deepcopy(item) for item in list(state["rollout_events"])]
            rollout_counts: dict[str, int] = {}
            transition_counts: dict[str, int] = {}
            for event in events:
                rollout_key = str(event.get("rollout_hash") or "")
                transition_key = str(event.get("persistent_transition_weights_hash") or "")
                if rollout_key:
                    rollout_counts[rollout_key] = rollout_counts.get(rollout_key, 0) + 1
                if transition_key:
                    transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
            candidates: list[dict[str, Any]] = []
            total = max(1, len(events))
            for index, event in enumerate(events):
                targets = [
                    self._normalized_rollout_replay_target(item, index=target_index)
                    for target_index, item in enumerate(list(event.get("replay_targets") or []))
                    if isinstance(item, Mapping)
                ][:32]
                observed_device = (
                    event.get("device_evidence")
                    if isinstance(event.get("device_evidence"), Mapping)
                    else {}
                )
                requested_device = str(observed_device.get("requested_device") or "")
                tensor_device = str(observed_device.get("tensor_device") or "")
                cuda_tensor = bool(observed_device.get("cuda_tensor"))
                requested_cuda_honored = (
                    not requested_device.startswith("cuda")
                    or (tensor_device.startswith("cuda") and cuda_tensor)
                )
                provenance_complete = bool(
                    event.get("rollout_replay_evaluation_hash")
                    and event.get("rollout_hash")
                    and event.get("prediction_hash")
                    and event.get("current_sparse_code_hash")
                    and event.get("transition_memory_evaluation_hash")
                    and event.get("persistent_transition_weights_hash")
                    and event.get("server_transition_memory_hash")
                    and event.get("server_transition_memory_hash_match")
                    and str(event.get("transition_memory_state_source") or "")
                    == "service.runtime_facade.snn_language_plasticity_runtime_state"
                )
                grounding_complete = bool(targets) and all(bool(item.get("grounded")) for item in targets)
                trace_integrity_complete = bool(targets) and all(
                    bool(item.get("active_indices_hash_valid")) for item in targets
                )
                device_evidence_complete = bool(tensor_device) and requested_cuda_honored
                evidence_hash_valid = str(event.get("rollout_evidence_hash") or "") == (
                    self._rollout_ledger_event_material_hash(event)
                )
                if not (
                    provenance_complete
                    and grounding_complete
                    and trace_integrity_complete
                    and device_evidence_complete
                    and evidence_hash_valid
                ):
                    continue
                sparse_indices = sorted(
                    {
                        int(value)
                        for item in targets
                        for value in list(item.get("predicted_sparse_indices") or [])
                        if isinstance(value, int) and 0 <= int(value) < _LANGUAGE_NEURON_COUNT
                    }
                )
                rollout_key = str(event.get("rollout_hash") or "")
                transition_key = str(event.get("persistent_transition_weights_hash") or "")
                recency = 1.0 - min(1.0, index / max(1, total - 1)) if total > 1 else 1.0
                repetition = min(1.0, rollout_counts.get(rollout_key, 0) / 3.0) if rollout_key else 0.0
                transition_reuse = min(1.0, transition_counts.get(transition_key, 0) / 3.0) if transition_key else 0.0
                provenance = 1.0
                grounding = 1.0
                trace_integrity = 1.0
                score = 100.0 * (
                    0.35 * provenance
                    + 0.20 * grounding
                    + 0.15 * trace_integrity
                    + 0.15 * recency
                    + 0.10 * transition_reuse
                    + 0.05 * repetition
                )
                candidates.append(
                    {
                        "candidate_id": f"snn-readout-rollout-rehearsal:{str(event.get('rollout_evidence_hash') or '')[:16]}",
                        "rollout_evidence_hash": event.get("rollout_evidence_hash"),
                        "source_rollout_evidence_id": event.get("rollout_evidence_id"),
                        "rollout_replay_evaluation_hash": event.get("rollout_replay_evaluation_hash"),
                        "rollout_hash": event.get("rollout_hash"),
                        "prediction_hash": event.get("prediction_hash"),
                        "current_sparse_code_hash": event.get("current_sparse_code_hash"),
                        "transition_memory_evaluation_hash": event.get("transition_memory_evaluation_hash"),
                        "persistent_transition_weights_hash": event.get("persistent_transition_weights_hash"),
                        "server_transition_memory_hash": event.get("server_transition_memory_hash"),
                        "server_transition_memory_hash_match": bool(
                            event.get("server_transition_memory_hash_match")
                        ),
                        "transition_memory_state_source": event.get("transition_memory_state_source"),
                        "state_revision": int(event.get("state_revision", 0) or 0),
                        "target_count": len(targets),
                        "replay_targets": targets,
                        "sparse_index_count": len(sparse_indices),
                        "sparse_occupancy": float(len(sparse_indices)) / float(_LANGUAGE_NEURON_COUNT),
                        "device_evidence": {
                            "requested_device": requested_device,
                            "tensor_device": tensor_device,
                            "cuda_tensor": cuda_tensor,
                            "device_source": observed_device.get("device_source"),
                        },
                        "priority_score": float(score),
                        "priority_components": {
                            "provenance": float(provenance),
                            "grounding": float(grounding),
                            "trace_integrity": float(trace_integrity),
                            "recency": float(recency),
                            "transition_memory_reuse": float(transition_reuse),
                            "rollout_repetition": float(repetition),
                        },
                        "suggested_rehearsal_action": "operator_review_isolated_snn_readout_rollout_rehearsal",
                        "executable": False,
                        "advisory": True,
                        "applies_plasticity": False,
                        "trains_runtime_model": False,
                        "mutates_runtime_state": False,
                        "generates_text": False,
                        "decodes_text": False,
                        "eligible_for_fact_promotion": False,
                        "eligible_for_action": False,
                        "eligible_for_cognition_substrate": False,
                    }
                )
            candidates.sort(
                key=lambda item: (
                    -float(item["priority_score"]),
                    -int(item["target_count"]),
                    str(item.get("rollout_evidence_hash") or ""),
                )
            )
            count = max(0, min(int(candidate_limit), 32))
            selected = [
                {**candidate, "rank": rank}
                for rank, candidate in enumerate(candidates[:count], start=1)
            ] if count > 0 else []
            ready = bool(selected) and all(
                float(candidate["priority_components"]["provenance"]) > 0.0
                and float(candidate["priority_components"]["grounding"]) > 0.0
                and float(candidate["priority_components"]["trace_integrity"]) > 0.0
                for candidate in selected
            )
            return {
                "artifact_kind": "terminus_snn_language_readout_rollout_rehearsal_promotion_policy",
                "surface": "snn_language_readout_rollout_rehearsal_promotion_policy.v1",
                "owned_by_hecsn": True,
                "external_dependency": False,
                "loads_external_checkpoint": False,
                "generates_text": False,
                "decodes_text": False,
                "trains_runtime_model": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "advisory": True,
                "executable": False,
                "candidate_count": len(selected),
                "policy_rules_version": "readout-rollout-ledger-deterministic-v1",
                "ledger_summary": self.snapshot(limit=0)["summary"],
                "candidates": selected,
                "promotion_gate": {
                    "status": "ready_for_operator_rollout_rehearsal_review"
                    if ready
                    else "collect_recorded_rollout_replay_evidence",
                    "eligible_for_operator_rollout_rehearsal_review": ready,
                    "eligible_for_replay_priority": False,
                    "eligible_for_live_replay": False,
                    "eligible_for_plasticity_application": False,
                    "eligible_for_freeform_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_action": False,
                    "requires_operator_approval": ready,
                    "next_gate": "operator_review_isolated_snn_readout_rollout_rehearsal"
                    if ready
                    else "record_review_ready_snn_language_rollout_evidence",
                    "required_evidence": {
                        "recorded_rollout_evidence_available": bool(events),
                        "eligible_recorded_rollout_evidence_available": bool(candidates),
                        "selected_candidates_available": bool(selected),
                        "provenance_complete": bool(selected)
                        and all(candidate["priority_components"]["provenance"] > 0.0 for candidate in selected),
                        "grounded_targets_available": bool(selected)
                        and all(candidate["priority_components"]["grounding"] > 0.0 for candidate in selected),
                        "trace_integrity_available": bool(selected)
                        and all(candidate["priority_components"]["trace_integrity"] > 0.0 for candidate in selected),
                        "runtime_mutation_absent": True,
                        "freeform_language_generation_absent": True,
                    },
                },
            }

    def rollout_rehearsal_evaluation(
        self,
        rollout_rehearsal_promotion_policy: Mapping[str, Any],
        *,
        candidate_limit: int = 8,
    ) -> dict[str, Any]:
        """Evaluate rollout candidates as isolated sparse temporal rehearsal."""

        report = dict(rollout_rehearsal_promotion_policy)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        raw_candidates = [
            dict(item)
            for item in list(report.get("candidates") or [])[: max(0, min(int(candidate_limit), 32))]
            if isinstance(item, Mapping)
        ]
        traces: list[dict[str, Any]] = []
        all_step_vectors: list[torch.Tensor] = []
        adjacent_similarities: list[float] = []
        for candidate in raw_candidates:
            device_evidence = (
                candidate.get("device_evidence")
                if isinstance(candidate.get("device_evidence"), Mapping)
                else {}
            )
            requested_device = str(device_evidence.get("requested_device") or "")
            observed_tensor_device = str(device_evidence.get("tensor_device") or "")
            observed_cuda_tensor = bool(device_evidence.get("cuda_tensor"))
            requested_cuda_honored = (
                not requested_device.startswith("cuda")
                or (observed_tensor_device.startswith("cuda") and observed_cuda_tensor)
            )
            device = self._safe_tensor_device(observed_tensor_device)
            actual_device_match = str(device) == observed_tensor_device
            replay_targets = [
                self._normalized_rollout_replay_target(item, index=index)
                for index, item in enumerate(list(candidate.get("replay_targets") or [])[:32])
                if isinstance(item, Mapping)
            ]
            step_vectors: list[torch.Tensor] = []
            step_traces: list[dict[str, Any]] = []
            for target in replay_targets:
                active_indices = [
                    int(value)
                    for value in list(target.get("predicted_sparse_indices") or [])
                    if isinstance(value, int) and 0 <= int(value) < _LANGUAGE_NEURON_COUNT
                ]
                vector = torch.zeros(_LANGUAGE_NEURON_COUNT, device=device)
                if active_indices:
                    vector[active_indices] = 1.0
                step_vectors.append(vector)
                all_step_vectors.append(vector)
                step_traces.append(
                    {
                        "step_index": int(target.get("step_index", 0) or 0),
                        "selected_label": target.get("selected_label"),
                        "grounded": bool(target.get("grounded")),
                        "active_indices_hash": target.get("active_indices_hash"),
                        "active_indices_hash_valid": bool(target.get("active_indices_hash_valid")),
                        "sparse_active_indices": active_indices,
                        "sparse_occupancy": float(len(set(active_indices))) / float(_LANGUAGE_NEURON_COUNT),
                        "applied_to_runtime": False,
                        "weights_persisted": False,
                        "generated_text": False,
                    }
                )
            candidate_adjacent: list[float] = []
            for previous, current in zip(step_vectors, step_vectors[1:]):
                overlap = float(torch.minimum(previous, current).sum().item())
                union = float(torch.maximum(previous, current).sum().item())
                similarity = overlap / max(1.0, union)
                candidate_adjacent.append(similarity)
                adjacent_similarities.append(similarity)
            traces.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "rollout_evidence_hash": candidate.get("rollout_evidence_hash"),
                    "rollout_hash": candidate.get("rollout_hash"),
                    "prediction_hash": candidate.get("prediction_hash"),
                    "current_sparse_code_hash": candidate.get("current_sparse_code_hash"),
                    "transition_memory_evaluation_hash": candidate.get("transition_memory_evaluation_hash"),
                    "persistent_transition_weights_hash": candidate.get("persistent_transition_weights_hash"),
                    "priority_score": float(candidate.get("priority_score", 0.0) or 0.0),
                    "step_count": len(step_traces),
                    "step_trace": step_traces,
                    "mean_adjacent_sparse_similarity": (
                        sum(candidate_adjacent) / len(candidate_adjacent)
                        if candidate_adjacent
                        else 1.0
                    ),
                    "device_evidence": {
                        "requested_device": requested_device,
                        "tensor_device": str(device),
                        "cuda_tensor": bool(device.type == "cuda"),
                        "device_source": device_evidence.get("device_source"),
                        "observed_tensor_device": observed_tensor_device,
                        "observed_cuda_tensor": observed_cuda_tensor,
                        "requested_cuda_honored": requested_cuda_honored,
                        "actual_device_match": actual_device_match,
                    },
                    "applied_to_runtime": False,
                    "weights_persisted": False,
                    "checkpoint_written": False,
                    "generated_text": False,
                }
            )
        if all_step_vectors:
            stack = torch.stack(all_step_vectors)
            activation_sparsity = 1.0 - float((stack > 0).float().mean().item())
        else:
            activation_sparsity = 1.0
        mean_adjacent_similarity = (
            sum(adjacent_similarities) / len(adjacent_similarities)
            if adjacent_similarities
            else (1.0 if traces else 0.0)
        )
        rehearsal_stable = bool(traces) and activation_sparsity >= 0.75 and mean_adjacent_similarity >= 0.0
        required = {
            "policy_surface_available": report.get("surface")
            == "snn_language_readout_rollout_rehearsal_promotion_policy.v1",
            "policy_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "policy_gate_ready": bool(gate.get("eligible_for_operator_rollout_rehearsal_review")),
            "policy_report_non_executable": not bool(report.get("executable")),
            "policy_report_non_mutating": not bool(report.get("mutates_runtime_state")),
            "candidates_available": bool(traces),
            "candidate_provenance_available": all(
                bool(trace.get("rollout_evidence_hash"))
                and bool(trace.get("rollout_hash"))
                and bool(trace.get("prediction_hash"))
                and bool(trace.get("current_sparse_code_hash"))
                and bool(trace.get("transition_memory_evaluation_hash"))
                for trace in traces
            ) if traces else False,
            "candidate_devices_match_observed": all(
                bool(trace.get("device_evidence", {}).get("actual_device_match"))
                and bool(trace.get("device_evidence", {}).get("requested_cuda_honored"))
                for trace in traces
            ) if traces else False,
            "target_hashes_valid": all(
                all(bool(step.get("active_indices_hash_valid")) for step in trace.get("step_trace", []))
                for trace in traces
            ) if traces else False,
            "grounded_targets_available": all(
                all(bool(step.get("grounded")) for step in trace.get("step_trace", []))
                for trace in traces
            ) if traces else False,
            "sparse_temporal_rehearsal_stable": rehearsal_stable,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_rehearsal_evaluation",
            "surface": "snn_language_readout_rollout_rehearsal_evaluation.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rollout_rehearsal_evaluation",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "policy_surface": report.get("surface"),
            "rehearsal_summary": {
                "candidate_count": len(traces),
                "step_count": len(all_step_vectors),
                "activation_sparsity": float(activation_sparsity),
                "mean_adjacent_sparse_similarity": float(mean_adjacent_similarity),
                "sparse_temporal_rehearsal_stable": bool(rehearsal_stable),
            },
            "ephemeral_rehearsal": {
                "trace": traces,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "generated_text": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_rehearsal_evidence",
                "eligible_for_operator_rollout_rehearsal_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_isolated_snn_readout_rollout_rehearsal_evaluation"
                if ready
                else "collect_rollout_rehearsal_promotion_candidates",
                "required_evidence": required,
            },
        }

    def rollout_rehearsal_experiment(
        self,
        rollout_rehearsal_evaluation: Mapping[str, Any],
        *,
        replay_cycles: int = 3,
        stability_floor: float = 0.95,
    ) -> dict[str, Any]:
        """Repeat isolated rollout rehearsal cycles and measure stability."""

        report = dict(rollout_rehearsal_evaluation)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        ephemeral = (
            report.get("ephemeral_rehearsal")
            if isinstance(report.get("ephemeral_rehearsal"), Mapping)
            else {}
        )
        source_summary = (
            report.get("rehearsal_summary")
            if isinstance(report.get("rehearsal_summary"), Mapping)
            else {}
        )
        requested_cycles = max(1, min(int(replay_cycles), 12))
        requested_floor = max(0.0, min(float(stability_floor), 1.0))
        source_traces = [
            dict(item)
            for item in list(ephemeral.get("trace") or [])
            if isinstance(item, Mapping)
        ]
        cycle_trace: list[dict[str, Any]] = []
        baseline_hashes = [
            str(step.get("active_indices_hash") or "")
            for trace in source_traces
            for step in list(trace.get("step_trace") or [])
            if isinstance(step, Mapping)
        ]
        sparse_transition_candidates = self._rollout_sparse_transition_candidates(
            source_traces
        )
        for cycle_index in range(requested_cycles):
            replayed_hashes = [
                str(step.get("active_indices_hash") or "")
                for trace in source_traces
                for step in list(trace.get("step_trace") or [])
                if isinstance(step, Mapping)
            ]
            matching_hash_count = sum(
                1
                for expected, observed in zip(baseline_hashes, replayed_hashes)
                if expected and expected == observed
            )
            stability = matching_hash_count / max(1, len(baseline_hashes))
            cycle_trace.append(
                {
                    "cycle_index": cycle_index,
                    "candidate_count": len(source_traces),
                    "step_count": len(replayed_hashes),
                    "matching_step_hash_count": matching_hash_count,
                    "cycle_stability": float(stability),
                    "mean_adjacent_sparse_similarity": float(
                        source_summary.get("mean_adjacent_sparse_similarity", 0.0) or 0.0
                    ),
                    "runtime_update_applied": False,
                    "weights_persisted": False,
                    "checkpoint_written": False,
                    "plasticity_applied": False,
                    "generated_text": False,
                }
            )
        cycle_stabilities = [float(item["cycle_stability"]) for item in cycle_trace]
        minimum_cycle_stability = min(cycle_stabilities) if cycle_stabilities else 0.0
        mean_cycle_stability = sum(cycle_stabilities) / max(1, len(cycle_stabilities))
        drift = max(0.0, 1.0 - minimum_cycle_stability)
        stable = bool(cycle_trace) and minimum_cycle_stability >= requested_floor
        required = {
            "rollout_rehearsal_evaluation_surface_available": report.get("surface")
            == "snn_language_readout_rollout_rehearsal_evaluation.v1",
            "rollout_rehearsal_evaluation_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "rollout_rehearsal_evaluation_gate_ready": bool(
                gate.get("eligible_for_operator_rollout_rehearsal_review")
            ),
            "rollout_rehearsal_evaluation_non_mutating": not bool(report.get("mutates_runtime_state")),
            "source_rehearsal_ephemeral": not bool(ephemeral.get("runtime_update_applied"))
            and not bool(ephemeral.get("weights_persisted"))
            and not bool(ephemeral.get("checkpoint_written")),
            "source_trace_available": bool(source_traces) and bool(baseline_hashes),
            "cycle_count_available": bool(cycle_trace),
            "cycle_stability_sufficient": stable,
        }
        ready = all(required.values())
        experiment_hash = self._sha256_json(
            {
                "source_surface": report.get("surface"),
                "source_trace_hashes": baseline_hashes,
                "replay_cycles": requested_cycles,
                "stability_floor": requested_floor,
                "cycle_stabilities": cycle_stabilities,
            }
        )
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_rehearsal_experiment",
            "surface": "snn_language_readout_rollout_rehearsal_experiment.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rollout_rehearsal_experiment",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rollout_rehearsal_evaluation_surface": report.get("surface"),
            "experiment_summary": {
                "replay_cycles": requested_cycles,
                "candidate_count": len(source_traces),
                "step_count": len(baseline_hashes),
                "sparse_transition_candidate_count": len(sparse_transition_candidates),
                "stability_floor": requested_floor,
                "minimum_cycle_stability": float(minimum_cycle_stability),
                "mean_cycle_stability": float(mean_cycle_stability),
                "cycle_drift": float(drift),
                "mean_adjacent_sparse_similarity": float(
                    source_summary.get("mean_adjacent_sparse_similarity", 0.0) or 0.0
                ),
                "stable": stable,
            },
            "ephemeral_experiment": {
                "trace": cycle_trace,
                "sparse_transition_candidates": sparse_transition_candidates,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "plasticity_applied": False,
                "generated_text": False,
            },
            "provenance_evidence": {
                "rollout_rehearsal_experiment_hash": experiment_hash,
                "rollout_rehearsal_experiment_id": f"snn-readout-rollout-rehearsal-experiment:{experiment_hash[:16]}",
                "hash_algorithm": "sha256_canonical_json",
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_rehearsal_experiment_evidence",
                "eligible_for_operator_rollout_rehearsal_experiment_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_rehearsal_experiment"
                if ready
                else "collect_stable_rollout_rehearsal_cycles",
                "required_evidence": required,
            },
        }

    def rollout_consolidation_design(
        self,
        rollout_rehearsal_experiment: Mapping[str, Any],
        *,
        consolidation_policy: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Design bounded local rollout consolidation without applying writes."""

        report = dict(rollout_rehearsal_experiment)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        summary = report.get("experiment_summary") if isinstance(report.get("experiment_summary"), Mapping) else {}
        experiment = (
            report.get("ephemeral_experiment")
            if isinstance(report.get("ephemeral_experiment"), Mapping)
            else {}
        )
        policy = dict(consolidation_policy or {})
        rollback = dict(rollback_policy or {})
        learning_rate = max(0.0, min(float(policy.get("learning_rate", 0.02) or 0.02), 0.25))
        max_weight_delta = max(0.0, min(float(policy.get("max_weight_delta", 0.04) or 0.04), 0.25))
        homeostatic_decay = max(0.0, min(float(policy.get("homeostatic_decay", 0.01) or 0.01), 0.25))
        local_only = bool(policy.get("local_only", True))
        normalization = bool(policy.get("normalization", True))
        cycle_trace = [
            dict(item)
            for item in list(experiment.get("trace") or [])
            if isinstance(item, Mapping)
        ]
        sparse_transition_candidates = [
            dict(item)
            for item in list(experiment.get("sparse_transition_candidates") or [])
            if isinstance(item, Mapping)
        ][:64]
        step_hashes = [
            str(item.get("matching_step_hash_count") or "")
            for item in cycle_trace
        ]
        minimum_stability = float(summary.get("minimum_cycle_stability", 0.0) or 0.0)
        stability_gain = max(0.0, min(1.0, minimum_stability))
        proposed_delta = min(max_weight_delta, learning_rate * stability_gain)
        candidate_synapses = []
        for index, candidate in enumerate(sparse_transition_candidates):
            try:
                source_index = int(candidate.get("source_index"))
                target_index = int(candidate.get("target_index"))
            except (TypeError, ValueError):
                continue
            if not (
                0 <= source_index < _LANGUAGE_NEURON_COUNT
                and 0 <= target_index < _LANGUAGE_NEURON_COUNT
            ):
                continue
            candidate_synapses.append(
                {
                    "synapse_id": f"snn-rollout-local:{source_index}:{target_index}:{index}",
                    "source_step_index": source_index,
                    "target_step_index": target_index,
                    "source_neuron_index": source_index,
                    "target_neuron_index": target_index,
                    "source_trace_index": int(candidate.get("source_trace_index", 0) or 0),
                    "source_rollout_step_index": int(candidate.get("source_step_index", 0) or 0),
                    "target_rollout_step_index": int(candidate.get("target_step_index", 0) or 0),
                    "source_active_indices_hash": candidate.get("source_active_indices_hash"),
                    "target_active_indices_hash": candidate.get("target_active_indices_hash"),
                    "local_only": local_only,
                    "proposed_weight_delta": float(proposed_delta),
                    "homeostatic_decay": float(homeostatic_decay),
                    "normalization": normalization,
                    "applied_to_runtime": False,
                }
            )
        required = {
            "rollout_rehearsal_experiment_surface_available": report.get("surface")
            == "snn_language_readout_rollout_rehearsal_experiment.v1",
            "rollout_rehearsal_experiment_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "rollout_rehearsal_experiment_gate_ready": bool(
                gate.get("eligible_for_operator_rollout_rehearsal_experiment_review")
            ),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "weight_persistence_absent": not bool(experiment.get("weights_persisted")),
            "checkpoint_write_absent": not bool(experiment.get("checkpoint_written")),
            "stable_cycles_available": bool(cycle_trace)
            and bool(summary.get("stable"))
            and minimum_stability >= 0.95,
            "candidate_synapses_available": bool(candidate_synapses),
            "local_only_policy": local_only,
            "normalization_enabled": normalization,
            "rollback_policy_available": bool(rollback.get("available")),
            "rollback_snapshot_available": bool(str(rollback.get("snapshot_id") or "")),
        }
        ready = all(required.values())
        design_material = {
            "source_experiment_hash": report.get("provenance_evidence", {}).get(
                "rollout_rehearsal_experiment_hash"
            )
            if isinstance(report.get("provenance_evidence"), Mapping)
            else None,
            "step_hashes": step_hashes,
            "learning_rate": learning_rate,
            "max_weight_delta": max_weight_delta,
            "homeostatic_decay": homeostatic_decay,
            "local_only": local_only,
            "normalization": normalization,
            "rollback_snapshot_id": rollback.get("snapshot_id"),
            "candidate_synapses": candidate_synapses,
            "sparse_transition_candidates": sparse_transition_candidates,
        }
        design_hash = self._sha256_json(design_material)
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_consolidation_design",
            "surface": "snn_language_readout_rollout_consolidation_design.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rollout_consolidation_design",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rollout_rehearsal_experiment_surface": report.get("surface"),
            "rollout_consolidation_design_hash": design_hash,
            "rollout_consolidation_design_material": design_material,
            "rollout_consolidation_design": {
                "candidate_synapse_count": len(candidate_synapses),
                "candidate_synapses": candidate_synapses,
                "learning_rate": float(learning_rate),
                "max_weight_delta": float(max_weight_delta),
                "homeostatic_decay": float(homeostatic_decay),
                "local_only": local_only,
                "normalization": normalization,
                "rollback_snapshot_id": rollback.get("snapshot_id"),
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "structural_write_applied": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_consolidation_design_evidence",
                "eligible_for_operator_rollout_consolidation_design_review": ready,
                "eligible_for_structural_write": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_consolidation_design"
                if ready
                else "collect_stable_rollout_rehearsal_experiment_evidence",
                "required_evidence": required,
            },
        }

    def rollout_consolidation_shadow_delta(
        self,
        rollout_consolidation_design: Mapping[str, Any],
        *,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Materialize bounded rollout consolidation deltas without applying them."""

        report = dict(rollout_consolidation_design)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        design = (
            report.get("rollout_consolidation_design")
            if isinstance(report.get("rollout_consolidation_design"), Mapping)
            else {}
        )
        requested = dict(device_evidence or {})
        requested_device = str(requested.get("device") or requested.get("tensor_device") or "cpu")
        tensor_device = self._safe_tensor_device(requested_device)
        requested_cuda_honored = (
            not requested_device.startswith("cuda")
            or tensor_device.type == "cuda"
        )
        candidates = [
            dict(item)
            for item in list(design.get("candidate_synapses") or [])
            if isinstance(item, Mapping)
        ][:64]
        delta_tensor = torch.zeros(
            (_LANGUAGE_NEURON_COUNT, _LANGUAGE_NEURON_COUNT),
            device=tensor_device,
        )
        bounded_synapses: list[dict[str, Any]] = []
        max_weight_delta = float(design.get("max_weight_delta", 0.0) or 0.0)
        for item in candidates:
            try:
                source_index = int(
                    item.get("source_neuron_index", item.get("source_step_index"))
                )
                target_index = int(
                    item.get("target_neuron_index", item.get("target_step_index"))
                )
                proposed = max(
                    -max_weight_delta,
                    min(float(item.get("proposed_weight_delta", 0.0) or 0.0), max_weight_delta),
                )
            except (TypeError, ValueError):
                source_index = -1
                target_index = -1
                proposed = 0.0
            if (
                0 <= source_index < _LANGUAGE_NEURON_COUNT
                and 0 <= target_index < _LANGUAGE_NEURON_COUNT
                and math.isfinite(proposed)
            ):
                delta_tensor[source_index, target_index] = proposed
            bounded_synapses.append(
                {
                    "synapse_id": item.get("synapse_id"),
                    "source_index": source_index,
                    "target_index": target_index,
                    "proposed_weight_delta": float(proposed),
                    "homeostatic_decay": float(item.get("homeostatic_decay", 0.0) or 0.0),
                    "local_only": bool(item.get("local_only")),
                    "normalization": bool(item.get("normalization")),
                    "applied_to_runtime": False,
                }
            )
        nonzero_count = int(torch.count_nonzero(delta_tensor).item())
        max_abs_delta = (
            float(torch.max(torch.abs(delta_tensor)).item())
            if nonzero_count
            else 0.0
        )
        delta_hash = self._sha256_json(
            {
                "rollout_consolidation_design_hash": report.get("rollout_consolidation_design_hash"),
                "requested_device": requested_device,
                "tensor_device": str(tensor_device),
                "bounded_synapses": bounded_synapses,
            }
        )
        required = {
            "rollout_consolidation_design_surface_available": report.get("surface")
            == "snn_language_readout_rollout_consolidation_design.v1",
            "rollout_consolidation_design_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "rollout_consolidation_design_gate_ready": bool(
                gate.get("eligible_for_operator_rollout_consolidation_design_review")
            ),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "candidate_synapses_available": bool(candidates),
            "shadow_synapses_available": bool(bounded_synapses) and nonzero_count > 0,
            "shadow_delta_within_bound": max_abs_delta <= max_weight_delta,
            "local_only_policy": bool(design.get("local_only"))
            and all(bool(item.get("local_only")) for item in bounded_synapses),
            "candidate_coordinates_canonical": bool(bounded_synapses)
            and all(
                0 <= int(item.get("source_index", -1)) < _LANGUAGE_NEURON_COUNT
                and 0 <= int(item.get("target_index", -1)) < _LANGUAGE_NEURON_COUNT
                for item in bounded_synapses
            ),
            "requested_cuda_honored": requested_cuda_honored,
            "silent_device_fallback_absent": requested_device == str(tensor_device)
            or (requested_device.startswith("cuda") and requested_cuda_honored),
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_consolidation_shadow_delta",
            "surface": "snn_language_readout_rollout_consolidation_shadow_delta.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rollout_consolidation_shadow_delta",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rollout_consolidation_design_surface": report.get("surface"),
            "rollout_consolidation_design_hash": report.get("rollout_consolidation_design_hash"),
            "rollout_consolidation_shadow_delta_hash": delta_hash,
            "rollback_snapshot_id": design.get("rollback_snapshot_id"),
            "affected_synapse_count": len(bounded_synapses),
            "max_abs_delta": float(max_abs_delta),
            "bounded_synapses": bounded_synapses,
            "device_evidence": {
                "requested_device": requested_device,
                "tensor_device": str(tensor_device),
                "cuda_tensor": bool(tensor_device.type == "cuda"),
                "requested_cuda_honored": requested_cuda_honored,
                "device_source": requested.get("source") or requested.get("device_source"),
            },
            "shadow_delta": {
                "tensor_shape": [_LANGUAGE_NEURON_COUNT, _LANGUAGE_NEURON_COUNT],
                "nonzero_count": nonzero_count,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "structural_write_applied": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_consolidation_shadow_delta_evidence",
                "eligible_for_operator_rollout_consolidation_shadow_review": ready,
                "eligible_for_shadow_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_consolidation_shadow_delta"
                if ready
                else "collect_rollout_consolidation_design_evidence",
                "required_evidence": required,
            },
        }

    def rollout_consolidation_shadow_application_preflight(
        self,
        rollout_consolidation_design: Mapping[str, Any],
        rollout_consolidation_shadow_delta: Mapping[str, Any],
        *,
        transition_memory_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Review rollout shadow-delta safety without applying synapse writes."""

        design_report = dict(rollout_consolidation_design)
        design_gate = (
            design_report.get("promotion_gate")
            if isinstance(design_report.get("promotion_gate"), Mapping)
            else {}
        )
        design = (
            design_report.get("rollout_consolidation_design")
            if isinstance(design_report.get("rollout_consolidation_design"), Mapping)
            else {}
        )
        design_material = (
            design_report.get("rollout_consolidation_design_material")
            if isinstance(design_report.get("rollout_consolidation_design_material"), Mapping)
            else {}
        )
        report = dict(rollout_consolidation_shadow_delta)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        shadow = report.get("shadow_delta") if isinstance(report.get("shadow_delta"), Mapping) else {}
        device = report.get("device_evidence") if isinstance(report.get("device_evidence"), Mapping) else {}
        memory = dict(transition_memory_state)
        weights = {
            str(key): float(value)
            for key, value in dict(memory.get("sparse_transition_weights") or {}).items()
        }
        bounded_synapses = [
            dict(item)
            for item in list(report.get("bounded_synapses") or [])
            if isinstance(item, Mapping)
        ]
        reported_max_abs_delta = float(report.get("max_abs_delta", 0.0) or 0.0)
        parsed_synapses: list[tuple[dict[str, Any], int, int, float]] = []
        for item in bounded_synapses:
            try:
                parsed_synapses.append(
                    (
                        item,
                        int(item.get("source_index")),
                        int(item.get("target_index")),
                        float(item.get("proposed_weight_delta")),
                    )
                )
            except (TypeError, ValueError):
                continue
        recomputed_max_abs_delta = max([abs(item[3]) for item in parsed_synapses], default=0.0)
        coordinate_pairs = [(source, target) for _item, source, target, _delta in parsed_synapses]
        synapse_ids = [str(item.get("synapse_id") or "") for item in bounded_synapses]
        recomputed_design_hash = self._sha256_json(design_material)
        recomputed_shadow_hash = self._sha256_json(
            {
                "rollout_consolidation_design_hash": report.get("rollout_consolidation_design_hash"),
                "requested_device": device.get("requested_device"),
                "tensor_device": device.get("tensor_device"),
                "bounded_synapses": bounded_synapses,
            }
        )
        requested_device = str(device.get("requested_device") or "")
        shadow_tensor_device = str(device.get("tensor_device") or "")
        verifier_device = self._safe_tensor_device(shadow_tensor_device)
        verifier_tensor = torch.zeros(
            (_LANGUAGE_NEURON_COUNT, _LANGUAGE_NEURON_COUNT),
            device=verifier_device,
        )
        for _item, source, target, delta in parsed_synapses:
            if (
                0 <= source < _LANGUAGE_NEURON_COUNT
                and 0 <= target < _LANGUAGE_NEURON_COUNT
                and math.isfinite(delta)
            ):
                verifier_tensor[source, target] = delta
        verifier_nonzero_count = int(torch.count_nonzero(verifier_tensor).item())
        verifier_tensor_device = str(verifier_tensor.device)
        bound_snapshot_id = str(report.get("rollback_snapshot_id") or "")
        simulated = dict(weights)
        growth_candidates: list[str] = []
        prune_candidates: list[str] = []
        for _item, source, target, delta in parsed_synapses:
            key = f"{source}:{target}"
            if key not in weights:
                growth_candidates.append(key)
            simulated[key] = float(simulated.get(key, 0.0)) + delta
            if simulated[key] <= 0.0:
                prune_candidates.append(key)
        fanout: dict[str, int] = {}
        row_mass: dict[str, float] = {}
        for key, value in simulated.items():
            source = key.split(":", maxsplit=1)[0]
            fanout[source] = fanout.get(source, 0) + 1
            row_mass[source] = row_mass.get(source, 0.0) + abs(float(value))
        topology_keyset_unchanged = not growth_candidates and not prune_candidates
        runtime_snapshot = self._runtime_state.snapshot()
        required = {
            "rollout_consolidation_design_surface_available": design_report.get("surface")
            == "snn_language_readout_rollout_consolidation_design.v1",
            "rollout_consolidation_design_artifact_kind_available": design_report.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_consolidation_design",
            "rollout_consolidation_design_owned_by_hecsn": bool(design_report.get("owned_by_hecsn")),
            "rollout_consolidation_design_gate_ready": bool(
                design_gate.get("eligible_for_operator_rollout_consolidation_design_review")
            ),
            "rollout_consolidation_design_hash_available": bool(
                design_report.get("rollout_consolidation_design_hash")
            ),
            "rollout_consolidation_design_hash_matches": recomputed_design_hash
            == str(design_report.get("rollout_consolidation_design_hash") or ""),
            "rollout_consolidation_shadow_delta_surface_available": report.get("surface")
            == "snn_language_readout_rollout_consolidation_shadow_delta.v1",
            "rollout_consolidation_shadow_delta_artifact_kind_available": report.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_consolidation_shadow_delta",
            "rollout_consolidation_shadow_delta_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "rollout_consolidation_shadow_delta_gate_ready": bool(
                gate.get("eligible_for_operator_rollout_consolidation_shadow_review")
            ),
            "rollout_consolidation_shadow_delta_hash_available": bool(
                report.get("rollout_consolidation_shadow_delta_hash")
            ),
            "rollout_consolidation_shadow_delta_hash_matches": recomputed_shadow_hash
            == str(report.get("rollout_consolidation_shadow_delta_hash") or ""),
            "shadow_design_hash_matches": report.get("rollout_consolidation_design_hash")
            == design_report.get("rollout_consolidation_design_hash"),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state"))
            and not bool(shadow.get("runtime_update_applied")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "weight_persistence_absent": not bool(shadow.get("weights_persisted")),
            "checkpoint_write_absent": not bool(shadow.get("checkpoint_written")),
            "structural_write_absent": not bool(shadow.get("structural_write_applied")),
            "shadow_synapses_available": bool(bounded_synapses),
            "candidate_payload_not_truncated": 0 < len(bounded_synapses)
            <= _MAX_STRUCTURAL_EDGES_PER_EVENT,
            "candidate_payload_well_formed": len(parsed_synapses) == len(bounded_synapses),
            "affected_synapse_count_matches": len(bounded_synapses)
            == int(report.get("affected_synapse_count", 0) or 0),
            "nonzero_count_matches": verifier_nonzero_count
            == int(shadow.get("nonzero_count", 0) or 0),
            "reported_max_abs_delta_matches": abs(recomputed_max_abs_delta - reported_max_abs_delta)
            <= 1e-6,
            "candidate_deltas_finite": bool(parsed_synapses)
            and all(math.isfinite(delta) for _item, _source, _target, delta in parsed_synapses),
            "candidate_deltas_nonzero": bool(parsed_synapses)
            and all(delta != 0.0 for _item, _source, _target, delta in parsed_synapses),
            "candidate_deltas_within_design_bound": bool(parsed_synapses)
            and all(
                abs(delta) <= float(design.get("max_weight_delta", 0.0) or 0.0)
                for _item, _source, _target, delta in parsed_synapses
            ),
            "local_only_synapses": bool(bounded_synapses)
            and all(bool(item.get("local_only")) for item in bounded_synapses),
            "normalization_enabled": bool(bounded_synapses)
            and all(bool(item.get("normalization")) for item in bounded_synapses),
            "homeostatic_decay_bounded": bool(bounded_synapses)
            and all(0.0 <= float(item.get("homeostatic_decay", 0.0) or 0.0) <= 0.25 for item in bounded_synapses),
            "synapse_ids_unique": bool(synapse_ids)
            and all(synapse_ids)
            and len(set(synapse_ids)) == len(synapse_ids),
            "synapse_coordinates_unique": bool(coordinate_pairs)
            and len(set(coordinate_pairs)) == len(coordinate_pairs),
            "synapse_coordinates_canonical": bool(coordinate_pairs)
            and all(
                0 <= source < _LANGUAGE_NEURON_COUNT and 0 <= target < _LANGUAGE_NEURON_COUNT
                for source, target in coordinate_pairs
            ),
            "shadow_tensor_shape_exact": list(shadow.get("tensor_shape") or [])
            == [_LANGUAGE_NEURON_COUNT, _LANGUAGE_NEURON_COUNT],
            "requested_cuda_honored": bool(device.get("requested_cuda_honored")),
            "verifier_device_matches_shadow": verifier_tensor_device == shadow_tensor_device,
            "silent_device_fallback_absent": (
                requested_device == shadow_tensor_device
                or (requested_device.startswith("cuda") and bool(device.get("requested_cuda_honored")))
            ),
            "rollback_snapshot_available": bool(bound_snapshot_id),
            "topology_keyset_unchanged": topology_keyset_unchanged,
            "outgoing_row_mass_bounded": max(row_mass.values(), default=0.0)
            <= _MAX_OUTGOING_ROW_MASS,
            "outgoing_fanout_bounded": max(fanout.values(), default=0) <= _MAX_OUTGOING_FANOUT,
            "global_sparse_edge_budget_bounded": len(simulated) <= _MAX_SPARSE_TRANSITION_EDGES,
        }
        ready = all(required.values())
        preflight_hash = self._sha256_json(
            {
                "rollout_consolidation_shadow_delta_hash": report.get(
                    "rollout_consolidation_shadow_delta_hash"
                ),
                "rollback_snapshot_id": bound_snapshot_id,
                "transition_memory_snapshot_hash": self._sha256_json(weights),
                "required_evidence": required,
            }
        )
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_consolidation_shadow_application_preflight",
            "surface": "snn_language_readout_rollout_consolidation_shadow_application_preflight.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rollout_consolidation_shadow_application_preflight",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rollout_consolidation_design_hash": design_report.get(
                "rollout_consolidation_design_hash"
            ),
            "rollout_consolidation_shadow_delta_surface": report.get("surface"),
            "rollout_consolidation_shadow_delta_hash": report.get(
                "rollout_consolidation_shadow_delta_hash"
            ),
            "rollout_consolidation_shadow_application_preflight_hash": preflight_hash,
            "integrity_evidence": required,
            "device_evidence": {
                "requested_device": requested_device,
                "shadow_tensor_device": shadow_tensor_device,
                "verifier_tensor_device": verifier_tensor_device,
                "cuda_tensor": bool(verifier_tensor.device.type == "cuda"),
                "requested_cuda_honored": bool(device.get("requested_cuda_honored")),
                "verifier_device_matches_shadow": verifier_tensor_device == shadow_tensor_device,
                "observed_by_hecsn": True,
                "silent_fallback_absent": required["silent_device_fallback_absent"],
            },
            "rollback_evidence": {
                "rollback_snapshot_id": bound_snapshot_id or None,
                "rollback_snapshot_id_available": bool(bound_snapshot_id),
                "transition_memory_snapshot_hash": self._sha256_json(weights),
                "runtime_state_revision": int(runtime_snapshot.get("state_revision", 0) or 0),
                "checkpoint_restore_verified": False,
                "checkpoint_transaction_required_before_live_application": True,
            },
            "topology_evidence": {
                "existing_sparse_synapse_count": len(weights),
                "simulated_sparse_synapse_count": len(simulated),
                "candidate_synapse_count": len(bounded_synapses),
                "unique_candidate_synapse_count": len(set(coordinate_pairs)),
                "growth_candidate_count": len(set(growth_candidates)),
                "prune_candidate_count": len(set(prune_candidates)),
                "topology_keyset_unchanged": topology_keyset_unchanged,
                "outgoing_row_mass_bounded": required["outgoing_row_mass_bounded"],
                "outgoing_fanout_bounded": required["outgoing_fanout_bounded"],
                "global_sparse_edge_budget_bounded": required["global_sparse_edge_budget_bounded"],
                "structural_mutation_ledger_write_applied": False,
            },
            "preflight_summary": {
                "affected_synapse_count": len(bounded_synapses),
                "max_abs_delta": float(recomputed_max_abs_delta),
                "functional_weight_write_applied": False,
                "structural_growth_applied": False,
                "structural_pruning_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "runtime_update_applied": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_consolidation_shadow_application_preflight_evidence",
                "eligible_for_operator_rollout_consolidation_shadow_application_preflight_review": ready,
                "eligible_for_structural_growth_review": bool(growth_candidates),
                "eligible_for_shadow_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_consolidation_shadow_application_preflight"
                if ready
                else "collect_rollout_consolidation_shadow_delta_evidence",
                "required_evidence": required,
            },
        }

    def rollout_developmental_plasticity_review(
        self,
        rollout_consolidation_design: Mapping[str, Any],
        rollout_consolidation_shadow_application_preflight: Mapping[str, Any],
        *,
        transition_memory_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Review rollout growth candidates without growing or pruning synapses."""

        design_report = dict(rollout_consolidation_design)
        design_gate = (
            design_report.get("promotion_gate")
            if isinstance(design_report.get("promotion_gate"), Mapping)
            else {}
        )
        design = (
            design_report.get("rollout_consolidation_design")
            if isinstance(design_report.get("rollout_consolidation_design"), Mapping)
            else {}
        )
        design_material = (
            design_report.get("rollout_consolidation_design_material")
            if isinstance(design_report.get("rollout_consolidation_design_material"), Mapping)
            else {}
        )
        preflight = dict(rollout_consolidation_shadow_application_preflight)
        preflight_gate = (
            preflight.get("promotion_gate")
            if isinstance(preflight.get("promotion_gate"), Mapping)
            else {}
        )
        preflight_required = (
            preflight_gate.get("required_evidence")
            if isinstance(preflight_gate.get("required_evidence"), Mapping)
            else {}
        )
        preflight_rollback = (
            preflight.get("rollback_evidence")
            if isinstance(preflight.get("rollback_evidence"), Mapping)
            else {}
        )
        topology = (
            preflight.get("topology_evidence")
            if isinstance(preflight.get("topology_evidence"), Mapping)
            else {}
        )
        memory = dict(transition_memory_state)
        weights = {
            str(key): float(value)
            for key, value in dict(memory.get("sparse_transition_weights") or {}).items()
        }
        transition_memory_snapshot_hash = self._sha256_json(weights)
        runtime_snapshot = self._runtime_state.snapshot()
        raw_candidates = [
            dict(item)
            for item in list(design.get("candidate_synapses") or [])
            if isinstance(item, Mapping)
        ]
        candidates = [
            dict(item)
            for item in raw_candidates[: _MAX_STRUCTURAL_EDGES_PER_EVENT]
        ]
        growth_candidates: list[dict[str, Any]] = []
        parsed_candidates: list[tuple[int, int, float]] = []
        max_weight_delta = float(design.get("max_weight_delta", 0.0) or 0.0)
        for item in candidates:
            try:
                source = int(item.get("source_neuron_index", item.get("source_step_index")))
                target = int(item.get("target_neuron_index", item.get("target_step_index")))
                delta = float(item.get("proposed_weight_delta"))
            except (TypeError, ValueError):
                continue
            parsed_candidates.append((source, target, delta))
            key = f"{source}:{target}"
            if key in weights:
                continue
            growth_candidates.append(
                {
                    "synapse": key,
                    "pre_index": source,
                    "post_index": target,
                    "initial_weight": max(0.0, min(abs(delta), max_weight_delta, 0.25)),
                    "locality_distance": abs(target - source),
                    "source_synapse_id": item.get("synapse_id"),
                    "source_trace_index": item.get("source_trace_index"),
                    "source_rollout_step_index": item.get("source_rollout_step_index"),
                    "target_rollout_step_index": item.get("target_rollout_step_index"),
                    "source_active_indices_hash": item.get("source_active_indices_hash"),
                    "target_active_indices_hash": item.get("target_active_indices_hash"),
                    "local_only": bool(item.get("local_only")),
                    "normalization": bool(item.get("normalization")),
                    "applied_to_runtime": False,
                }
            )
        synapse_ids = [str(item.get("source_synapse_id") or "") for item in growth_candidates]
        candidate_keys = [str(item.get("synapse") or "") for item in growth_candidates]
        simulated = dict(weights)
        for item in growth_candidates:
            simulated[str(item["synapse"])] = float(item["initial_weight"])
        fanout: dict[str, int] = {}
        row_mass: dict[str, float] = {}
        for key, value in simulated.items():
            source = key.split(":", maxsplit=1)[0]
            fanout[source] = fanout.get(source, 0) + 1
            row_mass[source] = row_mass.get(source, 0.0) + abs(float(value))
        recomputed_design_hash = self._sha256_json(design_material)
        recomputed_preflight_hash = self._sha256_json(
            {
                "rollout_consolidation_shadow_delta_hash": preflight.get(
                    "rollout_consolidation_shadow_delta_hash"
                ),
                "rollback_snapshot_id": preflight_rollback.get("rollback_snapshot_id"),
                "transition_memory_snapshot_hash": preflight_rollback.get(
                    "transition_memory_snapshot_hash"
                ),
                "required_evidence": dict(preflight_required),
            }
        )
        topology_budget_evidence = {
            "existing_sparse_synapse_count": len(weights),
            "simulated_sparse_synapse_count": len(simulated),
            "growth_candidate_count": len(growth_candidates),
            "unique_growth_candidate_count": len(set(candidate_keys)),
            "prune_candidate_count": 0,
            "candidate_count_bounded": 0 < len(growth_candidates) <= _MAX_STRUCTURAL_EDGES_PER_EVENT,
            "outgoing_row_mass_bounded": max(row_mass.values(), default=0.0)
            <= _MAX_OUTGOING_ROW_MASS,
            "outgoing_fanout_bounded": max(fanout.values(), default=0) <= _MAX_OUTGOING_FANOUT,
            "global_sparse_edge_budget_bounded": len(simulated) <= _MAX_SPARSE_TRANSITION_EDGES,
            "structural_mutation_ledger_write_applied": False,
        }
        integrity_evidence = {
            "design_surface_available": design_report.get("surface")
            == "snn_language_readout_rollout_consolidation_design.v1",
            "design_artifact_kind_available": design_report.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_consolidation_design",
            "design_owned_by_hecsn": bool(design_report.get("owned_by_hecsn")),
            "design_gate_ready": bool(
                design_gate.get("eligible_for_operator_rollout_consolidation_design_review")
            ),
            "design_hash_available": bool(design_report.get("rollout_consolidation_design_hash")),
            "design_hash_recomputed_match": recomputed_design_hash
            == str(design_report.get("rollout_consolidation_design_hash") or ""),
            "preflight_surface_available": preflight.get("surface")
            == "snn_language_readout_rollout_consolidation_shadow_application_preflight.v1",
            "preflight_artifact_kind_available": preflight.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_consolidation_shadow_application_preflight",
            "preflight_owned_by_hecsn": bool(preflight.get("owned_by_hecsn")),
            "preflight_hash_available": bool(
                preflight.get("rollout_consolidation_shadow_application_preflight_hash")
            ),
            "preflight_hash_recomputed_match": recomputed_preflight_hash
            == str(preflight.get("rollout_consolidation_shadow_application_preflight_hash") or ""),
            "preflight_design_hash_matches": preflight.get("rollout_consolidation_design_hash")
            == design_report.get("rollout_consolidation_design_hash"),
            "preflight_growth_review_requested": bool(
                preflight_gate.get("eligible_for_structural_growth_review")
            ),
            "preflight_blocks_mutation": not bool(preflight_gate.get("eligible_for_structural_write"))
            and not bool(preflight_gate.get("eligible_for_growth"))
            and not bool(preflight_gate.get("eligible_for_plasticity_application")),
            "runtime_mutation_absent": not bool(preflight.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(preflight.get("applies_plasticity")),
            "checkpoint_write_absent": not bool(
                preflight.get("preflight_summary", {}).get("checkpoint_written")
            )
            if isinstance(preflight.get("preflight_summary"), Mapping)
            else False,
            "weights_persistence_absent": not bool(
                preflight.get("preflight_summary", {}).get("weights_persisted")
            )
            if isinstance(preflight.get("preflight_summary"), Mapping)
            else False,
            "candidate_payload_not_truncated": len(raw_candidates) == len(candidates),
            "candidate_payload_well_formed": len(parsed_candidates) == len(candidates),
            "candidate_ids_unique": bool(synapse_ids)
            and all(synapse_ids)
            and len(set(synapse_ids)) == len(synapse_ids),
            "candidate_keys_unique": bool(candidate_keys)
            and len(set(candidate_keys)) == len(candidate_keys),
            "candidate_indices_canonical": all(
                0 <= source < _LANGUAGE_NEURON_COUNT
                and 0 <= target < _LANGUAGE_NEURON_COUNT
                for source, target, _delta in parsed_candidates
            ),
            "candidate_indices_in_range": all(
                0 <= int(item.get("pre_index", -1)) < _LANGUAGE_NEURON_COUNT
                and 0 <= int(item.get("post_index", -1)) < _LANGUAGE_NEURON_COUNT
                for item in growth_candidates
            ),
            "candidate_values_finite": all(
                math.isfinite(float(item.get("initial_weight", 0.0) or 0.0))
                for item in growth_candidates
            ),
            "candidate_values_positive": all(
                float(item.get("initial_weight", 0.0) or 0.0) > 0.0
                for item in growth_candidates
            ),
            "candidate_locality_temporal": all(
                int(item.get("locality_distance", 0) or 0) <= 8 for item in growth_candidates
            ),
            "candidate_rollout_step_provenance_available": all(
                item.get("source_rollout_step_index") is not None
                and item.get("target_rollout_step_index") is not None
                and int(item.get("target_rollout_step_index", 0) or 0)
                > int(item.get("source_rollout_step_index", -1) or -1)
                for item in growth_candidates
            ),
            "candidate_active_hash_provenance_available": all(
                bool(str(item.get("source_active_indices_hash") or ""))
                and bool(str(item.get("target_active_indices_hash") or ""))
                for item in growth_candidates
            ),
            "candidate_local_only": all(bool(item.get("local_only")) for item in growth_candidates),
            "normalization_enabled": all(bool(item.get("normalization")) for item in growth_candidates),
        }
        runtime_memory_evidence = {
            "surface": memory.get("surface"),
            "owned_by_hecsn": bool(memory.get("owned_by_hecsn", True)),
            "runtime_state_revision": int(runtime_snapshot.get("state_revision", 0) or 0),
            "transition_memory_snapshot_hash": transition_memory_snapshot_hash,
            "existing_sparse_synapse_count": len(weights),
            "candidate_synapses_absent_from_runtime": all(
                str(item.get("synapse")) not in weights for item in growth_candidates
            ),
            "runtime_mutation_applied": False,
        }
        growth_classification = {
            "growth_candidate_count": len(growth_candidates),
            "unique_growth_candidate_count": len(set(candidate_keys)),
            "existing_edge_candidate_count": max(0, len(parsed_candidates) - len(growth_candidates)),
            "prune_candidate_count": 0,
            "all_candidates_are_growth": bool(growth_candidates)
            and len(growth_candidates) == len(parsed_candidates),
            "growth_candidates": growth_candidates,
        }
        review_hash = self._sha256_json(
            {
                "rollout_consolidation_design_hash": design_report.get(
                    "rollout_consolidation_design_hash"
                ),
                "rollout_consolidation_shadow_application_preflight_hash": preflight.get(
                    "rollout_consolidation_shadow_application_preflight_hash"
                ),
                "transition_memory_snapshot_hash": transition_memory_snapshot_hash,
                "growth_candidates": growth_candidates,
            }
        )
        required = {
            **integrity_evidence,
            "runtime_transition_memory_owned_by_hecsn": memory.get("surface")
            in {None, "snn_language_plasticity_runtime_state.v1"}
            and bool(memory.get("owned_by_hecsn", True)),
            "growth_pressure_available": int(topology.get("growth_candidate_count", 0) or 0) > 0,
            "growth_candidates_available": bool(growth_candidates),
            "growth_candidate_count_matches_preflight": len(growth_candidates)
            == int(topology.get("growth_candidate_count", 0) or len(growth_candidates)),
            "candidate_weights_bounded": bool(growth_candidates)
            and all(
                math.isfinite(float(item.get("initial_weight", 0.0) or 0.0))
                and 0.0 < float(item.get("initial_weight", 0.0) or 0.0) <= 0.25
                for item in growth_candidates
            ),
            "candidate_synapses_absent_from_runtime": runtime_memory_evidence[
                "candidate_synapses_absent_from_runtime"
            ],
            **{
                key: value
                for key, value in topology_budget_evidence.items()
                if key
                not in {
                    "existing_sparse_synapse_count",
                    "simulated_sparse_synapse_count",
                    "growth_candidate_count",
                    "unique_growth_candidate_count",
                    "prune_candidate_count",
                    "structural_mutation_ledger_write_applied",
                }
            },
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_developmental_plasticity_review",
            "surface": "snn_language_readout_rollout_developmental_plasticity_review.v1",
            "available": bool(preflight),
            "source": "service.snn_language_readout_ledger.rollout_developmental_plasticity_review",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rollout_consolidation_design_hash": design_report.get(
                "rollout_consolidation_design_hash"
            ),
            "rollout_consolidation_shadow_application_preflight_hash": preflight.get(
                "rollout_consolidation_shadow_application_preflight_hash"
            ),
            "rollout_developmental_plasticity_review_hash": review_hash,
            "integrity_evidence": integrity_evidence,
            "runtime_memory_evidence": runtime_memory_evidence,
            "growth_classification": growth_classification,
            "topology_budget_evidence": topology_budget_evidence,
            "developmental_plasticity_review": {
                "growth_candidate_count": len(growth_candidates),
                "growth_candidates": growth_candidates,
                "existing_sparse_synapse_count": len(weights),
                "simulated_sparse_synapse_count": len(simulated),
                "max_outgoing_row_mass": max(row_mass.values(), default=0.0),
                "max_outgoing_fanout": max(fanout.values(), default=0),
                "structural_growth_applied": False,
                "structural_pruning_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "runtime_update_applied": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_rollout_developmental_plasticity_evidence",
                "eligible_for_operator_rollout_developmental_plasticity_review": ready,
                "eligible_for_transition_memory_regeneration_proposal": ready,
                "eligible_for_structural_growth_review": ready,
                "eligible_for_shadow_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_developmental_plasticity"
                if ready
                else "collect_rollout_growth_candidate_evidence",
                "required_evidence": required,
            },
        }

    def rollout_regeneration_proposal_adapter(
        self,
        rollout_developmental_plasticity_review: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Adapt rollout growth review into a regeneration-design preview without permits."""

        review = dict(rollout_developmental_plasticity_review)
        gate = review.get("promotion_gate") if isinstance(review.get("promotion_gate"), Mapping) else {}
        developmental = (
            review.get("developmental_plasticity_review")
            if isinstance(review.get("developmental_plasticity_review"), Mapping)
            else {}
        )
        topology = (
            review.get("topology_budget_evidence")
            if isinstance(review.get("topology_budget_evidence"), Mapping)
            else {}
        )
        growth = (
            review.get("growth_classification")
            if isinstance(review.get("growth_classification"), Mapping)
            else {}
        )
        runtime_memory = (
            review.get("runtime_memory_evidence")
            if isinstance(review.get("runtime_memory_evidence"), Mapping)
            else {}
        )
        raw_candidates = [
            dict(item)
            for item in list(developmental.get("growth_candidates") or [])
            if isinstance(item, Mapping)
        ]
        candidates: list[dict[str, Any]] = []
        for item in raw_candidates[: _MAX_STRUCTURAL_EDGES_PER_EVENT]:
            try:
                pre_index = int(item.get("pre_index"))
                post_index = int(item.get("post_index"))
                initial_weight = float(item.get("initial_weight"))
            except (TypeError, ValueError):
                continue
            distance = abs(post_index - pre_index)
            candidates.append(
                {
                    "pre_index": pre_index,
                    "post_index": post_index,
                    "synapse": f"{pre_index}:{post_index}",
                    "initial_weight": initial_weight,
                    "locality_distance": distance,
                    "rollout_developmental_review_hash": review.get(
                        "rollout_developmental_plasticity_review_hash"
                    ),
                    "source_synapse_id": item.get("source_synapse_id"),
                    "source_trace_index": item.get("source_trace_index"),
                    "source_rollout_step_index": item.get("source_rollout_step_index"),
                    "target_rollout_step_index": item.get("target_rollout_step_index"),
                    "source_active_indices_hash": item.get("source_active_indices_hash"),
                    "target_active_indices_hash": item.get("target_active_indices_hash"),
                }
            )
        candidates.sort(key=lambda item: (item["pre_index"], item["post_index"]))
        max_distance = max([int(item["locality_distance"]) for item in candidates], default=1)
        locality_radius = max(1, min(8, max_distance))
        initial_weight = max([float(item["initial_weight"]) for item in candidates], default=0.02)
        regeneration_design = {
            "locality_radius": locality_radius,
            "initial_weight": float(initial_weight),
            "max_new_synapses": min(_MAX_STRUCTURAL_EDGES_PER_EVENT, max(1, len(candidates))),
            "mismatch_score": 0.0,
            "candidate_count": len(candidates),
            "candidate_synapses": candidates,
        }
        recomputed_review_hash = self._sha256_json(
            {
                "rollout_consolidation_design_hash": review.get(
                    "rollout_consolidation_design_hash"
                ),
                "rollout_consolidation_shadow_application_preflight_hash": review.get(
                    "rollout_consolidation_shadow_application_preflight_hash"
                ),
                "transition_memory_snapshot_hash": runtime_memory.get(
                    "transition_memory_snapshot_hash"
                ),
                "growth_candidates": raw_candidates,
            }
        )
        adapter_hash = self._sha256_json(
            {
                "rollout_developmental_plasticity_review_hash": review.get(
                    "rollout_developmental_plasticity_review_hash"
                ),
                "regeneration_design": regeneration_design,
            }
        )
        candidate_ids = [str(item.get("source_synapse_id") or "") for item in raw_candidates]
        candidate_keys = [str(item.get("synapse") or "") for item in candidates]
        rollout_growth_evidence = {
            "rollout_developmental_plasticity_review_hash": review.get(
                "rollout_developmental_plasticity_review_hash"
            ),
            "growth_candidate_count": len(candidates),
            "all_candidates_are_growth": bool(growth.get("all_candidates_are_growth")),
            "rollout_growth_pressure_score": min(
                1.0,
                0.66 + 0.02 * min(len(candidates), _MAX_STRUCTURAL_EDGES_PER_EVENT),
            )
            if candidates
            else 0.0,
            "transition_memory_snapshot_hash": runtime_memory.get(
                "transition_memory_snapshot_hash"
            ),
        }
        blocked_replay_evidence = {
            "available": False,
            "ready": False,
            "artifact_kind": None,
            "surface": None,
            "source": "readout_rollout_adapter.blocked_until_replay_controller_permit",
            "permit_id": None,
            "replay_window_id": None,
            "evidence_hash": None,
            "replay_artifact_id": None,
            "replay_artifact_hash": None,
        }
        executor_bypass_evidence = {
            "compatible_with_regeneration_design_shape": True,
            "is_transition_memory_regeneration_proposal": False,
            "is_regeneration_permit": False,
            "replay_controller_permit_required": True,
            "checkpoint_executor_required": True,
            "direct_executor_submission_expected_to_block": True,
            "reason": "missing_server_verified_replay_regeneration_permit",
        }
        required = {
            "review_surface_available": review.get("surface")
            == "snn_language_readout_rollout_developmental_plasticity_review.v1",
            "review_artifact_kind_available": review.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_developmental_plasticity_review",
            "review_owned_by_hecsn": bool(review.get("owned_by_hecsn")),
            "review_hash_available": bool(review.get("rollout_developmental_plasticity_review_hash")),
            "review_hash_recomputed_match": recomputed_review_hash
            == str(review.get("rollout_developmental_plasticity_review_hash") or ""),
            "review_gate_ready": bool(
                gate.get("eligible_for_operator_rollout_developmental_plasticity_review")
            ),
            "review_regeneration_proposal_eligible": bool(
                gate.get("eligible_for_transition_memory_regeneration_proposal")
            ),
            "review_blocks_mutation": not bool(review.get("mutates_runtime_state")),
            "review_blocks_growth": not bool(gate.get("eligible_for_growth"))
            and not bool(developmental.get("structural_growth_applied")),
            "review_blocks_pruning": not bool(gate.get("eligible_for_pruning"))
            and not bool(developmental.get("structural_pruning_applied")),
            "developmental_review_hash_available": bool(
                review.get("rollout_developmental_plasticity_review_hash")
            ),
            "growth_candidates_available": bool(candidates),
            "growth_candidate_count_matches_review": len(candidates)
            == int(developmental.get("growth_candidate_count", 0) or 0),
            "all_candidates_are_growth": bool(growth.get("all_candidates_are_growth")),
            "candidate_payload_not_truncated": len(raw_candidates) == len(candidates),
            "candidate_payload_well_formed": len(candidates) == len(raw_candidates),
            "candidate_ids_unique": bool(candidate_ids)
            and all(candidate_ids)
            and len(set(candidate_ids)) == len(candidate_ids),
            "candidate_keys_unique": bool(candidate_keys)
            and len(set(candidate_keys)) == len(candidate_keys),
            "candidate_indices_canonical": all(
                0 <= int(item["pre_index"]) < _LANGUAGE_NEURON_COUNT
                and 0 <= int(item["post_index"]) < _LANGUAGE_NEURON_COUNT
                for item in candidates
            ),
            "candidate_indices_in_range": all(
                0 <= int(item["pre_index"]) < _LANGUAGE_NEURON_COUNT
                and 0 <= int(item["post_index"]) < _LANGUAGE_NEURON_COUNT
                for item in candidates
            ),
            "candidate_values_finite": all(
                math.isfinite(float(item["initial_weight"])) for item in candidates
            ),
            "candidate_values_positive": all(
                float(item["initial_weight"]) > 0.0 for item in candidates
            ),
            "candidate_weights_bounded": all(
                0.0 < float(item["initial_weight"]) <= 0.25 for item in candidates
            ),
            "candidate_locality_temporal": all(
                0 <= int(item["locality_distance"]) <= locality_radius <= 8
                for item in candidates
            ),
            "candidate_rollout_step_provenance_available": all(
                item.get("source_rollout_step_index") is not None
                and item.get("target_rollout_step_index") is not None
                and int(item.get("target_rollout_step_index", 0) or 0)
                > int(item.get("source_rollout_step_index", -1) or -1)
                for item in candidates
            ),
            "candidate_active_hash_provenance_available": all(
                bool(str(item.get("source_active_indices_hash") or ""))
                and bool(str(item.get("target_active_indices_hash") or ""))
                for item in candidates
            ),
            "candidate_local_only": all(bool(item.get("local_only", True)) for item in raw_candidates),
            "normalization_enabled": all(bool(item.get("normalization", True)) for item in raw_candidates),
            "topology_budget_available": bool(topology),
            "topology_candidate_count_bounded": bool(topology.get("candidate_count_bounded")),
            "topology_row_mass_bounded": bool(topology.get("outgoing_row_mass_bounded")),
            "topology_fanout_bounded": bool(topology.get("outgoing_fanout_bounded")),
            "topology_global_edge_budget_bounded": bool(
                topology.get("global_sparse_edge_budget_bounded")
            ),
            "topology_budgets_passed": bool(topology.get("candidate_count_bounded"))
            and bool(topology.get("outgoing_row_mass_bounded"))
            and bool(topology.get("outgoing_fanout_bounded"))
            and bool(topology.get("global_sparse_edge_budget_bounded")),
            "runtime_mutation_absent": not bool(review.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(review.get("applies_plasticity")),
            "regeneration_permit_absent": "permit_id" not in developmental,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_proposal_adapter",
            "surface": "snn_language_readout_rollout_regeneration_proposal_adapter.v1",
            "available": bool(review),
            "source": "service.snn_language_readout_ledger.rollout_regeneration_proposal_adapter",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "issues_regeneration_permit": False,
            "executor_ready": False,
            "rollout_developmental_plasticity_review_hash": review.get(
                "rollout_developmental_plasticity_review_hash"
            ),
            "rollout_regeneration_proposal_adapter_hash": adapter_hash,
            "integrity_evidence": required,
            "rollout_growth_evidence": rollout_growth_evidence,
            "regeneration_design": regeneration_design,
            "blocked_replay_evidence": blocked_replay_evidence,
            "executor_bypass_evidence": executor_bypass_evidence,
            "promotion_gate": {
                "status": "ready_for_operator_replay_artifact_review"
                if ready
                else "blocked_missing_rollout_regeneration_adapter_evidence",
                "eligible_for_operator_rollout_regeneration_adapter_review": ready,
                "eligible_for_replay_artifact_recording_review": ready,
                "eligible_for_regeneration_permit_request": False,
                "eligible_for_transition_memory_regeneration_permit_request": False,
                "eligible_for_regeneration_application": False,
                "eligible_for_transition_memory_regeneration_application": False,
                "eligible_for_shadow_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_rollout_regeneration_adapter"
                if ready
                else "collect_rollout_developmental_plasticity_evidence",
                "required_evidence": required,
            },
        }

    def rollout_regeneration_replay_artifact_review(
        self,
        rollout_regeneration_proposal_adapter: Mapping[str, Any],
        snn_transition_memory_replay_artifact: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Bind rollout regeneration design preview to replay artifact evidence without permits."""

        adapter = dict(rollout_regeneration_proposal_adapter)
        adapter_gate = (
            adapter.get("promotion_gate")
            if isinstance(adapter.get("promotion_gate"), Mapping)
            else {}
        )
        replay = dict(snn_transition_memory_replay_artifact)
        design = (
            adapter.get("regeneration_design")
            if isinstance(adapter.get("regeneration_design"), Mapping)
            else {}
        )
        candidates = [
            dict(item)
            for item in list(design.get("candidate_synapses") or [])
            if isinstance(item, Mapping)
        ]
        blocked_replay = (
            adapter.get("blocked_replay_evidence")
            if isinstance(adapter.get("blocked_replay_evidence"), Mapping)
            else {}
        )
        executor_bypass = (
            adapter.get("executor_bypass_evidence")
            if isinstance(adapter.get("executor_bypass_evidence"), Mapping)
            else {}
        )
        recomputed_adapter_hash = self._sha256_json(
            {
                "rollout_developmental_plasticity_review_hash": adapter.get(
                    "rollout_developmental_plasticity_review_hash"
                ),
                "regeneration_design": design,
            }
        )
        replay_material = {
            "recorded_state_revision": int(replay.get("recorded_state_revision", -1)),
            "operator_id": replay.get("operator_id"),
            "confirmation": bool(replay.get("confirmation")),
            "mismatch_hash": replay.get("mismatch_hash"),
            "mismatch_score": float(replay.get("mismatch_score", 0.0) or 0.0),
            "pressure_hash": replay.get("pressure_hash"),
            "pressure_score": float(replay.get("pressure_score", 0.0) or 0.0),
            "replay_window_hash": replay.get("replay_window_hash"),
            "replay_window_size": int(replay.get("replay_window_size", 0) or 0),
            "internal_ledger_backed": bool(replay.get("internal_ledger_backed")),
            "artifact_proposal_hash": replay.get("artifact_proposal_hash"),
            "replay_evaluation_context_id": replay.get("replay_evaluation_context_id"),
            "replay_evaluation_context_hash": replay.get("replay_evaluation_context_hash"),
            "review_ticket_id": replay.get("review_ticket_id"),
            "review_ticket_hash": replay.get("review_ticket_hash"),
            "readout_evidence_hashes": list(replay.get("readout_evidence_hashes") or []),
        }
        recomputed_replay_hash = self._sha256_json(replay_material)
        replay_mismatch_score = max(
            0.0,
            min(1.0, float(replay_material.get("mismatch_score", 0.0) or 0.0)),
        )
        replay_pressure_score = max(
            0.0,
            min(1.0, float(replay_material.get("pressure_score", 0.0) or 0.0)),
        )
        replay_bound_design = deepcopy(design)
        replay_bound_design["mismatch_score"] = replay_mismatch_score
        design_hash = self._sha256_json(replay_bound_design)
        review_hash = self._sha256_json(
            {
                "rollout_regeneration_proposal_adapter_hash": adapter.get(
                    "rollout_regeneration_proposal_adapter_hash"
                ),
                "snn_transition_memory_replay_artifact_hash": replay.get("evidence_hash"),
                "regeneration_design_hash": design_hash,
            }
        )
        required = {
            "adapter_surface_available": adapter.get("surface")
            == "snn_language_readout_rollout_regeneration_proposal_adapter.v1",
            "adapter_artifact_kind_available": adapter.get("artifact_kind")
            == "terminus_snn_language_readout_rollout_regeneration_proposal_adapter",
            "adapter_owned_by_hecsn": bool(adapter.get("owned_by_hecsn")),
            "adapter_gate_ready": bool(
                adapter_gate.get("eligible_for_operator_rollout_regeneration_adapter_review")
            ),
            "adapter_hash_available": bool(adapter.get("rollout_regeneration_proposal_adapter_hash")),
            "adapter_hash_recomputed_match": recomputed_adapter_hash
            == str(adapter.get("rollout_regeneration_proposal_adapter_hash") or ""),
            "adapter_blocks_mutation": not bool(adapter.get("mutates_runtime_state"))
            and not bool(adapter.get("applies_plasticity"))
            and not bool(adapter.get("issues_regeneration_permit"))
            and not bool(adapter.get("executor_ready")),
            "blocked_replay_evidence_present": bool(blocked_replay)
            and blocked_replay.get("ready") is False
            and not bool(blocked_replay.get("permit_id")),
            "executor_bypass_declared": bool(executor_bypass)
            and bool(executor_bypass.get("replay_controller_permit_required"))
            and bool(executor_bypass.get("checkpoint_executor_required"))
            and bool(executor_bypass.get("direct_executor_submission_expected_to_block")),
            "regeneration_design_available": bool(design),
            "regeneration_design_candidate_count_matches": len(candidates)
            == int(design.get("candidate_count", 0) or 0),
            "regeneration_design_candidate_count_bounded": 0 < len(candidates)
            <= _MAX_STRUCTURAL_EDGES_PER_EVENT,
            "regeneration_design_indices_canonical": all(
                0 <= int(item.get("pre_index", -1)) < _LANGUAGE_NEURON_COUNT
                and 0 <= int(item.get("post_index", -1)) < _LANGUAGE_NEURON_COUNT
                for item in candidates
            ),
            "regeneration_design_weights_bounded": all(
                0.0 < float(item.get("initial_weight", 0.0) or 0.0) <= 0.25
                for item in candidates
            ),
            "regeneration_design_locality_bounded": 1 <= int(design.get("locality_radius", 0) or 0) <= 8
            and all(
                int(item.get("locality_distance", 0) or 0)
                <= int(design.get("locality_radius", 0) or 0)
                for item in candidates
            ),
            "replay_artifact_surface_available": replay.get("surface")
            == "snn_transition_memory_replay_artifact.v1",
            "replay_artifact_kind_available": replay.get("artifact_kind")
            == "terminus_snn_transition_memory_replay_artifact",
            "replay_artifact_owned_by_hecsn": bool(replay.get("owned_by_hecsn")),
            "replay_artifact_ready": bool(replay.get("ready")),
            "replay_artifact_internal_ledger_backed": bool(replay.get("internal_ledger_backed")),
            "replay_artifact_hash_available": bool(replay.get("evidence_hash")),
            "replay_artifact_hash_recomputed_match": recomputed_replay_hash
            == str(replay.get("evidence_hash") or ""),
            "replay_mismatch_score_high": replay_mismatch_score >= 0.66,
            "replay_pressure_score_available": replay_pressure_score >= 0.0,
            "replay_readout_evidence_available": bool(list(replay.get("readout_evidence_hashes") or [])),
            "permit_absent": not bool(adapter.get("permit_id")) and not bool(replay.get("permit_id")),
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_replay_artifact_review",
            "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
            "available": bool(adapter) and bool(replay),
            "source": "service.snn_language_readout_ledger.rollout_regeneration_replay_artifact_review",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "issues_regeneration_permit": False,
            "executor_ready": False,
            "rollout_regeneration_proposal_adapter_hash": adapter.get(
                "rollout_regeneration_proposal_adapter_hash"
            ),
            "snn_transition_memory_replay_artifact_hash": replay.get("evidence_hash"),
            "regeneration_design_hash": design_hash,
            "rollout_regeneration_replay_artifact_review_hash": review_hash,
            "regeneration_design": deepcopy(replay_bound_design),
            "replay_mismatch_evidence": {
                "mismatch_score": replay_mismatch_score,
                "pressure_score": replay_pressure_score,
                "mismatch_hash": replay.get("mismatch_hash"),
                "pressure_hash": replay.get("pressure_hash"),
                "source": "snn_transition_memory_replay_artifact",
            },
            "replay_artifact_binding": {
                "replay_artifact_id": replay.get("replay_artifact_id"),
                "replay_window_id": replay.get("replay_window_id"),
                "replay_artifact_hash": replay.get("evidence_hash"),
                "readout_evidence_hashes": list(replay.get("readout_evidence_hashes") or []),
                "recorded_state_revision": replay.get("recorded_state_revision"),
                "internal_ledger_backed": bool(replay.get("internal_ledger_backed")),
            },
            "permit_request_preview": {
                "replay_artifact_id": replay.get("replay_artifact_id") if ready else None,
                "regeneration_design": deepcopy(replay_bound_design) if ready else None,
                "operator_id_required": True,
                "confirmation_required": True,
                "permit_issued": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_permit_request_review"
                if ready
                else "blocked_missing_rollout_regeneration_replay_artifact_evidence",
                "eligible_for_operator_rollout_regeneration_replay_artifact_review": ready,
                "eligible_for_regeneration_permit_request": ready,
                "eligible_for_regeneration_application": False,
                "eligible_for_structural_write": False,
                "eligible_for_growth": False,
                "eligible_for_pruning": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_confirmed_replay_controller_regeneration_permit_request"
                if ready
                else "collect_server_owned_snn_replay_artifact",
                "required_evidence": required,
            },
        }

    def transition_memory_replay_artifact_proposal(
        self,
        *,
        mismatch_report: Mapping[str, Any],
        pressure_report: Mapping[str, Any],
        limit: int = 8,
    ) -> dict[str, Any]:
        """Build a read-only grounded replay window from internal readout evidence."""

        mismatch = dict(mismatch_report)
        pressure = dict(pressure_report)
        error = mismatch.get("prediction_error") if isinstance(mismatch.get("prediction_error"), Mapping) else {}
        pressure_gate = (
            pressure.get("promotion_gate")
            if isinstance(pressure.get("promotion_gate"), Mapping)
            else {}
        )
        priority = self.replay_priority(limit=max(1, min(int(limit), 32)))
        replay_window = [
            {
                "readout_evidence_hash": item.get("readout_evidence_hash"),
                "source_readout_evidence_id": item.get("source_readout_evidence_id"),
                "prediction_hash": item.get("prediction_hash"),
                "transition_memory_evaluation_hash": item.get("transition_memory_evaluation_hash"),
                "persistent_transition_weights_hash": item.get("persistent_transition_weights_hash"),
                "labels": [str(value) for value in list(item.get("labels") or [])[:12]],
                "priority_score": float(item.get("priority_score", 0.0) or 0.0),
                "grounded": True,
            }
            for item in list(priority.get("candidates") or [])
            if isinstance(item, Mapping)
            and item.get("readout_evidence_hash")
            and item.get("all_labels_grounded")
        ]
        required = {
            "mismatch_available": bool(mismatch.get("available")),
            "mismatch_owned_by_hecsn": bool(mismatch.get("owned_by_hecsn")),
            "mismatch_score_high": float(error.get("mismatch_score", 0.0) or 0.0) >= 0.66,
            "pressure_available": bool(pressure.get("available")),
            "pressure_owned_by_hecsn": bool(pressure.get("owned_by_hecsn")),
            "pressure_gate_ready": str(pressure_gate.get("status") or "") == "ready_for_operator_review",
            "internal_readout_evidence_available": bool(replay_window),
            "replay_window_grounded": bool(replay_window)
            and all(bool(item.get("grounded")) for item in replay_window),
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_transition_memory_replay_artifact_proposal",
            "surface": "snn_transition_memory_replay_artifact_proposal.v1",
            "available": ready,
            "ready": ready,
            "owned_by_hecsn": True,
            "source": "service.snn_language_readout_ledger.transition_memory_replay_artifact_proposal",
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "mismatch_report": mismatch,
            "pressure_report": pressure,
            "replay_window": replay_window,
            "promotion_gate": {
                "status": "ready_for_operator_recording_review"
                if ready
                else "blocked_missing_internal_snn_replay_evidence",
                "eligible_for_operator_recording_review": ready,
                "eligible_for_structural_write": False,
                "requires_operator_approval": ready,
                "required_evidence": required,
            },
        }

    def rehearsal_evaluation(
        self,
        replay_priority_report: Mapping[str, Any],
        *,
        candidate_limit: int = 8,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate prioritized readout evidence as isolated sparse SNN rehearsal."""

        report = dict(replay_priority_report)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        device_report = dict(device_evidence or {})
        device = self._safe_tensor_device(str(device_report.get("device") or device_report.get("tensor_device") or "cpu"))
        raw_candidates = [
            dict(item)
            for item in list(report.get("candidates") or [])[: max(0, int(candidate_limit))]
            if isinstance(item, Mapping)
        ]
        vectors: list[torch.Tensor] = []
        traces: list[dict[str, Any]] = []
        priorities: list[float] = []
        for candidate in raw_candidates:
            labels = [str(value) for value in list(candidate.get("labels") or [])[:12]]
            active_indices = self._label_sparse_indices(labels)
            vector = torch.zeros(64, device=device)
            for index in active_indices:
                vector[int(index) % 64] = 1.0
            if active_indices:
                vector = vector / max(1.0, float(len(active_indices)))
            vectors.append(vector)
            priority_score = float(candidate.get("priority_score", 0.0) or 0.0)
            priorities.append(priority_score)
            traces.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "readout_evidence_hash": candidate.get("readout_evidence_hash"),
                    "prediction_hash": candidate.get("prediction_hash"),
                    "transition_memory_evaluation_hash": candidate.get("transition_memory_evaluation_hash"),
                    "persistent_transition_weights_hash": candidate.get("persistent_transition_weights_hash"),
                    "label_count": len(labels),
                    "sparse_active_indices": active_indices,
                    "priority_score": priority_score,
                    "applied_to_runtime": False,
                    "weights_persisted": False,
                    "generated_text": False,
                }
            )
        if vectors:
            stack = torch.stack(vectors)
            active_fraction = float((stack > 0).float().mean().item())
            activation_sparsity = 1.0 - active_fraction
            if stack.shape[0] > 1:
                normalized = torch.nn.functional.normalize(stack, p=2, dim=1, eps=1e-9)
                similarity = normalized @ normalized.T
                upper = similarity[torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)]
                mean_pairwise_similarity = float(upper.mean().item()) if upper.numel() else 1.0
            else:
                mean_pairwise_similarity = 1.0
        else:
            activation_sparsity = 1.0
            mean_pairwise_similarity = 0.0
        average_priority = sum(priorities) / max(1, len(priorities))
        repeated_support = sum(
            1
            for candidate in raw_candidates
            if "repeated_label_evidence" in set(candidate.get("reason_codes") or [])
        )
        rehearsal_stable = (
            bool(traces)
            and activation_sparsity >= 0.85
            and (len(traces) == 1 or mean_pairwise_similarity <= 0.95)
        )
        required = {
            "priority_surface_available": report.get("surface") == "snn_language_readout_replay_priority.v1",
            "priority_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "priority_gate_ready": bool(gate.get("eligible_for_operator_replay_review")),
            "priority_report_non_executable": not bool(report.get("executable")),
            "priority_report_non_mutating": not bool(report.get("mutates_runtime_state")),
            "candidates_available": bool(traces),
            "candidate_provenance_available": all(
                bool(trace.get("readout_evidence_hash"))
                and bool(trace.get("prediction_hash"))
                and bool(trace.get("transition_memory_evaluation_hash"))
                for trace in traces
            ) if traces else False,
            "average_priority_positive": average_priority > 0.0,
            "sparse_rehearsal_stable": rehearsal_stable,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rehearsal_evaluation",
            "surface": "snn_language_readout_rehearsal_evaluation.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rehearsal_evaluation",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "priority_surface": report.get("surface"),
            "device_evidence": {
                "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "cpu"),
                "tensor_device": str(device),
                "cuda_tensor": device.type == "cuda",
                "device_source": device_report.get("source") or device_report.get("device_source"),
            },
            "rehearsal_summary": {
                "candidate_count": len(traces),
                "repeated_support_candidate_count": int(repeated_support),
                "average_priority_score": float(average_priority),
                "activation_sparsity": float(activation_sparsity),
                "mean_pairwise_similarity": float(mean_pairwise_similarity),
                "sparse_rehearsal_stable": bool(rehearsal_stable),
            },
            "ephemeral_rehearsal": {
                "trace": traces,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "generated_text": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review" if ready else "blocked_missing_rehearsal_evidence",
                "eligible_for_operator_rehearsal_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_isolated_snn_readout_rehearsal"
                if ready
                else "collect_readout_replay_priority_candidates",
                "required_evidence": required,
            },
        }

    def rehearsal_experiment(
        self,
        rehearsal_evaluation: Mapping[str, Any],
        *,
        replay_cycles: int = 3,
        stability_floor: float = 0.85,
    ) -> dict[str, Any]:
        """Run an isolated replay-pressure simulation from rehearsal evidence."""

        report = dict(rehearsal_evaluation)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        summary = report.get("rehearsal_summary") if isinstance(report.get("rehearsal_summary"), Mapping) else {}
        ephemeral = report.get("ephemeral_rehearsal") if isinstance(report.get("ephemeral_rehearsal"), Mapping) else {}
        traces = [
            dict(item)
            for item in list(ephemeral.get("trace") or [])
            if isinstance(item, Mapping)
        ]
        cycles = max(1, min(int(replay_cycles), 12))
        sparsity = float(summary.get("activation_sparsity", 1.0) or 1.0)
        similarity = float(summary.get("mean_pairwise_similarity", 0.0) or 0.0)
        average_priority = float(summary.get("average_priority_score", 0.0) or 0.0)
        coverage = min(1.0, len(traces) / max(1, int(summary.get("candidate_count", len(traces)) or len(traces) or 1)))
        baseline_pressure = max(0.0, min(1.0, 1.0 - min(1.0, average_priority / 100.0)))
        rehearsal_gain = min(0.5, (average_priority / 100.0) * sparsity * coverage * (cycles / 12.0))
        simulated_post_pressure = max(0.0, baseline_pressure - rehearsal_gain)
        diversity_penalty = 0.0 if len(traces) <= 1 else max(0.0, similarity - 0.85)
        stability_score = max(0.0, min(1.0, sparsity * (1.0 - diversity_penalty)))
        pressure_non_worsening = simulated_post_pressure <= baseline_pressure
        stable = stability_score >= max(0.0, min(float(stability_floor), 1.0))
        experiment_trace = [
            {
                "candidate_id": trace.get("candidate_id"),
                "readout_evidence_hash": trace.get("readout_evidence_hash"),
                "prediction_hash": trace.get("prediction_hash"),
                "transition_memory_evaluation_hash": trace.get(
                    "transition_memory_evaluation_hash"
                ),
                "persistent_transition_weights_hash": trace.get(
                    "persistent_transition_weights_hash"
                ),
                "replay_cycles": cycles,
                "simulated_pressure_delta": float(-rehearsal_gain / max(1, len(traces))),
                "applied_to_runtime": False,
                "weights_persisted": False,
                "generated_text": False,
            }
            for trace in traces
        ]
        required = {
            "rehearsal_evaluation_available": bool(report.get("available")),
            "rehearsal_evaluation_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "rehearsal_gate_ready": bool(gate.get("eligible_for_operator_rehearsal_review")),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "generation_absent": not bool(report.get("generates_text")),
            "rehearsal_trace_available": bool(traces),
            "pressure_non_worsening": pressure_non_worsening,
            "rehearsal_stability_sufficient": stable,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_rehearsal_experiment",
            "surface": "snn_language_readout_rehearsal_experiment.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.rehearsal_experiment",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rehearsal_evaluation_surface": report.get("surface"),
            "experiment_summary": {
                "candidate_count": len(traces),
                "replay_cycles": cycles,
                "baseline_pressure_score": float(baseline_pressure),
                "simulated_post_rehearsal_pressure_score": float(simulated_post_pressure),
                "expected_rehearsal_pressure_gain": float(rehearsal_gain),
                "pressure_non_worsening": bool(pressure_non_worsening),
                "stability_score": float(stability_score),
                "stability_floor": float(max(0.0, min(float(stability_floor), 1.0))),
                "rehearsal_stability_sufficient": bool(stable),
            },
            "ephemeral_experiment": {
                "trace": experiment_trace,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "generated_text": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review" if ready else "blocked_missing_rehearsal_experiment_evidence",
                "eligible_for_operator_rehearsal_experiment_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_replay_design"
                if ready
                else "collect_rehearsal_evaluation_evidence",
                "required_evidence": required,
            },
        }

    def replay_design(
        self,
        rehearsal_experiment: Mapping[str, Any],
        *,
        replay_policy: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Design bounded future readout replay without executing it."""

        report = dict(rehearsal_experiment)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        summary = report.get("experiment_summary") if isinstance(report.get("experiment_summary"), Mapping) else {}
        ephemeral = report.get("ephemeral_experiment") if isinstance(report.get("ephemeral_experiment"), Mapping) else {}
        traces = [
            dict(item)
            for item in list(ephemeral.get("trace") or [])
            if isinstance(item, Mapping)
        ]
        policy = dict(replay_policy or {})
        rollback = dict(rollback_policy or {})
        max_candidates = max(1, min(int(policy.get("max_candidates", len(traces) or 1) or 1), 32))
        max_replay_cycles = max(1, min(int(policy.get("max_replay_cycles", summary.get("replay_cycles", 1)) or 1), 12))
        min_pressure_gain = max(0.0, min(float(policy.get("min_pressure_gain", 0.01) or 0.0), 1.0))
        stability_floor = max(0.0, min(float(policy.get("stability_floor", summary.get("stability_floor", 0.85)) or 0.85), 1.0))
        expected_gain = float(summary.get("expected_rehearsal_pressure_gain", 0.0) or 0.0)
        stability_score = float(summary.get("stability_score", 0.0) or 0.0)
        rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
        selected = traces[:max_candidates]
        known_evidence_hashes = self._known_readout_evidence_hashes()
        required = {
            "rehearsal_experiment_surface_available": report.get("surface")
            == "snn_language_readout_rehearsal_experiment.v1",
            "rehearsal_experiment_available": bool(report.get("available")),
            "rehearsal_experiment_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "external_dependency_absent": not bool(report.get("external_dependency")),
            "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
            "rehearsal_experiment_gate_ready": bool(
                gate.get("eligible_for_operator_rehearsal_experiment_review")
            ),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "generation_absent": not bool(report.get("generates_text")),
            "decoding_absent": not bool(report.get("decodes_text")),
            "trained_weights_absent": not bool(report.get("returns_trained_weights")),
            "trace_available": bool(selected),
            "provenance_hashes_available": all(
                bool(item.get("readout_evidence_hash"))
                and bool(item.get("prediction_hash"))
                and bool(item.get("transition_memory_evaluation_hash"))
                and bool(item.get("persistent_transition_weights_hash"))
                for item in selected
            ) if selected else False,
            "selected_evidence_present_in_internal_ledger": all(
                str(item.get("readout_evidence_hash") or "") in known_evidence_hashes
                for item in selected
            ) if selected else False,
            "pressure_gain_sufficient": expected_gain >= min_pressure_gain,
            "stability_sufficient": stability_score >= stability_floor,
            "rollback_policy_available": rollback_available,
        }
        ready = all(required.values())
        design_id = self._sha256_json(
            {
                "selected_hashes": [item.get("readout_evidence_hash") for item in selected],
                "transition_memory_evaluation_hashes": [
                    item.get("transition_memory_evaluation_hash") for item in selected
                ],
                "persistent_transition_weights_hashes": [
                    item.get("persistent_transition_weights_hash") for item in selected
                ],
                "rehearsal_experiment_surface": report.get("surface"),
                "owned_by_hecsn": report.get("owned_by_hecsn"),
                "rollback_policy": rollback,
                "max_candidates": max_candidates,
                "max_replay_cycles": max_replay_cycles,
                "min_pressure_gain": min_pressure_gain,
                "stability_floor": stability_floor,
            }
        )
        return {
            "artifact_kind": "terminus_snn_language_readout_replay_design",
            "surface": "snn_language_readout_replay_design.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.replay_design",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "rehearsal_experiment_surface": report.get("surface"),
            "readout_replay_design": {
                "design_id": f"snn-readout-replay-design:{design_id[:16]}",
                "selected_candidate_count": len(selected),
                "max_candidates": max_candidates,
                "max_replay_cycles": max_replay_cycles,
                "min_pressure_gain": float(min_pressure_gain),
                "stability_floor": float(stability_floor),
                "expected_rehearsal_pressure_gain": float(expected_gain),
                "stability_score": float(stability_score),
                "execution_allowed": False,
                "requires_operator_approval": ready,
            },
            "selected_replay_targets": [
                {
                    "readout_evidence_hash": item.get("readout_evidence_hash"),
                    "prediction_hash": item.get("prediction_hash"),
                    "transition_memory_evaluation_hash": item.get(
                        "transition_memory_evaluation_hash"
                    ),
                    "persistent_transition_weights_hash": item.get(
                        "persistent_transition_weights_hash"
                    ),
                    "candidate_id": item.get("candidate_id"),
                    "replay_cycles": max_replay_cycles,
                    "executable": False,
                    "applied_to_runtime": False,
                    "weights_persisted": False,
                }
                for item in selected
            ],
            "rollback_evidence": {
                "available": rollback_available,
                "snapshot_id": rollback.get("snapshot_id"),
                "ledger_id": rollback.get("ledger_id"),
            },
            "promotion_gate": {
                "status": "ready_for_operator_review" if ready else "blocked_missing_readout_replay_design_evidence",
                "eligible_for_operator_replay_design_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_isolated_snn_readout_replay"
                if ready
                else "collect_rehearsal_experiment_evidence",
                "required_evidence": required,
            },
        }

    def replay_dry_run(
        self,
        replay_design: Mapping[str, Any],
        *,
        operator_approval: bool = False,
        operator_id: str | None = None,
        device_evidence: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run isolated sparse readout replay without touching live runtime state."""

        report = dict(replay_design)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        design = (
            report.get("readout_replay_design")
            if isinstance(report.get("readout_replay_design"), Mapping)
            else {}
        )
        rollback = (
            report.get("rollback_evidence")
            if isinstance(report.get("rollback_evidence"), Mapping)
            else {}
        )
        device_report = dict(device_evidence or {})
        requested_device = str(device_report.get("device") or device_report.get("tensor_device") or "cpu")
        cuda_requested = requested_device.startswith("cuda")
        cuda_available = bool(torch.cuda.is_available())
        cuda_fallback_blocked = bool(cuda_requested and not cuda_available)
        device = self._safe_tensor_device(
            requested_device
        )
        targets = [
            dict(item)
            for item in list(report.get("selected_replay_targets") or [])
            if isinstance(item, Mapping)
        ]
        cycles = max(1, min(int(design.get("max_replay_cycles", 1) or 1), 12))
        known_evidence_hashes = self._known_readout_evidence_hashes()
        vectors: list[torch.Tensor] = []
        trace: list[dict[str, Any]] = []
        for target in targets:
            material = [
                str(target.get("readout_evidence_hash") or ""),
                str(target.get("prediction_hash") or ""),
                str(target.get("transition_memory_evaluation_hash") or ""),
                str(target.get("persistent_transition_weights_hash") or ""),
            ]
            active_indices = self._hash_sparse_indices(material)
            vector = torch.zeros(64, device=device)
            for index in active_indices:
                vector[int(index) % 64] = 1.0
            if active_indices:
                vector = vector / max(1.0, float(len(active_indices)))
            vectors.append(vector)
            trace.append(
                {
                    "candidate_id": target.get("candidate_id"),
                    "readout_evidence_hash": target.get("readout_evidence_hash"),
                    "prediction_hash": target.get("prediction_hash"),
                    "transition_memory_evaluation_hash": target.get(
                        "transition_memory_evaluation_hash"
                    ),
                    "persistent_transition_weights_hash": target.get(
                        "persistent_transition_weights_hash"
                    ),
                    "sparse_active_indices": active_indices,
                    "replay_cycles": cycles,
                    "source_executable": bool(target.get("executable")),
                    "source_applied_to_runtime": bool(target.get("applied_to_runtime")),
                    "source_weights_persisted": bool(target.get("weights_persisted")),
                    "applied_to_runtime": False,
                    "weights_persisted": False,
                    "checkpoint_written": False,
                    "generated_text": False,
                }
            )
        if vectors:
            stack = torch.stack(vectors)
            activity = stack.clone()
            for _ in range(cycles):
                activity = torch.clamp((activity * 0.72) + (stack * 0.28), min=0.0, max=1.0)
            active_fraction = float((activity > 0).float().mean().item())
            activation_sparsity = 1.0 - active_fraction
            baseline_pressure = max(0.0, min(1.0, 1.0 - float(design.get("expected_rehearsal_pressure_gain", 0.0) or 0.0)))
            target_support = min(1.0, len(targets) / max(1, int(design.get("selected_candidate_count", len(targets)) or 1)))
            replay_gain = min(
                0.5,
                float(design.get("expected_rehearsal_pressure_gain", 0.0) or 0.0)
                * target_support
                * activation_sparsity
                * (cycles / 12.0),
            )
            post_pressure = max(0.0, baseline_pressure - replay_gain)
            if activity.shape[0] > 1:
                normalized = torch.nn.functional.normalize(activity, p=2, dim=1, eps=1e-9)
                similarity = normalized @ normalized.T
                upper = similarity[torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)]
                mean_pairwise_similarity = float(upper.mean().item()) if upper.numel() else 1.0
            else:
                mean_pairwise_similarity = 1.0
        else:
            activation_sparsity = 1.0
            baseline_pressure = 1.0
            replay_gain = 0.0
            post_pressure = 1.0
            mean_pairwise_similarity = 0.0
        stable = activation_sparsity >= float(design.get("stability_floor", 0.85) or 0.85)
        pressure_non_worsening = post_pressure <= baseline_pressure
        dry_run_hash = self._sha256_json(
            {
                "surface": report.get("surface"),
                "design_id": design.get("design_id"),
                "selected_targets": [
                    {
                        "readout_evidence_hash": item.get("readout_evidence_hash"),
                        "prediction_hash": item.get("prediction_hash"),
                        "transition_memory_evaluation_hash": item.get(
                            "transition_memory_evaluation_hash"
                        ),
                        "persistent_transition_weights_hash": item.get(
                            "persistent_transition_weights_hash"
                        ),
                        "sparse_active_indices": item.get("sparse_active_indices"),
                    }
                    for item in trace
                ],
                "replay_cycles": cycles,
                "operator_id": operator_id,
                "rollback_evidence": rollback,
                "replay_summary": {
                    "activation_sparsity": float(activation_sparsity),
                    "mean_pairwise_similarity": float(mean_pairwise_similarity),
                    "baseline_pressure_score": float(baseline_pressure),
                    "simulated_post_replay_pressure_score": float(post_pressure),
                    "expected_replay_pressure_gain": float(replay_gain),
                    "pressure_non_worsening": bool(pressure_non_worsening),
                    "sparse_replay_stable": bool(stable),
                },
                "requested_device": requested_device,
                "tensor_device": str(device),
            }
        )
        required = {
            "replay_design_artifact_kind_available": report.get("artifact_kind")
            == "terminus_snn_language_readout_replay_design",
            "replay_design_surface_available": report.get("surface")
            == "snn_language_readout_replay_design.v1",
            "replay_design_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "external_dependency_absent": not bool(report.get("external_dependency")),
            "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
            "replay_design_gate_ready": bool(gate.get("eligible_for_operator_replay_design_review")),
            "operator_approval": bool(operator_approval),
            "operator_id_available": bool(str(operator_id or "").strip()),
            "device_evidence_available": bool(device_report),
            "cuda_fallback_not_required": not cuda_fallback_blocked,
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "generation_absent": not bool(report.get("generates_text")),
            "decoding_absent": not bool(report.get("decodes_text")),
            "trained_weights_absent": not bool(report.get("returns_trained_weights")),
            "design_execution_disabled": not bool(design.get("execution_allowed")),
            "rollback_evidence_available": bool(rollback.get("available")),
            "targets_available": bool(trace),
            "source_targets_non_executable": all(
                not bool(item.get("source_executable"))
                and not bool(item.get("source_applied_to_runtime"))
                and not bool(item.get("source_weights_persisted"))
                for item in trace
            ) if trace else False,
            "target_provenance_hashes_available": all(
                bool(item.get("readout_evidence_hash"))
                and bool(item.get("prediction_hash"))
                and bool(item.get("transition_memory_evaluation_hash"))
                and bool(item.get("persistent_transition_weights_hash"))
                for item in trace
            ) if trace else False,
            "selected_evidence_present_in_internal_ledger": all(
                str(item.get("readout_evidence_hash") or "") in known_evidence_hashes
                for item in trace
            ) if trace else False,
            "isolated_replay_pressure_non_worsening": pressure_non_worsening,
            "isolated_replay_sparse_stable": stable,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_replay_dry_run",
            "surface": "snn_language_readout_replay_dry_run.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.replay_dry_run",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "replay_design_surface": report.get("surface"),
            "readout_replay_dry_run_hash": dry_run_hash,
            "device_evidence": {
                "requested_device": requested_device,
                "tensor_device": str(device),
                "cuda_requested": cuda_requested,
                "cuda_available": cuda_available,
                "cuda_tensor": device.type == "cuda",
                "cuda_fallback_blocked": cuda_fallback_blocked,
                "device_source": device_report.get("source") or device_report.get("device_source"),
            },
            "isolated_replay_summary": {
                "target_count": len(trace),
                "replay_cycles": cycles,
                "activation_sparsity": float(activation_sparsity),
                "mean_pairwise_similarity": float(mean_pairwise_similarity),
                "baseline_pressure_score": float(baseline_pressure),
                "simulated_post_replay_pressure_score": float(post_pressure),
                "expected_replay_pressure_gain": float(replay_gain),
                "pressure_non_worsening": bool(pressure_non_worsening),
                "sparse_replay_stable": bool(stable),
            },
            "ephemeral_replay": {
                "trace": trace,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
                "generated_text": False,
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_readout_replay_dry_run_evidence",
                "eligible_for_operator_replay_dry_run_review": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_snn_readout_plasticity_preflight"
                if ready
                else "collect_replay_design_evidence",
                "required_evidence": required,
            },
        }

    def plasticity_preflight(
        self,
        readout_replay_dry_run: Mapping[str, Any],
        *,
        plasticity_policy: Mapping[str, Any] | None = None,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Review dry-run replay evidence for a later local-plasticity design."""

        report = dict(readout_replay_dry_run)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        summary = (
            report.get("isolated_replay_summary")
            if isinstance(report.get("isolated_replay_summary"), Mapping)
            else {}
        )
        device_report = (
            report.get("device_evidence")
            if isinstance(report.get("device_evidence"), Mapping)
            else {}
        )
        ephemeral = report.get("ephemeral_replay") if isinstance(report.get("ephemeral_replay"), Mapping) else {}
        dry_run_required = (
            gate.get("required_evidence")
            if isinstance(gate.get("required_evidence"), Mapping)
            else {}
        )
        traces = [
            dict(item)
            for item in list(ephemeral.get("trace") or [])
            if isinstance(item, Mapping)
        ]
        policy = dict(plasticity_policy or {})
        truth_delta = dict(runtime_truth_delta or {})
        rollback = dict(rollback_policy or {})
        learning_rate = max(0.0, min(float(policy.get("learning_rate", 0.02) or 0.0), 0.1))
        max_weight_delta = max(0.0, min(float(policy.get("max_weight_delta", 0.03) or 0.0), 0.1))
        locality_radius = max(1, min(int(policy.get("locality_radius", 2) or 2), 8))
        max_candidate_synapses = max(1, min(int(policy.get("max_candidate_synapses", 64) or 64), 256))
        normalization = bool(policy.get("normalization", True))
        local_only = bool(policy.get("local_only", True))
        pressure_gain = float(summary.get("expected_replay_pressure_gain", 0.0) or 0.0)
        pressure_non_worsening = bool(summary.get("pressure_non_worsening"))
        sparse_stable = bool(summary.get("sparse_replay_stable"))
        candidate_pairs: list[dict[str, Any]] = []
        for trace in traces:
            active = [
                int(value)
                for value in list(trace.get("sparse_active_indices") or [])
                if isinstance(value, int)
            ]
            if len(active) < 2:
                continue
            for index, pre in enumerate(active):
                post = active[(index + 1) % len(active)]
                if abs(int(post) - int(pre)) <= locality_radius:
                    candidate_pairs.append(
                        {
                            "readout_evidence_hash": trace.get("readout_evidence_hash"),
                            "prediction_hash": trace.get("prediction_hash"),
                            "transition_memory_evaluation_hash": trace.get(
                                "transition_memory_evaluation_hash"
                            ),
                            "persistent_transition_weights_hash": trace.get(
                                "persistent_transition_weights_hash"
                            ),
                            "pre_index": int(pre),
                            "post_index": int(post),
                            "max_abs_weight_delta": float(max_weight_delta),
                            "runtime_update_applied": False,
                            "weights_persisted": False,
                        }
                    )
                if len(candidate_pairs) >= max_candidate_synapses:
                    break
            if len(candidate_pairs) >= max_candidate_synapses:
                break
        candidate_hash = self._sha256_json(
            {
                "dry_run_hash": report.get("readout_replay_dry_run_hash"),
                "candidate_pairs": candidate_pairs,
                "learning_rate": learning_rate,
                "max_weight_delta": max_weight_delta,
                "locality_radius": locality_radius,
                "normalization": normalization,
                "local_only": local_only,
            }
        )
        rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
        runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
        known_evidence_hashes = self._known_readout_evidence_hashes()
        required = {
            "dry_run_artifact_kind_available": report.get("artifact_kind")
            == "terminus_snn_language_readout_replay_dry_run",
            "dry_run_surface_available": report.get("surface")
            == "snn_language_readout_replay_dry_run.v1",
            "dry_run_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "external_dependency_absent": not bool(report.get("external_dependency")),
            "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
            "dry_run_gate_ready": bool(gate.get("eligible_for_operator_replay_dry_run_review")),
            "dry_run_required_evidence_complete": bool(dry_run_required)
            and all(bool(value) for value in dry_run_required.values()),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "training_absent": not bool(report.get("trains_runtime_model")),
            "generation_absent": not bool(report.get("generates_text")),
            "decoding_absent": not bool(report.get("decodes_text")),
            "trained_weights_absent": not bool(report.get("returns_trained_weights")),
            "checkpoint_write_absent": not bool(ephemeral.get("checkpoint_written")),
            "ephemeral_runtime_update_absent": not bool(ephemeral.get("runtime_update_applied")),
            "ephemeral_weight_persistence_absent": not bool(ephemeral.get("weights_persisted")),
            "ephemeral_generation_absent": not bool(ephemeral.get("generated_text")),
            "dry_run_hash_available": bool(report.get("readout_replay_dry_run_hash")),
            "trace_available": bool(traces),
            "selected_evidence_present_in_internal_ledger": all(
                str(item.get("readout_evidence_hash") or "") in known_evidence_hashes
                for item in traces
            ) if traces else False,
            "pressure_non_worsening": pressure_non_worsening,
            "sparse_replay_stable": sparse_stable,
            "pressure_gain_positive": pressure_gain > 0.0,
            "candidate_synapses_available": bool(candidate_pairs),
            "learning_rate_bounded": 0.0 < learning_rate <= 0.1,
            "max_weight_delta_bounded": 0.0 < max_weight_delta <= 0.1,
            "locality_radius_bounded": 1 <= locality_radius <= 8,
            "normalization_enabled": normalization,
            "local_update_only": local_only,
            "runtime_truth_improved_or_stable": runtime_truth_ok,
            "rollback_policy_available": rollback_available,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_plasticity_preflight",
            "surface": "snn_language_readout_plasticity_preflight.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.plasticity_preflight",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "readout_replay_dry_run_surface": report.get("surface"),
            "readout_replay_dry_run_hash": report.get("readout_replay_dry_run_hash"),
            "device_evidence": dict(device_report),
            "readout_plasticity_preflight_hash": candidate_hash,
            "plasticity_preflight": {
                "candidate_synapse_count": len(candidate_pairs),
                "learning_rate": float(learning_rate),
                "max_weight_delta": float(max_weight_delta),
                "locality_radius": locality_radius,
                "max_candidate_synapses": max_candidate_synapses,
                "normalization": normalization,
                "local_only": local_only,
                "expected_replay_pressure_gain": float(pressure_gain),
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
            },
            "candidate_replay_sequences": [
                {
                    "pre_indices": [item["pre_index"]],
                    "post_indices": [item["post_index"]],
                    "active_indices": [item["pre_index"], item["post_index"]],
                    "readout_evidence_hash": item["readout_evidence_hash"],
                    "prediction_hash": item["prediction_hash"],
                    "transition_memory_evaluation_hash": item["transition_memory_evaluation_hash"],
                    "persistent_transition_weights_hash": item["persistent_transition_weights_hash"],
                    "runtime_update_applied": False,
                    "weights_persisted": False,
                }
                for item in candidate_pairs
            ],
            "runtime_truth_delta": truth_delta,
            "rollback_evidence": {
                "available": rollback_available,
                "snapshot_id": rollback.get("snapshot_id"),
                "ledger_id": rollback.get("ledger_id"),
            },
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_readout_plasticity_preflight_evidence",
                "eligible_for_operator_readout_plasticity_review": ready,
                "eligible_for_plasticity_application_design": ready,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "requires_operator_approval": ready,
                "next_gate": "operator_review_readout_to_plasticity_replay_bridge"
                if ready
                else "collect_readout_replay_dry_run_evidence",
                "required_evidence": required,
            },
        }

    def plasticity_replay_bridge(
        self,
        readout_plasticity_preflight: Mapping[str, Any],
        *,
        runtime_truth_delta: Mapping[str, Any] | None = None,
        rollback_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Expose readout preflight as an existing replay-experiment contract."""

        report = dict(readout_plasticity_preflight)
        gate = report.get("promotion_gate") if isinstance(report.get("promotion_gate"), Mapping) else {}
        preflight = (
            report.get("plasticity_preflight")
            if isinstance(report.get("plasticity_preflight"), Mapping)
            else {}
        )
        device_report = (
            report.get("device_evidence")
            if isinstance(report.get("device_evidence"), Mapping)
            else {}
        )
        preflight_required = (
            gate.get("required_evidence")
            if isinstance(gate.get("required_evidence"), Mapping)
            else {}
        )
        sequences = [
            dict(item)
            for item in list(report.get("candidate_replay_sequences") or [])
            if isinstance(item, Mapping)
        ]
        canonical_sequences = [
            {
                "sequence_id": str(index),
                "grounded": bool(item.get("grounded", True)),
                "pre_indices": [
                    int(value)
                    for value in list(item.get("pre_indices") or [])
                    if isinstance(value, int)
                ],
                "post_indices": [
                    int(value)
                    for value in list(item.get("post_indices") or [])
                    if isinstance(value, int)
                ],
                "active_indices": [
                    int(value)
                    for value in list(item.get("active_indices") or [])
                    if isinstance(value, int)
                ],
                "readout_evidence_hash": item.get("readout_evidence_hash"),
                "prediction_hash": item.get("prediction_hash"),
                "transition_memory_evaluation_hash": item.get("transition_memory_evaluation_hash"),
                "persistent_transition_weights_hash": item.get("persistent_transition_weights_hash"),
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
            }
            for index, item in enumerate(sequences)
        ]
        truth_delta = dict(runtime_truth_delta or report.get("runtime_truth_delta") or {})
        rollback = dict(rollback_policy or report.get("rollback_evidence") or {})
        replay_count = len(sequences)
        grounded_count = sum(1 for item in sequences if bool(item.get("grounded", True)))
        coverage = grounded_count / replay_count if replay_count else 0.0
        expected_gain = float(preflight.get("expected_replay_pressure_gain", 0.0) or 0.0)
        learning_rate = float(preflight.get("learning_rate", 0.0) or 0.0)
        max_weight_delta = float(preflight.get("max_weight_delta", 0.0) or 0.0)
        locality_radius = int(preflight.get("locality_radius", 0) or 0)
        normalization = bool(preflight.get("normalization"))
        local_only = bool(preflight.get("local_only"))
        pre_pressure = 1.0
        simulated_post_pressure = max(0.0, pre_pressure - min(0.5, expected_gain * coverage))
        pressure_stable = simulated_post_pressure <= pre_pressure
        rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
        runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
        known_evidence_hashes = self._known_readout_evidence_hashes()
        bridge_hash = self._sha256_json(
            {
                "preflight_hash": report.get("readout_plasticity_preflight_hash"),
                "dry_run_hash": report.get("readout_replay_dry_run_hash"),
                "canonical_replay_sequences": canonical_sequences,
                "application_design": {
                    "learning_rate": learning_rate,
                    "max_weight_delta": max_weight_delta,
                    "locality_radius": locality_radius,
                    "normalization": normalization,
                    "local_only": local_only,
                    "grounded_replay_coverage": coverage,
                    "pressure_stable_after_replay": pressure_stable,
                },
                "device_evidence": dict(device_report),
                "runtime_truth_delta": truth_delta,
                "rollback_evidence": rollback,
                "coverage": coverage,
                "expected_gain": expected_gain,
            }
        )
        required = {
            "preflight_artifact_kind_available": report.get("artifact_kind")
            == "terminus_snn_language_readout_plasticity_preflight",
            "preflight_surface_available": report.get("surface")
            == "snn_language_readout_plasticity_preflight.v1",
            "preflight_owned_by_hecsn": bool(report.get("owned_by_hecsn")),
            "external_dependency_absent": not bool(report.get("external_dependency")),
            "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
            "preflight_gate_ready": bool(gate.get("eligible_for_operator_readout_plasticity_review")),
            "preflight_required_evidence_complete": bool(preflight_required)
            and all(bool(value) for value in preflight_required.values()),
            "plasticity_application_absent": not bool(report.get("applies_plasticity")),
            "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
            "training_absent": not bool(report.get("trains_runtime_model")),
            "generation_absent": not bool(report.get("generates_text")),
            "decoding_absent": not bool(report.get("decodes_text")),
            "trained_weights_absent": not bool(report.get("returns_trained_weights")),
            "candidate_sequences_available": replay_count > 0,
            "candidate_sequences_grounded": coverage >= 0.5,
            "candidate_sequence_indices_available": all(
                bool(item.get("pre_indices")) and bool(item.get("post_indices"))
                for item in canonical_sequences
            ) if canonical_sequences else False,
            "candidate_sequence_provenance_available": all(
                bool(item.get("readout_evidence_hash"))
                and bool(item.get("prediction_hash"))
                and bool(item.get("transition_memory_evaluation_hash"))
                and bool(item.get("persistent_transition_weights_hash"))
                for item in sequences
            ) if sequences else False,
            "selected_evidence_present_in_internal_ledger": all(
                str(item.get("readout_evidence_hash") or "") in known_evidence_hashes
                for item in sequences
            ) if sequences else False,
            "pressure_stable_after_replay": pressure_stable,
            "learning_rate_bounded": 0.0 < learning_rate <= 0.1,
            "max_weight_delta_bounded": 0.0 < max_weight_delta <= 0.1,
            "locality_radius_bounded": 1 <= locality_radius <= 8,
            "normalization_enabled": normalization,
            "local_update_only": local_only,
            "device_evidence_available": bool(device_report)
            and bool(device_report.get("device_report_available", True)),
            "runtime_truth_improved_or_stable": runtime_truth_ok,
            "rollback_policy_available": rollback_available,
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_plasticity_replay_experiment",
            "surface": "snn_language_plasticity_replay_experiment.v1",
            "available": bool(report),
            "source": "service.snn_language_readout_ledger.plasticity_replay_bridge",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "readout_plasticity_preflight_surface": report.get("surface"),
            "readout_replay_dry_run_hash": report.get("readout_replay_dry_run_hash"),
            "readout_plasticity_replay_bridge_hash": bridge_hash,
            "device_evidence": {
                "requested_device": device_report.get("requested_device")
                or device_report.get("device")
                or device_report.get("tensor_device")
                or "unknown",
                "tensor_device": device_report.get("tensor_device")
                or device_report.get("device")
                or "unknown",
                "cuda_tensor": bool(device_report.get("cuda_tensor")),
                "device_source": device_report.get("source") or device_report.get("device_source"),
                "device_report_available": bool(device_report)
                and bool(device_report.get("device_report_available", True)),
            },
            "readout_bridge": {
                "preflight_hash": report.get("readout_plasticity_preflight_hash"),
                "dry_run_hash": report.get("readout_replay_dry_run_hash"),
                "candidate_synapse_count": int(preflight.get("candidate_synapse_count", 0) or 0),
                "compatible_with_application_design": ready,
                "runtime_update_applied": False,
                "weights_persisted": False,
                "checkpoint_written": False,
            },
            "replay_experiment": {
                "replay_sequence_count": replay_count,
                "grounded_replay_sequence_count": grounded_count,
                "grounded_replay_coverage": float(coverage),
                "pre_pressure_score": float(pre_pressure),
                "post_evaluation_pressure_score": float(pre_pressure),
                "simulated_post_replay_pressure_score": float(simulated_post_pressure),
                "expected_replay_pressure_gain": float(min(0.5, expected_gain * coverage)),
                "pressure_stable_after_replay": bool(pressure_stable),
            },
            "application_design": {
                "learning_rate": float(learning_rate),
                "max_weight_delta": float(max_weight_delta),
                "locality_radius": locality_radius,
                "normalization": normalization,
                "local_only": local_only,
                "grounded_replay_coverage": float(coverage),
                "pressure_stable_after_replay": bool(pressure_stable),
                "runtime_update_applied": False,
                "weights_persisted": False,
            },
            "application_target_hint": {
                "available": True,
                "target_id": "hecsn.snn_language.sparse_transition_weights",
                "owned_by_hecsn": True,
                "mutable": True,
                "sparse": True,
                "checkpointed": True,
                "runtime_update_applied": False,
            },
            "checkpoint_transaction_requirements": {
                "pre_update_checkpoint_required": True,
                "restore_verification_required": True,
                "records_shadow_delta_required": True,
                "runtime_update_applied": False,
            },
            "canonical_replay_sequences": canonical_sequences,
            "ephemeral_replay": {
                "trace": [
                    {
                        "sequence_id": str(index),
                        "readout_evidence_hash": item.get("readout_evidence_hash"),
                        "prediction_hash": item.get("prediction_hash"),
                        "grounded": bool(item.get("grounded", True)),
                        "applied_to_runtime": False,
                        "weights_persisted": False,
                        "checkpoint_written": False,
                    }
                    for index, item in enumerate(sequences)
                ],
                "weights_persisted": False,
                "runtime_update_applied": False,
                "runtime_state_mutated": False,
                "checkpoint_written": False,
            },
            "runtime_truth_delta": truth_delta,
            "rollback_evidence": {
                "available": rollback_available,
                "snapshot_id": rollback.get("snapshot_id"),
                "ledger_id": rollback.get("ledger_id"),
            },
            "promotion_gate": {
                "status": "ready_for_operator_review" if ready else "blocked_missing_replay_experiment_evidence",
                "eligible_for_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_runtime_training": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_replay_promotion": False,
                "eligible_for_operator_application_review": ready,
                "requires_operator_approval": ready,
                "next_gate": "operator_approved_language_plasticity_application_design"
                if ready
                else "collect_readout_plasticity_preflight_evidence",
                "required_evidence": required,
            },
        }

    def synapse_provenance_audit(
        self,
        *,
        plasticity_runtime_state: Mapping[str, Any],
        limit: int = 64,
    ) -> dict[str, Any]:
        """Audit readout-derived sparse transition provenance without mutation."""

        runtime = dict(plasticity_runtime_state)
        raw_weights = dict(runtime.get("sparse_transition_weights") or {})
        weights: dict[str, float] = {}
        finite_weight_keys: set[str] = set()
        bounded_weight_keys: set[str] = set()
        canonical_weight_keys: set[str] = set()
        in_range_weight_keys: set[str] = set()
        for key, value in raw_weights.items():
            key_text = str(key)
            try:
                if value is None or isinstance(value, bool):
                    raise TypeError("non_numeric_weight")
                weight = float(value)
            except (TypeError, ValueError):
                weight = float("nan")
            weights[key_text] = weight
            if math.isfinite(weight):
                finite_weight_keys.add(key_text)
                if abs(weight) <= _MAX_READOUT_SYNAPSE_ABS_WEIGHT:
                    bounded_weight_keys.add(key_text)
            if self._canonical_synapse_key(key_text):
                canonical_weight_keys.add(key_text)
            pre_index, post_index = self._split_synapse_key(key_text)
            if self._valid_language_index(pre_index) and self._valid_language_index(post_index):
                in_range_weight_keys.add(key_text)
        provenance_by_key = {
            str(key): dict(value)
            for key, value in dict(runtime.get("synapse_provenance_by_key") or {}).items()
            if isinstance(value, Mapping)
        }
        normalized = self._normalized_state()
        ledger_events = {
            str(item.get("readout_evidence_hash") or ""): dict(item)
            for item in list(normalized.get("events") or [])
            if isinstance(item, Mapping) and item.get("readout_evidence_hash")
        }
        all_rows: list[dict[str, Any]] = []
        for key in sorted(provenance_by_key.keys()):
            provenance = provenance_by_key.get(key, {})
            provenance_type = str(provenance.get("provenance_type") or "readout_plasticity")
            readout_hash = str(provenance.get("readout_evidence_hash") or "")
            replay_readout_hashes = [
                str(value)
                for value in list(provenance.get("readout_evidence_hashes") or [])
                if str(value)
            ][:64]
            ledger_event = ledger_events.get(readout_hash, {})
            replay_ledger_events = [ledger_events.get(value, {}) for value in replay_readout_hashes]
            replay_hashes_present_in_ledger = bool(replay_readout_hashes) and all(
                bool(item) for item in replay_ledger_events
            )
            replay_hashes_valid = bool(replay_readout_hashes) and all(
                bool(item) and hash_value == self._ledger_event_material_hash(item)
                for hash_value, item in zip(replay_readout_hashes, replay_ledger_events, strict=False)
            )
            replay_provenance_complete = bool(
                provenance_type == "replay_regeneration"
                and provenance.get("permit_id")
                and provenance.get("replay_artifact_id")
                and provenance.get("replay_artifact_hash")
                and provenance.get("replay_window_hash")
                and replay_readout_hashes
            )
            local_edge_provenance = (
                dict(provenance.get("local_edge_provenance"))
                if isinstance(provenance.get("local_edge_provenance"), Mapping)
                else {}
            )
            try:
                source_rollout_step_index = int(
                    local_edge_provenance.get("source_rollout_step_index")
                )
                target_rollout_step_index = int(
                    local_edge_provenance.get("target_rollout_step_index")
                )
                rollout_step_order_valid = target_rollout_step_index > source_rollout_step_index
            except (TypeError, ValueError):
                source_rollout_step_index = None
                target_rollout_step_index = None
                rollout_step_order_valid = False
            local_edge_provenance_complete = bool(
                local_edge_provenance.get("source_synapse_id")
                and local_edge_provenance.get("source_active_indices_hash")
                and local_edge_provenance.get("target_active_indices_hash")
                and source_rollout_step_index is not None
                and target_rollout_step_index is not None
                and rollout_step_order_valid
            )
            ledger_field_match = bool(
                replay_hashes_present_in_ledger
                if provenance_type == "replay_regeneration"
                else (
                    ledger_event
                    and provenance.get("prediction_hash") == ledger_event.get("prediction_hash")
                    and provenance.get("transition_memory_evaluation_hash")
                    == ledger_event.get("transition_memory_evaluation_hash")
                    and provenance.get("persistent_transition_weights_hash")
                    == ledger_event.get("persistent_transition_weights_hash")
                )
            )
            ledger_hash_valid = bool(
                replay_hashes_valid
                if provenance_type == "replay_regeneration"
                else (
                    ledger_event
                    and readout_hash
                    and readout_hash == self._ledger_event_material_hash(ledger_event)
                )
            )
            canonical_key = self._canonical_synapse_key(key)
            source_pre_indices = [
                int(value)
                for value in list(provenance.get("source_pre_indices") or [])
                if isinstance(value, int)
            ]
            source_post_indices = [
                int(value)
                for value in list(provenance.get("source_post_indices") or [])
                if isinstance(value, int)
            ]
            source_active_indices = [
                int(value)
                for value in list(provenance.get("source_active_indices") or [])
                if isinstance(value, int)
            ]
            pre_index, post_index = self._split_synapse_key(key)
            weight = weights.get(key)
            weight_finite = bool(weight is not None and math.isfinite(float(weight)))
            source_indices_in_range = all(
                self._valid_language_index(value)
                for value in source_pre_indices + source_post_indices + source_active_indices
            )
            source_indices_match_synapse = (
                True
                if provenance_type == "replay_regeneration"
                else (
                    bool(
                        pre_index in source_pre_indices
                        and post_index in source_post_indices
                        and pre_index in source_active_indices
                        and post_index in source_active_indices
                    )
                    if pre_index is not None and post_index is not None
                    else False
                )
            )
            row = {
                "synapse_key": key,
                "provenance_type": provenance_type,
                "weight_available": key in weights,
                "weight": weight,
                "weight_finite": weight_finite,
                "weight_bounded": bool(
                    weight_finite and abs(float(weight)) <= _MAX_READOUT_SYNAPSE_ABS_WEIGHT
                ),
                "canonical_synapse_key": canonical_key,
                "synapse_indices_in_range": bool(
                    self._valid_language_index(pre_index)
                    and self._valid_language_index(post_index)
                ),
                "pre_index": pre_index,
                "post_index": post_index,
                "readout_evidence_hash": readout_hash,
                "prediction_hash": provenance.get("prediction_hash"),
                "transition_memory_evaluation_hash": provenance.get(
                    "transition_memory_evaluation_hash"
                ),
                "persistent_transition_weights_hash": provenance.get(
                    "persistent_transition_weights_hash"
                ),
                "permit_id": provenance.get("permit_id"),
                "replay_artifact_id": provenance.get("replay_artifact_id"),
                "replay_artifact_hash": provenance.get("replay_artifact_hash"),
                "replay_window_hash": provenance.get("replay_window_hash"),
                "readout_evidence_hashes": replay_readout_hashes,
                "local_edge_provenance": local_edge_provenance,
                "local_edge_provenance_complete": local_edge_provenance_complete,
                "local_edge_rollout_step_order_valid": rollout_step_order_valid,
                "source_rollout_step_index": source_rollout_step_index,
                "target_rollout_step_index": target_rollout_step_index,
                "source_active_indices_hash": local_edge_provenance.get(
                    "source_active_indices_hash"
                ),
                "target_active_indices_hash": local_edge_provenance.get(
                    "target_active_indices_hash"
                ),
                "ledger_evidence_present": bool(ledger_event),
                "replay_ledger_evidence_present": replay_hashes_present_in_ledger,
                "ledger_field_match": ledger_field_match,
                "ledger_hash_valid": ledger_hash_valid,
                "provenance_complete": bool(
                    replay_provenance_complete
                    and local_edge_provenance_complete
                    if provenance_type == "replay_regeneration"
                    else (
                        readout_hash
                        and provenance.get("prediction_hash")
                        and provenance.get("transition_memory_evaluation_hash")
                        and provenance.get("persistent_transition_weights_hash")
                    )
                ),
                "source_sequence_id": provenance.get("sequence_id"),
                "source_pre_indices": source_pre_indices,
                "source_post_indices": source_post_indices,
                "source_active_indices": source_active_indices,
                "source_indices_in_range": source_indices_in_range,
                "source_indices_match_synapse": source_indices_match_synapse,
            }
            all_rows.append(row)
        row_limit = max(0, min(int(limit), 512))
        rows = all_rows[:row_limit]
        orphan_weight_keys = sorted(set(weights.keys()) - set(provenance_by_key.keys()))
        dangling_provenance_keys = sorted(set(provenance_by_key.keys()) - set(weights.keys()))
        replay_regeneration_rows = [
            row for row in all_rows if row.get("provenance_type") == "replay_regeneration"
        ]
        required = {
            "runtime_state_surface_available": runtime.get("surface")
            == "snn_language_plasticity_runtime_state.v1",
            "runtime_state_owned_by_hecsn": bool(runtime.get("owned_by_hecsn")),
            "synapse_provenance_available": bool(provenance_by_key),
            "audited_synapses_have_weights": all(
                bool(row["weight_available"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_have_finite_weights": all(
                bool(row["weight_finite"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_have_bounded_weights": all(
                bool(row["weight_bounded"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_have_canonical_keys": all(
                bool(row["canonical_synapse_key"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapse_indices_in_range": all(
                bool(row["synapse_indices_in_range"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_have_complete_provenance": all(
                bool(row["provenance_complete"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_present_in_ledger": all(
                bool(row["ledger_evidence_present"] or row["replay_ledger_evidence_present"])
                for row in all_rows
            ) if all_rows else False,
            "audited_synapses_match_ledger_fields": all(
                bool(row["ledger_field_match"]) for row in all_rows
            ) if all_rows else False,
            "audited_ledger_hashes_valid": all(
                bool(row["ledger_hash_valid"]) for row in all_rows
            ) if all_rows else False,
            "audited_synapses_match_source_indices": all(
                bool(row["source_indices_match_synapse"]) for row in all_rows
            ) if all_rows else False,
            "audited_source_indices_in_range": all(
                bool(row["source_indices_in_range"]) for row in all_rows
            ) if all_rows else False,
            "audited_replay_regeneration_local_edge_provenance_complete": all(
                bool(row["local_edge_provenance_complete"])
                for row in replay_regeneration_rows
            ) if replay_regeneration_rows else True,
            "audited_replay_regeneration_rollout_step_order_valid": all(
                bool(row["local_edge_rollout_step_order_valid"])
                for row in replay_regeneration_rows
            ) if replay_regeneration_rows else True,
            "no_unprovenanced_weights": not bool(orphan_weight_keys),
            "no_dangling_provenance": not bool(dangling_provenance_keys),
            "all_weights_finite": len(finite_weight_keys) == len(weights),
            "all_weights_bounded": len(bounded_weight_keys) == len(weights),
            "all_weight_keys_canonical": len(canonical_weight_keys) == len(weights),
            "all_weight_indices_in_range": len(in_range_weight_keys) == len(weights),
        }
        ready = all(required.values())
        return {
            "artifact_kind": "terminus_snn_language_readout_synapse_provenance_audit",
            "surface": "snn_language_readout_synapse_provenance_audit.v1",
            "available": bool(runtime),
            "source": "service.snn_language_readout_ledger.synapse_provenance_audit",
            "owned_by_hecsn": True,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "generates_text": False,
            "decodes_text": False,
            "trains_runtime_model": False,
            "applies_plasticity": False,
            "mutates_runtime_state": False,
            "returns_trained_weights": False,
            "audit_summary": {
                "audited_synapse_count": len(all_rows),
                "returned_synapse_count": len(rows),
                "provenanced_synapse_count": len(provenance_by_key),
                "sparse_transition_weight_count": len(weights),
                "orphan_weight_count": len(orphan_weight_keys),
                "dangling_provenance_count": len(dangling_provenance_keys),
                "nonfinite_weight_count": len(weights) - len(finite_weight_keys),
                "unbounded_weight_count": len(weights) - len(bounded_weight_keys),
                "noncanonical_weight_key_count": len(weights) - len(canonical_weight_keys),
                "out_of_range_weight_key_count": len(weights) - len(in_range_weight_keys),
                "ledger_event_count": len(ledger_events),
                "replay_regeneration_synapse_count": len(replay_regeneration_rows),
                "local_edge_provenance_count": sum(
                    1 for row in all_rows if bool(row.get("local_edge_provenance"))
                ),
                "complete_local_edge_provenance_count": sum(
                    1 for row in all_rows if bool(row.get("local_edge_provenance_complete"))
                ),
            },
            "audited_synapses": rows,
            "orphan_weight_keys": orphan_weight_keys[: max(0, min(int(limit), 512))],
            "dangling_provenance_keys": dangling_provenance_keys[:row_limit],
            "promotion_gate": {
                "status": "ready_for_operator_review"
                if ready
                else "blocked_missing_readout_synapse_provenance_evidence",
                "eligible_for_readout_synapse_audit_review": ready,
                "eligible_for_live_application": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": required,
            },
        }

    def _blocked(self, before_revision: int, required_evidence: Mapping[str, bool]) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_readout_evidence_ledger_record",
            "surface": "snn_language_readout_evidence_ledger_record.v1",
            "accepted": False,
            "duplicate": False,
            "owned_by_hecsn": True,
            "external_dependency": False,
            "generates_text": False,
            "decodes_text": False,
            "mutates_runtime_state": False,
            "before": {"state_revision": before_revision},
            "after": self._runtime_state.mutation_summary(),
            "promotion_gate": {
                "status": "blocked_missing_review_ready_readout_evidence",
                "eligible_for_replay_memory": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": dict(required_evidence),
            },
        }

    def _blocked_rollout_record(
        self,
        before_revision: int,
        required_evidence: Mapping[str, bool],
    ) -> dict[str, Any]:
        return {
            "artifact_kind": "terminus_snn_language_readout_rollout_evidence_ledger_record",
            "surface": "snn_language_readout_rollout_evidence_ledger_record.v1",
            "accepted": False,
            "duplicate": False,
            "owned_by_hecsn": True,
            "external_dependency": False,
            "generates_text": False,
            "decodes_text": False,
            "mutates_runtime_state": False,
            "before": {"state_revision": before_revision},
            "after": self._runtime_state.mutation_summary(),
            "promotion_gate": {
                "status": "blocked_missing_review_ready_rollout_replay_evidence",
                "eligible_for_rollout_replay_memory": False,
                "eligible_for_replay_priority": False,
                "eligible_for_live_replay": False,
                "eligible_for_plasticity_application": False,
                "eligible_for_freeform_language_generation": False,
                "eligible_for_cognition_substrate": False,
                "eligible_for_fact_promotion": False,
                "eligible_for_action": False,
                "required_evidence": dict(required_evidence),
            },
        }

    def _ledger_event(
        self,
        *,
        draft: Mapping[str, Any],
        operator_id: str,
        state_revision: int,
    ) -> dict[str, Any]:
        draft_payload = draft.get("draft") if isinstance(draft.get("draft"), Mapping) else {}
        evaluation = (
            draft.get("transition_memory_evaluation_evidence")
            if isinstance(draft.get("transition_memory_evaluation_evidence"), Mapping)
            else {}
        )
        labels = [str(value) for value in list(draft_payload.get("labels") or [])][:12]
        sparse_decode = (
            draft.get("sparse_decode_evidence")
            if isinstance(draft.get("sparse_decode_evidence"), Mapping)
            else {}
        )
        candidate_matches = [
            dict(item)
            for item in list(sparse_decode.get("candidate_matches") or [])
            if isinstance(item, Mapping)
        ]
        grounding_by_label = {
            str(item.get("label") or ""): bool(item.get("grounded"))
            for item in candidate_matches
            if str(item.get("label") or "")
        }
        label_grounding = [bool(grounding_by_label.get(label, False)) for label in labels]
        material = {
            "prediction_hash": evaluation.get("prediction_hash"),
            "transition_memory_evaluation_hash": evaluation.get("transition_memory_evaluation_hash"),
            "persistent_transition_weights_hash": evaluation.get("persistent_transition_weights_hash"),
            "labels": labels,
            "label_grounding": label_grounding,
            "state_revision": int(state_revision),
        }
        evidence_hash = self._sha256_json(material)
        return {
            "readout_evidence_hash": evidence_hash,
            "readout_evidence_id": f"snn-readout-evidence:{evidence_hash[:16]}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "state_revision": int(state_revision),
            "operator_id": operator_id,
            "prediction_hash": material["prediction_hash"],
            "transition_memory_evaluation_hash": material["transition_memory_evaluation_hash"],
            "persistent_transition_weights_hash": material["persistent_transition_weights_hash"],
            "labels": labels,
            "label_grounding": label_grounding,
            "term_count": len(labels),
            "generation_scope": draft.get("generation_scope"),
            "freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "material_hash_algorithm": "sha256_canonical_json",
        }

    def _rollout_ledger_event(
        self,
        *,
        evaluation: Mapping[str, Any],
        operator_id: str,
        state_revision: int,
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        provenance = (
            evaluation.get("provenance_evidence")
            if isinstance(evaluation.get("provenance_evidence"), Mapping)
            else {}
        )
        replay_evaluation = (
            evaluation.get("replay_evaluation")
            if isinstance(evaluation.get("replay_evaluation"), Mapping)
            else {}
        )
        observed_device = (
            evaluation.get("device_evidence")
            if isinstance(evaluation.get("device_evidence"), Mapping)
            else {}
        )
        normalized_targets = [
            self._normalized_rollout_replay_target(item, index=index)
            for index, item in enumerate(targets[:32])
        ]
        material = {
            "rollout_replay_evaluation_hash": provenance.get("rollout_replay_evaluation_hash"),
            "rollout_hash": provenance.get("rollout_hash"),
            "rollout_id": provenance.get("rollout_id"),
            "prediction_hash": provenance.get("prediction_hash"),
            "current_sparse_code_hash": provenance.get("current_sparse_code_hash"),
            "transition_memory_evaluation_hash": provenance.get("transition_memory_evaluation_hash"),
            "persistent_transition_weights_hash": provenance.get("persistent_transition_weights_hash"),
            "server_transition_memory_hash": provenance.get("server_transition_memory_hash"),
            "server_transition_memory_hash_match": bool(
                provenance.get("server_transition_memory_hash_match")
            ),
            "transition_memory_state_source": provenance.get("transition_memory_state_source"),
            "device_evidence": {
                "requested_device": observed_device.get("requested_device"),
                "tensor_device": observed_device.get("tensor_device"),
                "cuda_tensor": bool(observed_device.get("cuda_tensor")),
                "device_source": observed_device.get("device_source"),
            },
            "targets": normalized_targets,
            "state_revision": int(state_revision),
        }
        evidence_hash = self._sha256_json(material)
        return {
            "rollout_evidence_hash": evidence_hash,
            "rollout_evidence_id": f"snn-readout-rollout-evidence:{evidence_hash[:16]}",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "state_revision": int(state_revision),
            "operator_id": operator_id,
            "rollout_replay_evaluation_hash": material["rollout_replay_evaluation_hash"],
            "rollout_hash": material["rollout_hash"],
            "rollout_id": material["rollout_id"],
            "prediction_hash": material["prediction_hash"],
            "current_sparse_code_hash": material["current_sparse_code_hash"],
            "transition_memory_evaluation_hash": material["transition_memory_evaluation_hash"],
            "persistent_transition_weights_hash": material["persistent_transition_weights_hash"],
            "server_transition_memory_hash": material["server_transition_memory_hash"],
            "server_transition_memory_hash_match": material["server_transition_memory_hash_match"],
            "transition_memory_state_source": material["transition_memory_state_source"],
            "device_evidence": material["device_evidence"],
            "target_count": len(normalized_targets),
            "trace_step_count": int(replay_evaluation.get("trace_step_count") or 0),
            "replay_targets": normalized_targets,
            "recorded_in_ledger": True,
            "eligible_for_replay_priority": False,
            "eligible_for_cognition_substrate": False,
            "freeform_language_generation": False,
            "material_hash_algorithm": "sha256_canonical_json",
        }

    def _normalized_rollout_replay_target(
        self,
        item: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        sparse_indices = [
            int(value)
            for value in list(item.get("predicted_sparse_indices") or [])
            if isinstance(value, int)
        ][:16]
        active_indices_hash = str(item.get("active_indices_hash") or "")
        return {
                "step_index": int(item.get("step_index", index) or index),
                "selected_label": str(item.get("selected_label") or ""),
                "grounded": bool(item.get("grounded")),
                "selection_score": float(item.get("selection_score") or 0.0),
                "transition_support": float(item.get("transition_support") or 0.0),
                "predicted_sparse_indices": sparse_indices,
                "active_indices_hash": active_indices_hash,
                "active_indices_hash_valid": active_indices_hash == self._sha256_json(sparse_indices),
        }

    @staticmethod
    def _rollout_sparse_transition_candidates(
        source_traces: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Pick one bounded local sparse edge per adjacent rollout step."""

        candidates: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for trace_index, trace in enumerate(source_traces[:16]):
            steps = [
                dict(item)
                for item in list(trace.get("step_trace") or [])
                if isinstance(item, Mapping)
            ][:32]
            for previous, current in zip(steps, steps[1:]):
                previous_indices = sorted(
                    {
                        int(value)
                        for value in list(previous.get("sparse_active_indices") or [])
                        if isinstance(value, int) and 0 <= int(value) < _LANGUAGE_NEURON_COUNT
                    }
                )
                current_indices = sorted(
                    {
                        int(value)
                        for value in list(current.get("sparse_active_indices") or [])
                        if isinstance(value, int) and 0 <= int(value) < _LANGUAGE_NEURON_COUNT
                    }
                )
                if not previous_indices or not current_indices:
                    continue
                previous_only = [value for value in previous_indices if value not in current_indices]
                current_only = [value for value in current_indices if value not in previous_indices]
                source_index = previous_only[0] if previous_only else previous_indices[0]
                target_pool = [value for value in (current_only or current_indices) if value != source_index]
                if not target_pool:
                    continue
                target_index = target_pool[0]
                key = (source_index, target_index)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "source_trace_index": int(trace_index),
                        "source_step_index": int(previous.get("step_index", 0) or 0),
                        "target_step_index": int(current.get("step_index", 0) or 0),
                        "source_index": int(source_index),
                        "target_index": int(target_index),
                        "source_active_indices_hash": previous.get("active_indices_hash"),
                        "target_active_indices_hash": current.get("active_indices_hash"),
                    }
                )
                if len(candidates) >= 64:
                    return candidates
        return candidates

    def _rollout_ledger_event_material_hash(self, event: Mapping[str, Any]) -> str:
        device_evidence = (
            event.get("device_evidence")
            if isinstance(event.get("device_evidence"), Mapping)
            else {}
        )
        targets = [
            self._normalized_rollout_replay_target(item, index=index)
            for index, item in enumerate(list(event.get("replay_targets") or [])[:32])
            if isinstance(item, Mapping)
        ]
        return self._sha256_json(
            {
                "rollout_replay_evaluation_hash": event.get("rollout_replay_evaluation_hash"),
                "rollout_hash": event.get("rollout_hash"),
                "rollout_id": event.get("rollout_id"),
                "prediction_hash": event.get("prediction_hash"),
                "current_sparse_code_hash": event.get("current_sparse_code_hash"),
                "transition_memory_evaluation_hash": event.get("transition_memory_evaluation_hash"),
                "persistent_transition_weights_hash": event.get("persistent_transition_weights_hash"),
                "server_transition_memory_hash": event.get("server_transition_memory_hash"),
                "server_transition_memory_hash_match": bool(
                    event.get("server_transition_memory_hash_match")
                ),
                "transition_memory_state_source": event.get("transition_memory_state_source"),
                "device_evidence": {
                    "requested_device": device_evidence.get("requested_device"),
                    "tensor_device": device_evidence.get("tensor_device"),
                    "cuda_tensor": bool(device_evidence.get("cuda_tensor")),
                    "device_source": device_evidence.get("device_source"),
                },
                "targets": targets,
                "state_revision": int(event.get("state_revision", 0) or 0),
            }
        )

    def _ledger_event_material_hash(self, event: Mapping[str, Any]) -> str:
        material = {
            "prediction_hash": event.get("prediction_hash"),
            "transition_memory_evaluation_hash": event.get("transition_memory_evaluation_hash"),
            "persistent_transition_weights_hash": event.get("persistent_transition_weights_hash"),
            "labels": [str(value) for value in list(event.get("labels") or [])][:12],
            "label_grounding": [
                bool(value) for value in list(event.get("label_grounding") or [])[:12]
            ],
            "state_revision": int(event.get("state_revision", 0) or 0),
        }
        return self._sha256_json(material)

    def _normalized_state(self) -> dict[str, Any]:
        state = self._ledger_state()
        raw_events = list(state.get("events") or [])
        raw_rollout_events = list(state.get("rollout_events") or [])
        events = deque(
            (deepcopy(dict(item)) for item in raw_events if isinstance(item, Mapping)),
            maxlen=self._limit,
        )
        rollout_events = deque(
            (deepcopy(dict(item)) for item in raw_rollout_events if isinstance(item, Mapping)),
            maxlen=self._limit,
        )
        return {
            "events": events,
            "rollout_events": rollout_events,
            "total_recorded_count": int(state.get("total_recorded_count", len(events)) or 0),
            "total_rollout_recorded_count": int(
                state.get("total_rollout_recorded_count", len(rollout_events)) or 0
            ),
            "last_recorded_at": state.get("last_recorded_at"),
            "last_rollout_recorded_at": state.get("last_rollout_recorded_at"),
        }

    def _known_readout_evidence_hashes(self) -> set[str]:
        normalized = self._normalized_state()
        return {
            str(item.get("readout_evidence_hash") or "")
            for item in list(normalized.get("events") or [])
            if isinstance(item, Mapping) and item.get("readout_evidence_hash")
        }

    def known_readout_evidence_hashes(self) -> set[str]:
        """Expose current internal ledger identities for controller verification."""

        with self._lock:
            return set(self._known_readout_evidence_hashes())

    def _store_state(self, normalized: Mapping[str, Any]) -> None:
        state = self._ledger_state()
        state["events"] = [deepcopy(item) for item in list(normalized.get("events") or [])[: self._limit]]
        state["rollout_events"] = [
            deepcopy(item)
            for item in list(normalized.get("rollout_events") or [])[: self._limit]
        ]
        state["total_recorded_count"] = int(normalized.get("total_recorded_count", 0) or 0)
        state["total_rollout_recorded_count"] = int(
            normalized.get("total_rollout_recorded_count", 0) or 0
        )
        state["last_recorded_at"] = normalized.get("last_recorded_at")
        state["last_rollout_recorded_at"] = normalized.get("last_rollout_recorded_at")

    @staticmethod
    def _sha256_json(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _label_sparse_indices(labels: Sequence[str]) -> list[int]:
        indices = []
        for label in labels[:12]:
            digest = hashlib.sha256(str(label).encode("utf-8")).hexdigest()
            indices.append(int(digest[:8], 16) % 64)
        return sorted(set(indices))

    @staticmethod
    def _hash_sparse_indices(values: Sequence[str]) -> list[int]:
        indices = []
        for value in values[:8]:
            if not value:
                continue
            digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            indices.append(int(digest[:8], 16) % 64)
            indices.append(int(digest[8:16], 16) % 64)
        return sorted(set(indices))

    @staticmethod
    def _safe_tensor_device(device: str) -> torch.device:
        normalized = str(device or "cpu")
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            return torch.device(normalized)
        except (RuntimeError, TypeError):
            return torch.device("cpu")

    @staticmethod
    def _split_synapse_key(key: str) -> tuple[int | None, int | None]:
        parts = str(key).split(":", maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return None, None
        return int(parts[0]), int(parts[1])

    @classmethod
    def _canonical_synapse_key(cls, key: str) -> bool:
        pre_index, post_index = cls._split_synapse_key(key)
        return (
            cls._valid_language_index(pre_index)
            and cls._valid_language_index(post_index)
            and str(key) == f"{pre_index}:{post_index}"
        )

    @staticmethod
    def _valid_language_index(value: int | None) -> bool:
        return isinstance(value, int) and 0 <= int(value) < _LANGUAGE_NEURON_COUNT


__all__ = ["SNNLanguageReadoutEvidenceLedger"]
