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

    def snapshot(self, *, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            state = self._normalized_state()
            count = max(0, int(limit))
            events = list(state["events"])[:count] if count > 0 else []
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
                    "total_recorded_count": int(state.get("total_recorded_count", 0) or 0),
                    "unique_prediction_count": len(prediction_hashes),
                    "unique_transition_memory_count": len(transition_memory_hashes),
                    "last_recorded_at": state.get("last_recorded_at"),
                },
                "events": [deepcopy(item) for item in events],
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
                "ledger_evidence_present": bool(ledger_event),
                "replay_ledger_evidence_present": replay_hashes_present_in_ledger,
                "ledger_field_match": ledger_field_match,
                "ledger_hash_valid": ledger_hash_valid,
                "provenance_complete": bool(
                    replay_provenance_complete
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
        events = deque(
            (deepcopy(dict(item)) for item in raw_events if isinstance(item, Mapping)),
            maxlen=self._limit,
        )
        return {
            "events": events,
            "total_recorded_count": int(state.get("total_recorded_count", len(events)) or 0),
            "last_recorded_at": state.get("last_recorded_at"),
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
        state["total_recorded_count"] = int(normalized.get("total_recorded_count", 0) or 0)
        state["last_recorded_at"] = normalized.get("last_recorded_at")

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
