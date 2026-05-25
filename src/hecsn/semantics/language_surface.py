from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_cognitive_signal_language_surface(cognitive_signal: Mapping[str, Any]) -> dict[str, Any]:
    """Decode Cognitive Signal telemetry into bounded operator language."""

    prediction_error_mean = _float(cognitive_signal.get("prediction_error_mean"), 0.0)
    prediction_error_max = _float(cognitive_signal.get("prediction_error_max"), 0.0)
    confidence_mean = _float(cognitive_signal.get("predictive_confidence_mean"), 0.5)
    confidence_min = _float(cognitive_signal.get("predictive_confidence_min"), 0.5)
    dopamine = _float(cognitive_signal.get("dopamine"), 0.0)
    norepinephrine = _float(cognitive_signal.get("norepinephrine"), 0.0)
    concept_candidates = _concept_candidates(cognitive_signal.get("concept_candidates"))
    recent_concepts = _strings(cognitive_signal.get("recent_concepts"), limit=4)

    focus = _focus_label(concept_candidates, recent_concepts)
    error_band = _band(prediction_error_mean, low=0.10, high=0.35)
    confidence_band = _confidence_band(confidence_mean)
    arousal_band = _band(max(abs(dopamine), abs(norepinephrine)), low=0.20, high=0.65)
    state_text = (
        f"Cognitive Signal reports {error_band} prediction error "
        f"and {confidence_band} predictive confidence."
    )
    if focus:
        state_text += f" Focus: {focus}."
    if arousal_band != "low":
        state_text += f" Neuromodulator pressure is {arousal_band}."

    return {
        "surface": "subcortical_language.v1",
        "available": True,
        "source": "service.status_read_model.cognitive_signal",
        "state_text": state_text,
        "grounded": True,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "candidate_phrases": _candidate_phrases(concept_candidates, recent_concepts),
        "grounding": {
            "prediction_error_mean": prediction_error_mean,
            "prediction_error_max": prediction_error_max,
            "predictive_confidence_mean": confidence_mean,
            "predictive_confidence_min": confidence_min,
            "dopamine": dopamine,
            "norepinephrine": norepinephrine,
            "concept_focus": focus or None,
            "concept_count": len(concept_candidates),
        },
        "control_hint": _control_hint(
            error_band=error_band,
            confidence_band=confidence_band,
            arousal_band=arousal_band,
            has_focus=bool(focus),
        ),
        "limitations": [
            "Deterministic decode over Cognitive Signal telemetry, not an autonomous generator.",
        ],
    }


def build_subcortical_deliberation_surface(cognitive_signal: Mapping[str, Any]) -> dict[str, Any]:
    """Build grounded control hypotheses from Cognitive Signal pressure."""

    prediction_error_mean = _float(cognitive_signal.get("prediction_error_mean"), 0.0)
    prediction_error_max = _float(cognitive_signal.get("prediction_error_max"), 0.0)
    confidence_mean = _float(cognitive_signal.get("predictive_confidence_mean"), 0.5)
    confidence_min = _float(cognitive_signal.get("predictive_confidence_min"), 0.5)
    dopamine = _float(cognitive_signal.get("dopamine"), 0.0)
    norepinephrine = _float(cognitive_signal.get("norepinephrine"), 0.0)
    concept_candidates = _concept_candidates(cognitive_signal.get("concept_candidates"))
    recent_concepts = _strings(cognitive_signal.get("recent_concepts"), limit=4)
    focus = _focus_label(concept_candidates, recent_concepts)
    error_band = _band(prediction_error_mean, low=0.10, high=0.35)
    confidence_band = _confidence_band(confidence_mean)
    arousal_band = _band(max(abs(dopamine), abs(norepinephrine)), low=0.20, high=0.65)
    control_hint = _control_hint(
        error_band=error_band,
        confidence_band=confidence_band,
        arousal_band=arousal_band,
        has_focus=bool(focus),
    )
    candidates = _deliberation_candidates(
        control_hint=control_hint,
        focus=focus,
        concept_candidates=concept_candidates,
        recent_concepts=recent_concepts,
        prediction_error_mean=prediction_error_mean,
        prediction_error_max=prediction_error_max,
        confidence_mean=confidence_mean,
        confidence_min=confidence_min,
        arousal_band=arousal_band,
    )
    promotion_summary = _promotion_summary(candidates)
    return {
        "surface": "subcortical_control_candidates.v1",
        "available": True,
        "source": "service.status_read_model.cognitive_signal",
        "grounded": True,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "control_hint": control_hint,
        "candidates": candidates,
        "promotion_summary": promotion_summary,
        "grounding": {
            "prediction_error_mean": prediction_error_mean,
            "prediction_error_max": prediction_error_max,
            "predictive_confidence_mean": confidence_mean,
            "predictive_confidence_min": confidence_min,
            "dopamine": dopamine,
            "norepinephrine": norepinephrine,
            "concept_focus": focus or None,
            "concept_count": len(concept_candidates),
        },
        "limitations": [
            "Bounded control hypotheses from Cognitive Signal, not LLM ThoughtLoop output.",
            "Candidates are advisory until replay, policy, or operator evidence promotes them.",
        ],
    }


def build_subcortical_self_repair_surface(spike_health: Mapping[str, Any]) -> dict[str, Any]:
    """Build advisory self-repair candidates from Subcortex Spike Health evidence."""

    health = dict(spike_health)
    activity_state = _text(health.get("activity_state")) or "unknown"
    correlation = health.get("correlation") if isinstance(health.get("correlation"), Mapping) else {}
    correlation_status = _text(correlation.get("status")) or "unknown"
    silent_fraction = _float(health.get("silent_fraction"), 0.0)
    saturated_fraction = _float(health.get("saturated_fraction"), 0.0)
    stale_fraction = _float(health.get("stale_fraction"), 0.0)
    correlation_available = bool(health.get("correlation_evidence_available"))
    candidates = _self_repair_candidates(
        activity_state=activity_state,
        correlation_status=correlation_status,
        silent_fraction=silent_fraction,
        saturated_fraction=saturated_fraction,
        stale_fraction=stale_fraction,
        correlation_available=correlation_available,
        spike_health=health,
    )
    promotion_summary = _promotion_summary(candidates)
    return {
        "surface": "subcortical_self_repair_candidates.v1",
        "available": True,
        "source": "service.status_read_model.runtime_scope.spike_health",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "candidates": candidates,
        "promotion_summary": {
            **promotion_summary,
            "eligible_for_structural_mutation": False,
        },
        "grounding": {
            "activity_state": activity_state,
            "correlation_status": correlation_status,
            "correlation_evidence_available": correlation_available,
            "silent_fraction": silent_fraction,
            "saturated_fraction": saturated_fraction,
            "stale_fraction": stale_fraction,
            "not_liveness_claim": bool(health.get("not_liveness_claim")),
        },
        "limitations": [
            "Advisory self-repair candidates only; status reads must not revive, prune, grow, or mutate runtime state.",
            "Structural mutation requires a separate replay/deep-sleep/operator promotion gate.",
        ],
    }


def _promotion_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    for candidate in candidates:
        gate = candidate.get("promotion_gate") if isinstance(candidate.get("promotion_gate"), Mapping) else {}
        status = _text(gate.get("status")) or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "candidate_count": len(candidates),
        "ready_for_replay_review": statuses.get("ready_for_replay_review", 0),
        "blocked_missing_grounding": statuses.get("blocked_missing_grounding", 0),
        "advisory_monitor_only": statuses.get("advisory_monitor_only", 0),
        "insufficient_evidence": statuses.get("insufficient_evidence", 0),
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "status_counts": statuses,
    }


def attach_cognitive_signal_language_surface(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a Cognitive Signal payload with a language surface attached."""

    enriched = dict(payload)
    if enriched and not isinstance(enriched.get("subcortical_language"), Mapping):
        enriched["subcortical_language"] = build_cognitive_signal_language_surface(enriched)
    if enriched and not isinstance(enriched.get("subcortical_deliberation"), Mapping):
        enriched["subcortical_deliberation"] = build_subcortical_deliberation_surface(enriched)
    return enriched


def _deliberation_candidates(
    *,
    control_hint: str,
    focus: str,
    concept_candidates: Sequence[Mapping[str, Any]],
    recent_concepts: Sequence[str],
    prediction_error_mean: float,
    prediction_error_max: float,
    confidence_mean: float,
    confidence_min: float,
    arousal_band: str,
) -> list[dict[str, Any]]:
    concept_terms = _candidate_terms(concept_candidates, recent_concepts)
    focus_text = focus or "current sensory evidence"
    candidates: list[dict[str, Any]] = []

    if control_hint == "stabilize_prediction":
        candidates.append(
            _candidate(
                intent="stabilize_prediction",
                phase="observe",
                candidate_text=f"Reduce prediction error around {focus_text}.",
                priority=0.90,
                evidence_terms=concept_terms,
                rationale="prediction error is high while confidence is not high",
            )
        )
        candidates.append(
            _candidate(
                intent="replay_counterexample",
                phase="replay",
                candidate_text=f"Replay contrasting evidence before acting on {focus_text}.",
                priority=0.74,
                evidence_terms=concept_terms,
                rationale="unstable prediction needs support or contradiction",
            )
        )
    elif control_hint == "collect_grounding":
        candidates.append(
            _candidate(
                intent="collect_grounding",
                phase="acquire",
                candidate_text="Collect more grounded observations before forming a stronger hypothesis.",
                priority=0.82,
                evidence_terms=concept_terms,
                rationale="no stable concept focus is available",
            )
        )
    elif control_hint == "slow_replay_before_action":
        candidates.append(
            _candidate(
                intent="slow_replay_before_action",
                phase="replay",
                candidate_text=f"Replay recent evidence for {focus_text} before selecting an action.",
                priority=0.86,
                evidence_terms=concept_terms,
                rationale=f"neuromodulator pressure is {arousal_band}",
            )
        )
    elif control_hint == "maintain_current_focus":
        candidates.append(
            _candidate(
                intent="maintain_current_focus",
                phase="monitor",
                candidate_text=f"Maintain current focus on {focus_text} and watch for drift.",
                priority=0.62,
                evidence_terms=concept_terms,
                rationale="prediction error is low and confidence is high",
            )
        )
    else:
        candidates.append(
            _candidate(
                intent="monitor_and_replay",
                phase="monitor",
                candidate_text=f"Monitor {focus_text} and schedule replay if uncertainty persists.",
                priority=0.68,
                evidence_terms=concept_terms,
                rationale="signal pressure is mixed",
            )
        )

    if focus and len(candidates) < 3:
        candidates.append(
            _candidate(
                intent="test_concept_boundary",
                phase="question",
                candidate_text=f"Test what would falsify the current focus: {focus}.",
                priority=max(0.50, min(0.80, 0.45 + prediction_error_max + (1.0 - confidence_min) * 0.20)),
                evidence_terms=concept_terms,
                rationale="concept focus should stay falsifiable",
            )
        )

    for candidate in candidates:
        candidate["grounding"].update(
            {
                "prediction_error_mean": prediction_error_mean,
                "prediction_error_max": prediction_error_max,
                "predictive_confidence_mean": confidence_mean,
                "predictive_confidence_min": confidence_min,
                "concept_focus": focus or None,
            }
        )
        candidate["promotion_gate"] = _candidate_promotion_gate(candidate)
    candidates.sort(key=lambda item: float(item["priority"]), reverse=True)
    return candidates[:3]


def _self_repair_candidates(
    *,
    activity_state: str,
    correlation_status: str,
    silent_fraction: float,
    saturated_fraction: float,
    stale_fraction: float,
    correlation_available: bool,
    spike_health: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if activity_state == "silent_risk":
        candidates.append(
            _candidate(
                intent="review_column_revival",
                phase="repair_review",
                candidate_text="Review dead-column revival or grounding acquisition before allowing more wake training.",
                priority=max(0.78, min(0.95, 0.70 + silent_fraction * 0.25)),
                evidence_terms=["silent_fraction", "last_post_spike_fraction", "win_rate_ema"],
                rationale="Subcortex Spike Health reports broad silence risk",
            )
        )
    if activity_state == "saturation_risk":
        candidates.append(
            _candidate(
                intent="review_inhibitory_balance",
                phase="repair_review",
                candidate_text="Review inhibitory balance, thresholds, or pruning of overused routes before extending autonomy.",
                priority=max(0.76, min(0.94, 0.70 + saturated_fraction * 0.30)),
                evidence_terms=["saturated_fraction", "last_post_spike_fraction", "thresholds"],
                rationale="Subcortex Spike Health reports saturation risk",
            )
        )
    if activity_state == "stale_routing_risk":
        candidates.append(
            _candidate(
                intent="review_stale_column_revival",
                phase="repair_review",
                candidate_text="Review stale routing columns for deep-sleep revival or replay-guided repair.",
                priority=max(0.74, min(0.90, 0.68 + stale_fraction * 0.30)),
                evidence_terms=["stale_fraction", "steps_since_win", "dead_column_steps"],
                rationale="Subcortex Spike Health reports stale routing risk",
            )
        )
    if correlation_status == "overcorrelated_risk":
        candidates.append(
            _candidate(
                intent="review_decorrelation_or_prune",
                phase="repair_review",
                candidate_text="Review decorrelation, inhibitory feedback, or pruning of redundant routes.",
                priority=0.84,
                evidence_terms=["mean_abs_offdiag_correlation", "recent_spike_window", "valid_pairs"],
                rationale="recent winner vectors show over-correlated activity",
            )
        )
    if not candidates:
        candidates.append(
            _candidate(
                intent="monitor_spike_health",
                phase="monitor",
                candidate_text="Continue monitoring Subcortex Spike Health before proposing repair.",
                priority=0.46 if correlation_available else 0.38,
                evidence_terms=["activity_state", "correlation_status", "spike_health"],
                rationale=(
                    "spike-health evidence does not currently justify a repair review"
                    if correlation_available
                    else "spike-health evidence window is not yet sufficient"
                ),
            )
        )

    for candidate in candidates:
        candidate["grounding"].update(
            {
                "activity_state": activity_state,
                "correlation_status": correlation_status,
                "silent_fraction": silent_fraction,
                "saturated_fraction": saturated_fraction,
                "stale_fraction": stale_fraction,
                "correlation_evidence_available": correlation_available,
                "not_liveness_claim": bool(spike_health.get("not_liveness_claim")),
            }
        )
        correlation = spike_health.get("correlation") if isinstance(spike_health.get("correlation"), Mapping) else {}
        if correlation:
            candidate["grounding"]["correlation"] = {
                "sample_count": int(_float(correlation.get("sample_count"), 0.0)),
                "active_columns": int(_float(correlation.get("active_columns"), 0.0)),
                "valid_pairs": int(_float(correlation.get("valid_pairs"), 0.0)),
                "mean_abs_offdiag_correlation": correlation.get("mean_abs_offdiag_correlation"),
            }
        candidate["promotion_gate"] = _self_repair_promotion_gate(candidate)
    candidates.sort(key=lambda item: float(item["priority"]), reverse=True)
    return candidates[:3]


def _self_repair_promotion_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    priority = _float(candidate.get("priority"), 0.0)
    intent = _text(candidate.get("intent"))
    grounding = candidate.get("grounding") if isinstance(candidate.get("grounding"), Mapping) else {}
    evidence_available = bool(grounding.get("correlation_evidence_available")) or intent != "monitor_spike_health"
    review_relevant = intent in {
        "review_column_revival",
        "review_inhibitory_balance",
        "review_stale_column_revival",
        "review_decorrelation_or_prune",
    }
    satisfied = {
        "has_spike_health_evidence": bool(grounding),
        "repair_relevant_intent": review_relevant,
        "priority_at_least_0_70": priority >= 0.70,
        "correlation_window_available_or_not_required": evidence_available,
    }
    if not satisfied["has_spike_health_evidence"] or not evidence_available:
        status = "insufficient_evidence"
        next_gate = "collect_spike_window"
    elif review_relevant and priority >= 0.70:
        status = "ready_for_replay_review"
        next_gate = "deep_sleep_or_replay_repair_gate"
    else:
        status = "advisory_monitor_only"
        next_gate = "continue_monitoring"
    return {
        "status": status,
        "next_gate": next_gate,
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_structural_mutation": False,
        "eligible_for_replay_review": status == "ready_for_replay_review",
        "satisfied_conditions": satisfied,
    }


def _candidate(
    *,
    intent: str,
    phase: str,
    candidate_text: str,
    priority: float,
    evidence_terms: Sequence[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "phase": phase,
        "candidate_text": candidate_text,
        "priority": float(max(0.0, min(1.0, priority))),
        "grounded": True,
        "rationale": rationale,
        "evidence_terms": list(evidence_terms[:6]),
        "grounding": {},
    }


def _candidate_promotion_gate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    intent = _text(candidate.get("intent"))
    phase = _text(candidate.get("phase"))
    evidence_terms = _strings(candidate.get("evidence_terms"), limit=6)
    grounding = candidate.get("grounding") if isinstance(candidate.get("grounding"), Mapping) else {}
    concept_focus = _text(grounding.get("concept_focus"))
    priority = _float(candidate.get("priority"), 0.0)
    replay_relevant = phase in {"replay", "question", "observe"} or intent in {
        "stabilize_prediction",
        "replay_counterexample",
        "test_concept_boundary",
        "slow_replay_before_action",
    }
    satisfied = {
        "has_evidence_terms": bool(evidence_terms),
        "has_concept_focus": bool(concept_focus),
        "priority_at_least_0_70": priority >= 0.70,
        "replay_relevant_phase": replay_relevant,
    }
    if not satisfied["has_evidence_terms"] or not satisfied["has_concept_focus"]:
        status = "blocked_missing_grounding"
        next_gate = "collect_grounding"
    elif replay_relevant and priority >= 0.70:
        status = "ready_for_replay_review"
        next_gate = "operator_or_replay_gate"
    else:
        status = "advisory_monitor_only"
        next_gate = "continue_monitoring"
    return {
        "status": status,
        "next_gate": next_gate,
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_replay_review": status == "ready_for_replay_review",
        "satisfied_conditions": satisfied,
    }


def _candidate_terms(
    concept_candidates: Sequence[Mapping[str, Any]],
    recent_concepts: Sequence[str],
) -> list[str]:
    terms: list[str] = []
    for candidate in concept_candidates:
        for value in [candidate.get("label"), *_strings(candidate.get("top_terms"), limit=4)]:
            text = _text(value)
            if text and text not in terms:
                terms.append(text)
            if len(terms) >= 6:
                return terms
    for concept in recent_concepts:
        if concept and concept not in terms:
            terms.append(concept)
        if len(terms) >= 6:
            break
    return terms


def _candidate_phrases(
    concept_candidates: Sequence[Mapping[str, Any]],
    recent_concepts: Sequence[str],
) -> list[str]:
    phrases: list[str] = []
    for candidate in concept_candidates:
        label = _text(candidate.get("label"))
        top_terms = _strings(candidate.get("top_terms"), limit=3)
        if label and top_terms:
            phrases.append(f"{label}: {', '.join(top_terms)}")
        elif label:
            phrases.append(label)
        elif top_terms:
            phrases.append(", ".join(top_terms))
        if len(phrases) >= 3:
            return phrases
    for concept in recent_concepts:
        if concept not in phrases:
            phrases.append(concept)
        if len(phrases) >= 3:
            break
    return phrases


def _concept_candidates(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    candidates: list[Mapping[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            candidates.append(item)
    return candidates[:6]


def _control_hint(
    *,
    error_band: str,
    confidence_band: str,
    arousal_band: str,
    has_focus: bool,
) -> str:
    if error_band == "high" and confidence_band != "high":
        return "stabilize_prediction"
    if not has_focus:
        return "collect_grounding"
    if arousal_band == "high":
        return "slow_replay_before_action"
    if confidence_band == "high" and error_band == "low":
        return "maintain_current_focus"
    return "monitor_and_replay"


def _focus_label(
    concept_candidates: Sequence[Mapping[str, Any]],
    recent_concepts: Sequence[str],
) -> str:
    scored: list[tuple[float, str]] = []
    for candidate in concept_candidates:
        label = _text(candidate.get("label"))
        if not label:
            continue
        observations = _float(candidate.get("observations"), _float(candidate.get("match_count"), 0.0))
        coherence = _float(candidate.get("temporal_coherence"), 0.0)
        uncertainty = _float(candidate.get("uncertainty"), 1.0)
        scored.append((observations + coherence - uncertainty, label))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1]
    return recent_concepts[0] if recent_concepts else ""


def _band(value: float, *, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "moderate"
    return "low"


def _confidence_band(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "moderate"
    return "low"


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _strings(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[str] = []
    for item in value:
        text = _text(item)
        if text:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()
