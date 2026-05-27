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


def build_subcortical_spike_readout_evidence(
    cognitive_signal: Mapping[str, Any],
    runtime_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build HECSN-owned spike-language readout evidence without generating text."""

    prediction_error_mean = _float(cognitive_signal.get("prediction_error_mean"), 0.0)
    prediction_error_max = _float(cognitive_signal.get("prediction_error_max"), 0.0)
    confidence_mean = _float(cognitive_signal.get("predictive_confidence_mean"), 0.5)
    confidence_min = _float(cognitive_signal.get("predictive_confidence_min"), 0.5)
    dopamine = _float(cognitive_signal.get("dopamine"), 0.0)
    norepinephrine = _float(cognitive_signal.get("norepinephrine"), 0.0)
    concept_candidates = _concept_candidates(cognitive_signal.get("concept_candidates"))
    recent_concepts = _strings(cognitive_signal.get("recent_concepts"), limit=4)
    focus = _focus_label(concept_candidates, recent_concepts)
    cuda_runtime = (
        runtime_scope.get("cuda_first_runtime")
        if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
        else {}
    )
    subcortex_devices = (
        cuda_runtime.get("subcortex_tensor_devices")
        if isinstance(cuda_runtime.get("subcortex_tensor_devices"), Mapping)
        else {}
    )
    device = _spike_readout_device(cuda_runtime, subcortex_devices)
    pressure = max(prediction_error_mean, 1.0 - confidence_min, abs(dopamine), abs(norepinephrine))
    pressure_band = _band(pressure, low=0.20, high=0.65)
    readout_slots = _spike_readout_slots(
        concept_candidates=concept_candidates,
        recent_concepts=recent_concepts,
        focus=focus,
        pressure_band=pressure_band,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_spike_readout_evidence",
        "surface": "subcortical_spike_readout_evidence.v1",
        "available": True,
        "source": "service.status_read_model.cognitive_signal_and_runtime_scope",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "generates_text": False,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "device_evidence": {
            "device": device,
            "cuda_report_available": bool(cuda_runtime),
            "subcortex_device_evidence_available": bool(subcortex_devices),
            "cuda_device_selected": _is_cuda_device(device),
        },
        "population_code": {
            "prediction_error_band": _band(prediction_error_mean, low=0.10, high=0.35),
            "prediction_error_peak_band": _band(prediction_error_max, low=0.20, high=0.60),
            "confidence_band": _confidence_band(confidence_mean),
            "confidence_floor_band": _confidence_band(confidence_min),
            "neuromodulator_pressure_band": _band(max(abs(dopamine), abs(norepinephrine)), low=0.20, high=0.65),
            "readout_pressure_band": pressure_band,
            "concept_focus": focus or None,
            "concept_count": len(concept_candidates),
        },
        "readout_slots": readout_slots,
        "promotion_constraints": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_language_generation": False,
            "requires_hecsn_owned_decoder": True,
            "requires_grounding_support": True,
            "requires_sparsity_evidence": True,
            "requires_device_evidence": True,
            "requires_replay_or_eval_dataset_report": True,
        },
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
            "Readout evidence only; it does not decode, generate, sample, or train language.",
            "The population code is deterministic telemetry evidence for a future HECSN-owned spike decoder.",
        ],
    }


def build_snn_language_readiness_surface(
    cognitive_signal: Mapping[str, Any],
    runtime_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a read-only readiness gate for future SNN-native language generation."""

    language_surface = build_cognitive_signal_language_surface(cognitive_signal)
    deliberation_surface = build_subcortical_deliberation_surface(cognitive_signal)
    spike_readout_evidence = build_subcortical_spike_readout_evidence(cognitive_signal, runtime_scope)
    cuda_runtime = (
        runtime_scope.get("cuda_first_runtime")
        if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
        else {}
    )
    subcortex_devices = (
        cuda_runtime.get("subcortex_tensor_devices")
        if isinstance(cuda_runtime.get("subcortex_tensor_devices"), Mapping)
        else {}
    )
    readiness_checks = _snn_language_readiness_checks(
        language_surface=language_surface,
        deliberation_surface=deliberation_surface,
        spike_readout_evidence=spike_readout_evidence,
        cuda_runtime=cuda_runtime,
        subcortex_devices=subcortex_devices,
    )
    promotion_gate = _snn_language_promotion_gate(readiness_checks)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_snn_native_language_readiness_gate",
        "surface": "snn_native_language_readiness.v1",
        "available": True,
        "endpoint": "/terminus/snn-language-readiness",
        "review_role": "operator_hecsn_native_snn_language_review_only",
        "source": "service.status_read_model.cognitive_signal_and_runtime_scope",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "current_language_surface": {
            "surface": language_surface.get("surface"),
            "source": language_surface.get("source"),
            "grounded": bool(language_surface.get("grounded")),
            "not_cognition_substrate": bool(language_surface.get("not_cognition_substrate")),
            "retired_runtime_dependency": bool(language_surface.get("retired_runtime_dependency")),
            "concept_focus": (language_surface.get("grounding") or {}).get("concept_focus")
            if isinstance(language_surface.get("grounding"), Mapping)
            else None,
            "candidate_phrase_count": len(list(language_surface.get("candidate_phrases") or [])),
        },
        "current_deliberation_surface": {
            "surface": deliberation_surface.get("surface"),
            "grounded": bool(deliberation_surface.get("grounded")),
            "not_cognition_substrate": bool(deliberation_surface.get("not_cognition_substrate")),
            "retired_runtime_dependency": bool(deliberation_surface.get("retired_runtime_dependency")),
            "candidate_count": int(
                (deliberation_surface.get("promotion_summary") or {}).get("candidate_count", 0)
                if isinstance(deliberation_surface.get("promotion_summary"), Mapping)
                else 0
            ),
            "ready_for_replay_review": int(
                (deliberation_surface.get("promotion_summary") or {}).get("ready_for_replay_review", 0)
                if isinstance(deliberation_surface.get("promotion_summary"), Mapping)
                else 0
            ),
        },
        "current_spike_readout_evidence": {
            "surface": spike_readout_evidence.get("surface"),
            "grounded": bool(spike_readout_evidence.get("grounded")),
            "not_cognition_substrate": bool(spike_readout_evidence.get("not_cognition_substrate")),
            "retired_runtime_dependency": bool(spike_readout_evidence.get("retired_runtime_dependency")),
            "generates_text": bool(spike_readout_evidence.get("generates_text")),
            "readout_slot_count": len(list(spike_readout_evidence.get("readout_slots") or [])),
            "device": (
                (spike_readout_evidence.get("device_evidence") or {}).get("device")
                if isinstance(spike_readout_evidence.get("device_evidence"), Mapping)
                else None
            ),
            "cuda_device_selected": bool(
                (spike_readout_evidence.get("device_evidence") or {}).get("cuda_device_selected")
                if isinstance(spike_readout_evidence.get("device_evidence"), Mapping)
                else False
            ),
        },
        "research_candidates": [
            {
                "name": "NeuronSpark",
                "kind": "pure_snn_language_implementation_reference",
                "reported_scale": "0.9B parameters",
                "integration_status": "reference_for_hecsn_owned_reimplementation",
                "required_local_evidence": [
                    "hecsn_owned_language_neuron_module",
                    "hecsn_native_snn_decoder",
                    "hecsn_controlled_training_loop",
                    "sparsity_report",
                    "cuda_or_accelerator_device_report",
                    "grounding_support_report",
                    "repo_owned_training_or_replay_evaluation",
                ],
            },
            {
                "name": "Nord-AI",
                "kind": "pure_snn_language_implementation_reference",
                "reported_scale": "144M to 1B-class community checkpoints",
                "integration_status": "reference_for_hecsn_owned_reimplementation",
                "required_local_evidence": [
                    "hecsn_owned_language_neuron_module",
                    "hecsn_native_snn_decoder",
                    "hecsn_controlled_training_loop",
                    "activation_sparsity_report",
                    "cuda_or_accelerator_device_report",
                    "grounding_support_report",
                    "repo_owned_training_or_replay_evaluation",
                ],
            },
        ],
        "readiness_checks": readiness_checks,
        "promotion_gate": promotion_gate,
        "success_evidence": [
            "local_snn_generator_device_report",
            "activation_sparsity_report",
            "grounding_support_report",
            "runtime_truth_delta",
            "replay_or_eval_dataset_report",
            "retired_runtime_dependency_absent",
        ],
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "requires_operator_approval": True,
            "requires_grounding_support": True,
            "requires_sparsity_evidence": True,
            "requires_device_evidence": True,
            "requires_no_retired_runtime_dependency": True,
            "requires_hecsn_owned_implementation": True,
            "requires_hecsn_controlled_training": True,
        },
        "limitations": [
            "Readiness gate only; it does not load external SNN language checkpoints, add external dependencies, or generate text.",
            "NeuronSpark and Nord-AI are implementation references only; HECSN must own the language neurons, decoder, training loop, grounding, telemetry, and promotion gates.",
            "Language remains a grounded Subcortex surface until a HECSN-native SNN generator proves sparsity, device placement, grounding support, and operator-controlled training.",
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
    promotion_gate = _self_repair_surface_promotion_gate(candidates)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_self_repair_gate_plan",
        "surface": "subcortical_self_repair_candidates.v1",
        "available": True,
        "endpoint": "/terminus/subcortical-self-repair",
        "review_role": "operator_replay_deep_sleep_review_only",
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
        "promotion_gate": promotion_gate,
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


def build_subcortical_self_repair_evaluation_surface(spike_health: Mapping[str, Any]) -> dict[str, Any]:
    """Build a read-only evaluation plan for gated Subcortex self-repair pressure."""

    repair_surface = build_subcortical_self_repair_surface(spike_health)
    promotion_gate = (
        repair_surface.get("promotion_gate")
        if isinstance(repair_surface.get("promotion_gate"), Mapping)
        else {}
    )
    evaluation_cases = _self_repair_evaluation_cases(repair_surface)
    evaluation_gate = _self_repair_evaluation_gate(promotion_gate, evaluation_cases)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_self_repair_evaluation_plan",
        "surface": "subcortical_self_repair_evaluation.v1",
        "available": True,
        "endpoint": "/terminus/subcortical-self-repair/evaluation",
        "review_role": "operator_replay_deep_sleep_evaluation_only",
        "source": "service.status_read_model.runtime_scope.spike_health",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "repair_surface": {
            "surface": repair_surface.get("surface"),
            "artifact_kind": repair_surface.get("artifact_kind"),
            "promotion_status": promotion_gate.get("status"),
            "next_gate": promotion_gate.get("next_gate"),
            "candidate_count": int(
                (repair_surface.get("promotion_summary") or {}).get("candidate_count", 0)
                if isinstance(repair_surface.get("promotion_summary"), Mapping)
                else 0
            ),
        },
        "evaluation_gate": evaluation_gate,
        "evaluation_cases": evaluation_cases,
        "success_evidence": [
            "pre_repair_spike_health_snapshot",
            "post_repair_spike_health_snapshot",
            "runtime_truth_delta",
            "replay_or_deep_sleep_run_id",
            "device_evidence_report",
        ],
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "requires_operator_approval": True,
            "requires_isolated_replay_or_deep_sleep": True,
            "requires_runtime_truth_improvement": True,
            "requires_device_evidence": True,
        },
        "limitations": [
            "Evaluation plan only; it does not run replay, deep sleep, pruning, growth, or column revival.",
            "Promotion still requires separate operator approval and measured Runtime Truth improvement.",
        ],
    }


def build_subcortical_structural_plasticity_surface(
    concept_store: Mapping[str, Any],
    runtime_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a read-only structural-plasticity promotion artifact."""

    growth = concept_store.get("growth") if isinstance(concept_store.get("growth"), Mapping) else {}
    cuda_runtime = (
        runtime_scope.get("cuda_first_runtime")
        if isinstance(runtime_scope.get("cuda_first_runtime"), Mapping)
        else {}
    )
    subcortex_devices = (
        cuda_runtime.get("subcortex_tensor_devices")
        if isinstance(cuda_runtime.get("subcortex_tensor_devices"), Mapping)
        else {}
    )
    binding_report = (
        subcortex_devices.get("binding")
        if isinstance(subcortex_devices.get("binding"), Mapping)
        else {}
    )
    competitive_report = (
        subcortex_devices.get("competitive")
        if isinstance(subcortex_devices.get("competitive"), Mapping)
        else {}
    )
    local_plasticity_report = (
        competitive_report.get("local_plasticity")
        if isinstance(competitive_report.get("local_plasticity"), Mapping)
        else {}
    )
    spike_health = runtime_scope.get("spike_health") if isinstance(runtime_scope.get("spike_health"), Mapping) else {}
    weight_distribution = (
        runtime_scope.get("weight_distribution")
        if isinstance(runtime_scope.get("weight_distribution"), Mapping)
        else {}
    )
    structural_cases = _structural_plasticity_cases(
        growth,
        binding_report,
        cuda_runtime,
        local_plasticity_report,
        spike_health,
        weight_distribution,
    )
    promotion_gate = _structural_plasticity_promotion_gate(structural_cases, binding_report)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_structural_plasticity_gate_plan",
        "surface": "subcortical_structural_plasticity.v1",
        "available": True,
        "endpoint": "/terminus/subcortical-structural-plasticity",
        "review_role": "operator_structural_plasticity_review_only",
        "source": "service.status_read_model.concept_store_and_runtime_scope",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "retired_runtime_dependency": False,
        "promotion_gate": promotion_gate,
        "structural_cases": structural_cases,
        "concept_growth": {
            "growth_ready": bool(growth.get("growth_ready")),
            "requested_output_dim": int(_float(growth.get("requested_output_dim"), 0.0)),
            "current_output_dim": int(_float(growth.get("current_output_dim"), 0.0)),
            "max_output_dim": int(_float(growth.get("max_output_dim"), 0.0)),
            "expansion_events": int(_float(growth.get("expansion_events"), 0.0)),
            "prune_events": int(_float(growth.get("prune_events"), 0.0)),
            "active_growth_concepts": [
                _structural_growth_concept(item)
                for item in list(growth.get("active_growth_concepts") or [])[:4]
                if isinstance(item, Mapping)
            ],
        },
        "binding_topology": _binding_structural_summary(binding_report),
        "local_plasticity": _local_plasticity_summary(
            local_plasticity_report,
            spike_health,
            weight_distribution,
        ),
        "device_evidence": _structural_plasticity_device_evidence(
            cuda_runtime,
            binding_report,
            local_plasticity_report,
        ),
        "success_evidence": [
            "pre_mutation_structural_snapshot",
            "post_mutation_structural_snapshot",
            "local_plasticity_stability_delta",
            "runtime_truth_delta",
            "spike_health_delta",
            "device_evidence_report",
            "rollback_policy",
        ],
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "requires_operator_approval": True,
            "requires_isolated_evaluation": True,
            "requires_runtime_truth_improvement": True,
            "requires_reversible_mutation_ledger": True,
            "requires_device_evidence": True,
        },
        "limitations": [
            "Read-only structural-plasticity pressure; it does not call ConceptStore.refresh_structural_capacity or binding growth/prune methods.",
            "Local STDP, homeostatic scaling, and inhibitory-balance evidence can support readiness, but this artifact cannot change synapses or topology.",
            "Structural mutation remains disabled until a separate isolated evaluation proves Runtime Truth, spike-health, rollback, and device evidence improved.",
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


def _self_repair_evaluation_cases(repair_surface: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for candidate in list(repair_surface.get("candidates") or []):
        if not isinstance(candidate, Mapping):
            continue
        gate = candidate.get("promotion_gate") if isinstance(candidate.get("promotion_gate"), Mapping) else {}
        grounding = candidate.get("grounding") if isinstance(candidate.get("grounding"), Mapping) else {}
        status = _text(gate.get("status"))
        intent = _text(candidate.get("intent"))
        cases.append(
            {
                "intent": intent,
                "phase": _text(candidate.get("phase")),
                "priority": _float(candidate.get("priority"), 0.0),
                "promotion_status": status or "unknown",
                "ready_for_evaluation": status == "ready_for_replay_review",
                "evaluation_target": _self_repair_evaluation_target(intent),
                "baseline_metrics": _self_repair_baseline_metrics(grounding),
                "required_evidence": [
                    "baseline_spike_health",
                    "candidate_specific_counterfactual",
                    "post_evaluation_spike_health",
                    "runtime_truth_delta",
                    "rollback_policy",
                ],
                "promotion_constraints": {
                    "eligible_for_action": False,
                    "eligible_for_fact_promotion": False,
                    "eligible_for_structural_mutation": False,
                    "next_gate": "operator_approved_isolated_evaluation"
                    if status == "ready_for_replay_review"
                    else gate.get("next_gate"),
                },
            }
        )
    return cases[:3]


def _snn_language_readiness_checks(
    *,
    language_surface: Mapping[str, Any],
    deliberation_surface: Mapping[str, Any],
    spike_readout_evidence: Mapping[str, Any],
    cuda_runtime: Mapping[str, Any],
    subcortex_devices: Mapping[str, Any],
) -> dict[str, bool]:
    readout_device = (
        spike_readout_evidence.get("device_evidence")
        if isinstance(spike_readout_evidence.get("device_evidence"), Mapping)
        else {}
    )
    return {
        "grounded_language_surface_available": bool(language_surface.get("grounded")),
        "language_surface_not_cognition_substrate": bool(language_surface.get("not_cognition_substrate")),
        "deliberation_surface_available": bool(deliberation_surface.get("grounded")),
        "deliberation_not_llm_thought_loop": not bool(deliberation_surface.get("retired_runtime_dependency")),
        "hecsn_spike_readout_evidence_available": bool(spike_readout_evidence.get("available")),
        "hecsn_spike_readout_grounded": bool(spike_readout_evidence.get("grounded")),
        "hecsn_spike_readout_non_generative": not bool(spike_readout_evidence.get("generates_text")),
        "hecsn_spike_readout_device_evidence_available": bool(
            readout_device.get("cuda_report_available")
            or readout_device.get("subcortex_device_evidence_available")
        ),
        "cuda_runtime_scope_available": bool(cuda_runtime),
        "subcortex_device_evidence_available": bool(subcortex_devices),
        "local_snn_language_generator_available": bool(cuda_runtime.get("snn_language_generator_device_report")),
        "activation_sparsity_report_available": bool(cuda_runtime.get("snn_language_activation_sparsity")),
        "grounding_support_report_available": bool(cuda_runtime.get("snn_language_grounding_support")),
        "retired_runtime_dependency_absent": not bool(language_surface.get("retired_runtime_dependency")),
    }


def _snn_language_promotion_gate(readiness_checks: Mapping[str, bool]) -> dict[str, Any]:
    required_now = {
        "grounded_language_surface_available": bool(readiness_checks.get("grounded_language_surface_available")),
        "language_surface_not_cognition_substrate": bool(readiness_checks.get("language_surface_not_cognition_substrate")),
        "deliberation_not_llm_thought_loop": bool(readiness_checks.get("deliberation_not_llm_thought_loop")),
        "hecsn_spike_readout_evidence_available": bool(
            readiness_checks.get("hecsn_spike_readout_evidence_available")
        ),
        "hecsn_spike_readout_grounded": bool(readiness_checks.get("hecsn_spike_readout_grounded")),
        "hecsn_spike_readout_non_generative": bool(
            readiness_checks.get("hecsn_spike_readout_non_generative")
        ),
        "hecsn_spike_readout_device_evidence_available": bool(
            readiness_checks.get("hecsn_spike_readout_device_evidence_available")
        ),
        "cuda_runtime_scope_available": bool(readiness_checks.get("cuda_runtime_scope_available")),
        "subcortex_device_evidence_available": bool(readiness_checks.get("subcortex_device_evidence_available")),
        "retired_runtime_dependency_absent": bool(readiness_checks.get("retired_runtime_dependency_absent")),
    }
    generator_ready = (
        bool(readiness_checks.get("local_snn_language_generator_available"))
        and bool(readiness_checks.get("activation_sparsity_report_available"))
        and bool(readiness_checks.get("grounding_support_report_available"))
    )
    if not all(required_now.values()):
        status = "blocked_missing_subcortex_evidence"
        next_gate = "complete_grounded_subcortex_language_evidence"
    elif not generator_ready:
        status = "research_candidate_only"
        next_gate = "build_local_snn_language_generator_adapter"
    else:
        status = "ready_for_isolated_language_evaluation"
        next_gate = "operator_approved_snn_language_evaluation"
    return {
        "status": status,
        "next_gate": next_gate,
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_cognition_substrate": False,
        "eligible_for_language_generation": status == "ready_for_isolated_language_evaluation",
        "requires_operator_approval": status == "ready_for_isolated_language_evaluation",
        "required_subcortex_conditions": required_now,
        "missing_generator_conditions": {
            "local_snn_language_generator_available": not bool(
                readiness_checks.get("local_snn_language_generator_available")
            ),
            "activation_sparsity_report_available": not bool(
                readiness_checks.get("activation_sparsity_report_available")
            ),
            "grounding_support_report_available": not bool(
                readiness_checks.get("grounding_support_report_available")
            ),
        },
    }


def _structural_plasticity_cases(
    growth: Mapping[str, Any],
    binding_report: Mapping[str, Any],
    cuda_runtime: Mapping[str, Any],
    local_plasticity_report: Mapping[str, Any],
    spike_health: Mapping[str, Any],
    weight_distribution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if growth:
        growth_ready = bool(growth.get("growth_ready"))
        active_growth_concepts = [
            _structural_growth_concept(item)
            for item in list(growth.get("active_growth_concepts") or [])[:4]
            if isinstance(item, Mapping)
        ]
        cases.append(
            {
                "intent": "evaluate_concept_capacity_growth",
                "phase": "structural_review",
                "ready_for_evaluation": growth_ready,
                "promotion_status": "ready_for_structural_review" if growth_ready else "monitor_only",
                "priority": 0.82 if growth_ready else 0.42,
                "evaluation_target": "increase_abstraction_capacity_without_memory_or_spike_health_regression",
                "baseline_metrics": {
                    "requested_output_dim": int(_float(growth.get("requested_output_dim"), 0.0)),
                    "current_output_dim": int(_float(growth.get("current_output_dim"), 0.0)),
                    "max_output_dim": int(_float(growth.get("max_output_dim"), 0.0)),
                    "growth_ready": growth_ready,
                    "active_growth_concept_count": len(active_growth_concepts),
                    "expansion_events": int(_float(growth.get("expansion_events"), 0.0)),
                    "prune_events": int(_float(growth.get("prune_events"), 0.0)),
                },
                "evidence_terms": ["concept_growth", "split_bias", "growth_pressure", "abstraction_capacity"],
                "promotion_constraints": _structural_promotion_constraints(growth_ready),
            }
        )
    binding_summary = _binding_structural_summary(binding_report)
    mutation_count = int(binding_summary["growth_events"]) + int(binding_summary["prune_events"])
    if binding_report:
        cases.append(
            {
                "intent": "evaluate_binding_topology_stability",
                "phase": "structural_review",
                "ready_for_evaluation": mutation_count > 0,
                "promotion_status": "ready_for_structural_review" if mutation_count > 0 else "monitor_only",
                "priority": 0.78 if mutation_count > 0 else 0.40,
                "evaluation_target": "preserve_sparse_binding_reachability_with_auditable_edge_ledger",
                "baseline_metrics": binding_summary,
                "evidence_terms": ["binding_topology", "structural_mutation_ledger", "sparse_edges", "device_report"],
                "promotion_constraints": _structural_promotion_constraints(mutation_count > 0),
            }
        )
    local_summary = _local_plasticity_summary(local_plasticity_report, spike_health, weight_distribution)
    if local_plasticity_report:
        local_pressure = (
            bool(local_summary["spike_health_risk"])
            or bool(local_summary["synaptic_validation_failed"])
            or bool(growth.get("growth_ready"))
        )
        ready_for_local_review = (
            bool(local_summary["device_evidence_available"])
            and bool(local_summary["eligibility_traces_available"])
            and local_pressure
        )
        cases.append(
            {
                "intent": "evaluate_local_plasticity_growth_prune_pressure",
                "phase": "structural_review",
                "ready_for_evaluation": ready_for_local_review,
                "promotion_status": "ready_for_structural_review" if ready_for_local_review else "monitor_only",
                "priority": 0.76 if ready_for_local_review else 0.38,
                "evaluation_target": "prove_local_stdp_scaling_and_inhibition_stabilize_growth_or_prune_pressure",
                "baseline_metrics": local_summary,
                "evidence_terms": [
                    "local_plasticity",
                    "eligibility_traces",
                    "homeostatic_scaling",
                    "inhibitory_balance",
                    "device_report",
                ],
                "promotion_constraints": _structural_promotion_constraints(ready_for_local_review),
            }
        )
    if not cases:
        cases.append(
            {
                "intent": "collect_structural_plasticity_evidence",
                "phase": "monitor",
                "ready_for_evaluation": False,
                "promotion_status": "insufficient_evidence",
                "priority": 0.30,
                "evaluation_target": "collect_concept_growth_and_binding_device_evidence",
                "baseline_metrics": {
                    "runtime_scope_has_cuda_first_runtime": bool(cuda_runtime),
                    "binding_report_available": bool(binding_report),
                    "local_plasticity_report_available": bool(local_plasticity_report),
                },
                "evidence_terms": [
                    "concept_store_growth",
                    "binding_device_report",
                    "local_plasticity_device_report",
                    "runtime_scope",
                ],
                "promotion_constraints": _structural_promotion_constraints(False),
            }
        )
    cases.sort(key=lambda item: float(item["priority"]), reverse=True)
    return cases[:3]


def _structural_plasticity_promotion_gate(
    structural_cases: Sequence[Mapping[str, Any]],
    binding_report: Mapping[str, Any],
) -> dict[str, Any]:
    ready_cases = [case for case in structural_cases if bool(case.get("ready_for_evaluation"))]
    device_evidence_available = bool(binding_report) or any(
        "concept_growth" in list(case.get("evidence_terms") or [])
        or "local_plasticity" in list(case.get("evidence_terms") or [])
        for case in structural_cases
    )
    if ready_cases and device_evidence_available:
        status = "ready_for_isolated_structural_evaluation"
        next_gate = "operator_approved_structural_plasticity_evaluation"
    elif not device_evidence_available:
        status = "insufficient_device_evidence"
        next_gate = "collect_cuda_structural_device_report"
    else:
        status = "monitor_only"
        next_gate = "continue_monitoring"
    return {
        "status": status,
        "next_gate": next_gate,
        "ready_case_count": len(ready_cases),
        "case_count": len(structural_cases),
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_structural_mutation": False,
        "eligible_for_replay_review": status == "ready_for_isolated_structural_evaluation",
        "requires_operator_approval": status == "ready_for_isolated_structural_evaluation",
    }


def _structural_promotion_constraints(ready: bool) -> dict[str, Any]:
    return {
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_structural_mutation": False,
        "next_gate": "operator_approved_structural_evaluation" if ready else "continue_monitoring",
    }


def _structural_growth_concept(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "concept_id": _text(item.get("concept_id")),
        "label": _text(item.get("label")),
        "growth_pressure": _float(item.get("growth_pressure"), 0.0),
        "split_bias": _float(item.get("split_bias"), 0.0),
        "observations": int(_float(item.get("observations"), 0.0)),
        "top_terms": _strings(item.get("top_terms"), limit=6),
    }


def _binding_structural_summary(binding_report: Mapping[str, Any]) -> dict[str, Any]:
    structural_mutations = (
        binding_report.get("structural_mutations")
        if isinstance(binding_report.get("structural_mutations"), Mapping)
        else {}
    )
    return {
        "module": binding_report.get("module"),
        "topology": binding_report.get("topology"),
        "growth_events": int(_float(structural_mutations.get("growth_events"), 0.0)),
        "prune_events": int(_float(structural_mutations.get("prune_events"), 0.0)),
        "edges_added_total": int(_float(structural_mutations.get("edges_added_total"), 0.0)),
        "edges_removed_total": int(_float(structural_mutations.get("edges_removed_total"), 0.0)),
        "recent_events": [
            dict(item)
            for item in list(structural_mutations.get("recent_events") or [])[:6]
            if isinstance(item, Mapping)
        ],
    }


def _local_plasticity_summary(
    local_plasticity_report: Mapping[str, Any],
    spike_health: Mapping[str, Any] | None = None,
    weight_distribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    health = spike_health or {}
    weights = weight_distribution or {}
    device_keys = _local_plasticity_device_keys(local_plasticity_report)
    eligibility_keys = {
        "input_eligibility_device",
        "projection_eligibility_device",
        "assembly_projection_eligibility_device",
    }
    homeostatic_keys = {
        "firing_rate_ema_device",
        "synaptic_scale_device",
        "inhibitory_trace_device",
        "inhibitory_tone_device",
    }
    synaptic_validation_passed = bool(weights.get("validates_full_log_stdp_weight_target", False))
    activity_state = _text(health.get("activity_state")) or "unknown"
    correlation = health.get("correlation") if isinstance(health.get("correlation"), Mapping) else {}
    correlation_status = _text(correlation.get("status"))
    spike_health_risk = activity_state in {
        "silent_risk",
        "saturation_risk",
        "stale_routing_risk",
    } or correlation_status == "overcorrelated_risk"
    return {
        "available": bool(local_plasticity_report),
        "module": local_plasticity_report.get("module"),
        "device": local_plasticity_report.get("device"),
        "spike_backend": local_plasticity_report.get("spike_backend"),
        "plasticity_rule": local_plasticity_report.get("plasticity_rule"),
        "activity_state": activity_state,
        "silent_fraction": _float(health.get("silent_fraction"), 0.0),
        "saturated_fraction": _float(health.get("saturated_fraction"), 0.0),
        "stale_fraction": _float(health.get("stale_fraction"), 0.0),
        "spike_health_risk": bool(spike_health_risk),
        "synaptic_validation_available": "validates_full_log_stdp_weight_target" in weights,
        "synaptic_validation_passed": synaptic_validation_passed,
        "synaptic_validation_failed": "validates_full_log_stdp_weight_target" in weights
        and not synaptic_validation_passed,
        "device_evidence_available": bool(device_keys),
        "eligibility_traces_available": eligibility_keys.issubset(device_keys),
        "homeostatic_state_available": homeostatic_keys.issubset(device_keys),
        "adex_state_available": bool(local_plasticity_report.get("adex_device")),
        "device_key_count": len(device_keys),
    }


def _local_plasticity_device_keys(local_plasticity_report: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, value in local_plasticity_report.items()
        if str(key).endswith("_device") and value is not None
    }


def _structural_plasticity_device_evidence(
    cuda_runtime: Mapping[str, Any],
    binding_report: Mapping[str, Any],
    local_plasticity_report: Mapping[str, Any],
) -> dict[str, Any]:
    keys = [
        "device",
        "binding_state_device",
        "neighbor_ids_device",
        "neighbor_weights_device",
        "learned_weights_device",
        "degree_device",
    ]
    local_keys = [
        "device",
        "pre_trace_device",
        "post_trace_device",
        "input_eligibility_device",
        "projection_eligibility_device",
        "assembly_projection_eligibility_device",
        "firing_rate_ema_device",
        "synaptic_scale_device",
        "inhibitory_trace_device",
        "inhibitory_tone_device",
        "adex_device",
        "adex_voltage_device",
        "adex_adaptation_device",
        "adex_spike_times_device",
    ]
    return {
        "cuda_first_runtime_available": bool(cuda_runtime),
        "tensor_device": cuda_runtime.get("tensor_device"),
        "routing_search_device": cuda_runtime.get("routing_search_device"),
        "binding_report_available": bool(binding_report),
        "local_plasticity_report_available": bool(local_plasticity_report),
        "binding_devices": {
            key: binding_report.get(key)
            for key in keys
            if key in binding_report
        },
        "local_plasticity_devices": {
            key: local_plasticity_report.get(key)
            for key in local_keys
            if key in local_plasticity_report
        },
    }


def _self_repair_evaluation_gate(
    promotion_gate: Mapping[str, Any],
    evaluation_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    promotion_status = _text(promotion_gate.get("status"))
    ready_cases = [case for case in evaluation_cases if bool(case.get("ready_for_evaluation"))]
    if promotion_status == "ready_for_replay_review" and ready_cases:
        status = "ready_for_isolated_evaluation"
        next_gate = "operator_approved_deep_sleep_or_replay_evaluation"
    elif promotion_status == "insufficient_evidence":
        status = "blocked_missing_spike_window"
        next_gate = "collect_spike_window"
    else:
        status = "monitor_only"
        next_gate = "continue_monitoring"
    return {
        "status": status,
        "next_gate": next_gate,
        "ready_case_count": len(ready_cases),
        "case_count": len(evaluation_cases),
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_structural_mutation": False,
        "eligible_for_replay_review": status == "ready_for_isolated_evaluation",
        "requires_operator_approval": status == "ready_for_isolated_evaluation",
    }


def _self_repair_evaluation_target(intent: str) -> str:
    if intent in {"review_column_revival", "review_stale_column_revival"}:
        return "reduce_silence_or_stale_routing_without_saturation"
    if intent == "review_inhibitory_balance":
        return "reduce_saturation_without_collapsing_activity"
    if intent == "review_decorrelation_or_prune":
        return "reduce_overcorrelation_without_losing_sparse_responsiveness"
    return "collect_stable_spike_health_window"


def _spike_readout_device(cuda_runtime: Mapping[str, Any], subcortex_devices: Mapping[str, Any]) -> str:
    direct_device = _text(cuda_runtime.get("tensor_device"))
    if direct_device:
        return direct_device
    for report in subcortex_devices.values():
        if not isinstance(report, Mapping):
            continue
        for key in (
            "prototype_device",
            "location_state_device",
            "binding_matrix_device",
            "encoder_device",
        ):
            candidate = _text(report.get(key))
            if candidate:
                return candidate
    return "unknown"


def _is_cuda_device(device: str) -> bool:
    return device.lower().startswith("cuda")


def _spike_readout_slots(
    *,
    concept_candidates: Sequence[Mapping[str, Any]],
    recent_concepts: Sequence[str],
    focus: str,
    pressure_band: str,
) -> list[dict[str, Any]]:
    labels: list[str] = []
    for candidate in concept_candidates:
        label = _text(candidate.get("label"))
        if label and label not in labels:
            labels.append(label)
    for concept in recent_concepts:
        if concept and concept not in labels:
            labels.append(concept)
    if not labels and focus:
        labels.append(focus)
    if not labels:
        labels.append("unfocused_subcortex_state")
    return [
        {
            "slot_id": f"spike_readout_{index}",
            "kind": "concept_pressure",
            "label": label,
            "pressure_band": pressure_band,
            "grounded": label != "unfocused_subcortex_state",
        }
        for index, label in enumerate(labels[:4])
    ]


def _self_repair_baseline_metrics(grounding: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "activity_state": grounding.get("activity_state"),
        "correlation_status": grounding.get("correlation_status"),
        "silent_fraction": grounding.get("silent_fraction"),
        "saturated_fraction": grounding.get("saturated_fraction"),
        "stale_fraction": grounding.get("stale_fraction"),
        "correlation_evidence_available": bool(grounding.get("correlation_evidence_available")),
    }
    correlation = grounding.get("correlation") if isinstance(grounding.get("correlation"), Mapping) else {}
    if correlation:
        metrics["correlation"] = dict(correlation)
    return metrics


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


def _self_repair_surface_promotion_gate(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_gates = [
        candidate.get("promotion_gate")
        for candidate in candidates
        if isinstance(candidate.get("promotion_gate"), Mapping)
    ]
    ready = [
        gate for gate in candidate_gates
        if _text(gate.get("status")) == "ready_for_replay_review"
    ]
    insufficient = [
        gate for gate in candidate_gates
        if _text(gate.get("status")) == "insufficient_evidence"
    ]
    if ready:
        status = "ready_for_replay_review"
        next_gate = "deep_sleep_or_replay_repair_gate"
    elif insufficient:
        status = "insufficient_evidence"
        next_gate = "collect_spike_window"
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
        "requires_operator_approval": status == "ready_for_replay_review",
        "candidate_gate_count": len(candidate_gates),
        "ready_candidate_count": len(ready),
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
