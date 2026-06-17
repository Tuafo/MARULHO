from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from marulho.semantics.spike_language_decoder import build_spike_language_decoder_probe
from marulho.semantics.spike_language_neurons import build_spike_language_neuron_adapter


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
            "Bounded control hypotheses from Cognitive Signal, not generated text output.",
            "Candidates are advisory until replay, policy, or operator evidence promotes them.",
        ],
    }


def build_subcortical_spike_readout_evidence(
    cognitive_signal: Mapping[str, Any],
    runtime_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build MARULHO-owned spike-language readout evidence without generating text."""

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
    device_evidence = _spike_readout_device_evidence(cuda_runtime, subcortex_devices)
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
        "device_evidence": {
            "device": device_evidence["device"],
            "cuda_report_available": bool(cuda_runtime),
            "subcortex_device_evidence_available": bool(subcortex_devices),
            "cuda_device_selected": bool(device_evidence["cuda_device_selected"]),
            "source": device_evidence["source"],
            "observed_device_count": device_evidence["observed_device_count"],
            "observed_devices": device_evidence["observed_devices"],
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
            "requires_marulho_owned_decoder": True,
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
            "The population code is deterministic telemetry evidence for a future MARULHO-owned spike decoder.",
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
    decoder_probe = build_spike_language_decoder_probe(spike_readout_evidence)
    language_neuron_adapter = build_spike_language_neuron_adapter(decoder_probe)
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
        decoder_probe=decoder_probe,
        language_neuron_adapter=language_neuron_adapter,
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
        "review_role": "operator_marulho_native_snn_language_review_only",
        "source": "service.status_read_model.cognitive_signal_and_runtime_scope",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "current_language_surface": {
            "surface": language_surface.get("surface"),
            "source": language_surface.get("source"),
            "grounded": bool(language_surface.get("grounded")),
            "not_cognition_substrate": bool(language_surface.get("not_cognition_substrate")),
            "concept_focus": (language_surface.get("grounding") or {}).get("concept_focus")
            if isinstance(language_surface.get("grounding"), Mapping)
            else None,
            "candidate_phrase_count": len(list(language_surface.get("candidate_phrases") or [])),
        },
        "current_deliberation_surface": {
            "surface": deliberation_surface.get("surface"),
            "grounded": bool(deliberation_surface.get("grounded")),
            "not_cognition_substrate": bool(deliberation_surface.get("not_cognition_substrate")),
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
        "current_decoder_probe_evidence": {
            "surface": decoder_probe.get("surface"),
            "owned_by_marulho": bool(decoder_probe.get("owned_by_marulho")),
            "external_dependency": bool(decoder_probe.get("external_dependency")),
            "generates_text": bool(decoder_probe.get("generates_text")),
            "executable": bool(decoder_probe.get("executable")),
            "mutates_runtime_state": bool(decoder_probe.get("mutates_runtime_state")),
            "decodes_text": bool(decoder_probe.get("decodes_text")),
            "tensor_device": (
                (decoder_probe.get("device_evidence") or {}).get("tensor_device")
                if isinstance(decoder_probe.get("device_evidence"), Mapping)
                else None
            ),
            "mean_sparsity": (
                (decoder_probe.get("sparsity_evidence") or {}).get("mean_sparsity")
                if isinstance(decoder_probe.get("sparsity_evidence"), Mapping)
                else None
            ),
            "readout_slot_count": (
                (decoder_probe.get("support_evidence") or {}).get("readout_slot_count")
                if isinstance(decoder_probe.get("support_evidence"), Mapping)
                else 0
            ),
            "grounded_slot_count": (
                (decoder_probe.get("support_evidence") or {}).get("grounded_slot_count")
                if isinstance(decoder_probe.get("support_evidence"), Mapping)
                else 0
            ),
            "supported": (
                (decoder_probe.get("support_evidence") or {}).get("supported")
                if isinstance(decoder_probe.get("support_evidence"), Mapping)
                else False
            ),
            "dynamic_state_available": (
                (decoder_probe.get("temporal_state_evidence") or {}).get("dynamic_state_available")
                if isinstance(decoder_probe.get("temporal_state_evidence"), Mapping)
                else False
            ),
            "active_transition_count": (
                (decoder_probe.get("temporal_state_evidence") or {}).get("active_transition_count")
                if isinstance(decoder_probe.get("temporal_state_evidence"), Mapping)
                else 0
            ),
        },
        "current_language_neuron_adapter_evidence": {
            "surface": language_neuron_adapter.get("surface"),
            "owned_by_marulho": bool(language_neuron_adapter.get("owned_by_marulho")),
            "external_dependency": bool(language_neuron_adapter.get("external_dependency")),
            "generates_text": bool(language_neuron_adapter.get("generates_text")),
            "executable": bool(language_neuron_adapter.get("executable")),
            "mutates_runtime_state": bool(language_neuron_adapter.get("mutates_runtime_state")),
            "decodes_text": bool(language_neuron_adapter.get("decodes_text")),
            "tensor_device": (
                (language_neuron_adapter.get("device_evidence") or {}).get("tensor_device")
                if isinstance(language_neuron_adapter.get("device_evidence"), Mapping)
                else None
            ),
            "active_spike_count": (
                (language_neuron_adapter.get("neuron_dynamics") or {}).get("active_spike_count")
                if isinstance(language_neuron_adapter.get("neuron_dynamics"), Mapping)
                else 0
            ),
            "adaptive_timesteps": (
                (language_neuron_adapter.get("neuron_dynamics") or {}).get("adaptive_timesteps")
                if isinstance(language_neuron_adapter.get("neuron_dynamics"), Mapping)
                else False
            ),
            "activation_sparsity": (
                (language_neuron_adapter.get("sparsity_evidence") or {}).get("activation_sparsity")
                if isinstance(language_neuron_adapter.get("sparsity_evidence"), Mapping)
                else None
            ),
        },
        "research_candidates": [
            {
                "name": "NeuronSpark",
                "kind": "pure_snn_language_implementation_reference",
                "reported_scale": "0.9B parameters",
                "integration_status": "reference_for_marulho_owned_reimplementation",
                "required_local_evidence": [
                    "marulho_owned_language_neuron_module",
                    "marulho_native_snn_decoder",
                    "marulho_controlled_training_loop",
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
                "integration_status": "reference_for_marulho_owned_reimplementation",
                "required_local_evidence": [
                    "marulho_owned_language_neuron_module",
                    "marulho_native_snn_decoder",
                    "marulho_controlled_training_loop",
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
            "marulho_spike_language_decoder_probe_report",
            "marulho_spike_language_neuron_adapter_report",
            "activation_sparsity_report",
            "grounding_support_report",
            "runtime_truth_delta",
            "replay_or_eval_dataset_report",
        ],
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "requires_operator_approval": True,
            "requires_grounding_support": True,
            "requires_sparsity_evidence": True,
            "requires_device_evidence": True,
            "requires_marulho_decoder_probe": True,
            "requires_marulho_owned_implementation": True,
            "requires_marulho_controlled_training": True,
        },
        "limitations": [
            "Readiness gate only; it does not load external SNN language checkpoints, add external dependencies, or generate text.",
            "NeuronSpark and Nord-AI are implementation references only; MARULHO must own the language neurons, decoder, training loop, grounding, telemetry, and promotion gates.",
            "Language remains a grounded Subcortex surface until a MARULHO-native SNN generator proves sparsity, device placement, grounding support, and operator-controlled training.",
        ],
    }


def build_snn_language_evaluation_surface(
    cognitive_signal: Mapping[str, Any],
    runtime_scope: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a read-only gate for isolated SNN language-adapter evaluation."""

    readiness = build_snn_language_readiness_surface(cognitive_signal, runtime_scope)
    checks = readiness.get("readiness_checks") if isinstance(readiness.get("readiness_checks"), Mapping) else {}
    adapter = (
        readiness.get("current_language_neuron_adapter_evidence")
        if isinstance(readiness.get("current_language_neuron_adapter_evidence"), Mapping)
        else {}
    )
    decoder = (
        readiness.get("current_decoder_probe_evidence")
        if isinstance(readiness.get("current_decoder_probe_evidence"), Mapping)
        else {}
    )
    required = {
        "grounded_language_surface_available": bool(checks.get("grounded_language_surface_available")),
        "marulho_spike_decoder_probe_grounding_supported": bool(
            checks.get("marulho_spike_decoder_probe_grounding_supported")
        ),
        "marulho_spike_decoder_probe_temporal_state": bool(checks.get("marulho_spike_decoder_probe_temporal_state")),
        "marulho_spike_language_neuron_adapter_available": bool(
            checks.get("marulho_spike_language_neuron_adapter_available")
        ),
        "marulho_spike_language_neuron_adapter_owned": bool(checks.get("marulho_spike_language_neuron_adapter_owned")),
        "marulho_spike_language_neuron_adapter_sparse": bool(checks.get("marulho_spike_language_neuron_adapter_sparse")),
        "marulho_spike_language_neuron_adapter_dynamic": bool(
            checks.get("marulho_spike_language_neuron_adapter_dynamic")
        ),
        "marulho_spike_language_neuron_adapter_device_evidence_available": bool(
            checks.get("marulho_spike_language_neuron_adapter_device_evidence_available")
        ),
        "subcortex_device_evidence_available": bool(checks.get("subcortex_device_evidence_available")),
    }
    ready = all(required.values())
    status = "ready_for_isolated_adapter_evaluation" if ready else "blocked_missing_adapter_evidence"
    next_gate = (
        "operator_approved_isolated_language_adapter_evaluation"
        if ready
        else "complete_grounded_snn_adapter_evidence"
    )
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_snn_language_adapter_evaluation_gate",
        "surface": "snn_language_adapter_evaluation.v1",
        "endpoint": "/terminus/snn-language-evaluation",
        "review_role": "operator_snn_language_adapter_evaluation_review_only",
        "source": "service.status_read_model.snn_language_readiness",
        "grounded": bool(readiness.get("grounded")),
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "promotion_gate": {
            "status": status,
            "next_gate": next_gate,
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_language_generation": False,
            "eligible_for_training": False,
            "requires_operator_approval": ready,
            "required_evidence": required,
        },
        "evaluation_cases": [
            {
                "case_id": "snn_language_adapter_sparse_grounded_eval",
                "target": "spike_language_neuron_adapter",
                "phase": "isolated_evaluation_review",
                "ready_for_evaluation": ready,
                "promotion_status": status,
                "baseline": {
                    "decoder_mean_sparsity": decoder.get("mean_sparsity"),
                    "adapter_activation_sparsity": adapter.get("activation_sparsity"),
                    "adapter_active_spike_count": int(_float(adapter.get("active_spike_count"), 0.0)),
                    "adapter_adaptive_timesteps": bool(adapter.get("adaptive_timesteps")),
                    "tensor_device": adapter.get("tensor_device"),
                },
                "evaluation_target": "prove_sparse_adapter_spikes_improve_grounded_readout_support_without_text_generation",
                "required_evidence": [
                    "pre_evaluation_readiness_snapshot",
                    "heldout_grounded_readout_slots",
                    "post_evaluation_grounding_delta",
                    "adapter_activation_sparsity_delta",
                    "runtime_truth_delta",
                    "rollback_policy",
                ],
                "promotion_constraints": {
                    "eligible_for_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "eligible_for_training": False,
                    "next_gate": next_gate,
                },
            }
        ],
        "success_evidence": [
            "heldout_grounded_readout_slots",
            "post_evaluation_grounding_delta",
            "adapter_activation_sparsity_delta",
            "runtime_truth_delta",
            "cuda_or_accelerator_device_report",
            "rollback_policy",
        ],
        "limitations": [
            "Evaluation gate only; it does not train, mutate runtime state, decode text, or generate language.",
            "A passing adapter evaluation can only justify a later operator-approved training loop, not immediate cognition-substrate promotion.",
        ],
    }


def build_snn_language_training_readiness_surface(
    heldout_evaluation: Mapping[str, Any],
    *,
    runtime_truth_delta: Mapping[str, Any] | None = None,
    rollback_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only gate for designing a local SNN language training loop."""

    heldout = dict(heldout_evaluation)
    heldout_summary = (
        heldout.get("heldout_summary")
        if isinstance(heldout.get("heldout_summary"), Mapping)
        else {}
    )
    adapter_delta = heldout.get("adapter_delta") if isinstance(heldout.get("adapter_delta"), Mapping) else {}
    gate = heldout.get("promotion_gate") if isinstance(heldout.get("promotion_gate"), Mapping) else {}
    truth_delta = dict(runtime_truth_delta or {})
    rollback = dict(rollback_policy or {})
    runtime_truth_ok = bool(truth_delta.get("improved_or_stable") or not truth_delta)
    rollback_available = bool(rollback.get("available") or rollback.get("reversible"))
    required = {
        "heldout_evaluation_available": bool(heldout.get("available")),
        "heldout_owned_by_marulho": bool(heldout.get("owned_by_marulho")),
        "heldout_external_dependency_absent": not bool(heldout.get("external_dependency")),
        "heldout_external_checkpoint_absent": not bool(heldout.get("loads_external_checkpoint")),
        "heldout_generation_absent": not bool(heldout.get("generates_text")),
        "heldout_training_absent": not bool(heldout.get("trains")),
        "heldout_mutation_absent": not bool(heldout.get("mutates_runtime_state")),
        "heldout_cases_supported": int(_float(heldout_summary.get("unsupported_case_count"), 1.0)) == 0
        and int(_float(heldout_summary.get("case_count"), 0.0)) > 0,
        "activation_sparsity_floor_met": _float(adapter_delta.get("min_activation_sparsity"), 0.0)
        >= _float(adapter_delta.get("target_min_activation_sparsity"), 0.85),
        "heldout_gate_ready": _text(gate.get("status")) == "ready_for_operator_review",
        "runtime_truth_improved_or_stable": runtime_truth_ok,
        "rollback_policy_available": rollback_available,
    }
    ready = all(required.values())
    status = "ready_for_training_loop_design_review" if ready else "blocked_missing_training_design_evidence"
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_snn_language_training_readiness_gate",
        "surface": "snn_language_training_readiness.v1",
        "endpoint": "/terminus/snn-language-training/readiness",
        "review_role": "operator_snn_language_training_design_review_only",
        "source": "semantics.snn_language_training_readiness",
        "grounded": bool(required["heldout_cases_supported"]),
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "not_cognition_substrate": True,
        "heldout_evaluation_surface": heldout.get("surface"),
        "heldout_summary": {
            "case_count": int(_float(heldout_summary.get("case_count"), 0.0)),
            "supported_case_count": int(_float(heldout_summary.get("supported_case_count"), 0.0)),
            "unsupported_case_count": int(_float(heldout_summary.get("unsupported_case_count"), 0.0)),
            "mean_grounded_fraction": _float(heldout_summary.get("mean_grounded_fraction"), 0.0),
        },
        "adapter_training_constraints": {
            "min_activation_sparsity": _float(adapter_delta.get("min_activation_sparsity"), 0.0),
            "target_min_activation_sparsity": _float(adapter_delta.get("target_min_activation_sparsity"), 0.85),
            "mean_active_spike_count": _float(adapter_delta.get("mean_active_spike_count"), 0.0),
            "requires_local_marulho_trainer": True,
            "requires_cuda_or_accelerator_device_report": True,
            "requires_grounded_sequence_objective": True,
            "requires_no_external_checkpoint": True,
        },
        "runtime_truth_delta": dict(truth_delta),
        "rollback_evidence": {
            "available": rollback_available,
            "snapshot_id": rollback.get("snapshot_id"),
            "ledger_id": rollback.get("ledger_id"),
        },
        "promotion_gate": {
            "status": status,
            "next_gate": (
                "operator_approved_local_snn_language_trainer"
                if ready
                else "collect_training_design_evidence"
            ),
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_language_generation": False,
            "eligible_for_training": False,
            "eligible_for_training_loop_design": ready,
            "requires_operator_approval": True,
            "required_evidence": required,
        },
        "training_design_cases": [
            {
                "case_id": "local_snn_language_sequence_trainer_design",
                "target": "marulho_owned_snn_language_trainer",
                "ready_for_design_review": ready,
                "objective": "learn_grounded_sequence_transitions_from_spike_readout_slots_without_external_llm_checkpoint",
                "required_evidence": [
                    "heldout_language_adapter_evaluation",
                    "grounded_sequence_readout_dataset",
                    "local_plasticity_or_surrogate_training_rule",
                    "cuda_or_accelerator_device_report",
                    "runtime_truth_delta",
                    "rollback_policy",
                ],
                "promotion_constraints": {
                    "eligible_for_training": False,
                    "eligible_for_language_generation": False,
                    "eligible_for_cognition_substrate": False,
                    "next_gate": "operator_approved_local_snn_language_trainer" if ready else "collect_training_design_evidence",
                },
            }
        ],
        "success_evidence": [
            "heldout_language_adapter_evaluation",
            "grounded_sequence_readout_dataset",
            "training_rule_design",
            "cuda_or_accelerator_device_report",
            "runtime_truth_delta",
            "rollback_policy",
        ],
        "limitations": [
            "Training readiness only; it does not update weights, decode text, generate text, or promote a cognition substrate.",
            "Readiness can only authorize design review for a MARULHO-owned local trainer, not use of external SNN language checkpoints.",
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
    promotion_gate = _structural_plasticity_promotion_gate(
        structural_cases,
        binding_report,
        local_plasticity_report,
    )
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


def build_binding_growth_trial_design(
    column_runtime: Mapping[str, Any],
    binding_plan: Mapping[str, Any],
    *,
    state_revision: int,
) -> dict[str, Any]:
    """Bind repeated prediction failures to a read-only sparse topology trial."""
    runtime = dict(column_runtime or {})
    growth = runtime.get("growth_gate") if isinstance(runtime.get("growth_gate"), Mapping) else {}
    plan = dict(binding_plan or {})
    growth_candidates = [
        int(value)
        for value in list(growth.get("candidate_column_ids_sample") or [])
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    plan_candidates = [
        int(value)
        for value in list(plan.get("candidate_column_ids") or [])
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    proposed_edges = [
        [int(edge[0]), int(edge[1])]
        for edge in list(plan.get("proposed_edges") or [])
        if isinstance(edge, (list, tuple))
        and len(edge) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in edge)
    ]
    max_total_edge_delta = int(_float(plan.get("max_total_edge_delta"), 0.0))
    proposed_total_edge_delta = int(_float(plan.get("proposed_total_edge_delta"), 0.0))
    plan_material = {
        "n_columns": int(_float(runtime.get("total_columns"), 0.0)),
        "candidate_column_ids": plan_candidates,
        "max_total_edge_delta": max_total_edge_delta,
        "baseline_topology_hash": plan.get("baseline_topology_hash"),
        "proposed_edges": proposed_edges,
    }
    recomputed_plan_hash = _sha256_json(plan_material)
    supplied_plan_hash = _text(plan.get("plan_hash"))
    required_evidence = {
        "column_runtime_surface_available": runtime.get("surface") == "column_runtime_metabolism.v1",
        "repeated_failure_growth_gate_ready": bool(growth.get("ready")),
        "repeated_failure_evidence_available": bool(growth.get("repeated_surprise_available")),
        "binding_plan_surface_available": plan.get("surface")
        == "binding_candidate_hub_topology_plan.v1",
        "candidate_columns_bound": bool(plan_candidates) and plan_candidates == growth_candidates[: len(plan_candidates)],
        "baseline_topology_hash_available": len(_text(plan.get("baseline_topology_hash"))) == 64,
        "plan_hash_available": len(supplied_plan_hash) == 64,
        "plan_hash_recomputed_match": supplied_plan_hash == recomputed_plan_hash,
        "nonzero_edge_delta": proposed_total_edge_delta > 0,
        "edge_delta_matches_plan": proposed_total_edge_delta == len(proposed_edges),
        "edge_delta_within_budget": proposed_total_edge_delta <= max_total_edge_delta,
        "source_tensor_device_observed": bool(_text(plan.get("source_tensor_device"))),
        "runtime_mutation_absent": not bool(plan.get("mutates_runtime_state")),
        "checkpoint_write_absent": not bool(plan.get("writes_checkpoint")),
        "topology_refresh_absent": not bool(plan.get("calls_topology_refresh")),
    }
    ready = all(bool(value) for value in required_evidence.values())
    design_material = {
        "state_revision": int(state_revision),
        "growth_gate": {
            "candidate_column_ids": plan_candidates,
            "streak_threshold": int(_float(growth.get("streak_threshold"), 0.0)),
            "repeated_surprise_count": int(_float(growth.get("repeated_surprise_count"), 0.0)),
        },
        "binding_plan_hash": supplied_plan_hash,
        "baseline_topology_hash": plan.get("baseline_topology_hash"),
        "max_total_edge_delta": max_total_edge_delta,
        "proposed_edges": proposed_edges,
    }
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_binding_growth_trial_design",
        "surface": "binding_growth_trial_design.v1",
        "available": True,
        "source": "core.column_runtime_and_hypercube_binding_plan",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "runs_isolated_trial": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": False,
        "calls_topology_refresh": False,
        "state_revision": int(state_revision),
        "binding_growth_trial_design_hash": _sha256_json(design_material),
        "growth_evidence": {
            "candidate_column_ids": plan_candidates,
            "candidate_column_count": len(plan_candidates),
            "streak_threshold": int(_float(growth.get("streak_threshold"), 0.0)),
            "repeated_surprise_count": int(_float(growth.get("repeated_surprise_count"), 0.0)),
            "evidence": growth.get("evidence"),
        },
        "topology_trial": {
            "binding_plan_hash": supplied_plan_hash or None,
            "baseline_topology_hash": plan.get("baseline_topology_hash"),
            "max_total_edge_delta": max_total_edge_delta,
            "proposed_total_edge_delta": proposed_total_edge_delta,
            "proposed_edges": proposed_edges,
            "per_source": [
                dict(item)
                for item in list(plan.get("per_source") or [])
                if isinstance(item, Mapping)
            ],
            "source_tensor_device": plan.get("source_tensor_device"),
            "plan_compute_device": plan.get("plan_compute_device"),
            "device_transfer_count": int(_float(plan.get("device_transfer_count"), 0.0)),
            "snapshot_bytes": int(_float(plan.get("snapshot_bytes"), 0.0)),
            "hot_path_effect": plan.get("hot_path_effect"),
        },
        "promotion_gate": {
            "status": "ready_for_isolated_binding_growth_trial"
            if ready
            else "blocked_missing_binding_growth_trial_evidence",
            "next_gate": "checkpoint_clone_binding_growth_trial_runner"
            if ready
            else "collect_repeated_failure_and_bounded_topology_plan",
            "eligible_for_isolated_trial": ready,
            "eligible_for_structural_mutation": False,
            "eligible_for_action": False,
            "required_evidence": required_evidence,
        },
        "limitations": [
            "Design only; it does not clone a checkpoint, run cognition, refresh topology, or apply edges.",
            "Prediction failure identifies candidate sources but does not prove the proposed edges improve prediction, Runtime Truth, or spike health.",
        ],
    }


def evaluate_subcortical_structural_plasticity_isolated(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    *,
    rollback_policy: Mapping[str, Any] | None = None,
    candidate_evidence: Mapping[str, Any] | None = None,
    cost_evidence: Mapping[str, Any] | None = None,
    runtime_truth_summary: Mapping[str, Any] | None = None,
    no_mutation_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a bounded structural-plasticity trial without mutating runtime state."""

    structural_delta = _isolated_structural_delta(pre_snapshot, post_snapshot)
    device_evidence = _isolated_structural_device_evidence(pre_snapshot, post_snapshot)
    spike_health_delta = _isolated_spike_health_delta(pre_snapshot, post_snapshot)
    runtime_truth_delta = _isolated_runtime_truth_delta(pre_snapshot, post_snapshot)
    snapshot_binding = _isolated_structural_snapshot_binding(pre_snapshot, post_snapshot)
    rollback_evidence = _isolated_rollback_evidence(
        rollback_policy or {},
        pre_snapshot_hash=str(snapshot_binding["pre_snapshot_hash"]),
    )
    candidate_binding = _isolated_candidate_evidence(
        candidate_evidence or {},
        pre_snapshot_hash=str(snapshot_binding["pre_snapshot_hash"]),
    )
    cost_impact = _isolated_cost_usefulness_impact(cost_evidence or {}, candidate_binding)
    truth_summary = _isolated_runtime_truth_summary(
        pre_snapshot,
        post_snapshot,
        runtime_truth_summary or {},
    )
    no_mutation_proof = _isolated_no_mutation_proof(no_mutation_evidence or {})

    bounded_delta = (
        structural_delta["edges_added_delta"] >= 0
        and structural_delta["edges_removed_delta"] >= 0
        and structural_delta["growth_events_delta"] >= 0
        and structural_delta["prune_events_delta"] >= 0
        and structural_delta["total_edge_delta"] <= structural_delta["bounded_edge_delta_limit"]
    )
    ready_for_review = bool(
        bounded_delta
        and snapshot_binding["structural_delta_present"]
        and snapshot_binding["snapshot_hashes_distinct"]
        and device_evidence["consistent"]
        and spike_health_delta["improved_or_stable"]
        and runtime_truth_delta["improved_or_stable"]
        and rollback_evidence["available"]
        and rollback_evidence["bound_to_pre_snapshot"]
    )
    promotion_status = "ready_for_operator_review" if ready_for_review else "blocked_evidence_incomplete"
    checkpointed_candidate_gate = _checkpointed_candidate_gate(
        ready_for_review=ready_for_review,
        candidate_binding=candidate_binding,
        cost_impact=cost_impact,
        truth_summary=truth_summary,
        rollback_evidence=rollback_evidence,
        no_mutation_proof=no_mutation_proof,
    )
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_structural_plasticity_isolated_evaluation",
        "surface": "subcortical_structural_plasticity_isolated_evaluation.v1",
        "available": True,
        "review_role": "operator_structural_plasticity_evaluation_review_only",
        "source": "semantics.subcortical_structural_plasticity_isolated_evaluator",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "structural_delta": structural_delta,
        "device_evidence": device_evidence,
        "spike_health_delta": spike_health_delta,
        "runtime_truth_delta": runtime_truth_delta,
        "runtime_truth_summary": truth_summary,
        "rollback_evidence": rollback_evidence,
        "snapshot_binding": snapshot_binding,
        "checkpointed_candidate_evidence": candidate_binding,
        "cost_usefulness_impact": cost_impact,
        "no_mutation_proof": no_mutation_proof,
        "checkpointed_candidate_gate": checkpointed_candidate_gate,
        "promotion_gate": {
            "status": promotion_status,
            "next_gate": (
                "operator_approved_structural_mutation_design"
                if ready_for_review
                else "collect_isolated_structural_evaluation_evidence"
            ),
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "requires_operator_approval": True,
            "requires_reversible_mutation_ledger": True,
            "requires_runtime_truth_improvement": True,
            "requires_device_evidence": True,
            "requires_bound_snapshot_hashes": True,
            "requires_nonzero_structural_delta": True,
            "requires_rollback_pre_snapshot_binding": True,
            "checkpointed_candidate_gate_status": checkpointed_candidate_gate["status"],
            "requires_checkpointed_candidate_evidence": True,
        },
        "success_evidence": [
            "pre_mutation_structural_snapshot",
            "post_mutation_structural_snapshot",
            "pre_post_snapshot_hash_binding",
            "candidate_reason_and_baseline_hash",
            "cost_usefulness_latency_ram_vram_impact",
            "runtime_truth_delta",
            "runtime_truth_summary",
            "spike_health_delta",
            "device_evidence_report",
            "rollback_policy",
            "rollback_artifact",
            "no_mutation_proof",
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
            "requires_bound_snapshot_hashes": True,
            "requires_nonzero_structural_delta": True,
            "requires_rollback_pre_snapshot_binding": True,
            "requires_checkpointed_candidate_evidence": True,
            "requires_no_mutation_proof": True,
        },
        "limitations": [
            "Evaluation only; it compares snapshots and never calls growth, pruning, or binding mutation code.",
            "Ready-for-review status does not authorize runtime structural mutation.",
            "Checkpointed candidate readiness requires ticket, baseline hash, cost/usefulness impact, Runtime Truth summary, rollback artifact, and no-mutation proof.",
        ],
    }


def build_subcortical_structural_mutation_design(
    isolated_evaluation: Mapping[str, Any],
    *,
    operator_id: str | None = None,
    confirmation: bool = False,
    mutation_reason: str | None = None,
    max_total_edge_delta: int = 16,
) -> dict[str, Any]:
    """Build a read-only structural mutation design from bound isolated evidence."""

    evaluation = dict(isolated_evaluation or {})
    gate = evaluation.get("promotion_gate") if isinstance(evaluation.get("promotion_gate"), Mapping) else {}
    structural_delta = (
        evaluation.get("structural_delta")
        if isinstance(evaluation.get("structural_delta"), Mapping)
        else {}
    )
    snapshot_binding = (
        evaluation.get("snapshot_binding")
        if isinstance(evaluation.get("snapshot_binding"), Mapping)
        else {}
    )
    rollback = (
        evaluation.get("rollback_evidence")
        if isinstance(evaluation.get("rollback_evidence"), Mapping)
        else {}
    )
    device = (
        evaluation.get("device_evidence")
        if isinstance(evaluation.get("device_evidence"), Mapping)
        else {}
    )
    spike_delta = (
        evaluation.get("spike_health_delta")
        if isinstance(evaluation.get("spike_health_delta"), Mapping)
        else {}
    )
    truth_delta = (
        evaluation.get("runtime_truth_delta")
        if isinstance(evaluation.get("runtime_truth_delta"), Mapping)
        else {}
    )
    candidate_gate = (
        evaluation.get("checkpointed_candidate_gate")
        if isinstance(evaluation.get("checkpointed_candidate_gate"), Mapping)
        else {}
    )
    candidate_binding = (
        evaluation.get("checkpointed_candidate_evidence")
        if isinstance(evaluation.get("checkpointed_candidate_evidence"), Mapping)
        else {}
    )
    cost_impact = (
        evaluation.get("cost_usefulness_impact")
        if isinstance(evaluation.get("cost_usefulness_impact"), Mapping)
        else {}
    )

    bounded_edge_delta = int(_float(structural_delta.get("total_edge_delta"), 0.0)) <= int(max_total_edge_delta)
    required_evidence = {
        "isolated_evaluation_surface_available": evaluation.get("surface")
        == "subcortical_structural_plasticity_isolated_evaluation.v1",
        "isolated_evaluation_ready": gate.get("status") == "ready_for_operator_review",
        "snapshot_hashes_bound": bool(snapshot_binding.get("snapshot_hashes_distinct")),
        "structural_delta_present": bool(snapshot_binding.get("structural_delta_present")),
        "rollback_bound_to_pre_snapshot": bool(rollback.get("bound_to_pre_snapshot")),
        "device_evidence_consistent": bool(device.get("consistent")),
        "spike_health_improved_or_stable": bool(spike_delta.get("improved_or_stable")),
        "runtime_truth_improved_or_stable": bool(truth_delta.get("improved_or_stable")),
        "bounded_total_edge_delta": bounded_edge_delta,
        "operator_confirmation": bool(confirmation),
        "operator_id_available": bool(_text(operator_id)),
        "mutation_reason_available": bool(_text(mutation_reason)),
        "checkpointed_candidate_gate_ready": (
            not bool(candidate_gate)
            or candidate_gate.get("status") == "ready_for_checkpointed_candidate_review"
        ),
    }
    ready = all(bool(value) for value in required_evidence.values())
    design_material = {
        "surface": evaluation.get("surface"),
        "pre_snapshot_hash": snapshot_binding.get("pre_snapshot_hash"),
        "post_snapshot_hash": snapshot_binding.get("post_snapshot_hash"),
        "pre_state_revision": snapshot_binding.get("pre_state_revision"),
        "post_state_revision": snapshot_binding.get("post_state_revision"),
        "structural_delta": dict(structural_delta),
        "rollback_snapshot_id": rollback.get("snapshot_id"),
        "rollback_pre_snapshot_hash": rollback.get("pre_snapshot_hash"),
        "max_total_edge_delta": int(max_total_edge_delta),
        "operator_id": _text(operator_id),
        "confirmation": bool(confirmation),
        "mutation_target": "marulho.subcortex.binding.hub_topology",
        "mutation_reason": _text(mutation_reason),
        "checkpointed_candidate_gate_status": candidate_gate.get("status"),
        "candidate_evidence_hash": candidate_binding.get("candidate_evidence_hash"),
        "candidate_baseline_hash": candidate_binding.get("baseline_hash"),
        "candidate_reason": candidate_binding.get("candidate_reason"),
        "cost_impact_summary_hash": cost_impact.get("impact_summary_hash"),
    }
    design_hash = _sha256_json(design_material)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_structural_mutation_design",
        "surface": "subcortical_structural_mutation_design.v1",
        "available": True,
        "review_role": "operator_structural_mutation_design_review_only",
        "source": "semantics.subcortical_structural_mutation_design",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": False,
        "calls_growth_or_prune": False,
        "applies_structural_mutation": False,
        "structural_mutation_design_hash": design_hash,
        "hash_algorithm": "sha256_canonical_json",
        "evaluation_binding": {
            "evaluation_surface": evaluation.get("surface"),
            "evaluation_artifact_kind": evaluation.get("artifact_kind"),
            "pre_snapshot_hash": snapshot_binding.get("pre_snapshot_hash"),
            "post_snapshot_hash": snapshot_binding.get("post_snapshot_hash"),
            "pre_state_revision": snapshot_binding.get("pre_state_revision"),
            "post_state_revision": snapshot_binding.get("post_state_revision"),
            "rollback_snapshot_id": rollback.get("snapshot_id"),
            "rollback_pre_snapshot_hash": rollback.get("pre_snapshot_hash"),
            "rollback_bound_to_pre_snapshot": bool(rollback.get("bound_to_pre_snapshot")),
        },
        "checkpointed_candidate_binding": {
            "gate_status": candidate_gate.get("status"),
            "ticket_id": candidate_binding.get("ticket_id"),
            "kind": candidate_binding.get("kind"),
            "column_id": candidate_binding.get("column_id"),
            "candidate_reason": candidate_binding.get("candidate_reason"),
            "candidate_evidence_hash": candidate_binding.get("candidate_evidence_hash"),
            "candidate_evidence_hash_recomputed_match": bool(
                candidate_binding.get("candidate_evidence_hash_recomputed_match")
            ),
            "baseline_hash": candidate_binding.get("baseline_hash"),
            "baseline_hash_available": bool(candidate_binding.get("baseline_hash_available")),
            "candidate_reason_proves_growth_or_prune_pressure": bool(
                candidate_binding.get("candidate_reason_proves_growth_or_prune_pressure")
            ),
            "cost_impact_summary_hash": cost_impact.get("impact_summary_hash"),
            "latency_ram_vram_impact_available": bool(
                cost_impact.get("latency_ram_vram_impact_available")
            ),
        },
        "application_target": {
            "target_id": "marulho.subcortex.binding.hub_topology",
            "mutation_method": "HypercubeBindingLayer.refresh_hub_topology",
            "mutation_reason": _text(mutation_reason) or None,
        },
        "structural_mutation_design": {
            "edges_added_delta": int(_float(structural_delta.get("edges_added_delta"), 0.0)),
            "edges_removed_delta": int(_float(structural_delta.get("edges_removed_delta"), 0.0)),
            "growth_events_delta": int(_float(structural_delta.get("growth_events_delta"), 0.0)),
            "prune_events_delta": int(_float(structural_delta.get("prune_events_delta"), 0.0)),
            "total_edge_delta": int(_float(structural_delta.get("total_edge_delta"), 0.0)),
            "max_total_edge_delta": int(max_total_edge_delta),
            "bounded": bool(structural_delta.get("bounded")) and bounded_edge_delta,
            "runtime_update_applied": False,
            "checkpoint_written": False,
            "growth_or_prune_called": False,
        },
        "operator_review": {
            "operator_id": _text(operator_id) or None,
            "confirmation": bool(confirmation),
            "mutation_reason": _text(mutation_reason) or None,
        },
        "promotion_gate": {
            "status": "ready_for_structural_mutation_preflight_review"
            if ready
            else "blocked_missing_structural_mutation_design_evidence",
            "next_gate": "subcortical_structural_mutation_preflight.v1"
            if ready
            else "collect_bound_structural_mutation_design_evidence",
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "eligible_for_structural_mutation_preflight_review": ready,
            "requires_operator_approval": True,
            "requires_checkpoint_transaction": True,
            "requires_rollback_pre_snapshot_binding": True,
            "requires_checkpointed_candidate_evidence": bool(candidate_gate),
            "required_evidence": required_evidence,
        },
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "requires_operator_approval": True,
            "requires_checkpoint_transaction": True,
            "requires_preflight": True,
            "requires_rollback_pre_snapshot_binding": True,
            "requires_checkpointed_candidate_evidence": bool(candidate_gate),
        },
        "limitations": [
            "Design only; it does not call growth, pruning, binding mutation, checkpoint save, or checkpoint restore.",
            "Ready design can only feed a separate checkpoint-backed preflight review.",
        ],
    }


def build_subcortical_structural_mutation_preflight(
    structural_mutation_design: Mapping[str, Any],
    *,
    expected_state_revision: int,
    current_state_revision: int,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Review structural mutation design before a future checkpoint-backed write."""

    design = dict(structural_mutation_design or {})
    binding = (
        design.get("evaluation_binding")
        if isinstance(design.get("evaluation_binding"), Mapping)
        else {}
    )
    mutation = (
        design.get("structural_mutation_design")
        if isinstance(design.get("structural_mutation_design"), Mapping)
        else {}
    )
    operator = (
        design.get("operator_review")
        if isinstance(design.get("operator_review"), Mapping)
        else {}
    )
    gate = design.get("promotion_gate") if isinstance(design.get("promotion_gate"), Mapping) else {}
    target = (
        design.get("application_target")
        if isinstance(design.get("application_target"), Mapping)
        else {}
    )
    candidate = (
        design.get("checkpointed_candidate_binding")
        if isinstance(design.get("checkpointed_candidate_binding"), Mapping)
        else {}
    )
    design_material = {
        "surface": binding.get("evaluation_surface"),
        "pre_snapshot_hash": binding.get("pre_snapshot_hash"),
        "post_snapshot_hash": binding.get("post_snapshot_hash"),
        "pre_state_revision": binding.get("pre_state_revision"),
        "post_state_revision": binding.get("post_state_revision"),
        "structural_delta": {
            "edges_added_delta": mutation.get("edges_added_delta"),
            "edges_removed_delta": mutation.get("edges_removed_delta"),
            "growth_events_delta": mutation.get("growth_events_delta"),
            "prune_events_delta": mutation.get("prune_events_delta"),
            "total_edge_delta": mutation.get("total_edge_delta"),
            "bounded_edge_delta_limit": mutation.get("max_total_edge_delta"),
            "bounded": mutation.get("bounded"),
        },
        "rollback_snapshot_id": binding.get("rollback_snapshot_id"),
        "rollback_pre_snapshot_hash": binding.get("rollback_pre_snapshot_hash"),
        "max_total_edge_delta": mutation.get("max_total_edge_delta"),
        "operator_id": _text(operator.get("operator_id")),
        "confirmation": bool(operator.get("confirmation")),
        "mutation_target": target.get("target_id"),
        "mutation_reason": _text(operator.get("mutation_reason")),
        "checkpointed_candidate_gate_status": candidate.get("gate_status"),
        "candidate_evidence_hash": candidate.get("candidate_evidence_hash"),
        "candidate_baseline_hash": candidate.get("baseline_hash"),
        "candidate_reason": candidate.get("candidate_reason"),
        "cost_impact_summary_hash": candidate.get("cost_impact_summary_hash"),
    }
    recomputed_design_hash = _sha256_json(design_material)
    supplied_design_hash = str(design.get("structural_mutation_design_hash") or "")
    checkpoint = _text(checkpoint_path)
    required_evidence = {
        "design_surface_available": design.get("surface") == "subcortical_structural_mutation_design.v1",
        "design_artifact_kind_available": design.get("artifact_kind")
        == "terminus_subcortical_structural_mutation_design",
        "design_hash_available": len(supplied_design_hash) == 64,
        "design_hash_recomputed_match": recomputed_design_hash == supplied_design_hash,
        "design_gate_ready": gate.get("status") == "ready_for_structural_mutation_preflight_review",
        "design_blocks_direct_mutation": not bool(gate.get("eligible_for_structural_mutation"))
        and not bool(design.get("applies_structural_mutation")),
        "rollback_bound_to_pre_snapshot": bool(binding.get("rollback_bound_to_pre_snapshot")),
        "checkpoint_path_available": bool(checkpoint),
        "expected_state_revision_current": int(expected_state_revision) == int(current_state_revision),
        "design_bounded": bool(mutation.get("bounded")),
        "design_did_not_write_checkpoint": not bool(design.get("writes_checkpoint"))
        and not bool(mutation.get("checkpoint_written")),
        "design_did_not_mutate_runtime": not bool(design.get("mutates_runtime_state"))
        and not bool(mutation.get("runtime_update_applied")),
        "binding_hub_topology_target_bound": target.get("target_id")
        == "marulho.subcortex.binding.hub_topology",
        "binding_hub_refresh_method_bound": target.get("mutation_method")
        == "HypercubeBindingLayer.refresh_hub_topology",
        "mutation_reason_available": bool(_text(operator.get("mutation_reason"))),
        "checkpointed_candidate_gate_ready": (
            not bool(candidate)
            or candidate.get("gate_status") == "ready_for_checkpointed_candidate_review"
        ),
        "checkpointed_candidate_hash_bound": (
            not bool(candidate)
            or len(_text(candidate.get("candidate_evidence_hash"))) == 64
        ),
        "checkpointed_candidate_baseline_hash_bound": (
            not bool(candidate)
            or len(_text(candidate.get("baseline_hash"))) == 64
        ),
        "checkpointed_candidate_cost_impact_bound": (
            not bool(candidate)
            or bool(candidate.get("latency_ram_vram_impact_available"))
        ),
    }
    ready = all(bool(value) for value in required_evidence.values())
    preflight_material = {
        "structural_mutation_design_hash": supplied_design_hash,
        "mutation_target": target.get("target_id"),
        "mutation_method": target.get("mutation_method"),
        "mutation_reason": _text(operator.get("mutation_reason")),
        "max_total_edge_delta": int(_float(mutation.get("max_total_edge_delta"), 0.0)),
        "checkpointed_candidate_gate_status": candidate.get("gate_status"),
        "candidate_evidence_hash": candidate.get("candidate_evidence_hash"),
        "candidate_baseline_hash": candidate.get("baseline_hash"),
        "candidate_reason": candidate.get("candidate_reason"),
        "cost_impact_summary_hash": candidate.get("cost_impact_summary_hash"),
        "expected_state_revision": int(expected_state_revision),
        "current_state_revision": int(current_state_revision),
        "checkpoint_path": checkpoint or None,
        "required_evidence": required_evidence,
    }
    preflight_hash = _sha256_json(preflight_material)
    return {
        "schema_version": 1,
        "artifact_kind": "terminus_subcortical_structural_mutation_preflight",
        "surface": "subcortical_structural_mutation_preflight.v1",
        "available": True,
        "review_role": "operator_structural_mutation_preflight_review_only",
        "source": "semantics.subcortical_structural_mutation_preflight",
        "grounded": True,
        "advisory": True,
        "executable": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": False,
        "calls_growth_or_prune": False,
        "applies_structural_mutation": False,
        "structural_mutation_preflight_hash": preflight_hash,
        "hash_algorithm": "sha256_canonical_json",
        "design_binding": {
            "structural_mutation_design_hash": supplied_design_hash,
            "design_hash_available": len(supplied_design_hash) == 64,
            "recomputed_design_hash": recomputed_design_hash,
            "design_hash_recomputed_match": recomputed_design_hash == supplied_design_hash,
            "pre_snapshot_hash": binding.get("pre_snapshot_hash"),
            "post_snapshot_hash": binding.get("post_snapshot_hash"),
            "rollback_snapshot_id": binding.get("rollback_snapshot_id"),
            "rollback_pre_snapshot_hash": binding.get("rollback_pre_snapshot_hash"),
            "mutation_target": target.get("target_id"),
            "mutation_method": target.get("mutation_method"),
            "mutation_reason": _text(operator.get("mutation_reason")) or None,
            "max_total_edge_delta": int(_float(mutation.get("max_total_edge_delta"), 0.0)),
            "checkpointed_candidate_gate_status": candidate.get("gate_status"),
            "candidate_evidence_hash": candidate.get("candidate_evidence_hash"),
            "candidate_baseline_hash": candidate.get("baseline_hash"),
            "candidate_reason": candidate.get("candidate_reason"),
            "cost_impact_summary_hash": candidate.get("cost_impact_summary_hash"),
        },
        "checkpoint_transaction_requirements": {
            "expected_state_revision": int(expected_state_revision),
            "current_state_revision": int(current_state_revision),
            "expected_state_revision_current": int(expected_state_revision) == int(current_state_revision),
            "checkpoint_path": checkpoint or None,
            "checkpoint_path_available": bool(checkpoint),
            "pre_mutation_checkpoint_required": True,
            "restore_verification_required": True,
            "commit_transaction_required": True,
        },
        "promotion_gate": {
            "status": "ready_for_operator_structural_mutation_execution_review"
            if ready
            else "blocked_missing_structural_mutation_preflight_evidence",
            "next_gate": "operator_confirmed_checkpoint_backed_structural_mutation_executor"
            if ready
            else "collect_structural_mutation_preflight_evidence",
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "eligible_for_operator_execution_review": ready,
            "requires_operator_approval": True,
            "requires_checkpoint_transaction": True,
            "requires_restore_verification": True,
            "required_evidence": required_evidence,
        },
        "safety_invariants": {
            "eligible_for_action": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_structural_mutation": False,
            "requires_operator_approval": True,
            "requires_checkpoint_transaction": True,
            "requires_restore_verification": True,
        },
        "limitations": [
            "Preflight only; it verifies design and checkpoint requirements without saving checkpoints or mutating topology.",
            "Execution remains blocked until a separate operator-confirmed checkpoint-backed executor exists.",
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
    decoder_probe: Mapping[str, Any],
    language_neuron_adapter: Mapping[str, Any],
    cuda_runtime: Mapping[str, Any],
    subcortex_devices: Mapping[str, Any],
) -> dict[str, bool]:
    readout_device = (
        spike_readout_evidence.get("device_evidence")
        if isinstance(spike_readout_evidence.get("device_evidence"), Mapping)
        else {}
    )
    decoder_sparsity = (
        decoder_probe.get("sparsity_evidence")
        if isinstance(decoder_probe.get("sparsity_evidence"), Mapping)
        else {}
    )
    decoder_device = (
        decoder_probe.get("device_evidence") if isinstance(decoder_probe.get("device_evidence"), Mapping) else {}
    )
    decoder_support = (
        decoder_probe.get("support_evidence")
        if isinstance(decoder_probe.get("support_evidence"), Mapping)
        else {}
    )
    decoder_temporal = (
        decoder_probe.get("temporal_state_evidence")
        if isinstance(decoder_probe.get("temporal_state_evidence"), Mapping)
        else {}
    )
    adapter_device = (
        language_neuron_adapter.get("device_evidence")
        if isinstance(language_neuron_adapter.get("device_evidence"), Mapping)
        else {}
    )
    adapter_dynamics = (
        language_neuron_adapter.get("neuron_dynamics")
        if isinstance(language_neuron_adapter.get("neuron_dynamics"), Mapping)
        else {}
    )
    adapter_sparsity = (
        language_neuron_adapter.get("sparsity_evidence")
        if isinstance(language_neuron_adapter.get("sparsity_evidence"), Mapping)
        else {}
    )
    return {
        "grounded_language_surface_available": bool(language_surface.get("grounded")),
        "language_surface_not_cognition_substrate": bool(language_surface.get("not_cognition_substrate")),
        "deliberation_surface_available": bool(deliberation_surface.get("grounded")),
        "deliberation_grounded_subcortex_control": bool(deliberation_surface.get("grounded")),
        "marulho_spike_readout_evidence_available": bool(spike_readout_evidence.get("available")),
        "marulho_spike_readout_grounded": bool(spike_readout_evidence.get("grounded")),
        "marulho_spike_readout_non_generative": not bool(spike_readout_evidence.get("generates_text")),
        "marulho_spike_readout_device_evidence_available": bool(
            readout_device.get("cuda_report_available")
            or readout_device.get("subcortex_device_evidence_available")
        ),
        "marulho_spike_decoder_probe_available": bool(decoder_probe.get("available")),
        "marulho_spike_decoder_probe_owned": bool(decoder_probe.get("owned_by_marulho")),
        "marulho_spike_decoder_probe_non_generative": not bool(decoder_probe.get("generates_text")),
        "marulho_spike_decoder_probe_sparse": bool(decoder_sparsity.get("meets_sparse_readout_floor")),
        "marulho_spike_decoder_probe_device_evidence_available": bool(decoder_device.get("tensor_device")),
        "marulho_spike_decoder_probe_grounding_supported": bool(decoder_support.get("supported")),
        "marulho_spike_decoder_probe_temporal_state": bool(decoder_temporal.get("dynamic_state_available")),
        "marulho_spike_language_neuron_adapter_available": bool(language_neuron_adapter.get("available")),
        "marulho_spike_language_neuron_adapter_owned": bool(language_neuron_adapter.get("owned_by_marulho")),
        "marulho_spike_language_neuron_adapter_non_generative": not bool(
            language_neuron_adapter.get("generates_text")
        ),
        "marulho_spike_language_neuron_adapter_device_evidence_available": bool(adapter_device.get("tensor_device")),
        "marulho_spike_language_neuron_adapter_sparse": bool(
            adapter_sparsity.get("meets_sparse_activation_floor")
        ),
        "marulho_spike_language_neuron_adapter_dynamic": bool(adapter_dynamics.get("active_spike_count")),
        "cuda_runtime_scope_available": bool(cuda_runtime),
        "subcortex_device_evidence_available": bool(subcortex_devices),
        "local_snn_language_generator_available": bool(cuda_runtime.get("snn_language_generator_device_report")),
        "activation_sparsity_report_available": bool(cuda_runtime.get("snn_language_activation_sparsity")),
        "grounding_support_report_available": bool(cuda_runtime.get("snn_language_grounding_support")),
    }


def _snn_language_promotion_gate(readiness_checks: Mapping[str, bool]) -> dict[str, Any]:
    required_now = {
        "grounded_language_surface_available": bool(readiness_checks.get("grounded_language_surface_available")),
        "language_surface_not_cognition_substrate": bool(readiness_checks.get("language_surface_not_cognition_substrate")),
        "deliberation_grounded_subcortex_control": bool(readiness_checks.get("deliberation_grounded_subcortex_control")),
        "marulho_spike_readout_evidence_available": bool(
            readiness_checks.get("marulho_spike_readout_evidence_available")
        ),
        "marulho_spike_readout_grounded": bool(readiness_checks.get("marulho_spike_readout_grounded")),
        "marulho_spike_readout_non_generative": bool(
            readiness_checks.get("marulho_spike_readout_non_generative")
        ),
        "marulho_spike_readout_device_evidence_available": bool(
            readiness_checks.get("marulho_spike_readout_device_evidence_available")
        ),
        "marulho_spike_decoder_probe_available": bool(
            readiness_checks.get("marulho_spike_decoder_probe_available")
        ),
        "marulho_spike_decoder_probe_owned": bool(
            readiness_checks.get("marulho_spike_decoder_probe_owned")
        ),
        "marulho_spike_decoder_probe_non_generative": bool(
            readiness_checks.get("marulho_spike_decoder_probe_non_generative")
        ),
        "marulho_spike_decoder_probe_sparse": bool(
            readiness_checks.get("marulho_spike_decoder_probe_sparse")
        ),
        "marulho_spike_decoder_probe_device_evidence_available": bool(
            readiness_checks.get("marulho_spike_decoder_probe_device_evidence_available")
        ),
        "marulho_spike_decoder_probe_grounding_supported": bool(
            readiness_checks.get("marulho_spike_decoder_probe_grounding_supported")
        ),
        "marulho_spike_decoder_probe_temporal_state": bool(
            readiness_checks.get("marulho_spike_decoder_probe_temporal_state")
        ),
        "marulho_spike_language_neuron_adapter_available": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_available")
        ),
        "marulho_spike_language_neuron_adapter_owned": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_owned")
        ),
        "marulho_spike_language_neuron_adapter_non_generative": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_non_generative")
        ),
        "marulho_spike_language_neuron_adapter_device_evidence_available": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_device_evidence_available")
        ),
        "marulho_spike_language_neuron_adapter_sparse": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_sparse")
        ),
        "marulho_spike_language_neuron_adapter_dynamic": bool(
            readiness_checks.get("marulho_spike_language_neuron_adapter_dynamic")
        ),
        "cuda_runtime_scope_available": bool(readiness_checks.get("cuda_runtime_scope_available")),
        "subcortex_device_evidence_available": bool(readiness_checks.get("subcortex_device_evidence_available")),
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
    binding_device_evidence_available = bool(_binding_device_keys(binding_report))
    local_device_evidence_available = bool(_local_plasticity_device_keys(local_plasticity_report))
    structural_device_evidence_available = bool(
        binding_device_evidence_available or local_device_evidence_available
    )
    if growth:
        growth_ready = bool(growth.get("growth_ready"))
        ready_for_growth_review = bool(growth_ready and structural_device_evidence_available)
        active_growth_concepts = [
            _structural_growth_concept(item)
            for item in list(growth.get("active_growth_concepts") or [])[:4]
            if isinstance(item, Mapping)
        ]
        cases.append(
            {
                "intent": "evaluate_concept_capacity_growth",
                "phase": "structural_review",
                "ready_for_evaluation": ready_for_growth_review,
                "promotion_status": "ready_for_structural_review" if ready_for_growth_review else "monitor_only",
                "priority": 0.82 if ready_for_growth_review else 0.42,
                "evaluation_target": "increase_abstraction_capacity_without_memory_or_spike_health_regression",
                "baseline_metrics": {
                    "requested_output_dim": int(_float(growth.get("requested_output_dim"), 0.0)),
                    "current_output_dim": int(_float(growth.get("current_output_dim"), 0.0)),
                    "max_output_dim": int(_float(growth.get("max_output_dim"), 0.0)),
                    "growth_ready": growth_ready,
                    "active_growth_concept_count": len(active_growth_concepts),
                    "expansion_events": int(_float(growth.get("expansion_events"), 0.0)),
                    "prune_events": int(_float(growth.get("prune_events"), 0.0)),
                    "structural_device_evidence_available": structural_device_evidence_available,
                },
                "evidence_terms": [
                    "concept_growth",
                    "split_bias",
                    "growth_pressure",
                    "abstraction_capacity",
                    "structural_device_report",
                ],
                "promotion_constraints": _structural_promotion_constraints(ready_for_growth_review),
            }
        )
    binding_summary = _binding_structural_summary(binding_report)
    mutation_count = int(binding_summary["growth_events"]) + int(binding_summary["prune_events"])
    if binding_report:
        ready_for_binding_review = bool(mutation_count > 0 and binding_device_evidence_available)
        cases.append(
            {
                "intent": "evaluate_binding_topology_stability",
                "phase": "structural_review",
                "ready_for_evaluation": ready_for_binding_review,
                "promotion_status": "ready_for_structural_review" if ready_for_binding_review else "monitor_only",
                "priority": 0.78 if ready_for_binding_review else 0.40,
                "evaluation_target": "preserve_sparse_binding_reachability_with_auditable_edge_ledger",
                "baseline_metrics": {
                    **binding_summary,
                    "binding_device_evidence_available": binding_device_evidence_available,
                },
                "evidence_terms": ["binding_topology", "structural_mutation_ledger", "sparse_edges", "device_report"],
                "promotion_constraints": _structural_promotion_constraints(ready_for_binding_review),
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
            and bool(local_summary["homeostatic_state_available"])
            and not bool(local_summary["synaptic_validation_failed"])
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
    local_plasticity_report: Mapping[str, Any],
) -> dict[str, Any]:
    ready_cases = [case for case in structural_cases if bool(case.get("ready_for_evaluation"))]
    device_evidence_available = bool(
        _binding_device_keys(binding_report) or _local_plasticity_device_keys(local_plasticity_report)
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


def _binding_device_keys(binding_report: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, value in binding_report.items()
        if (str(key) == "device" or str(key).endswith("_device")) and value is not None
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


def _isolated_structural_delta(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pre_topology = _snapshot_binding_topology(pre_snapshot)
    post_topology = _snapshot_binding_topology(post_snapshot)
    edges_added_delta = int(_float(post_topology.get("edges_added_total"), 0.0)) - int(
        _float(pre_topology.get("edges_added_total"), 0.0)
    )
    edges_removed_delta = int(_float(post_topology.get("edges_removed_total"), 0.0)) - int(
        _float(pre_topology.get("edges_removed_total"), 0.0)
    )
    growth_events_delta = int(_float(post_topology.get("growth_events"), 0.0)) - int(
        _float(pre_topology.get("growth_events"), 0.0)
    )
    prune_events_delta = int(_float(post_topology.get("prune_events"), 0.0)) - int(
        _float(pre_topology.get("prune_events"), 0.0)
    )
    total_edge_delta = abs(edges_added_delta) + abs(edges_removed_delta)
    return {
        "edges_added_delta": edges_added_delta,
        "edges_removed_delta": edges_removed_delta,
        "growth_events_delta": growth_events_delta,
        "prune_events_delta": prune_events_delta,
        "total_edge_delta": total_edge_delta,
        "bounded_edge_delta_limit": 16,
        "bounded": (
            edges_added_delta >= 0
            and edges_removed_delta >= 0
            and growth_events_delta >= 0
            and prune_events_delta >= 0
            and total_edge_delta <= 16
        ),
    }


def _isolated_structural_snapshot_binding(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pre_hash = _sha256_json(dict(pre_snapshot))
    post_hash = _sha256_json(dict(post_snapshot))
    pre_revision = _snapshot_revision(pre_snapshot)
    post_revision = _snapshot_revision(post_snapshot)
    pre_topology = _snapshot_binding_topology(pre_snapshot)
    post_topology = _snapshot_binding_topology(post_snapshot)
    structural_delta_present = any(
        int(_float(post_topology.get(key), 0.0)) != int(_float(pre_topology.get(key), 0.0))
        for key in (
            "edges_added_total",
            "edges_removed_total",
            "growth_events",
            "prune_events",
        )
    )
    return {
        "hash_algorithm": "sha256_canonical_json",
        "pre_snapshot_hash": pre_hash,
        "post_snapshot_hash": post_hash,
        "snapshot_hashes_distinct": pre_hash != post_hash,
        "pre_state_revision": pre_revision,
        "post_state_revision": post_revision,
        "state_revision_order_valid": (
            pre_revision is None
            or post_revision is None
            or int(post_revision) >= int(pre_revision)
        ),
        "structural_delta_present": structural_delta_present,
        "raw_snapshots_exposed": False,
    }


def _snapshot_revision(snapshot: Mapping[str, Any]) -> int | None:
    for key in ("current_state_revision", "state_revision", "runtime_state_revision"):
        value = snapshot.get(key)
        if isinstance(value, bool):
            continue
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _snapshot_binding_topology(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    topology = snapshot.get("binding_topology") if isinstance(snapshot.get("binding_topology"), Mapping) else {}
    if topology:
        return topology
    binding = snapshot.get("binding") if isinstance(snapshot.get("binding"), Mapping) else {}
    structural_mutations = (
        binding.get("structural_mutations")
        if isinstance(binding.get("structural_mutations"), Mapping)
        else {}
    )
    return structural_mutations


def _isolated_structural_device_evidence(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pre_devices = _snapshot_structural_devices(pre_snapshot)
    post_devices = _snapshot_structural_devices(post_snapshot)
    return {
        "pre_devices": pre_devices,
        "post_devices": post_devices,
        "pre_device_keys": sorted(pre_devices.keys()),
        "post_device_keys": sorted(post_devices.keys()),
        "available": bool(pre_devices and post_devices),
        "consistent": bool(pre_devices and pre_devices == post_devices),
    }


def _snapshot_structural_devices(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    evidence = snapshot.get("device_evidence") if isinstance(snapshot.get("device_evidence"), Mapping) else {}
    devices: dict[str, Any] = {}
    for group_name in ("binding_devices", "local_plasticity_devices"):
        group = evidence.get(group_name) if isinstance(evidence.get(group_name), Mapping) else {}
        devices.update({f"{group_name}.{key}": value for key, value in group.items() if value is not None})
    return devices


def _isolated_spike_health_delta(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pre_health = pre_snapshot.get("spike_health") if isinstance(pre_snapshot.get("spike_health"), Mapping) else {}
    post_health = post_snapshot.get("spike_health") if isinstance(post_snapshot.get("spike_health"), Mapping) else {}
    pre_risk = _spike_health_risk_score(pre_health)
    post_risk = _spike_health_risk_score(post_health)
    return {
        "pre_risk_score": pre_risk,
        "post_risk_score": post_risk,
        "risk_delta": post_risk - pre_risk,
        "improved_or_stable": post_risk <= pre_risk,
    }


def _spike_health_risk_score(spike_health: Mapping[str, Any]) -> float:
    return round(
        _float(spike_health.get("silent_fraction"), 0.0)
        + _float(spike_health.get("saturated_fraction"), 0.0)
        + _float(spike_health.get("stale_fraction"), 0.0),
        6,
    )


def _isolated_runtime_truth_delta(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pre_truth = pre_snapshot.get("runtime_truth") if isinstance(pre_snapshot.get("runtime_truth"), Mapping) else {}
    post_truth = post_snapshot.get("runtime_truth") if isinstance(post_snapshot.get("runtime_truth"), Mapping) else {}
    pre_verdict = _text(pre_truth.get("verdict")) or "unknown"
    post_verdict = _text(post_truth.get("verdict")) or "unknown"
    pre_rank = _runtime_truth_rank(pre_verdict)
    post_rank = _runtime_truth_rank(post_verdict)
    return {
        "pre_verdict": pre_verdict,
        "post_verdict": post_verdict,
        "pre_rank": pre_rank,
        "post_rank": post_rank,
        "rank_delta": post_rank - pre_rank,
        "improved_or_stable": post_rank >= pre_rank,
    }


def _runtime_truth_rank(verdict: str) -> int:
    ranks = {
        "unknown": 0,
        "dead": 1,
        "failed": 1,
        "degraded": 2,
        "partial": 3,
        "alive": 4,
    }
    return ranks.get(verdict, 0)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _isolated_rollback_evidence(
    rollback_policy: Mapping[str, Any],
    *,
    pre_snapshot_hash: str | None = None,
) -> dict[str, Any]:
    available = bool(rollback_policy.get("available") or rollback_policy.get("reversible"))
    artifact = (
        rollback_policy.get("rollback_artifact")
        if isinstance(rollback_policy.get("rollback_artifact"), Mapping)
        else {}
    )
    rollback_hash = (
        rollback_policy.get("pre_snapshot_hash")
        or rollback_policy.get("rollback_snapshot_hash")
        or rollback_policy.get("snapshot_hash")
        or artifact.get("pre_snapshot_hash")
    )
    rollback_hash_text = str(rollback_hash) if rollback_hash is not None else None
    hash_match = bool(pre_snapshot_hash and rollback_hash_text == pre_snapshot_hash)
    artifact_path = _text(rollback_policy.get("artifact_path") or artifact.get("artifact_path"))
    artifact_hash = _text(
        rollback_policy.get("artifact_hash")
        or rollback_policy.get("rollback_artifact_hash")
        or artifact.get("artifact_hash")
    )
    tombstone_hash = _text(
        rollback_policy.get("tombstone_manifest_hash")
        or artifact.get("tombstone_manifest_hash")
    )
    artifact_available = bool(available and (artifact_path or artifact_hash or tombstone_hash))
    return {
        "available": available,
        "reversible": available,
        "snapshot_id": rollback_policy.get("snapshot_id"),
        "ledger_id": rollback_policy.get("ledger_id"),
        "pre_snapshot_hash": rollback_hash_text,
        "expected_pre_snapshot_hash": pre_snapshot_hash,
        "pre_snapshot_hash_match": hash_match,
        "bound_to_pre_snapshot": available and hash_match,
        "rollback_artifact": {
            "available": artifact_available,
            "artifact_path": artifact_path or None,
            "artifact_hash": artifact_hash or None,
            "tombstone_manifest_hash": tombstone_hash or None,
        },
    }


def _isolated_candidate_evidence(
    candidate_evidence: Mapping[str, Any],
    *,
    pre_snapshot_hash: str,
) -> dict[str, Any]:
    ticket = (
        candidate_evidence.get("ticket")
        if isinstance(candidate_evidence.get("ticket"), Mapping)
        else candidate_evidence.get("candidate_ticket")
        if isinstance(candidate_evidence.get("candidate_ticket"), Mapping)
        else {}
    )
    if not ticket and isinstance(candidate_evidence.get("tickets_sample"), Sequence):
        for item in candidate_evidence.get("tickets_sample") or []:
            if isinstance(item, Mapping):
                ticket = item
                break
    if not ticket and candidate_evidence.get("surface") == "column_structural_review_ticket.v1":
        ticket = candidate_evidence

    evidence = ticket.get("evidence") if isinstance(ticket.get("evidence"), Mapping) else {}
    checkpoint_baseline = (
        candidate_evidence.get("checkpoint_baseline")
        if isinstance(candidate_evidence.get("checkpoint_baseline"), Mapping)
        else {}
    )
    baseline_hash = _text(
        candidate_evidence.get("baseline_hash")
        or candidate_evidence.get("queue_state_hash")
        or checkpoint_baseline.get("queue_state_hash")
        or candidate_evidence.get("pre_snapshot_hash")
    )
    ticket_hash = _text(ticket.get("candidate_evidence_hash") or candidate_evidence.get("candidate_evidence_hash"))
    recomputed_ticket_hash = _sha256_json(dict(evidence)) if evidence else ""
    reason = _text(ticket.get("candidate_reason") or ticket.get("reason") or candidate_evidence.get("candidate_reason"))
    kind = _text(ticket.get("kind") or candidate_evidence.get("kind"))
    metrics = {
        "prediction_error": _float(evidence.get("prediction_error"), 0.0),
        "confidence": _float(evidence.get("confidence"), 1.0),
        "prediction_failure_streak": int(_float(evidence.get("prediction_failure_streak"), 0.0)),
        "estimated_cost": _float(evidence.get("estimated_cost"), 0.0),
        "usefulness": _float(evidence.get("usefulness"), 0.5),
        "memory_pressure": _float(evidence.get("memory_pressure"), 0.0),
    }
    repeated_failure_ready = (
        metrics["prediction_failure_streak"] >= 3
        and metrics["prediction_error"] >= 0.60
        and metrics["confidence"] <= 0.45
    )
    prune_pressure_ready = (
        metrics["memory_pressure"] >= 0.95
        or (metrics["estimated_cost"] >= 0.85 and metrics["confidence"] <= 0.25)
        or (metrics["estimated_cost"] >= 0.85 and metrics["usefulness"] <= 0.25)
    )
    return {
        "available": bool(candidate_evidence),
        "source": candidate_evidence.get("source") or ticket.get("source"),
        "ticket_surface": ticket.get("surface"),
        "ticket_id": ticket.get("ticket_id"),
        "kind": kind or None,
        "column_id": ticket.get("column_id"),
        "candidate_reason": reason or None,
        "candidate_reason_available": bool(reason),
        "candidate_evidence_hash": ticket_hash or None,
        "candidate_evidence_hash_recomputed": recomputed_ticket_hash or None,
        "candidate_evidence_hash_recomputed_match": bool(
            ticket_hash and recomputed_ticket_hash and ticket_hash == recomputed_ticket_hash
        ),
        "baseline_hash": baseline_hash or None,
        "baseline_hash_available": len(baseline_hash) == 64,
        "baseline_hash_matches_pre_snapshot": bool(
            baseline_hash and baseline_hash == pre_snapshot_hash
        ),
        "hash_algorithm": "sha256_canonical_json",
        "metrics": metrics,
        "repeated_failure_ready": repeated_failure_ready,
        "prune_or_sleep_pressure_ready": prune_pressure_ready,
        "candidate_reason_proves_growth_or_prune_pressure": bool(
            repeated_failure_ready or prune_pressure_ready
        ),
        "mutates_runtime_state": False,
        "calls_growth_or_prune": False,
        "writes_checkpoint": False,
    }


def _metric_delta(payload: Mapping[str, Any], prefix: str) -> tuple[float | None, float | None, float | None]:
    before = payload.get(f"{prefix}_before")
    after = payload.get(f"{prefix}_after")
    delta = payload.get(f"{prefix}_delta")
    before_value = None if before is None else _float(before, 0.0)
    after_value = None if after is None else _float(after, 0.0)
    delta_value = None if delta is None else _float(delta, 0.0)
    if delta_value is None and before_value is not None and after_value is not None:
        delta_value = after_value - before_value
    return before_value, after_value, delta_value


def _isolated_cost_usefulness_impact(
    cost_evidence: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = (
        candidate_binding.get("metrics")
        if isinstance(candidate_binding.get("metrics"), Mapping)
        else {}
    )
    latency_before, latency_after, latency_delta = _metric_delta(cost_evidence, "latency_ms")
    ram_before, ram_after, ram_delta = _metric_delta(cost_evidence, "ram_bytes")
    vram_before, vram_after, vram_delta = _metric_delta(cost_evidence, "vram_bytes")
    latency_available = latency_delta is not None
    ram_available = ram_delta is not None
    vram_available = vram_delta is not None
    return {
        "available": bool(cost_evidence) or bool(metrics),
        "estimated_cost": _float(metrics.get("estimated_cost"), 0.0),
        "usefulness": _float(metrics.get("usefulness"), 0.5),
        "memory_pressure": _float(metrics.get("memory_pressure"), 0.0),
        "latency_ms_before": latency_before,
        "latency_ms_after": latency_after,
        "latency_ms_delta": latency_delta,
        "ram_bytes_before": ram_before,
        "ram_bytes_after": ram_after,
        "ram_bytes_delta": ram_delta,
        "vram_bytes_before": vram_before,
        "vram_bytes_after": vram_after,
        "vram_bytes_delta": vram_delta,
        "latency_impact_available": latency_available,
        "ram_impact_available": ram_available,
        "vram_impact_available": vram_available,
        "latency_ram_vram_impact_available": bool(
            latency_available and ram_available and vram_available
        ),
        "impact_summary_hash": _sha256_json(
            {
                "cost_evidence": dict(cost_evidence),
                "candidate_metrics": dict(metrics),
            }
        ),
    }


def _isolated_runtime_truth_summary(
    pre_snapshot: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    runtime_truth_summary: Mapping[str, Any],
) -> dict[str, Any]:
    pre_truth = pre_snapshot.get("runtime_truth") if isinstance(pre_snapshot.get("runtime_truth"), Mapping) else {}
    post_truth = post_snapshot.get("runtime_truth") if isinstance(post_snapshot.get("runtime_truth"), Mapping) else {}
    supplied = dict(runtime_truth_summary or {})
    pre_verdict = _text(supplied.get("pre_verdict") or pre_truth.get("verdict")) or "unknown"
    post_verdict = _text(supplied.get("post_verdict") or post_truth.get("verdict")) or "unknown"
    summary = {
        "available": bool(supplied or pre_truth or post_truth),
        "pre_verdict": pre_verdict,
        "post_verdict": post_verdict,
        "pre_rank": _runtime_truth_rank(pre_verdict),
        "post_rank": _runtime_truth_rank(post_verdict),
        "tick_latency_ms_before": supplied.get("tick_latency_ms_before"),
        "tick_latency_ms_after": supplied.get("tick_latency_ms_after"),
        "throughput_tokens_per_sec_before": supplied.get("throughput_tokens_per_sec_before"),
        "throughput_tokens_per_sec_after": supplied.get("throughput_tokens_per_sec_after"),
        "candidate_runtime_truth_status": supplied.get("candidate_runtime_truth_status"),
        "report_path": supplied.get("report_path"),
    }
    summary["rank_delta"] = int(summary["post_rank"]) - int(summary["pre_rank"])
    summary["improved_or_stable"] = int(summary["post_rank"]) >= int(summary["pre_rank"])
    summary["summary_hash"] = _sha256_json(
        {"pre_runtime_truth": dict(pre_truth), "post_runtime_truth": dict(post_truth), "supplied": supplied}
    )
    return summary


def _isolated_no_mutation_proof(no_mutation_evidence: Mapping[str, Any]) -> dict[str, Any]:
    before_revision = no_mutation_evidence.get("state_revision_before")
    after_revision = no_mutation_evidence.get("state_revision_after")
    revision_stable = (
        before_revision is None
        or after_revision is None
        or int(_float(before_revision, 0.0)) == int(_float(after_revision, 0.0))
    )
    mutates_runtime_state = bool(no_mutation_evidence.get("mutates_runtime_state"))
    calls_growth_or_prune = bool(no_mutation_evidence.get("calls_growth_or_prune"))
    writes_checkpoint = bool(no_mutation_evidence.get("writes_checkpoint"))
    applies_structural_mutation = bool(no_mutation_evidence.get("applies_structural_mutation"))
    valid = (
        revision_stable
        and not mutates_runtime_state
        and not calls_growth_or_prune
        and not writes_checkpoint
        and not applies_structural_mutation
    )
    return {
        "available": bool(no_mutation_evidence),
        "source": no_mutation_evidence.get("source"),
        "state_revision_before": before_revision,
        "state_revision_after": after_revision,
        "state_revision_stable": revision_stable,
        "mutates_runtime_state": mutates_runtime_state,
        "calls_growth_or_prune": calls_growth_or_prune,
        "writes_checkpoint": writes_checkpoint,
        "applies_structural_mutation": applies_structural_mutation,
        "valid": valid,
        "proof_hash": _sha256_json(dict(no_mutation_evidence)) if no_mutation_evidence else None,
    }


def _checkpointed_candidate_gate(
    *,
    ready_for_review: bool,
    candidate_binding: Mapping[str, Any],
    cost_impact: Mapping[str, Any],
    truth_summary: Mapping[str, Any],
    rollback_evidence: Mapping[str, Any],
    no_mutation_proof: Mapping[str, Any],
) -> dict[str, Any]:
    rollback_artifact = (
        rollback_evidence.get("rollback_artifact")
        if isinstance(rollback_evidence.get("rollback_artifact"), Mapping)
        else {}
    )
    required = {
        "isolated_evaluation_ready": bool(ready_for_review),
        "candidate_evidence_available": bool(candidate_binding.get("available")),
        "candidate_reason_available": bool(candidate_binding.get("candidate_reason_available")),
        "candidate_evidence_hash_bound": bool(
            candidate_binding.get("candidate_evidence_hash_recomputed_match")
        ),
        "exact_baseline_hash_available": bool(candidate_binding.get("baseline_hash_available")),
        "candidate_reason_proves_growth_or_prune_pressure": bool(
            candidate_binding.get("candidate_reason_proves_growth_or_prune_pressure")
        ),
        "cost_usefulness_metrics_available": bool(cost_impact.get("available")),
        "latency_ram_vram_impact_available": bool(
            cost_impact.get("latency_ram_vram_impact_available")
        ),
        "runtime_truth_summary_available": bool(truth_summary.get("available")),
        "rollback_artifact_available": bool(rollback_artifact.get("available")),
        "rollback_bound_to_pre_snapshot": bool(rollback_evidence.get("bound_to_pre_snapshot")),
        "no_mutation_proof_available": bool(no_mutation_proof.get("available")),
        "no_mutation_proof_valid": bool(no_mutation_proof.get("valid")),
    }
    ready = all(bool(value) for value in required.values())
    return {
        "status": (
            "ready_for_checkpointed_candidate_review"
            if ready
            else "blocked_missing_checkpointed_candidate_evidence"
        ),
        "next_gate": (
            "operator_approved_structural_mutation_design"
            if ready
            else "collect_candidate_baseline_cost_runtime_rollback_no_mutation_evidence"
        ),
        "eligible_for_action": False,
        "eligible_for_fact_promotion": False,
        "eligible_for_structural_mutation": False,
        "eligible_for_operator_review": ready,
        "required_evidence": required,
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


def _spike_readout_device_evidence(
    cuda_runtime: Mapping[str, Any],
    subcortex_devices: Mapping[str, Any],
) -> dict[str, Any]:
    observed_devices = _observed_tensor_devices(subcortex_devices)
    if observed_devices:
        cuda_devices = [device for device in observed_devices if _is_cuda_device(device)]
        selected_device = cuda_devices[0] if cuda_devices else observed_devices[0]
        return {
            "device": selected_device,
            "cuda_device_selected": bool(cuda_devices),
            "source": "observed_subcortex_tensor_devices",
            "observed_device_count": len(observed_devices),
            "observed_devices": observed_devices[:12],
        }
    direct_device = _text(cuda_runtime.get("tensor_device"))
    if direct_device:
        return {
            "device": direct_device,
            "cuda_device_selected": _is_cuda_device(direct_device),
            "source": "configured_tensor_device_fallback",
            "observed_device_count": 0,
            "observed_devices": [],
        }
    return {
        "device": "unknown",
        "cuda_device_selected": False,
        "source": "missing_device_evidence",
        "observed_device_count": 0,
        "observed_devices": [],
    }


def _observed_tensor_devices(report: Mapping[str, Any]) -> list[str]:
    devices: list[str] = []

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key, candidate in value.items():
            if isinstance(candidate, Mapping):
                visit(candidate)
                continue
            if key == "device" or key.endswith("_device"):
                device = _text(candidate)
                if device and device not in devices:
                    devices.append(device)

    visit(report)
    return devices


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
    evidence_available = bool(grounding.get("correlation_evidence_available"))
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
        "correlation_window_available": evidence_available,
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
