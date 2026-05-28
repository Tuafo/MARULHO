from __future__ import annotations

import unittest

from hecsn.semantics import (
    build_snn_language_readiness_surface,
    build_spike_language_decoder_probe,
    build_subcortical_spike_readout_evidence,
    build_subcortical_deliberation_surface,
    build_subcortical_self_repair_evaluation_surface,
    build_subcortical_self_repair_surface,
    build_subcortical_structural_plasticity_surface,
)


class SubcorticalDeliberationSurfaceTests(unittest.TestCase):
    def test_high_prediction_error_yields_stabilize_candidate(self) -> None:
        surface = build_subcortical_deliberation_surface(
            {
                "prediction_error_mean": 0.42,
                "prediction_error_max": 0.71,
                "predictive_confidence_mean": 0.51,
                "predictive_confidence_min": 0.34,
                "dopamine": 0.1,
                "norepinephrine": 0.2,
                "recent_concepts": ["coral thermal memory"],
                "concept_candidates": [
                    {
                        "label": "coral thermal memory",
                        "top_terms": ["coral", "thermal", "memory"],
                        "observations": 4,
                        "uncertainty": 0.3,
                        "temporal_coherence": 0.6,
                    }
                ],
            }
        )

        self.assertEqual(surface["surface"], "subcortical_control_candidates.v1")
        self.assertEqual(surface["control_hint"], "stabilize_prediction")
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertEqual(surface["candidates"][0]["intent"], "stabilize_prediction")
        self.assertIn("coral thermal memory", surface["candidates"][0]["candidate_text"])
        self.assertNotIn("prompt", surface["candidates"][0])
        self.assertIn("prediction_error_mean", surface["candidates"][0]["grounding"])
        for replay_key in ("candidate_id", "target_type", "suggested_endpoint", "suggested_input", "reason_codes"):
            self.assertNotIn(replay_key, surface["candidates"][0])
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "ready_for_replay_review")
        self.assertTrue(surface["candidates"][0]["promotion_gate"]["eligible_for_replay_review"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_fact_promotion"])
        self.assertEqual(surface["promotion_summary"]["ready_for_replay_review"], 3)
        self.assertFalse(surface["promotion_summary"]["eligible_for_action"])

    def test_missing_focus_yields_collect_grounding(self) -> None:
        surface = build_subcortical_deliberation_surface(
            {
                "prediction_error_mean": 0.04,
                "prediction_error_max": 0.08,
                "predictive_confidence_mean": 0.60,
                "predictive_confidence_min": 0.50,
                "recent_concepts": [],
                "concept_candidates": [],
            }
        )

        self.assertEqual(surface["control_hint"], "collect_grounding")
        self.assertEqual(surface["candidates"][0]["intent"], "collect_grounding")
        self.assertEqual(surface["grounding"]["concept_focus"], None)
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "blocked_missing_grounding")
        self.assertEqual(surface["promotion_summary"]["blocked_missing_grounding"], 1)
        self.assertLessEqual(len(surface["candidates"]), 3)


class SNNLanguageReadinessSurfaceTests(unittest.TestCase):
    def test_decoder_probe_consumes_readout_slots_without_generating_text(self) -> None:
        evidence = build_subcortical_spike_readout_evidence(
            {
                "prediction_error_mean": 0.31,
                "predictive_confidence_min": 0.41,
                "recent_concepts": ["spike language"],
                "concept_candidates": [{"label": "spike language", "observations": 4}],
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cpu",
                    "subcortex_tensor_devices": {"competitive": {"prototypes_device": "cpu"}},
                }
            },
        )

        probe = build_spike_language_decoder_probe(evidence)

        self.assertEqual(probe["artifact_kind"], "terminus_hecsn_spike_language_decoder_probe")
        self.assertEqual(probe["surface"], "snn_language_decoder_probe_evidence.v1")
        self.assertTrue(probe["owned_by_hecsn"])
        self.assertFalse(probe["external_dependency"])
        self.assertFalse(probe["loads_external_checkpoint"])
        self.assertFalse(probe["generates_text"])
        self.assertFalse(probe["executable"])
        self.assertFalse(probe["decodes_text"])
        self.assertFalse(probe["mutates_runtime_state"])
        self.assertEqual(probe["support_evidence"]["grounded_slot_labels"], ["spike language"])
        self.assertFalse(probe["promotion_constraints"]["eligible_for_language_generation"])

    def test_decoder_probe_reports_device_sparsity_and_support_evidence(self) -> None:
        probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [
                    {
                        "slot_id": "slot_a",
                        "label": "thermal memory",
                        "pressure_band": "medium",
                        "grounded": True,
                    }
                ],
                "device_evidence": {"device": "cpu", "source": "observed_subcortex_tensor_devices"},
            }
        )

        self.assertEqual(probe["device_evidence"]["tensor_device"], "cpu")
        self.assertEqual(probe["device_evidence"]["device_source"], "observed_subcortex_tensor_devices")
        self.assertEqual(probe["sparsity_evidence"]["code_dim"], 32)
        self.assertTrue(probe["sparsity_evidence"]["meets_sparse_readout_floor"])
        self.assertEqual(probe["support_evidence"]["readout_slot_count"], 1)
        self.assertEqual(probe["support_evidence"]["grounded_slot_count"], 1)
        self.assertEqual(probe["support_evidence"]["unsupported_slot_count"], 0)
        self.assertTrue(probe["support_evidence"]["supported"])

    def test_spike_readout_evidence_is_owned_cuda_aware_and_non_generative(self) -> None:
        evidence = build_subcortical_spike_readout_evidence(
            {
                "prediction_error_mean": 0.31,
                "prediction_error_max": 0.56,
                "predictive_confidence_mean": 0.67,
                "predictive_confidence_min": 0.41,
                "dopamine": 0.18,
                "norepinephrine": 0.28,
                "recent_concepts": ["spike language"],
                "concept_candidates": [{"label": "spike language", "observations": 4}],
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {"competitive": {"prototypes_device": "cuda:0"}},
                }
            },
        )

        self.assertEqual(evidence["artifact_kind"], "terminus_subcortical_spike_readout_evidence")
        self.assertEqual(evidence["surface"], "subcortical_spike_readout_evidence.v1")
        self.assertTrue(evidence["grounded"])
        self.assertTrue(evidence["advisory"])
        self.assertFalse(evidence["executable"])
        self.assertFalse(evidence["mutates_runtime_state"])
        self.assertFalse(evidence["generates_text"])
        self.assertNotIn("retired_runtime_dependency", evidence)
        self.assertEqual(evidence["device_evidence"]["device"], "cuda:0")
        self.assertTrue(evidence["device_evidence"]["cuda_device_selected"])
        self.assertEqual(evidence["device_evidence"]["source"], "observed_subcortex_tensor_devices")
        self.assertEqual(evidence["population_code"]["concept_focus"], "spike language")
        self.assertEqual(evidence["readout_slots"][0]["label"], "spike language")
        self.assertFalse(evidence["promotion_constraints"]["eligible_for_language_generation"])

    def test_spike_readout_device_evidence_prefers_observed_tensors_over_configured_cuda(self) -> None:
        evidence = build_subcortical_spike_readout_evidence(
            {
                "prediction_error_mean": 0.21,
                "predictive_confidence_mean": 0.71,
                "recent_concepts": ["device evidence"],
                "concept_candidates": [{"label": "device evidence", "observations": 2}],
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {"prototypes_device": "cpu"},
                        "predictive": {"location_device": "cpu"},
                        "binding": {"binding_state_device": "cpu"},
                    },
                }
            },
        )

        device_evidence = evidence["device_evidence"]
        self.assertEqual(device_evidence["device"], "cpu")
        self.assertFalse(device_evidence["cuda_device_selected"])
        self.assertEqual(device_evidence["source"], "observed_subcortex_tensor_devices")
        self.assertEqual(device_evidence["observed_device_count"], 1)

    def test_readiness_tracks_pure_snn_language_research_without_enabling_generation(self) -> None:
        surface = build_snn_language_readiness_surface(
            {
                "prediction_error_mean": 0.22,
                "prediction_error_max": 0.48,
                "predictive_confidence_mean": 0.62,
                "predictive_confidence_min": 0.44,
                "dopamine": 0.1,
                "norepinephrine": 0.2,
                "recent_concepts": ["spiking language"],
                "concept_candidates": [
                    {
                        "label": "spiking language",
                        "top_terms": ["spiking", "language"],
                        "observations": 5,
                        "uncertainty": 0.2,
                        "temporal_coherence": 0.7,
                    }
                ],
            },
            {
                "spike_health": {
                    "activity_state": "stale_routing_risk",
                    "silent_fraction": 0.0,
                    "saturated_fraction": 0.0,
                    "stale_fraction": 0.30,
                },
                "weight_distribution": {"validates_full_log_stdp_weight_target": True},
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {"prototypes_device": "cuda:0"},
                        "predictive": {"location_device": "cuda:0"},
                    },
                }
            },
        )

        self.assertEqual(surface["schema_version"], 1)
        self.assertEqual(surface["artifact_kind"], "terminus_snn_native_language_readiness_gate")
        self.assertEqual(surface["surface"], "snn_native_language_readiness.v1")
        self.assertEqual(surface["endpoint"], "/terminus/snn-language-readiness")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertEqual(surface["promotion_gate"]["status"], "research_candidate_only")
        self.assertEqual(surface["promotion_gate"]["next_gate"], "build_local_snn_language_generator_adapter")
        self.assertFalse(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(
            [candidate["name"] for candidate in surface["research_candidates"]],
            ["NeuronSpark", "Nord-AI"],
        )
        self.assertEqual(
            surface["research_candidates"][0]["integration_status"],
            "reference_for_hecsn_owned_reimplementation",
        )
        self.assertIn("hecsn_owned_language_neuron_module", surface["research_candidates"][0]["required_local_evidence"])
        self.assertIn("hecsn_native_snn_decoder", surface["research_candidates"][0]["required_local_evidence"])
        self.assertTrue(surface["safety_invariants"]["requires_hecsn_owned_implementation"])
        self.assertTrue(surface["safety_invariants"]["requires_hecsn_controlled_training"])
        self.assertIn("HECSN must own", " ".join(surface["limitations"]))
        self.assertTrue(surface["readiness_checks"]["grounded_language_surface_available"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_readout_evidence_available"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_readout_grounded"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_readout_non_generative"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_readout_device_evidence_available"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_available"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_owned"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_non_generative"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_sparse"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_grounding_supported"])
        self.assertEqual(surface["current_spike_readout_evidence"]["surface"], "subcortical_spike_readout_evidence.v1")
        self.assertEqual(surface["current_spike_readout_evidence"]["device"], "cuda:0")
        self.assertTrue(surface["current_spike_readout_evidence"]["cuda_device_selected"])
        self.assertFalse(surface["current_spike_readout_evidence"]["generates_text"])
        self.assertEqual(
            surface["current_decoder_probe_evidence"]["surface"],
            "snn_language_decoder_probe_evidence.v1",
        )
        self.assertTrue(surface["current_decoder_probe_evidence"]["owned_by_hecsn"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["executable"])
        self.assertFalse(surface["readiness_checks"]["local_snn_language_generator_available"])
        self.assertIn("activation_sparsity_report", surface["success_evidence"])

    def test_ready_generator_requires_sparsity_grounding_and_device_evidence(self) -> None:
        surface = build_snn_language_readiness_surface(
            {
                "prediction_error_mean": 0.05,
                "predictive_confidence_mean": 0.9,
                "recent_concepts": ["grounded decoder"],
                "concept_candidates": [{"label": "grounded decoder", "observations": 3}],
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {"competitive": {"prototypes_device": "cuda:0"}},
                    "snn_language_generator_device_report": {"device": "cuda:0"},
                    "snn_language_activation_sparsity": {"mean_sparsity": 0.93},
                    "snn_language_grounding_support": {"supported": True},
                }
            },
        )

        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_isolated_language_evaluation")
        self.assertTrue(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertTrue(surface["readiness_checks"]["hecsn_spike_decoder_probe_available"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(surface["safety_invariants"]["eligible_for_cognition_substrate"])


class SubcorticalSelfRepairSurfaceTests(unittest.TestCase):
    def test_silent_spike_health_without_correlation_window_stays_insufficient(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "silent_risk",
                "silent_fraction": 0.91,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.4,
                "correlation_evidence_available": False,
                "correlation": {"status": "insufficient_window", "sample_count": 2},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["schema_version"], 1)
        self.assertEqual(surface["artifact_kind"], "terminus_subcortical_self_repair_gate_plan")
        self.assertEqual(surface["endpoint"], "/terminus/subcortical-self-repair")
        self.assertEqual(surface["review_role"], "operator_replay_deep_sleep_review_only")
        self.assertEqual(surface["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertTrue(surface["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", surface)
        self.assertEqual(surface["candidates"][0]["intent"], "review_column_revival")
        self.assertIn("promotion_gate", surface["candidates"][0])
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "insufficient_evidence")
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_action"])
        self.assertFalse(surface["candidates"][0]["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(surface["promotion_summary"]["eligible_for_structural_mutation"])
        self.assertEqual(surface["promotion_gate"]["status"], "insufficient_evidence")
        self.assertEqual(surface["promotion_gate"]["next_gate"], "collect_spike_window")
        self.assertFalse(surface["promotion_gate"]["requires_operator_approval"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_structural_mutation"])

    def test_silent_spike_health_with_correlation_window_yields_repair_review_candidate(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "silent_risk",
                "silent_fraction": 0.91,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.4,
                "correlation_evidence_available": True,
                "correlation": {"status": "balanced", "sample_count": 8},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["candidates"][0]["intent"], "review_column_revival")
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "ready_for_replay_review")
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_replay_review")

    def test_overcorrelated_spike_health_yields_decorrelation_candidate(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "sparse_responsive",
                "silent_fraction": 0.0,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.0,
                "correlation_evidence_available": True,
                "correlation": {
                    "status": "overcorrelated_risk",
                    "sample_count": 8,
                    "active_columns": 3,
                    "valid_pairs": 6,
                    "mean_abs_offdiag_correlation": 0.91,
                },
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["candidates"][0]["intent"], "review_decorrelation_or_prune")
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["next_gate"], "deep_sleep_or_replay_repair_gate")
        self.assertEqual(surface["promotion_summary"]["ready_for_replay_review"], 1)
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_replay_review")

    def test_insufficient_spike_window_stays_monitor_only(self) -> None:
        surface = build_subcortical_self_repair_surface(
            {
                "activity_state": "sparse_responsive",
                "silent_fraction": 0.0,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.0,
                "correlation_evidence_available": False,
                "correlation": {"status": "insufficient_window", "sample_count": 0},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["candidates"][0]["intent"], "monitor_spike_health")
        self.assertEqual(surface["candidates"][0]["promotion_gate"]["status"], "insufficient_evidence")
        self.assertEqual(surface["promotion_summary"]["insufficient_evidence"], 1)
        self.assertEqual(surface["promotion_gate"]["status"], "insufficient_evidence")
        self.assertEqual(surface["promotion_gate"]["next_gate"], "collect_spike_window")


class SubcorticalSelfRepairEvaluationSurfaceTests(unittest.TestCase):
    def test_ready_repair_candidate_yields_isolated_evaluation_plan(self) -> None:
        surface = build_subcortical_self_repair_evaluation_surface(
            {
                "activity_state": "stale_routing_risk",
                "silent_fraction": 0.1,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.86,
                "correlation_evidence_available": True,
                "correlation": {"status": "balanced", "sample_count": 8},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["schema_version"], 1)
        self.assertEqual(surface["artifact_kind"], "terminus_subcortical_self_repair_evaluation_plan")
        self.assertEqual(surface["surface"], "subcortical_self_repair_evaluation.v1")
        self.assertEqual(surface["endpoint"], "/terminus/subcortical-self-repair/evaluation")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertEqual(surface["evaluation_gate"]["status"], "ready_for_isolated_evaluation")
        self.assertEqual(
            surface["evaluation_gate"]["next_gate"],
            "operator_approved_deep_sleep_or_replay_evaluation",
        )
        self.assertFalse(surface["evaluation_gate"]["eligible_for_structural_mutation"])
        self.assertEqual(surface["evaluation_cases"][0]["intent"], "review_stale_column_revival")
        self.assertTrue(surface["evaluation_cases"][0]["ready_for_evaluation"])
        self.assertEqual(
            surface["evaluation_cases"][0]["evaluation_target"],
            "reduce_silence_or_stale_routing_without_saturation",
        )
        self.assertIn("runtime_truth_delta", surface["success_evidence"])
        for forbidden_key in ("candidate_id", "suggested_endpoint", "suggested_input", "operation"):
            self.assertNotIn(forbidden_key, surface["evaluation_cases"][0])

    def test_insufficient_spike_window_blocks_evaluation(self) -> None:
        surface = build_subcortical_self_repair_evaluation_surface(
            {
                "activity_state": "sparse_responsive",
                "silent_fraction": 0.0,
                "saturated_fraction": 0.0,
                "stale_fraction": 0.0,
                "correlation_evidence_available": False,
                "correlation": {"status": "insufficient_window", "sample_count": 0},
                "not_liveness_claim": True,
            }
        )

        self.assertEqual(surface["evaluation_gate"]["status"], "blocked_missing_spike_window")
        self.assertEqual(surface["evaluation_gate"]["next_gate"], "collect_spike_window")
        self.assertFalse(surface["evaluation_cases"][0]["ready_for_evaluation"])
        self.assertFalse(surface["safety_invariants"]["eligible_for_structural_mutation"])


class SubcorticalStructuralPlasticitySurfaceTests(unittest.TestCase):
    def test_growth_and_binding_evidence_yields_isolated_structural_evaluation(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {
                "growth": {
                    "growth_ready": True,
                    "requested_output_dim": 2,
                    "current_output_dim": 1,
                    "max_output_dim": 4,
                    "expansion_events": 1,
                    "prune_events": 0,
                    "active_growth_concepts": [
                        {
                            "concept_id": "c1",
                            "label": "thermal memory",
                            "growth_pressure": 0.61,
                            "split_bias": 0.52,
                            "observations": 5,
                            "top_terms": ["thermal", "memory"],
                        }
                    ],
                }
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "routing_search_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {
                            "module": "competitive_columns",
                            "device": "cuda:0",
                            "local_plasticity": {
                                "module": "local_plasticity",
                                "device": "cuda:0",
                                "spike_backend": "adex",
                                "plasticity_rule": "triplet",
                                "input_eligibility_device": "cuda:0",
                                "projection_eligibility_device": "cuda:0",
                                "assembly_projection_eligibility_device": "cuda:0",
                                "firing_rate_ema_device": "cuda:0",
                                "synaptic_scale_device": "cuda:0",
                                "inhibitory_trace_device": "cuda:0",
                                "inhibitory_tone_device": "cuda:0",
                                "adex_device": "cuda:0",
                            },
                        },
                        "binding": {
                            "module": "hypercube_binding",
                            "device": "cuda:0",
                            "binding_state_device": "cuda:0",
                            "neighbor_ids_device": "cuda:0",
                            "neighbor_weights_device": "cuda:0",
                            "learned_weights_device": "cuda:0",
                            "degree_device": "cuda:0",
                            "topology": {"module": "hypercube_topology"},
                            "structural_mutations": {
                                "growth_events": 1,
                                "prune_events": 0,
                                "edges_added_total": 3,
                                "edges_removed_total": 0,
                                "recent_events": [{"type": "grow", "added_edges": 3}],
                            },
                        }
                    },
                }
            },
        )

        self.assertEqual(surface["schema_version"], 1)
        self.assertEqual(surface["artifact_kind"], "terminus_subcortical_structural_plasticity_gate_plan")
        self.assertEqual(surface["surface"], "subcortical_structural_plasticity.v1")
        self.assertEqual(surface["endpoint"], "/terminus/subcortical-structural-plasticity")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_isolated_structural_evaluation")
        self.assertFalse(surface["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertEqual(surface["concept_growth"]["active_growth_concepts"][0]["concept_id"], "c1")
        self.assertEqual(surface["binding_topology"]["growth_events"], 1)
        self.assertTrue(surface["local_plasticity"]["available"])
        self.assertTrue(surface["local_plasticity"]["eligibility_traces_available"])
        self.assertTrue(surface["local_plasticity"]["homeostatic_state_available"])
        self.assertEqual(surface["local_plasticity"]["spike_backend"], "adex")
        self.assertEqual(surface["device_evidence"]["binding_devices"]["binding_state_device"], "cuda:0")
        self.assertEqual(
            surface["device_evidence"]["local_plasticity_devices"]["input_eligibility_device"],
            "cuda:0",
        )
        self.assertIn("rollback_policy", surface["success_evidence"])
        self.assertIn("local_plasticity_stability_delta", surface["success_evidence"])
        for case in surface["structural_cases"]:
            self.assertFalse(case["promotion_constraints"]["eligible_for_structural_mutation"])
            self.assertNotIn("suggested_endpoint", case)
            self.assertNotIn("suggested_input", case)

    def test_local_plasticity_evidence_can_trigger_structural_review_without_mutation(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {"growth": {}},
            {
                "spike_health": {
                    "activity_state": "stale_routing_risk",
                    "silent_fraction": 0.0,
                    "saturated_fraction": 0.0,
                    "stale_fraction": 0.30,
                },
                "weight_distribution": {"validates_full_log_stdp_weight_target": True},
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {
                            "module": "competitive_columns",
                            "device": "cuda:0",
                            "local_plasticity": {
                                "module": "local_plasticity",
                                "device": "cuda:0",
                                "spike_backend": "proxy",
                                "plasticity_rule": "pair",
                                "input_eligibility_device": "cuda:0",
                                "projection_eligibility_device": "cuda:0",
                                "assembly_projection_eligibility_device": "cuda:0",
                                "firing_rate_ema_device": "cuda:0",
                                "synaptic_scale_device": "cuda:0",
                                "inhibitory_trace_device": "cuda:0",
                                "inhibitory_tone_device": "cuda:0",
                            },
                        }
                    },
                }
            },
        )

        local_cases = [
            case
            for case in surface["structural_cases"]
            if case["intent"] == "evaluate_local_plasticity_growth_prune_pressure"
        ]
        self.assertEqual(len(local_cases), 1)
        self.assertTrue(local_cases[0]["ready_for_evaluation"])
        self.assertEqual(
            local_cases[0]["evaluation_target"],
            "prove_local_stdp_scaling_and_inhibition_stabilize_growth_or_prune_pressure",
        )
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_isolated_structural_evaluation")
        self.assertFalse(surface["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertFalse(local_cases[0]["promotion_constraints"]["eligible_for_structural_mutation"])
        self.assertTrue(surface["device_evidence"]["local_plasticity_report_available"])

    def test_local_plasticity_without_homeostatic_state_stays_monitor_only(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {"growth": {}},
            {
                "spike_health": {
                    "activity_state": "stale_routing_risk",
                    "silent_fraction": 0.0,
                    "saturated_fraction": 0.0,
                    "stale_fraction": 0.30,
                },
                "weight_distribution": {"validates_full_log_stdp_weight_target": True},
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {
                            "module": "competitive_columns",
                            "device": "cuda:0",
                            "local_plasticity": {
                                "module": "local_plasticity",
                                "device": "cuda:0",
                                "input_eligibility_device": "cuda:0",
                                "projection_eligibility_device": "cuda:0",
                                "assembly_projection_eligibility_device": "cuda:0",
                            },
                        }
                    },
                },
            },
        )

        local_case = surface["structural_cases"][0]
        self.assertEqual(local_case["intent"], "evaluate_local_plasticity_growth_prune_pressure")
        self.assertFalse(local_case["ready_for_evaluation"])
        self.assertEqual(surface["promotion_gate"]["status"], "monitor_only")
        self.assertFalse(surface["local_plasticity"]["homeostatic_state_available"])

    def test_failed_synaptic_validation_blocks_local_plasticity_review(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {"growth": {}},
            {
                "spike_health": {
                    "activity_state": "stale_routing_risk",
                    "silent_fraction": 0.0,
                    "saturated_fraction": 0.0,
                    "stale_fraction": 0.30,
                },
                "weight_distribution": {"validates_full_log_stdp_weight_target": False},
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "competitive": {
                            "module": "competitive_columns",
                            "device": "cuda:0",
                            "local_plasticity": {
                                "module": "local_plasticity",
                                "device": "cuda:0",
                                "input_eligibility_device": "cuda:0",
                                "projection_eligibility_device": "cuda:0",
                                "assembly_projection_eligibility_device": "cuda:0",
                                "firing_rate_ema_device": "cuda:0",
                                "synaptic_scale_device": "cuda:0",
                                "inhibitory_trace_device": "cuda:0",
                                "inhibitory_tone_device": "cuda:0",
                            },
                        }
                    },
                },
            },
        )

        local_case = surface["structural_cases"][0]
        self.assertEqual(local_case["intent"], "evaluate_local_plasticity_growth_prune_pressure")
        self.assertFalse(local_case["ready_for_evaluation"])
        self.assertEqual(surface["promotion_gate"]["status"], "monitor_only")
        self.assertTrue(surface["local_plasticity"]["synaptic_validation_failed"])

    def test_missing_structural_evidence_collects_more_evidence(self) -> None:
        surface = build_subcortical_structural_plasticity_surface({}, {})

        self.assertEqual(surface["promotion_gate"]["status"], "insufficient_device_evidence")
        self.assertEqual(surface["promotion_gate"]["next_gate"], "collect_cuda_structural_device_report")
        self.assertFalse(surface["structural_cases"][0]["ready_for_evaluation"])
        self.assertFalse(surface["safety_invariants"]["eligible_for_structural_mutation"])

    def test_concept_growth_without_structural_device_evidence_is_not_ready(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {
                "growth": {
                    "growth_ready": True,
                    "requested_output_dim": 2,
                    "current_output_dim": 1,
                    "max_output_dim": 4,
                    "active_growth_concepts": [
                        {
                            "concept_id": "c1",
                            "label": "thermal memory",
                            "growth_pressure": 0.61,
                        }
                    ],
                }
            },
            {"cuda_first_runtime": {"tensor_device": "cuda:0"}},
        )

        growth_case = surface["structural_cases"][0]
        self.assertEqual(growth_case["intent"], "evaluate_concept_capacity_growth")
        self.assertFalse(growth_case["ready_for_evaluation"])
        self.assertFalse(growth_case["baseline_metrics"]["structural_device_evidence_available"])
        self.assertEqual(surface["promotion_gate"]["status"], "insufficient_device_evidence")
        self.assertEqual(surface["promotion_gate"]["next_gate"], "collect_cuda_structural_device_report")

    def test_binding_mutation_without_device_keys_is_not_ready(self) -> None:
        surface = build_subcortical_structural_plasticity_surface(
            {},
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {
                        "binding": {
                            "module": "hypercube_binding",
                            "topology": {"module": "hypercube_topology"},
                            "structural_mutations": {
                                "growth_events": 1,
                                "prune_events": 0,
                                "edges_added_total": 3,
                            },
                        }
                    },
                }
            },
        )

        binding_case = surface["structural_cases"][0]
        self.assertEqual(binding_case["intent"], "evaluate_binding_topology_stability")
        self.assertFalse(binding_case["ready_for_evaluation"])
        self.assertFalse(binding_case["baseline_metrics"]["binding_device_evidence_available"])
        self.assertEqual(surface["promotion_gate"]["status"], "insufficient_device_evidence")


if __name__ == "__main__":
    unittest.main()
