from __future__ import annotations

import hashlib
import json
import unittest

from marulho.semantics import (
    build_snn_language_readiness_surface,
    build_snn_language_evaluation_surface,
    build_snn_language_training_readiness_surface,
    build_spike_language_plasticity_application_design,
    build_spike_language_plasticity_pressure,
    build_spike_language_plasticity_shadow_delta,
    build_snn_language_transition_memory_prediction_evaluation,
    build_snn_language_transition_memory_sleep_policy,
    build_snn_language_transition_memory_regeneration_proposal,
    build_snn_language_readout_emission,
    build_snn_language_readout_trajectory_evidence,
    evaluate_spike_language_adapter_heldout,
    evaluate_spike_language_plasticity_live_application_preflight,
    evaluate_spike_language_plasticity_live_application_readiness,
    evaluate_spike_language_plasticity_shadow_application,
    evaluate_spike_language_plasticity_replay,
    evaluate_spike_language_sequence_mismatch,
    evaluate_spike_language_trainer_dry_run,
    evaluate_snn_language_readout_rollout_replay,
    generate_snn_language_readout_draft,
    predict_spike_language_sequence,
    rollout_snn_language_readout_candidate,
    run_spike_language_plasticity_replay_experiment,
    run_spike_language_plasticity_trial,
    run_spike_language_trainer_dry_run,
    build_spike_language_neuron_adapter,
    build_spike_language_decoder_probe,
    build_subcortical_spike_readout_evidence,
    build_subcortical_deliberation_surface,
    build_subcortical_self_repair_evaluation_surface,
    build_subcortical_self_repair_surface,
    build_subcortical_structural_mutation_design,
    build_subcortical_structural_mutation_preflight,
    build_subcortical_structural_plasticity_surface,
    evaluate_subcortical_structural_plasticity_isolated,
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

        self.assertEqual(probe["artifact_kind"], "terminus_marulho_spike_language_decoder_probe")
        self.assertEqual(probe["surface"], "snn_language_decoder_probe_evidence.v1")
        self.assertTrue(probe["owned_by_marulho"])
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

    def test_decoder_probe_reports_sparse_recurrent_state_without_text_generation(self) -> None:
        probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [
                    {
                        "slot_id": "slot_a",
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    },
                    {
                        "slot_id": "slot_b",
                        "label": "concept focus",
                        "pressure_band": "medium",
                        "grounded": True,
                    },
                ],
                "device_evidence": {"device": "cpu", "source": "observed_subcortex_tensor_devices"},
            }
        )

        temporal = probe["temporal_state_evidence"]
        sparse_code = probe["sparse_code_evidence"]
        self.assertTrue(temporal["dynamic_state_available"])
        self.assertEqual(temporal["timestep_count"], 2)
        self.assertEqual(temporal["state_device"], "cpu")
        self.assertGreaterEqual(temporal["active_transition_count"], 1)
        self.assertTrue(temporal["has_temporal_order"])
        self.assertGreater(sparse_code["active_index_count"], 0)
        self.assertLessEqual(sparse_code["active_index_count"], probe["sparsity_evidence"]["code_dim"])
        self.assertFalse(probe["generates_text"])
        self.assertFalse(probe["promotion_constraints"]["eligible_for_language_generation"])

    def test_language_neuron_adapter_consumes_probe_without_generating_text(self) -> None:
        probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [
                    {
                        "slot_id": "slot_a",
                        "label": "prediction error",
                        "pressure_band": "high",
                        "grounded": True,
                    },
                    {
                        "slot_id": "slot_b",
                        "label": "concept focus",
                        "pressure_band": "medium",
                        "grounded": True,
                    },
                ],
                "device_evidence": {"device": "cpu", "source": "observed_subcortex_tensor_devices"},
            }
        )

        adapter = build_spike_language_neuron_adapter(probe)

        self.assertEqual(adapter["artifact_kind"], "terminus_marulho_spike_language_neuron_adapter")
        self.assertEqual(adapter["surface"], "snn_language_neuron_adapter_evidence.v1")
        self.assertTrue(adapter["owned_by_marulho"])
        self.assertFalse(adapter["external_dependency"])
        self.assertFalse(adapter["loads_external_checkpoint"])
        self.assertFalse(adapter["generates_text"])
        self.assertFalse(adapter["decodes_text"])
        self.assertFalse(adapter["trains"])
        self.assertFalse(adapter["mutates_runtime_state"])
        self.assertEqual(adapter["device_evidence"]["tensor_device"], "cpu")
        self.assertGreater(adapter["neuron_dynamics"]["active_spike_count"], 0)
        self.assertGreaterEqual(adapter["neuron_dynamics"]["timestep_count"], 2)
        self.assertTrue(adapter["neuron_dynamics"]["adaptive_timesteps"])
        self.assertTrue(adapter["sparsity_evidence"]["meets_sparse_activation_floor"])
        self.assertFalse(adapter["promotion_constraints"]["eligible_for_language_generation"])

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
            "reference_for_marulho_owned_reimplementation",
        )
        self.assertIn("marulho_owned_language_neuron_module", surface["research_candidates"][0]["required_local_evidence"])
        self.assertIn("marulho_native_snn_decoder", surface["research_candidates"][0]["required_local_evidence"])
        self.assertTrue(surface["safety_invariants"]["requires_marulho_owned_implementation"])
        self.assertTrue(surface["safety_invariants"]["requires_marulho_controlled_training"])
        self.assertIn("MARULHO must own", " ".join(surface["limitations"]))
        self.assertTrue(surface["readiness_checks"]["grounded_language_surface_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_evidence_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_grounded"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_non_generative"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_readout_device_evidence_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_owned"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_non_generative"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_sparse"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_temporal_state"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_grounding_supported"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_available"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_owned"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_sparse"])
        self.assertTrue(surface["readiness_checks"]["marulho_spike_language_neuron_adapter_dynamic"])
        self.assertEqual(surface["current_spike_readout_evidence"]["surface"], "subcortical_spike_readout_evidence.v1")
        self.assertEqual(surface["current_spike_readout_evidence"]["device"], "cuda:0")
        self.assertTrue(surface["current_spike_readout_evidence"]["cuda_device_selected"])
        self.assertFalse(surface["current_spike_readout_evidence"]["generates_text"])
        self.assertEqual(
            surface["current_decoder_probe_evidence"]["surface"],
            "snn_language_decoder_probe_evidence.v1",
        )
        self.assertTrue(surface["current_decoder_probe_evidence"]["owned_by_marulho"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["executable"])
        self.assertTrue(surface["current_decoder_probe_evidence"]["dynamic_state_available"])
        self.assertEqual(
            surface["current_language_neuron_adapter_evidence"]["surface"],
            "snn_language_neuron_adapter_evidence.v1",
        )
        self.assertFalse(surface["current_language_neuron_adapter_evidence"]["generates_text"])
        self.assertFalse(surface["current_language_neuron_adapter_evidence"]["executable"])
        self.assertTrue(surface["current_language_neuron_adapter_evidence"]["adaptive_timesteps"])
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
        self.assertTrue(surface["readiness_checks"]["marulho_spike_decoder_probe_available"])
        self.assertFalse(surface["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(surface["safety_invariants"]["eligible_for_cognition_substrate"])

    def test_language_evaluation_surface_gates_adapter_without_generation(self) -> None:
        surface = build_snn_language_evaluation_surface(
            {
                "prediction_error_mean": 0.22,
                "prediction_error_max": 0.48,
                "predictive_confidence_mean": 0.62,
                "predictive_confidence_min": 0.44,
                "recent_concepts": ["spiking language"],
                "concept_candidates": [{"label": "spiking language", "observations": 5}],
            },
            {
                "cuda_first_runtime": {
                    "tensor_device": "cuda:0",
                    "subcortex_tensor_devices": {"competitive": {"prototypes_device": "cuda:0"}},
                }
            },
        )

        self.assertEqual(surface["artifact_kind"], "terminus_snn_language_adapter_evaluation_gate")
        self.assertEqual(surface["surface"], "snn_language_adapter_evaluation.v1")
        self.assertEqual(surface["endpoint"], "/terminus/snn-language-evaluation")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_isolated_adapter_evaluation")
        self.assertTrue(surface["promotion_gate"]["requires_operator_approval"])
        self.assertEqual(surface["evaluation_cases"][0]["target"], "spike_language_neuron_adapter")
        self.assertTrue(surface["evaluation_cases"][0]["ready_for_evaluation"])
        self.assertIn("post_evaluation_grounding_delta", surface["success_evidence"])
        self.assertIn("adapter_activation_sparsity_delta", surface["success_evidence"])

    def test_heldout_language_adapter_evaluator_reports_support_without_training(self) -> None:
        report = evaluate_spike_language_adapter_heldout(
            [
                [
                    {"label": "prediction error", "pressure_band": "high", "grounded": True},
                    {"label": "concept focus", "pressure_band": "medium", "grounded": True},
                ],
                [
                    {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
                    {"label": "unsupported drift", "pressure_band": "low", "grounded": False},
                ],
            ],
            {"device": "cpu", "source": "heldout_readout_fixture"},
        )

        self.assertEqual(report["artifact_kind"], "terminus_snn_language_adapter_heldout_evaluation")
        self.assertEqual(report["surface"], "snn_language_adapter_heldout_evaluation.v1")
        self.assertTrue(report["owned_by_marulho"])
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["trains"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(report["heldout_summary"]["case_count"], 2)
        self.assertEqual(report["heldout_summary"]["supported_case_count"], 2)
        self.assertGreaterEqual(report["heldout_summary"]["mean_grounded_fraction"], 0.5)
        self.assertGreater(report["adapter_delta"]["mean_active_spike_count"], 0.0)
        self.assertGreaterEqual(report["adapter_delta"]["min_activation_sparsity"], 0.85)
        self.assertEqual(report["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertFalse(report["promotion_gate"]["eligible_for_language_generation"])

    def test_language_training_readiness_gate_requires_heldout_and_rollback_without_training(self) -> None:
        heldout = evaluate_spike_language_adapter_heldout(
            [
                [
                    {"label": "prediction error", "pressure_band": "high", "grounded": True},
                    {"label": "concept focus", "pressure_band": "medium", "grounded": True},
                ]
            ],
            {"device": "cpu", "source": "training_readiness_fixture"},
        )
        surface = build_snn_language_training_readiness_surface(
            heldout,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-training"},
        )

        self.assertEqual(surface["artifact_kind"], "terminus_snn_language_training_readiness_gate")
        self.assertEqual(surface["surface"], "snn_language_training_readiness.v1")
        self.assertEqual(surface["endpoint"], "/terminus/snn-language-training/readiness")
        self.assertTrue(surface["advisory"])
        self.assertFalse(surface["executable"])
        self.assertFalse(surface["mutates_runtime_state"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_training"])
        self.assertTrue(surface["promotion_gate"]["eligible_for_training_loop_design"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(surface["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(surface["promotion_gate"]["status"], "ready_for_training_loop_design_review")
        self.assertEqual(surface["training_design_cases"][0]["target"], "marulho_owned_snn_language_trainer")
        self.assertIn("training_rule_design", surface["success_evidence"])

    def test_spike_language_trainer_dry_run_learns_sequence_evidence_without_weights(self) -> None:
        report = run_spike_language_trainer_dry_run(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            {"device": "cpu", "source": "trainer_dry_run_fixture"},
            learning_rate=0.12,
            epochs=3,
        )

        self.assertEqual(report["artifact_kind"], "terminus_snn_language_trainer_dry_run")
        self.assertEqual(report["surface"], "snn_language_trainer_dry_run.v1")
        self.assertTrue(report["owned_by_marulho"])
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["decodes_text"])
        self.assertFalse(report["trains_runtime_model"])
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(report["training_rule"]["rule"], "local_hebbian_outer_product_with_row_normalization")
        self.assertGreater(report["training_rule"]["training_transition_count"], 0)
        self.assertGreaterEqual(report["weight_evidence"]["weight_sparsity"], 0.85)
        self.assertGreater(report["validation_summary"]["mean_transition_support"], 0.0)
        self.assertFalse(report["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(report["promotion_gate"]["eligible_for_runtime_training"])

    def test_spike_language_trainer_evaluation_gates_dry_run_without_promotion(self) -> None:
        dry_run = run_spike_language_trainer_dry_run(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            {"device": "cpu", "source": "trainer_evaluation_fixture"},
        )
        evaluation = evaluate_spike_language_trainer_dry_run(
            dry_run,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-trainer-eval"},
        )

        self.assertEqual(evaluation["artifact_kind"], "terminus_snn_language_trainer_isolated_evaluation")
        self.assertEqual(evaluation["surface"], "snn_language_trainer_isolated_evaluation.v1")
        self.assertTrue(evaluation["owned_by_marulho"])
        self.assertFalse(evaluation["generates_text"])
        self.assertFalse(evaluation["trains_runtime_model"])
        self.assertFalse(evaluation["promotes_runtime_trainer"])
        self.assertFalse(evaluation["mutates_runtime_state"])
        self.assertGreater(evaluation["validation_summary"]["mean_transition_support"], 0.0)
        self.assertGreaterEqual(evaluation["weight_evidence"]["weight_sparsity"], 0.85)
        self.assertEqual(evaluation["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_runtime_training"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_trainer_promotion"])

    def test_spike_language_sequence_prediction_probe_returns_sparse_code_not_text(self) -> None:
        report = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "sequence_prediction_fixture"},
            top_k=4,
        )

        self.assertEqual(report["artifact_kind"], "terminus_snn_language_sequence_prediction_probe")
        self.assertEqual(report["surface"], "snn_language_sequence_prediction_probe.v1")
        self.assertTrue(report["owned_by_marulho"])
        self.assertFalse(report["generates_text"])
        self.assertFalse(report["decodes_text"])
        self.assertFalse(report["trains_runtime_model"])
        self.assertFalse(report["returns_trained_weights"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertEqual(report["prediction"]["top_k"], 4)
        self.assertEqual(len(report["prediction"]["predicted_sparse_indices"]), 4)
        self.assertGreater(report["prediction"]["support_strength"], 0.0)
        self.assertFalse(report["persistent_transition_evidence"]["influenced_prediction"])
        self.assertGreaterEqual(report["training_evidence"]["weight_sparsity"], 0.85)
        self.assertFalse(report["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(report["promotion_gate"]["eligible_for_cognition_substrate"])

    def test_spike_language_sequence_prediction_uses_persistent_transition_weights(self) -> None:
        baseline = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "persistent_transition_fixture"},
            top_k=4,
        )
        current_indices = baseline["current_sparse_code"]["active_indices"]
        self.assertGreater(len(current_indices), 0)
        target_index = (int(current_indices[0]) + 7) % 64
        influenced = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "persistent_transition_fixture"},
            top_k=4,
            persistent_transition_weights={f"{int(current_indices[0])}:{target_index}": 0.5},
        )

        self.assertEqual(
            influenced["persistent_transition_evidence"]["surface"],
            "snn_language_persistent_transition_evidence.v1",
        )
        self.assertTrue(influenced["persistent_transition_evidence"]["available"])
        self.assertTrue(influenced["persistent_transition_evidence"]["influenced_prediction"])
        self.assertGreater(influenced["persistent_transition_evidence"]["support_strength"], 0.0)
        self.assertIn(target_index, influenced["prediction"]["predicted_sparse_indices"])
        self.assertFalse(influenced["generates_text"])
        self.assertFalse(influenced["decodes_text"])
        self.assertFalse(influenced["mutates_runtime_state"])

    def test_snn_language_readout_draft_generates_bounded_grounded_text_from_sparse_prediction(self) -> None:
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        vocabulary = [
            {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
            {"label": "prediction error", "pressure_band": "high", "grounded": True},
        ]
        baseline = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            current,
            {"device": "cpu", "source": "readout_draft_fixture"},
            top_k=4,
        )
        current_index = baseline["current_sparse_code"]["active_indices"][0]
        target_probe = build_spike_language_decoder_probe(
            {
                "readout_slots": [vocabulary[0]],
                "device_evidence": {"device": "cpu", "source": "readout_draft_fixture"},
            }
        )
        target_index = target_probe["sparse_code_evidence"]["active_indices"][0]
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            current,
            {"device": "cpu", "source": "readout_draft_fixture"},
            top_k=4,
            persistent_transition_weights={f"{current_index}:{target_index}": 0.9},
        )
        transition_memory_evaluation = build_snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            [current, [vocabulary[0]]],
            {"sparse_transition_weights": {f"{current_index}:{target_index}": 0.9}},
            {"device": "cpu", "source": "readout_draft_fixture"},
            top_k=4,
        )

        draft = generate_snn_language_readout_draft(
            prediction,
            vocabulary,
            {"device": "cpu", "source": "readout_draft_fixture"},
            transition_memory_evaluation,
        )
        pending_evaluation_draft = generate_snn_language_readout_draft(
            prediction,
            vocabulary,
            {"device": "cpu", "source": "readout_draft_fixture"},
        )
        worsened_evaluation = dict(transition_memory_evaluation)
        worsened_evaluation["evaluation_summary"] = dict(transition_memory_evaluation["evaluation_summary"])
        worsened_evaluation["evaluation_summary"]["mean_mismatch_delta"] = -0.2
        worsened_evaluation["evaluation_summary"]["worsened_sequence_count"] = 1
        worsened_evaluation["promotion_gate"] = dict(transition_memory_evaluation["promotion_gate"])
        worsened_evaluation["promotion_gate"]["eligible_for_bounded_readout_generation_review"] = False
        blocked_worsened_draft = generate_snn_language_readout_draft(
            prediction,
            vocabulary,
            {"device": "cpu", "source": "readout_draft_fixture"},
            worsened_evaluation,
        )
        external_prediction = dict(prediction)
        external_prediction["external_dependency"] = True
        blocked_external_draft = generate_snn_language_readout_draft(
            external_prediction,
            vocabulary,
            {"device": "cpu", "source": "readout_draft_fixture"},
            transition_memory_evaluation,
        )
        emission = build_snn_language_readout_emission(draft)
        pending_emission = build_snn_language_readout_emission(pending_evaluation_draft)

        self.assertEqual(draft["artifact_kind"], "terminus_snn_language_readout_draft")
        self.assertEqual(draft["surface"], "snn_language_readout_draft.v1")
        self.assertTrue(draft["owned_by_marulho"])
        self.assertFalse(draft["external_dependency"])
        self.assertFalse(draft["loads_external_checkpoint"])
        self.assertTrue(draft["generates_text"])
        self.assertTrue(draft["decodes_text"])
        self.assertEqual(draft["generation_scope"], "bounded_grounded_readout_label_draft")
        self.assertFalse(draft["freeform_language_generation"])
        self.assertFalse(draft["mutates_runtime_state"])
        self.assertIn("memory pressure", draft["draft"]["text"])
        self.assertTrue(draft["persistent_transition_evidence"]["influenced_prediction"])
        self.assertTrue(draft["transition_memory_evaluation_evidence"]["review_ready"])
        self.assertTrue(draft["transition_memory_evaluation_evidence"]["provenance_match"])
        trajectory = draft["readout_trajectory_evidence"]
        self.assertEqual(
            trajectory["surface"],
            "snn_language_readout_trajectory_evidence.v1",
        )
        self.assertEqual(trajectory["trajectory_kind"], "draft")
        self.assertTrue(trajectory["bounded_output"])
        self.assertFalse(trajectory["freeform_language_generation"])
        self.assertFalse(trajectory["mutates_runtime_state"])
        self.assertTrue(
            trajectory["promotion_gate"][
                "eligible_for_bounded_snn_language_readout"
            ]
        )
        self.assertTrue(
            trajectory["promotion_gate"]["required_evidence"][
                "event_sparsity_preserved"
            ]
        )
        self.assertTrue(
            trajectory["promotion_gate"]["required_evidence"][
                "transition_memory_bound"
            ]
        )
        self.assertEqual(trajectory["device_evidence"]["tensor_device"], "cpu")
        self.assertEqual(
            draft["transition_memory_evaluation_evidence"]["prediction_hash"],
            prediction["provenance_evidence"]["prediction_hash"],
        )
        self.assertEqual(
            draft["transition_memory_evaluation_evidence"]["transition_memory_evaluation_hash"],
            transition_memory_evaluation["provenance_evidence"]["evaluation_hash"],
        )
        self.assertTrue(draft["promotion_gate"]["eligible_for_bounded_readout_generation"])
        self.assertTrue(
            draft["promotion_gate"]["required_evidence"]["transition_memory_prediction_evaluation_ready"]
        )
        self.assertFalse(draft["promotion_gate"]["eligible_for_freeform_language_generation"])
        self.assertFalse(draft["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertTrue(pending_evaluation_draft["generates_text"])
        self.assertEqual(
            pending_evaluation_draft["promotion_gate"]["status"],
            "collect_transition_memory_prediction_evaluation",
        )
        self.assertFalse(
            pending_evaluation_draft["promotion_gate"]["eligible_for_bounded_readout_generation"]
        )
        self.assertFalse(pending_evaluation_draft["transition_memory_evaluation_evidence"]["non_worsening"])
        self.assertFalse(
            blocked_worsened_draft["promotion_gate"]["eligible_for_bounded_readout_generation"]
        )
        self.assertEqual(
            blocked_worsened_draft["promotion_gate"]["status"],
            "blocked_transition_memory_prediction_evaluation",
        )
        self.assertFalse(
            blocked_worsened_draft["promotion_gate"]["required_evidence"][
                "transition_memory_prediction_non_worsening"
            ]
        )
        self.assertFalse(
            blocked_external_draft["promotion_gate"]["eligible_for_bounded_readout_generation"]
        )
        self.assertFalse(
            blocked_external_draft["promotion_gate"]["required_evidence"]["external_dependency_absent"]
        )
        self.assertEqual(emission["artifact_kind"], "terminus_snn_language_readout_emission")
        self.assertEqual(emission["surface"], "snn_language_readout_emission.v1")
        self.assertTrue(emission["ready"])
        self.assertTrue(emission["generates_text"])
        self.assertTrue(emission["decodes_text"])
        self.assertEqual(
            emission["generation_scope"],
            "operator_visible_bounded_snn_readout_emission",
        )
        self.assertFalse(emission["freeform_language_generation"])
        self.assertFalse(emission["mutates_runtime_state"])
        self.assertFalse(emission["applies_plasticity"])
        self.assertFalse(emission["writes_checkpoint"])
        self.assertFalse(emission["promotes_fact"])
        self.assertFalse(emission["promotes_action"])
        self.assertFalse(emission["cognition_substrate"])
        self.assertEqual(emission["language_output"]["text"], draft["draft"]["text"])
        self.assertIn("memory pressure", emission["language_output"]["text"])
        self.assertEqual(len(emission["emission_hash"]), 64)
        self.assertEqual(
            emission["emission_binding"]["transition_memory_evaluation_hash"],
            transition_memory_evaluation["provenance_evidence"]["evaluation_hash"],
        )
        self.assertEqual(
            emission["emission_binding"]["trajectory_hash"],
            draft["readout_trajectory_evidence"]["provenance_evidence"]["trajectory_hash"],
        )
        self.assertTrue(emission["promotion_gate"]["eligible_for_operator_display"])
        self.assertFalse(
            emission["promotion_gate"]["eligible_for_freeform_language_generation"]
        )
        self.assertFalse(emission["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(pending_emission["ready"])
        self.assertFalse(pending_emission["generates_text"])
        self.assertEqual(pending_emission["language_output"]["text"], "")
        self.assertFalse(
            pending_emission["promotion_gate"]["required_evidence"][
                "draft_bounded_generation_ready"
            ]
        )
        stale_evaluation = build_snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "unrelated focus", "pressure_band": "medium", "grounded": True}],
            ],
            [
                [{"label": "unrelated focus", "pressure_band": "medium", "grounded": True}],
                [vocabulary[0]],
            ],
            {"sparse_transition_weights": {f"{current_index}:{target_index}": 0.9}},
            {"device": "cpu", "source": "readout_draft_fixture"},
            top_k=4,
        )
        blocked_stale_evaluation_draft = generate_snn_language_readout_draft(
            prediction,
            vocabulary,
            {"device": "cpu", "source": "readout_draft_fixture"},
            stale_evaluation,
        )
        self.assertFalse(
            blocked_stale_evaluation_draft["transition_memory_evaluation_evidence"]["provenance_match"]
        )
        self.assertFalse(
            blocked_stale_evaluation_draft["promotion_gate"]["eligible_for_bounded_readout_generation"]
        )

    def test_snn_language_readout_rollout_candidate_uses_persistent_sparse_memory_without_mutation(self) -> None:
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        vocabulary = [
            {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
            {"label": "prediction error", "pressure_band": "high", "grounded": True},
            {"label": "ungrounded lure", "pressure_band": "high", "grounded": False},
        ]
        device = {"device": "cpu", "source": "readout_rollout_fixture"}
        current_probe = build_spike_language_decoder_probe({"readout_slots": current, "device_evidence": device})
        memory_probe = build_spike_language_decoder_probe({"readout_slots": [vocabulary[0]], "device_evidence": device})
        error_probe = build_spike_language_decoder_probe({"readout_slots": [vocabulary[1]], "device_evidence": device})
        current_index = current_probe["sparse_code_evidence"]["active_indices"][0]
        memory_index = memory_probe["sparse_code_evidence"]["active_indices"][0]
        error_index = error_probe["sparse_code_evidence"]["active_indices"][0]
        transition_weights = {
            f"{current_index}:{memory_index}": 0.9,
            f"{memory_index}:{error_index}": 0.8,
        }
        server_transition_state = {
            "sparse_transition_weights": transition_weights,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
            "current_state_revision": 7,
        }
        server_transition_state = {
            "sparse_transition_weights": transition_weights,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
            "current_state_revision": 7,
        }
        training_batches = [
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            current,
        ]
        prediction = predict_spike_language_sequence(
            training_batches,
            current,
            device,
            top_k=4,
            persistent_transition_weights=transition_weights,
        )
        transition_memory_evaluation = build_snn_language_transition_memory_prediction_evaluation(
            training_batches,
            [current, [vocabulary[0]], [vocabulary[1]]],
            {"sparse_transition_weights": transition_weights},
            device,
            top_k=4,
        )

        rollout = rollout_snn_language_readout_candidate(
            prediction,
            vocabulary,
            server_transition_state,
            device,
            transition_memory_evaluation,
            rollout_steps=3,
            top_k=4,
        )
        blocked_without_memory = rollout_snn_language_readout_candidate(
            prediction,
            vocabulary,
            {"sparse_transition_weights": {}},
            device,
            transition_memory_evaluation,
            rollout_steps=3,
            top_k=4,
        )
        blocked_bad_parameters = rollout_snn_language_readout_candidate(
            prediction,
            vocabulary,
            server_transition_state,
            device,
            transition_memory_evaluation,
            rollout_steps=13,
            top_k="wide",
        )
        blocked_caller_memory = rollout_snn_language_readout_candidate(
            prediction,
            vocabulary,
            {"sparse_transition_weights": transition_weights},
            device,
            transition_memory_evaluation,
            rollout_steps=3,
            top_k=4,
        )

        self.assertEqual(rollout["artifact_kind"], "terminus_snn_language_readout_rollout_candidate")
        self.assertEqual(rollout["surface"], "snn_language_readout_rollout_candidate.v1")
        self.assertTrue(rollout["owned_by_marulho"])
        self.assertFalse(rollout["external_dependency"])
        self.assertFalse(rollout["loads_external_checkpoint"])
        self.assertTrue(rollout["generates_text"])
        self.assertFalse(rollout["freeform_language_generation"])
        self.assertFalse(rollout["decodes_text"])
        self.assertFalse(rollout["trains_runtime_model"])
        self.assertFalse(rollout["returns_trained_weights"])
        self.assertFalse(rollout["applies_plasticity"])
        self.assertFalse(rollout["mutates_runtime_state"])
        self.assertIn("memory pressure", rollout["rollout"]["labels"])
        self.assertIn("prediction error", rollout["rollout"]["labels"])
        self.assertNotIn("ungrounded lure", rollout["rollout"]["labels"])
        self.assertEqual(rollout["readout_rollout_evidence"]["persistent_transition_weight_count"], 2)
        self.assertTrue(
            rollout["readout_rollout_evidence"]["server_transition_memory_hash_match"]
        )
        self.assertTrue(
            rollout["readout_rollout_evidence"][
                "caller_transition_memory_state_absent_or_ignored"
            ]
        )
        trajectory = rollout["readout_trajectory_evidence"]
        self.assertEqual(
            trajectory["surface"],
            "snn_language_readout_trajectory_evidence.v1",
        )
        self.assertEqual(trajectory["trajectory_kind"], "rollout")
        self.assertTrue(trajectory["trajectory"]["adaptive_timesteps"])
        self.assertTrue(trajectory["trajectory"]["event_sparse"])
        self.assertFalse(trajectory["freeform_language_generation"])
        self.assertFalse(trajectory["applies_plasticity"])
        self.assertFalse(trajectory["mutates_runtime_state"])
        self.assertEqual(
            trajectory["transition_memory_evidence"][
                "persistent_transition_weight_count"
            ],
            2,
        )
        self.assertEqual(trajectory["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(
            trajectory["promotion_gate"][
                "eligible_for_bounded_snn_language_readout"
            ]
        )
        self.assertTrue(rollout["promotion_gate"]["eligible_for_bounded_readout_rollout"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_freeform_language_generation"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_fact_promotion"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_action"])
        self.assertFalse(blocked_without_memory["available"])
        self.assertFalse(
            blocked_without_memory["promotion_gate"]["required_evidence"][
                "persistent_transition_memory_available"
            ]
        )
        self.assertFalse(
            blocked_bad_parameters["promotion_gate"][
                "eligible_for_bounded_readout_rollout"
            ]
        )
        self.assertFalse(
            blocked_bad_parameters["promotion_gate"]["required_evidence"][
                "bounded_parameters_accepted_as_is"
            ]
        )
        self.assertFalse(
            blocked_bad_parameters["readout_rollout_evidence"]["bounded_parameters"][
                "accepted_as_is"
            ]
        )
        self.assertFalse(
            blocked_bad_parameters["readout_rollout_evidence"]["bounded_parameters"][
                "top_k_parse_valid"
            ]
        )
        self.assertEqual(
            blocked_bad_parameters["readout_rollout_evidence"]["bounded_parameters"][
                "rollout_steps"
            ],
            13,
        )
        self.assertFalse(blocked_bad_parameters["generates_text"])
        self.assertEqual(blocked_bad_parameters["rollout"]["labels"], [])
        self.assertFalse(
            blocked_bad_parameters["readout_trajectory_evidence"]["promotion_gate"][
                "eligible_for_bounded_snn_language_readout"
            ]
        )
        self.assertFalse(
            blocked_caller_memory["promotion_gate"][
                "eligible_for_bounded_readout_rollout"
            ]
        )
        self.assertFalse(
            blocked_caller_memory["promotion_gate"]["required_evidence"][
                "server_transition_memory_hash_match"
            ]
        )
        self.assertFalse(
            blocked_caller_memory["promotion_gate"]["required_evidence"][
                "caller_transition_memory_state_absent_or_ignored"
            ]
        )

    def test_snn_language_readout_trajectory_evidence_fails_closed_without_memory_binding(self) -> None:
        evidence = build_snn_language_readout_trajectory_evidence(
            trajectory_kind="rollout",
            labels=["memory pressure"],
            sparse_steps=[
                {
                    "step_index": 0,
                    "predicted_sparse_indices": [1, 2, 3],
                    "support_strength": 0.9,
                }
            ],
            grounding_evidence={
                "grounded_fraction": 1.0,
                "grounded_rollout_label_count": 1,
            },
            transition_memory_evidence={
                "persistent_transition_weight_count": 0,
            },
            device_evidence={
                "requested_device": "cpu",
                "tensor_device": "cpu",
                "cuda_tensor": False,
                "device_source": "trajectory_fixture",
            },
        )

        self.assertEqual(
            evidence["surface"],
            "snn_language_readout_trajectory_evidence.v1",
        )
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["bounded_output"])
        self.assertFalse(evidence["freeform_language_generation"])
        self.assertFalse(evidence["mutates_runtime_state"])
        self.assertFalse(
            evidence["promotion_gate"][
                "eligible_for_bounded_snn_language_readout"
            ]
        )
        self.assertFalse(
            evidence["promotion_gate"]["required_evidence"][
                "transition_memory_bound"
            ]
        )
        self.assertEqual(evidence["promotion_gate"]["status"], "collect_sparse_readout_trajectory_evidence")

    def test_snn_language_readout_rollout_replay_evaluation_builds_review_targets_without_mutation(self) -> None:
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        vocabulary = [
            {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
            {"label": "prediction error", "pressure_band": "high", "grounded": True},
            {"label": "ungrounded lure", "pressure_band": "high", "grounded": False},
        ]
        device = {"device": "cpu", "source": "readout_rollout_replay_fixture"}
        current_probe = build_spike_language_decoder_probe({"readout_slots": current, "device_evidence": device})
        memory_probe = build_spike_language_decoder_probe({"readout_slots": [vocabulary[0]], "device_evidence": device})
        error_probe = build_spike_language_decoder_probe({"readout_slots": [vocabulary[1]], "device_evidence": device})
        current_index = current_probe["sparse_code_evidence"]["active_indices"][0]
        memory_index = memory_probe["sparse_code_evidence"]["active_indices"][0]
        error_index = error_probe["sparse_code_evidence"]["active_indices"][0]
        transition_weights = {
            f"{current_index}:{memory_index}": 0.9,
            f"{memory_index}:{error_index}": 0.8,
        }
        server_transition_state = {
            "sparse_transition_weights": transition_weights,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
            "current_state_revision": 7,
        }
        training_batches = [
            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
            current,
        ]
        prediction = predict_spike_language_sequence(
            training_batches,
            current,
            device,
            top_k=4,
            persistent_transition_weights=transition_weights,
        )
        transition_memory_evaluation = build_snn_language_transition_memory_prediction_evaluation(
            training_batches,
            [current, [vocabulary[0]], [vocabulary[1]]],
            {"sparse_transition_weights": transition_weights},
            device,
            top_k=4,
        )
        rollout = rollout_snn_language_readout_candidate(
            prediction,
            vocabulary,
            server_transition_state,
            device,
            transition_memory_evaluation,
            rollout_steps=3,
            top_k=4,
        )

        evaluation = evaluate_snn_language_readout_rollout_replay(
            rollout,
            candidate_limit=4,
            device_evidence=device,
        )
        repeat = evaluate_snn_language_readout_rollout_replay(
            rollout,
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_external = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "external_dependency": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_checkpoint = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "loads_external_checkpoint": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_freeform = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "freeform_language_generation": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_mutating = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "mutates_runtime_state": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_plasticity = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "applies_plasticity": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_training = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "trains_runtime_model": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_weights = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "returns_trained_weights": True},
            candidate_limit=4,
            device_evidence=device,
        )
        blocked_decoding = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "decodes_text": True},
            candidate_limit=4,
            device_evidence=device,
        )
        tampered_trace = [dict(item) for item in rollout["rollout_trace"]]
        tampered_trace[0]["selected_label"] = "prediction error"
        blocked_hash_tamper = evaluate_snn_language_readout_rollout_replay(
            {**rollout, "rollout_trace": tampered_trace},
            candidate_limit=4,
            device_evidence=device,
        )

        self.assertEqual(
            evaluation["artifact_kind"],
            "terminus_snn_language_readout_rollout_replay_evaluation",
        )
        self.assertEqual(evaluation["surface"], "snn_language_readout_rollout_replay_evaluation.v1")
        self.assertTrue(evaluation["owned_by_marulho"])
        self.assertFalse(evaluation["external_dependency"])
        self.assertFalse(evaluation["loads_external_checkpoint"])
        self.assertFalse(evaluation["generates_text"])
        self.assertFalse(evaluation["freeform_language_generation"])
        self.assertFalse(evaluation["decodes_text"])
        self.assertFalse(evaluation["trains_runtime_model"])
        self.assertFalse(evaluation["applies_plasticity"])
        self.assertFalse(evaluation["mutates_runtime_state"])
        self.assertFalse(evaluation["recorded_in_ledger"])
        self.assertFalse(evaluation["eligible_for_replay_priority"])
        self.assertTrue(evaluation["promotion_gate"]["eligible_for_readout_rollout_ledger_recording_review"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_replay_priority"])
        self.assertEqual(
            evaluation["promotion_gate"]["next_gate"],
            "operator_review_record_snn_language_readout_rollout_evidence",
        )
        targets = evaluation["replay_evaluation"]["replay_targets"]
        self.assertGreaterEqual(len(targets), 1)
        self.assertTrue(all(target["grounded"] for target in targets))
        self.assertTrue(all(target["active_indices_hash_valid"] for target in targets))
        self.assertIn("memory pressure", [target["selected_label"] for target in targets])
        self.assertEqual(
            evaluation["provenance_evidence"]["rollout_replay_evaluation_hash"],
            repeat["provenance_evidence"]["rollout_replay_evaluation_hash"],
        )
        self.assertEqual(
            evaluation["provenance_evidence"]["server_transition_memory_hash"],
            rollout["readout_rollout_evidence"]["server_transition_memory_hash"],
        )
        self.assertTrue(evaluation["provenance_evidence"]["server_transition_memory_hash_match"])
        self.assertEqual(
            evaluation["provenance_evidence"]["transition_memory_state_source"],
            "service.runtime_facade.snn_language_plasticity_runtime_state",
        )
        self.assertTrue(
            evaluation["promotion_gate"]["required_evidence"][
                "server_transition_memory_hash_available"
            ]
        )
        self.assertTrue(
            evaluation["promotion_gate"]["required_evidence"][
                "server_transition_memory_state_source_bound"
            ]
        )
        self.assertFalse(
            blocked_external["promotion_gate"]["required_evidence"]["external_dependency_absent"]
        )
        self.assertFalse(
            blocked_external["promotion_gate"]["eligible_for_readout_rollout_ledger_recording_review"]
        )
        self.assertFalse(
            blocked_checkpoint["promotion_gate"]["required_evidence"][
                "external_checkpoint_absent"
            ]
        )
        self.assertFalse(
            blocked_freeform["promotion_gate"]["required_evidence"][
                "freeform_generation_absent"
            ]
        )
        self.assertFalse(
            blocked_mutating["promotion_gate"]["required_evidence"][
                "runtime_mutation_absent"
            ]
        )
        self.assertFalse(
            blocked_plasticity["promotion_gate"]["required_evidence"][
                "plasticity_absent"
            ]
        )
        self.assertFalse(
            blocked_training["promotion_gate"]["required_evidence"][
                "training_absent"
            ]
        )
        self.assertFalse(
            blocked_weights["promotion_gate"]["required_evidence"][
                "trained_weights_absent"
            ]
        )
        self.assertFalse(
            blocked_decoding["promotion_gate"]["required_evidence"][
                "text_decoding_absent"
            ]
        )
        self.assertFalse(
            blocked_hash_tamper["promotion_gate"]["required_evidence"][
                "rollout_hash_valid"
            ]
        )

    def test_transition_memory_prediction_evaluation_compares_memory_against_baseline_without_generation(self) -> None:
        current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
        observed = [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}]
        current_probe = build_spike_language_decoder_probe(
            {"readout_slots": current, "device_evidence": {"device": "cpu", "source": "prediction_eval_fixture"}}
        )
        observed_probe = build_spike_language_decoder_probe(
            {"readout_slots": observed, "device_evidence": {"device": "cpu", "source": "prediction_eval_fixture"}}
        )
        current_index = current_probe["sparse_code_evidence"]["active_indices"][0]
        observed_index = observed_probe["sparse_code_evidence"]["active_indices"][0]

        evaluation = build_snn_language_transition_memory_prediction_evaluation(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ],
            [current, observed],
            {"sparse_transition_weights": {f"{current_index}:{observed_index}": 0.9}},
            {"device": "cpu", "source": "prediction_eval_fixture"},
            top_k=4,
        )

        self.assertEqual(
            evaluation["artifact_kind"],
            "terminus_snn_language_transition_memory_prediction_evaluation",
        )
        self.assertEqual(evaluation["surface"], "snn_language_transition_memory_prediction_evaluation.v1")
        self.assertTrue(evaluation["owned_by_marulho"])
        self.assertFalse(evaluation["generates_text"])
        self.assertFalse(evaluation["decodes_text"])
        self.assertFalse(evaluation["trains_runtime_model"])
        self.assertFalse(evaluation["applies_plasticity"])
        self.assertFalse(evaluation["mutates_runtime_state"])
        summary = evaluation["evaluation_summary"]
        self.assertEqual(summary["evaluation_pair_count"], 1)
        self.assertEqual(summary["persistent_transition_weight_count"], 1)
        self.assertGreaterEqual(summary["mean_mismatch_delta"], 0.0)
        self.assertGreaterEqual(summary["influenced_prediction_count"], 1)
        self.assertEqual(summary["worsened_sequence_count"], 0)
        self.assertEqual(evaluation["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_cognition_substrate"])

    def test_transition_memory_sleep_policy_recommends_operator_review_without_mutation(self) -> None:
        policy = build_snn_language_transition_memory_sleep_policy(
            {
                "sparse_transition_weight_count": 4,
                "homeostatic_maintenance_count": 1,
            },
            subcortex_sleep_pressure={"pressure": 0.8, "source": "living_loop.subcortex_sleep_pressure"},
            replay_evidence={
                "available": True,
                "ready": True,
                "source": "replay_controller",
                "replay_window_id": "replay-window-1",
                "evidence_hash": "sha256:replay-window-1",
            },
        )

        self.assertEqual(policy["surface"], "snn_language_transition_memory_sleep_policy.v1")
        self.assertFalse(policy["generates_text"])
        self.assertFalse(policy["decodes_text"])
        self.assertFalse(policy["applies_plasticity"])
        self.assertFalse(policy["mutates_runtime_state"])
        self.assertTrue(policy["recommendation"]["recommended"])
        self.assertFalse(policy["recommendation"]["executable"])
        self.assertEqual(
            policy["recommendation"]["suggested_endpoint"],
            "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
        )
        self.assertFalse(policy["subcortex_sleep_pressure"]["retired_runtime_dependency"])

    def test_transition_memory_sleep_policy_routes_replay_bound_rollout_growth(self) -> None:
        permit_policy = build_snn_language_transition_memory_sleep_policy(
            {"sparse_transition_weight_count": 2, "homeostatic_maintenance_count": 0},
            subcortex_sleep_pressure={"pressure": 0.2},
            rollout_regeneration_evidence={
                "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
                "promotion_gate": {"eligible_for_regeneration_permit_request": True},
            },
            readout_ledger_evidence={"summary": {"event_count": 1, "rollout_event_count": 1}},
        )
        application_policy = build_snn_language_transition_memory_sleep_policy(
            {"sparse_transition_weight_count": 2, "homeostatic_maintenance_count": 0},
            rollout_regeneration_evidence={
                "surface": "snn_language_readout_rollout_regeneration_application_preflight.v1",
                "ready": True,
                "promotion_gate": {
                    "eligible_for_checkpoint_backed_regeneration_executor": True,
                },
            },
        )
        maintenance_policy = build_snn_language_transition_memory_sleep_policy(
            {
                "sparse_transition_weight_count": 3,
                "homeostatic_maintenance_count": 0,
                "regeneration_count": 1,
                "regenerated_synapse_count_total": 1,
            },
            subcortex_sleep_pressure={"pressure": 0.7},
            rollout_regeneration_evidence={
                "surface": "snn_language_readout_rollout_regeneration_application.v1",
                "accepted": True,
                "mutates_runtime_state": True,
            },
        )

        self.assertEqual(
            permit_policy["recommendation"]["action"],
            "review_rollout_regeneration_permit_request",
        )
        self.assertEqual(
            permit_policy["recommendation"]["suggested_endpoint"],
            "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request",
        )
        self.assertTrue(
            permit_policy["rollout_regeneration_evidence"]["replay_artifact_review_ready"]
        )
        self.assertFalse(permit_policy["mutates_runtime_state"])
        self.assertEqual(
            application_policy["recommendation"]["action"],
            "review_rollout_regeneration_application",
        )
        self.assertEqual(
            application_policy["recommendation"]["suggested_endpoint"],
            "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application",
        )
        self.assertTrue(
            application_policy["rollout_regeneration_evidence"]["application_preflight_ready"]
        )
        self.assertEqual(
            maintenance_policy["recommendation"]["action"],
            "review_transition_memory_homeostatic_maintenance",
        )
        self.assertIn(
            "post_growth_homeostatic_maintenance_due",
            maintenance_policy["recommendation"]["reason_codes"],
        )
        self.assertFalse(maintenance_policy["applies_plasticity"])
        self.assertFalse(maintenance_policy["mutates_runtime_state"])

    def test_transition_memory_regeneration_proposal_requires_replay_backed_local_mismatch(self) -> None:
        mismatch = {
            "available": True,
            "surface": "snn_language_sequence_mismatch_probe.v1",
            "prediction_error": {"mismatch_score": 0.9},
            "sparse_code_delta": {
                "predicted_only_indices": [3],
                "observed_only_indices": [4, 12],
            },
        }
        proposal = build_snn_language_transition_memory_regeneration_proposal(
            mismatch,
            {"surface": "snn_language_plasticity_runtime_state.v1"},
            replay_evidence={
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "artifact_kind": "terminus_snn_language_transition_memory_regeneration_permit",
                "source": "replay_controller.regeneration_permit",
                "permit_id": "permit-1",
                "replay_window_id": "replay-window-1",
                "evidence_hash": "sha256:replay-window-1",
                "mismatch_hash": hashlib.sha256(
                    json.dumps(mismatch, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            },
            locality_radius=2,
            initial_weight=0.02,
        )

        self.assertEqual(proposal["surface"], "snn_language_transition_memory_regeneration_proposal.v1")
        self.assertFalse(proposal["applies_plasticity"])
        self.assertFalse(proposal["mutates_runtime_state"])
        self.assertEqual(proposal["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertEqual(proposal["regeneration_design"]["candidate_count"], 1)
        self.assertEqual(proposal["regeneration_design"]["candidate_synapses"][0]["synapse"], "3:4")

    def test_spike_language_sequence_mismatch_probe_reports_prediction_error_without_learning(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "mismatch_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "mismatch_fixture"},
        )

        self.assertEqual(mismatch["artifact_kind"], "terminus_snn_language_sequence_mismatch_probe")
        self.assertEqual(mismatch["surface"], "snn_language_sequence_mismatch_probe.v1")
        self.assertTrue(mismatch["owned_by_marulho"])
        self.assertFalse(mismatch["generates_text"])
        self.assertFalse(mismatch["decodes_text"])
        self.assertFalse(mismatch["trains_runtime_model"])
        self.assertFalse(mismatch["returns_trained_weights"])
        self.assertFalse(mismatch["mutates_runtime_state"])
        self.assertIn(mismatch["prediction_error"]["prediction_error_band"], {"low", "medium", "high"})
        self.assertGreaterEqual(mismatch["prediction_error"]["mismatch_score"], 0.0)
        self.assertFalse(mismatch["promotion_gate"]["eligible_for_learning_signal"])
        self.assertFalse(mismatch["promotion_gate"]["eligible_for_language_generation"])

    def test_spike_language_plasticity_pressure_gates_mismatch_without_learning(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_pressure_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_pressure_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(pressure["artifact_kind"], "terminus_snn_language_plasticity_pressure_gate")
        self.assertEqual(pressure["surface"], "snn_language_plasticity_pressure.v1")
        self.assertTrue(pressure["owned_by_marulho"])
        self.assertFalse(pressure["generates_text"])
        self.assertFalse(pressure["decodes_text"])
        self.assertFalse(pressure["trains_runtime_model"])
        self.assertFalse(pressure["applies_plasticity"])
        self.assertFalse(pressure["mutates_runtime_state"])
        self.assertIn(pressure["plasticity_pressure"]["pressure_band"], {"medium", "high"})
        self.assertEqual(pressure["candidate_update"]["target"], "local_snn_language_sequence_transition_weights")
        self.assertFalse(pressure["promotion_gate"]["eligible_for_learning_signal"])
        self.assertFalse(pressure["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertTrue(pressure["promotion_gate"]["eligible_for_plasticity_design_review"])

    def test_spike_language_plasticity_trial_simulates_update_without_applying_it(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_trial_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_trial_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = run_spike_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(trial["artifact_kind"], "terminus_snn_language_plasticity_trial")
        self.assertEqual(trial["surface"], "snn_language_plasticity_trial.v1")
        self.assertFalse(trial["generates_text"])
        self.assertFalse(trial["decodes_text"])
        self.assertFalse(trial["trains_runtime_model"])
        self.assertFalse(trial["applies_plasticity"])
        self.assertFalse(trial["returns_trained_weights"])
        self.assertFalse(trial["mutates_runtime_state"])
        self.assertGreater(trial["trial_summary"]["expected_pressure_reduction"], 0.0)
        self.assertLessEqual(
            trial["trial_summary"]["post_pressure_score"],
            trial["trial_summary"]["pre_pressure_score"],
        )
        self.assertFalse(trial["ephemeral_update"]["weights_persisted"])
        self.assertFalse(trial["ephemeral_update"]["runtime_update_applied"])
        self.assertFalse(trial["promotion_gate"]["eligible_for_plasticity_application"])

    def test_spike_language_plasticity_replay_evaluation_gates_trial_without_promotion(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_replay_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_replay_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = run_spike_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = evaluate_spike_language_plasticity_replay(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(replay["artifact_kind"], "terminus_snn_language_plasticity_replay_evaluation")
        self.assertEqual(replay["surface"], "snn_language_plasticity_replay_evaluation.v1")
        self.assertFalse(replay["generates_text"])
        self.assertFalse(replay["decodes_text"])
        self.assertFalse(replay["trains_runtime_model"])
        self.assertFalse(replay["applies_plasticity"])
        self.assertFalse(replay["mutates_runtime_state"])
        self.assertGreater(replay["replay_evidence"]["expected_pressure_reduction"], 0.0)
        self.assertTrue(replay["replay_evidence"]["pressure_non_worsening"])
        self.assertFalse(replay["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(replay["promotion_gate"]["eligible_for_replay_promotion"])
        self.assertTrue(replay["promotion_gate"]["eligible_for_operator_replay_review"])

    def test_spike_language_plasticity_replay_experiment_stays_ephemeral(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_replay_experiment_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_replay_experiment_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = run_spike_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = evaluate_spike_language_plasticity_replay(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        experiment = run_spike_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(experiment["artifact_kind"], "terminus_snn_language_plasticity_replay_experiment")
        self.assertEqual(experiment["surface"], "snn_language_plasticity_replay_experiment.v1")
        self.assertFalse(experiment["generates_text"])
        self.assertFalse(experiment["decodes_text"])
        self.assertFalse(experiment["trains_runtime_model"])
        self.assertFalse(experiment["applies_plasticity"])
        self.assertFalse(experiment["mutates_runtime_state"])
        self.assertFalse(experiment["returns_trained_weights"])
        self.assertTrue(experiment["replay_experiment"]["pressure_stable_after_replay"])
        self.assertEqual(experiment["replay_experiment"]["replay_sequence_count"], 1)
        self.assertFalse(experiment["ephemeral_replay"]["weights_persisted"])
        self.assertFalse(experiment["ephemeral_replay"]["runtime_update_applied"])
        self.assertFalse(experiment["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(experiment["promotion_gate"]["eligible_for_replay_promotion"])
        self.assertTrue(experiment["promotion_gate"]["eligible_for_operator_application_review"])

    def test_spike_language_plasticity_application_design_does_not_apply_learning(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_application_design_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_application_design_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = run_spike_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = evaluate_spike_language_plasticity_replay(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        experiment = run_spike_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        design = build_spike_language_plasticity_application_design(
            experiment,
            application_policy={
                "learning_rate": 0.03,
                "max_weight_delta": 0.04,
                "locality_radius": 2,
                "normalization": True,
                "local_only": True,
            },
            device_evidence={"device": "cpu", "source": "plasticity_application_design_fixture"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        blocked_without_device = build_spike_language_plasticity_application_design(
            experiment,
            application_policy={
                "learning_rate": 0.03,
                "max_weight_delta": 0.04,
                "locality_radius": 2,
                "normalization": True,
                "local_only": True,
            },
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(design["artifact_kind"], "terminus_snn_language_plasticity_application_design")
        self.assertEqual(design["surface"], "snn_language_plasticity_application_design.v1")
        self.assertFalse(design["generates_text"])
        self.assertFalse(design["decodes_text"])
        self.assertFalse(design["trains_runtime_model"])
        self.assertFalse(design["applies_plasticity"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["returns_trained_weights"])
        self.assertEqual(design["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(design["device_evidence"]["cuda_tensor"])
        self.assertTrue(design["device_evidence"]["device_report_available"])
        self.assertEqual(design["application_design"]["learning_rate"], 0.03)
        self.assertEqual(design["application_design"]["locality_radius"], 2)
        self.assertFalse(design["application_design"]["runtime_update_applied"])
        self.assertFalse(design["application_design"]["weights_persisted"])
        self.assertFalse(design["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(design["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(design["promotion_gate"]["eligible_for_operator_application_review"])
        self.assertEqual(blocked_without_device["promotion_gate"]["status"], "blocked_missing_application_design_evidence")
        self.assertFalse(
            blocked_without_device["promotion_gate"]["required_evidence"]["device_evidence_available"]
        )
        self.assertFalse(blocked_without_device["promotion_gate"]["eligible_for_operator_application_review"])

    def test_spike_language_plasticity_shadow_application_verifies_without_mutation(self) -> None:
        prediction = predict_spike_language_sequence(
            [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
            ],
            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
            {"device": "cpu", "source": "plasticity_shadow_application_fixture"},
            top_k=4,
        )
        mismatch = evaluate_spike_language_sequence_mismatch(
            prediction,
            [{"label": "novel mismatch", "pressure_band": "high", "grounded": True}],
            {"device": "cpu", "source": "plasticity_shadow_application_fixture"},
        )
        pressure = build_spike_language_plasticity_pressure(
            mismatch,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        trial = run_spike_language_plasticity_trial(
            pressure,
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        replay = evaluate_spike_language_plasticity_replay(
            trial,
            replay_window=[{"case_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        experiment = run_spike_language_plasticity_replay_experiment(
            replay,
            replay_sequences=[{"sequence_id": "sequence-replay-1", "grounded": True}],
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        design = build_spike_language_plasticity_application_design(
            experiment,
            application_policy={"learning_rate": 0.03, "max_weight_delta": 0.04, "locality_radius": 2},
            device_evidence={"device": "cpu", "source": "plasticity_shadow_application_fixture"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        measured_delta = build_spike_language_plasticity_shadow_delta(
            design,
            [
                {
                    "pre_indices": [2, 3],
                    "post_indices": [3, 4],
                    "grounded": True,
                    "readout_evidence_hash": "readout-evidence-1",
                    "prediction_hash": "prediction-1",
                    "transition_memory_evaluation_hash": "evaluation-1",
                    "persistent_transition_weights_hash": "weights-1",
                }
            ],
            device_evidence={"device": "cpu", "source": "plasticity_shadow_application_fixture"},
        )
        shadow = evaluate_spike_language_plasticity_shadow_application(
            design,
            shadow_delta=measured_delta,
            device_evidence={"device": "cpu", "source": "plasticity_shadow_application_fixture"},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        live_readiness = evaluate_spike_language_plasticity_live_application_readiness(
            shadow,
            rollback_readiness={
                "checkpoint_available": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_endpoint_available": True,
            },
            operator_approval={
                "approved": True,
                "operator_id": "operator-test",
                "approval_id": "approval-1",
            },
        )
        live_readiness_without_operator = evaluate_spike_language_plasticity_live_application_readiness(
            shadow,
            rollback_readiness={
                "checkpoint_available": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_endpoint_available": True,
            },
        )
        preflight = evaluate_spike_language_plasticity_live_application_preflight(
            live_readiness,
            application_target={
                "available": True,
                "target_id": "marulho.snn_language.sparse_transition_weights",
                "owned_by_marulho": True,
                "mutable": True,
                "sparse": True,
                "checkpointed": True,
            },
            checkpoint_transaction={
                "pre_update_checkpoint_saved": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_verified": True,
                "records_shadow_delta": True,
            },
        )
        preflight_without_target = evaluate_spike_language_plasticity_live_application_preflight(
            live_readiness,
            checkpoint_transaction={
                "pre_update_checkpoint_saved": True,
                "checkpoint_path": "checkpoint://pre-language-plasticity",
                "restore_verified": True,
                "records_shadow_delta": True,
            },
        )
        design_without_device = build_spike_language_plasticity_application_design(
            experiment,
            application_policy={"learning_rate": 0.03, "max_weight_delta": 0.04, "locality_radius": 2},
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )
        shadow_without_device = evaluate_spike_language_plasticity_shadow_application(
            design_without_device,
            shadow_delta={
                "max_abs_weight_delta": 0.03,
                "affected_synapse_count": 4,
                "locality_radius": 2,
                "pressure_before": 0.4,
                "pressure_after": 0.35,
            },
            runtime_truth_delta={"improved_or_stable": True},
            rollback_policy={"available": True, "snapshot_id": "pre-language-plasticity"},
        )

        self.assertEqual(measured_delta["artifact_kind"], "terminus_snn_language_plasticity_shadow_delta")
        self.assertEqual(measured_delta["surface"], "snn_language_plasticity_shadow_delta.v1")
        self.assertTrue(measured_delta["available"])
        self.assertEqual(measured_delta["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(measured_delta["applies_plasticity"])
        self.assertFalse(measured_delta["mutates_runtime_state"])
        self.assertGreater(measured_delta["affected_synapse_count"], 0)
        self.assertEqual(measured_delta["bounded_synapses"][0]["readout_evidence_hash"], "readout-evidence-1")
        self.assertEqual(measured_delta["bounded_synapses"][0]["source_pre_indices"], [2, 3])
        self.assertLessEqual(measured_delta["max_abs_weight_delta"], 0.04)
        self.assertEqual(shadow["artifact_kind"], "terminus_snn_language_plasticity_shadow_application")
        self.assertEqual(shadow["surface"], "snn_language_plasticity_shadow_application.v1")
        self.assertFalse(shadow["generates_text"])
        self.assertFalse(shadow["decodes_text"])
        self.assertFalse(shadow["trains_runtime_model"])
        self.assertFalse(shadow["applies_plasticity"])
        self.assertFalse(shadow["mutates_runtime_state"])
        self.assertFalse(shadow["returns_trained_weights"])
        self.assertEqual(shadow["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(shadow["device_evidence"]["cuda_tensor"])
        self.assertTrue(shadow["device_evidence"]["device_report_available"])
        self.assertTrue(shadow["shadow_application"]["pressure_non_worsening"])
        self.assertFalse(shadow["shadow_application"]["runtime_update_applied"])
        self.assertFalse(shadow["shadow_application"]["weights_persisted"])
        self.assertFalse(shadow["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(shadow["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(shadow["promotion_gate"]["eligible_for_operator_live_application_review"])
        self.assertEqual(
            live_readiness["artifact_kind"],
            "terminus_snn_language_plasticity_live_application_readiness",
        )
        self.assertEqual(live_readiness["surface"], "snn_language_plasticity_live_application_readiness.v1")
        self.assertFalse(live_readiness["generates_text"])
        self.assertFalse(live_readiness["decodes_text"])
        self.assertFalse(live_readiness["trains_runtime_model"])
        self.assertFalse(live_readiness["applies_plasticity"])
        self.assertFalse(live_readiness["mutates_runtime_state"])
        self.assertFalse(live_readiness["returns_trained_weights"])
        self.assertTrue(live_readiness["rollback_readiness"]["checkpoint_available"])
        self.assertTrue(live_readiness["rollback_readiness"]["restore_endpoint_available"])
        self.assertTrue(live_readiness["operator_approval"]["approved"])
        self.assertEqual(live_readiness["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertFalse(live_readiness["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(live_readiness["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(live_readiness["promotion_gate"]["eligible_for_operator_live_application_review"])
        self.assertEqual(
            live_readiness_without_operator["promotion_gate"]["status"],
            "blocked_missing_live_application_readiness",
        )
        self.assertFalse(
            live_readiness_without_operator["promotion_gate"]["required_evidence"]["operator_approval_available"]
        )
        self.assertEqual(preflight["artifact_kind"], "terminus_snn_language_plasticity_live_application_preflight")
        self.assertEqual(preflight["surface"], "snn_language_plasticity_live_application_preflight.v1")
        self.assertFalse(preflight["applies_plasticity"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["returns_trained_weights"])
        self.assertEqual(preflight["promotion_gate"]["status"], "ready_for_operator_execution_review")
        self.assertFalse(preflight["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(preflight["promotion_gate"]["eligible_for_operator_execution_review"])
        self.assertEqual(
            preflight_without_target["promotion_gate"]["status"],
            "blocked_missing_live_application_preflight",
        )
        self.assertFalse(
            preflight_without_target["promotion_gate"]["required_evidence"]["application_target_available"]
        )
        self.assertEqual(
            shadow_without_device["promotion_gate"]["status"],
            "blocked_missing_shadow_application_evidence",
        )
        self.assertFalse(shadow_without_device["promotion_gate"]["required_evidence"]["device_evidence_available"])
        self.assertFalse(shadow_without_device["promotion_gate"]["eligible_for_operator_live_application_review"])


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

    def test_structural_plasticity_isolated_evaluator_reports_deltas_without_mutation(self) -> None:
        pre_snapshot = {
            "current_state_revision": 10,
            "binding_topology": {
                "edges_added_total": 4,
                "edges_removed_total": 1,
                "growth_events": 1,
                "prune_events": 0,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.20, "saturated_fraction": 0.05, "stale_fraction": 0.30},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 11,
            "binding_topology": {
                "edges_added_total": 5,
                "edges_removed_total": 2,
                "growth_events": 2,
                "prune_events": 1,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.10, "saturated_fraction": 0.02, "stale_fraction": 0.20},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()

        report = evaluate_subcortical_structural_plasticity_isolated(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-structural-eval",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )

        self.assertEqual(report["artifact_kind"], "terminus_subcortical_structural_plasticity_isolated_evaluation")
        self.assertEqual(report["surface"], "subcortical_structural_plasticity_isolated_evaluation.v1")
        self.assertFalse(report["executable"])
        self.assertFalse(report["mutates_runtime_state"])
        self.assertFalse(report["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertEqual(report["structural_delta"]["edges_added_delta"], 1)
        self.assertEqual(report["structural_delta"]["edges_removed_delta"], 1)
        self.assertTrue(report["spike_health_delta"]["improved_or_stable"])
        self.assertTrue(report["runtime_truth_delta"]["improved_or_stable"])
        self.assertTrue(report["rollback_evidence"]["available"])
        self.assertTrue(report["rollback_evidence"]["bound_to_pre_snapshot"])
        self.assertTrue(report["rollback_evidence"]["pre_snapshot_hash_match"])
        self.assertEqual(report["snapshot_binding"]["hash_algorithm"], "sha256_canonical_json")
        self.assertEqual(len(report["snapshot_binding"]["pre_snapshot_hash"]), 64)
        self.assertEqual(len(report["snapshot_binding"]["post_snapshot_hash"]), 64)
        self.assertTrue(report["snapshot_binding"]["snapshot_hashes_distinct"])
        self.assertTrue(report["snapshot_binding"]["structural_delta_present"])
        self.assertEqual(report["snapshot_binding"]["pre_state_revision"], 10)
        self.assertEqual(report["snapshot_binding"]["post_state_revision"], 11)
        self.assertTrue(report["snapshot_binding"]["state_revision_order_valid"])
        self.assertFalse(report["snapshot_binding"]["raw_snapshots_exposed"])
        self.assertIn("pre_post_snapshot_hash_binding", report["success_evidence"])
        self.assertTrue(report["promotion_gate"]["requires_bound_snapshot_hashes"])
        self.assertTrue(report["promotion_gate"]["requires_nonzero_structural_delta"])
        self.assertTrue(report["promotion_gate"]["requires_rollback_pre_snapshot_binding"])
        self.assertEqual(report["promotion_gate"]["status"], "ready_for_operator_review")

    def test_structural_plasticity_isolated_evaluator_blocks_identical_snapshots(self) -> None:
        snapshot = {
            "current_state_revision": 12,
            "binding_topology": {
                "edges_added_total": 4,
                "edges_removed_total": 1,
                "growth_events": 1,
                "prune_events": 0,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.10, "saturated_fraction": 0.02, "stale_fraction": 0.20},
            "runtime_truth": {"verdict": "alive"},
        }

        report = evaluate_subcortical_structural_plasticity_isolated(
            snapshot,
            dict(snapshot),
            rollback_policy={"available": True, "snapshot_id": "pre-structural-eval"},
        )

        self.assertFalse(report["snapshot_binding"]["snapshot_hashes_distinct"])
        self.assertFalse(report["snapshot_binding"]["structural_delta_present"])
        self.assertEqual(report["promotion_gate"]["status"], "blocked_evidence_incomplete")
        self.assertFalse(report["promotion_gate"]["eligible_for_structural_mutation"])

    def test_structural_plasticity_isolated_evaluator_blocks_unbound_rollback(self) -> None:
        pre_snapshot = {
            "current_state_revision": 20,
            "binding_topology": {
                "edges_added_total": 4,
                "edges_removed_total": 1,
                "growth_events": 1,
                "prune_events": 0,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.20, "saturated_fraction": 0.05, "stale_fraction": 0.30},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 21,
            "binding_topology": {
                "edges_added_total": 5,
                "edges_removed_total": 2,
                "growth_events": 2,
                "prune_events": 1,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.10, "saturated_fraction": 0.02, "stale_fraction": 0.20},
            "runtime_truth": {"verdict": "alive"},
        }

        report = evaluate_subcortical_structural_plasticity_isolated(
            pre_snapshot,
            post_snapshot,
            rollback_policy={"available": True, "snapshot_id": "pre-structural-eval"},
        )

        self.assertTrue(report["rollback_evidence"]["available"])
        self.assertFalse(report["rollback_evidence"]["bound_to_pre_snapshot"])
        self.assertFalse(report["rollback_evidence"]["pre_snapshot_hash_match"])
        self.assertEqual(report["promotion_gate"]["status"], "blocked_evidence_incomplete")

    def test_structural_mutation_design_requires_bound_evaluation_and_operator_confirmation(self) -> None:
        pre_snapshot = {
            "current_state_revision": 30,
            "binding_topology": {
                "edges_added_total": 4,
                "edges_removed_total": 1,
                "growth_events": 1,
                "prune_events": 0,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.20, "saturated_fraction": 0.05, "stale_fraction": 0.30},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 31,
            "binding_topology": {
                "edges_added_total": 5,
                "edges_removed_total": 2,
                "growth_events": 2,
                "prune_events": 1,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.10, "saturated_fraction": 0.02, "stale_fraction": 0.20},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        evaluation = evaluate_subcortical_structural_plasticity_isolated(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-structural-design",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )

        design = build_subcortical_structural_mutation_design(
            evaluation,
            operator_id="operator-structural-design",
            confirmation=True,
        )
        blocked = build_subcortical_structural_mutation_design(
            evaluation,
            operator_id="operator-structural-design",
            confirmation=False,
        )

        self.assertEqual(design["artifact_kind"], "terminus_subcortical_structural_mutation_design")
        self.assertEqual(design["surface"], "subcortical_structural_mutation_design.v1")
        self.assertFalse(design["executable"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["writes_checkpoint"])
        self.assertFalse(design["calls_growth_or_prune"])
        self.assertFalse(design["applies_structural_mutation"])
        self.assertEqual(len(design["structural_mutation_design_hash"]), 64)
        self.assertTrue(design["evaluation_binding"]["rollback_bound_to_pre_snapshot"])
        self.assertEqual(design["structural_mutation_design"]["total_edge_delta"], 2)
        self.assertFalse(design["structural_mutation_design"]["runtime_update_applied"])
        self.assertFalse(design["structural_mutation_design"]["checkpoint_written"])
        self.assertEqual(
            design["promotion_gate"]["status"],
            "ready_for_structural_mutation_preflight_review",
        )
        self.assertFalse(design["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(
            design["promotion_gate"]["eligible_for_structural_mutation_preflight_review"]
        )
        self.assertEqual(
            blocked["promotion_gate"]["status"],
            "blocked_missing_structural_mutation_design_evidence",
        )
        self.assertFalse(
            blocked["promotion_gate"]["required_evidence"]["operator_confirmation"]
        )

    def test_structural_mutation_preflight_requires_design_hash_revision_and_checkpoint(self) -> None:
        pre_snapshot = {
            "current_state_revision": 40,
            "binding_topology": {
                "edges_added_total": 4,
                "edges_removed_total": 1,
                "growth_events": 1,
                "prune_events": 0,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.20, "saturated_fraction": 0.05, "stale_fraction": 0.30},
            "runtime_truth": {"verdict": "degraded"},
        }
        post_snapshot = {
            "current_state_revision": 41,
            "binding_topology": {
                "edges_added_total": 5,
                "edges_removed_total": 2,
                "growth_events": 2,
                "prune_events": 1,
            },
            "device_evidence": {
                "binding_devices": {"binding_state_device": "cuda:0"},
                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
            },
            "spike_health": {"silent_fraction": 0.10, "saturated_fraction": 0.02, "stale_fraction": 0.20},
            "runtime_truth": {"verdict": "alive"},
        }
        pre_snapshot_hash = hashlib.sha256(
            json.dumps(pre_snapshot, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        evaluation = evaluate_subcortical_structural_plasticity_isolated(
            pre_snapshot,
            post_snapshot,
            rollback_policy={
                "available": True,
                "snapshot_id": "pre-structural-design",
                "pre_snapshot_hash": pre_snapshot_hash,
            },
        )
        design = build_subcortical_structural_mutation_design(
            evaluation,
            operator_id="operator-structural-design",
            confirmation=True,
        )

        preflight = build_subcortical_structural_mutation_preflight(
            design,
            expected_state_revision=7,
            current_state_revision=7,
            checkpoint_path="checkpoint://pre-structural-mutation",
        )
        stale = build_subcortical_structural_mutation_preflight(
            design,
            expected_state_revision=6,
            current_state_revision=7,
            checkpoint_path="checkpoint://pre-structural-mutation",
        )

        self.assertEqual(preflight["artifact_kind"], "terminus_subcortical_structural_mutation_preflight")
        self.assertEqual(preflight["surface"], "subcortical_structural_mutation_preflight.v1")
        self.assertFalse(preflight["executable"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["writes_checkpoint"])
        self.assertFalse(preflight["calls_growth_or_prune"])
        self.assertFalse(preflight["applies_structural_mutation"])
        self.assertEqual(len(preflight["structural_mutation_preflight_hash"]), 64)
        self.assertTrue(preflight["design_binding"]["design_hash_recomputed_match"])
        self.assertTrue(
            preflight["checkpoint_transaction_requirements"]["expected_state_revision_current"]
        )
        self.assertTrue(preflight["checkpoint_transaction_requirements"]["checkpoint_path_available"])
        self.assertEqual(
            preflight["promotion_gate"]["status"],
            "ready_for_operator_structural_mutation_execution_review",
        )
        self.assertFalse(preflight["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(preflight["promotion_gate"]["eligible_for_operator_execution_review"])
        self.assertEqual(
            stale["promotion_gate"]["status"],
            "blocked_missing_structural_mutation_preflight_evidence",
        )
        self.assertFalse(
            stale["promotion_gate"]["required_evidence"]["expected_state_revision_current"]
        )


if __name__ == "__main__":
    unittest.main()
