from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT,
)
from marulho.service.snn_language_readout_ledger import (
    SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
)
from marulho.semantics.spike_language_neurons import (
    SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
)


class JsonPathPredicateAssertion(BaseModel):
    path: str = Field(..., min_length=1)
    op: Literal["contains", "regex", "gt", "gte", "lt", "lte", "between", "startswith", "endswith", "any_contains", "any_regex", "all_contains", "all_regex", "none_contains", "none_regex"]
    value: Any


class JsonPathPredicateGroupAssertion(BaseModel):
    logic: Literal["all", "any", "none"]
    predicates: list[JsonPathPredicateAssertion] = Field(default_factory=list)
    groups: list["JsonPathPredicateGroupAssertion"] = Field(default_factory=list)


try:
    JsonPathPredicateGroupAssertion.model_rebuild()
except AttributeError:  # pragma: no cover - pydantic v1 fallback
    JsonPathPredicateGroupAssertion.update_forward_refs()


class FeedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    context_text: str | None = None
    top_k_candidates: int = Field(5, ge=1, le=64)
    top_k_memories: int = Field(5, ge=1, le=64)
    top_chars: int = Field(6, ge=1, le=32)


class RespondRequest(QueryRequest):
    learn_mode: Literal["none", "user_only", "user_and_selected_evidence"] = "user_and_selected_evidence"
    max_evidence_items: int = Field(3, ge=1, le=8)


class TerminusSourceSpec(BaseModel):
    name: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    source_type: Literal["auto", "file", "hf", "web"] = "auto"
    text_field: str = Field("text", min_length=1)
    hf_config: str | None = None
    topic_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class TerminusSensorySourceSpec(BaseModel):
    name: str = Field(..., min_length=1)
    adapter: Literal["s1_mmalign", "audiocaps"]
    source: str | None = None
    split: str = Field("train", min_length=1)
    year_prefixes: list[str] = Field(default_factory=list)
    sample_rate: int | None = Field(None, ge=1000, le=192000)
    n_fft: int | None = Field(None, ge=64, le=8192)
    max_text_chars: int | None = Field(None, ge=32, le=4000)
    audio_candidates_per_item: int | None = Field(None, ge=1, le=64)
    topic_terms: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class TerminusCatalogEntrySpec(BaseModel):
    name: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    source_type: Literal["auto", "hf", "web"] = "auto"
    summary: str | None = None
    title: str | None = None
    description: str | None = None
    text_field: str = Field("text", min_length=1)
    hf_config: str | None = None
    query_text: str | None = None
    provider: str | None = None
    tags: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    catalog_priority: float | None = None
    prior_weight: float | None = None


class TerminusCatalogSpec(BaseModel):
    name: str | None = None
    catalog_mode: Literal["semantic_registry", "live_remote_search"]
    catalog_entries: list[TerminusCatalogEntrySpec] = Field(default_factory=list)
    catalog_limit: int = Field(8, ge=1, le=128)
    catalog_probe_pool_limit: int | None = Field(None, ge=1, le=256)
    catalog_focus_text: str | None = None
    catalog_focus_terms: list[str] = Field(default_factory=list)
    catalog_diversity_weight: float = 0.20
    catalog_semantic_weight: float = 1.0
    catalog_prior_weight: float = 1.0
    catalog_exclude_sources: list[str] = Field(default_factory=list)
    catalog_exclude_names: list[str] = Field(default_factory=list)
    catalog_providers: list[str] = Field(default_factory=list)
    catalog_queries_per_provider: int = Field(2, ge=1, le=16)
    catalog_provider_result_limit: int = Field(4, ge=1, le=32)
    catalog_provider_timeout_seconds: float = Field(15.0, ge=1.0, le=60.0)


TerminusCandidateSpec = TerminusSourceSpec | TerminusCatalogSpec


class TerminusAutonomyConfig(BaseModel):
    enabled: bool = True
    policy: Literal["active", "round_robin"] = "active"
    candidate_bank: list[TerminusCandidateSpec] = Field(default_factory=list)
    trigger_interval_tokens: int = Field(4096, ge=1, le=200000)
    candidate_train_tokens: int = Field(768, ge=1, le=20000)
    probe_tokens: int = Field(96, ge=1, le=20000)
    acquisition_tokens: int = Field(512, ge=1, le=20000)
    acquisition_slots: int = Field(1, ge=1, le=16)
    gap_exploration_bonus: float = 0.03
    gap_ambiguity_weight: float = 0.4
    gap_switch_weight: float = 0.2
    gap_margin_reference: float = 0.12
    coverage_balance_penalty: float = 0.2
    gap_focus_margin: float = 0.05
    scout_commit_tokens: int = Field(0, ge=0, le=20000)
    scout_top_k: int = Field(1, ge=1, le=16)
    semantic_shortlist_size: int = Field(0, ge=0, le=32)
    semantic_shortlist_gap_weight: float = 0.5
    semantic_shortlist_affinity_weight: float = 0.5


class TerminusSensoryConfig(BaseModel):
    enabled: bool = True
    source_bank: list[TerminusSensorySourceSpec] = Field(default_factory=list)
    episode_interval_tokens: int = Field(1536, ge=1, le=200000)
    items_per_episode: int = Field(2, ge=1, le=16)
    base_windows_per_item: int = Field(4, ge=1, le=128)
    max_windows_per_item: int = Field(10, ge=1, le=128)
    confidence_window_gain: float = Field(3.0, ge=0.0, le=64.0)
    semantic_window_gain: float = Field(3.0, ge=0.0, le=64.0)
    item_retrieval_lookahead: int = Field(6, ge=1, le=32)
    item_retrieval_semantic_weight: float = Field(0.72, ge=0.0, le=1.0)
    modality_target_confidence: float = Field(0.70, ge=0.1, le=1.0)
    observation_salience: float = Field(0.82, ge=0.1, le=1.0)
    cooldown_seconds: float = Field(8.0, ge=0.1, le=600.0)
    repeat_sources: bool = True
    queue_target_items: int = Field(6, ge=1, le=128)
    prewarm_on_startup: bool = False
    prewarm_max_seconds: float = Field(5.0, ge=0.01, le=600.0)


class TerminusIngestionConfig(BaseModel):
    enabled: bool = True
    queue_target_tokens: int = Field(256, ge=1, le=200000)
    prewarm_on_startup: bool = False
    prewarm_max_seconds: float = Field(5.0, ge=0.01, le=600.0)


class TerminusConfigureRequest(BaseModel):
    source_bank: list[TerminusSourceSpec] = Field(..., min_length=1)
    tick_tokens: int = Field(128, ge=1, le=20000)
    source_concept_observation_tick_interval: int = Field(4, ge=1, le=1024)
    sleep_interval_seconds: float = Field(0.25, ge=0.01, le=60.0)
    execution_quantum_tokens: int = Field(16, ge=1, le=128)
    execution_yield_seconds: float = Field(0.0, ge=0.0, le=1.0)
    repeat_sources: bool = True
    autonomy: TerminusAutonomyConfig | None = None
    sensory: TerminusSensoryConfig | None = None
    ingestion: TerminusIngestionConfig | None = None


class TerminusTickRequest(BaseModel):
    steps: int = Field(1, ge=1, le=128)


class DigitalActionRequest(BaseModel):
    action_type: Literal["workspace_search", "workspace_read", "web_fetch", "api_request"]
    query_text: str | None = Field(None, min_length=1)
    path: str | None = None
    url: str | None = None
    method: Literal["GET", "POST"] = "GET"
    params: dict[str, Any] | None = None
    json_body: Any | None = None
    expected_json_paths: list[str] = Field(default_factory=list)
    expected_json_values: dict[str, Any] | None = None
    expected_json_predicates: list[JsonPathPredicateAssertion] = Field(default_factory=list)
    expected_json_predicate_groups: list[JsonPathPredicateGroupAssertion] = Field(default_factory=list)
    expected_response_shape: Literal["object", "array", "scalar", "null"] | None = None
    predicted_outcome: str | None = None
    root_path: str | None = None
    max_hits: int = Field(6, ge=1, le=32)
    max_files: int = Field(256, ge=1, le=5000)
    max_file_bytes: int = Field(200000, ge=1000, le=5000000)
    max_response_bytes: int = Field(200000, ge=1000, le=5000000)
    timeout_seconds: float = Field(10.0, ge=0.1, le=120.0)


class RuntimeFeedbackRequest(BaseModel):
    target_type: Literal["runtime_episode", "action"]
    target_id: str = Field(..., min_length=1, max_length=160)
    verdict: Literal["verified", "contradicted", "unverified"]
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    summary: str | None = Field(None, max_length=2000)
    corrected_output: Any | None = None
    evidence: list[Any] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    evaluator_id: str | None = Field(None, max_length=160)


class SNNLanguageHeldoutReadoutSlot(BaseModel):
    label: str = Field(..., min_length=1, max_length=200)
    pressure_band: Literal["low", "medium", "high"] = "low"
    grounded: bool = True
    slot_id: str | None = Field(None, max_length=160)


class SNNLanguageHeldoutEvaluationRequest(BaseModel):
    heldout_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=1,
        max_length=16,
    )
    device_evidence: dict[str, Any] | None = None


class SNNLanguageTrainingReadinessRequest(BaseModel):
    heldout_evaluation: dict[str, Any] = Field(..., min_length=1)
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageTrainerDryRunRequest(BaseModel):
    training_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=2,
        max_length=32,
    )
    validation_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=2,
        max_length=32,
    )
    device_evidence: dict[str, Any] | None = None
    learning_rate: float = Field(0.08, ge=0.0, le=1.0)
    epochs: int = Field(2, ge=1, le=8)


class SNNLanguageTrainerEvaluationRequest(BaseModel):
    dry_run_report: dict[str, Any] = Field(..., min_length=1)
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageSequencePredictionRequest(BaseModel):
    training_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=2,
        max_length=32,
    )
    current_readout_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(..., min_length=1, max_length=16)
    device_evidence: dict[str, Any] | None = None
    learning_rate: float = Field(0.08, ge=0.0, le=1.0)
    epochs: int = Field(2, ge=1, le=8)
    top_k: int = Field(8, ge=1, le=16)
    persistent_transition_weights: dict[str, Any] | None = None


class SNNLanguageSequenceMismatchRequest(BaseModel):
    prediction_report: dict[str, Any] = Field(..., min_length=1)
    observed_readout_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(..., min_length=1, max_length=16)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageReadoutDraftRequest(BaseModel):
    prediction_report: dict[str, Any] = Field(..., min_length=1)
    readout_vocabulary_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(..., min_length=1, max_length=32)
    device_evidence: dict[str, Any] | None = None
    transition_memory_evaluation: dict[str, Any] | None = None
    max_draft_terms: int = Field(6, ge=1, le=12)


class SNNLanguageReadoutEmissionRequest(BaseModel):
    readout_draft: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageReadoutEmissionReviewRequest(BaseModel):
    readout_emission: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageReadoutEmissionReplayEvaluationDesignRequest(BaseModel):
    emission_replay_evaluation_policy: dict[str, Any] = Field(..., min_length=1)
    design_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationEvaluationDesignRequest(BaseModel):
    dense_label_candidate_calibration_policy: dict[str, Any] = Field(..., min_length=1)
    heldout_label_evidence: dict[str, Any] | None = None
    design_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationEvaluationPreflightRequest(BaseModel):
    dense_label_candidate_calibration_evaluation_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationEvaluationRequest(BaseModel):
    dense_label_candidate_calibration_evaluation_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    heldout_label_evidence: dict[str, Any] = Field(..., min_length=1)
    bin_count: int = Field(5, ge=2, le=16)


class SNNLanguageDenseLabelCandidateCalibrationEvaluationReviewRequest(BaseModel):
    dense_label_candidate_calibration_evaluation: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    review_policy: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationUpdateDesignRequest(BaseModel):
    dense_label_candidate_calibration_evaluation_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    update_policy: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationUpdatePreflightRequest(BaseModel):
    dense_label_candidate_calibration_update_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str = Field(..., min_length=1, max_length=1024)
    rollback_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationRequest(BaseModel):
    dense_label_candidate_calibration_update_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationReviewRequest(BaseModel):
    dense_label_candidate_calibration_update_application: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidatePostCalibrationObservationWindowRequest(BaseModel):
    dense_label_candidate_calibration_update_application_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    observation_evidence: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    window_policy: dict[str, Any] | None = None


class SNNLanguageDenseLabelCandidatePostCalibrationOperatorReviewRequest(BaseModel):
    dense_label_candidate_post_calibration_observation_window: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    review_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceUseDesignRequest(BaseModel):
    dense_label_candidate_post_calibration_operator_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    confidence_use_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceUsePreflightRequest(BaseModel):
    dense_label_confidence_use_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    candidate_evidence: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceUseExecutorRequest(BaseModel):
    calibrated_dense_label_confidence_use_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    candidate_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceOperatorDisplayReviewRequest(BaseModel):
    calibrated_dense_label_confidence_use_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    review_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceInternalStabilityReviewRequest(BaseModel):
    calibrated_dense_label_confidence_use_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    stability_evidence: dict[str, Any] | None = None
    review_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewDesignRequest(BaseModel):
    calibrated_dense_label_confidence_internal_stability_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    replay_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewPreflightRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_replay_review_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewExecutorRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_replay_review_preflight: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    replay_cycle_evidence: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationDesignRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_replay_review_executor: dict[
        str, Any
    ] = Field(..., min_length=1)
    recalibration_policy: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationPreflightRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_recalibration_design: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationExecutorRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_recalibration_preflight: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationApplicationReviewRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_recalibration_executor: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationObservationWindowRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_recalibration_application_review: dict[
        str, Any
    ] = Field(..., min_length=1)
    observation_evidence: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    window_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationStabilityReviewRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_post_calibration_observation_window: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    stability_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseDesignRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_post_calibration_stability_review: dict[
        str, Any
    ] = Field(..., min_length=1)
    confidence_use_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousUsePreflightRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_use_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    candidate_evidence: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseExecutorRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_use_preflight: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    candidate_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseEventReviewRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_use_executor: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousHashReadoutBindingDesignRequest(BaseModel):
    calibrated_dense_label_confidence_autonomous_use_event_review: dict[
        str, Any
    ] = Field(..., min_length=1)
    readout_vocabulary_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(
        ..., min_length=1, max_length=32
    )
    binding_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousHashReadoutBindingPreflightRequest(BaseModel):
    autonomous_hash_readout_binding_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousHashReadoutBindingExecutorRequest(BaseModel):
    autonomous_hash_readout_binding_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousHashReadoutBindingEventReviewRequest(BaseModel):
    autonomous_hash_readout_binding_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundReadoutObservationDesignRequest(BaseModel):
    autonomous_hash_readout_binding_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    observation_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundReadoutObservationPreflightRequest(BaseModel):
    autonomous_bound_readout_observation_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundReadoutObservationExecutorRequest(BaseModel):
    autonomous_bound_readout_observation_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    observation_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundReadoutObservationEventReviewRequest(BaseModel):
    autonomous_bound_readout_observation_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousReadoutTrainingWindowDesignRequest(BaseModel):
    autonomous_bound_readout_observation_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    training_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousReadoutTrainingWindowPreflightRequest(BaseModel):
    autonomous_readout_training_window_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousReadoutTrainingWindowExecutorRequest(BaseModel):
    autonomous_readout_training_window_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    training_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousReadoutTrainingWindowEventReviewRequest(BaseModel):
    autonomous_readout_training_window_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousDecoderProbeDesignRequest(BaseModel):
    autonomous_readout_training_window_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    probe_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousDecoderProbePreflightRequest(BaseModel):
    autonomous_decoder_probe_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousDecoderProbeExecutorRequest(BaseModel):
    autonomous_decoder_probe_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    probe_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousDecoderProbeEventReviewRequest(BaseModel):
    autonomous_decoder_probe_executor: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousLanguageOutputDesignRequest(BaseModel):
    autonomous_decoder_probe_event_review: dict[str, Any] = Field(..., min_length=1)
    output_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousLanguageOutputPreflightRequest(BaseModel):
    autonomous_language_output_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousLanguageOutputExecutorRequest(BaseModel):
    autonomous_language_output_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    output_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousLanguageOutputEventReviewRequest(BaseModel):
    autonomous_language_output_executor: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousDecodedOutputDesignRequest(BaseModel):
    autonomous_language_output_event_review: dict[str, Any] = Field(..., min_length=1)
    vocabulary_binding: dict[str, Any] | None = None
    decode_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousDecodedOutputPreflightRequest(BaseModel):
    autonomous_decoded_output_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousDecodedOutputExecutorRequest(BaseModel):
    autonomous_decoded_output_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    decode_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousDecodedOutputEventReviewRequest(BaseModel):
    autonomous_decoded_output_executor: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedTextEmissionDesignRequest(BaseModel):
    autonomous_decoded_output_event_review: dict[str, Any] = Field(..., min_length=1)
    text_surface_binding: dict[str, Any] | None = None
    emission_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedTextEmissionPreflightRequest(BaseModel):
    autonomous_bounded_text_emission_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedTextEmissionExecutorRequest(BaseModel):
    autonomous_bounded_text_emission_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    emission_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedTextEmissionEventReviewRequest(BaseModel):
    autonomous_bounded_text_emission_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceSequenceReviewRequest(BaseModel):
    autonomous_bounded_text_emission_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    sequence_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceCommitDesignRequest(BaseModel):
    autonomous_text_surface_sequence_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    commit_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceCommitPreflightRequest(BaseModel):
    autonomous_text_surface_commit_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceCommitExecutorRequest(BaseModel):
    autonomous_text_surface_commit_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    commit_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceCommitEventReviewRequest(BaseModel):
    autonomous_text_surface_commit_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceMaterializationDesignRequest(BaseModel):
    autonomous_text_surface_commit_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    materialization_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceMaterializationPreflightRequest(BaseModel):
    autonomous_text_surface_materialization_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceMaterializationExecutorRequest(BaseModel):
    autonomous_text_surface_materialization_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    materialization_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousTextSurfaceMaterializationEventReviewRequest(BaseModel):
    autonomous_text_surface_materialization_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceReviewRequest(BaseModel):
    autonomous_text_surface_materialization_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    language_surface_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceCommitDesignRequest(BaseModel):
    autonomous_bounded_language_surface_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    commit_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceCommitPreflightRequest(BaseModel):
    autonomous_bounded_language_surface_commit_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceCommitExecutorRequest(BaseModel):
    autonomous_bounded_language_surface_commit_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    commit_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceCommitEventReviewRequest(BaseModel):
    autonomous_bounded_language_surface_commit_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceUseReviewRequest(BaseModel):
    autonomous_bounded_language_surface_commit_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    use_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceUsePreflightRequest(BaseModel):
    autonomous_bounded_language_surface_use_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceUseExecutorRequest(BaseModel):
    autonomous_bounded_language_surface_use_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    use_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousBoundedLanguageSurfaceUseEventReviewRequest(BaseModel):
    autonomous_bounded_language_surface_use_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageGenerationDesignRequest(BaseModel):
    autonomous_bounded_language_surface_use_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    generation_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageGenerationPreflightRequest(BaseModel):
    autonomous_snn_language_generation_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageGenerationExecutorRequest(BaseModel):
    autonomous_snn_language_generation_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    generation_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageGenerationEventReviewRequest(BaseModel):
    autonomous_snn_language_generation_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageDecodingDesignRequest(BaseModel):
    autonomous_snn_language_generation_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    decoding_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageDecodingPreflightRequest(BaseModel):
    autonomous_snn_language_decoding_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    decoder_capabilities: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageDecodingExecutorRequest(BaseModel):
    autonomous_snn_language_decoding_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    decoding_evidence: dict[str, Any] | None = None
    execution_policy: dict[str, Any] | None = None


class SNNLanguageAutonomousSNNLanguageDecodingEventReviewRequest(BaseModel):
    autonomous_snn_language_decoding_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageSurfaceDesignRequest(BaseModel):
    autonomous_snn_language_decoding_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    surface_policy: dict[str, Any] | None = None


class SNNLanguageSurfacePreflightRequest(BaseModel):
    snn_language_surface_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageSurfaceExecutorRequest(BaseModel):
    snn_language_surface_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    execution_policy: dict[str, Any] | None = None


class SNNLanguageSurfaceEventReviewRequest(BaseModel):
    snn_language_surface_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageMemoryDesignRequest(BaseModel):
    snn_language_surface_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    memory_policy: dict[str, Any] | None = None


class SNNLanguageMemoryPreflightRequest(BaseModel):
    snn_language_memory_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageMemoryExecutorRequest(BaseModel):
    snn_language_memory_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    execution_policy: dict[str, Any] | None = None


class SNNLanguageMemoryEventReviewRequest(BaseModel):
    snn_language_memory_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageConsolidationDesignRequest(BaseModel):
    snn_language_memory_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    consolidation_policy: dict[str, Any] | None = None


class SNNLanguageConsolidationPreflightRequest(BaseModel):
    snn_language_consolidation_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageConsolidationExecutorRequest(BaseModel):
    snn_language_consolidation_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    execution_policy: dict[str, Any] | None = None


class SNNLanguageConsolidationEventReviewRequest(BaseModel):
    snn_language_consolidation_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageStructuralPlasticityDesignRequest(BaseModel):
    snn_language_consolidation_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    structural_policy: dict[str, Any] | None = None


class SNNLanguageStructuralPlasticityPreflightRequest(BaseModel):
    snn_language_structural_plasticity_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageStructuralPlasticityExecutorRequest(BaseModel):
    snn_language_structural_plasticity_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    execution_policy: dict[str, Any] | None = None


class SNNLanguageStructuralPlasticityEventReviewRequest(BaseModel):
    snn_language_structural_plasticity_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    review_policy: dict[str, Any] | None = None


class SNNLanguageCapacityMutationDesignRequest(BaseModel):
    snn_language_structural_plasticity_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    capacity_policy: dict[str, Any] | None = None


class SNNLanguageCapacityMutationPreflightRequest(BaseModel):
    snn_language_capacity_mutation_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_transaction: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageCapacityMutationExecutorRequest(BaseModel):
    snn_language_capacity_mutation_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1, max_length=4096)
    requested_device: str | None = Field(None, min_length=1, max_length=80)


class SNNLanguageCapacityMutationEventReviewRequest(BaseModel):
    snn_language_capacity_mutation_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)


class SNNLanguageNewbornNeuronIntegrationDesignRequest(BaseModel):
    snn_language_capacity_mutation_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    integration_policy: dict[str, Any] | None = None


class SNNLanguageNewbornNeuronIntegrationPreflightRequest(BaseModel):
    snn_language_newborn_neuron_integration_design: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    live_spike_evidence: dict[str, Any] = Field(..., min_length=1)
    checkpoint_transaction: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageNewbornNeuronIntegrationExecutorRequest(BaseModel):
    snn_language_newborn_neuron_integration_preflight: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1, max_length=4096)


class SNNLanguageNewbornNeuronIntegrationEventReviewRequest(BaseModel):
    snn_language_newborn_neuron_integration_executor: dict[str, Any] = Field(
        ..., min_length=1
    )
    expected_state_revision: int = Field(..., ge=0)


class SNNLanguageNewbornNeuronCriticalPeriodLearningDesignRequest(BaseModel):
    snn_language_newborn_neuron_integration_event_review: dict[str, Any] = Field(
        ..., min_length=1
    )
    learning_policy: dict[str, Any] | None = None


class SNNLanguageNewbornNeuronCriticalPeriodLearningPreflightRequest(BaseModel):
    snn_language_newborn_neuron_critical_period_learning_design: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    critical_period_activity_evidence: dict[str, Any] = Field(
        ..., min_length=1
    )
    checkpoint_transaction: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageNewbornNeuronCriticalPeriodLearningExecutorRequest(BaseModel):
    snn_language_newborn_neuron_critical_period_learning_preflight: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1, max_length=4096)


class SNNLanguageNewbornNeuronCriticalPeriodLearningEventReviewRequest(BaseModel):
    snn_language_newborn_neuron_critical_period_learning_executor: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)


class SNNLanguageNewbornNeuronCriticalPeriodLearningContinuationDesignRequest(
    BaseModel
):
    snn_language_newborn_neuron_critical_period_learning_event_review: dict[
        str, Any
    ] = Field(..., min_length=1)


class SNNLanguageNewbornNeuronMaturationOutcomeReviewRequest(BaseModel):
    snn_language_newborn_neuron_critical_period_learning_event_review: dict[
        str, Any
    ] = Field(..., min_length=1)


class SNNLanguageNewbornSynapsePruningDesignRequest(BaseModel):
    snn_language_newborn_neuron_maturation_outcome_review: dict[
        str, Any
    ] = Field(..., min_length=1)


class SNNLanguageNewbornSynapsePruningPreflightRequest(BaseModel):
    snn_language_newborn_synapse_pruning_design: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_transaction: dict[str, Any] | None = None
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageNewbornSynapsePruningExecutorRequest(BaseModel):
    snn_language_newborn_synapse_pruning_preflight: dict[
        str, Any
    ] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1, max_length=4096)


class SNNLanguageReadoutEmissionReplayContextReviewRequest(BaseModel):
    emission_replay_evaluation_design: dict[str, Any] = Field(..., min_length=1)
    prediction_report: dict[str, Any] = Field(..., min_length=1)
    observed_readout_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(
        ..., min_length=1, max_length=SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    device_evidence: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageReadoutRolloutCandidateRequest(BaseModel):
    prediction_report: dict[str, Any] = Field(..., min_length=1)
    readout_vocabulary_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(..., min_length=1, max_length=32)
    device_evidence: dict[str, Any] | None = None
    transition_memory_evaluation: dict[str, Any] | None = None
    rollout_steps: int = Field(4, ge=1, le=12)
    top_k: int = Field(4, ge=1, le=8)


class SNNLanguageReadoutRolloutReplayEvaluationRequest(BaseModel):
    readout_rollout_candidate: dict[str, Any] = Field(..., min_length=1)
    candidate_limit: int = Field(8, ge=1, le=32)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageReadoutLedgerRecordRequest(BaseModel):
    readout_draft: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageReadoutRolloutLedgerRecordRequest(BaseModel):
    readout_rollout_replay_evaluation: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageReadoutRolloutRehearsalEvaluationRequest(BaseModel):
    rollout_rehearsal_promotion_policy: dict[str, Any] = Field(..., min_length=1)
    candidate_limit: int = Field(8, ge=0, le=32)


class SNNLanguageReadoutRolloutRehearsalExperimentRequest(BaseModel):
    rollout_rehearsal_evaluation: dict[str, Any] = Field(..., min_length=1)
    replay_cycles: int = Field(3, ge=1, le=12)
    stability_floor: float = Field(0.95, ge=0.0, le=1.0)


class SNNLanguageReadoutRolloutConsolidationDesignRequest(BaseModel):
    rollout_rehearsal_experiment: dict[str, Any] = Field(..., min_length=1)
    consolidation_policy: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageReadoutRolloutConsolidationShadowDeltaRequest(BaseModel):
    rollout_consolidation_design: dict[str, Any] = Field(..., min_length=1)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageReadoutRolloutConsolidationShadowApplicationPreflightRequest(BaseModel):
    rollout_consolidation_design: dict[str, Any] = Field(..., min_length=1)
    rollout_consolidation_shadow_delta: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageReadoutRolloutDevelopmentalPlasticityReviewRequest(BaseModel):
    rollout_consolidation_design: dict[str, Any] = Field(..., min_length=1)
    rollout_consolidation_shadow_application_preflight: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageReadoutRolloutRegenerationProposalAdapterRequest(BaseModel):
    rollout_developmental_plasticity_review: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageReadoutRolloutRegenerationReplayArtifactReviewRequest(BaseModel):
    rollout_regeneration_proposal_adapter: dict[str, Any] = Field(..., min_length=1)
    snn_transition_memory_replay_artifact: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageReadoutRolloutRegenerationPermitRequestRequest(BaseModel):
    rollout_regeneration_replay_artifact_review: dict[str, Any] = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageReadoutRolloutRegenerationApplicationPreflightRequest(BaseModel):
    rollout_regeneration_permit_request: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1)


class SNNLanguageReadoutRolloutRegenerationApplicationRequest(BaseModel):
    rollout_regeneration_application_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)
    max_outgoing_row_mass: float = Field(1.0, gt=0.0, le=4.0)


class SNNLanguageReadoutRehearsalEvaluationRequest(BaseModel):
    replay_priority_report: dict[str, Any] = Field(..., min_length=1)
    candidate_limit: int = Field(8, ge=0, le=32)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageReadoutRehearsalExperimentRequest(BaseModel):
    rehearsal_evaluation: dict[str, Any] = Field(..., min_length=1)
    replay_cycles: int = Field(3, ge=1, le=12)
    stability_floor: float = Field(0.85, ge=0.0, le=1.0)


class SNNLanguageReadoutReplayDesignRequest(BaseModel):
    rehearsal_experiment: dict[str, Any] = Field(..., min_length=1)
    replay_policy: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageReadoutReplayDryRunRequest(BaseModel):
    replay_design: dict[str, Any] = Field(..., min_length=1)
    operator_approval: bool = False
    operator_id: str | None = Field(None, max_length=160)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageReadoutPlasticityPreflightRequest(BaseModel):
    readout_replay_dry_run: dict[str, Any] = Field(..., min_length=1)
    plasticity_policy: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageReadoutPlasticityReplayBridgeRequest(BaseModel):
    readout_plasticity_preflight: dict[str, Any] = Field(..., min_length=1)
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageTransitionMemoryPredictionEvaluationRequest(BaseModel):
    training_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=2,
        max_length=32,
    )
    evaluation_readout_slot_batches: list[list[SNNLanguageHeldoutReadoutSlot]] = Field(
        ...,
        min_length=2,
        max_length=32,
    )
    transition_memory_state: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    learning_rate: float = Field(0.08, ge=0.0, le=1.0)
    epochs: int = Field(2, ge=1, le=8)
    top_k: int = Field(8, ge=1, le=16)


class SNNLanguagePlasticityPressureRequest(BaseModel):
    mismatch_report: dict[str, Any] = Field(..., min_length=1)
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityTrialRequest(BaseModel):
    pressure_report: dict[str, Any] = Field(..., min_length=1)
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityReplayEvaluationRequest(BaseModel):
    trial_report: dict[str, Any] = Field(..., min_length=1)
    replay_window: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
    )
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityReplayExperimentRequest(BaseModel):
    replay_evaluation: dict[str, Any] = Field(..., min_length=1)
    replay_sequences: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
    )
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityApplicationDesignRequest(BaseModel):
    replay_experiment: dict[str, Any] = Field(..., min_length=1)
    application_policy: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityShadowApplicationRequest(BaseModel):
    application_design: dict[str, Any] = Field(..., min_length=1)
    shadow_delta: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguagePlasticityShadowDeltaRequest(BaseModel):
    application_design: dict[str, Any] = Field(..., min_length=1)
    replay_sequences: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
    )
    device_evidence: dict[str, Any] | None = None


class SNNLanguagePlasticityLiveApplicationReadinessRequest(BaseModel):
    shadow_application: dict[str, Any] = Field(..., min_length=1)
    rollback_readiness: dict[str, Any] | None = None
    operator_approval: dict[str, Any] | None = None


class SNNLanguagePlasticityLiveApplicationPreflightRequest(BaseModel):
    live_application_readiness: dict[str, Any] = Field(..., min_length=1)
    application_target: dict[str, Any] | None = None
    checkpoint_transaction: dict[str, Any] | None = None


class SNNLanguagePlasticityLiveApplicationRequest(BaseModel):
    live_application_readiness: dict[str, Any] = Field(..., min_length=1)
    shadow_delta: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)


class SNNLanguageCapacityExpansionDesignRequest(BaseModel):
    capacity_pressure: dict[str, Any] = Field(..., min_length=1)
    device_evidence: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None
    max_neuron_growth_factor: float = Field(2.0, ge=1.0, le=4.0)


class SNNLanguageCapacityExpansionPreflightRequest(BaseModel):
    capacity_expansion_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_transaction: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageCapacityResizeCompatibilityAuditRequest(BaseModel):
    capacity_expansion_preflight: dict[str, Any] = Field(..., min_length=1)
    language_capacity_state: dict[str, Any] | None = None


class SNNLanguageDenseReadoutResizePlanRequest(BaseModel):
    capacity_pressure: dict[str, Any] = Field(..., min_length=1)
    fixed_boundaries: dict[str, Any] = Field(..., min_length=1)


class SNNLanguageDenseReadoutResizePreflightRequest(BaseModel):
    dense_readout_resize_plan: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_transaction: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None


class SNNLanguageDenseReadoutResizeTransactionProposalRequest(BaseModel):
    dense_readout_resize_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageDenseReadoutResizeExecutorReadinessAuditRequest(BaseModel):
    dense_readout_resize_transaction_proposal: dict[str, Any] = Field(..., min_length=1)
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageDenseReadoutLayoutMigrationRequest(BaseModel):
    dense_readout_resize_transaction_proposal: dict[str, Any] = Field(..., min_length=1)
    dense_readout_resize_executor_readiness_audit: dict[str, Any] = Field(
        ...,
        min_length=1,
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)


class SNNLanguageDenseReadoutTensorMaterializationReadinessRequest(BaseModel):
    dense_readout_layout_migration: dict[str, Any] = Field(..., min_length=1)
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageDenseReadoutTensorMaterializationRequest(BaseModel):
    dense_readout_tensor_materialization_readiness: dict[str, Any] = Field(
        ...,
        min_length=1,
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)
    requested_device: str | None = Field(None, min_length=1, max_length=64)


class SNNLanguageDenseReadoutTrainingReadinessRequest(BaseModel):
    dense_readout_tensor_integrity: dict[str, Any] = Field(..., min_length=1)
    heldout_evaluation: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageDenseReadoutTrainingLoopDesignRequest(BaseModel):
    dense_readout_training_readiness: dict[str, Any] = Field(..., min_length=1)
    training_plan: dict[str, Any] | None = None
    device_evidence: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageDenseReadoutTrainingLoopPreflightRequest(BaseModel):
    dense_readout_training_loop_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1)
    executor_capabilities: dict[str, Any] | None = None


class SNNLanguageDenseReadoutTrainingRequest(BaseModel):
    dense_readout_training_loop_preflight: dict[str, Any] = Field(..., min_length=1)
    training_transitions: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT,
    )
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)


class SNNLanguageDenseReadoutPostTrainingEvaluationRequest(BaseModel):
    dense_readout_training: dict[str, Any] = Field(..., min_length=1)
    dense_readout_tensor_integrity: dict[str, Any] = Field(..., min_length=1)
    heldout_evaluation: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNLanguageDenseReadoutDecoderProbeDesignRequest(BaseModel):
    dense_readout_post_training_evaluation: dict[str, Any] = Field(..., min_length=1)
    readout_slots: list[dict[str, Any]] = Field(..., min_length=1, max_length=8)
    device_evidence: dict[str, Any] | None = None
    decoder_design: dict[str, Any] | None = None


class SNNLanguageDenseReadoutDecoderProbePreflightRequest(BaseModel):
    dense_readout_decoder_probe_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    device_evidence: dict[str, Any] | None = None


class SNNLanguageDenseReadoutDecoderProbeExecutionRequest(BaseModel):
    dense_readout_decoder_probe_preflight: dict[str, Any] = Field(..., min_length=1)
    max_candidate_labels: int = Field(4, ge=1, le=8)


class SNNLanguageDenseReadoutLabelCandidateReviewRequest(BaseModel):
    dense_readout_decoder_probe_execution: dict[str, Any] = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    review_note: str | None = Field(None, max_length=512)


class SNNLanguageDenseReadoutLabelCandidateEvidenceRecordRequest(BaseModel):
    dense_readout_label_candidate_review: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageTransitionMemoryHomeostaticMaintenanceRequest(BaseModel):
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)
    decay_factor: float = Field(0.98, gt=0.0, le=1.0)
    prune_below: float = Field(0.005, ge=0.0, le=0.25)
    max_outgoing_row_mass: float = Field(1.0, gt=0.0, le=4.0)


class SNNLanguageTransitionMemorySleepPolicyRequest(BaseModel):
    transition_memory_state: dict[str, Any] | None = None
    subcortex_sleep_pressure: dict[str, Any] | None = None
    replay_evidence: dict[str, Any] | None = None
    rollout_regeneration_evidence: dict[str, Any] | None = None
    readout_ledger_evidence: dict[str, Any] | None = None


class SNNSleepPlasticityReviewTicketRequest(BaseModel):
    sleep_policy: dict[str, Any] = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNSleepPlasticitySchedulerDesignReviewTicketRequest(BaseModel):
    limit: int = Field(20, ge=1, le=64)
    cycles: int = Field(4, ge=3, le=16)
    min_stable_cycles: int = Field(3, ge=3, le=16)
    max_review_interval_seconds: float = Field(300.0, ge=60.0, le=3600.0)
    expected_state_revision: int = Field(..., ge=0)
    scheduler_design_hash: str = Field(..., min_length=64, max_length=64)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNSleepPlasticityReviewSchedulerInstallationRequest(BaseModel):
    limit: int = Field(20, ge=1, le=64)
    expected_state_revision: int = Field(..., ge=0)
    scheduler_installation_preflight_hash: str = Field(..., min_length=64, max_length=64)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNSleepPlasticityReviewSchedulerCycleAcknowledgmentRequest(BaseModel):
    expected_state_revision: int = Field(..., ge=0)
    scheduler_installation_id: str = Field(..., min_length=1, max_length=240)
    scheduler_installation_evidence_hash: str = Field(..., min_length=64, max_length=64)
    review_ticket_id: str = Field(..., min_length=1, max_length=240)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageTransitionMemoryRegenerationProposalRequest(BaseModel):
    mismatch_report: dict[str, Any] = Field(..., min_length=1)
    transition_memory_state: dict[str, Any] | None = None
    replay_evidence: dict[str, Any] | None = None
    locality_radius: int = Field(2, ge=1, le=8)
    initial_weight: float = Field(0.02, gt=0.0, le=0.25)
    max_new_synapses: int = Field(8, ge=1, le=32)


class SNNLanguageTransitionMemoryRegenerationPermitRequest(BaseModel):
    replay_artifact_id: str = Field(..., min_length=1, max_length=240)
    regeneration_design: dict[str, Any] = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNReplayEvaluationContextRequest(BaseModel):
    prediction_report: dict[str, Any] = Field(..., min_length=1)
    observed_readout_slots: list[SNNLanguageHeldoutReadoutSlot] = Field(
        ..., min_length=1, max_length=SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
    )
    device_evidence: dict[str, Any] | None = None
    runtime_truth_delta: dict[str, Any] | None = None
    rollback_policy: dict[str, Any] | None = None


class SNNTransitionMemoryReplayArtifactProposalRequest(BaseModel):
    replay_evaluation_context_id: str = Field(..., min_length=1, max_length=240)
    limit: int = Field(8, ge=1, le=32)


class SNNEvaluatedTransitionMemoryReplayArtifactRequest(BaseModel):
    replay_evaluation_context_id: str = Field(..., min_length=1, max_length=240)
    review_ticket_id: str = Field(..., min_length=1, max_length=240)
    limit: int = Field(8, ge=1, le=32)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNReplayArtifactRecordingReviewTicketRequest(BaseModel):
    limit: int = Field(8, ge=1, le=32)
    min_priority_score: float = Field(66.0, ge=0.0, le=100.0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNDueCycleReplayArtifactRecordingReviewTicketRequest(BaseModel):
    limit: int = Field(8, ge=1, le=32)
    max_candidates: int = Field(1, ge=1, le=8)
    min_priority_score: float = Field(66.0, ge=0.0, le=100.0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False


class SNNLanguageTransitionMemoryRegenerationRequest(BaseModel):
    regeneration_proposal: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)
    max_outgoing_row_mass: float = Field(1.0, gt=0.0, le=4.0)


class StructuralPlasticityIsolatedEvaluationRequest(BaseModel):
    pre_snapshot: dict[str, Any] = Field(..., min_length=1)
    post_snapshot: dict[str, Any] = Field(..., min_length=1)
    rollback_policy: dict[str, Any] | None = None
    candidate_evidence: dict[str, Any] | None = None
    cost_evidence: dict[str, Any] | None = None
    runtime_truth_summary: dict[str, Any] | None = None
    no_mutation_evidence: dict[str, Any] | None = None


class StructuralMutationDesignRequest(BaseModel):
    isolated_evaluation: dict[str, Any] = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    mutation_reason: str = Field(..., min_length=1, max_length=240)
    max_total_edge_delta: int = Field(16, ge=1, le=64)


class StructuralMutationPreflightRequest(BaseModel):
    structural_mutation_design: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    checkpoint_path: str | None = Field(None, min_length=1)


class StructuralMutationApplicationRequest(BaseModel):
    structural_mutation_preflight: dict[str, Any] = Field(..., min_length=1)
    expected_state_revision: int = Field(..., ge=0)
    operator_id: str = Field(..., min_length=1, max_length=160)
    confirmation: bool = False
    checkpoint_path: str | None = Field(None, min_length=1)


class CheckpointSaveRequest(BaseModel):
    path: str | None = None


class CheckpointRestoreRequest(BaseModel):
    path: str = Field(..., min_length=1)


class CheckpointRecord(BaseModel):
    path: str
    name: str
    size_bytes: int
    modified_at: str


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointRecord]


class AcquisitionPresetListResponse(BaseModel):
    presets: list[str]


class CheckpointActionResponse(BaseModel):
    path: str
    dirty_state: bool
    state_revision: int
    token_count: int


class FeedResponse(BaseModel):
    feed_summary: dict[str, Any]
    runtime_episode: dict[str, Any] | None = None
    dirty_state: bool
    state_revision: int


class StatusResponse(BaseModel):
    checkpoint_path: str
    dirty_state: bool
    state_revision: int
    token_count: int
    last_winner: int | None
    context_supported: bool
    context_state_norm: float
    trace_history_size: int
    trace_storage_dir: str
    last_trace_id: str | None
    last_trace_created_at: str | None
    checkpoint_metadata: dict[str, Any]
    dopamine: float = 0.0
    serotonin: float = 0.0
    acetylcholine: float = 0.0
    norepinephrine: float = 0.0
    runtime_scope: dict[str, Any]
    memory_store: dict[str, Any]
    concept_store: dict[str, Any]
    terminus_runtime: dict[str, Any]
    replay_dataset_summary: dict[str, Any] | None = None
    snn_sleep_plasticity_autonomy_proposal: dict[str, Any] | None = None
    snn_sleep_plasticity_scheduler_installation_autonomy_proposal: dict[str, Any] | None = None
    snn_due_cycle_bounded_replay_selection_proposal: dict[str, Any] | None = None
    runtime_truth: dict[str, Any] | None = None


class TerminusRuntimeResponse(BaseModel):
    terminus_runtime: dict[str, Any]
    tick_summaries: list[dict[str, Any]] | None = None
    dirty_state: bool
    state_revision: int
    token_count: int
    multimodal: dict[str, Any] | None = None
    runtime_scope: dict[str, Any] | None = None
    memory_store: dict[str, Any] | None = None
    replay_dataset_summary: dict[str, Any] | None = None
    snn_sleep_plasticity_autonomy_proposal: dict[str, Any] | None = None
    snn_sleep_plasticity_scheduler_installation_autonomy_proposal: dict[str, Any] | None = None
    snn_due_cycle_bounded_replay_selection_proposal: dict[str, Any] | None = None
    runtime_truth: dict[str, Any] | None = None


class DigitalActionResponse(BaseModel):
    accepted: bool
    reason: str | None = None
    action_type: str | None = None
    message: str | None = None
    workspace_root: str | None = None
    result: dict[str, Any] | None = None
    terminus_runtime: dict[str, Any] | None = None
    state_revision: int | None = None


class ActionHistoryResponse(BaseModel):
    count: int
    root_path: str
    supported_actions: list[str]
    actions: list[dict[str, Any]]


class RuntimeFeedbackResponse(BaseModel):
    accepted: bool
    target_type: str
    target_id: str
    feedback: dict[str, Any]
    target: dict[str, Any]
    dirty_state: bool
    state_revision: int
    terminus_runtime: dict[str, Any] | None = None


class PolicyActuatorReason(BaseModel):
    code: str
    detail: str


class PolicyActuatorResponse(BaseModel):
    schema_version: int
    recommendation: str
    action: str
    reasons: list[PolicyActuatorReason]
    risk: float
    expected_information_gain: float
    expected_goal_progress: float
    expected_cost: float
    uncertainty: float
    advisory: bool = True
    executable: bool = False
    target_episode_id: str | None = None
    target_action_id: str | None = None
    action_id: str | None = None
    suggested_endpoint: str
    suggested_input: dict[str, Any]
    input: dict[str, Any]
    created_at: str
    subcortical_control_candidates: dict[str, Any] | None = None
    subcortical_self_repair_candidates: dict[str, Any] | None = None
    snn_sleep_plasticity_autonomy_proposal: dict[str, Any] | None = None
    snn_sleep_plasticity_scheduler_installation_autonomy_proposal: dict[str, Any] | None = None
    snn_due_cycle_bounded_replay_selection_proposal: dict[str, Any] | None = None


class ReplayCandidateResponse(BaseModel):
    candidate_id: str
    rank: int
    target_type: str
    target_id: str
    target_ids: list[str]
    operation: str
    created_at: str
    completed_at: str
    reason_codes: list[str]
    priority_score: float
    priority_components: dict[str, float]
    suggested_consolidation_action: str
    suggested_endpoint: str
    suggested_input: dict[str, Any]
    summary: str
    provenance: dict[str, Any]
    risk: float
    uncertainty: float
    latency: dict[str, Any]
    memory_health: dict[str, Any]
    feedback: dict[str, Any]
    policy: dict[str, Any]


class ReplayPlanResponse(BaseModel):
    schema_version: int
    generated_at: str
    advisory: bool = True
    executable: bool = False
    endpoint: str
    limit: int
    count: int
    state_revision: int
    token_count: int
    snapshot_counts: dict[str, int]
    priority_rules_version: str
    priority_weights: dict[str, float]
    plan_reason_codes: list[str]
    candidates: list[ReplayCandidateResponse]


class ReplaySampleRequest(BaseModel):
    mode: Literal["dry_run", "sample"] = "sample"
    candidate_id: str | None = Field(None, min_length=1, max_length=160)
    target_type: str | None = Field(None, min_length=1, max_length=64)
    target_id: str | None = Field(None, min_length=1, max_length=160)
    operator_id: str = Field(..., min_length=1, max_length=160)
    operator_note: str | None = Field(None, max_length=2000)
    confirmation: bool = False
    limit: int | None = Field(None, ge=1, le=20)
    count: int | None = Field(None, ge=1, le=20)
    alpha: float = Field(1.0, ge=0.0, le=4.0)
    seed: int | None = None


class ReplaySampleResponse(BaseModel):
    schema_version: int
    replay_sample_id: str
    created_at: str
    mode: Literal["dry_run", "sample"]
    status: str
    reason: str
    endpoint: str
    operator_id: str
    operator_note: str
    requested_candidate_id: str | None = None
    target_type: str | None = None
    target_id: str | None = None
    requested_count: int
    alpha: float
    seed: int | None = None
    candidate_ids: list[str]
    selected_candidate_ids: list[str]
    selected_candidates: list[dict[str, Any]]
    safety_checks: dict[str, Any]
    safety_flags: dict[str, Any]
    before: dict[str, int]
    after: dict[str, int]
    plan_summary: dict[str, Any]


class ReplaySampleHistoryResponse(BaseModel):
    schema_version: int
    endpoint: str
    count: int
    limit: int
    history: list[ReplaySampleResponse]


class ResponseBundle(BaseModel):
    trace_id: str
    trace_path: str
    created_at: str
    query_result: dict[str, Any]
    response: dict[str, Any]
    learning: dict[str, Any] | None
    runtime_episode: dict[str, Any] | None = None
    dirty_state: bool
    state_revision: int


class TraceHistoryResponse(BaseModel):
    traces: list[dict[str, Any]]


class RuntimeTraceExportResponse(BaseModel):
    export_kind: str
    schema_version: int
    training_role: str
    description: str
    limit: int
    max_limit: int
    endpoint: str | None = None
    count: int
    policy_decision: dict[str, Any] | None = None
    replay_plan_summary: dict[str, Any] | None = None
    replay_sample_summary: dict[str, Any] | None = None
    replay_dataset_summary: dict[str, Any] | None = None
    examples: list[dict[str, Any]]
    excluded_fields: list[str]


class ReplayDatasetPreviewResponse(BaseModel):
    schema_version: int
    export_kind: str
    training_role: str
    description: str
    created_at: str
    latest_export_timestamp: str | None = None
    latest_history_timestamp: str | None = None
    endpoint: str
    limit: int
    max_limit: int
    filter_endpoint: str | None = None
    count: int
    positive_count: int
    negative_count: int
    provenance_counts: dict[str, int]
    example_type_counts: dict[str, int]
    policy_decision: dict[str, Any] | None = None
    replay_plan_summary: dict[str, Any] | None = None
    replay_sample_summary: dict[str, Any] | None = None
    safety_flags: dict[str, Any]
    before: dict[str, int]
    after: dict[str, int]
    items: list[dict[str, Any]]
    excluded_fields: list[str]
    empty_reason: str | None = None


class ReplayDatasetCandidatesResponse(BaseModel):
    schema_version: int
    export_kind: str
    training_role: str
    created_at: str
    endpoint: str
    limit: int
    max_limit: int
    count: int
    candidates: list[dict[str, Any]]
    replay_plan_summary: dict[str, Any] | None = None
    safety_flags: dict[str, Any]
    excluded_fields: list[str]


class ReplayDatasetHistoryResponse(BaseModel):
    schema_version: int
    export_kind: str
    training_role: str
    created_at: str
    endpoint: str
    source_endpoint: str
    limit: int
    max_limit: int
    count: int
    history: list[dict[str, Any]]
    replay_sample_summary: dict[str, Any] | None = None
    safety_flags: dict[str, Any]
    excluded_fields: list[str]


class ReplayDatasetBundleRequest(BaseModel):
    operator_id: str = Field(..., min_length=1, max_length=160)
    operator_note: str | None = Field(None, max_length=2000)
    confirmation: bool = False
    limit: int = Field(20, ge=1, le=50)
    endpoint: str | None = Field(None, min_length=1, max_length=32)
    holdout_fraction: float = Field(0.2, ge=0.0, le=0.8)
    eval_fraction: float = Field(0.2, ge=0.0, le=0.8)
    seed: int | None = None
    retention_days: int = Field(3650, ge=0)
    decontamination_terms: list[str] | None = None


class ReplayDatasetBundleResponse(BaseModel):
    schema_version: int
    export_kind: str
    training_role: str
    description: str
    created_at: str
    endpoint: str
    source_endpoint: str
    limit: int
    max_limit: int
    filter_endpoint: str | None = None
    bundle_id: str
    bundle_version: str
    bundle_hash: str
    source_preview_hash: str
    operator_approval: dict[str, Any]
    packaging_policy: dict[str, Any]
    source_count: int
    count: int
    excluded_count: int
    positive_count: int
    negative_count: int
    preference_pair_count: int
    sft_count: int
    negative_only_count: int
    split_counts: dict[str, int]
    split_summaries: dict[str, dict[str, Any]]
    source_preview_summary: dict[str, Any]
    manifest: dict[str, Any]
    training_gate: dict[str, Any]
    splits: dict[str, list[dict[str, Any]]]
    excluded_items: list[dict[str, Any]]
    safety_flags: dict[str, Any]
    before: dict[str, int]
    after: dict[str, int]
    excluded_fields: list[str]
    empty_reason: str | None = None


class QueryResponse(BaseModel):
    query_summary: dict[str, Any]
    concept_summary: dict[str, Any]
    gap_plan: dict[str, Any]
    service_state: dict[str, Any]
    runtime_episode: dict[str, Any] | None = None
