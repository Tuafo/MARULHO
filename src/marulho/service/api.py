from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .manager import MarulhoServiceManager
from .runtime_evidence import MAX_RUNTIME_TRACE_EXPORT_LIMIT
from .runtime_control import RuntimeControl
from .schemas import (
    ActionHistoryResponse,
    CheckpointActionResponse,
    CheckpointListResponse,
    CheckpointRecord,
    CheckpointRestoreRequest,
    CheckpointSaveRequest,
    DigitalActionRequest,
    DigitalActionResponse,
    FeedRequest,
    FeedResponse,
    PolicyActuatorResponse,
    QueryRequest,
    QueryResponse,
    ReplayDatasetBundleRequest,
    ReplayDatasetBundleResponse,
    ReplayDatasetPreviewResponse,
    ReplayPlanResponse,
    ReplaySampleHistoryResponse,
    ReplaySampleRequest,
    ReplaySampleResponse,
    RespondRequest,
    ResponseBundle,
    RuntimeFeedbackRequest,
    RuntimeFeedbackResponse,
    RuntimeTraceExportResponse,
    SNNLanguageHeldoutEvaluationRequest,
    SNNLanguagePlasticityApplicationDesignRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewDesignRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewExecutorRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewPreflightRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationObservationWindowRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationStabilityReviewRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationApplicationReviewRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationDesignRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationExecutorRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationPreflightRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseDesignRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseExecutorRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseEventReviewRequest,
    SNNLanguageCalibratedDenseLabelConfidenceAutonomousUsePreflightRequest,
    SNNLanguageAutonomousHashReadoutBindingDesignRequest,
    SNNLanguageAutonomousHashReadoutBindingExecutorRequest,
    SNNLanguageAutonomousHashReadoutBindingEventReviewRequest,
    SNNLanguageAutonomousHashReadoutBindingPreflightRequest,
    SNNLanguageAutonomousDecoderProbeDesignRequest,
    SNNLanguageAutonomousDecoderProbeEventReviewRequest,
    SNNLanguageAutonomousDecoderProbeExecutorRequest,
    SNNLanguageAutonomousDecoderProbePreflightRequest,
    SNNLanguageAutonomousDecodedOutputDesignRequest,
    SNNLanguageAutonomousDecodedOutputExecutorRequest,
    SNNLanguageAutonomousDecodedOutputEventReviewRequest,
    SNNLanguageAutonomousDecodedOutputPreflightRequest,
    SNNLanguageAutonomousBoundedTextEmissionDesignRequest,
    SNNLanguageAutonomousBoundedTextEmissionExecutorRequest,
    SNNLanguageAutonomousBoundedTextEmissionEventReviewRequest,
    SNNLanguageAutonomousBoundedTextEmissionPreflightRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceCommitDesignRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceCommitExecutorRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceCommitEventReviewRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceCommitPreflightRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceReviewRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceUseExecutorRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceUseEventReviewRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceUsePreflightRequest,
    SNNLanguageAutonomousBoundedLanguageSurfaceUseReviewRequest,
    SNNLanguageAutonomousSNNLanguageGenerationDesignRequest,
    SNNLanguageAutonomousSNNLanguageDecodingDesignRequest,
    SNNLanguageAutonomousSNNLanguageDecodingExecutorRequest,
    SNNLanguageAutonomousSNNLanguageDecodingEventReviewRequest,
    SNNLanguageAutonomousSNNLanguageDecodingPreflightRequest,
    SNNLanguageConsolidationDesignRequest,
    SNNLanguageConsolidationEventReviewRequest,
    SNNLanguageConsolidationExecutorRequest,
    SNNLanguageConsolidationPreflightRequest,
    SNNLanguageCapacityMutationDesignRequest,
    SNNLanguageCapacityMutationEventReviewRequest,
    SNNLanguageCapacityMutationExecutorRequest,
    SNNLanguageCapacityMutationPreflightRequest,
    SNNLanguageNewbornNeuronIntegrationDesignRequest,
    SNNLanguageNewbornNeuronIntegrationEventReviewRequest,
    SNNLanguageNewbornNeuronIntegrationExecutorRequest,
    SNNLanguageNewbornNeuronIntegrationPreflightRequest,
    SNNLanguageNewbornNeuronCriticalPeriodLearningDesignRequest,
    SNNLanguageNewbornNeuronCriticalPeriodLearningContinuationDesignRequest,
    SNNLanguageNewbornNeuronCriticalPeriodLearningEventReviewRequest,
    SNNLanguageNewbornNeuronCriticalPeriodLearningExecutorRequest,
    SNNLanguageNewbornNeuronCriticalPeriodLearningPreflightRequest,
    SNNLanguageNewbornNeuronMaturationOutcomeReviewRequest,
    SNNLanguageNewbornSynapsePruningDesignRequest,
    SNNLanguageNewbornSynapsePruningExecutorRequest,
    SNNLanguageNewbornSynapsePruningPreflightRequest,
    SNNLanguageStructuralPlasticityDesignRequest,
    SNNLanguageStructuralPlasticityExecutorRequest,
    SNNLanguageStructuralPlasticityEventReviewRequest,
    SNNLanguageStructuralPlasticityPreflightRequest,
    SNNLanguageMemoryDesignRequest,
    SNNLanguageMemoryExecutorRequest,
    SNNLanguageMemoryEventReviewRequest,
    SNNLanguageMemoryPreflightRequest,
    SNNLanguageSurfaceDesignRequest,
    SNNLanguageSurfaceExecutorRequest,
    SNNLanguageSurfaceEventReviewRequest,
    SNNLanguageSurfacePreflightRequest,
    SNNLanguageAutonomousSNNLanguageGenerationExecutorRequest,
    SNNLanguageAutonomousSNNLanguageGenerationEventReviewRequest,
    SNNLanguageAutonomousSNNLanguageGenerationPreflightRequest,
    SNNLanguageAutonomousTextSurfaceCommitDesignRequest,
    SNNLanguageAutonomousTextSurfaceCommitExecutorRequest,
    SNNLanguageAutonomousTextSurfaceCommitEventReviewRequest,
    SNNLanguageAutonomousTextSurfaceCommitPreflightRequest,
    SNNLanguageAutonomousTextSurfaceMaterializationDesignRequest,
    SNNLanguageAutonomousTextSurfaceMaterializationExecutorRequest,
    SNNLanguageAutonomousTextSurfaceMaterializationEventReviewRequest,
    SNNLanguageAutonomousTextSurfaceMaterializationPreflightRequest,
    SNNLanguageAutonomousTextSurfaceSequenceReviewRequest,
    SNNLanguageAutonomousLanguageOutputDesignRequest,
    SNNLanguageAutonomousLanguageOutputExecutorRequest,
    SNNLanguageAutonomousLanguageOutputEventReviewRequest,
    SNNLanguageAutonomousLanguageOutputPreflightRequest,
    SNNLanguageAutonomousBoundReadoutObservationDesignRequest,
    SNNLanguageAutonomousBoundReadoutObservationEventReviewRequest,
    SNNLanguageAutonomousBoundReadoutObservationExecutorRequest,
    SNNLanguageAutonomousBoundReadoutObservationPreflightRequest,
    SNNLanguageAutonomousReadoutTrainingWindowDesignRequest,
    SNNLanguageAutonomousReadoutTrainingWindowEventReviewRequest,
    SNNLanguageAutonomousReadoutTrainingWindowExecutorRequest,
    SNNLanguageAutonomousReadoutTrainingWindowPreflightRequest,
    SNNLanguageCalibratedDenseLabelConfidenceInternalStabilityReviewRequest,
    SNNLanguageCalibratedDenseLabelConfidenceOperatorDisplayReviewRequest,
    SNNLanguageCalibratedDenseLabelConfidenceUseDesignRequest,
    SNNLanguageCalibratedDenseLabelConfidenceUseExecutorRequest,
    SNNLanguageCalibratedDenseLabelConfidenceUsePreflightRequest,
    SNNLanguageCapacityExpansionDesignRequest,
    SNNLanguageCapacityExpansionPreflightRequest,
    SNNLanguageCapacityResizeCompatibilityAuditRequest,
    SNNLanguageDenseReadoutLayoutMigrationRequest,
    SNNLanguageDenseReadoutResizeExecutorReadinessAuditRequest,
    SNNLanguageDenseReadoutResizePreflightRequest,
    SNNLanguageDenseReadoutResizePlanRequest,
    SNNLanguageDenseReadoutTensorMaterializationRequest,
    SNNLanguageDenseReadoutTensorMaterializationReadinessRequest,
    SNNLanguageDenseReadoutDecoderProbeDesignRequest,
    SNNLanguageDenseReadoutDecoderProbeExecutionRequest,
    SNNLanguageDenseReadoutDecoderProbePreflightRequest,
    SNNLanguageDenseLabelCandidateCalibrationEvaluationRequest,
    SNNLanguageDenseLabelCandidateCalibrationEvaluationDesignRequest,
    SNNLanguageDenseLabelCandidateCalibrationEvaluationPreflightRequest,
    SNNLanguageDenseLabelCandidateCalibrationEvaluationReviewRequest,
    SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationRequest,
    SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationReviewRequest,
    SNNLanguageDenseLabelCandidateCalibrationUpdateDesignRequest,
    SNNLanguageDenseLabelCandidateCalibrationUpdatePreflightRequest,
    SNNLanguageDenseLabelCandidatePostCalibrationObservationWindowRequest,
    SNNLanguageDenseLabelCandidatePostCalibrationOperatorReviewRequest,
    SNNLanguageDenseReadoutLabelCandidateEvidenceRecordRequest,
    SNNLanguageDenseReadoutLabelCandidateReviewRequest,
    SNNLanguageDenseReadoutPostTrainingEvaluationRequest,
    SNNLanguageDenseReadoutTrainingLoopDesignRequest,
    SNNLanguageDenseReadoutTrainingLoopPreflightRequest,
    SNNLanguageDenseReadoutTrainingRequest,
    SNNLanguageDenseReadoutTrainingReadinessRequest,
    SNNLanguageDenseReadoutResizeTransactionProposalRequest,
    SNNLanguagePlasticityLiveApplicationPreflightRequest,
    SNNLanguagePlasticityLiveApplicationRequest,
    SNNLanguagePlasticityLiveApplicationReadinessRequest,
    SNNLanguagePlasticityPressureRequest,
    SNNLanguagePlasticityReplayEvaluationRequest,
    SNNLanguagePlasticityReplayExperimentRequest,
    SNNLanguageReadoutDraftRequest,
    SNNLanguageReadoutEmissionRequest,
    SNNLanguageReadoutEmissionReplayContextReviewRequest,
    SNNLanguageReadoutEmissionReplayEvaluationDesignRequest,
    SNNLanguageReadoutEmissionReviewRequest,
    SNNLanguageReadoutLedgerRecordRequest,
    SNNLanguageReadoutPlasticityPreflightRequest,
    SNNLanguageReadoutPlasticityReplayBridgeRequest,
    SNNLanguageReadoutRehearsalEvaluationRequest,
    SNNLanguageReadoutRehearsalExperimentRequest,
    SNNLanguageReadoutRolloutCandidateRequest,
    SNNLanguageReadoutRolloutConsolidationDesignRequest,
    SNNLanguageReadoutRolloutConsolidationShadowApplicationPreflightRequest,
    SNNLanguageReadoutRolloutConsolidationShadowDeltaRequest,
    SNNLanguageReadoutRolloutDevelopmentalPlasticityReviewRequest,
    SNNLanguageReadoutRolloutRegenerationApplicationRequest,
    SNNLanguageReadoutRolloutLedgerRecordRequest,
    SNNLanguageReadoutRolloutRegenerationApplicationPreflightRequest,
    SNNLanguageReadoutRolloutRegenerationProposalAdapterRequest,
    SNNLanguageReadoutRolloutRegenerationPermitRequestRequest,
    SNNLanguageReadoutRolloutRegenerationReplayArtifactReviewRequest,
    SNNLanguageReadoutRolloutReplayEvaluationRequest,
    SNNLanguageReadoutRolloutRehearsalEvaluationRequest,
    SNNLanguageReadoutRolloutRehearsalExperimentRequest,
    SNNLanguageReadoutReplayDesignRequest,
    SNNLanguageReadoutReplayDryRunRequest,
    SNNLanguageTransitionMemoryPredictionEvaluationRequest,
    SNNLanguagePlasticityShadowApplicationRequest,
    SNNLanguagePlasticityShadowDeltaRequest,
    SNNLanguagePlasticityTrialRequest,
    SNNLanguageTrainingReadinessRequest,
    SNNLanguageSequenceMismatchRequest,
    SNNLanguageSequencePredictionRequest,
    SNNLanguageTransitionMemoryHomeostaticMaintenanceRequest,
    SNNLanguageTransitionMemoryRegenerationProposalRequest,
    SNNLanguageTransitionMemoryRegenerationPermitRequest,
    SNNLanguageTransitionMemoryRegenerationRequest,
    SNNLanguageTransitionMemorySleepPolicyRequest,
    SNNSleepPlasticityReviewTicketRequest,
    SNNSleepPlasticitySchedulerDesignReviewTicketRequest,
    SNNSleepPlasticityReviewSchedulerInstallationRequest,
    SNNSleepPlasticityReviewSchedulerCycleAcknowledgmentRequest,
    SNNEvaluatedTransitionMemoryReplayArtifactRequest,
    SNNDueCycleReplayArtifactRecordingReviewTicketRequest,
    SNNReplayArtifactRecordingReviewTicketRequest,
    SNNReplayEvaluationContextRequest,
    SNNTransitionMemoryReplayArtifactProposalRequest,
    SNNLanguageTrainerDryRunRequest,
    SNNLanguageTrainerEvaluationRequest,
    StructuralPlasticityIsolatedEvaluationRequest,
    StructuralMutationApplicationRequest,
    StructuralMutationDesignRequest,
    StructuralMutationPreflightRequest,
    StatusResponse,
    TerminusConfigureRequest,
    TerminusRuntimeResponse,
    TerminusTickRequest,
    TraceHistoryResponse,
)
from .terminus_hf_sources import current_runtime_datasets


DEFAULT_WEB_DIST_DIR = Path("MARULHO_UI") / "dist"
REPORT_SUMMARY_KINDS = {
    "terminus_multi_hour_live_validation",
    "terminus_bounded_self_improvement_readiness",
    "terminus_live_long_run_validation",
    "terminus_approved_action_level2",
    "terminus_service_benchmark",
    "marulho_service_benchmark_regression_gate",
    "marulho_service_benchmark_accepted_baseline",
    "marulho_service_benchmark_baseline_run_bundle",
}
BENCHMARK_EVIDENCE_FRESH_HOURS = 24.0
BENCHMARK_EVIDENCE_WARN_HOURS = 72.0


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_report_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _apply_benchmark_freshness(summary: dict[str, Any], *, generated_at: Any) -> None:
    generated = _parse_report_datetime(generated_at)
    summary["evidence_freshness_max_age_hours"] = BENCHMARK_EVIDENCE_FRESH_HOURS
    summary["evidence_stale_after_hours"] = BENCHMARK_EVIDENCE_WARN_HOURS
    if generated is None:
        summary["evidence_age_hours"] = None
        summary["evidence_freshness_status"] = "unknown_timestamp"
        summary["evidence_freshness_hint"] = (
            "Report has no parseable generated_at timestamp; rerun the benchmark slow path before using it as current evidence."
        )
        return
    age_hours = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0)
    summary["evidence_age_hours"] = round(age_hours, 3)
    if age_hours <= BENCHMARK_EVIDENCE_FRESH_HOURS:
        summary["evidence_freshness_status"] = "fresh"
        summary["evidence_freshness_hint"] = "Benchmark evidence is fresh enough for current operator review."
    elif age_hours <= BENCHMARK_EVIDENCE_WARN_HOURS:
        summary["evidence_freshness_status"] = "aging"
        summary["evidence_freshness_hint"] = (
            "Benchmark evidence is older than the fresh window; consider a fresh slow-path run before relying on it."
        )
    else:
        summary["evidence_freshness_status"] = "stale"
        summary["evidence_freshness_hint"] = (
            "Benchmark evidence is stale; rerun the accepted-baseline benchmark bundle before treating hot-path evidence as current."
        )


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    return getattr(model, "dict")()


_PUBLIC_SNN_LANGUAGE_NAME_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("terminus_snn_language_autonomous_snn_language_thought", "terminus_snn_language_readout"),
    ("snn_language_autonomous_snn_language_thought", "snn_language_readout"),
    ("autonomous_snn_language_thought", "snn_language_readout"),
    ("language_thought", "language_readout"),
    ("thought_capacity", "readout_capacity"),
    ("thought_newborn", "readout_newborn"),
    ("thought_trace", "readout_trace"),
    ("thought_driven", "readout_driven"),
    ("missing_thought", "missing_readout"),
    ("thought-capacity", "readout-capacity"),
)
_INTERNAL_SNN_LANGUAGE_NAME_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("terminus_snn_language_readout_newborn", "terminus_snn_language_autonomous_snn_language_thought_newborn"),
    ("terminus_snn_language_readout_capacity", "terminus_snn_language_autonomous_snn_language_thought_capacity"),
    ("snn_language_readout_newborn", "snn_language_autonomous_snn_language_thought_newborn"),
    ("snn_language_readout_capacity", "snn_language_autonomous_snn_language_thought_capacity"),
    ("readout_capacity", "thought_capacity"),
    ("readout_newborn", "thought_newborn"),
    ("readout_trace", "thought_trace"),
    ("readout_driven", "thought_driven"),
    ("missing_readout", "missing_thought"),
    ("readout-capacity", "thought-capacity"),
)


def _replace_snn_language_names(value: str, replacements: tuple[tuple[str, str], ...]) -> str:
    result = value
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def _map_snn_language_names(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, dict):
        mapped: dict[str, Any] = {}
        for key, item in value.items():
            next_key = _replace_snn_language_names(str(key), replacements)
            next_value = _map_snn_language_names(item, replacements)
            mapped[next_key] = next_value
            if replacements is _INTERNAL_SNN_LANGUAGE_NAME_REPLACEMENTS and next_key.startswith(
                "snn_language_autonomous_snn_language_thought"
            ):
                mapped[
                    next_key.replace(
                        "snn_language_autonomous_snn_language_thought",
                        "autonomous_snn_language_thought",
                        1,
                    )
                ] = next_value
        return mapped
    if isinstance(value, list):
        return [_map_snn_language_names(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_snn_language_names(item, replacements) for item in value)
    if isinstance(value, str):
        return _replace_snn_language_names(value, replacements)
    return value


def _public_snn_language_payload(value: Any) -> Any:
    return _map_snn_language_names(value, _PUBLIC_SNN_LANGUAGE_NAME_REPLACEMENTS)


def _internal_snn_language_payload(value: Any) -> Any:
    return _map_snn_language_names(value, _INTERNAL_SNN_LANGUAGE_NAME_REPLACEMENTS)


def _report_root(manager: MarulhoServiceManager) -> Path:
    env_root = getattr(manager, "_env_root", None)
    return ((Path(env_root) if env_root is not None else Path.cwd()) / "reports").resolve()


def _safe_report_path(manager: MarulhoServiceManager, path_value: str) -> Path:
    root = _report_root(manager)
    candidate = (root / path_value).resolve()
    if root != candidate and root not in candidate.parents:
        raise HTTPException(status_code=400, detail="Report path must stay inside the reports directory.")
    if candidate.suffix.lower() not in {".json", ".md"}:
        raise HTTPException(status_code=400, detail="Only JSON and Markdown reports can be read.")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return candidate


def _summarize_report(path: Path, root: Path) -> dict[str, Any]:
    relative_path = path.relative_to(root).as_posix()
    stat = path.stat()
    summary: dict[str, Any] = {
        "path": relative_path,
        "file_name": path.name,
        "modified_at": datetime_from_timestamp(stat.st_mtime),
        "size_bytes": stat.st_size,
    }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        summary["artifact_kind"] = "unreadable_json_report"
        summary["status"] = "unreadable"
        return summary
    if not isinstance(payload, dict):
        summary["artifact_kind"] = "unknown_json_report"
        summary["status"] = "unknown"
        return summary
    summary.update(
        {
            "artifact_kind": payload.get("artifact_kind", ""),
            "status": payload.get("status", payload.get("health_verdict", "")),
            "passed": payload.get("passed", payload.get("approved", None)),
            "generated_at": payload.get("generated_at", payload.get("end_time", "")),
            "health_verdict": payload.get("health_verdict", ""),
            "runtime_truth_verdict": payload.get("runtime_truth_verdict", payload.get("final_runtime_truth", {}).get("verdict", "") if isinstance(payload.get("final_runtime_truth"), dict) else ""),
            "recommended_operator_action": payload.get("recommended_operator_action", ""),
            "readme_path": (path.parent / "README.md").relative_to(root).as_posix()
            if (path.parent / "README.md").exists()
            else "",
        }
    )
    operator_report = payload.get("operator_visible_report")
    if isinstance(operator_report, dict):
        summary["summary"] = operator_report.get("summary", "")
    if payload.get("artifact_kind") in {
        "marulho_service_benchmark_regression_gate",
        "marulho_service_benchmark_accepted_baseline",
        "marulho_service_benchmark_baseline_run_bundle",
    }:
        _apply_benchmark_freshness(summary, generated_at=payload.get("generated_at"))
    if payload.get("artifact_kind") == "marulho_service_benchmark_regression_gate":
        runtime_truth = payload.get("runtime_truth")
        hot_path = payload.get("hot_path")
        endpoint_grouping = payload.get("endpoint_grouping")
        configured_source = payload.get("configured_source")
        accepted_baseline = payload.get("accepted_baseline")
        checks = payload.get("checks")
        if isinstance(runtime_truth, dict):
            summary["runtime_truth_verdict"] = runtime_truth.get("after", "")
            summary["runtime_truth_before"] = runtime_truth.get("before", "")
            summary["runtime_truth_regressed"] = bool(runtime_truth.get("regressed", False))
        if isinstance(hot_path, dict):
            summary["hot_path_p95_ms"] = hot_path.get("after_p95_ms")
            summary["hot_path_total_ms"] = hot_path.get("after_total_ms")
            summary["hot_path_allowed_p95_ms"] = hot_path.get("allowed_after_p95_ms")
            summary["hot_path_allowed_total_ms"] = hot_path.get("allowed_after_total_ms")
            summary["hot_path_regression_tolerance"] = hot_path.get("regression_tolerance")
        if isinstance(endpoint_grouping, dict):
            summary["setup_leaked_into_hot_path"] = bool(endpoint_grouping.get("setup_leaked_into_hot_path", False))
            summary["slow_path_leaked_into_hot_path"] = bool(endpoint_grouping.get("slow_path_leaked_into_hot_path", False))
        if isinstance(configured_source, dict):
            summary["configured_source"] = configured_source.get("source_name")
            summary["configured_source_tick_tokens"] = configured_source.get("tick_tokens_processed")
        if isinstance(accepted_baseline, dict):
            summary["accepted_baseline_id"] = accepted_baseline.get("baseline_id", "")
            summary["accepted_baseline_label"] = accepted_baseline.get("label", "")
            summary["accepted_baseline_by"] = accepted_baseline.get("accepted_by", "")
            summary["accepted_baseline_path"] = accepted_baseline.get("baseline_path", "")
        if isinstance(checks, dict):
            summary["failed_checks"] = sorted(str(name) for name, passed in checks.items() if not bool(passed))
    if payload.get("artifact_kind") == "marulho_service_benchmark_accepted_baseline":
        operator_review = payload.get("operator_review")
        source_report = payload.get("source_report")
        baseline_snapshot = payload.get("baseline_report_snapshot")
        checks = payload.get("checks")
        acceptance_integrity_status = "legacy_unbound"
        acceptance_hash_match = None
        if isinstance(operator_review, dict):
            summary["accepted_baseline_by"] = operator_review.get("accepted_by", "")
            summary["accepted_baseline_note"] = operator_review.get("note", "")
            summary["accepted_baseline_at"] = operator_review.get("accepted_at", "")
            acceptance_hash = str(operator_review.get("acceptance_hash", "") or "")
            acceptance_material = operator_review.get("acceptance_material")
            summary["acceptance_hash"] = acceptance_hash
            if acceptance_hash:
                actual_acceptance_hash = (
                    _sha256_json(acceptance_material)
                    if isinstance(acceptance_material, dict)
                    else ""
                )
                acceptance_hash_match = bool(
                    actual_acceptance_hash and actual_acceptance_hash == acceptance_hash
                )
                acceptance_integrity_status = "verified" if acceptance_hash_match else "failed"
                summary["acceptance_material_hash"] = actual_acceptance_hash
                summary["acceptance_hash_match"] = acceptance_hash_match
            else:
                summary["acceptance_hash_match"] = None
            summary["acceptance_integrity_status"] = acceptance_integrity_status
        if isinstance(source_report, dict):
            expected_hash = str(source_report.get("sha256_canonical_json", "") or "")
            actual_hash = _sha256_json(baseline_snapshot) if isinstance(baseline_snapshot, dict) else ""
            hash_match = bool(expected_hash and actual_hash and expected_hash == actual_hash)
            summary["accepted_baseline_id"] = payload.get("baseline_id", "")
            summary["accepted_baseline_label"] = payload.get("label", "")
            summary["baseline_report_hash"] = expected_hash
            summary["baseline_snapshot_hash"] = actual_hash
            summary["baseline_hash_match"] = hash_match
            summary["baseline_integrity_status"] = "verified" if hash_match else "failed"
            if hash_match:
                if acceptance_integrity_status == "verified":
                    summary["baseline_operator_action_hint"] = (
                        "Baseline and operator approval integrity verified; use --run-against-baseline for fresh slow-path comparisons."
                    )
                    summary["baseline_operator_action_commands"] = [
                        "python -m marulho.evaluation.service_benchmark --run-against-baseline <accepted-baseline.json> --checkpoint <checkpoint.pt> --output <reports/service_benchmark_baseline_fresh_cycle> --configure-local-source --local-source-tick-steps 1"
                    ]
                elif acceptance_integrity_status == "failed":
                    summary["baseline_operator_action_hint"] = (
                        "Baseline snapshot is intact, but operator approval hash mismatch invalidates this accepted baseline; re-accept from the source report after review."
                    )
                    summary["baseline_operator_action_commands"] = [
                        "python -m marulho.evaluation.service_benchmark --accept-baseline-from <service-benchmark.json> --accepted-by <operator> --baseline-label <label> --output <reports/service_benchmark_baseline/accepted-baseline.json>"
                    ]
                else:
                    summary["baseline_operator_action_hint"] = (
                        "Baseline snapshot integrity verified, but operator approval is legacy-unbound; re-accept the baseline to bind review metadata before using it as a durable anchor."
                    )
                    summary["baseline_operator_action_commands"] = [
                        "python -m marulho.evaluation.service_benchmark --accept-baseline-from <service-benchmark.json> --accepted-by <operator> --baseline-label <label> --output <reports/service_benchmark_baseline/accepted-baseline.json>"
                    ]
            else:
                summary["baseline_operator_action_hint"] = (
                    "Baseline snapshot hash mismatch; treat this baseline as invalid, rerun a fresh configured-source benchmark, then accept a new baseline from that report."
                )
                summary["baseline_operator_action_commands"] = [
                    "python -m marulho.evaluation.service_benchmark --checkpoint <checkpoint.pt> --output <reports/service_benchmark_cycle_configured/service-benchmark.json> --configure-local-source --local-source-tick-steps 1",
                    "python -m marulho.evaluation.service_benchmark --accept-baseline-from <service-benchmark.json> --accepted-by <operator> --baseline-label <label> --output <reports/service_benchmark_baseline/accepted-baseline.json>",
                ]
            summary["runtime_truth_verdict"] = source_report.get("runtime_truth_verdict", "")
            summary["hot_path_p95_ms"] = source_report.get("hot_path_p95_ms")
            summary["hot_path_total_ms"] = source_report.get("hot_path_total_ms")
            summary["source_report_path"] = source_report.get("path", "")
        if isinstance(checks, dict):
            failed_checks = [str(name) for name, passed in checks.items() if not bool(passed)]
            if summary.get("baseline_integrity_status") == "failed":
                failed_checks.append("baseline_snapshot_hash_match")
            if summary.get("acceptance_integrity_status") == "failed":
                failed_checks.append("baseline_acceptance_hash_match")
            summary["failed_checks"] = sorted(set(failed_checks))
    if payload.get("artifact_kind") == "marulho_service_benchmark_baseline_run_bundle":
        runtime_truth = payload.get("runtime_truth")
        hot_path = payload.get("hot_path")
        configured_source = payload.get("configured_source")
        accepted_baseline = payload.get("accepted_baseline")
        paths = payload.get("paths")
        checks = payload.get("checks")
        summary["passed"] = bool(payload.get("success"))
        summary["success"] = bool(payload.get("success"))
        if isinstance(runtime_truth, dict):
            summary["runtime_truth_verdict"] = runtime_truth.get("after", "")
            summary["runtime_truth_before"] = runtime_truth.get("before", "")
            summary["runtime_truth_regressed"] = bool(runtime_truth.get("regressed", False))
        if isinstance(hot_path, dict):
            summary["hot_path_p95_ms"] = hot_path.get("after_p95_ms")
            summary["hot_path_total_ms"] = hot_path.get("after_total_ms")
            summary["hot_path_allowed_p95_ms"] = hot_path.get("allowed_after_p95_ms")
            summary["hot_path_allowed_total_ms"] = hot_path.get("allowed_after_total_ms")
            summary["hot_path_regression_tolerance"] = hot_path.get("regression_tolerance")
        if isinstance(configured_source, dict):
            summary["configured_source"] = configured_source.get("source_name")
            summary["configured_source_tick_tokens"] = configured_source.get("tick_tokens_processed")
            summary["configured_source_count"] = configured_source.get("source_count")
        if isinstance(accepted_baseline, dict):
            summary["accepted_baseline_id"] = accepted_baseline.get("baseline_id", "")
            summary["accepted_baseline_label"] = accepted_baseline.get("label", "")
            summary["accepted_baseline_by"] = accepted_baseline.get("accepted_by", "")
            summary["baseline_report_hash"] = accepted_baseline.get("baseline_report_hash", "")
            summary["after_report_hash"] = accepted_baseline.get("after_report_hash", "")
        if isinstance(paths, dict):
            summary["bundle_dir"] = paths.get("bundle_dir", "")
            summary["fresh_benchmark_path"] = paths.get("benchmark", "")
            summary["comparison_report_path"] = paths.get("comparison", "")
            summary["accepted_baseline_path"] = paths.get("baseline", "")
        if isinstance(checks, dict):
            summary["failed_checks"] = sorted(str(name) for name, passed in checks.items() if not bool(passed))
    return summary


def datetime_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from MARULHO_CORS_ORIGINS env var (comma-separated).

    Falls back to the canonical set of local dev origins when not set.
    """
    env_val = os.environ.get("MARULHO_CORS_ORIGINS", "").strip()
    if env_val:
        return [o.strip() for o in env_val.split(",") if o.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


def create_app(
    checkpoint_path: str | Path,
    trace_history_limit: int = 200,
    trace_dir: str | Path | None = None,
    web_dist_dir: str | Path | None = None,
    env_root: str | Path | None = None,
) -> FastAPI:
    manager = MarulhoServiceManager(
        checkpoint_path=checkpoint_path,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir,
        env_root=env_root,
    )
    app = FastAPI(
        title="MARULHO Local Service",
        version="0.1.0",
        description="Strict-evidence local service for querying and steering a checkpoint-backed MARULHO Terminus runtime.",
    )
    app.state.marulho_manager = manager
    runtime = manager.runtime_facade
    app.state.marulho_runtime = runtime
    app.router.on_shutdown.append(manager.close)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dist_dir = Path(web_dist_dir) if web_dist_dir is not None else DEFAULT_WEB_DIST_DIR
    app.state.web_dist_dir = dist_dir
    if dist_dir.exists():
        app.mount("/app", StaticFiles(directory=dist_dir, html=True), name="app")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        app_hint = '<p>Built frontend available at <a href="/app">/app</a>.</p>' if dist_dir.exists() else ""
        return (
            "<html><body style=\"font-family:Segoe UI, sans-serif; padding: 24px;\">"
            "<h1>MARULHO Local Service</h1>"
            f"{app_hint}"
            "<p>Interactive API docs are available at <a href=\"/docs\">/docs</a>.</p>"
            "</body></html>"
        )

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(**runtime.status())

    @app.get("/checkpoints", response_model=CheckpointListResponse)
    def checkpoints() -> CheckpointListResponse:
        return CheckpointListResponse(checkpoints=[CheckpointRecord(**item) for item in runtime.checkpoint_list()])

    @app.post("/checkpoint/save", response_model=CheckpointActionResponse)
    def save_checkpoint(request: CheckpointSaveRequest) -> CheckpointActionResponse:
        try:
            return CheckpointActionResponse(**runtime.save_checkpoint(request.path))
        except TimeoutError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/checkpoint/restore", response_model=CheckpointActionResponse)
    def restore_checkpoint(request: CheckpointRestoreRequest) -> CheckpointActionResponse:
        return CheckpointActionResponse(**runtime.restore_checkpoint(request.path))

    @app.post("/feed", response_model=FeedResponse)
    def feed(request: FeedRequest) -> FeedResponse:
        return FeedResponse(**runtime.feed(text=request.text))

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = runtime.query(
            query_text=request.query_text,
            context_text=request.context_text,
            top_k_candidates=request.top_k_candidates,
            top_k_memories=request.top_k_memories,
            top_chars=request.top_chars,
        )
        return QueryResponse(
            query_summary=result.get("query_summary") or {},
            concept_summary=result.get("concept_summary") or {},
            gap_plan=result.get("gap_plan") or {},
            service_state=result.get("service_state") or {},
            runtime_episode=result.get("runtime_episode"),
        )

    @app.post("/respond", response_model=ResponseBundle)
    def respond(request: RespondRequest) -> ResponseBundle:
        try:
            return ResponseBundle(
                **runtime.respond(
                    query_text=request.query_text,
                    context_text=request.context_text,
                    top_k_candidates=request.top_k_candidates,
                    top_k_memories=request.top_k_memories,
                    top_chars=request.top_chars,
                    max_evidence_items=request.max_evidence_items,
                    learn_mode=request.learn_mode,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/terminus", response_model=TerminusRuntimeResponse)
    def terminus_status() -> TerminusRuntimeResponse:
        return TerminusRuntimeResponse(**runtime.terminus_status())

    @app.get("/terminus/living-loop")
    def terminus_living_loop() -> dict[str, Any]:
        return runtime.living_loop_status()

    @app.get("/terminus/policy-actuator", response_model=PolicyActuatorResponse)
    def terminus_policy_actuator() -> PolicyActuatorResponse:
        return PolicyActuatorResponse(**runtime.policy_actuator_status())

    @app.get("/terminus/cognitive-signal")
    def terminus_cognitive_signal() -> dict[str, Any]:
        return runtime.cognitive_signal_state()

    @app.get("/terminus/subcortical-language")
    def terminus_subcortical_language() -> dict[str, Any]:
        return runtime.subcortical_language_surface()

    @app.get("/terminus/subcortical-deliberation")
    def terminus_subcortical_deliberation() -> dict[str, Any]:
        return runtime.subcortical_deliberation_surface()

    @app.get("/terminus/snn-language-readiness")
    def terminus_snn_language_readiness() -> dict[str, Any]:
        return runtime.snn_language_readiness_surface()

    @app.get("/terminus/snn-language-evaluation")
    def terminus_snn_language_evaluation() -> dict[str, Any]:
        return runtime.snn_language_evaluation_surface()

    @app.post("/terminus/snn-language-evaluation/heldout")
    def terminus_snn_language_heldout_evaluation(request: SNNLanguageHeldoutEvaluationRequest) -> dict[str, Any]:
        return runtime.snn_language_adapter_heldout_evaluation(
            heldout_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.heldout_readout_slot_batches
            ],
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-training/readiness")
    def terminus_snn_language_training_readiness(request: SNNLanguageTrainingReadinessRequest) -> dict[str, Any]:
        return runtime.snn_language_training_readiness(
            heldout_evaluation=request.heldout_evaluation,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-training/dry-run")
    def terminus_snn_language_training_dry_run(request: SNNLanguageTrainerDryRunRequest) -> dict[str, Any]:
        return runtime.snn_language_trainer_dry_run(
            training_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.training_readout_slot_batches
            ],
            validation_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.validation_readout_slot_batches
            ],
            device_evidence=request.device_evidence,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
        )

    @app.post("/terminus/snn-language-training/evaluate")
    def terminus_snn_language_training_evaluation(request: SNNLanguageTrainerEvaluationRequest) -> dict[str, Any]:
        return runtime.snn_language_trainer_isolated_evaluation(
            dry_run_report=request.dry_run_report,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/predict")
    def terminus_snn_language_sequence_prediction(request: SNNLanguageSequencePredictionRequest) -> dict[str, Any]:
        return runtime.snn_language_sequence_prediction_probe(
            training_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.training_readout_slot_batches
            ],
            current_readout_slots=[_model_to_dict(slot) for slot in request.current_readout_slots],
            device_evidence=request.device_evidence,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            top_k=request.top_k,
            persistent_transition_weights=request.persistent_transition_weights,
        )

    @app.post("/terminus/snn-language-sequence/mismatch")
    def terminus_snn_language_sequence_mismatch(request: SNNLanguageSequenceMismatchRequest) -> dict[str, Any]:
        return runtime.snn_language_sequence_mismatch_probe(
            prediction_report=request.prediction_report,
            observed_readout_slots=[_model_to_dict(slot) for slot in request.observed_readout_slots],
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-draft")
    def terminus_snn_language_readout_draft(request: SNNLanguageReadoutDraftRequest) -> dict[str, Any]:
        return runtime.snn_language_readout_draft(
            prediction_report=request.prediction_report,
            readout_vocabulary_slots=[
                _model_to_dict(slot)
                for slot in request.readout_vocabulary_slots
            ],
            device_evidence=request.device_evidence,
            transition_memory_evaluation=request.transition_memory_evaluation,
            max_draft_terms=request.max_draft_terms,
        )

    @app.post("/terminus/snn-language-sequence/readout-emission")
    def terminus_snn_language_readout_emission(request: SNNLanguageReadoutEmissionRequest) -> dict[str, Any]:
        return runtime.snn_language_readout_emission(
            readout_draft=request.readout_draft,
        )

    @app.post("/terminus/snn-language-sequence/readout-emission/operator-review")
    def terminus_snn_language_readout_emission_operator_review(
        request: SNNLanguageReadoutEmissionReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_emission_review_record(
            readout_emission=request.readout_emission,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.get("/terminus/snn-language-sequence/readout-emission/operator-review/history")
    def terminus_snn_language_readout_emission_operator_review_history(
        limit: int = Query(20, ge=0, le=128),
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_emission_review_history(limit=limit)

    @app.get("/terminus/snn-language-sequence/readout-emission/operator-review/replay-evaluation-policy")
    def terminus_snn_language_readout_emission_operator_review_replay_evaluation_policy(
        limit: int = Query(12, ge=0, le=128),
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_emission_replay_evaluation_policy(limit=limit)

    @app.post("/terminus/snn-language-sequence/readout-emission/operator-review/replay-evaluation-design")
    def terminus_snn_language_readout_emission_operator_review_replay_evaluation_design(
        request: SNNLanguageReadoutEmissionReplayEvaluationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_emission_replay_evaluation_design(
            emission_replay_evaluation_policy=request.emission_replay_evaluation_policy,
            design_policy=request.design_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-emission/operator-review/replay-context-review")
    def terminus_snn_language_readout_emission_operator_review_replay_context_review(
        request: SNNLanguageReadoutEmissionReplayContextReviewRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_language_readout_emission_replay_context_review(
                emission_replay_evaluation_design=request.emission_replay_evaluation_design,
                prediction_report=request.prediction_report,
                observed_readout_slots=request.observed_readout_slots,
                device_evidence=request.device_evidence,
                runtime_truth_delta=request.runtime_truth_delta,
                rollback_policy=request.rollback_policy,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/terminus/snn-language-sequence/readout-rollout-candidate")
    def terminus_snn_language_readout_rollout_candidate(
        request: SNNLanguageReadoutRolloutCandidateRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_candidate(
            prediction_report=request.prediction_report,
            readout_vocabulary_slots=[
                _model_to_dict(slot)
                for slot in request.readout_vocabulary_slots
            ],
            device_evidence=request.device_evidence,
            transition_memory_evaluation=request.transition_memory_evaluation,
            rollout_steps=request.rollout_steps,
            top_k=request.top_k,
        )

    @app.post("/terminus/snn-language-sequence/readout-rollout-candidate/replay-evaluation")
    def terminus_snn_language_readout_rollout_replay_evaluation(
        request: SNNLanguageReadoutRolloutReplayEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_replay_evaluation(
            readout_rollout_candidate=request.readout_rollout_candidate,
            candidate_limit=request.candidate_limit,
            device_evidence=request.device_evidence,
        )

    @app.get("/terminus/snn-language-sequence/readout-ledger")
    def terminus_snn_language_readout_ledger(limit: int = Query(20, ge=0, le=128)) -> dict[str, Any]:
        return runtime.snn_language_readout_evidence_ledger(limit=limit)

    @app.get("/terminus/snn-language-sequence/readout-ledger/replay-priority")
    def terminus_snn_language_readout_replay_priority(limit: int = Query(12, ge=0, le=128)) -> dict[str, Any]:
        return runtime.snn_language_readout_replay_priority(limit=limit)

    @app.get("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-history")
    def terminus_snn_language_dense_label_candidate_history(
        limit: int = Query(20, ge=0, le=128),
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_history(limit=limit)

    @app.get("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-policy")
    def terminus_snn_language_dense_label_candidate_calibration_policy(
        limit: int = Query(12, ge=0, le=128),
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_policy(
            limit=limit,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-design")
    def terminus_snn_language_dense_label_candidate_calibration_evaluation_design(
        request: SNNLanguageDenseLabelCandidateCalibrationEvaluationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_evaluation_design(
            dense_label_candidate_calibration_policy=(
                request.dense_label_candidate_calibration_policy
            ),
            heldout_label_evidence=request.heldout_label_evidence,
            design_policy=request.design_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-preflight")
    def terminus_snn_language_dense_label_candidate_calibration_evaluation_preflight(
        request: SNNLanguageDenseLabelCandidateCalibrationEvaluationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_evaluation_preflight(
            dense_label_candidate_calibration_evaluation_design=(
                request.dense_label_candidate_calibration_evaluation_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation")
    def terminus_snn_language_dense_label_candidate_calibration_evaluation(
        request: SNNLanguageDenseLabelCandidateCalibrationEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_evaluation(
            dense_label_candidate_calibration_evaluation_preflight=(
                request.dense_label_candidate_calibration_evaluation_preflight
            ),
            heldout_label_evidence=request.heldout_label_evidence,
            bin_count=request.bin_count,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-review")
    def terminus_snn_language_dense_label_candidate_calibration_evaluation_review(
        request: SNNLanguageDenseLabelCandidateCalibrationEvaluationReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_evaluation_review(
            dense_label_candidate_calibration_evaluation=(
                request.dense_label_candidate_calibration_evaluation
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-design")
    def terminus_snn_language_dense_label_candidate_calibration_update_design(
        request: SNNLanguageDenseLabelCandidateCalibrationUpdateDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_update_design(
            dense_label_candidate_calibration_evaluation_review=(
                request.dense_label_candidate_calibration_evaluation_review
            ),
            update_policy=request.update_policy,
            rollback_policy=request.rollback_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-preflight")
    def terminus_snn_language_dense_label_candidate_calibration_update_preflight(
        request: SNNLanguageDenseLabelCandidateCalibrationUpdatePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_update_preflight(
            dense_label_candidate_calibration_update_design=(
                request.dense_label_candidate_calibration_update_design
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
            rollback_policy=request.rollback_policy,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-application")
    def terminus_snn_language_dense_label_candidate_calibration_update_application(
        request: SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_update_application(
            dense_label_candidate_calibration_update_preflight=(
                request.dense_label_candidate_calibration_update_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-application-review")
    def terminus_snn_language_dense_label_candidate_calibration_update_application_review(
        request: SNNLanguageDenseLabelCandidateCalibrationUpdateApplicationReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_calibration_update_application_review(
            dense_label_candidate_calibration_update_application=(
                request.dense_label_candidate_calibration_update_application
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-post-calibration-observation-window")
    def terminus_snn_language_dense_label_candidate_post_calibration_observation_window(
        request: SNNLanguageDenseLabelCandidatePostCalibrationObservationWindowRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_post_calibration_observation_window(
            dense_label_candidate_calibration_update_application_review=(
                request.dense_label_candidate_calibration_update_application_review
            ),
            observation_evidence=request.observation_evidence,
            expected_state_revision=request.expected_state_revision,
            window_policy=request.window_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-post-calibration-operator-review")
    def terminus_snn_language_dense_label_candidate_post_calibration_operator_review(
        request: SNNLanguageDenseLabelCandidatePostCalibrationOperatorReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_label_candidate_post_calibration_operator_review(
            dense_label_candidate_post_calibration_observation_window=(
                request.dense_label_candidate_post_calibration_observation_window
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-design")
    def terminus_snn_language_calibrated_dense_label_confidence_use_design(
        request: SNNLanguageCalibratedDenseLabelConfidenceUseDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_use_design(
            dense_label_candidate_post_calibration_operator_review=(
                request.dense_label_candidate_post_calibration_operator_review
            ),
            confidence_use_policy=request.confidence_use_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-preflight")
    def terminus_snn_language_calibrated_dense_label_confidence_use_preflight(
        request: SNNLanguageCalibratedDenseLabelConfidenceUsePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_use_preflight(
            dense_label_confidence_use_design=request.dense_label_confidence_use_design,
            expected_state_revision=request.expected_state_revision,
            candidate_evidence=request.candidate_evidence,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-executor")
    def terminus_snn_language_calibrated_dense_label_confidence_use_executor(
        request: SNNLanguageCalibratedDenseLabelConfidenceUseExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_use_executor(
            calibrated_dense_label_confidence_use_preflight=(
                request.calibrated_dense_label_confidence_use_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            candidate_evidence=request.candidate_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-operator-display-review")
    def terminus_snn_language_calibrated_dense_label_confidence_operator_display_review(
        request: SNNLanguageCalibratedDenseLabelConfidenceOperatorDisplayReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_operator_display_review(
            calibrated_dense_label_confidence_use_executor=(
                request.calibrated_dense_label_confidence_use_executor
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-internal-stability-review")
    def terminus_snn_language_calibrated_dense_label_confidence_internal_stability_review(
        request: SNNLanguageCalibratedDenseLabelConfidenceInternalStabilityReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_internal_stability_review(
            calibrated_dense_label_confidence_use_executor=(
                request.calibrated_dense_label_confidence_use_executor
            ),
            expected_state_revision=request.expected_state_revision,
            stability_evidence=request.stability_evidence,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-design")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_replay_review_design(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_replay_review_design(
            calibrated_dense_label_confidence_internal_stability_review=(
                request.calibrated_dense_label_confidence_internal_stability_review
            ),
            replay_policy=request.replay_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-preflight")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_replay_review_preflight(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_replay_review_preflight(
            calibrated_dense_label_confidence_autonomous_replay_review_design=(
                request.calibrated_dense_label_confidence_autonomous_replay_review_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-executor")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_replay_review_executor(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousReplayReviewExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_replay_review_executor(
            calibrated_dense_label_confidence_autonomous_replay_review_preflight=(
                request.calibrated_dense_label_confidence_autonomous_replay_review_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            replay_cycle_evidence=request.replay_cycle_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-design")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_recalibration_design(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_recalibration_design(
            calibrated_dense_label_confidence_autonomous_replay_review_executor=(
                request.calibrated_dense_label_confidence_autonomous_replay_review_executor
            ),
            recalibration_policy=request.recalibration_policy,
            rollback_policy=request.rollback_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-preflight")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_recalibration_preflight(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_recalibration_preflight(
            calibrated_dense_label_confidence_autonomous_recalibration_design=(
                request.calibrated_dense_label_confidence_autonomous_recalibration_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-executor")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_recalibration_executor(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_recalibration_executor(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight=(
                request.calibrated_dense_label_confidence_autonomous_recalibration_preflight
            ),
            expected_state_revision=request.expected_state_revision,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-application-review")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_recalibration_application_review(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousRecalibrationApplicationReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_recalibration_application_review(
            calibrated_dense_label_confidence_autonomous_recalibration_executor=(
                request.calibrated_dense_label_confidence_autonomous_recalibration_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-post-calibration-observation-window")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationObservationWindowRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_observation_window(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review=(
                request.calibrated_dense_label_confidence_autonomous_recalibration_application_review
            ),
            observation_evidence=request.observation_evidence,
            expected_state_revision=request.expected_state_revision,
            window_policy=request.window_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-post-calibration-stability-review")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousPostCalibrationStabilityReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_stability_review(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation_window=(
                request.calibrated_dense_label_confidence_autonomous_post_calibration_observation_window
            ),
            expected_state_revision=request.expected_state_revision,
            stability_policy=request.stability_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-design")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_use_design(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_use_design(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review=(
                request.calibrated_dense_label_confidence_autonomous_post_calibration_stability_review
            ),
            confidence_use_policy=request.confidence_use_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-preflight")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_use_preflight(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousUsePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_use_preflight(
            calibrated_dense_label_confidence_autonomous_use_design=(
                request.calibrated_dense_label_confidence_autonomous_use_design
            ),
            expected_state_revision=request.expected_state_revision,
            candidate_evidence=request.candidate_evidence,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-executor")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_use_executor(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_use_executor(
            calibrated_dense_label_confidence_autonomous_use_preflight=(
                request.calibrated_dense_label_confidence_autonomous_use_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            candidate_evidence=request.candidate_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-event-review")
    def terminus_snn_language_calibrated_dense_label_confidence_autonomous_use_event_review(
        request: SNNLanguageCalibratedDenseLabelConfidenceAutonomousUseEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_calibrated_dense_label_confidence_autonomous_use_event_review(
            calibrated_dense_label_confidence_autonomous_use_executor=(
                request.calibrated_dense_label_confidence_autonomous_use_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-design")
    def terminus_snn_language_autonomous_hash_readout_binding_design(
        request: SNNLanguageAutonomousHashReadoutBindingDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_hash_readout_binding_design(
            calibrated_dense_label_confidence_autonomous_use_event_review=(
                request.calibrated_dense_label_confidence_autonomous_use_event_review
            ),
            readout_vocabulary_slots=[
                item.model_dump() if hasattr(item, "model_dump") else dict(item)
                for item in request.readout_vocabulary_slots
            ],
            binding_policy=request.binding_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-preflight")
    def terminus_snn_language_autonomous_hash_readout_binding_preflight(
        request: SNNLanguageAutonomousHashReadoutBindingPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_hash_readout_binding_preflight(
            autonomous_hash_readout_binding_design=(
                request.autonomous_hash_readout_binding_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-executor")
    def terminus_snn_language_autonomous_hash_readout_binding_executor(
        request: SNNLanguageAutonomousHashReadoutBindingExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_hash_readout_binding_executor(
            autonomous_hash_readout_binding_preflight=(
                request.autonomous_hash_readout_binding_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-event-review")
    def terminus_snn_language_autonomous_hash_readout_binding_event_review(
        request: SNNLanguageAutonomousHashReadoutBindingEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_hash_readout_binding_event_review(
            autonomous_hash_readout_binding_executor=(
                request.autonomous_hash_readout_binding_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-design")
    def terminus_snn_language_autonomous_bound_readout_observation_design(
        request: SNNLanguageAutonomousBoundReadoutObservationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bound_readout_observation_design(
            autonomous_hash_readout_binding_event_review=(
                request.autonomous_hash_readout_binding_event_review
            ),
            observation_policy=request.observation_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-preflight")
    def terminus_snn_language_autonomous_bound_readout_observation_preflight(
        request: SNNLanguageAutonomousBoundReadoutObservationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bound_readout_observation_preflight(
            autonomous_bound_readout_observation_design=(
                request.autonomous_bound_readout_observation_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-executor")
    def terminus_snn_language_autonomous_bound_readout_observation_executor(
        request: SNNLanguageAutonomousBoundReadoutObservationExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bound_readout_observation_executor(
            autonomous_bound_readout_observation_preflight=(
                request.autonomous_bound_readout_observation_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            observation_evidence=request.observation_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-event-review")
    def terminus_snn_language_autonomous_bound_readout_observation_event_review(
        request: SNNLanguageAutonomousBoundReadoutObservationEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bound_readout_observation_event_review(
            autonomous_bound_readout_observation_executor=(
                request.autonomous_bound_readout_observation_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-design")
    def terminus_snn_language_autonomous_readout_training_window_design(
        request: SNNLanguageAutonomousReadoutTrainingWindowDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_readout_training_window_design(
            autonomous_bound_readout_observation_event_review=(
                request.autonomous_bound_readout_observation_event_review
            ),
            training_policy=request.training_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-preflight")
    def terminus_snn_language_autonomous_readout_training_window_preflight(
        request: SNNLanguageAutonomousReadoutTrainingWindowPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_readout_training_window_preflight(
            autonomous_readout_training_window_design=(
                request.autonomous_readout_training_window_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-executor")
    def terminus_snn_language_autonomous_readout_training_window_executor(
        request: SNNLanguageAutonomousReadoutTrainingWindowExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_readout_training_window_executor(
            autonomous_readout_training_window_preflight=(
                request.autonomous_readout_training_window_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            training_evidence=request.training_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-event-review")
    def terminus_snn_language_autonomous_readout_training_window_event_review(
        request: SNNLanguageAutonomousReadoutTrainingWindowEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_readout_training_window_event_review(
            autonomous_readout_training_window_executor=(
                request.autonomous_readout_training_window_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-design")
    def terminus_snn_language_autonomous_decoder_probe_design(
        request: SNNLanguageAutonomousDecoderProbeDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoder_probe_design(
            autonomous_readout_training_window_event_review=(
                request.autonomous_readout_training_window_event_review
            ),
            probe_policy=request.probe_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-preflight")
    def terminus_snn_language_autonomous_decoder_probe_preflight(
        request: SNNLanguageAutonomousDecoderProbePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoder_probe_preflight(
            autonomous_decoder_probe_design=request.autonomous_decoder_probe_design,
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-executor")
    def terminus_snn_language_autonomous_decoder_probe_executor(
        request: SNNLanguageAutonomousDecoderProbeExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoder_probe_executor(
            autonomous_decoder_probe_preflight=(
                request.autonomous_decoder_probe_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            probe_evidence=request.probe_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-event-review")
    def terminus_snn_language_autonomous_decoder_probe_event_review(
        request: SNNLanguageAutonomousDecoderProbeEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoder_probe_event_review(
            autonomous_decoder_probe_executor=request.autonomous_decoder_probe_executor,
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-design")
    def terminus_snn_language_autonomous_language_output_design(
        request: SNNLanguageAutonomousLanguageOutputDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_language_output_design(
            autonomous_decoder_probe_event_review=(
                request.autonomous_decoder_probe_event_review
            ),
            output_policy=request.output_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-preflight")
    def terminus_snn_language_autonomous_language_output_preflight(
        request: SNNLanguageAutonomousLanguageOutputPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_language_output_preflight(
            autonomous_language_output_design=request.autonomous_language_output_design,
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-executor")
    def terminus_snn_language_autonomous_language_output_executor(
        request: SNNLanguageAutonomousLanguageOutputExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_language_output_executor(
            autonomous_language_output_preflight=(
                request.autonomous_language_output_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            output_evidence=request.output_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-event-review")
    def terminus_snn_language_autonomous_language_output_event_review(
        request: SNNLanguageAutonomousLanguageOutputEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_language_output_event_review(
            autonomous_language_output_executor=(
                request.autonomous_language_output_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-design")
    def terminus_snn_language_autonomous_decoded_output_design(
        request: SNNLanguageAutonomousDecodedOutputDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoded_output_design(
            autonomous_language_output_event_review=(
                request.autonomous_language_output_event_review
            ),
            vocabulary_binding=request.vocabulary_binding,
            decode_policy=request.decode_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-preflight")
    def terminus_snn_language_autonomous_decoded_output_preflight(
        request: SNNLanguageAutonomousDecodedOutputPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoded_output_preflight(
            autonomous_decoded_output_design=request.autonomous_decoded_output_design,
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-executor")
    def terminus_snn_language_autonomous_decoded_output_executor(
        request: SNNLanguageAutonomousDecodedOutputExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoded_output_executor(
            autonomous_decoded_output_preflight=(
                request.autonomous_decoded_output_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            decode_evidence=request.decode_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-event-review")
    def terminus_snn_language_autonomous_decoded_output_event_review(
        request: SNNLanguageAutonomousDecodedOutputEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_decoded_output_event_review(
            autonomous_decoded_output_executor=(
                request.autonomous_decoded_output_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-design")
    def terminus_snn_language_autonomous_bounded_text_emission_design(
        request: SNNLanguageAutonomousBoundedTextEmissionDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_text_emission_design(
            autonomous_decoded_output_event_review=(
                request.autonomous_decoded_output_event_review
            ),
            text_surface_binding=request.text_surface_binding,
            emission_policy=request.emission_policy,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-preflight")
    def terminus_snn_language_autonomous_bounded_text_emission_preflight(
        request: SNNLanguageAutonomousBoundedTextEmissionPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_text_emission_preflight(
            autonomous_bounded_text_emission_design=(
                request.autonomous_bounded_text_emission_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-executor")
    def terminus_snn_language_autonomous_bounded_text_emission_executor(
        request: SNNLanguageAutonomousBoundedTextEmissionExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_text_emission_executor(
            autonomous_bounded_text_emission_preflight=(
                request.autonomous_bounded_text_emission_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            emission_evidence=request.emission_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-event-review")
    def terminus_snn_language_autonomous_bounded_text_emission_event_review(
        request: SNNLanguageAutonomousBoundedTextEmissionEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_text_emission_event_review(
            autonomous_bounded_text_emission_executor=(
                request.autonomous_bounded_text_emission_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-sequence-review")
    def terminus_snn_language_autonomous_text_surface_sequence_review(
        request: SNNLanguageAutonomousTextSurfaceSequenceReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_sequence_review(
            autonomous_bounded_text_emission_event_review=(
                request.autonomous_bounded_text_emission_event_review
            ),
            sequence_policy=request.sequence_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-design")
    def terminus_snn_language_autonomous_text_surface_commit_design(
        request: SNNLanguageAutonomousTextSurfaceCommitDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_commit_design(
            autonomous_text_surface_sequence_review=(
                request.autonomous_text_surface_sequence_review
            ),
            commit_policy=request.commit_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-preflight")
    def terminus_snn_language_autonomous_text_surface_commit_preflight(
        request: SNNLanguageAutonomousTextSurfaceCommitPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_commit_preflight(
            autonomous_text_surface_commit_design=(
                request.autonomous_text_surface_commit_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-executor")
    def terminus_snn_language_autonomous_text_surface_commit_executor(
        request: SNNLanguageAutonomousTextSurfaceCommitExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_commit_executor(
            autonomous_text_surface_commit_preflight=(
                request.autonomous_text_surface_commit_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            commit_evidence=request.commit_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-event-review")
    def terminus_snn_language_autonomous_text_surface_commit_event_review(
        request: SNNLanguageAutonomousTextSurfaceCommitEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_commit_event_review(
            autonomous_text_surface_commit_executor=(
                request.autonomous_text_surface_commit_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-design")
    def terminus_snn_language_autonomous_text_surface_materialization_design(
        request: SNNLanguageAutonomousTextSurfaceMaterializationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_materialization_design(
            autonomous_text_surface_commit_event_review=(
                request.autonomous_text_surface_commit_event_review
            ),
            materialization_policy=request.materialization_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-preflight")
    def terminus_snn_language_autonomous_text_surface_materialization_preflight(
        request: SNNLanguageAutonomousTextSurfaceMaterializationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_materialization_preflight(
            autonomous_text_surface_materialization_design=(
                request.autonomous_text_surface_materialization_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-executor")
    def terminus_snn_language_autonomous_text_surface_materialization_executor(
        request: SNNLanguageAutonomousTextSurfaceMaterializationExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_materialization_executor(
            autonomous_text_surface_materialization_preflight=(
                request.autonomous_text_surface_materialization_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            materialization_evidence=request.materialization_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-event-review")
    def terminus_snn_language_autonomous_text_surface_materialization_event_review(
        request: SNNLanguageAutonomousTextSurfaceMaterializationEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_text_surface_materialization_event_review(
            autonomous_text_surface_materialization_executor=(
                request.autonomous_text_surface_materialization_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-review")
    def terminus_snn_language_autonomous_bounded_language_surface_review(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_review(
            autonomous_text_surface_materialization_event_review=(
                request.autonomous_text_surface_materialization_event_review
            ),
            language_surface_policy=request.language_surface_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-design")
    def terminus_snn_language_autonomous_bounded_language_surface_commit_design(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceCommitDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_commit_design(
            autonomous_bounded_language_surface_review=(
                request.autonomous_bounded_language_surface_review
            ),
            commit_policy=request.commit_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-preflight")
    def terminus_snn_language_autonomous_bounded_language_surface_commit_preflight(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceCommitPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_commit_preflight(
            autonomous_bounded_language_surface_commit_design=(
                request.autonomous_bounded_language_surface_commit_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-executor")
    def terminus_snn_language_autonomous_bounded_language_surface_commit_executor(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceCommitExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_commit_executor(
            autonomous_bounded_language_surface_commit_preflight=(
                request.autonomous_bounded_language_surface_commit_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            commit_evidence=request.commit_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-event-review")
    def terminus_snn_language_autonomous_bounded_language_surface_commit_event_review(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceCommitEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_commit_event_review(
            autonomous_bounded_language_surface_commit_executor=(
                request.autonomous_bounded_language_surface_commit_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-review")
    def terminus_snn_language_autonomous_bounded_language_surface_use_review(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceUseReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_use_review(
            autonomous_bounded_language_surface_commit_event_review=(
                request.autonomous_bounded_language_surface_commit_event_review
            ),
            use_policy=request.use_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-preflight")
    def terminus_snn_language_autonomous_bounded_language_surface_use_preflight(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceUsePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_use_preflight(
            autonomous_bounded_language_surface_use_review=(
                request.autonomous_bounded_language_surface_use_review
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-executor")
    def terminus_snn_language_autonomous_bounded_language_surface_use_executor(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceUseExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_use_executor(
            autonomous_bounded_language_surface_use_preflight=(
                request.autonomous_bounded_language_surface_use_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            use_evidence=request.use_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-event-review")
    def terminus_snn_language_autonomous_bounded_language_surface_use_event_review(
        request: SNNLanguageAutonomousBoundedLanguageSurfaceUseEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_bounded_language_surface_use_event_review(
            autonomous_bounded_language_surface_use_executor=(
                request.autonomous_bounded_language_surface_use_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-design")
    def terminus_snn_language_autonomous_snn_language_generation_design(
        request: SNNLanguageAutonomousSNNLanguageGenerationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_generation_design(
            autonomous_bounded_language_surface_use_event_review=(
                request.autonomous_bounded_language_surface_use_event_review
            ),
            generation_policy=request.generation_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-preflight")
    def terminus_snn_language_autonomous_snn_language_generation_preflight(
        request: SNNLanguageAutonomousSNNLanguageGenerationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_generation_preflight(
            autonomous_snn_language_generation_design=(
                request.autonomous_snn_language_generation_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-executor")
    def terminus_snn_language_autonomous_snn_language_generation_executor(
        request: SNNLanguageAutonomousSNNLanguageGenerationExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_generation_executor(
            autonomous_snn_language_generation_preflight=(
                request.autonomous_snn_language_generation_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            generation_evidence=request.generation_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-event-review")
    def terminus_snn_language_autonomous_snn_language_generation_event_review(
        request: SNNLanguageAutonomousSNNLanguageGenerationEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_generation_event_review(
            autonomous_snn_language_generation_executor=(
                request.autonomous_snn_language_generation_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-design")
    def terminus_snn_language_autonomous_snn_language_decoding_design(
        request: SNNLanguageAutonomousSNNLanguageDecodingDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_decoding_design(
            autonomous_snn_language_generation_event_review=(
                request.autonomous_snn_language_generation_event_review
            ),
            decoding_policy=request.decoding_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-preflight")
    def terminus_snn_language_autonomous_snn_language_decoding_preflight(
        request: SNNLanguageAutonomousSNNLanguageDecodingPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_decoding_preflight(
            autonomous_snn_language_decoding_design=(
                request.autonomous_snn_language_decoding_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            decoder_capabilities=request.decoder_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-executor")
    def terminus_snn_language_autonomous_snn_language_decoding_executor(
        request: SNNLanguageAutonomousSNNLanguageDecodingExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_decoding_executor(
            autonomous_snn_language_decoding_preflight=(
                request.autonomous_snn_language_decoding_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            decoding_evidence=request.decoding_evidence,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-event-review")
    def terminus_snn_language_autonomous_snn_language_decoding_event_review(
        request: SNNLanguageAutonomousSNNLanguageDecodingEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_autonomous_snn_language_decoding_event_review(
            autonomous_snn_language_decoding_executor=(
                request.autonomous_snn_language_decoding_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-surface-design")
    def terminus_snn_language_surface_design(
        request: SNNLanguageSurfaceDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_surface_design(
            autonomous_snn_language_decoding_event_review=(
                request.autonomous_snn_language_decoding_event_review
            ),
            surface_policy=request.surface_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-surface-preflight")
    def terminus_snn_language_surface_preflight(
        request: SNNLanguageSurfacePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_surface_preflight(
            snn_language_readout_surface_design=(
                request.snn_language_surface_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-surface-executor")
    def terminus_snn_language_surface_executor(
        request: SNNLanguageSurfaceExecutorRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_surface_executor(
            snn_language_readout_surface_preflight=(
                request.snn_language_surface_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            execution_policy=request.execution_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-surface-event-review")
    def terminus_snn_language_surface_event_review(
        request: SNNLanguageSurfaceEventReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_surface_event_review(
            snn_language_readout_surface_executor=(
                request.snn_language_surface_executor
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=request.review_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-memory-design")
    def terminus_snn_language_memory_design(
        request: SNNLanguageMemoryDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_memory_design(
            snn_language_readout_surface_event_review=(
                request.snn_language_surface_event_review
            ),
            memory_policy=_internal_snn_language_payload(request.memory_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-memory-preflight")
    def terminus_snn_language_memory_preflight(
        request: SNNLanguageMemoryPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_memory_preflight(
            snn_language_readout_memory_design=(
                _internal_snn_language_payload(request.snn_language_memory_design)
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=_internal_snn_language_payload(request.device_evidence),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-memory-executor")
    def terminus_snn_language_memory_executor(
        request: SNNLanguageMemoryExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_memory_executor(
            snn_language_readout_memory_preflight=(
                _internal_snn_language_payload(request.snn_language_memory_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            execution_policy=_internal_snn_language_payload(request.execution_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-memory-event-review")
    def terminus_snn_language_memory_event_review(
        request: SNNLanguageMemoryEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_memory_event_review(
            snn_language_readout_memory_executor=(
                _internal_snn_language_payload(request.snn_language_memory_executor)
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=_internal_snn_language_payload(request.review_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-design")
    def terminus_snn_language_consolidation_design(
        request: SNNLanguageConsolidationDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_consolidation_design(
            snn_language_readout_memory_event_review=(
                _internal_snn_language_payload(request.snn_language_memory_event_review)
            ),
            consolidation_policy=_internal_snn_language_payload(request.consolidation_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-preflight")
    def terminus_snn_language_consolidation_preflight(
        request: SNNLanguageConsolidationPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_consolidation_preflight(
            snn_language_readout_consolidation_design=(
                _internal_snn_language_payload(request.snn_language_consolidation_design)
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=_internal_snn_language_payload(request.device_evidence),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-executor")
    def terminus_snn_language_consolidation_executor(
        request: SNNLanguageConsolidationExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_consolidation_executor(
            snn_language_readout_consolidation_preflight=(
                _internal_snn_language_payload(request.snn_language_consolidation_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            execution_policy=_internal_snn_language_payload(request.execution_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-event-review")
    def terminus_snn_language_consolidation_event_review(
        request: SNNLanguageConsolidationEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_consolidation_event_review(
            snn_language_readout_consolidation_executor=(
                _internal_snn_language_payload(request.snn_language_consolidation_executor)
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=_internal_snn_language_payload(request.review_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-design")
    def terminus_snn_language_structural_plasticity_design(
        request: SNNLanguageStructuralPlasticityDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_structural_plasticity_design(
            snn_language_readout_consolidation_event_review=(
                _internal_snn_language_payload(request.snn_language_consolidation_event_review)
            ),
            structural_policy=_internal_snn_language_payload(request.structural_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-preflight")
    def terminus_snn_language_structural_plasticity_preflight(
        request: SNNLanguageStructuralPlasticityPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_structural_plasticity_preflight(
            snn_language_readout_structural_plasticity_design=(
                _internal_snn_language_payload(request.snn_language_structural_plasticity_design)
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=_internal_snn_language_payload(request.device_evidence),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-executor")
    def terminus_snn_language_structural_plasticity_executor(
        request: SNNLanguageStructuralPlasticityExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_structural_plasticity_executor(
            snn_language_readout_structural_plasticity_preflight=(
                _internal_snn_language_payload(request.snn_language_structural_plasticity_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            execution_policy=_internal_snn_language_payload(request.execution_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-event-review")
    def terminus_snn_language_structural_plasticity_event_review(
        request: SNNLanguageStructuralPlasticityEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_readout_structural_plasticity_event_review(
            snn_language_readout_structural_plasticity_executor=(
                _internal_snn_language_payload(request.snn_language_structural_plasticity_executor)
            ),
            expected_state_revision=request.expected_state_revision,
            review_policy=_internal_snn_language_payload(request.review_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-design")
    def terminus_snn_language_capacity_mutation_design(
        request: SNNLanguageCapacityMutationDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_capacity_mutation_design(
            snn_language_readout_structural_plasticity_event_review=(
                _internal_snn_language_payload(request.snn_language_structural_plasticity_event_review)
            ),
            capacity_policy=_internal_snn_language_payload(request.capacity_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-preflight")
    def terminus_snn_language_capacity_mutation_preflight(
        request: SNNLanguageCapacityMutationPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_capacity_mutation_preflight(
            autonomous_snn_language_thought_capacity_mutation_design=(
                _internal_snn_language_payload(request.snn_language_capacity_mutation_design)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_transaction=_internal_snn_language_payload(request.checkpoint_transaction),
            device_evidence=_internal_snn_language_payload(request.device_evidence),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-executor")
    def terminus_snn_language_capacity_mutation_executor(
        request: SNNLanguageCapacityMutationExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_capacity_mutation_executor(
            autonomous_snn_language_thought_capacity_mutation_preflight=(
                _internal_snn_language_payload(request.snn_language_capacity_mutation_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
            requested_device=request.requested_device,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-event-review")
    def terminus_snn_language_capacity_mutation_event_review(
        request: SNNLanguageCapacityMutationEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_capacity_mutation_event_review(
            autonomous_snn_language_thought_capacity_mutation_executor=(
                _internal_snn_language_payload(request.snn_language_capacity_mutation_executor)
            ),
            expected_state_revision=request.expected_state_revision,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-design")
    def terminus_snn_language_newborn_neuron_integration_design(
        request: SNNLanguageNewbornNeuronIntegrationDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_integration_design(
            autonomous_snn_language_thought_capacity_mutation_event_review=(
                _internal_snn_language_payload(request.snn_language_capacity_mutation_event_review)
            ),
            integration_policy=_internal_snn_language_payload(request.integration_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-preflight")
    def terminus_snn_language_newborn_neuron_integration_preflight(
        request: SNNLanguageNewbornNeuronIntegrationPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_integration_preflight(
            autonomous_snn_language_thought_newborn_neuron_integration_design=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_integration_design)
            ),
            expected_state_revision=request.expected_state_revision,
            live_spike_evidence=_internal_snn_language_payload(request.live_spike_evidence),
            checkpoint_transaction=_internal_snn_language_payload(request.checkpoint_transaction),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-executor")
    def terminus_snn_language_newborn_neuron_integration_executor(
        request: SNNLanguageNewbornNeuronIntegrationExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_integration_executor(
            autonomous_snn_language_thought_newborn_neuron_integration_preflight=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_integration_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-event-review")
    def terminus_snn_language_newborn_neuron_integration_event_review(
        request: SNNLanguageNewbornNeuronIntegrationEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_integration_event_review(
            autonomous_snn_language_thought_newborn_neuron_integration_executor=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_integration_executor)
            ),
            expected_state_revision=request.expected_state_revision,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-design")
    def terminus_snn_language_newborn_neuron_critical_period_learning_design(
        request: SNNLanguageNewbornNeuronCriticalPeriodLearningDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design(
            autonomous_snn_language_thought_newborn_neuron_integration_event_review=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_integration_event_review)
            ),
            learning_policy=_internal_snn_language_payload(request.learning_policy),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-preflight")
    def terminus_snn_language_newborn_neuron_critical_period_learning_preflight(
        request: SNNLanguageNewbornNeuronCriticalPeriodLearningPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight(
            autonomous_snn_language_thought_newborn_neuron_critical_period_learning_design=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_critical_period_learning_design)
            ),
            expected_state_revision=request.expected_state_revision,
            critical_period_activity_evidence=(
                _internal_snn_language_payload(request.critical_period_activity_evidence)
            ),
            checkpoint_transaction=_internal_snn_language_payload(request.checkpoint_transaction),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-executor")
    def terminus_snn_language_newborn_neuron_critical_period_learning_executor(
        request: SNNLanguageNewbornNeuronCriticalPeriodLearningExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_executor(
            autonomous_snn_language_thought_newborn_neuron_critical_period_learning_preflight=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_critical_period_learning_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-event-review")
    def terminus_snn_language_newborn_neuron_critical_period_learning_event_review(
        request: SNNLanguageNewbornNeuronCriticalPeriodLearningEventReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review(
            autonomous_snn_language_thought_newborn_neuron_critical_period_learning_executor=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_critical_period_learning_executor)
            ),
            expected_state_revision=request.expected_state_revision,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-continuation-design")
    def terminus_snn_language_newborn_neuron_critical_period_learning_continuation_design(
        request: SNNLanguageNewbornNeuronCriticalPeriodLearningContinuationDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_critical_period_learning_continuation_design(
            autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_critical_period_learning_event_review)
            )
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-maturation-outcome-review")
    def terminus_snn_language_newborn_neuron_maturation_outcome_review(
        request: SNNLanguageNewbornNeuronMaturationOutcomeReviewRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review(
            autonomous_snn_language_thought_newborn_neuron_critical_period_learning_event_review=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_critical_period_learning_event_review)
            )
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-design")
    def terminus_snn_language_newborn_synapse_pruning_design(
        request: SNNLanguageNewbornSynapsePruningDesignRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_design(
            autonomous_snn_language_thought_newborn_neuron_maturation_outcome_review=(
                _internal_snn_language_payload(request.snn_language_newborn_neuron_maturation_outcome_review)
            )
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-preflight")
    def terminus_snn_language_newborn_synapse_pruning_preflight(
        request: SNNLanguageNewbornSynapsePruningPreflightRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_preflight(
            autonomous_snn_language_thought_newborn_synapse_pruning_design=(
                _internal_snn_language_payload(request.snn_language_newborn_synapse_pruning_design)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_transaction=_internal_snn_language_payload(request.checkpoint_transaction),
            executor_capabilities=_internal_snn_language_payload(request.executor_capabilities),
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-executor")
    def terminus_snn_language_newborn_synapse_pruning_executor(
        request: SNNLanguageNewbornSynapsePruningExecutorRequest,
    ) -> dict[str, Any]:
        return _public_snn_language_payload(runtime.snn_language_autonomous_snn_language_thought_newborn_synapse_pruning_executor(
            autonomous_snn_language_thought_newborn_synapse_pruning_preflight=(
                _internal_snn_language_payload(request.snn_language_newborn_synapse_pruning_preflight)
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
        ))

    @app.post("/terminus/snn-language-sequence/readout-ledger/rehearsal-evaluation")
    def terminus_snn_language_readout_rehearsal_evaluation(
        request: SNNLanguageReadoutRehearsalEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rehearsal_evaluation(
            replay_priority_report=request.replay_priority_report,
            candidate_limit=request.candidate_limit,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rehearsal-experiment")
    def terminus_snn_language_readout_rehearsal_experiment(
        request: SNNLanguageReadoutRehearsalExperimentRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rehearsal_experiment(
            rehearsal_evaluation=request.rehearsal_evaluation,
            replay_cycles=request.replay_cycles,
            stability_floor=request.stability_floor,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/replay-design")
    def terminus_snn_language_readout_replay_design(
        request: SNNLanguageReadoutReplayDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_replay_design(
            rehearsal_experiment=request.rehearsal_experiment,
            replay_policy=request.replay_policy,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/replay-dry-run")
    def terminus_snn_language_readout_replay_dry_run(
        request: SNNLanguageReadoutReplayDryRunRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_replay_dry_run(
            replay_design=request.replay_design,
            operator_approval=request.operator_approval,
            operator_id=request.operator_id,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/plasticity-preflight")
    def terminus_snn_language_readout_plasticity_preflight(
        request: SNNLanguageReadoutPlasticityPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_plasticity_preflight(
            readout_replay_dry_run=request.readout_replay_dry_run,
            plasticity_policy=request.plasticity_policy,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/plasticity-replay-bridge")
    def terminus_snn_language_readout_plasticity_replay_bridge(
        request: SNNLanguageReadoutPlasticityReplayBridgeRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_plasticity_replay_bridge(
            readout_plasticity_preflight=request.readout_plasticity_preflight,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.get("/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit")
    def terminus_snn_language_readout_synapse_provenance_audit(
        limit: int = Query(64, ge=0, le=512),
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_synapse_provenance_audit(limit=limit)

    @app.post("/terminus/snn-language-sequence/readout-ledger/record")
    def terminus_snn_language_readout_ledger_record(
        request: SNNLanguageReadoutLedgerRecordRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_evidence_ledger_record(
            readout_draft=request.readout_draft,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/record-rollout-replay-evaluation")
    def terminus_snn_language_readout_rollout_ledger_record(
        request: SNNLanguageReadoutRolloutLedgerRecordRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_evidence_ledger_record(
            readout_rollout_replay_evaluation=request.readout_rollout_replay_evaluation,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.get("/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-promotion-policy")
    def terminus_snn_language_readout_rollout_rehearsal_promotion_policy(
        limit: int = Query(8, ge=0, le=32),
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_rehearsal_promotion_policy(
            candidate_limit=limit,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-evaluation")
    def terminus_snn_language_readout_rollout_rehearsal_evaluation(
        request: SNNLanguageReadoutRolloutRehearsalEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_rehearsal_evaluation(
            rollout_rehearsal_promotion_policy=request.rollout_rehearsal_promotion_policy,
            candidate_limit=request.candidate_limit,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-experiment")
    def terminus_snn_language_readout_rollout_rehearsal_experiment(
        request: SNNLanguageReadoutRolloutRehearsalExperimentRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_rehearsal_experiment(
            rollout_rehearsal_evaluation=request.rollout_rehearsal_evaluation,
            replay_cycles=request.replay_cycles,
            stability_floor=request.stability_floor,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-design")
    def terminus_snn_language_readout_rollout_consolidation_design(
        request: SNNLanguageReadoutRolloutConsolidationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_consolidation_design(
            rollout_rehearsal_experiment=request.rollout_rehearsal_experiment,
            consolidation_policy=request.consolidation_policy,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-shadow-delta")
    def terminus_snn_language_readout_rollout_consolidation_shadow_delta(
        request: SNNLanguageReadoutRolloutConsolidationShadowDeltaRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_consolidation_shadow_delta(
            rollout_consolidation_design=request.rollout_consolidation_design,
            device_evidence=request.device_evidence,
        )

    @app.post(
        "/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-shadow-application-preflight"
    )
    def terminus_snn_language_readout_rollout_consolidation_shadow_application_preflight(
        request: SNNLanguageReadoutRolloutConsolidationShadowApplicationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_consolidation_shadow_application_preflight(
            rollout_consolidation_design=request.rollout_consolidation_design,
            rollout_consolidation_shadow_delta=request.rollout_consolidation_shadow_delta,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-developmental-plasticity-review")
    def terminus_snn_language_readout_rollout_developmental_plasticity_review(
        request: SNNLanguageReadoutRolloutDevelopmentalPlasticityReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_developmental_plasticity_review(
            rollout_consolidation_design=request.rollout_consolidation_design,
            rollout_consolidation_shadow_application_preflight=(
                request.rollout_consolidation_shadow_application_preflight
            ),
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-proposal-adapter")
    def terminus_snn_language_readout_rollout_regeneration_proposal_adapter(
        request: SNNLanguageReadoutRolloutRegenerationProposalAdapterRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_regeneration_proposal_adapter(
            rollout_developmental_plasticity_review=request.rollout_developmental_plasticity_review,
        )

    @app.post(
        "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-replay-artifact-review"
    )
    def terminus_snn_language_readout_rollout_regeneration_replay_artifact_review(
        request: SNNLanguageReadoutRolloutRegenerationReplayArtifactReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_regeneration_replay_artifact_review(
            rollout_regeneration_proposal_adapter=request.rollout_regeneration_proposal_adapter,
            snn_transition_memory_replay_artifact=request.snn_transition_memory_replay_artifact,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request")
    def terminus_snn_language_readout_rollout_regeneration_permit_request(
        request: SNNLanguageReadoutRolloutRegenerationPermitRequestRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_regeneration_permit_request(
            rollout_regeneration_replay_artifact_review=(
                request.rollout_regeneration_replay_artifact_review
            ),
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application-preflight")
    def terminus_snn_language_readout_rollout_regeneration_application_preflight(
        request: SNNLanguageReadoutRolloutRegenerationApplicationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_regeneration_application_preflight(
            rollout_regeneration_permit_request=request.rollout_regeneration_permit_request,
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application")
    def terminus_snn_language_readout_rollout_regeneration_application(
        request: SNNLanguageReadoutRolloutRegenerationApplicationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_readout_rollout_regeneration_application(
            rollout_regeneration_application_preflight=(
                request.rollout_regeneration_application_preflight
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
            max_outgoing_row_mass=request.max_outgoing_row_mass,
        )

    @app.post("/terminus/snn-language-sequence/transition-memory-prediction-evaluation")
    def terminus_snn_language_transition_memory_prediction_evaluation(
        request: SNNLanguageTransitionMemoryPredictionEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_transition_memory_prediction_evaluation(
            training_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.training_readout_slot_batches
            ],
            evaluation_readout_slot_batches=[
                [_model_to_dict(slot) for slot in batch]
                for batch in request.evaluation_readout_slot_batches
            ],
            transition_memory_state=request.transition_memory_state,
            device_evidence=request.device_evidence,
            learning_rate=request.learning_rate,
            epochs=request.epochs,
            top_k=request.top_k,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-pressure")
    def terminus_snn_language_plasticity_pressure(
        request: SNNLanguagePlasticityPressureRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_pressure(
            mismatch_report=request.mismatch_report,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-trial")
    def terminus_snn_language_plasticity_trial(
        request: SNNLanguagePlasticityTrialRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_trial(
            pressure_report=request.pressure_report,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-replay-evaluation")
    def terminus_snn_language_plasticity_replay_evaluation(
        request: SNNLanguagePlasticityReplayEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_replay_evaluation(
            trial_report=request.trial_report,
            replay_window=request.replay_window,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-replay-experiment")
    def terminus_snn_language_plasticity_replay_experiment(
        request: SNNLanguagePlasticityReplayExperimentRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_replay_experiment(
            replay_evaluation=request.replay_evaluation,
            replay_sequences=request.replay_sequences,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-application-design")
    def terminus_snn_language_plasticity_application_design(
        request: SNNLanguagePlasticityApplicationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_application_design(
            replay_experiment=request.replay_experiment,
            application_policy=request.application_policy,
            device_evidence=request.device_evidence,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-shadow-application")
    def terminus_snn_language_plasticity_shadow_application(
        request: SNNLanguagePlasticityShadowApplicationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_shadow_application(
            application_design=request.application_design,
            shadow_delta=request.shadow_delta,
            device_evidence=request.device_evidence,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-shadow-delta")
    def terminus_snn_language_plasticity_shadow_delta(
        request: SNNLanguagePlasticityShadowDeltaRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_shadow_delta(
            application_design=request.application_design,
            replay_sequences=request.replay_sequences,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-live-application-readiness")
    def terminus_snn_language_plasticity_live_application_readiness(
        request: SNNLanguagePlasticityLiveApplicationReadinessRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_live_application_readiness(
            shadow_application=request.shadow_application,
            rollback_readiness=request.rollback_readiness,
            operator_approval=request.operator_approval,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-live-application-preflight")
    def terminus_snn_language_plasticity_live_application_preflight(
        request: SNNLanguagePlasticityLiveApplicationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_live_application_preflight(
            live_application_readiness=request.live_application_readiness,
            application_target=request.application_target,
            checkpoint_transaction=request.checkpoint_transaction,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-live-application")
    def terminus_snn_language_plasticity_live_application(
        request: SNNLanguagePlasticityLiveApplicationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_plasticity_live_application(
            live_application_readiness=request.live_application_readiness,
            shadow_delta=request.shadow_delta,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
        )

    @app.get("/terminus/snn-language-sequence/plasticity-runtime-state")
    def terminus_snn_language_plasticity_runtime_state() -> dict[str, Any]:
        return runtime.snn_language_plasticity_runtime_state()

    @app.post("/terminus/snn-language-sequence/capacity-expansion-design")
    def terminus_snn_language_capacity_expansion_design(
        request: SNNLanguageCapacityExpansionDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_capacity_expansion_design(
            capacity_pressure=request.capacity_pressure,
            device_evidence=request.device_evidence,
            rollback_policy=request.rollback_policy,
            max_neuron_growth_factor=request.max_neuron_growth_factor,
        )

    @app.post("/terminus/snn-language-sequence/capacity-expansion-preflight")
    def terminus_snn_language_capacity_expansion_preflight(
        request: SNNLanguageCapacityExpansionPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_capacity_expansion_preflight(
            capacity_expansion_design=request.capacity_expansion_design,
            expected_state_revision=request.expected_state_revision,
            checkpoint_transaction=request.checkpoint_transaction,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/capacity-resize-compatibility-audit")
    def terminus_snn_language_capacity_resize_compatibility_audit(
        request: SNNLanguageCapacityResizeCompatibilityAuditRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_capacity_resize_compatibility_audit(
            capacity_expansion_preflight=request.capacity_expansion_preflight,
            language_capacity_state=request.language_capacity_state,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-resize-plan")
    def terminus_snn_language_dense_readout_resize_plan(
        request: SNNLanguageDenseReadoutResizePlanRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_resize_plan(
            capacity_pressure=request.capacity_pressure,
            fixed_boundaries=request.fixed_boundaries,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-resize-preflight")
    def terminus_snn_language_dense_readout_resize_preflight(
        request: SNNLanguageDenseReadoutResizePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_resize_preflight(
            dense_readout_resize_plan=request.dense_readout_resize_plan,
            expected_state_revision=request.expected_state_revision,
            checkpoint_transaction=request.checkpoint_transaction,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-resize-transaction-proposal")
    def terminus_snn_language_dense_readout_resize_transaction_proposal(
        request: SNNLanguageDenseReadoutResizeTransactionProposalRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_resize_transaction_proposal(
            dense_readout_resize_preflight=request.dense_readout_resize_preflight,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-resize-executor-readiness-audit")
    def terminus_snn_language_dense_readout_resize_executor_readiness_audit(
        request: SNNLanguageDenseReadoutResizeExecutorReadinessAuditRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_resize_executor_readiness_audit(
            dense_readout_resize_transaction_proposal=(
                request.dense_readout_resize_transaction_proposal
            ),
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-layout-migration")
    def terminus_snn_language_dense_readout_layout_migration(
        request: SNNLanguageDenseReadoutLayoutMigrationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_layout_migration(
            dense_readout_resize_transaction_proposal=(
                request.dense_readout_resize_transaction_proposal
            ),
            dense_readout_resize_executor_readiness_audit=(
                request.dense_readout_resize_executor_readiness_audit
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-tensor-materialization-readiness")
    def terminus_snn_language_dense_readout_tensor_materialization_readiness(
        request: SNNLanguageDenseReadoutTensorMaterializationReadinessRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_tensor_materialization_readiness(
            dense_readout_layout_migration=request.dense_readout_layout_migration,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-tensor-materialization")
    def terminus_snn_language_dense_readout_tensor_materialization(
        request: SNNLanguageDenseReadoutTensorMaterializationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_tensor_materialization(
            dense_readout_tensor_materialization_readiness=(
                request.dense_readout_tensor_materialization_readiness
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
            requested_device=request.requested_device,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-training-readiness")
    def terminus_snn_language_dense_readout_training_readiness(
        request: SNNLanguageDenseReadoutTrainingReadinessRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_training_readiness(
            dense_readout_tensor_integrity=request.dense_readout_tensor_integrity,
            heldout_evaluation=request.heldout_evaluation,
            device_evidence=request.device_evidence,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-training-loop-design")
    def terminus_snn_language_dense_readout_training_loop_design(
        request: SNNLanguageDenseReadoutTrainingLoopDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_training_loop_design(
            dense_readout_training_readiness=(
                request.dense_readout_training_readiness
            ),
            training_plan=request.training_plan,
            device_evidence=request.device_evidence,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-training-loop-preflight")
    def terminus_snn_language_dense_readout_training_loop_preflight(
        request: SNNLanguageDenseReadoutTrainingLoopPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_training_loop_preflight(
            dense_readout_training_loop_design=(
                request.dense_readout_training_loop_design
            ),
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
            executor_capabilities=request.executor_capabilities,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-training")
    def terminus_snn_language_dense_readout_training(
        request: SNNLanguageDenseReadoutTrainingRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_training(
            dense_readout_training_loop_preflight=(
                request.dense_readout_training_loop_preflight
            ),
            training_transitions=request.training_transitions,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-post-training-evaluation")
    def terminus_snn_language_dense_readout_post_training_evaluation(
        request: SNNLanguageDenseReadoutPostTrainingEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_post_training_evaluation(
            dense_readout_training=request.dense_readout_training,
            dense_readout_tensor_integrity=request.dense_readout_tensor_integrity,
            heldout_evaluation=request.heldout_evaluation,
            runtime_truth_delta=request.runtime_truth_delta,
            rollback_policy=request.rollback_policy,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-decoder-probe-design")
    def terminus_snn_language_dense_readout_decoder_probe_design(
        request: SNNLanguageDenseReadoutDecoderProbeDesignRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_decoder_probe_design(
            dense_readout_post_training_evaluation=(
                request.dense_readout_post_training_evaluation
            ),
            readout_slots=request.readout_slots,
            device_evidence=request.device_evidence,
            decoder_design=request.decoder_design,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-decoder-probe-preflight")
    def terminus_snn_language_dense_readout_decoder_probe_preflight(
        request: SNNLanguageDenseReadoutDecoderProbePreflightRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_decoder_probe_preflight(
            dense_readout_decoder_probe_design=(
                request.dense_readout_decoder_probe_design
            ),
            expected_state_revision=request.expected_state_revision,
            device_evidence=request.device_evidence,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-decoder-probe")
    def terminus_snn_language_dense_readout_decoder_probe(
        request: SNNLanguageDenseReadoutDecoderProbeExecutionRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_decoder_probe_execution(
            dense_readout_decoder_probe_preflight=(
                request.dense_readout_decoder_probe_preflight
            ),
            max_candidate_labels=request.max_candidate_labels,
        )

    @app.post("/terminus/snn-language-sequence/dense-readout-label-candidate-review")
    def terminus_snn_language_dense_readout_label_candidate_review(
        request: SNNLanguageDenseReadoutLabelCandidateReviewRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_label_candidate_review(
            dense_readout_decoder_probe_execution=(
                request.dense_readout_decoder_probe_execution
            ),
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            review_note=request.review_note,
        )

    @app.post("/terminus/snn-language-sequence/readout-ledger/record-dense-label-candidate-review")
    def terminus_snn_language_dense_readout_label_candidate_evidence_record(
        request: SNNLanguageDenseReadoutLabelCandidateEvidenceRecordRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_dense_readout_label_candidate_evidence_record(
            dense_readout_label_candidate_review=(
                request.dense_readout_label_candidate_review
            ),
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-homeostatic-maintenance")
    def terminus_snn_language_transition_memory_homeostatic_maintenance(
        request: SNNLanguageTransitionMemoryHomeostaticMaintenanceRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_transition_memory_homeostatic_maintenance(
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
            decay_factor=request.decay_factor,
            prune_below=request.prune_below,
            max_outgoing_row_mass=request.max_outgoing_row_mass,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-sleep-policy")
    def terminus_snn_language_transition_memory_sleep_policy(
        request: SNNLanguageTransitionMemorySleepPolicyRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_transition_memory_sleep_policy(
            transition_memory_state=request.transition_memory_state,
            subcortex_sleep_pressure=request.subcortex_sleep_pressure,
            replay_evidence=request.replay_evidence,
            rollout_regeneration_evidence=request.rollout_regeneration_evidence,
            readout_ledger_evidence=request.readout_ledger_evidence,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-sleep-policy/review-ticket")
    def terminus_snn_sleep_plasticity_review_ticket(
        request: SNNSleepPlasticityReviewTicketRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_sleep_plasticity_review_ticket(
                sleep_policy=request.sleep_policy,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/terminus/snn-language-sequence/plasticity-sleep-policy/review-tickets")
    def terminus_snn_sleep_plasticity_review_ticket_queue(
        limit: int = Query(20, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_review_ticket_queue(limit=limit)

    @app.get("/terminus/snn-language-sequence/plasticity-sleep-policy/autonomy-proposal")
    def terminus_snn_sleep_plasticity_autonomy_proposal(
        limit: int = Query(20, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_autonomy_proposal(limit=limit)

    @app.get("/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment")
    def terminus_snn_sleep_plasticity_scheduler_experiment(
        limit: int = Query(20, ge=1, le=64),
        cycles: int = Query(4, ge=1, le=16),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_scheduler_experiment(
            limit=limit,
            cycles=cycles,
        )

    @app.get("/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design")
    def terminus_snn_sleep_plasticity_scheduler_design(
        limit: int = Query(20, ge=1, le=64),
        cycles: int = Query(4, ge=3, le=16),
        min_stable_cycles: int = Query(3, ge=3, le=16),
        max_review_interval_seconds: float = Query(300.0, ge=60.0, le=3600.0),
    ) -> dict[str, Any]:
        if min_stable_cycles > cycles:
            raise HTTPException(
                status_code=422,
                detail="min_stable_cycles must not exceed cycles.",
            )
        return runtime.snn_sleep_plasticity_scheduler_design(
            limit=limit,
            cycles=cycles,
            min_stable_cycles=min_stable_cycles,
            max_review_interval_seconds=max_review_interval_seconds,
        )

    @app.post(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design/review-ticket"
    )
    def terminus_snn_sleep_plasticity_scheduler_design_review_ticket(
        request: SNNSleepPlasticitySchedulerDesignReviewTicketRequest,
    ) -> dict[str, Any]:
        if request.min_stable_cycles > request.cycles:
            raise HTTPException(
                status_code=422,
                detail="min_stable_cycles must not exceed cycles.",
            )
        try:
            return runtime.snn_sleep_plasticity_scheduler_design_review_ticket(
                limit=request.limit,
                cycles=request.cycles,
                min_stable_cycles=request.min_stable_cycles,
                max_review_interval_seconds=request.max_review_interval_seconds,
                expected_state_revision=request.expected_state_revision,
                scheduler_design_hash=request.scheduler_design_hash,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design/review-tickets"
    )
    def terminus_snn_sleep_plasticity_scheduler_design_review_ticket_queue(
        limit: int = Query(20, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_scheduler_design_review_ticket_queue(
            limit=limit
        )

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "scheduler-installation-autonomy-proposal"
    )
    def terminus_snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
        limit: int = Query(20, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_scheduler_installation_autonomy_proposal(
            limit=limit
        )

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "scheduler-installation-preflight"
    )
    def terminus_snn_sleep_plasticity_scheduler_installation_preflight(
        limit: int = Query(20, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_scheduler_installation_preflight(
            limit=limit
        )

    @app.post(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler/install"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_installation(
        request: SNNSleepPlasticityReviewSchedulerInstallationRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_sleep_plasticity_review_scheduler_installation(
                limit=request.limit,
                expected_state_revision=request.expected_state_revision,
                scheduler_installation_preflight_hash=(
                    request.scheduler_installation_preflight_hash
                ),
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_runtime() -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_review_scheduler_runtime()

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "review-scheduler/cycle-inspection"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_cycle_inspection() -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_review_scheduler_cycle_inspection()

    @app.post(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "review-scheduler/cycle-acknowledgment"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_cycle_acknowledgment(
        request: SNNSleepPlasticityReviewSchedulerCycleAcknowledgmentRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment(
                expected_state_revision=request.expected_state_revision,
                scheduler_installation_id=request.scheduler_installation_id,
                scheduler_installation_evidence_hash=(
                    request.scheduler_installation_evidence_hash
                ),
                review_ticket_id=request.review_ticket_id,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "review-scheduler/cycle-acknowledgment-preflight"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
        scheduler_installation_id: str = Query(..., min_length=1, max_length=240),
        scheduler_installation_evidence_hash: str = Query(..., min_length=64, max_length=64),
        review_ticket_id: str = Query(..., min_length=1, max_length=240),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_review_scheduler_cycle_acknowledgment_preflight(
            scheduler_installation_id=scheduler_installation_id,
            scheduler_installation_evidence_hash=scheduler_installation_evidence_hash,
            review_ticket_id=review_ticket_id,
        )

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "review-scheduler/cycle-autonomy-proposal"
    )
    def terminus_snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal() -> dict[str, Any]:
        return runtime.snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal()

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
        "review-scheduler/due-cycle-bounded-replay-selection-proposal"
    )
    def terminus_snn_due_cycle_bounded_replay_selection_proposal(
        limit: int = Query(8, ge=0, le=32),
        max_candidates: int = Query(1, ge=1, le=8),
    ) -> dict[str, Any]:
        return runtime.snn_due_cycle_bounded_replay_selection_proposal(
            limit=limit,
            max_candidates=max_candidates,
        )

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler/"
        "due-cycle-replay-artifact-recording-review-proposal"
    )
    def terminus_snn_due_cycle_replay_artifact_recording_review_proposal(
        limit: int = Query(8, ge=0, le=32),
        max_candidates: int = Query(1, ge=1, le=8),
        min_priority_score: float = Query(66.0, ge=0.0, le=100.0),
    ) -> dict[str, Any]:
        return runtime.snn_due_cycle_replay_artifact_recording_review_proposal(
            limit=limit,
            max_candidates=max_candidates,
            policy={"min_priority_score": min_priority_score},
        )

    @app.post(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler/"
        "due-cycle-replay-artifact-recording-review-ticket"
    )
    def terminus_snn_due_cycle_replay_artifact_recording_review_ticket(
        request: SNNDueCycleReplayArtifactRecordingReviewTicketRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_due_cycle_replay_artifact_recording_review_ticket(
                limit=request.limit,
                max_candidates=request.max_candidates,
                policy={"min_priority_score": request.min_priority_score},
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler/"
        "sleep-phase-separation-proposal"
    )
    def terminus_snn_sleep_phase_separation_proposal(
        limit: int = Query(8, ge=0, le=32),
        max_candidates: int = Query(1, ge=1, le=8),
    ) -> dict[str, Any]:
        return runtime.snn_sleep_phase_separation_proposal(
            limit=limit,
            max_candidates=max_candidates,
        )

    @app.get(
        "/terminus/snn-language-sequence/plasticity-sleep-policy/review-scheduler/"
        "rem-like-homeostatic-stabilization-preflight"
    )
    def terminus_snn_rem_like_homeostatic_stabilization_preflight(
        limit: int = Query(8, ge=0, le=32),
        max_candidates: int = Query(1, ge=1, le=8),
        decay_factor: float = Query(0.98, gt=0.0, le=1.0),
        prune_below: float = Query(0.005, ge=0.0, le=0.25),
        max_outgoing_row_mass: float = Query(1.0, gt=0.0, le=4.0),
    ) -> dict[str, Any]:
        return runtime.snn_rem_like_homeostatic_stabilization_preflight(
            limit=limit,
            max_candidates=max_candidates,
            maintenance_policy={
                "decay_factor": decay_factor,
                "prune_below": prune_below,
                "max_outgoing_row_mass": max_outgoing_row_mass,
            },
        )

    @app.post("/terminus/snn-language-sequence/plasticity-regeneration-proposal")
    def terminus_snn_language_transition_memory_regeneration_proposal(
        request: SNNLanguageTransitionMemoryRegenerationProposalRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_transition_memory_regeneration_proposal(
            mismatch_report=request.mismatch_report,
            transition_memory_state=request.transition_memory_state,
            replay_evidence=request.replay_evidence,
            locality_radius=request.locality_radius,
            initial_weight=request.initial_weight,
            max_new_synapses=request.max_new_synapses,
        )

    @app.post("/terminus/snn-language-sequence/plasticity-regeneration-permit")
    def terminus_snn_language_transition_memory_regeneration_permit(
        request: SNNLanguageTransitionMemoryRegenerationPermitRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_language_transition_memory_regeneration_permit(
                replay_artifact_id=request.replay_artifact_id,
                regeneration_design=request.regeneration_design,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/terminus/snn-language-sequence/replay-evaluation-context")
    def terminus_snn_replay_evaluation_context(
        request: SNNReplayEvaluationContextRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_replay_evaluation_context(
                prediction_report=request.prediction_report,
                observed_readout_slots=request.observed_readout_slots,
                device_evidence=request.device_evidence,
                runtime_truth_delta=request.runtime_truth_delta,
                rollback_policy=request.rollback_policy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/terminus/snn-language-sequence/replay-consolidation-priority-queue")
    def terminus_snn_replay_consolidation_priority_queue(
        limit: int = Query(8, ge=0, le=32),
    ) -> dict[str, Any]:
        return runtime.snn_replay_consolidation_priority_queue(limit=limit)

    @app.get("/terminus/snn-language-sequence/replay-artifact-recording-policy")
    def terminus_snn_replay_artifact_recording_policy(
        limit: int = Query(8, ge=0, le=32),
        min_priority_score: float = Query(66.0, ge=0.0, le=100.0),
    ) -> dict[str, Any]:
        return runtime.snn_replay_artifact_recording_policy_proposal(
            limit=limit,
            policy={"min_priority_score": min_priority_score},
        )

    @app.post("/terminus/snn-language-sequence/replay-artifact-recording-review-ticket")
    def terminus_snn_replay_artifact_recording_review_ticket(
        request: SNNReplayArtifactRecordingReviewTicketRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_replay_artifact_recording_review_ticket(
                limit=request.limit,
                policy={"min_priority_score": request.min_priority_score},
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/terminus/snn-language-sequence/transition-memory-replay-artifact/proposal")
    def terminus_snn_transition_memory_replay_artifact_proposal(
        request: SNNTransitionMemoryReplayArtifactProposalRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_transition_memory_replay_artifact_proposal(
                replay_evaluation_context_id=request.replay_evaluation_context_id,
                limit=request.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/terminus/snn-language-sequence/transition-memory-replay-artifact/evaluated-record")
    def terminus_snn_transition_memory_evaluated_replay_artifact(
        request: SNNEvaluatedTransitionMemoryReplayArtifactRequest,
    ) -> dict[str, Any]:
        try:
            return runtime.snn_transition_memory_evaluated_replay_artifact(
                replay_evaluation_context_id=request.replay_evaluation_context_id,
                review_ticket_id=request.review_ticket_id,
                limit=request.limit,
                operator_id=request.operator_id,
                confirmation=request.confirmation,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/terminus/snn-language-sequence/plasticity-regeneration")
    def terminus_snn_language_transition_memory_regeneration(
        request: SNNLanguageTransitionMemoryRegenerationRequest,
    ) -> dict[str, Any]:
        return runtime.snn_language_transition_memory_regeneration(
            regeneration_proposal=request.regeneration_proposal,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
            max_outgoing_row_mass=request.max_outgoing_row_mass,
        )

    @app.get("/terminus/subcortical-self-repair")
    def terminus_subcortical_self_repair() -> dict[str, Any]:
        return runtime.subcortical_self_repair_surface()

    @app.get("/terminus/subcortical-self-repair/evaluation")
    def terminus_subcortical_self_repair_evaluation() -> dict[str, Any]:
        return runtime.subcortical_self_repair_evaluation_surface()

    @app.get("/terminus/subcortical-structural-plasticity")
    def terminus_subcortical_structural_plasticity() -> dict[str, Any]:
        return runtime.subcortical_structural_plasticity_surface()

    @app.get("/terminus/subcortical-structural-plasticity/binding-growth-trial")
    def terminus_binding_growth_trial_design(
        max_candidates: int = Query(8, ge=1, le=16),
        max_total_edge_delta: int = Query(16, ge=1, le=64),
    ) -> dict[str, Any]:
        return runtime.binding_growth_trial_design(
            max_candidates=max_candidates,
            max_total_edge_delta=max_total_edge_delta,
        )

    @app.post("/terminus/subcortical-structural-plasticity/evaluate")
    def terminus_subcortical_structural_plasticity_evaluation(
        request: StructuralPlasticityIsolatedEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.subcortical_structural_plasticity_isolated_evaluation(
            pre_snapshot=request.pre_snapshot,
            post_snapshot=request.post_snapshot,
            rollback_policy=request.rollback_policy,
            candidate_evidence=request.candidate_evidence,
            cost_evidence=request.cost_evidence,
            runtime_truth_summary=request.runtime_truth_summary,
            no_mutation_evidence=request.no_mutation_evidence,
        )

    @app.post("/terminus/subcortical-structural-plasticity/mutation-design")
    def terminus_subcortical_structural_mutation_design(
        request: StructuralMutationDesignRequest,
    ) -> dict[str, Any]:
        return runtime.subcortical_structural_mutation_design(
            isolated_evaluation=request.isolated_evaluation,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            mutation_reason=request.mutation_reason,
            max_total_edge_delta=request.max_total_edge_delta,
        )

    @app.post("/terminus/subcortical-structural-plasticity/mutation-preflight")
    def terminus_subcortical_structural_mutation_preflight(
        request: StructuralMutationPreflightRequest,
    ) -> dict[str, Any]:
        return runtime.subcortical_structural_mutation_preflight(
            structural_mutation_design=request.structural_mutation_design,
            expected_state_revision=request.expected_state_revision,
            checkpoint_path=request.checkpoint_path,
        )

    @app.post("/terminus/subcortical-structural-plasticity/mutation-application")
    def terminus_subcortical_structural_mutation_application(
        request: StructuralMutationApplicationRequest,
    ) -> dict[str, Any]:
        return runtime.subcortical_structural_mutation_application(
            structural_mutation_preflight=request.structural_mutation_preflight,
            expected_state_revision=request.expected_state_revision,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
            checkpoint_path=request.checkpoint_path,
        )

    @app.get("/terminus/replay-plan", response_model=ReplayPlanResponse)
    def terminus_replay_plan(limit: int = Query(20, ge=1, le=50)) -> ReplayPlanResponse:
        return ReplayPlanResponse(**runtime.replay_plan_status(limit=limit))

    def _replay_sample_response(request: ReplaySampleRequest) -> ReplaySampleResponse:
        try:
            return ReplaySampleResponse(
                **runtime.replay_sample(
                    mode=request.mode,
                    candidate_id=request.candidate_id,
                    target_type=request.target_type,
                    target_id=request.target_id,
                    operator_id=request.operator_id,
                    operator_note=request.operator_note,
                    confirmation=request.confirmation,
                    limit=request.limit,
                    count=request.count,
                    alpha=request.alpha,
                    seed=request.seed,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/replay-sample", response_model=ReplaySampleResponse)
    def terminus_replay_sample(request: ReplaySampleRequest) -> ReplaySampleResponse:
        return _replay_sample_response(request)

    @app.get("/terminus/replay-sample/history", response_model=ReplaySampleHistoryResponse)
    def terminus_replay_sample_history(limit: int = Query(20, ge=1, le=256)) -> ReplaySampleHistoryResponse:
        return ReplaySampleHistoryResponse(**runtime.replay_sample_history(limit=limit))

    @app.get("/terminus/runtime-traces/export", response_model=RuntimeTraceExportResponse)
    def terminus_runtime_trace_export(
        limit: int = Query(20, ge=1, le=MAX_RUNTIME_TRACE_EXPORT_LIMIT),
        endpoint: str | None = Query(None, min_length=1, max_length=32),
        trace_type: str | None = Query(None, alias="type", min_length=1, max_length=32),
    ) -> RuntimeTraceExportResponse:
        return RuntimeTraceExportResponse(
            **runtime.export_runtime_trace_examples(limit=limit, endpoint=endpoint or trace_type)
        )

    @app.get("/terminus/replay-dataset/preview", response_model=ReplayDatasetPreviewResponse)
    def terminus_replay_dataset_preview(
        limit: int = Query(20, ge=1, le=MAX_RUNTIME_TRACE_EXPORT_LIMIT),
        endpoint: str | None = Query(None, min_length=1, max_length=32),
        trace_type: str | None = Query(None, alias="type", min_length=1, max_length=32),
    ) -> ReplayDatasetPreviewResponse:
        return ReplayDatasetPreviewResponse(
            **runtime.replay_dataset_preview(limit=limit, endpoint=endpoint or trace_type)
        )

    @app.post("/terminus/replay-dataset/bundle", response_model=ReplayDatasetBundleResponse)
    def terminus_replay_dataset_bundle(request: ReplayDatasetBundleRequest) -> ReplayDatasetBundleResponse:
        try:
            return ReplayDatasetBundleResponse(
                **runtime.replay_dataset_bundle(
                    operator_id=request.operator_id,
                    operator_note=request.operator_note,
                    confirmation=request.confirmation,
                    limit=request.limit,
                    endpoint=request.endpoint,
                    holdout_fraction=request.holdout_fraction,
                    eval_fraction=request.eval_fraction,
                    seed=request.seed,
                    retention_days=request.retention_days,
                    decontamination_terms=request.decontamination_terms,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/configure", response_model=TerminusRuntimeResponse)
    def terminus_configure(request: TerminusConfigureRequest) -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(
                **runtime.configure_terminus(
                    source_bank=[_model_to_dict(item) for item in request.source_bank],
                    tick_tokens=request.tick_tokens,
                    source_concept_observation_tick_interval=(
                        request.source_concept_observation_tick_interval
                    ),
                    sleep_interval_seconds=request.sleep_interval_seconds,
                    execution_quantum_tokens=request.execution_quantum_tokens,
                    execution_yield_seconds=request.execution_yield_seconds,
                    repeat_sources=request.repeat_sources,
                    autonomy=None if request.autonomy is None else _model_to_dict(request.autonomy),
                    sensory=None if request.sensory is None else _model_to_dict(request.sensory),
                    ingestion=None if request.ingestion is None else _model_to_dict(request.ingestion),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/start", response_model=TerminusRuntimeResponse)
    def terminus_start() -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(**runtime.start_terminus())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/stop", response_model=TerminusRuntimeResponse)
    def terminus_stop() -> TerminusRuntimeResponse:
        return TerminusRuntimeResponse(**runtime.stop_terminus())

    @app.post("/terminus/tick", response_model=TerminusRuntimeResponse)
    def terminus_tick(request: TerminusTickRequest) -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(**runtime.terminus_tick(steps=request.steps))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/quick-start")
    def terminus_quick_start(preset: str = Query("curriculum")) -> dict[str, Any]:
        try:
            return runtime.quick_start_terminus(preset=preset)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/terminus/presets")
    def terminus_presets() -> list[dict[str, Any]]:
        return RuntimeControl.quick_start_presets()

    @app.get("/terminus/actions", response_model=ActionHistoryResponse)
    def terminus_actions(limit: int = Query(20, ge=1, le=100)) -> ActionHistoryResponse:
        return ActionHistoryResponse(**runtime.action_history(limit=limit))

    @app.get("/terminus/validation/reports")
    def terminus_validation_reports(limit: int = Query(40, ge=1, le=200)) -> dict[str, Any]:
        root = _report_root(manager)
        reports: list[dict[str, Any]] = []
        if root.exists():
            candidates = sorted(root.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
            for path in candidates:
                summary = _summarize_report(path, root)
                if summary.get("artifact_kind") in REPORT_SUMMARY_KINDS or str(summary.get("status", "")):
                    reports.append(summary)
                if len(reports) >= limit:
                    break
        phase_status = {
            "phase14": next(
                (item for item in reports if item.get("artifact_kind") == "terminus_multi_hour_live_validation"),
                None,
            ),
            "phase15": next(
                (item for item in reports if item.get("artifact_kind") == "terminus_bounded_self_improvement_readiness"),
                None,
            ),
        }
        return {
            "root": str(root),
            "reports": reports,
            "phase_status": phase_status,
            "latest": reports[0] if reports else None,
        }

    @app.get("/terminus/validation/report")
    def terminus_validation_report(path: str = Query(..., min_length=1, max_length=512)) -> dict[str, Any]:
        report_path = _safe_report_path(manager, path)
        if report_path.suffix.lower() == ".md":
            return {
                "path": report_path.relative_to(_report_root(manager)).as_posix(),
                "media_type": "text/markdown",
                "content": report_path.read_text(encoding="utf-8"),
            }
        return {
            "path": report_path.relative_to(_report_root(manager)).as_posix(),
            "media_type": "application/json",
            "content": json.loads(report_path.read_text(encoding="utf-8")),
        }

    @app.post("/terminus/action", response_model=DigitalActionResponse)
    def terminus_action(request: DigitalActionRequest) -> DigitalActionResponse:
        try:
            return DigitalActionResponse(**runtime.execute_digital_action(_model_to_dict(request)))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/runtime-feedback", response_model=RuntimeFeedbackResponse)
    def terminus_runtime_feedback(request: RuntimeFeedbackRequest) -> RuntimeFeedbackResponse:
        try:
            return RuntimeFeedbackResponse(**runtime.record_runtime_feedback(_model_to_dict(request)))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/terminus/sensory/recent")
    def terminus_sensory_recent(limit: int = Query(6, ge=1, le=12)) -> dict[str, Any]:
        """Recent real sensory previews (images/audio) for the UI."""
        return runtime.sensory_previews(limit=limit)

    @app.get("/traces", response_model=TraceHistoryResponse)
    def traces(limit: int = Query(20, ge=1, le=200)) -> TraceHistoryResponse:
        return TraceHistoryResponse(traces=runtime.recent_traces(limit=limit))

    @app.get("/architecture")
    def architecture() -> dict[str, Any]:
        return runtime.architecture_summary()

    @app.post("/grounding-probe/run")
    def grounding_probe_run() -> dict[str, Any]:
        return runtime.run_grounding_probe()

    @app.get("/stream/status")
    async def stream_status(interval: float = Query(1.0, ge=0.25, le=10.0)) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[str]:
            last_payload = ""
            heartbeat_counter = 0
            while True:
                payload = json.dumps(runtime.telemetry_snapshot())
                if payload != last_payload:
                    yield f"event: status\ndata: {payload}\n\n"
                    last_payload = payload
                else:
                    # Send a heartbeat comment every ~15 s to keep proxies alive.
                    heartbeat_counter += 1
                    if heartbeat_counter * interval >= 15.0:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                await asyncio.sleep(interval)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/datasets")
    async def datasets():
        """List the data sources used by the current Terminus runtime."""
        result = current_runtime_datasets()
        return {
            "datasets": result,
            "huggingface": {
                "token_configured": bool(
                    os.environ.get("HF_TOKEN")
                    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                    or os.environ.get("HUGGINGFACE_API_KEY")
                )
            },
        }

    return app
