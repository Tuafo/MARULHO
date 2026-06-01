from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .manager import HECSNServiceManager
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
    ReplayDatasetCandidatesResponse,
    ReplayDatasetHistoryResponse,
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
    SNNLanguagePlasticityLiveApplicationPreflightRequest,
    SNNLanguagePlasticityLiveApplicationRequest,
    SNNLanguagePlasticityLiveApplicationReadinessRequest,
    SNNLanguagePlasticityPressureRequest,
    SNNLanguagePlasticityReplayEvaluationRequest,
    SNNLanguagePlasticityReplayExperimentRequest,
    SNNLanguageReadoutDraftRequest,
    SNNLanguageReadoutLedgerRecordRequest,
    SNNLanguageReadoutPlasticityPreflightRequest,
    SNNLanguageReadoutPlasticityReplayBridgeRequest,
    SNNLanguageReadoutRehearsalEvaluationRequest,
    SNNLanguageReadoutRehearsalExperimentRequest,
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
    SNNLanguageTrainerDryRunRequest,
    SNNLanguageTrainerEvaluationRequest,
    StructuralPlasticityIsolatedEvaluationRequest,
    StatusResponse,
    TerminusConfigureRequest,
    TerminusRuntimeResponse,
    TerminusTickRequest,
    TraceHistoryResponse,
)
from .terminus_hf_sources import current_runtime_datasets


DEFAULT_WEB_DIST_DIR = Path("HECSN_UI") / "dist"
REPORT_SUMMARY_KINDS = {
    "terminus_multi_hour_live_validation",
    "terminus_bounded_self_improvement_readiness",
    "terminus_live_long_run_validation",
    "terminus_replay_adapter_promotion_gate",
    "terminus_approved_action_level2",
    "terminus_replay_adaptation_experiment_1",
    "terminus_service_benchmark",
}


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    return getattr(model, "dict")()


def _report_root(manager: HECSNServiceManager) -> Path:
    env_root = getattr(manager, "_env_root", None)
    return ((Path(env_root) if env_root is not None else Path.cwd()) / "reports").resolve()


def _safe_report_path(manager: HECSNServiceManager, path_value: str) -> Path:
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
    return summary


def datetime_from_timestamp(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from HECSN_CORS_ORIGINS env var (comma-separated).

    Falls back to the canonical set of local dev origins when not set.
    """
    env_val = os.environ.get("HECSN_CORS_ORIGINS", "").strip()
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
    manager = HECSNServiceManager(
        checkpoint_path=checkpoint_path,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir,
        env_root=env_root,
    )
    app = FastAPI(
        title="HECSN Local Service",
        version="0.1.0",
        description="Strict-evidence local service for querying and steering a checkpoint-backed HECSN Terminus runtime.",
    )
    app.state.hecsn_manager = manager
    runtime = manager.runtime_facade
    app.state.hecsn_runtime = runtime
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
            "<h1>HECSN Local Service</h1>"
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
        return CheckpointActionResponse(**runtime.save_checkpoint(request.path))

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

    @app.get("/terminus/snn-language-sequence/readout-ledger")
    def terminus_snn_language_readout_ledger(limit: int = Query(20, ge=0, le=128)) -> dict[str, Any]:
        return runtime.snn_language_readout_evidence_ledger(limit=limit)

    @app.get("/terminus/snn-language-sequence/readout-ledger/replay-priority")
    def terminus_snn_language_readout_replay_priority(limit: int = Query(12, ge=0, le=128)) -> dict[str, Any]:
        return runtime.snn_language_readout_replay_priority(limit=limit)

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
        return runtime.snn_language_transition_memory_regeneration_permit(
            mismatch_report=request.mismatch_report,
            pressure_report=request.pressure_report,
            replay_window=request.replay_window,
            operator_id=request.operator_id,
            confirmation=request.confirmation,
        )

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

    @app.post("/terminus/subcortical-structural-plasticity/evaluate")
    def terminus_subcortical_structural_plasticity_evaluation(
        request: StructuralPlasticityIsolatedEvaluationRequest,
    ) -> dict[str, Any]:
        return runtime.subcortical_structural_plasticity_isolated_evaluation(
            pre_snapshot=request.pre_snapshot,
            post_snapshot=request.post_snapshot,
            rollback_policy=request.rollback_policy,
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

    @app.post("/terminus/replay-execute", response_model=ReplaySampleResponse)
    def terminus_replay_execute(request: ReplaySampleRequest) -> ReplaySampleResponse:
        return _replay_sample_response(request)

    @app.get("/terminus/replay-sample/history", response_model=ReplaySampleHistoryResponse)
    def terminus_replay_sample_history(limit: int = Query(20, ge=1, le=256)) -> ReplaySampleHistoryResponse:
        return ReplaySampleHistoryResponse(**runtime.replay_sample_history(limit=limit))

    @app.get("/terminus/replay-execute/history", response_model=ReplaySampleHistoryResponse)
    def terminus_replay_execute_history(limit: int = Query(20, ge=1, le=256)) -> ReplaySampleHistoryResponse:
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

    @app.get("/terminus/replay-dataset/candidates", response_model=ReplayDatasetCandidatesResponse)
    def terminus_replay_dataset_candidates(
        limit: int = Query(20, ge=1, le=MAX_RUNTIME_TRACE_EXPORT_LIMIT),
    ) -> ReplayDatasetCandidatesResponse:
        return ReplayDatasetCandidatesResponse(**runtime.replay_dataset_candidates(limit=limit))

    @app.get("/terminus/replay-dataset/history", response_model=ReplayDatasetHistoryResponse)
    def terminus_replay_dataset_history(limit: int = Query(20, ge=1, le=256)) -> ReplayDatasetHistoryResponse:
        return ReplayDatasetHistoryResponse(**runtime.replay_dataset_history(limit=limit))

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
                    sleep_interval_seconds=request.sleep_interval_seconds,
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
