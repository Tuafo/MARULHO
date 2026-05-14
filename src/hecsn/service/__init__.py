"""HECSN Service layer -- REST API, Terminus brain loop, and autonomy.

Module structure:
- manager.py: HECSNServiceManager composition root and lifecycle cleanup
- runtime_facade.py: Operator-facing runtime interface over the deep modules
- runtime_state.py: Shared dirty-state, revision, and brain event container
- status_read_model.py: Read-only projection of runtime state for status/terminus/telemetry/living-loop/policy-actuator/cortex-signal snapshots
- runtime_evidence.py: Sanitized traces, replay dataset preview, and evidence exports
- runtime_feedback.py: Operator feedback normalization and application
- action_assist.py: Query action-assist and audited action evidence injection
- action_runtime.py: Digital action execution and audit summaries
- brain_runtime.py: Brain source rebuild, tick, source utility, autonomy, and runtime snapshots
- delayed_consequence.py: Long-horizon consequence record state machines
- persistence.py: Checkpoint, trace-history, and JSON-safe persistence helpers with explicit dependencies
- cortex_controller.py: Cortex ask/sleep/thought/action-intent control helpers
- reporting.py: Grounding-probe evaluation helper (architecture summary now delegates through status_read_model)
- replay_runtime.py: Advisory replay planning and operator-gated replay sampling
- interaction_pipeline.py: Constructor-injected query/feed/respond-turn seam and runtime trace payload behavior
- interaction_runtime.py: Query/feed/respond/acquire operator interaction flow helpers
- living_status.py: Living-loop and policy-actuator read-only status helpers (living loop and policy snapshots now delegate through status_read_model)
- runtime_config.py: Operator runtime/source configuration normalization with explicit priority callbacks
- runtime_control.py: Terminus configure/start/stop/tick runtime control with explicit dependency access
- runtime_prewarm.py: Remote warm promotion and ingestion prewarm loops
- runtime_sources.py: Runtime source streams, live-remote wrapping, and caches with explicit dependencies
- sensory_runtime.py: Multimodal sensory selection, prefetch, and injection
- source_focus.py: Text-source focus and semantic scoring with explicit dependencies
- autonomy_planner.py: AutonomyPlanner (focus planning, provider curriculum) with explicit dependency access
- status_runtime.py: Status, telemetry, and runtime warm-state summaries
- sensory_preview.py: Sensory preview payload helpers; status_read_model owns the active read path
- replay_dataset_bundle.py: Operator-approved preview-only replay dataset packaging
- terminus_presets.py: Quick-start preset configurations
- terminus_hf_sources.py: Recommended Hugging Face runtime sources
- terminus_sensory.py: Real Hugging Face multimodal stream adapters
- terminus_autonomy.py: Autonomy focus, provider curriculum, and shortlist helpers
- api.py: FastAPI route definitions
- schemas.py: Request/response Pydantic models
- server.py: CLI entry point (uvicorn launcher)
"""

from .manager import HECSNServiceManager


def create_app(*args, **kwargs):
    from .api import create_app as _create_app
    return _create_app(*args, **kwargs)


__all__ = ["create_app", "HECSNServiceManager"]
