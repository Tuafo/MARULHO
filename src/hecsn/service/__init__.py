"""HECSN Service layer -- REST API, Terminus brain loop, and autonomy.

Module structure:
- manager.py:          HECSNServiceManager facade and live runtime orchestration
- runtime_state.py:    Shared dirty-state, revision, and brain event container
- runtime_evidence.py: Sanitized traces, replay dataset preview, and evidence exports
- runtime_feedback.py: Operator feedback normalization and application
- action_assist.py:   Query action-assist and audited action evidence injection
- action_runtime.py:   Digital action execution and audit summaries
- brain_runtime.py:   Brain source rebuild, tick, autonomy, and runtime snapshots
- delayed_consequence.py: Long-horizon source/provider consequence learning
- persistence.py:      Checkpoint, trace-history, and JSON-safe persistence helpers
- cortex_runtime.py:   Cortex ask/sleep/thought/action-intent control helpers
- reporting.py:        Architecture and grounding-probe reporting helpers
- replay_runtime.py:   Advisory replay planning and operator-gated replay sampling
- interaction_runtime.py: Query/feed/respond/acquire operator interaction flow
- living_status.py:    Living-loop and policy-actuator read-only status helpers
- runtime_config.py:   Operator runtime/source configuration normalization
- runtime_control.py:  Terminus configure/start/stop/tick runtime control
- runtime_prewarm.py:  Remote warm promotion and ingestion prewarm loops
- runtime_sources.py:  Runtime source streams, live-remote wrapping, and caches
- sensory_runtime.py:  Multimodal sensory selection, prefetch, and injection
- source_focus.py:     Text-source focus, semantic scoring, and source utility
- status_runtime.py:   Status, telemetry, and runtime warm-state summaries
- sensory_preview.py:  Recent sensory preview payload helpers
- replay_dataset_bundle.py: Operator-approved preview-only replay dataset packaging
- terminus_presets.py: Quick-start preset configurations
- terminus_hf_sources.py: Recommended Hugging Face runtime sources
- terminus_sensory.py: Real Hugging Face multimodal stream adapters
- terminus_autonomy.py: TerminusAutonomyMixin (focus planning, provider curriculum)
- api.py:             FastAPI route definitions
- schemas.py:         Request/response Pydantic models
- server.py:          CLI entry point (uvicorn launcher)
"""

from .manager import HECSNServiceManager


def create_app(*args, **kwargs):
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app", "HECSNServiceManager"]
