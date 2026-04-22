"""HECSN Service layer -- REST API, Terminus brain loop, and autonomy.

Module structure:
- manager.py:          HECSNServiceManager (core orchestrator, ~2700 lines)
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
