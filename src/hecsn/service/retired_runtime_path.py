from __future__ import annotations

from typing import Any


def _build_retired_runtime_path_initial_state() -> dict[str, Any]:
    return {
        "_retired_runtime_path_available": False,
        "_retired_runtime_path_init_started": False,
        "_retired_runtime_path_init_finished": True,
        "_retired_runtime_path_init_timed_out": False,
        "_retired_runtime_path_init_error": (
            "retired LLM path is inactive; use Subcortex/Living Loop surfaces."
        ),
    }


RETIRED_RUNTIME_PATH_STATE_FIELDS = frozenset(_build_retired_runtime_path_initial_state())


class RetiredRuntimePathState:
    """Retired runtime path state holder.

    The former LLM/ThoughtLoop controller no longer owns ask/sleep/action
    compatibility behavior. Active service code only needs a stable retired
    snapshot. It has no ThoughtLoop slot, so loop injection cannot restart the path.
    """

    def __init__(self) -> None:
        for field_name, initial_value in _build_retired_runtime_path_initial_state().items():
            object.__setattr__(self, field_name, initial_value)

    def _retired_runtime_path_unavailable_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "retired": True,
            "reason": "retired_llm_path",
            "replacement": "subcortex_living_loop",
            "initialization": {
                "started": bool(getattr(self, "_retired_runtime_path_init_started", False)),
                "finished": bool(getattr(self, "_retired_runtime_path_init_finished", True)),
                "timed_out": bool(getattr(self, "_retired_runtime_path_init_timed_out", False)),
                "error": getattr(self, "_retired_runtime_path_init_error", None),
            },
        }
