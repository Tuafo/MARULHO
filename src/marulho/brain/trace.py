from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class BrainTrace:
    """Compact runtime telemetry for MarulhoBrain."""

    step: int
    event: str
    device: str
    token_count: int
    queued_tokens: int
    tick_tokens: int = 0
    trained_tokens: int = 0
    elapsed_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    executor: str = ""
    route_vote_mode: str = ""
    active_language_path: str = "local_transition_readout"
    cuda_available: bool = False
    generation_before: str = ""
    generation_after: str = ""
    replay_updates: int = 0
    growth_events: int = 0
    prune_events: int = 0
    checkpoint_path: str | None = None
    source: str | None = None
    note: str = ""
    created_at: str = ""

    surface: str = "marulho_brain_trace.v1"
    external_llm_used: bool = False
    thought_loop_used: bool = False
    cortex_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        created_at = self.created_at or datetime.now(timezone.utc).isoformat()
        return {
            "surface": self.surface,
            "step": int(self.step),
            "event": str(self.event),
            "created_at": created_at,
            "device": str(self.device),
            "token_count": int(self.token_count),
            "queued_tokens": int(self.queued_tokens),
            "tick_tokens": int(self.tick_tokens),
            "trained_tokens": int(self.trained_tokens),
            "elapsed_ms": round(float(self.elapsed_ms), 3),
            "throughput_tokens_per_sec": round(
                float(self.throughput_tokens_per_sec),
                3,
            ),
            "executor": str(self.executor),
            "route_vote_mode": str(self.route_vote_mode),
            "active_language_path": str(self.active_language_path),
            "cuda_available": bool(self.cuda_available),
            "generation": {
                "before": str(self.generation_before),
                "after": str(self.generation_after),
                "changed": str(self.generation_before) != str(self.generation_after),
                "owned_by_marulho": True,
                "external_dependency": False,
            },
            "replay_updates": int(self.replay_updates),
            "growth_events": int(self.growth_events),
            "prune_events": int(self.prune_events),
            "checkpoint_path": self.checkpoint_path,
            "source": self.source,
            "note": str(self.note),
            "retired_brain_surfaces": {
                "external_llm_used": bool(self.external_llm_used),
                "thought_loop_used": bool(self.thought_loop_used),
                "cortex_used": bool(self.cortex_used),
            },
        }
