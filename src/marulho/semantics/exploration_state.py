"""Active exploration target state for Subcortex control surfaces."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ExplorationState:
    """Current active-exploration target chosen to reduce uncertainty."""

    target: str = ""
    reason: str = ""
    source: str = ""
    score: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_target(
        cls,
        target: str,
        *,
        reason: str = "",
        source: str = "",
        score: float = 0.0,
        updated_at: float | None = None,
    ) -> "ExplorationState":
        cleaned = normalize_exploration_target(target)
        if not cleaned:
            return cls()
        return cls(
            target=cleaned,
            reason=" ".join(str(reason).split()).strip()[:160],
            source=str(source).strip()[:40],
            score=max(0.0, min(1.0, float(score))),
            updated_at=time.time() if updated_at is None else max(0.0, float(updated_at)),
        )

    def to_dict(self) -> dict[str, float | str]:
        return {
            "target": self.target,
            "reason": self.reason,
            "source": self.source,
            "score": float(self.score),
            "updated_at": float(self.updated_at),
        }


def normalize_exploration_target(target: str) -> str:
    return " ".join(str(target).replace("/", " ").replace("|", " ").split()).strip()[:120]
