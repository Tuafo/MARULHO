"""Retired Cortex compatibility primitives.

The active cognition path is Subcortex/Living Loop. This package remains only
for historical tests and reusable data primitives while the former LLM
ThoughtLoop code is deleted from runtime paths.
"""

from hecsn.cortex.core import (
    CorticalCore,
    ContextPacket,
    MemoryItem,
    ThoughtResult,
    ThinkingMode,
    ThoughtDepth,
    MockCortex,
)
from hecsn.cortex.episodic_memory import (
    EpisodicMemory,
    Episode,
    Provenance,
    SimpleEmbedder,
    NIMEmbedder,
)
from hecsn.cortex.drives import (
    DriveSystem,
    DriveState,
    ThalamicGate,
    AntiRuminationCircuit,
)
from hecsn.cortex.working_memory import WorkingMemory, WorkingMemoryItem, WMItemType
from hecsn.cortex.narrative_self import NarrativeSelf

__all__ = [
    "CorticalCore",
    "ContextPacket",
    "MemoryItem",
    "ThoughtResult",
    "ThinkingMode",
    "ThoughtDepth",
    "MockCortex",
    "EpisodicMemory",
    "Episode",
    "Provenance",
    "SimpleEmbedder",
    "NIMEmbedder",
    "DriveSystem",
    "DriveState",
    "ThalamicGate",
    "AntiRuminationCircuit",
    "WorkingMemory",
    "WorkingMemoryItem",
    "WMItemType",
    "NarrativeSelf",
]
