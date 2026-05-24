"""Cortical backend interfaces for the Terminus cortex-subcortex architecture.

The cortex module exposes a replaceable expressive/deliberative backend.
NIMCortex is the current external LLM adapter, while MockCortex and custom
CorticalCore implementations keep the control loop backend-neutral. The SNN
subcortical systems control when, what, and how the backend is invoked.
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
from hecsn.cortex.thought_loop import ThoughtLoop, BrainStats
from hecsn.cortex.working_memory import WorkingMemory, WorkingMemoryItem, WMItemType
from hecsn.cortex.narrative_self import NarrativeSelf
from hecsn.cortex.multi_cortex import (
    NIMCortex,
    MultiCortex,
    create_cortex_from_env,
    create_embedder_from_env,
)

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
    "ThoughtLoop",
    "BrainStats",
    "WorkingMemory",
    "WorkingMemoryItem",
    "WMItemType",
    "NarrativeSelf",
    "NIMCortex",
    "MultiCortex",
    "create_cortex_from_env",
    "create_embedder_from_env",
]
