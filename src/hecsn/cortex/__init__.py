"""Cortical core -- LLM neocortex for the hybrid SNN-LLM architecture.

The cortex module wraps NVIDIA NIM cloud LLM(s) as the language/reasoning
engine of Terminus. The SNN subcortical systems (drives, memory, surprise,
sleep) control *when*, *what*, and *how* the cortex thinks -- never the
reverse.

No local Ollama or other local LLM is used. All inference goes through
NVIDIA NIM (40 req/min free tier), with a shared budget across chat and
embedding calls. MockCortex is available for testing.
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
