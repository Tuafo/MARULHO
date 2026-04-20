"""Cortical core -- LLM neocortex for the hybrid SNN-LLM architecture.

The cortex module wraps NVIDIA NIM cloud LLM(s) as the language/reasoning
engine of Terminus. The SNN subcortical systems (drives, memory, surprise,
sleep) control *when*, *what*, and *how* the cortex thinks -- never the
reverse.

No local Ollama or other local LLM is used. All inference goes through
NVIDIA NIM (40 req/min free tier). MockCortex is available for testing.
"""

from hecsn.cortex.core import (
    CorticalCore,
    ContextPacket,
    MemoryItem,
    ThoughtResult,
    ThinkingMode,
    MockCortex,
    FakeCortex,  # backwards compat alias for MockCortex
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
from hecsn.cortex.multi_cortex import (
    NIMCortex,
    MultiCortex,
    create_cortex_from_env,
    create_embedder_from_env,
)
from hecsn.cortex.curriculum import CurriculumGenerator, CurriculumSegment

__all__ = [
    "CorticalCore",
    "ContextPacket",
    "MemoryItem",
    "ThoughtResult",
    "ThinkingMode",
    "MockCortex",
    "FakeCortex",
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
    "NIMCortex",
    "MultiCortex",
    "create_cortex_from_env",
    "create_embedder_from_env",
    "CurriculumGenerator",
    "CurriculumSegment",
]
