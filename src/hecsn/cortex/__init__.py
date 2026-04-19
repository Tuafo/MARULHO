"""Cortical core — LLM neocortex for the hybrid SNN-LLM architecture.

The cortex module wraps frozen LLM(s) (Ollama, NVIDIA NIM, or both)
as the language/reasoning engine of Terminus. The SNN subcortical
systems (drives, memory, surprise, sleep) control *when*, *what*,
and *how* the cortex thinks — but never the reverse.
"""

from hecsn.cortex.core import (
    CorticalCore,
    ContextPacket,
    MemoryItem,
    ThoughtResult,
    ThinkingMode,
    FakeCortex,
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
