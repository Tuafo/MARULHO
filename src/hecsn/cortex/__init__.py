"""Cortical core — LLM neocortex for the hybrid SNN-LLM architecture.

The cortex module wraps a frozen LLM (Gemma 4 via Ollama) as the language/reasoning
engine of Terminus.  The SNN subcortical systems (drives, memory, surprise, sleep)
control *when*, *what*, and *how* the cortex thinks — but never the reverse.
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
)
from hecsn.cortex.drives import (
    DriveSystem,
    DriveState,
    ThalamicGate,
    AntiRuminationCircuit,
)
from hecsn.cortex.thought_loop import ThoughtLoop, BrainStats

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
    "DriveSystem",
    "DriveState",
    "ThalamicGate",
    "AntiRuminationCircuit",
    "ThoughtLoop",
    "BrainStats",
]
