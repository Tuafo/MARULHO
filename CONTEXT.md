# HECSN / Terminus Domain Language

## Core Concepts

**Subcortex** — the grounded predictive spiking substrate. Owns sparse routing, multimodal grounding, predictive error, neuromodulation, replay, and curiosity pressure. Does not reason in language.

**Cortex** — the expressive language layer (NVIDIA NIM). Owns language, reasoning, answers, working memory, narrative self, dream-style hypothesis formation, and question formation. Gated by subcortical pressure, never free-running.

**ThoughtLoop** — the continuous cognition orchestrator. Manages wake cognition, sleep phases, working memory, episodic memory, drives, narrative self, and active exploration. The largest cortex module.

**DriveSystem** — converts predictive error, surprise, fatigue, and novelty into cognitive pressure and thalamic context.

**ThalamicGate** — assembles budgeted context packets for cortex calls from memory, drives, and source evidence.

**WorkingMemory** — chain-local global workspace. Active scratchpad with strength-based decay and broadcast compression.

**EpisodicMemory** — provenance-aware hippocampal memory with embedding-based retrieval, capacity-bounded eviction, and importance scoring.

**NarrativeSelf** — cross-session autobiographical continuity. Tracks interests, questions, and surprise over time.

**Predictive Columns** — SNN columns that predict their input. Prediction error drives surprise, learning, and curiosity.

**Replay** — hippocampal-style replay of past experiences for consolidation. Strictly evidence-only in the current runtime: no training, memory mutation, fact promotion, action execution, or sleep side effects from replay artifacts.

**Encoder** — transforms raw input (text, audio, visual) into sparse spike patterns for the SNN. Includes RTFEncoder, SemanticEncoder, EventCameraEncoder, CochleagramEncoder.

**Assembly** — a stable co-activated neuron group representing a learned concept. Decoded during query/response.

**Concept Store** — CRUD store for grounded concepts with match scoring, label generation, and expansion/contraction.

**Gap Planner** — identifies knowledge gaps from frontier analysis and produces query plans for source acquisition.

**Curiosity Controller** — geometric-curiosity-driven detection of concept gaps and synthesis of exploration queries.

**Evidence Responder** — hallucination-guarded response builder with source attribution and candidate scoring.

**Living Loop** — the autonomous runtime cycle: tick → train → think → replay → act → sleep → repeat. The core of the service runtime.

**Service Manager** — the facade through which the FastAPI layer reaches the runtime. Currently a God class composed of ~15 mixin classes.

**Autonomy Ladder** — levels 0–5 of measured autonomy: observe → propose → execute approved → recurring constrained → evaluated policy → bounded self-improvement.

**Replay Pipeline** — the staged evidence-to-learning pipeline: gate → approval → plan → isolated experiment → promotion gate. Each stage produces a hash-verified, schema-versioned artifact.

**Runtime Truth** — the liveness classification system: alive / degraded / dead / partial / failed, with evidence, safety flags, and recommended operator action.

**Delayed Consequence** — long-horizon utility tracking that connects earlier actions to later outcomes across queries and runs.

**Source Bank** — a named, ordered collection of training data sources (corpus, HF dataset, remote search) used by the subcortex for learning.

**Sensory Stream** — multimodal (visual, audio) observation stream for grounding, separate from text corpus.

## Key Relationships

- Subcortex drives Cortex via DriveSystem → ThalamicGate → ThoughtLoop
- ThoughtLoop reads from EpisodicMemory and WorkingMemory, writes to NarrativeSelf
- Gap Planner and Curiosity Controller feed Source Bank selection for autonomous acquisition
- Replay Pipeline feeds adapter experiments that never touch production runtime
- Service Manager orchestrates Living Loop, which runs ThoughtLoop against the SNN model
