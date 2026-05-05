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

**Living Loop** — the autonomous runtime cycle: tick → train → think → replay → act → sleep → repeat. The core of the service runtime. Split into five depth-aligned modules (ADR 0001):
  - **Living Loop Helpers** (`living_loop_helpers.py`) — shared cross-layer private helper functions (Layer 0 / Foundation). No dependency on any other Living Loop module.
  - **Runtime Records** (`living_loop_records.py`) — enums and frozen dataclasses for the Living Loop runtime record types (Layer A). Maps to **Living Loop records** in domain vocabulary. Depends on Helpers only.
  - **Policy Scoring** (`living_loop_policy.py`) — PolicyScore, WorldModelLiteSummary, PolicyActuatorRecommendation, and policy actuator decision logic (Layer B). Maps to **Policy Actuator** in domain vocabulary. Depends on Helpers and Records only.
  - **Replay Planning** (`living_loop_replay.py`) — ReplayCandidate, ReplayPlan, build_replay_plan, replay safety flags, and replay candidate ranking logic (Layer C). Maps to **Replay Pipeline planning stage** in domain vocabulary. Depends on Helpers, Records, and Policy only.
  - **Operational Self-Model** (`living_loop_self_model.py`) — OperationalSelfModel, build_runtime_benchmark_telemetry, and telemetry helpers (Layer D). Maps to **Runtime Truth / Living Loop self-model** in domain vocabulary. Depends on Helpers, Records, Policy, and Replay.
  - The original `living_loop.py` is now a backward-compatible re-export shim (no implementation code).

**Service Manager** — the composition root through which the FastAPI layer reaches the runtime. Wires and exposes the following deep modules but owns no business logic itself:

- **Runtime State** — owns the shared mutation flag (`dirty_state`), revision counter, and brain event log. Every other deep module receives it as a dependency.
- **Delayed Consequence Tracker** — long-horizon utility learning for sources and providers. Owns consequence records and the cooling/compaction/splitting/remerging state machines.
- **Autonomy Planner** — gap-based focus planning, provider curriculum prioritization, autonomy candidate selection, and query family scoring.
- **Brain Runtime** — source rebuilding, tick collection/training, grounded source observation injection, autonomy scheduling. Owns brain source runtimes, source utility, tick counters, and ingestion/sensory thread state.
- **Interaction Pipeline** — query/feed/respond/acquire behaviour and the evidence capture that turns live interactions into runtime traces. Owns query gap history and the dirty-state revision counter.
- **Feedback Applier** — verdict state machine, feedback normalization, and feedback application with provenance tracking.
- **Source Focus Scorer** — multi-factor selection scoring, semantic match scoring, utility EMA updates, and evidence weight mapping.
- **Runtime Controller** — runtime lifecycle state machine (configure/start/stop/tick), brain loop orchestration, thread lifecycle, and prewarm management.
- **Status Read Model** — read-only projection of all runtime state into status/telemetry/terminus snapshots. Absorbs the former LivingStatusMixin, SensoryPreviewMixin, and shallow reporting.
- **Action Executor** — digital action execution with path sandboxing, outcome calibration scoring, action history, and action-assist query augmentation.
- **Cortex Controller** — cortex ask/sleep/thoughts, action intent handling, cortex query hints.
- **Runtime Persistence** — checkpoint save/restore, trace persistence, brain event recording.
- **Runtime Config** — input validation and normalization gate for all operator configs. Stateless.
- **Runtime Sources** — stream construction, cache I/O, serialization, window reconstruction.
- **Replay Controller** — advisory replay planning, operator-gated sampling, dataset bundling with decontamination and splitting.

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
