# HECSN / Terminus Domain Language

HECSN/Terminus is a grounded Subcortex runtime for auditable autonomous cognition. "Living brain" is an aspiration only when backed by runtime evidence; the maintained claim is behavioral liveness under explicit validation conditions.

## Core Concepts

**Subcortex** — the grounded predictive spiking substrate. Owns sparse routing, multimodal grounding, predictive error, neuromodulation, replay, and curiosity pressure. Does not reason in language.
_Avoid_: SSN, raw SNN side

**Retired Cortex Path** — the former LLM/ThoughtLoop deliberation path. It is no longer an active runtime requirement because it added external dependency and code weight without being the living substrate. Public HTTP endpoints for this path are removed; any temporary internal compatibility surface must report the path as retired, not unavailable or required.
_Avoid_: treating LLM/NIM as the mind, mandatory reasoning core, or active production path

**Subcortex Deliberation** — the active replacement direction for cognition that needs planning, replay, or hypothesis formation. It should be implemented through SNN-compatible prediction, world-model, memory, and policy mechanisms rather than an external LLM loop. Early deliberation candidates are bounded control candidates derived from Cognitive Signal pressure and concept focus.

**Subcortex Language Surface** — an evidence-facing translation layer that can express runtime state in language without becoming the cognition substrate. It has two grounded slices: the interaction responder's native-decode surface, and the Cognitive Signal surface that turns prediction error, confidence, neuromodulator pressure, and concept focus into auditable operator text.
_Avoid_: treating text fluency as liveness, hidden LLM mind, or ungrounded thought generation

**Developmental Plasticity** — the Subcortex mechanism family for growing, pruning, and stabilizing assemblies, synapses, routing prototypes, and replay policies under evidence gates.
_Avoid_: self-replication as unchecked code mutation, permanent growth without pruning

**Structural Mutation Ledger** — the runtime evidence record for bounded topology growth/pruning events such as hypercube binding hub outreach. It records added/removed sparse edges and recent mutation samples so structural plasticity can be audited and rolled into Runtime Truth rather than treated as hidden model drift.
_Avoid_: undocumented rewiring, unbounded topology mutation, growth claims without prune evidence

**Path Retirement Gate** — the rule that a runtime path should be deprecated or removed when it adds complexity without improving liveness, grounding, efficiency, or evidence quality. Legacy paths survive only behind compatibility seams while replacement evidence is gathered.

**Terminus** — the whole Subcortex-centered architecture: the predictive spiking substrate plus the service/runtime surfaces that keep it observable, gated, and auditable.

**Living Brain** — an evidence-gated target state where the runtime continuously senses, learns, thinks, replays, acts, sleeps, and reports liveness without bypassing safety boundaries.
_Avoid_: using "living brain" as an unconditional production claim

**CUDA-first Runtime** — the runtime posture that uses CUDA/GPU execution for tensor-heavy subcortical work when available while keeping ordinary unit tests deterministic on CPU.
_Avoid_: GPU-only correctness, hidden CPU fallback in benchmark claims

**Routing Index** — the subcortical retrieval path that maps queries to candidate assemblies/prototypes. CUDA evidence requires actual cache/backend device telemetry, not only configured device intent.

**ThoughtLoop** — retired LLM cognition orchestrator. Historical tests may still cover it while the codebase is cleaned, but active liveness and architecture claims must not depend on it.

**DriveSystem** — converts predictive error, surprise, fatigue, and novelty into cognitive pressure and thalamic context.

**ThalamicGate** — assembles budgeted context packets for cortex calls from memory, drives, and source evidence.

**Cognitive Signal** — the typed Subcortex control packet carrying prediction error, predictive confidence, neuromodulator mirrors, recent concepts, source, and sample time.

**WorkingMemory** — chain-local global workspace. Active scratchpad with strength-based decay and broadcast compression.

**EpisodicMemory** — provenance-aware hippocampal memory with embedding-based retrieval, capacity-bounded eviction, and importance scoring.

**NarrativeSelf** — cross-session autobiographical continuity. Tracks interests, questions, and surprise over time.

**Predictive Columns** — SNN columns that predict their input. Prediction error drives surprise, learning, and curiosity.

**Neuron Dynamics** — executable spiking neuron state such as AdEx membrane voltage, adaptation, and spike timing. CUDA evidence requires live tensor device reports and checkpoint restore back onto the selected runtime device.

**Replay** — hippocampal-style replay of past experiences for consolidation. Strictly evidence-only in the current runtime: no training, memory mutation, fact promotion, action execution, or sleep side effects from replay artifacts.

**Encoder** — transforms raw input (text, audio, visual) into sparse spike patterns for the SNN. Includes RTFEncoder, SemanticEncoder, EventCameraEncoder, CochleagramEncoder. Tensor-backed encoder state follows the configured runtime device; parsing windows, string segmentation, and archival metadata remain CPU/control-plane work.

**Text Encoder** — the RTFEncoder or SemanticEncoder path for character/text input. CUDA evidence requires device reports for learned chunking codebooks, semantic bucket embeddings, adapter tensors, emitted feature vectors, and spike traces; it does not require moving Python string parsing to CUDA.

**Sensory Encoder** — a CUDA-first Encoder for real sensory streams whose episode metadata records the tensor device and encoder state used to produce visual or audio spikes.

**Assembly** — a stable co-activated neuron group representing a learned concept. Decoded during query/response.

**Memory Store** — CPU archival replay ledger for assemblies, input patterns, routing keys, raw windows, texts, tags, and PRP/consolidation metadata. CUDA-first applies when replay tensors are consumed by the model, not when evidence records are stored.

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

**Service Manager** — the composition root that wires the runtime (ADR 0003, ADR 0004). It constructs deep modules, owns lifecycle cleanup, and exposes the Runtime Facade. It owns no business logic or ADR-owned runtime state itself. It has no legacy inherited mixin stack, no manager-level catch-all attribute router, no manager-bound fallback path, no owner-forwarder helper module, no import-time dynamic delegate installer, and no module-level `*Mixin = ...` compatibility aliases; remaining manager methods are internal dependency callbacks, not the operator-facing runtime interface.

**Runtime Facade** — the operator-facing runtime interface introduced by ADR 0004. FastAPI routes and export runners call this facade instead of calling Service Manager runtime pass-through methods. It delegates to the owning deep modules and preserves the stable HTTP/runtime contract while the Service Manager stays a composition root.

- **Runtime State** — owns the shared mutation flag (`dirty_state`), revision counter, brain event history, and the externally visible `last_event` / `recent_events` payloads. Every other deep module receives it as a dependency.
- **Delayed Consequence Tracker** — long-horizon utility learning for sources and providers. Owns consequence records and the cooling/compaction/splitting/remerging state machines.
- **Autonomy Planner** — gap-based focus planning, provider curriculum prioritization, autonomy candidate selection, and query family scoring. Uses an explicit dependency adapter rather than manager-bound owner forwarding.
- **Brain Runtime** — source rebuilding, tick collection/training, grounded source observation injection, autonomy scheduling. Owns source runtimes, source utility, tick counters, stream epochs, and sensory episode/preview counters.
- **Interaction Pipeline** — query/feed/respond/acquire behaviour and the evidence capture that turns live interactions into runtime traces. Owns query gap history.
- **Feedback Applier** — verdict state machine, feedback normalization, and feedback application with provenance tracking.
- **Source Focus Scorer** — multi-factor selection scoring, semantic match scoring, utility EMA updates, and evidence weight mapping.
- **Runtime Controller** — runtime lifecycle state machine (configure/start/stop/tick), brain loop orchestration, active execution counters, thread lifecycle, and prewarm management. Uses an explicit dependency adapter rather than manager-bound owner forwarding.
- **Status Read Model** — read-only projection of all runtime state into status/telemetry/terminus snapshots, including direct sensory preview projection.
- **Action Executor** — digital action execution with path sandboxing, outcome calibration scoring, action history, and action-assist query augmentation.
- **Retired Cortex Compatibility Controller** — temporary internal adapter for the former LLM/ThoughtLoop path. It may expose retired status snapshots and cleanup hooks while old internals are removed, but it is not an operator-facing runtime surface and must not own liveness, action policy, or CUDA-first claims.
- **Runtime Persistence** — checkpoint save/restore and trace persistence. Runtime State owns the brain event history. Uses an explicit dependency object instead of owner-forwarded manager fields.
- **Runtime Config** — input validation and normalization gate for all operator configs. Stateless.
- **Runtime Sources** — stream construction, cache I/O, serialization, window reconstruction.
- **Replay Controller** — advisory replay planning, operator-gated sampling, dataset bundling with decontamination and splitting. Replay sampling intentionally uses Runtime State's dirty-without-revision path so audit-only samples stay dirty without advancing `state_revision`.

**Autonomy Ladder** — levels 0–5 of measured autonomy: observe → propose → execute approved → recurring constrained → evaluated policy → bounded self-improvement.

**Replay Pipeline** — the staged evidence-to-learning pipeline: gate → approval → plan → isolated experiment → promotion gate. Each stage produces a hash-verified, schema-versioned artifact.

**Runtime Truth** — the liveness classification system: alive / degraded / dead / partial / failed, with evidence, safety flags, and recommended operator action.

**Runtime Evidence Report** — operator-facing status evidence that joins model CUDA scope, trainer-owned encoder device reports, memory-store placement, runtime truth, and source configuration. It is read-only and must not advance runtime state.

**Subcortex Spike Health** — read-only operational stability evidence from competitive-column activity, bounded recent spike windows, local spike fraction, stale routing counters, visible silence/saturation thresholds, and windowed over-correlation risk. It is evidence for Runtime Truth, not a standalone liveness verdict.
_Avoid_: treating endpoint uptime as neural health, hiding threshold heuristics, treating one scalar correlation as full manifold health

**Subcortex Self-Repair Candidate** — an advisory repair hypothesis derived from Subcortex Spike Health, such as reviewing column revival, inhibitory balance, stale routing, or decorrelation/pruning. It is not an action; promotion requires replay, deep-sleep repair, or operator gates.
_Avoid_: automatic self-mutation from status reads, treating repair suggestions as executed growth/prune events

**Delayed Consequence** — long-horizon utility tracking that connects earlier actions to later outcomes across queries and runs.

**Source Bank** — a named, ordered collection of training data sources (corpus, HF dataset, remote search) used by the subcortex for learning.

**Sensory Stream** — multimodal (visual, audio) observation stream for grounding, separate from text corpus.

## Key Relationships

- Subcortex is the active cognition substrate; the former Cortex/ThoughtLoop path is retired from runtime liveness claims.
- Subcortex Language Surface may describe, narrate, or decode Subcortex state, but it must not own memory, policy, liveness, or Runtime Truth.
- Native-decode Subcortex Language is a bridge, not a generator: it may speak only from decoded assembly text and selected evidence, with support metrics attached.
- Cognitive Signal Subcortex Language is a status decoder: it may express runtime pressure and focus, but the numeric signal remains authoritative.
- Cognitive Signal is the canonical runtime signal surface. Any `cortex_signal` name is a retired compatibility alias and must not be used for new operator-facing paths.
- Subcortex Deliberation candidates are advisory control candidates until replay, policy, or operator evidence promotes them; they must not be stored as facts, treated as generated thoughts, or queued as LLM prompts.
- Living Loop status is the primary operational sidecar for Subcortex Deliberation candidates. Policy Actuator may display the same candidates as non-executable context, but policy status must not execute them or promote them beyond advisory evidence.
- Every Subcortex Deliberation candidate must carry a promotion gate. The gate may mark it ready for replay review or blocked by missing grounding, but it must keep action execution and fact promotion false until a separate replay/policy/operator path explicitly promotes it.
- Developmental Plasticity is the clean path for self-growth and pruning: runtime changes must be traceable, bounded, reversible, and evaluated before promotion.
- Structural Mutation Ledger is required when topology changes at runtime; growth/pruning must leave countable evidence before it can support Living Brain claims.
- Cognitive Signal is the telemetry/control contract that lets Subcortex update runtime pressure, concept alignment, and future Subcortex Deliberation modules.
- Retired ThoughtLoop code may remain temporarily during cleanup but must not block Runtime Truth, CUDA evidence, or long-run liveness.
- Gap Planner and Curiosity Controller feed Source Bank selection for autonomous acquisition
- Replay Pipeline feeds adapter experiments that never touch production runtime
- Service Manager wires the Runtime Facade and deep modules. Living Loop evidence is produced by Subcortex runtime state, replay, grounding, and policy surfaces; it must not require ThoughtLoop.
- CUDA-first Runtime applies to tensor-heavy Subcortex modules such as routing, predictive columns, neuron dynamics, binding, plasticity, cross-modal grounding, text encoders, and sensory encoders. The retired Cortex path is not a CUDA-first claim or architectural requirement.
- Runtime Evidence Report is the bridge from internal CUDA-first claims to operator-visible status; it must include trainer-owned Encoder evidence as well as model-owned Subcortex evidence.
- Subcortex Spike Health is the first operational-stability slice inside Runtime Evidence Report: it can flag silent, saturated, stale routing, or windowed over-correlation risk, while full operational-manifold health remains a future benchmark-level claim.
- Subcortex Self-Repair Candidates turn Spike Health into reviewable repair pressure for Living Loop and Policy Actuator sidecars. They must keep action execution, fact promotion, and structural mutation false until a separate promotion gate approves the repair.
- Path Retirement Gate now applies to Cortex: LLM-backed runtime paths are being removed from active architecture so focus returns to Subcortex, world-model, memory, and policy mechanisms.

## Flagged Ambiguities

- "SSN side" is not canonical in this project. Resolved: use **Subcortex** for the domain layer and **SNN** only when referring specifically to spiking neural network mechanics.
- "Living brain" must not erase the existing safety vocabulary. Resolved: use **Living Brain** only as an evidence-gated target state; use **Runtime Truth** for actual liveness classification.
- "language generation" is ambiguous in this project. Resolved: use **Subcortex Language Surface** for grounded expression of Subcortex state, and reserve **Subcortex Deliberation** for cognition mechanisms that can run without an LLM.
