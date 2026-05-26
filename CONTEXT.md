# HECSN / Terminus Domain Language

HECSN/Terminus is a grounded Subcortex runtime for auditable autonomous cognition. "Living brain" is an aspiration only when backed by runtime evidence; the maintained claim is behavioral liveness under explicit validation conditions.

## Core Concepts

**Subcortex** — the grounded predictive spiking substrate. Owns sparse routing, multimodal grounding, predictive error, neuromodulation, replay, and curiosity pressure. Does not reason in language.
_Avoid_: SSN, raw SNN side

**Retired Cortex Path** — the former LLM/ThoughtLoop deliberation path. It is no longer an active runtime requirement because it added external dependency and code weight without being the living substrate. Public HTTP endpoints for this path are removed; any temporary internal compatibility surface must report the path as retired, not unavailable or required.
_Avoid_: treating LLM/NIM as the mind, mandatory reasoning core, or active production path

**Subcortex Action Ledger** — the runtime-owned record of digital action execution, verification, contradiction, feedback, and consequence evidence. Action recording must not initialize the Retired Cortex Path; any retired-loop mirroring is compatibility-only and best-effort when an old loop already exists.
_Avoid_: action history as ThoughtLoop memory, booting Cortex to mirror an action

**Subcortex Grounded Observation** — the runtime-owned source or sensory evidence packet created from real text, visual, audio, or multimodal input. It carries grounded metadata, focus terms, salience, and device/encoder evidence without requiring ThoughtLoop observation or surprise injection.
_Avoid_: treating source/sensory observations as Cortex memory writes

**Grounding Diagnostics** — term-level evidence that a language-facing result is supported by observed source or sensory evidence. It records target terms, matched terms, evidence support, coverage, alignment, and recovery state; it is Subcortex language evidence, not a Cortex-only type.
_Avoid_: fluent answers without support terms, hidden recovery paths

**Active Exploration State** — the current bounded focus target selected to reduce uncertainty or resolve a control gap. It records normalized target text, reason, source, score, and sample time; it is Subcortex control evidence, not a ThoughtLoop-private field.
_Avoid_: free-form curiosity text without source/reason/score

**Subcortex Deliberation** — the active replacement direction for cognition that needs planning, replay, or hypothesis formation. It should be implemented through SNN-compatible prediction, world-model, memory, and policy mechanisms rather than an external LLM loop. Early deliberation candidates are bounded control candidates derived from Cognitive Signal pressure and concept focus.

**Deliberation Feedback** — the topic, grounding, valence, and confidence payload routed from a language-facing result back into Subcortex curiosity, drives, and context control.
_Avoid_: naming closed-loop control feedback as Cortex-owned

**Deliberation Text Merge** — deterministic language-surface helper that compresses multi-step deliberation outputs into one auditable result. It is a formatting/readout primitive, not a cognition substrate, and must be usable without instantiating ThoughtLoop.
_Avoid_: treating merge formatting as thought generation

**Brain Runtime Metrics** — observable counters and quality scores for language-facing runtime output, grounding, topic diversity, SNN alignment, and inference latency. They are telemetry, not proof of liveness, and must be usable without instantiating ThoughtLoop.
_Avoid_: treating generated text counts as Living Brain evidence by themselves

**Subcortex Language Surface** — an evidence-facing translation layer that can express runtime state in language without becoming the cognition substrate. It has two grounded slices: the interaction responder's native-decode surface, and the Cognitive Signal surface that turns prediction error, confidence, neuromodulator pressure, and concept focus into auditable operator text.
_Avoid_: treating text fluency as liveness, hidden LLM mind, or ungrounded thought generation

**SNN-Native Language Readiness Gate** — the operator-facing read-only artifact for future HECSN-owned language generation. External pure-SNN language projects may inform design, but HECSN must own the language neurons, decoder, training loop, grounding, telemetry, and promotion gates before language generation can move beyond a surface.
_Avoid_: loading external checkpoints as the brain, outsourcing language cognition, treating reference implementations as runtime dependencies

**Developmental Plasticity** — the Subcortex mechanism family for growing, pruning, and stabilizing assemblies, synapses, routing prototypes, and replay policies under evidence gates.
_Avoid_: self-replication as unchecked code mutation, permanent growth without pruning

**Local Plasticity Evidence** — the read-only report from local STDP eligibility traces, synaptic scaling, inhibitory balance, spike-health state, synaptic validation, and device placement. It can support Developmental Plasticity readiness, but it is not a command to change synapses or topology.
_Avoid_: treating plasticity telemetry as automatic self-modification

**Structural Mutation Ledger** — the runtime evidence record for bounded topology growth/pruning events such as hypercube binding hub outreach. It records added/removed sparse edges and recent mutation samples so structural plasticity can be audited and rolled into Runtime Truth rather than treated as hidden model drift.
_Avoid_: undocumented rewiring, unbounded topology mutation, growth claims without prune evidence

**Structural Plasticity Gate Artifact** — the operator-facing read-only artifact that combines ConceptStore growth pressure, binding topology mutation ledger evidence, and CUDA/device placement into structural-promotion readiness. It can mark a structural change ready for isolated evaluation, but it cannot mutate topology.
_Avoid_: calling concept observation, binding, grow/prune, or structural refresh from a readiness artifact

**Path Retirement Gate** — the rule that a runtime path should be deprecated or removed when it adds complexity without improving liveness, grounding, efficiency, or evidence quality. Legacy paths survive only behind compatibility seams while replacement evidence is gathered.

**Terminus** — the whole Subcortex-centered architecture: the predictive spiking substrate plus the service/runtime surfaces that keep it observable, gated, and auditable.

**Living Brain** — an evidence-gated target state where the runtime continuously senses, learns, thinks, replays, acts, sleeps, and reports liveness without bypassing safety boundaries.
_Avoid_: using "living brain" as an unconditional production claim

**CUDA-first Runtime** — the runtime posture that uses CUDA/GPU execution for tensor-heavy subcortical work when available while keeping ordinary unit tests deterministic on CPU.
_Avoid_: GPU-only correctness, hidden CPU fallback in benchmark claims

**Routing Index** — the subcortical retrieval path that maps queries to candidate assemblies/prototypes. CUDA evidence requires actual cache/backend device telemetry, not only configured device intent.

**ThoughtLoop** — retired LLM cognition orchestrator. `ThoughtLoop()` is non-instantiable and must raise a retirement error; historical tests may keep skipped behavior coverage while reusable primitive/static helpers are migrated.

**DriveSystem** — converts predictive error, surprise, fatigue, and novelty into cognitive pressure and thalamic context.

**ThalamicGate** — assembles budgeted language/readout context packets from memory, drives, and source evidence.

**Cognitive Signal** — the typed Subcortex control packet carrying prediction error, predictive confidence, neuromodulator mirrors, recent concepts, source, and sample time. It is owned by Subcortex/semantics code, not the Retired Cortex Path.

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

**Retired Runtime Path Evidence** — Runtime Truth evidence for a former runtime path that may still appear in compatibility payloads but no longer owns liveness, operator surfaces, or active runtime requirements. `retired_runtime_path` is the canonical status vocabulary; `cortex_*` booleans are temporary compatibility aliases only.

**Runtime Evidence Report** — operator-facing status evidence that joins model CUDA scope, trainer-owned encoder device reports, memory-store placement, runtime truth, and source configuration. It is read-only and must not advance runtime state.
_Avoid_: using retired Cortex snapshots as the evidence source of record

**Subcortex Spike Health** — read-only operational stability evidence from competitive-column activity, bounded recent spike windows, local spike fraction, stale routing counters, visible silence/saturation thresholds, and windowed over-correlation risk. It is evidence for Runtime Truth, not a standalone liveness verdict.
_Avoid_: treating endpoint uptime as neural health, hiding threshold heuristics, treating one scalar correlation as full manifold health

**Subcortex Self-Repair Candidate** — an advisory repair hypothesis derived from Subcortex Spike Health, such as reviewing column revival, inhibitory balance, stale routing, or decorrelation/pruning. It is not an action; promotion requires replay, deep-sleep repair, or operator gates.
_Avoid_: automatic self-mutation from status reads, treating repair suggestions as executed growth/prune events

**Self-Repair Promotion Gate** — the explicit gate on a Subcortex Self-Repair Candidate surface. It can request more spike-window evidence, mark candidates ready for replay/deep-sleep review, or keep them monitor-only; it must keep action, fact promotion, and structural mutation disabled.
_Avoid_: implicit repair execution, hidden replay mutation, structural mutation without a gate

**Self-Repair Gate Artifact** — the operator-facing read-only artifact that exposes Subcortex Self-Repair Candidates and their Self-Repair Promotion Gate without inserting those candidates into replay execution plans.
_Avoid_: replay candidate IDs, suggested endpoints, or execution payloads inside repair-gate status

**Self-Repair Evaluation Artifact** — the operator-facing read-only plan for measuring whether a Self-Repair Gate Artifact is safe to test in isolated replay or deep sleep. It names required evidence and success criteria, but does not run repair, mutate state, or promote structure.
_Avoid_: treating evaluation readiness as repair approval, hiding rollback/device/runtime-truth evidence requirements

**Delayed Consequence** — long-horizon utility tracking that connects earlier actions to later outcomes across queries and runs.

**Source Bank** — a named, ordered collection of training data sources (corpus, HF dataset, remote search) used by the subcortex for learning.

**Sensory Stream** — multimodal (visual, audio) observation stream for grounding, separate from text corpus.

## Key Relationships

- Subcortex is the active cognition substrate; the former Cortex/ThoughtLoop path is retired from runtime liveness claims.
- Subcortex Action Ledger owns action evidence. Digital action execution may update runtime history, provider calibration, consequences, and Runtime Truth evidence, but it must not initialize the Retired Cortex Path just to mirror action records.
- Subcortex Grounded Observations own source and sensory evidence. Focus selection comes from query gaps, autonomy plans, geometric curiosity, concept state, and source metadata; it must not depend on a retired ThoughtLoop exploration target.
- Grounding Diagnostics belong to the Subcortex Language Surface boundary. Retired compatibility code may consume them, but future SNN language/readout modules must be able to produce and inspect them without instantiating ThoughtLoop.
- Active Exploration State belongs to Subcortex control. ThalamicGate stores the canonical state object and may expose compatibility properties; future curiosity, source-focus, and SNN deliberation modules must be able to normalize and inspect exploration targets without instantiating ThoughtLoop.
- Runtime Truth owns retired-path vocabulary through `retired_runtime_path`; `cortex_available`, `cortex_retired`, `cortex_enabled`, and evidence `cortex_retired` are compatibility aliases while old consumers are removed.
- Living Loop may publish `retired_runtime_path` beside a temporary `cortex` compatibility snapshot, but new policy and benchmark consumers should use the retired-path vocabulary first.
- Long-test reports should describe the former Cortex as a Retired Runtime Path, not as an active model/component. Any `cortex_*` report fields are compatibility data until JSON consumers migrate.
- Subcortex Language Surface may describe, narrate, or decode Subcortex state, but it must not own memory, policy, liveness, or Runtime Truth.
- Native-decode Subcortex Language is a bridge, not a generator: it may speak only from decoded assembly text and selected evidence, with support metrics attached.
- Cognitive Signal Subcortex Language is a status decoder: it may express runtime pressure and focus, but the numeric signal remains authoritative.
- SNN-Native Language Readiness Gate keeps NeuronSpark/Nord-style work as implementation references only; the target is HECSN-owned language neurons and decoder machinery under CUDA/device, sparsity, grounding, replay/evaluation, and operator-control gates. Runtime Truth may include a compact readiness summary, while the full artifact remains the review surface.
- Cognitive Signal is the canonical runtime signal surface. Its state primitive must be importable without `ThoughtLoop`; any `cortex_signal` name is a retired compatibility alias and must not be used for new operator-facing paths.
- Subcortex Deliberation candidates are advisory control candidates until replay, policy, or operator evidence promotes them; they must not be stored as facts, treated as generated thoughts, or queued as LLM prompts.
- Deliberation Feedback belongs to Subcortex control. `emit_deliberation_feedback` is the active API; the old `emit_cortex_feedback` path is removed.
- Deliberation Text Merge belongs to the Subcortex Language Surface boundary. Active code and tests should call the merge helper directly instead of using `ThoughtLoop` as a namespace.
- Brain Runtime Metrics can summarize language-facing runtime output and grounding quality, but Runtime Truth and Subcortex Spike Health remain the authoritative liveness evidence.
- Living Loop status is the primary operational sidecar for Subcortex Deliberation candidates. Policy Actuator may display the same candidates as non-executable context, but policy status must not execute them or promote them beyond advisory evidence.
- Every Subcortex Deliberation candidate must carry a promotion gate. The gate may mark it ready for replay review or blocked by missing grounding, but it must keep action execution and fact promotion false until a separate replay/policy/operator path explicitly promotes it.
- Developmental Plasticity is the clean path for self-growth and pruning: runtime changes must be traceable, bounded, reversible, and evaluated before promotion.
- Local Plasticity Evidence is the first synapse-level support signal for Developmental Plasticity: local STDP traces, synaptic scaling, inhibitory balance, spike-health risk, synaptic validation, and device reports can make a case ready for isolated evaluation, but never directly mutate synapses or topology.
- Structural Mutation Ledger is required when topology changes at runtime; growth/pruning must leave countable evidence before it can support Living Brain claims.
- Structural Plasticity Gate Artifact is the read-only promotion surface for Developmental Plasticity: it can expose concept growth pressure, hypercube/binding mutation ledger state, and CUDA/device evidence, but structural mutation remains behind isolated evaluation and operator gates. Runtime Truth may include a compact gate summary, while the full artifact remains the review surface.
- Cognitive Signal is the telemetry/control contract that lets Subcortex update runtime pressure, concept alignment, and future Subcortex Deliberation modules.
- Retired ThoughtLoop code may remain temporarily during cleanup but must not block Runtime Truth, CUDA evidence, or long-run liveness. Cognitive Signal state, Active Exploration State, Grounding Diagnostics, Brain Runtime Metrics, and deliberation text merge belong to Subcortex-owned modules.
- Retired Cortex Compatibility Controller must not build, lazily initialize, or start ThoughtLoop. It may only report retired state and reject old compatibility calls while Subcortex/Living Loop surfaces take over.
- The top-level `hecsn.cortex` package is not an active cognition API. It may expose historical primitives during cleanup, but it must not export ThoughtLoop or the external LLM Cortex factory as product entry points.
- Gap Planner and Curiosity Controller feed Source Bank selection for autonomous acquisition
- Replay Pipeline feeds adapter experiments that never touch production runtime
- Service Manager wires the Runtime Facade and deep modules. Living Loop evidence is produced by Subcortex runtime state, replay, grounding, and policy surfaces; it must not require ThoughtLoop.
- CUDA-first Runtime applies to tensor-heavy Subcortex modules such as routing, predictive columns, neuron dynamics, binding, plasticity, cross-modal grounding, text encoders, and sensory encoders. The retired Cortex path is not a CUDA-first claim or architectural requirement.
- Runtime Evidence Report is the bridge from internal CUDA-first claims to operator-visible status; it must include trainer-owned Encoder evidence as well as model-owned Subcortex evidence.
- Runtime Evidence Report and replay/export planning should read `retired_runtime_path` as the canonical retired-path snapshot. A `cortex` payload may remain only as an external compatibility alias while clients migrate.
- Subcortex Spike Health is the first operational-stability slice inside Runtime Evidence Report: it can flag silent, saturated, stale routing, or windowed over-correlation risk, while full operational-manifold health remains a future benchmark-level claim.
- Subcortex Self-Repair Candidates turn Spike Health into reviewable repair pressure for Living Loop and Policy Actuator sidecars. The Self-Repair Promotion Gate must keep action execution, fact promotion, and structural mutation false until a separate replay/deep-sleep/operator path approves the repair.
- Self-Repair Gate Artifact is exposed separately from Replay Plan so repair pressure can be reviewed without becoming a replay execution candidate or mutating runtime state. Runtime Truth may include a compact gate summary, but the full artifact remains the review surface.
- Self-Repair Evaluation Artifact is the next gate after review readiness: it specifies isolated replay/deep-sleep evaluation evidence, rollback expectations, Runtime Truth delta, and CUDA/device evidence before any repair can be promoted.
- Path Retirement Gate now applies to Cortex: LLM-backed runtime paths are being removed from active architecture so focus returns to Subcortex, world-model, memory, and policy mechanisms.

## Flagged Ambiguities

- "SSN side" is not canonical in this project. Resolved: use **Subcortex** for the domain layer and **SNN** only when referring specifically to spiking neural network mechanics.
- "Living brain" must not erase the existing safety vocabulary. Resolved: use **Living Brain** only as an evidence-gated target state; use **Runtime Truth** for actual liveness classification.
- "language generation" is ambiguous in this project. Resolved: use **Subcortex Language Surface** for grounded expression of Subcortex state, and reserve **Subcortex Deliberation** for cognition mechanisms that can run without an LLM.
