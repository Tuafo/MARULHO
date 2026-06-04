# HECSN / Terminus Domain Language

HECSN/Terminus is a grounded Subcortex runtime for auditable autonomous cognition. "Living brain" is an aspiration only when backed by runtime evidence; the maintained claim is behavioral liveness under explicit validation conditions.

## Core Concepts

**Subcortex** — the grounded predictive spiking substrate. Owns sparse routing, multimodal grounding, predictive error, neuromodulation, replay, and curiosity pressure. Does not reason in language.
_Avoid_: SSN, raw SNN side

**Retired LLM Path** — the former external LLM/ThoughtLoop deliberation path. It is no longer an active runtime requirement because it added external dependency and code weight without being the living substrate. Public HTTP endpoints for this path are removed; any temporary internal compatibility surface must report the path as retired, not unavailable or required.
_Avoid_: treating LLM/NIM as the mind, mandatory reasoning core, or active production path

**Subcortex Action Ledger** — the runtime-owned record of digital action execution, verification, contradiction, feedback, and consequence evidence. Action recording must not initialize or mirror into the Retired LLM Path.
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

**Subcortex Spike Readout Evidence** — a HECSN-owned, read-only bridge from Cognitive Signal and CUDA/runtime placement evidence into bounded spike-language readout slots. It is deterministic population-code evidence for future SNN decoders; it must not generate text, mutate runtime state, load external checkpoints, or become the cognition substrate.
_Avoid_: treating readout slots as generated thoughts, importing reference checkpoints, or bypassing sparsity/device/grounding gates

**Subcortex Readout** — operator-facing language or status output decoded from HECSN-owned Subcortex state, grounded evidence, and readout metrics. It is reportable evidence, not a ThoughtLoop thought stream, narrative self, or proof of liveness by fluency.
_Avoid_: `thoughts` counters, narrative-self report sections, dormant LLM generation aliases

**SNN-Native Language Readiness Gate** — the operator-facing read-only artifact for future HECSN-owned language generation. External pure-SNN language projects may inform design, but HECSN must own the language neurons, decoder, training loop, grounding, telemetry, and promotion gates before language generation can move beyond a surface.
_Avoid_: loading external checkpoints as the brain, outsourcing language cognition, treating reference implementations as runtime dependencies

**Spike Language Decoder Probe** — a HECSN-owned, read-only probe that turns Subcortex Spike Readout Evidence into sparse population-code, leaky recurrent state, temporal transition, device, and support evidence. It is evidence for future SNN language machinery; it must not generate text, decode free-form language, train, mutate runtime state, or satisfy the language-generation gate by itself.
_Avoid_: treating sparse labels as generated thoughts, using the probe as a mock generator, or calling temporal-state evidence a trained language model

**Spike Language Neuron Adapter** — a HECSN-owned, read-only PLIF-style adapter that consumes Spike Language Decoder Probe sparse indices and reports bounded membrane/spike dynamics, activation sparsity, adaptive timestep use, and device placement. It is the first local language-neuron module boundary, but it is still evidence only until training, grounding evaluation, and operator gates promote a generator.
_Avoid_: treating adapter spikes as generated text, loading external language checkpoints, or using the adapter as a cognition substrate

**SNN Language Adapter Evaluation Gate** — the operator-facing read-only plan for testing the Spike Language Neuron Adapter in isolation. It names heldout grounded readout slots, grounding delta, activation-sparsity delta, Runtime Truth delta, and rollback evidence required before any later training loop can be approved; it cannot train, decode text, generate language, or promote the adapter as a cognition substrate.
_Avoid_: treating evaluation readiness as generation approval, hiding rollback requirements, or promoting adapter spikes directly into facts/actions

**Heldout Language Adapter Evaluation** — a deterministic, read-only evaluator that runs the local Spike Language Decoder Probe and Spike Language Neuron Adapter over heldout readout-slot batches, available to operators through `/terminus/snn-language-evaluation/heldout`. It reports grounded support, adapter spike counts, activation sparsity, and device evidence without training or decoding text.
_Avoid_: evaluating on generated prompts, mutating adapter weights during evaluation, or using heldout support as language-generation proof

**SNN Language Training Readiness Gate** — the read-only operator gate after Heldout Language Adapter Evaluation, available through `/terminus/snn-language-training/readiness`. It checks heldout support, activation sparsity, Runtime Truth delta, rollback evidence, and absence of external checkpoints before a HECSN-owned local SNN language trainer can be designed; it cannot train, generate text, or promote a cognition substrate.
_Avoid_: treating training-design readiness as trainer execution, loading external SNN language checkpoints, or skipping grounded heldout evidence

**SNN Language Trainer Dry Run** — an isolated HECSN-owned local-learning experiment over grounded spike readout slot sequences, available through `/terminus/snn-language-training/dry-run`. It may update ephemeral in-memory weights to measure transition support and sparsity, but it must discard weights, avoid text generation, and never mutate the live runtime model.
_Avoid_: treating dry-run weights as a checkpoint, decoding text from dry-run support, or calling dry-run success production training

**SNN Language Trainer Isolated Evaluation** — the read-only gate after SNN Language Trainer Dry Run, available through `/terminus/snn-language-training/evaluate`. It checks validation transition support, sparse weight evidence, device placement, Runtime Truth delta, and rollback evidence before a local trainer design can be reviewed; it still cannot promote runtime training, return weights, decode text, or generate language.
_Avoid_: treating trainer evaluation as a live learner, storing dry-run weights, or skipping operator review

**SNN Language Sequence Prediction Probe** — an isolated HECSN-owned probe that trains ephemeral sparse transition weights from grounded spike readout slot sequences and predicts the next sparse population-code indices, available through `/terminus/snn-language-sequence/predict`. It may report predicted spike indices and strengths, but it cannot decode text, generate language, store weights, or mutate runtime state.
_Avoid_: calling predicted sparse indices a sentence, fact, or thought; persisting probe weights; using the probe as a cognition substrate

**Persistent SNN Language Transition Memory** — checkpointed HECSN-owned sparse transition weights created by the SNN Language Plasticity Live Application Executor and read by the Sequence Prediction Probe. It can bias future sparse-code prediction and report influence evidence, but it is still not text generation, fact promotion, or an external model checkpoint.
_Avoid_: treating transition-memory influence as decoded language, accepting non-HECSN weights, or hiding whether persistent state changed prediction

**SNN Language Readout Draft** — the first HECSN-owned bounded text-producing surface over SNN language evidence. It maps sparse next-code prediction and persistent transition-memory influence onto grounded readout-slot labels to produce an operator-reviewable draft. A draft can be emitted before promotion, but bounded-readout generation readiness requires a non-worsening SNN Language Transition Memory Prediction Evaluation over grounded windows. It is not free-form language generation, fact promotion, action authority, or a cognition substrate.
_Avoid_: calling readout labels thoughts, bypassing grounding vocabulary, treating one influenced prediction as draft readiness, or treating a draft as autonomous truth

**SNN Language Readout Rollout Candidate** — a HECSN-owned, read-only multi-step sparse transition rollout over Persistent SNN Language Transition Memory. It starts from current sparse readout evidence, advances bounded sparse states, and maps each step only to grounded readout-vocabulary labels so operators can review short SNN-native language candidates without free-form decoding, external checkpoints, runtime mutation, fact promotion, or action authority.
_Avoid_: naming rollout candidates thoughts, using ungrounded vocabulary labels, treating rollout text as autonomous truth, or allowing rollout evidence to apply plasticity

**SNN Language Readout Rollout Replay Evaluation** — a HECSN-owned, read-only review gate that turns a bounded Readout Rollout Candidate into hash-addressed replay-review targets. It verifies candidate provenance, grounded labels, transition-memory review evidence, and trace hashes before any ledger recording or replay-priority step. It does not generate text, decode free-form language, train, apply plasticity, mutate runtime state, or load external NeuronSpark/Nord-AI artifacts.
_Avoid_: recording rollout replay targets directly, treating review targets as replay priority, importing reference-project code as runtime dependency, or using this gate as cognition-substrate approval

**SNN Language Readout Rollout Evidence Ledger Record** — an operator-approved durable record of a review-ready SNN Language Readout Rollout Replay Evaluation. It stores grounded rollout replay targets and provenance in the readout ledger's separate rollout-event channel so they can be audited later without entering draft replay priority, live replay, plasticity, fact promotion, action authority, or cognition-substrate status.
_Avoid_: mixing rollout evidence into draft replay priority, recording without operator confirmation, recording ungrounded or hash-invalid targets, or treating durable rollout evidence as learned language

**SNN Language Readout Rollout Rehearsal Promotion Policy** — a HECSN-owned, read-only deterministic ranking over recorded rollout evidence. It measures provenance, grounded replay targets, trace integrity, recency, transition-memory reuse, repetition, sparse occupancy, and CUDA/device placement to identify candidates for operator review of isolated rollout rehearsal. It does not execute rehearsal, enter draft replay priority, train, apply plasticity, promote facts/actions, or become the cognition substrate.
_Avoid_: ranking unrecorded rollouts, treating policy rank as replay execution, hiding CPU fallback when CUDA is requested, or bypassing operator review

**SNN Language Readout Rollout Rehearsal Evaluation** — an isolated HECSN-owned ephemeral SNN replay measurement over policy-approved rollout evidence. It reconstructs sparse step vectors on the recorded observed device, measures activation sparsity and adjacent-state temporal continuity, verifies device placement and target hashes, and discards every tensor after review. It does not persist weights, write checkpoints, mutate runtime state, execute live replay, or apply plasticity.
_Avoid_: accepting caller-selected devices, rehearsing unranked rollout records, persisting ephemeral tensors, or treating rehearsal stability as learned language

**SNN Language Readout Rollout Rehearsal Experiment** — a HECSN-owned repeated-cycle stability experiment over an approved Rollout Rehearsal Evaluation. It replays the isolated sparse trajectory for bounded cycles, measures cycle-to-cycle hash stability, drift, and continuity retention, and emits an operator-reviewable experiment hash. It does not persist weights, write checkpoints, mutate runtime state, execute live replay, or apply plasticity.
_Avoid_: calling repeated ephemeral cycles consolidation, accepting unstable cycles for promotion, persisting experiment tensors, or bypassing operator review

**SNN Language Readout Rollout Consolidation Design** — a HECSN-owned read-only proposal for bounded local synaptic adjustments after stable rollout rehearsal cycles. It derives local temporal synapse candidates, capped deltas, homeostatic decay, normalization, and rollback evidence for operator review. It does not write synapses, persist weights, mutate runtime state, execute replay, or apply plasticity.
_Avoid_: treating a proposal as consolidation, omitting rollback evidence, allowing non-local updates, or crossing directly into structural writes

**Sparse Rollout Consolidation Candidate Edge** — a local source-neuron to target-neuron proposal derived from adjacent sparse rollout trajectory states. Rollout step indices are provenance only; the candidate edge identity must come from canonical sparse neuron indices observed in the rehearsed transition, so later shadow deltas, growth review, and regeneration evidence stay tied to actual spike activity rather than placeholder sequence positions.
Its promotion evidence must preserve source/target rollout step indices and source/target active-index hashes through developmental growth review and regeneration design, so replay-bound growth remains auditable after the design is adapted for the checkpoint-backed executor.
_Avoid_: using rollout step numbers as synapse coordinates, fabricating edges without adjacent sparse states, dropping active-index hashes before regeneration review, importing reference-project topology, or treating one candidate edge as applied structural growth

**SNN Language Readout Rollout Consolidation Shadow Delta** — an isolated HECSN-owned sparse tensor materialization of a reviewed Rollout Consolidation Design. It computes bounded local synapse deltas on the requested CUDA/device surface, reports honest fallback evidence, and discards the tensor after review. It does not write synapses, persist weights, mutate runtime state, execute live replay, or apply plasticity.
_Avoid_: hiding CUDA fallback, applying shadow tensors, accepting empty designs, or treating shadow deltas as consolidated memory

**SNN Language Readout Rollout Consolidation Shadow Application Preflight** — a HECSN-owned, read-only integrity review of a Rollout Consolidation Shadow Delta before any application path exists. It verifies hash identity, bounded local coordinates, normalization, homeostatic decay, CUDA/device truth, rollback-snapshot binding, and topology invariants against server-held transition memory. Missing runtime edges are classified as growth candidates for separate structural review. It does not apply shadow tensors, grow or prune synapses, mutate runtime state, persist weights, execute live replay, or authorize plasticity.
_Avoid_: bypassing shadow hashes, accepting a rollback snapshot unrelated to the design, mixing functional consolidation with structural growth/pruning, or treating preflight review as application authority

**SNN Language Readout Rollout Developmental Plasticity Review** — a HECSN-owned, read-only structural review of rollout growth candidates produced when consolidation targets missing transition-memory edges. It compares the reviewed rollout design and shadow-application preflight with server-held sparse transition memory, verifies temporal locality and sparse topology budgets, and prepares candidate evidence for later regeneration review. It does not grow synapses, prune synapses, persist weights, issue permits, mutate runtime state, execute replay, or authorize plasticity.
_Avoid_: treating missing edges as weight updates, creating regeneration permits from rollout evidence directly, mixing pruning with growth review, or bypassing checkpoint-backed regeneration boundaries

**SNN Language Readout Rollout Regeneration Proposal Adapter** — a HECSN-owned, read-only adapter that converts a reviewed rollout developmental-plasticity artifact into a regeneration-design preview shaped for the existing transition-memory regeneration permit flow. It carries bounded growth candidates, locality radius, sparse topology budgets, and permit requirements, but it does not issue a permit, attach replay evidence, grow synapses, persist weights, mutate runtime state, execute replay, or make the executor ready.
_Avoid_: passing the adapter directly to the regeneration executor, fabricating replay evidence, treating rollout review as a server-issued permit, or skipping the checkpoint-backed regeneration boundary

**SNN Language Readout Rollout Regeneration Replay Artifact Review** — a HECSN-owned, read-only binding review between a Rollout Regeneration Proposal Adapter and a server-owned SNN transition-memory replay artifact. It recomputes adapter and replay-artifact hashes, checks internal-ledger replay evidence, and prepares the exact replay-artifact id plus regeneration design needed for a later operator-confirmed permit request. It does not issue permits, apply regeneration, grow synapses, persist weights, mutate runtime state, or execute replay.
_Avoid_: accepting unverified replay artifacts, treating a review as a permit, bypassing operator confirmation, or letting rollout growth evidence skip replay-controller provenance

**SNN Replay-Bound Regeneration Design** — the regeneration design after a server-owned SNN transition-memory replay artifact supplies hash-protected mismatch and pressure summaries. It replaces rollout-adapter structural intent with replay-derived mismatch pressure before permit issuance, so executor growth is justified by replay evidence rather than by a caller-provided or default score.
_Avoid_: issuing permits from adapter-only mismatch defaults, trusting opaque replay hashes without score summaries, or changing a design after permit hashing

**SNN Transition Memory Sleep Plasticity Policy** — a HECSN-owned advisory scheduler that reads transition-memory state, Subcortex sleep pressure, replay evidence, readout-ledger evidence, and rollout-regeneration evidence to recommend the next operator-reviewed growth, application, or homeostatic-maintenance gate. It never records replay artifacts, issues permits, writes checkpoints, grows/prunes synapses, or mutates runtime state.
_Avoid_: treating recommendations as hidden autonomy execution, skipping operator-confirmed permit/application gates, or running growth without a follow-up homeostatic-maintenance review

**SNN Sleep Plasticity Review Ticket** — an operator-confirmed, Replay Controller-owned record that makes one Sleep Plasticity Policy recommendation durable before any future scheduler may follow it. It binds the reviewed policy, recommended action, suggested endpoint, current Runtime State revision, operator identity, transition-memory summary, replay evidence, rollout-regeneration evidence, and readout-ledger evidence into a hash-verified ticket. It does not execute the recommendation, record replay artifacts, issue permits, write checkpoints, grow/prune synapses, apply plasticity, or mutate runtime state.
_Avoid_: treating the ticket as permission to execute, accepting no-op monitoring policies as review tickets, reusing stale revision tickets after mutation, or skipping the endpoint-specific gate named by the policy

**SNN Sleep Plasticity Review Ticket Queue** — the read-only Replay Controller surface that exposes current Sleep Plasticity Review Tickets, verification status, stale/tamper counts, pending action counts, and the latest reviewed next gate for autonomy or operator inspection. It is visibility over durable tickets, not a scheduler and not execution authority.
_Avoid_: using queue presence as automatic permission, following stale/tampered tickets, or bypassing the endpoint-specific gate named by the latest verified ticket

**SNN Sleep Plasticity Autonomy Proposal** — the non-executing autonomy candidate derived from the latest verified Sleep Plasticity Review Ticket. It lets autonomy or an operator see which reviewed sleep/plasticity gate should be inspected next, and appears as advisory evidence in Living Loop, Policy Actuator, Terminus, Runtime Status, and Runtime Truth, while leaving replay, permit issuance, checkpoint writes, growth/pruning, plasticity, fact promotion, action execution, and structural writes disabled.
_Avoid_: treating status visibility as a scheduler, executing the suggested endpoint automatically, or promoting stale/tampered ticket evidence into autonomy

**SNN Sleep Plasticity Scheduler Experiment** — the isolated, non-executing stability measurement over a verified Sleep Plasticity Autonomy Proposal. It repeatedly inspects the reviewed ticket, revision, action, and next-gate identity across bounded cycles, records deterministic hash provenance, and reports whether a future scheduler design is ready for operator review. It performs no tensor work, so CUDA is explicitly not applicable; it does not install a scheduler, call the suggested endpoint, replay activity, issue permits, write checkpoints, apply plasticity, grow/prune synapses, or mutate runtime state.
_Avoid_: calling repeated inspection scheduling, hiding endpoint calls inside an experiment, claiming CUDA acceleration for control-plane hashing, or promoting experiment readiness directly into scheduler installation

**SNN Sleep Plasticity Scheduler Design** — the HECSN-owned, read-only design proposal after a stable Scheduler Experiment. It binds the reviewed ticket and experiment hashes to bounded review cadence, minimum stable-cycle evidence, current-revision checks, and operator confirmation requirements for a future scheduler. It does not install a scheduler, call the suggested endpoint, replay activity, issue permits, write checkpoints, apply plasticity, grow/prune synapses, or mutate runtime state.
_Avoid_: treating a design as scheduler installation, hiding automatic endpoint execution behind review cadence, using stale experiment hashes, or turning operator-review intent into automatic plasticity

**SNN Sleep Plasticity Scheduler Design Review Ticket** — the operator-confirmed, Replay Controller-owned durable record that accepts one current Scheduler Design for later scheduler-installation review. The controller recomputes the design from verified internal ticket state before recording it, then binds operator identity, Runtime State revision, design hash, experiment hash, reviewed sleep-policy ticket, and named next gate. It does not install a scheduler, call the suggested endpoint, replay activity, issue permits, write checkpoints, apply plasticity, grow/prune synapses, or mutate transition memory.
_Avoid_: accepting caller-authored scheduler designs without controller recomputation, treating review as installation, reusing tickets after revision changes, or hiding endpoint execution behind operator confirmation

**SNN Sleep Plasticity Scheduler Design Review Ticket Queue** — the read-only Replay Controller visibility surface over accepted Scheduler Design Review Tickets. It verifies the current internal evidence chain, reports stale and tampered records, and exposes the latest verified accepted design for later autonomy planning without installing a scheduler or granting execution authority.
_Avoid_: treating queue presence as installation consent, surfacing stale designs as current, or calling reviewed endpoints from a visibility read

**SNN Sleep Plasticity Scheduler Installation Autonomy Proposal** — the non-executing autonomy candidate derived from the latest verified Scheduler Design Review Ticket. It exposes which accepted design is eligible for installation-preflight inspection while scheduler creation, timer registration, background workers, endpoint calls, replay, plasticity, and runtime mutation remain disabled.
_Avoid_: treating autonomy planning as installation, registering a timer from queue visibility, or calling the reviewed maintenance gate before installation review

**SNN Sleep Plasticity Scheduler Installation Preflight** — the deterministic read-only integrity review after the Scheduler Installation Autonomy Proposal. It binds accepted design evidence, review cadence, current Runtime State revision, and the named reviewed gate into an installation-review artifact without creating a scheduler, timer, worker, endpoint call, replay run, or plasticity update.
_Avoid_: treating preflight readiness as scheduler installation, hiding runtime work inside integrity checks, or bypassing operator installation review

**SNN Sleep Plasticity Review Scheduler Installation** — the operator-confirmed durable installation of one passive HECSN review scheduler configuration. It binds current installation-preflight evidence to a bounded cadence and computes when the next operator review inspection is due. It registers no operating-system timer, starts no worker, calls no endpoint, records no replay, and applies no plasticity.
_Avoid_: treating cadence tracking as automatic maintenance, hiding endpoint calls behind a due flag, or keeping an installation active after its revision-bound design evidence becomes stale

**SNN Sleep Plasticity Review Scheduler Cycle Inspection** — the read-only artifact emitted when an installed passive Review Scheduler is inspected. It becomes review-ready only after cadence is due and preserves the reviewed sleep-plasticity endpoint for operator inspection without calling that endpoint, running replay, or applying plasticity.
_Avoid_: treating a due flag as endpoint authority, combining timing inspection with replay execution, or allowing cycle inspection to mutate transition memory

**SNN Sleep Plasticity Review Scheduler Cycle Acknowledgment** — the operator-confirmed cadence-state transition after one due Review Scheduler Cycle Inspection has produced a verified due-cycle replay review ticket. It binds the active scheduler configuration, consumes that ticket once, appends an immutable successor configuration, advances the passive next-review deadline by exactly one accepted interval, and records the acknowledgment count while calling no endpoint, running no replay, and applying no plasticity.
_Avoid_: acknowledging a non-due cycle, reusing an inspection hash, skipping multiple cadence intervals in one acknowledgment, or hiding consolidation work inside scheduler bookkeeping

**SNN Sleep Plasticity Review Scheduler Cycle Acknowledgment Preflight** — the deterministic read-only integrity review before Cycle Acknowledgment. It verifies the active scheduler configuration hash, due-state inspection, current due-cycle replay review ticket, deadline binding, and one-time consumption rule without advancing cadence or executing consolidation work.
_Avoid_: treating preflight readiness as acknowledgment, accepting generic replay-review tickets, or letting callers supply timestamps or cadence parameters

**SNN Sleep Plasticity Review Scheduler Cycle Autonomy Proposal** — the non-executing autonomy candidate derived from a due Review Scheduler Cycle Inspection. It exposes the reviewed sleep-plasticity gate as the next operator inspection target while timer registration, workers, endpoint calls, replay, checkpoints, plasticity, and runtime mutation remain disabled.
_Avoid_: treating cadence readiness as replay selection, endpoint authority, or permission to modify synapses

**SNN Due-Cycle Bounded Replay-Selection Proposal** — the deterministic non-executing nomination of a bounded set of current verified replay-evaluation contexts after a passive sleep review cadence becomes due. It combines timing evidence with consolidation priority for operator inspection only; it does not select replay-window content, record an artifact, run replay, call an endpoint, or modify synapses.
_Avoid_: treating a nominated context as a replay payload, allowing caller-authored queue evidence, or turning salience ranking into structural-write authority

**SNN Due-Cycle Replay Artifact Recording Review Proposal** — the deterministic read-only adapter that binds one due-cycle nominated replay-evaluation context to the existing artifact-recording policy recommendation. It proves that cadence, priority, policy, and current-revision context evidence agree before operator review; it does not create a ticket, derive replay-window content, record an artifact, run replay, or modify synapses.
_Avoid_: creating a parallel artifact recorder, treating policy agreement as durable consent, or bypassing the existing internal-ledger proposal and ticket path

**SNN Sleep Phase Separation Proposal** — the read-only sleep-cycle artifact that separates NREM-like replay nomination from REM-like stabilization review. The NREM-like phase may nominate bounded replay evidence for operator inspection; the REM-like phase may only become review-ready after cycle-acknowledgment preflight is ready. Neither phase records artifacts, runs replay, writes checkpoints, or applies plasticity.
_Avoid_: treating phase labels as biological claims, merging replay nomination with stabilization, or using a ready NREM-like phase as mutation authority

**SNN REM-Like Homeostatic Stabilization Preflight** — the read-only bridge from REM-like Stabilization/Integration Review to the existing Transition Memory Homeostasis command. It verifies phase readiness, transition-memory pressure, and already-bounded maintenance parameters, then names the operator-reviewed maintenance endpoint without executing it, writing checkpoints, pruning synapses, or mutating runtime state.
_Avoid_: treating preflight as maintenance, bypassing operator confirmation, or inventing a second homeostatic mutation path

**SNN Readout Rollout Server-State Binding** — the evidence rule that bounded SNN readout rollout text must be derived from HECSN-held sparse transition memory at the runtime facade, not caller-carried weights, mocks, external checkpoints, or freeform decoding. The rollout records the server transition-memory hash, the rollout tensor hash, prediction/evaluation provenance matches, current state revision, bounded rollout parameters, and trajectory evidence while remaining non-mutating.
_Avoid_: letting API callers supply readout-rollout weights, silently accepting widened rollout bounds, treating a rollout label string as fact promotion, or hiding freeform generation behind sparse-readout terminology

**SNN Readout Rollout Runtime Truth Binding** — the compact Runtime Truth visibility gate for SNN Readout Rollout Server-State Binding. It exposes whether HECSN currently holds sparse transition memory for bounded rollout review, plus server memory hash/count and negative execution flags, without calling rollout, replay, ledger recording, checkpointing, plasticity, or text generation.
_Avoid_: treating the Runtime Truth binding as a rollout candidate, exposing labels/text/prediction reports inside status, accepting caller-carried transition weights, or promoting cognition/facts/actions from status visibility

**SNN Readout Rollout Consolidation Path Evidence** — the compact Runtime Truth visibility gate for the durable rollout rehearsal/consolidation path. It summarizes recorded rollout-replay evidence count, latest rollout and transition-memory hashes, and the next review gate while keeping rehearsal, consolidation, live replay, ledger recording, checkpointing, plasticity, and text generation disabled.
_Avoid_: treating recorded rollout visibility as rehearsal execution, exposing replay targets or labels inside Runtime Truth, promoting consolidation from status, or using this summary as a substitute for the detailed ledger artifacts

**SNN Readout Applied Synapse Provenance Evidence** — the compact Runtime Truth visibility gate for applied readout-derived sparse transition memory. It summarizes sparse weight count, provenance row count, replay-regeneration count, complete local-edge provenance count, missing local-edge provenance, invalid rollout step order, orphan weights, and dangling provenance without running the full audit, exposing raw weights, executing replay, writing checkpoints, applying plasticity, or generating language.
_Avoid_: treating Runtime Truth as the audit itself, exposing `synapse_provenance_by_key` or sparse weights in status, hiding missing rollout-local provenance after regeneration, or letting status reads trigger repair/plasticity

**Server-Bound Rollout Ledger Evidence** — durable rollout evidence whose replay evaluation, ledger material hash, and rehearsal policy candidate all preserve the server transition-memory hash, hash-match flag, and `service.runtime_facade.snn_language_plasticity_runtime_state` source. Missing or tampered server binding blocks ledger recording or removes the event from rehearsal eligibility.
_Avoid_: relying only on generic transition-weight hashes, rehearsing rollout records with missing server-memory provenance, or letting durable evidence forget that the public rollout path ignored caller-carried weights

**SNN Language Readout Rollout Regeneration Permit Request** — an operator-confirmed command that submits a reviewed rollout regeneration replay-artifact binding to the existing replay-controller regeneration permit issuer. It may durably record a permit through replay-controller provenance, but it does not apply regeneration, grow synapses, prune synapses, persist weights, write checkpoints, execute replay, or bypass the checkpoint-backed regeneration executor.
_Avoid_: issuing without operator confirmation, using fabricated replay artifacts, treating a permit as synapse mutation, or skipping the executor's checkpoint transaction

**SNN Language Readout Rollout Regeneration Application Preflight** — a HECSN-owned final preflight that converts a verified rollout regeneration permit request into an executor-shaped regeneration proposal only when the current state revision and checkpoint intent are present. It does not call the executor, save checkpoints, grow synapses, prune synapses, apply plasticity, or mutate runtime state.
_Avoid_: treating proposal assembly as application, omitting checkpoint intent, using stale state revisions, or bypassing the executor's operator confirmation

**SNN Language Readout Rollout Regeneration Application** — an operator-confirmed rollout command that delegates a preflight-approved, replay-bound regeneration proposal to the existing checkpoint-backed transition-memory regeneration executor. It may grow bounded sparse transition synapses only after server-owned replay mismatch evidence, replay-controller permit verification, checkpoint transaction, row-mass limits, and rollback path all align; it does not implement a second write path or decode/generate language.
_Avoid_: applying rollout growth directly, accepting stale preflight revisions, mismatching checkpoint intent, bypassing replay-permit verification, using adapter-only mismatch defaults, or treating executor-blocked proposals as partial mutation

**SNN Language Transition Memory Prediction Evaluation** — the read-only gate that compares baseline sparse next-code prediction with persistent-memory-assisted prediction across grounded evaluation windows. It measures mismatch deltas, influence count, improved/worsened sequence counts, and persistent weight coverage before a readout draft can be treated as useful evidence.
_Avoid_: claiming persistent memory helps because it exists, using one influenced prediction as utility proof, or treating evaluation as training

**SNN Language Readout Emission** — the operator-visible bounded language output derived from a ready SNN Language Readout Draft. It binds emitted labels/text to the draft, readout trajectory hash, sparse prediction hash, transition-memory evaluation hash, and persistent transition-weight hash while staying read-only: it cannot promote facts/actions, become a cognition substrate, write checkpoints, apply plasticity, or claim freeform language generation.
_Avoid_: displaying blocked drafts as output, treating emitted labels as truth, hiding freeform decoding behind readout terminology, or using emission as plasticity consent

**SNN Language Readout Emission Review** — the operator-confirmed display-history record for a ready SNN Language Readout Emission. It records that bounded SNN output was reviewed, with emission/prediction/trajectory/transition-memory hashes and operator evidence, but remains separate from replay memory, plasticity consent, fact promotion, action authority, and cognition-substrate evidence.
_Avoid_: treating display acknowledgment as learning, replay priority, truth, action approval, or permission to mutate transition memory

**SNN Language Readout Emission Review History** — the operator-facing read-only inspection surface for reviewed SNN Language Readout Emissions. It may display the already-reviewed bounded text and labels with their provenance hashes, but it returns only emission-review events and cannot expose draft ledger events, rollout events, replay targets, prediction reports, transition-memory evaluations, or any mutation authority.
_Avoid_: using display history as replay memory, training data, action/fact approval, a broad ledger export, or a source of unreviewed generated text

**SNN Emission Replay Evaluation Policy** — the read-only selector that compares reviewed SNN Language Readout Emissions against internal SNN Language Readout Evidence Ledger rows. A reviewed emission can become a replay-evaluation policy candidate only when prediction, transition-memory evaluation, persistent-weight, label, and grounding evidence match an existing HECSN-owned sparse readout evidence row; display text is reduced to hashes and is not a replay source.
_Avoid_: replaying from text history alone, accepting reviewed labels without matching sparse evidence, recording replay memory from policy output, or treating policy readiness as replay/plasticity permission

**SNN Emission Replay Evaluation Design** — the read-only design seed that turns policy-ready reviewed emissions into hash-bound replay-context review inputs. It verifies candidates still match the internal SNN Language Readout Evidence Ledger, requires device-review evidence, and points to server-computed replay-context recording as the next gate without recording that context, replaying activity, exposing display text, applying plasticity, or promoting facts/actions.
_Avoid_: treating the design as replay memory, accepting display text or labels as replay windows, recording replay contexts from design output alone, skipping server-computed mismatch/pressure evidence, or using design readiness as plasticity permission

**SNN Emission Replay Design Path Evidence** — the compact Runtime Truth visibility summary for the reviewed-emission replay-design path. It reports reviewed-emission counts, internal readout-evidence counts, matched hash-only candidate counts, latest provenance hashes, and the next review gate while keeping raw text, labels, candidate arrays, replay-context recording, replay execution, plasticity, checkpoints, facts/actions, and cognition-substrate promotion disabled.
_Avoid_: treating status visibility as the design artifact, exposing reviewed text or labels inside Runtime Truth, using matched counts as replay memory, or skipping the device-reviewed design and server-computed replay-context gates

**SNN Emission Replay Context Review** — the operator-confirmed bridge from SNN Emission Replay Evaluation Design into the existing server-held Evaluated SNN Transition-Memory Replay Context recorder. It verifies the design seed, prediction hash, observed sparse readout slots, device evidence, runtime-truth/rollback pressure gate, and operator confirmation before recording mismatch/pressure evidence through the Replay Controller; it does not replay activity, record replay memory, apply plasticity, write checkpoints, expose reviewed text, or promote facts/actions.
_Avoid_: recording replay contexts from text or labels alone, bypassing mismatch/pressure recomputation, accepting unconfirmed design seeds, treating context recording as replay execution, or using the bridge as plasticity permission

**SNN Emission Replay Context Lineage** — the hash-verified source metadata stored with an Evaluated SNN Transition-Memory Replay Context when it originates from reviewed SNN emission evidence. It binds the emission review, emission hash, readout evidence hash, replay-design hash, seed hash, prediction hash, and operator identity into the replay context so later queues/artifacts can audit why the context exists without reading display text.
_Avoid_: treating mismatch/pressure hashes as enough lineage, dropping emission provenance after context recording, storing raw display text as context metadata, or accepting lineage that does not verify against the context evidence hash

**SNN Replay Context Lineage Propagation** — the hash-only carrying of SNN Emission Replay Context Lineage from a verified Evaluated SNN Transition-Memory Replay Context into the SNN Replay Consolidation Priority Queue, SNN Replay Artifact Recording Policy Proposal, and SNN Replay Artifact Recording Review Ticket. It exposes the source metadata hash and compact emission/readout/prediction/design hashes for audit continuity while omitting raw source metadata, operator identity, display text, labels, replay execution, artifact recording, and structural-write authority.
_Avoid_: treating lineage presence as consolidation eligibility, copying raw source metadata or operator identity into queue candidates, accepting policy/ticket lineage that no longer matches the verified replay context, or using lineage hashes as a substitute for server-held context verification

**SNN Replay Artifact Lineage** — the hash-only continuation of verified SNN Replay Context Lineage Propagation into an Evaluated SNN Transition-Memory Replay Artifact and any Replay-Backed Regeneration Permit issued from it. It binds artifact and permit evidence back to the source metadata hash and compact emission/readout/prediction/design hashes while still requiring the server-held replay context, review ticket, internal readout evidence, and current Runtime State revision to verify.
_Avoid_: recording evaluated artifacts from lineage alone, using lineage hashes as regeneration authority, dropping source lineage at permit issuance, or exposing raw source metadata/operator identity/text/labels through artifact or permit evidence

**SNN Applied Replay Lineage Provenance** — the hash-only preservation of SNN Replay Artifact Lineage on applied replay-regenerated synapses. It lets Runtime Truth and Synapse Provenance Audit confirm that a structural growth record still points back to the replay artifact source hash and compact emission/readout/prediction lineage while separately requiring local-edge provenance, bounded weights, checkpoint-backed application, and server-verified permits.
_Avoid_: treating applied lineage as proof that a synapse should exist, accepting partial lineage when a replay-regenerated synapse claims source metadata, exposing raw text/labels/operator identity, or bypassing local-edge and checkpoint evidence

**SNN Durable Applied Replay Lineage Checkpoint Evidence** — the checkpoint-level summary proving that saved SNN language plasticity state contains complete hash-only Applied Replay Lineage Provenance for replay-regenerated synapses. It records lineage counts and a deterministic lineage material hash beside the saved plasticity state so restore audits can detect missing lineage without running replay, applying plasticity, or exposing raw text/labels/operator identity.
_Avoid_: treating checkpoint lineage summary as replay-artifact verification, permitting regeneration from a checkpoint summary alone, storing raw source metadata in checkpoint summaries, or replacing full Synapse Provenance Audit with the summary

**SNN Applied Replay Lineage Restore Validation** — the restore-time comparison between Durable Applied Replay Lineage Checkpoint Evidence and the replay-regenerated synapse provenance hydrated from a checkpoint. It recomputes lineage counts and deterministic material hash from restored plasticity state and reports whether the saved summary still matches, without replay execution, plasticity application, permit issuance, or raw text/labels/operator identity exposure.
_Avoid_: silently trusting a saved checkpoint summary, treating restore validation as mutation authority, accepting restored replay-regenerated synapses with mismatched lineage summaries, or running replay/plasticity from restore validation

**SNN Applied Replay Lineage Restore Visibility** — the compact Runtime Truth evidence over SNN Applied Replay Lineage Restore Validation. It reports whether restore validation is available, whether saved/restored lineage counts and material hashes match, and whether the restored checkpoint may proceed to readout synapse provenance audit review, while remaining advisory and unable to load checkpoints, run replay, apply plasticity, issue permits, or write checkpoints.
_Avoid_: hiding restore mismatches in raw checkpoint metadata, treating visibility as restore execution, using status as a checkpoint loader, or promoting restored synapses to replay/plasticity/fact/action authority from status alone

**SNN Restore-Gated Synapse Audit Readiness** — the rule that Runtime Truth may report applied replay-regenerated synapse provenance as ready for readout synapse provenance audit only when local provenance is complete and any available SNN Applied Replay Lineage Restore Validation is not mismatched. Fresh runtime state without restore validation may still proceed on complete local evidence, but restored checkpoint state with count or material-hash disagreement blocks audit readiness.
_Avoid_: letting complete local-edge provenance override a failed restore validation, requiring restore metadata for never-restored live state, or treating a passing readiness gate as permission to replay, mutate, issue permits, or promote facts/actions

**SNN Restore-Gated Synapse Audit Execution** — the non-mutating audit behavior that consumes SNN Applied Replay Lineage Restore Validation when present. The audit endpoint may still return an audit artifact for operator inspection, but a restored lineage mismatch is required evidence failure and blocks the audit artifact from becoming review-ready.
_Avoid_: making Runtime Truth stricter than the actual audit endpoint, hiding restored checkpoint mismatch during audit, throwing away inspectable audit rows on mismatch, or using audit execution as replay/plasticity/checkpoint authority

**SNN Readout Emission Review History Evidence** — the compact Runtime Truth visibility summary for reviewed bounded SNN language emissions. It reports review counts, unique emission/trajectory/transition-memory counts, latest hashes, and the next operator-inspection gate without exposing raw text/labels or becoming a ledger writer, replay source, plasticity signal, fact/action authority, checkpoint command, or cognition-substrate claim.
_Avoid_: exposing display text inside status, treating a reviewed emission as replay memory, applying plasticity from review history, or using compact Runtime Truth evidence as a substitute for the dedicated emission-review ledger

**SNN Language Evaluated Prediction Provenance** — the canonical-hash binding between a sparse next-code prediction, the training window, current readout slots, persistent transition-memory weights, and the transition-memory evaluation window that tested it. A readout draft may be review-ready only when its prediction hash appears in the evaluation's memory-backed prediction hashes and the training/current/memory hashes match.
_Avoid_: replaying a passing evaluation for an unrelated prediction, accepting hand-authored summary booleans, or treating provenance as optional metadata

**SNN Language Readout Evidence Ledger** — the HECSN-owned append-only memory of operator-confirmed, provenance-matched bounded readout drafts. It records prediction, evaluation, transition-memory, label, revision, and operator evidence for later replay/evaluation without treating text labels as thoughts, facts, actions, or a cognition substrate.
_Avoid_: hidden read-side ledger mutation, recording unready drafts, storing free-form generated text as memory, or using ledger presence as autonomous truth

**SNN Language Readout Replay Priority** — the read-only advisory ranking over SNN Language Readout Evidence Ledger entries. It prioritizes provenance-bound readout evidence for possible isolated SNN rehearsal review using deterministic recency, repetition, and transition-memory reuse signals; it cannot execute replay, apply plasticity, generate language, promote facts/actions, or become the cognition substrate.
_Avoid_: treating priority as replay execution, synthesizing new text from labels, training from priority alone, or using priority scores as truth

**SNN Language Readout Rehearsal Evaluation** — the isolated read-only evaluator after SNN Language Readout Replay Priority. It turns prioritized ledger candidates into sparse rehearsal vectors, measures activation sparsity, similarity, priority support, and device placement, and can mark evidence ready for operator rehearsal review without mutating runtime state, applying plasticity, training, or generating language.
_Avoid_: treating rehearsal evaluation as live replay, using sparse vectors as thoughts, applying weights from rehearsal, or promoting labels as facts/actions

**SNN Language Readout Rehearsal Experiment** — the isolated replay-pressure simulation after SNN Language Readout Rehearsal Evaluation. It estimates whether replay cycles over sparse readout evidence would be non-worsening and stable before any future replay design, while discarding traces, persisting no weights, and keeping live replay, plasticity, language generation, facts, actions, and cognition-substrate promotion disabled.
_Avoid_: treating simulated pressure gain as applied learning, running live replay from the experiment, or accepting rehearsal evidence as autonomous truth

**SNN Language Readout Replay Design** — the read-only operator-review design after SNN Language Readout Rehearsal Experiment. It selects hash-addressed readout evidence targets and bounds future isolated replay by candidate count, replay cycles, pressure-gain threshold, stability floor, and rollback evidence; it still cannot execute replay, apply plasticity, train, generate language, or promote facts/actions.
_Avoid_: treating replay design as replay execution, widening targets beyond recorded evidence, bypassing rollback policy, or allowing design artifacts to mutate transition memory

**SNN Language Readout Replay Dry Run** — the operator-approved isolated replay simulation after SNN Language Readout Replay Design. It replays only internal-ledger-backed, hash-addressed readout targets as ephemeral sparse tensors with explicit device/CUDA evidence, and reports pressure/stability metrics without writing checkpoints, mutating runtime state, applying plasticity, decoding/generating language, or promoting facts/actions.
_Avoid_: treating dry-run pressure gain as live learning, silently falling back from requested CUDA, accepting caller-only evidence not present in the internal ledger, or returning trained weights

**SNN Language Readout Plasticity Preflight** — the read-only bridge after SNN Language Readout Replay Dry Run. It checks whether dry-run sparse replay evidence is stable enough to review as local-plasticity intent and emits bounded candidate replay sequences for later application design; it still cannot apply weights, persist checkpoints, train, generate language, or promote facts/actions.
_Avoid_: treating preflight as an application design, using caller-only dry-run evidence, applying candidate synapses, or skipping the existing plasticity application/shadow/live gates

**SNN Language Readout Plasticity Replay Bridge** — the read-only adapter after SNN Language Readout Plasticity Preflight. It converts internal-ledger-backed readout preflight evidence into the existing `snn_language_plasticity_replay_experiment.v1` contract and carries canonical replay sequences, an application-design-compatible block, a HECSN sparse-transition target hint, and checkpoint-transaction requirements for the established shadow/readiness/preflight path; it cannot apply plasticity, mutate runtime state, persist checkpoints, generate language, or promote facts/actions.
_Avoid_: caller-side reshaping that drops provenance, spoofing canonical replay experiments, recomputing preflight bounds in the bridge, bypassing checkpoint/restore readiness, or treating bridge compatibility as live update permission

**Readout-Derived Live Plasticity Application** — the checkpoint-backed command path where SNN Language Readout Plasticity Replay Bridge evidence may finally update Persistent SNN Language Transition Memory through the existing SNN Language Plasticity Live Application Executor. It requires shadow-delta evidence, live readiness, live preflight, current Runtime State revision, explicit operator confirmation, and a restorable checkpoint; it still cannot generate or decode language, load external checkpoints, or promote facts/actions.
_Avoid_: adding a second readout-specific executor, applying without confirmation, skipping checkpoint restore evidence, or treating bounded readout labels as autonomous truth

**Readout-Derived Synapse Provenance** — the per-synapse audit trail attached to readout-derived live plasticity. It preserves the readout evidence hash, prediction hash, transition-memory evaluation hash, persistent transition-weight hash, source sequence, and source sparse indices on both shadow deltas and applied synapses, and persists a `synapse_provenance_by_key` map across checkpoints for restored-runtime audit.
_Avoid_: storing only numeric weights after readout-derived mutation, dropping provenance at the shadow-delta boundary, or making checkpoint restore unable to explain why a sparse transition exists

**SNN Language Readout Synapse Provenance Audit** — the read-only audit surface over readout-derived live plasticity. It compares persisted sparse transition weights and `synapse_provenance_by_key` against exact internal Readout Evidence Ledger entries, including canonical 64-neuron synapse keys, in-range source indices, finite bounded weights, recomputed ledger material hashes, matching prediction, transition-evaluation, persistent-weight hashes, and replay-regeneration local-edge provenance when a synapse was grown from rollout consolidation. It reports orphan weights, incomplete provenance, missing local rollout edge evidence, or invalid rollout step order, and can mark restored runtime state ready for operator audit review without applying plasticity, generating language, or promoting facts/actions.
_Avoid_: using audit success as permission to mutate, accepting weights without ledger-backed provenance, dropping rollout local-edge hashes from replay-regenerated synapses, treating hash-field mismatches or tampered ledger material as harmless, accepting malformed sparse keys or unbounded weights, or letting row-return limits shrink validation scope

**SNN Language Transition Memory Homeostasis** — the checkpoint-backed maintenance command for Persistent SNN Language Transition Memory. It decays transition weights, bounds outgoing row mass, prunes weak synapses, and records maintenance evidence through `/terminus/snn-language-sequence/plasticity-homeostatic-maintenance`; it requires explicit operator confirmation and current revision evidence.
_Avoid_: unbounded transition growth, hidden pruning during prediction reads, or maintenance without a rollback checkpoint

**Evaluated SNN Transition-Memory Replay Context** — the Replay Controller-owned, server-held, current-revision evidence record binding HECSN-computed sequence mismatch and plasticity-pressure reports for replay review. It is recorded from source inputs, not caller-carried reports, and artifacts or permits that depend on it must verify its server-held identity and content hash at the current Runtime State revision.
_Avoid_: accepting caller-authored mismatch or pressure payloads as permit evidence, reusing stale contexts after revision advance, or treating context recording as mutation authority

**SNN Replay Consolidation Priority Queue** — the read-only advisory ranking of current Evaluated SNN Transition-Memory Replay Contexts for operator consolidation review. It scores only verified server-held contexts using bounded prediction-error, plasticity-pressure, recency, and internal readout-support evidence; it may recommend which context to review next but cannot record artifacts, replay live activity, mutate transition memory, generate language, promote facts, or issue permits.
_Avoid_: treating queue rank as replay execution, artifact recording, structural-write permission, fact/action promotion, or evidence from stale/tampered contexts

**SNN Replay Artifact Recording Policy Proposal** — the read-only autonomy policy surface that selects a policy-ready SNN Replay Consolidation Priority Queue candidate for operator artifact-recording review. It applies bounded policy thresholds and verifies the recommended replay context is still current, but it cannot record an artifact, run replay, mutate transition memory, generate language, promote facts/actions, or issue permits.
_Avoid_: treating a recommended policy proposal as consent, artifact evidence, live replay execution, structural mutation, or a substitute for operator review

**SNN Replay Artifact Recording Review Ticket** — the Replay Controller-owned durable intent tag that binds a current SNN Replay Artifact Recording Policy Proposal, its Evaluated SNN Transition-Memory Replay Context, the current Runtime State revision, and the confirming operator before evaluated replay-artifact recording. When issued from a due sleep-review cycle it also binds the Due-Cycle Replay Artifact Recording Review Proposal and its source selection hash. It records consent to review the selected consolidation target, but it is not an artifact, replay execution, mutation authority, generation permission, or regeneration permit.
_Avoid_: accepting caller-copied policy proposals as consent, reusing stale tickets after revision advance, sharing tickets across operators, or treating ticket creation as artifact recording

**Evaluated SNN Transition-Memory Replay Artifact Proposal** — the read-only policy-derived consolidation proposal built from ranked internal Readout Evidence Ledger entries and a verified Evaluated SNN Transition-Memory Replay Context. It preserves per-label grounding and may propose an operator-recordable replay window only when every selected label is grounded, high mismatch is present, and plasticity pressure is ready for review.
_Avoid_: accepting caller-round-tripped replay windows as evaluated evidence, collapsing partial grounding into a single optimistic flag, or treating a proposal as mutation authority

**SNN Transition-Memory Replay Artifact** — the Replay Controller-owned durable consolidation context for structural review. A regeneration-permit-eligible artifact records an operator-confirmed, internal-ledger-backed Evaluated SNN Transition-Memory Replay Artifact Proposal together with a verified SNN Replay Artifact Recording Review Ticket, a verified Evaluated SNN Transition-Memory Replay Context, normalized operator identity, and the current Runtime State revision. Raw caller-window artifacts may remain auditable, but they cannot authorize regeneration.
_Avoid_: treating general replay audit samples as SNN consolidation evidence, accepting caller-authored artifact IDs, issuing regeneration permits from raw replay windows, or treating artifact recording as structural-write authority

**Replay-Window Synapse Provenance** — the hash-addressed ancestry carried from internal Readout Evidence Ledger entries through an evaluated replay artifact, regeneration permit, regeneration proposal, and regenerated sparse transition synapse. It lets later audits explain which grounded readout evidence justified a grown synapse without loading external SNN checkpoints or treating replay labels as generated language.
_Avoid_: growing transition-memory synapses from replay-window hashes alone, dropping readout evidence hashes at the permit boundary, or mixing caller-window artifacts with internal-ledger-backed regeneration

**SNN Language Transition Memory Sleep Policy** — retired wording for **SNN Transition Memory Sleep Plasticity Policy**. Use the canonical term so sleep pressure, replay evidence, rollout-regeneration evidence, and homeostatic maintenance review stay under one policy concept.
_Avoid_: creating a second sleep-policy surface, treating the alias as a compatibility API, or reviving retired Cortex sleep snapshots as inputs

**SNN Language Sequence Mismatch Probe** — the read-only prediction-error surface after SNN Language Sequence Prediction Probe, available through `/terminus/snn-language-sequence/mismatch`. It compares predicted sparse population-code indices with the next observed grounded sparse code and reports precision, recall, mismatch score, and sparse-code deltas without applying learning or generating text.
_Avoid_: treating mismatch as an automatic learning signal, fact promotion, generated thought, or runtime model update

**SNN Language Plasticity Pressure Gate** — the read-only gate after SNN Language Sequence Mismatch Probe, available through `/terminus/snn-language-sequence/plasticity-pressure`. It converts sparse prediction error into operator-reviewable local plasticity pressure and candidate update focus, but it cannot apply plasticity, train runtime weights, generate language, or promote facts.
_Avoid_: treating pressure as permission to learn, mutating sequence weights from status reads, or using mismatch pressure as text generation evidence

**SNN Language Plasticity Trial** — an isolated simulation after SNN Language Plasticity Pressure Gate, available through `/terminus/snn-language-sequence/plasticity-trial`. It estimates whether an error-modulated local sequence update would reduce sparse prediction pressure using ephemeral update evidence only; it cannot apply plasticity, persist weights, train runtime state, generate language, or promote facts.
_Avoid_: treating simulated pressure reduction as a live weight update, storing trial weights, or bypassing isolated replay/operator approval

**SNN Language Plasticity Replay Evaluation** — the read-only gate after SNN Language Plasticity Trial, available through `/terminus/snn-language-sequence/plasticity-replay-evaluation`. It checks replay-window evidence, expected pressure reduction, Runtime Truth delta, and rollback policy before a future isolated replay experiment can be reviewed; it cannot apply plasticity, persist weights, train runtime state, generate language, or promote facts.
_Avoid_: treating replay evaluation as replay execution, applying trial updates, or using replay readiness as fact/action promotion

**SNN Language Plasticity Replay Experiment** — the isolated sparse-code rehearsal after SNN Language Plasticity Replay Evaluation, available through `/terminus/snn-language-sequence/plasticity-replay-experiment`. It measures grounded replay coverage and simulated pressure stability from replay sequences while discarding replay traces and refusing runtime weight updates.
_Avoid_: treating replay rehearsal as live learning, storing replay weights, promoting replay outputs as facts, or using it as text generation

**SNN Language Plasticity Application Design** — the read-only design gate after SNN Language Plasticity Replay Experiment, available through `/terminus/snn-language-sequence/plasticity-application-design`. It bounds a possible future local update with learning-rate, weight-delta, locality, normalization, device evidence, Runtime Truth, and rollback constraints, but it still cannot apply plasticity or persist weights.
_Avoid_: treating an application design as a live update, bypassing rollback evidence, widening updates beyond local sparse support, or promoting replay evidence as language/facts

**SNN Language Plasticity Shadow Application** — the read-only verifier after SNN Language Plasticity Application Design, available through `/terminus/snn-language-sequence/plasticity-shadow-application`. It checks a proposed shadow update against weight-delta, locality, pressure-stability, device evidence, Runtime Truth, and rollback bounds before any future live application can be reviewed.
_Avoid_: treating shadow deltas as applied weights, skipping rollback, using shadow verification as fact promotion, or calling it runtime learning

**SNN Language Plasticity Shadow Delta** — the HECSN-owned read-only measurement that derives a bounded local weight-delta candidate from sparse replay indices and application-design limits, available through `/terminus/snn-language-sequence/plasticity-shadow-delta`. It can report affected synapse count, maximum absolute delta, locality radius, pressure change, and device placement, but it cannot apply or persist the update.
_Avoid_: hand-written shadow deltas as proof of live readiness, treating measured deltas as stored weights, or skipping the Shadow Application verifier

**SNN Language Plasticity Live Application Readiness** — the final read-only gate before any checkpoint-backed live language plasticity application. It checks Shadow Application evidence, checkpoint/restore rollback readiness, and explicit operator approval through `/terminus/snn-language-sequence/plasticity-live-application-readiness`; it cannot apply weights, mutate runtime state, train, generate, or promote facts.
_Avoid_: treating readiness as application, bypassing checkpoint restore, or using operator approval without measured shadow evidence

**SNN Language Plasticity Live Application Preflight** — the read-only pre-execution review after Live Application Readiness. It checks that the target is a HECSN-owned sparse mutable transition-weight surface and that a checkpoint transaction is saved, restorable, and records the shadow delta before `/terminus/snn-language-sequence/plasticity-live-application` may be called.
_Avoid_: treating preflight as mutation, accepting non-HECSN targets, or applying a delta without a saved rollback transaction

**SNN Language Plasticity Live Application Executor** — the command-shaped, checkpoint-backed mutation boundary for bounded local SNN language transition updates. It applies measured shadow deltas only to HECSN-owned sparse transition weights, requires current state revision, explicit confirmation, operator approval, real checkpoint save, and bounded non-worsening pressure evidence; it does not generate text, decode text, load external checkpoints, or promote facts.
_Avoid_: placing live mutation in read models, accepting spoofed readiness without revalidation, or applying broad dense updates outside sparse local support

**SNN Language Plasticity Path Evidence** — the compact Runtime Truth evidence summary for the SNN language plasticity chain. It lists the current gates through SNN Language Plasticity Live Application Preflight, reports checkpoint/restore rollback readiness, and records the invariant that future live application requires device evidence, Runtime Truth delta, and rollback evidence while generation, training, plasticity application, and runtime mutation remain false.
_Avoid_: treating the Runtime Truth summary as execution evidence, hiding detailed gate artifacts, or promoting live learning from summary presence alone

**Developmental Plasticity** — the Subcortex mechanism family for growing, pruning, and stabilizing assemblies, synapses, routing prototypes, and replay policies under evidence gates.
_Avoid_: self-replication as unchecked code mutation, permanent growth without pruning

**Local Plasticity Evidence** — the read-only report from local STDP eligibility traces, synaptic scaling, inhibitory balance, spike-health state, synaptic validation, and device placement. It can support Developmental Plasticity readiness, but it is not a command to change synapses or topology.
_Avoid_: treating plasticity telemetry as automatic self-modification

**Structural Mutation Ledger** — the runtime evidence record for bounded topology growth/pruning events such as hypercube binding hub outreach. It records added/removed sparse edges and recent mutation samples so structural plasticity can be audited and rolled into Runtime Truth rather than treated as hidden model drift.
_Avoid_: undocumented rewiring, unbounded topology mutation, growth claims without prune evidence

**Structural Plasticity Gate Artifact** — the operator-facing read-only artifact that combines ConceptStore growth pressure, binding topology mutation ledger evidence, and CUDA/device placement into structural-promotion readiness. It can mark a structural change ready for isolated evaluation, but it cannot mutate topology.

**Structural Plasticity Runtime Truth Evidence** — the compact Runtime Truth visibility summary for the Structural Plasticity Gate Artifact. It exposes promotion status, review readiness, local-plasticity rule/backend, synaptic-validation availability, spike-health risk, device-evidence counts, and required isolated-evaluation/rollback/Runtime Truth/device gates while omitting structural cases, recent topology events, active growth concepts, full device payloads, and mutation commands.
_Avoid_: using status as the structural-plasticity artifact, hiding rollback/runtime-truth requirements, or promoting growth/prune from compact visibility

**Isolated Structural Plasticity Evaluation** — a read-only comparison of pre/post structural snapshots after a bounded growth/prune trial outside the live runtime, available through `/terminus/subcortical-structural-plasticity/evaluate`. It reports edge deltas, spike-health delta, Runtime Truth delta, CUDA/device consistency, and rollback evidence; even when ready for operator review, it still cannot authorize structural mutation by itself.
_Avoid_: calling concept observation, binding, grow/prune, or structural refresh from a readiness artifact

**Structural Evaluation Snapshot Binding** — the hash-bound identity evidence for an Isolated Structural Plasticity Evaluation. It records canonical pre/post snapshot hashes, optional state revisions, revision-order validity, whether a nonzero structural delta exists, and whether rollback evidence is bound to the exact pre-snapshot hash, while omitting raw snapshots from compact promotion evidence.
_Avoid_: reviewing unbound snapshots, treating identical snapshots as growth/prune evidence, accepting rollback evidence for a different pre-state, or exposing raw structural state in compact status

**Structural Mutation Design** — the operator-confirmed read-only design after Isolated Structural Plasticity Evaluation. It binds the evaluation, pre/post snapshot hashes, rollback pre-snapshot hash, bounded edge delta, and operator confirmation into a design hash for later preflight review; it does not call growth/prune code, write checkpoints, or mutate topology.
_Avoid_: treating design readiness as mutation authority, accepting unbound evaluation evidence, or skipping the checkpoint-backed preflight gate

**Structural Mutation Preflight** — the read-only checkpoint gate after Structural Mutation Design. It recomputes the design hash, checks the expected runtime revision, requires a rollback checkpoint path and future restore verification, and still blocks direct structural mutation until the Structural Mutation Application executor receives explicit operator confirmation.
_Avoid_: treating preflight readiness as consent, writing checkpoints during preflight, or calling growth/prune before the application executor

**Structural Mutation Application** — the checkpoint-backed command path where a reviewed Structural Mutation Preflight may finally call HECSN-owned structural capacity refresh. It requires current Runtime State revision, bound design hash evidence, explicit operator confirmation, a restorable pre-mutation checkpoint, an observed concept-store structural delta, and a verified committed checkpoint; it does not load external checkpoints, use external SNN implementations as dependencies, or bypass rollback.
_Avoid_: claiming success on no-op refresh, importing reference model code as a runtime dependency, or creating a second unverified structural write path

**Path Retirement Gate** — the rule that a runtime path should be removed when it adds complexity without improving liveness, grounding, efficiency, or evidence quality. Legacy paths may exist only as short-lived migration scaffolding while replacement evidence is gathered; after the active Subcortex path exists, compatibility code is deleted rather than kept dormant. Negative regression tests and retirement notes may remain as boundary evidence, but runtime aliases, mocks, dormant compatibility APIs, and old extension points should not.

**Compatibility Is Not Permanence** — migration adapters are acceptable only while callers are actively moving to the real owner. Once the owner module exists and tests can target it directly, the old import path, wrapper, alias, or mock is deleted and covered by absence guards.
_Avoid_: preserving retired modules for historical users, keeping old names as convenience namespaces, or testing behavior through compatibility wrappers

**Terminus** — the whole Subcortex-centered architecture: the predictive spiking substrate plus the service/runtime surfaces that keep it observable, gated, and auditable.

**Living Brain** — an evidence-gated target state where the runtime continuously senses, learns, thinks, replays, acts, sleeps, and reports liveness without bypassing safety boundaries.
_Avoid_: using "living brain" as an unconditional production claim

**CUDA-first Runtime** — the runtime posture that uses CUDA/GPU execution for tensor-heavy subcortical work when available while keeping ordinary unit tests deterministic on CPU. Checkpoints and caches may serialize archival tensors on CPU, but active restore must load toward the selected runtime device rather than preserving a CPU-first path.
_Avoid_: GPU-only correctness, hidden CPU fallback in benchmark claims

**Routing Index** — the subcortical retrieval path that maps queries to candidate assemblies/prototypes. CUDA evidence requires actual cache/backend device telemetry, not only configured device intent.

**ThoughtLoop** — deleted LLM cognition orchestrator. The name survives only in retirement documentation and negative tests; no module, constructor, skipped behavior suite, or compatibility namespace should remain.

**DriveSystem** — converts predictive error, surprise, fatigue, and novelty into cognitive pressure and thalamic context.

**ThalamicGate** — assembles budgeted language/readout context packets from memory, drives, and source evidence.

**Cognitive Signal** — the typed Subcortex control packet carrying prediction error, predictive confidence, neuromodulator mirrors, recent concepts, source, and sample time. It is owned by Subcortex/semantics code, not the Retired LLM Path.

**WorkingMemory** — chain-local global workspace. Active scratchpad with strength-based decay and broadcast compression.

**EpisodicMemory** — provenance-aware hippocampal memory with embedding-based retrieval, capacity-bounded eviction, and importance scoring.

**NarrativeSelf** — cross-session autobiographical continuity. Tracks interests, questions, and surprise over time.

**Predictive Columns** — SNN columns that predict their input. Prediction error drives surprise, learning, and curiosity.

**Neuron Dynamics** — executable spiking neuron state such as AdEx membrane voltage, adaptation, and spike timing. CUDA evidence requires live tensor device reports and checkpoint restore back onto the selected runtime device.

**Replay** — hippocampal-style replay of past experiences for consolidation. Strictly evidence-only in the current runtime: no training, memory mutation, fact promotion, action execution, or sleep side effects from replay artifacts.

**Encoder** — transforms raw input (text, audio, visual) into sparse spike patterns for the SNN. Includes RTFEncoder, SemanticEncoder, EventCameraEncoder, CochleagramEncoder. Tensor-backed encoder state follows the configured runtime device; parsing windows, string segmentation, and archival metadata remain CPU/control-plane work.

**Text Encoder** — the RTFEncoder or SemanticEncoder path for character/text input. CUDA evidence requires device reports for learned chunking codebooks, semantic bucket embeddings, adapter tensors, emitted feature vectors, and spike traces; it does not require moving Python string parsing to CUDA.

**Sensory Encoder** — a CUDA-first Encoder for real sensory streams whose episode metadata records the tensor device, encoder state, and last emitted spike tensor shape/device used to produce visual or audio spikes.

**Multimodal Stream Loader** — the source adapter that feeds aligned text, visual frames, and audio chunks into sensory encoding. Generated and file-backed tensor modalities follow the resolved runtime device; CPU remains valid only when explicitly selected or when CUDA is unavailable, not as a hidden default.

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
- The original `living_loop.py` compatibility shim is deleted. Active code imports directly from the owning Living Loop depth module instead of using an aggregator namespace.

**Service Manager** — the composition root that wires the runtime (ADR 0003, ADR 0004). It constructs deep modules, owns lifecycle cleanup, and exposes the Runtime Facade. It owns no business logic, ADR-owned runtime state, owner-module constants, or owner callback wrappers itself. It has no legacy inherited mixin stack, no manager-level catch-all attribute router, no manager-bound fallback path, no owner-forwarder helper module, no import-time dynamic delegate installer, no generic mixin delegate trampoline, no interaction-mixin delegate wrapper methods, and no module-level `*Mixin = ...` compatibility aliases; remaining manager methods are internal dependency callbacks, not the operator-facing runtime interface.

**Runtime Facade** — the operator-facing runtime interface introduced by ADR 0004. FastAPI routes and export runners call this facade instead of calling Service Manager runtime pass-through methods. It delegates to the owning deep modules and preserves the stable HTTP/runtime contract while the Service Manager stays a composition root.

**Runtime Sources** — the owner of text/sensory stream construction, live-remote wrapping, runtime cache paths, stream readiness reads, and stream shutdown. Brain Runtime, Runtime Prewarmer, Sensory Runtime, and tests must patch or call Runtime Sources directly instead of using Service Manager stream-builder wrappers.

**Interaction Store** — the Interaction Pipeline-owned record of recent query gaps and runtime episode traces. Persistence, feedback, replay evidence, and tests read or mutate this store through Interaction Pipeline, not through Service Manager convenience methods.

**Owner Callback** — a constructor-injected callback that points directly at the module that owns the behavior or state, such as Delayed Consequence Tracker, Source Focus Scorer, Runtime Evidence Reporter, or Autonomy Planner. It must not be implemented as a Service Manager wrapper when the owner is already available in the composition root.
_Avoid_: manager-private callback wrappers, owner behavior hidden behind `self._...` manager methods

**Operator Interaction Runtime** — the domain-named service module for operator acquisition and interaction callbacks that do not belong on the Service Manager. It supplies query/feed/respond collaborator functions to Interaction Pipeline and owns the public acquisition flow behind Runtime Facade.
_Avoid_: resurrecting `InteractionRuntimeMixin`, manager-private interaction wrappers, or compatibility imports as active runtime surfaces

**Action Executor** — the domain-named owner for digital action execution, action history, action feedback routing, action-assist reuse, and Subcortex Action Ledger summaries. Runtime Facade delegates action execution and history reads to this module; delayed-consequence and interaction callbacks wire to it directly. The old `action_runtime.py` mixin module, standalone action-assist mixin module, and manager-private action delegate wrappers are deleted.
_Avoid_: action behavior in `ActionRuntimeMixin`, standalone action-assist mixin modules, manager-private action wrappers, retired-loop mirroring, or manager-owned action history

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
- **Runtime Persistence** — checkpoint save/restore and trace persistence. Runtime State owns the brain event history. Uses an explicit dependency object instead of owner-forwarded manager fields.
- **Runtime Config** — input validation and normalization gate for all operator configs. Stateless.
- **Runtime Sources** — stream construction, cache I/O, serialization, window reconstruction.
- **Replay Controller** — advisory replay planning, operator-gated sampling, dataset bundling with decontamination and splitting. Replay sampling intentionally uses Runtime State's dirty-without-revision path so audit-only samples stay dirty without advancing `state_revision`.

**Autonomy Ladder** — levels 0–5 of measured autonomy: observe → propose → execute approved → recurring constrained → evaluated policy → bounded self-improvement.

**Replay Pipeline** — the staged evidence-to-learning pipeline: gate → approval → plan → isolated experiment → promotion gate. Each stage produces a hash-verified, schema-versioned artifact.

**Runtime Truth** — the liveness classification system: alive / degraded / dead / partial / failed, with evidence, safety flags, and recommended operator action.

**Runtime Evidence Report** — operator-facing status evidence that joins model CUDA scope, trainer-owned encoder device reports, memory-store placement, runtime truth, and source configuration. It is read-only and must not advance runtime state.
_Avoid_: using retired LLM-path snapshots as the evidence source of record

**Subcortex Spike Health** — read-only operational stability evidence from competitive-column activity, bounded recent spike windows, local spike fraction, stale routing counters, visible silence/saturation thresholds, and windowed over-correlation risk. It is evidence for Runtime Truth, not a standalone liveness verdict.
_Avoid_: treating endpoint uptime as neural health, hiding threshold heuristics, treating one scalar correlation as full manifold health

**Subcortex Sleep Pressure** — active consolidation pressure derived from Subcortex memory pressure and trainer sleep counters. Policy and replay may use it to request consolidation or sleep; they must not derive fatigue, sleep, or memory pressure from a retired runtime snapshot.
_Avoid_: retired runtime fatigue, Cortex sleep state, compatibility snapshots as consolidation input

**Subcortex Self-Repair Candidate** — an advisory repair hypothesis derived from Subcortex Spike Health, such as reviewing column revival, inhibitory balance, stale routing, or decorrelation/pruning. It is not an action; promotion requires replay, deep-sleep repair, or operator gates.
_Avoid_: automatic self-mutation from status reads, treating repair suggestions as executed growth/prune events

**Self-Repair Promotion Gate** — the explicit gate on a Subcortex Self-Repair Candidate surface. It can request more spike-window evidence, mark candidates ready for replay/deep-sleep review, or keep them monitor-only; it must keep action, fact promotion, and structural mutation disabled.
_Avoid_: implicit repair execution, hidden replay mutation, structural mutation without a gate

**Self-Repair Gate Artifact** — the operator-facing read-only artifact that exposes Subcortex Self-Repair Candidates and their Self-Repair Promotion Gate without inserting those candidates into replay execution plans.
_Avoid_: replay candidate IDs, suggested endpoints, or execution payloads inside repair-gate status

**Self-Repair Evaluation Artifact** — the operator-facing read-only plan for measuring whether a Self-Repair Gate Artifact is safe to test in isolated replay or deep sleep. It names required evidence and success criteria, but does not run repair, mutate state, or promote structure.
_Avoid_: treating evaluation readiness as repair approval, hiding rollback/device/runtime-truth evidence requirements

**Self-Repair Evaluation Runtime Truth Evidence** — the compact Runtime Truth visibility summary for the Self-Repair Evaluation Artifact. It exposes evaluation-gate status, ready/case counts, next gate, success-evidence terms, and non-execution invariants while omitting evaluation cases, endpoints, suggested inputs, replay execution, deep sleep execution, repair mutation, and structural mutation.
_Avoid_: using Runtime Truth as the self-repair evaluator, exposing repair case payloads inside status, or promoting revive/prune/grow work from status visibility

**Delayed Consequence** — long-horizon utility tracking that connects earlier actions to later outcomes across queries and runs.

**Source Bank** — a named, ordered collection of training data sources (corpus, HF dataset, remote search) used by the subcortex for learning.

**Sensory Stream** — multimodal (visual, audio) observation stream for grounding, separate from text corpus.

## Key Relationships

- Subcortex is the active cognition substrate; the former external LLM/ThoughtLoop path is retired from runtime liveness claims.
- Subcortex Action Ledger owns action evidence. Digital action execution may update runtime history, provider calibration, consequences, and Runtime Truth evidence, but it must not initialize the Retired LLM Path just to mirror action records.
- Subcortex Grounded Observations own source and sensory evidence. Focus selection comes from query gaps, autonomy plans, geometric curiosity, concept state, and source metadata; it must not depend on a retired ThoughtLoop exploration target.
- Grounding Diagnostics belong to the Subcortex Language Surface boundary. Retired compatibility code may consume them, but future SNN language/readout modules must be able to produce and inspect them without instantiating ThoughtLoop.
- Active Exploration State belongs to Subcortex control. ThalamicGate stores the canonical state object and may expose compatibility properties; future curiosity, source-focus, and SNN deliberation modules must be able to normalize and inspect exploration targets without instantiating ThoughtLoop.
- Runtime Truth must not own retired-path vocabulary. `retired_runtime_path`, `cortex_available`, `cortex_retired`, `cortex_enabled`, and retired evidence aliases are deleted from active status and report contracts.
- Living Loop and Runtime Truth consumers should read active Subcortex evidence only; retired LLM/ThoughtLoop paths are absence checks, not status channels.
- Living Loop payloads must not emit a `cortex` sibling snapshot, `cortex_loop_snapshot` capability, or `retired_runtime_path` snapshot; replay pressure reads sleep/fatigue state from Subcortex Sleep Pressure.
- Long-test reports should emphasize Subcortex progress, Runtime Truth, memory pressure, spike health, and CUDA/device evidence. Retired LLM path fields are not compatibility payloads.
- Subcortex Language Surface may describe, narrate, or decode Subcortex state, but it must not own memory, policy, liveness, or Runtime Truth.
- Native-decode Subcortex Language is a bridge, not a generator: it may speak only from decoded assembly text and selected evidence, with support metrics attached.
- Cognitive Signal Subcortex Language is a status decoder: it may express runtime pressure and focus, but the numeric signal remains authoritative.
- SNN-Native Language Readiness Gate keeps NeuronSpark/Nord-style work as implementation references only; the target is HECSN-owned language neurons and decoder machinery under CUDA/device, sparsity, grounding, replay/evaluation, and operator-control gates. Runtime Truth may include a compact readiness summary, while the full artifact remains the review surface.
- Subcortex Spike Readout Evidence is the first owned SNN-language readiness primitive: it turns Cognitive Signal pressure, concept focus, and CUDA/Subcortex device reports into readout slots and population-code bands without decoding or generating text.
- Subcortex Spike Readout Evidence must prefer observed tensor placement from `subcortex_tensor_devices` over configured CUDA intent. A configured `tensor_device` is only fallback evidence when no live Subcortex tensor device report exists.
- Spike Language Decoder Probe is the next owned SNN-language primitive after spike readout evidence. It may report sparse code occupancy, leaky recurrent temporal state, grounded slot support, and tensor placement, but it remains non-generative and cannot make `eligible_for_language_generation` true without a separately trained/evaluated SNN generator.
- Spike Language Neuron Adapter is the first owned language-neuron module boundary after the decoder probe. It may report PLIF-style membrane/spike dynamics and activation sparsity, but it remains non-generative and must stay behind the SNN-Native Language Readiness Gate until a controlled training/evaluation loop exists.
- SNN Language Adapter Evaluation Gate is the next review boundary after the neuron adapter. It can mark the adapter ready for isolated evaluation when sparse dynamics, grounding support, and device evidence are present, but it must keep language generation, training, action, fact promotion, and cognition-substrate promotion false.
- Heldout Language Adapter Evaluation is the concrete evidence mechanism behind that gate. It may mark heldout evidence ready for operator review, but the next gate is still training-loop design review, not generation.
- SNN Language Training Readiness Gate is the design-review bridge after heldout adapter evidence. It can mark a local SNN trainer design review ready, but it keeps training, language generation, action, fact promotion, and cognition-substrate promotion false until a separate HECSN-owned trainer exists and passes evaluation.
- SNN Language Trainer Dry Run is the first local trainer experiment after design readiness. It may perform ephemeral local Hebbian sequence updates over spike readout slots for evidence, but it must return only metrics and discard weights.
- SNN Language Trainer Isolated Evaluation is the gate after a dry run. It can mark evidence ready for operator review, but it keeps runtime training, trainer promotion, language generation, and cognition-substrate promotion false.
- SNN Language Sequence Prediction Probe is the first local next-code prediction surface for the language path. It predicts sparse population-code indices from grounded spike readout sequences, but it does not decode those indices into text or promote them as thoughts.
- Persistent SNN Language Transition Memory lets applied live plasticity influence later sparse-code prediction through checkpointed HECSN-owned transition weights. Prediction must report whether persistent memory influenced the result.
- SNN Language Transition Memory Homeostasis gives that memory bounded decay, outgoing-row normalization, weak-edge pruning, checkpoint rollback, and a maintenance ledger. It is an explicit mutation command, never a read-side effect.
- SNN Language Transition Memory Sleep Policy can recommend that maintenance from active Subcortex Sleep Pressure and replay evidence, but it cannot execute the recommendation automatically.
- SNN Language Transition Memory Regeneration Proposal is a read-only advisory artifact for replay-window-backed local sparse regrowth after mismatch remains high. It must carry a replay-window identity, evidence hash, canonical neuron indices, and bounded locality while keeping topology mutation false.
- Replay-Backed Regeneration Permit is the Replay Controller-owned provenance artifact that authorizes one current-revision regeneration review. It binds high mismatch evidence, plasticity-pressure evidence, a grounded replay window, normalized operator identity, operator confirmation, Runtime State revision, and the canonical bounded regeneration design into a durable content hash. Caller-authored replay IDs, hashes, or post-permit candidate edits are not permits.
- SNN Language Transition Memory Regeneration is the checkpoint-backed operator-confirmed command that may add sparse transition edges after revalidation. It enforces fixed HECSN ceilings for canonical indices, event edge count, row fan-out, outgoing row mass, and global sparse topology size; duplicate-only proposals are blocked without advancing Runtime State.
- Applied regenerated synapses retain replay permit/artifact provenance and local sparse edge provenance, including rollout source/target step indices and active-index hashes when regeneration came from rollout consolidation.
- Replay Controller-issued permits narrow caller-authored structural-write authority to one server-held, revision-bound, design-bound artifact. Automatic structural adaptation remains blocked until permit issuance itself is driven by evaluated replay policy rather than an operator-confirmed command.
- Replay-Backed Regeneration Permit requires a server-held, internal-ledger-backed SNN Transition-Memory Replay Artifact whose SNN Replay Artifact Recording Review Ticket and Evaluated SNN Transition-Memory Replay Context are still current and content-verified. General replay audit-sampling history, raw caller-window artifacts, caller-carried mismatch or pressure reports, SNN Replay Consolidation Priority Queue rank, and SNN Replay Artifact Recording Policy Proposal recommendation must not be treated as equivalent evidence. Evaluated proposals now derive grounded replay windows from the internal Readout Evidence Ledger; policy proposals may choose review targets, but artifact recording remains operator-confirmed through a server-held review ticket.
- SNN language structural writes require an exact pre-mutation checkpoint round trip: the serialized transition-memory state and Runtime State revision must match the in-memory snapshot before live update, maintenance, or regeneration can proceed.
- Durable Structural Plasticity Transaction is the commit boundary for SNN language live update, maintenance, and regeneration. It requires a distinct pre-mutation rollback checkpoint, an atomically replaced post-mutation committed checkpoint, exact state/revision verification, and fail-closed in-memory recovery when commit verification fails.
- Runtime State revision continuity is checkpoint state. Service startup must hydrate the persisted clean revision exactly; explicit operator restore may advance revision to invalidate stale commands and permits.
- Published Current Checkpoint is the Runtime Persistence-owned atomic pointer to the last verified committed brain image. It references an immutable managed checkpoint object with size, SHA-256, and revision evidence; structural mutation may publish it only after committed state/revision verification. Startup validates the current descriptor, falls back to the previous committed descriptor when needed, and resolves the selected object before loading the trainer.

- Root Capture Refresh is the Service Manager-owned rebind step after a committed trainer, encoder, metadata, checkpoint-path, or action-root swap. Runtime Persistence invokes it after publication and explicit restore; Runtime Control invokes it after a preset rebuild. It refreshes concrete captures in active collaborators before runtime work resumes, so Terminus cannot split across different brain images.

- Operator Restore Commit is an explicit restore promoted into a new durable brain image. It imports the selected checkpoint, advances Runtime State beyond both the live and imported revisions, stages a fresh checkpoint, and publishes that staged image as the current managed object. Publication failure must recover the previous live image and leave the prior Published Current Checkpoint untouched.
- SNN Language Sequence Mismatch Probe is the first prediction-error surface for the language path. It can report how predicted spike-code indices differ from observed next spike-code indices, but it cannot apply learning or promote the mismatch as a fact/action.
- SNN Language Plasticity Pressure Gate turns prediction error into reviewable local learning pressure. It can identify observed-only and predicted-only sparse code targets for a future isolated plasticity trial, but it cannot apply plasticity by itself.
- SNN Language Plasticity Trial simulates the candidate local update from pressure evidence and reports expected pressure reduction. It must discard all update state and keep runtime training disabled.
- SNN Language Plasticity Replay Evaluation checks whether trial evidence is ready for operator replay review. It does not execute replay or apply the update.
- SNN Language Plasticity Replay Experiment rehearses sparse replay sequences in isolation. It can report replay coverage and simulated pressure stability, but it cannot persist weights, mutate runtime state, or promote language/facts.
- SNN Language Plasticity Application Design bounds a possible future live update. It can specify local update constraints and device evidence, but it remains read-only and cannot apply learning.
- SNN Language Plasticity Shadow Delta measures a proposed local update from sparse replay evidence. It is stronger than a hand-authored delta fixture but still cannot apply weights.
- SNN Language Plasticity Shadow Application verifies a proposed bounded update against design constraints and device evidence. It can report whether the shadow delta is locally stable, but it still cannot apply weights or mutate runtime state.
- SNN Language Plasticity Live Application Readiness checks whether shadow evidence has checkpoint/restore rollback coverage and explicit operator approval. It still keeps live application false until a separate application executor exists and is approved.
- SNN Language Plasticity Live Application Preflight checks the concrete mutable target and checkpoint transaction while staying read-only.
- SNN Language Plasticity Live Application Executor is the first command boundary that may mutate HECSN-owned sparse language transition weights after revalidation. It increments Runtime State and stays separate from Status Read Model.
- SNN Language Plasticity Path Evidence is the Runtime Truth summary for the chain. It exposes progress and invariants, but it is not a substitute for detailed gate artifacts or live application approval.
- Long-test reports use Subcortex Readout vocabulary. They must not publish ThoughtLoop-era `thoughts`, `thought_lifecycle`, `narrative_self`, `global_workspace`, or dream-verification report fields as active liveness evidence.
- Cognitive Signal is the canonical runtime signal surface. Its state primitive must be importable without `ThoughtLoop`; `cortex_signal` aliases are deleted and must not be reintroduced.
- Subcortex Deliberation candidates are advisory control candidates until replay, policy, or operator evidence promotes them; they must not be stored as facts, treated as generated thoughts, or queued as LLM prompts.
- Deliberation Feedback belongs to Subcortex control. `emit_deliberation_feedback` is the active API; the old `emit_cortex_feedback` path is removed.
- Deliberation Text Merge belongs to the Subcortex Language Surface boundary. Active code and tests should call the merge helper directly instead of using `ThoughtLoop` as a namespace.
- Brain Runtime Metrics can summarize language-facing runtime output and grounding quality, but Runtime Truth and Subcortex Spike Health remain the authoritative liveness evidence.
- Living Loop status is the primary operational sidecar for Subcortex Deliberation candidates. Policy Actuator may display the same candidates as non-executable context, but policy status must not execute them or promote them beyond advisory evidence.
- Policy Actuator reads sleep/fatigue pressure from Subcortex Sleep Pressure, not from `retired_runtime_path` or any Cortex snapshot.
- Replay planning reads memory/consolidation pressure from Subcortex Sleep Pressure, not from a retired runtime snapshot.
- Every Subcortex Deliberation candidate must carry a promotion gate. The gate may mark it ready for replay review or blocked by missing grounding, but it must keep action execution and fact promotion false until a separate replay/policy/operator path explicitly promotes it.
- Developmental Plasticity is the clean path for self-growth and pruning: runtime changes must be traceable, bounded, reversible, and evaluated before promotion.
- Local Plasticity Evidence is the first synapse-level support signal for Developmental Plasticity: local STDP traces, synaptic scaling, inhibitory balance, spike-health risk, synaptic validation, and device reports can make a case ready for isolated evaluation, but never directly mutate synapses or topology.
- Structural Mutation Ledger is required when topology changes at runtime; growth/pruning must leave countable evidence before it can support Living Brain claims.
- Structural Plasticity Gate Artifact is the read-only promotion surface for Developmental Plasticity: it can expose concept growth pressure, hypercube/binding mutation ledger state, and CUDA/device evidence, but structural mutation remains behind isolated evaluation and operator gates. Runtime Truth may include a compact gate summary, while the full artifact remains the review surface.
- Isolated Structural Plasticity Evaluation is the evidence bridge after the gate artifact: it can compare bounded pre/post topology snapshots and mark them ready for operator review, but the next gate remains an operator-approved structural mutation design, not automatic self-growth.
- Concept growth pressure and binding mutation ledgers are not enough by themselves. A structural case can become ready for isolated evaluation only when paired with observed binding or local-plasticity tensor device evidence.
- Local Plasticity Evidence can support growth/prune review only when eligibility traces, homeostatic state, device placement, and non-failed synaptic validation are all present. Failed synaptic validation is repair pressure, not promotion readiness.
- Subcortex Self-Repair Candidates require a sufficient spike/correlation evidence window before replay or deep-sleep review can be marked ready. A risk label without that window stays evidence collection.
- Cognitive Signal is the telemetry/control contract that lets Subcortex update runtime pressure, concept alignment, and future Subcortex Deliberation modules.
- Retired ThoughtLoop code is deleted and must not block Runtime Truth, CUDA evidence, or long-run liveness. Cognitive Signal state, Active Exploration State, Grounding Diagnostics, Brain Runtime Metrics, and deliberation text merge belong to Subcortex-owned modules.
- Retired Runtime Path State Holder is deleted. Active modules must not build, lazily initialize, start, store, snapshot, or report a ThoughtLoop-derived runtime path.
- The top-level `hecsn.cortex` package is deleted. Runtime, mock, memory, drive, prompt, narrative, and language primitives must not be reintroduced under that namespace.
- External LLM Cortex adapters are deleted. `NIMCortex`, `MultiCortex`, and Cortex environment factories are not valid runtime, testing, or extension points.
- External embedding adapters are deleted from Episodic Memory. Local sparse text encoders may remain as transitional indexing machinery, but remote API keys, NIM request accounting, and external embedding clients are not valid memory substrates.
- Remote API rate limiting is deleted with the external adapters. Runtime throttling should be reintroduced only for concrete maintained sources, not as Cortex/NIM budgeting.
- Mock Cortex is deleted. Tests may assert retirement boundaries, but `MockCortex` must not exist as a production-source class or stand in for language, thought, memory, sleep, or reasoning.
- ThoughtLoop's runnable body and module are deleted. There must be no hidden generation, sleep, or background-loop branches.
- Language Result is the Subcortex/semantics-owned result packet for language/readout quality metrics. Active code must import `LanguageResult` directly and must not use `ThoughtResult` aliases.
- Language Packet is the Subcortex/semantics-owned context contract for language/readout slots, mode, memory items, and depth. Active code must import `ContextPacket`, `MemoryItem`, `ReadoutMode`, and `DeliberationDepth` from `hecsn.semantics.language_packet`; `hecsn.cortex.core` is deleted.
- Cortex prompt templates are deleted. Language/readout steering should be semantics/Subcortex-owned and grounded by packet evidence, not preserved as Cortex-owned static LLM prompts.
- The `hecsn.cortex` package is deleted. Cortex-owned drives, episodic memory, working memory, narrative self, prompt, core, and ThoughtLoop modules must not be active extension points; reusable concepts belong under semantics, service runtime, or Subcortex modules.
- Gap Planner and Curiosity Controller feed Source Bank selection for autonomous acquisition
- Replay Pipeline feeds adapter experiments that never touch production runtime
- Service Manager wires the Runtime Facade and deep modules. Living Loop evidence is produced by Subcortex runtime state, replay, grounding, and policy surfaces; it must not require ThoughtLoop.
- CUDA-first Runtime applies to tensor-heavy Subcortex modules such as routing, predictive columns, neuron dynamics, binding, plasticity, cross-modal grounding, text encoders, and sensory encoders. The Retired LLM Path is not a CUDA-first claim or architectural requirement.
- Runtime Evidence Report is the bridge from internal CUDA-first claims to operator-visible status; it must include trainer-owned Encoder evidence as well as model-owned Subcortex evidence.
- Sensory Encoder device reports must include both persistent encoder state devices and the last emitted spike tensor device/shape; a configured device alone is not CUDA evidence.
- Multimodal Stream Loader tensors must resolve to the runtime device before encoding. Directory-loaded `.pt` visual/audio tensors and synthetic sensory tensors must not silently enter the Subcortex path through a CPU-only default.
- Checkpoint restore must select the active runtime device for `torch.load`; CPU serialization is allowed for archival payloads, not as the live Subcortex placement rule.
- Runtime Evidence Report and replay/export planning must read active Subcortex, Runtime Truth, spike-health, memory, and sleep-pressure evidence directly; they must not read `retired_runtime_path` as a source of record.
- Brain runtime snapshots must not expose `retired_runtime_path`, publish an active `cortex` sibling payload, or require `cortex_snapshot()` helpers.
- Subcortex Spike Health is the first operational-stability slice inside Runtime Evidence Report: it can flag silent, saturated, stale routing, or windowed over-correlation risk, while full operational-manifold health remains a future benchmark-level claim.
- Subcortex Self-Repair Candidates turn Spike Health into reviewable repair pressure for Living Loop and Policy Actuator sidecars. The Self-Repair Promotion Gate must keep action execution, fact promotion, and structural mutation false until a separate replay/deep-sleep/operator path approves the repair.
- Self-Repair Gate Artifact is exposed separately from Replay Plan so repair pressure can be reviewed without becoming a replay execution candidate or mutating runtime state. Runtime Truth may include a compact gate summary, but the full artifact remains the review surface.
- Self-Repair Evaluation Artifact is the next gate after review readiness: it specifies isolated replay/deep-sleep evaluation evidence, rollback expectations, Runtime Truth delta, and CUDA/device evidence before any repair can be promoted.
- Path Retirement Gate now applies to Cortex: LLM-backed runtime paths are being removed from active architecture so focus returns to Subcortex, world-model, memory, and policy mechanisms.

## Flagged Ambiguities

- "SSN side" is not canonical in this project. Resolved: use **Subcortex** for the domain layer and **SNN** only when referring specifically to spiking neural network mechanics.
- "Living brain" must not erase the existing safety vocabulary. Resolved: use **Living Brain** only as an evidence-gated target state; use **Runtime Truth** for actual liveness classification.
- "language generation" is ambiguous in this project. Resolved: use **Subcortex Language Surface** for grounded expression of Subcortex state, and reserve **Subcortex Deliberation** for cognition mechanisms that can run without an LLM.
