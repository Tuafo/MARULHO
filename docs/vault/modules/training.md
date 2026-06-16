---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.training

        ## Persistent Text Tick Executor

        `training.cuda_graph_route_transition.CudaGraphRouteTransition` implements the default promoted Persistent Text Tick Executor for eligible CUDA text ticks. Eligible text ticks use one fixed-address replay for projection, fused route-score reconstruction, device neuromodulation, sparse route/vote, in-place transition, and post-transition competitive-surprise measurement. The runtime mirrors one bounded result packet to `SurpriseMonitor` and Runtime Truth, so competitive surprise no longer launches a second CUDA norm/readback. Sensory ticks, pointer changes, unsupported configurations, and capture failures retain the pre-mutation fallback.

        The executor is not the full trainer or the full living loop. Cross-modal sensory grounding, archival memory, source handling, replay, checkpointing, and service orchestration remain outside the graph. Competitive-surprise measurement is device-owned for eligible text ticks, while its bounded history remains CPU control state. See [Hot-path latency](../benchmarks/hot-path-latency.md) and [ADR 0006](../../adr/0006-persistent-text-tick-executor.md).

        ## Responsibility

        Bootstrap, developmental, autonomy, consolidation, query, checkpointing, and long-run runners.

        ## Owns

        Offline or explicitly invoked training/evaluation workflows. Trainer checkpointing persists model-owned predictive column state, including prediction failure streaks used by the Column Runtime growth gate. The live trainer owns delayed promotion points for candidate-scoped column metabolism and the promoted `ColumnTransitionRuntime` lifecycle. That runtime performs compile-only Triton warmup for bounded candidate shapes, owns persistent work buffers, keeps single-winner selection and fallback evidence on CUDA, and fuses predictive vote plus candidate competition for the proven learned-chunk/no-extra-gain shape. Unsupported shapes reconstruct the retained vote before ordinary device selection. The runtime falls back only before mutation, fails closed after a mutating launch begins, and reports transition, selection, fused-vote execution, and fallback evidence through Runtime Truth. The trainer also keeps adaptive context state live every token while applying dense context-weight plasticity on the configured four-token cadence; explicit replay/offline calls keep their own update policy.

        The trainer reuses the transition's already-materialized CPU winner IDs when buffering prototype updates for the routing index. It must not issue a second winner-tensor CPU transfer before the existing bounded flush; revived-column paths still materialize the expanded ID set when needed.

        Trainer scalar telemetry refreshes on `trainer_telemetry_interval_tokens` instead of a hardcoded ten-token cadence. This does not skip cognitive updates; it only reduces host-visible metric reads between cached Runtime Truth values. The metrics report the interval and whether the current tick refreshed telemetry.

        The trainer also caches the term set for the currently cached episode text. This keeps raw-window archival text and learned-chunk segmentation unchanged while avoiding a repeated regex/set rebuild when deciding whether a new stream window needs an episode refresh.

        Text-only Cross-Modal Grounding now defaults to a 16-token idle probe interval. Accepted visual/audio evidence still wakes text updates immediately, while text-only ticks mostly record cached-idle trace decay. The interval is a trainer-owned metabolism policy and must stay visible in Runtime Truth; it is not evidence that grounding quality improved.

        Competitive-column homeostasis now has its own wake gate: after `candidate_homeostasis_start_tokens`, threshold and win-rate updates use the retrieved candidate set instead of all columns. This is separate from `dead_column_steps`; stale counters, spike health, and explicit deep-sleep/maintenance revival keep their prior structural-mutation boundary.

        Standalone compiled route/competition remains rejected, but the broader exact-cache route plus predictive-vote cluster is now the default eligible CUDA text path through `predictive_route_vote_mode=cuda_graph_text`, with `fused_triton_text` retained as an explicit benchmark/fallback mode. `ColumnTransitionRuntime` owns compile-only warmup, persistent score/candidate state, cache refresh counters, execution evidence, and sensory fallback. It runs only on text/idle ticks; visual/audio ticks retain ordinary tensor routing.

        `predictive_route_vote_mode=cuda_graph_text` widens that production boundary into a fixed-address text-tick island. It captures input normalization/projection, fused route-score reconstruction, fused route/vote, and the in-place transition after checkpoint restoration. Visual/audio ticks bypass graph pre-routing and retain ordinary routing. Runtime Truth reports capture latency, graph names, replay/bypass/failure counters, fixed-address status, device evidence, reconstruction source/update counters, route/vote kernel variant, and routing-cache clean fast-path/rebuild-check counters. Clean routing-cache ticks reuse the transition runtime's existing tensor pointers; dirty routing caches still rebuild through retrieval before graph replay.

        The maintained route/vote kernel variant is `two_stage_route_vote`: a Triton route-score kernel followed by the retained select/vote kernel. A direct one-block route/vote fusion for the live `1024 x 64` checkpoint passed parity but regressed complete runtime, so it was deleted instead of retained as dormant hot-path code. Runtime Truth keeps the variant field so future speed work can prove whether a real executor is running the retained route/vote family or a newly promoted replacement.

        The graph also reuses the captured bucket-consolidation tensor through a memory-store generation guard. Eligible warm-memory ticks compare `DualMemoryStore.bucket_consolidation_cache_generation` with the captured generation instead of calling `bucket_consolidation_tensor()` every tick. Generation or memory-warm-state changes deactivate the graph before replay; in-place cache adjustments keep the generation stable and preserve the pointer. Runtime Truth reports generation fast-path and mismatch counters.

        Post-transition graph bookkeeping honors the same ownership boundary: it does not reacquire the generation-validated consolidation tensor, and it reuses a persistent empty revival-id tensor instead of allocating one CUDA tensor per token. Runtime Truth reports `graph_consolidation_lookup_skip_count` and `graph_empty_revival_tensor_reuse_count`. Retained/sensory transitions and explicit revival maintenance remain unchanged.

        The graph now applies the same ownership pattern to retrieval's torch routing cache. It stores the routing-cache generation captured from `retrieval.hnsw_index`, skips dirty-bit/pointer validation while that generation is unchanged, and falls back through `routing_tensor_cache()` plus fixed-pointer validation when retrieval reports a new generation. Runtime Truth reports route-cache generation fast-path, mismatch, rebuild-check, and clean-cache counters.

        Eligible graph ticks also keep that exact device routing cache coherent inside the in-place Triton transition. The transition writes the normalized next winner prototype into the prevalidated cache row, and the trainer skips the duplicate per-token routing-index winner/vector enqueue. The CPU routing-index store is a stale-capable slow mirror synchronized from live prototypes before retained index mutation. Runtime Truth exposes device updates, skipped buffering, host-mirror synchronization, and mirror freshness.

        After a successful persistent graph replay, candidate routing for the same token reuses the graph-prepared candidate buffer instead of immediately repeating routing-cache and graph-eligibility checks. The reuse is scoped by token count, increments `route_vote_prepared_graph_reuse_count`, and does not cross sensory/bootstrap/fallback boundaries.

        Trainer-stage profiling now splits persistent graph preparation into parameter staging, recent-row fill, input staging, and replay sub-buckets. These buckets are evaluation evidence only; they do not run when profiling is disabled. The current evidence shows parameter/control staging is larger than the actual input-buffer copy, so the next production-velocity boundary is device-owned modulator/control state inside a broader persistent executor.

        The graph now owns the previous-routing flag after capture. Python still stages the competitive modulator before replay, but it no longer copies the already graph-persisted `has_previous_routing_key` flag from host every token. Runtime Truth reports `previous_flag_device_owned_count`. This is a small host-control cleanup with parity evidence; it is not a broad throughput promotion.

        The graph also owns the competitive learning-rate counter for graph-backed text ticks. It computes `lr_initial / (1 + lr_decay * update_count)` inside the captured replay and increments the device update-count scalar after each replay. If a sensory/bootstrap/fallback tick advances Python `update_count` outside the graph, the next graph preparation resynchronizes the device scalar before replay. Runtime Truth reports `learning_rate_device_owned_count` and `learning_rate_host_resync_count`.

        The remaining host-staged competitive modulator now uses a `SurpriseMonitor.modulator_revision` cache. The graph copies the modulator scalar only when CPU-visible surprise state changes through error records, CPU neuromodulator updates, or graph host-truth mirror updates; intervening graph ticks reuse the already-staged device scalar. Runtime Truth reports `modulator_stage_copy_count` and `modulator_stage_skip_count`.

        The graph input boundary is now the Persistent Quantum Input Ring.
        Brain Runtime offers each bounded sequential execution quantum without
        owning CUDA algorithms; `ColumnTransitionRuntime` stages the encoded
        tensors into a fixed 128-row CUDA ring and the captured graph advances
        the input slot and recent-spike-row cursor. Warm metric-free text
        sequences stage the longest safe segment up to the ring capacity after
        a non-mutating boundary preflight classifies each burst-sized slice as
        device-continuous. Host-truth, sleep, metrics, and other cognitive
        boundaries stop the staged segment instead of disabling the whole source
        tick. The eight-token burst executor then consumes pointer-checked
        slices from the staged window. Pointer-order validation preserves exact
        token order, while mismatch, sensory, and fallback boundaries skip or
        discard staged remainder and fall back before mutation. Runtime Truth
        exposes sequence-stage calls/tokens/skips plus graph stage, token reuse,
        fallback-copy, mismatch, discard, and device-owned cursor counters; only
        the real burst plan updates boundary counters.

        The production trainer now owns Boundary-Aware Text Burst execution.
        For exactly eight ordinary text ticks, it consumes an already staged
        wider quantum slice when available, otherwise stages the ring for that
        burst, replays the same one-tick graph eight times in order, and applies
        equivalent host bookkeeping in one bounded operation. On the promoted
        CUDA path, the promoted `conditional_while` sequence executor wraps the
        retained one-tick graph in a startup-warmed conditional-WHILE parent
        CUDA graph and launches that parent once per sixteen-token q16 sequence.
        The repeated-child native parent graph remains exact eight-token replay
        for fallback and explicit opt-out. Runtime Truth reports active,
        default, and allowed effective burst capacities plus separate
        repeated-child and sequence-loop capacities so a conditional q16
        promotion is not confused with the rejected native16 repeated-child
        prototype. Benchmark probes must keep native burst capacity aligned
        with the execution quantum; a warmed parent graph is not native
        coverage when q16 chunking forces `python_loop_partial_disabled`.
        ADR 0007 records conditional-WHILE q16 as the promoted lower-level
        sequence boundary and keeps the next direction below local graph
        composition in C++/CUDA, Triton, persistent-kernel, or hybrid ownership.
        Runtime Truth exposes whether native replay is configured, loaded,
        enabled, which backend ran, parent-graph count, launch attempts,
        successes, covered tokens, fallbacks, failures, and compile/build
        latency. It also exposes sequence-loop sequential-state parity and
        bounded-quality gate status fields so executor speed reports carry the
        validation boundary. The retained Python replay loop remains the
        explicit fallback before mutation. The in-place transition kernel writes
        the slim result
        packet and strong-event flag directly into the Device Strong-Event Ring,
        loading and copying
        assembly/routing rows only for threshold crossings; training drains
        those records at the host-truth boundary and archives all payloads on
        CPU. The same kernel also maintains a device-owned cumulative
        strong-event count, so ordinary no-strong drains skip CPU strong-flag
        scans while forced or natural strong captures still materialize exact
        flags and payload rows. Full-capacity cadence drains rely on the device
        slot's natural wrap instead of launching a redundant reset, with
        reset/skip counts and strong-count scan/skip counts exposed through
        Runtime Truth. Eligibility still fails closed at drift,
        telemetry, sleep, slow-memory cadence, cross-modal wake, host-truth,
        routing-mode, and metrics boundaries. The same classifier can preview those
        boundaries for quantum pre-staging without incrementing Runtime Truth
        counters; service only offers the encoded quantum and does not own burst
        algorithms. Runtime Truth exposes burst executions, burst tokens,
        failures, fallback-reason counts, strong events, ring ownership, and
        graph names.

        The promoted graph specialization has live service evidence on an opt-in checkpoint: one 24-token source tick executed the graph-backed CUDA path 24 times with zero failures. Fresh-process hot-window evidence improved mean throughput from `176.24` to `264.46 ticks/sec`, but the source tick still took about `1.24 s`. The next training-owned optimization boundary is the remaining host orchestration and per-token stages outside the graph, without moving algorithms into service.

        Background semantic observation now uses the Sampled Batched Concept Observation boundary. `service.brain_runtime.BrainRuntime` schedules first/eighth/final samples, `service.operator_interaction.OperatorInteractionRuntime` adapts those samples into ConceptStore observations, and `semantics.concepts.ConceptStore` owns concept assignment plus structural maintenance. Structural growth/pruning maintenance runs once at the source-window boundary; service does not own concept algorithms.

        Runtime-source cache persistence is not trainer-owned cognition. The 2026-06-13 cache-material skip leaves deterministic source cache ownership in `service.runtime_sources.RuntimeSources` and removes identical cache rewrites from tick preparation before the trainer runs.

        The 2026-06-14 Training-Owned Text Sequence gives `MarulhoTrainer` the complete ordered text-tick execution boundary while service retains source orchestration and semantic observation. Runtime Sources separately owns Prepared Source Generations, allowing consumption-only ticks to skip cache reconstruction in O(1). Ordinary service ticks now return Device-Burst Lightweight Metrics from the CUDA result packet; full per-token metrics are reserved for explicit evaluator evidence positions so source concept sampling cannot silently break burst ownership. See [[prepared-source-tick-executor]].

        The maintained service execution quantum is `16`. The promoted
        conditional-WHILE executor consumes that q16 quantum as one ordered
        native sequence loop, while the repeated-child native fallback/opt-out
        path still consumes exact ordered eight-token parent graphs. In both
        cases the persistent CUDA graph body, event ring, host-truth boundary,
        and SNN transition order remain unchanged. This removes the old q16
        fallback path where wider quanta bypassed burst execution and fell
        through to per-token `train_step`. The source-sequence input stage now
        spans multiple quanta when safe, but never crosses the same host-truth,
        sleep, metrics, or fallback boundaries that the real burst executor
        enforces.

        Slow replay-memory admission is no longer a fixed-cadence hot-path write. Every token still runs the promoted column transition, context, binding, cross-modal, surprise, and routing-index buffer policies, but expensive `DualMemoryStore.update()` admission and stream-text episode reconstruction run only on retained/fallback admission or high-surprise strong-capture events. Fixed cadence is counted as deferred maintenance by the cognitive boundary controller, not as a reason to break burst execution. Runtime Truth exposes deferred cadence, archive count, skip count, interval, and last archive reason through `memory_hot_path` and the boundary report.

        Drift maintenance is sync-free on burst ticks. The trainer refreshes drift without draining pending CUDA event evidence, uses winner-local drift only when the host winner mirror is already fresh, and reports global-drift refreshes when the mirror is stale. Drift-floor closure is CPU maintenance and no longer forces an event drain. See [[prepared-source-tick-executor]].

        The cognitive boundary classifier now uses range arithmetic for telemetry, drift, slow-memory cadence, sleep, and routing-index boundaries instead of scanning every token in Python for each proposed burst. Focused tests compare the range classifier against the previous loop semantics, and Runtime Truth exposes `classification_mode=range_arithmetic`.

        ## Should Not Own

        Service lifecycle, HTTP policy, or structural mutation authority that belongs behind explicit checkpoint/operator gates.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/training](../../../src/marulho/training)
        - [tests](../../../tests)

        ## Related Concepts

        [Replay Window](../concepts/replay-window.md), [Plasticity Gate](../concepts/plasticity-gate.md), [Metabolism](../concepts/metabolism.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "training" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
