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

        `training.cuda_graph_route_transition.CudaGraphRouteTransition` implements the checkpoint-opt-in Persistent Text Tick Executor. Eligible text ticks use one fixed-address replay for projection, reconstruction, device neuromodulation, sparse route/vote, in-place transition, and post-transition competitive-surprise measurement. The runtime mirrors one bounded result packet to `SurpriseMonitor` and Runtime Truth, so competitive surprise no longer launches a second CUDA norm/readback. Sensory ticks, pointer changes, unsupported configurations, and capture failures retain the pre-mutation fallback.

        The executor is not the full trainer or the full living loop. Cross-modal sensory grounding, archival memory, source handling, replay, checkpointing, and service orchestration remain outside the graph. Competitive-surprise measurement is device-owned for eligible text ticks, while its bounded history remains CPU control state. See [Hot-path latency](../benchmarks/hot-path-latency.md) and [ADR 0006](../../adr/0006-persistent-text-tick-executor.md).

        ## Responsibility

        Bootstrap, developmental, autonomy, consolidation, query, checkpointing, and long-run runners.

        ## Owns

        Offline or explicitly invoked training/evaluation workflows. Trainer checkpointing persists model-owned predictive column state, including prediction failure streaks used by the Column Runtime growth gate. The live trainer owns delayed promotion points for candidate-scoped column metabolism and the checkpoint-opt-in `ColumnTransitionRuntime` lifecycle. That runtime performs compile-only Triton warmup for bounded candidate shapes, owns persistent work buffers, keeps single-winner selection and fallback evidence on CUDA, and fuses predictive vote plus candidate competition for the proven learned-chunk/no-extra-gain shape. Unsupported shapes reconstruct the retained vote before ordinary device selection. The runtime falls back only before mutation, fails closed after a mutating launch begins, and reports transition, selection, fused-vote execution, and fallback evidence through Runtime Truth. The trainer also keeps adaptive context state live every token while applying dense context-weight plasticity on the configured four-token cadence; explicit replay/offline calls keep their own update policy.

        The trainer reuses the transition's already-materialized CPU winner IDs when buffering prototype updates for HNSW. It must not issue a second winner-tensor CPU transfer before the existing bounded flush; revived-column paths still materialize the expanded ID set when needed.

        Trainer scalar telemetry refreshes on `trainer_telemetry_interval_tokens` instead of a hardcoded ten-token cadence. This does not skip cognitive updates; it only reduces host-visible metric reads between cached Runtime Truth values. The metrics report the interval and whether the current tick refreshed telemetry.

        The trainer also caches the term set for the currently cached episode text. This keeps raw-window archival text and learned-chunk segmentation unchanged while avoiding a repeated regex/set rebuild when deciding whether a new stream window needs an episode refresh.

        Text-only Cross-Modal Grounding now defaults to a 16-token idle probe interval. Accepted visual/audio evidence still wakes text updates immediately, while text-only ticks mostly record cached-idle trace decay. The interval is a trainer-owned metabolism policy and must stay visible in Runtime Truth; it is not evidence that grounding quality improved.

        Competitive-column homeostasis now has its own wake gate: after `candidate_homeostasis_start_tokens`, threshold and win-rate updates use the retrieved candidate set instead of all columns. This is separate from `dead_column_steps`; stale counters, spike health, and explicit deep-sleep/maintenance revival keep their prior structural-mutation boundary.

        Standalone compiled route/competition remains rejected, but the broader exact-cache route plus predictive-vote cluster is now checkpoint-opt-in as `predictive_route_vote_mode=fused_triton_text`. `ColumnTransitionRuntime` owns compile-only warmup, persistent score/candidate state, cache refresh counters, execution evidence, and sensory fallback. It runs only on text/idle ticks; visual/audio ticks retain ordinary tensor routing.

        `predictive_route_vote_mode=cuda_graph_text` widens that production boundary into a fixed-address text-tick island. It captures input normalization/projection, exact fresh reconstruction distance, fused route/vote, and the in-place transition after checkpoint restoration. Visual/audio ticks bypass graph pre-routing and retain ordinary routing. Runtime Truth reports capture latency, graph names, replay/bypass/failure counters, fixed-address status, device evidence, and routing-cache clean fast-path/rebuild-check counters. Clean routing-cache ticks reuse the transition runtime's existing tensor pointers; dirty routing caches still rebuild through retrieval before graph replay.

        The graph also reuses the captured bucket-consolidation tensor through a memory-store generation guard. Eligible warm-memory ticks compare `DualMemoryStore.bucket_consolidation_cache_generation` with the captured generation instead of calling `bucket_consolidation_tensor()` every tick. Generation or memory-warm-state changes deactivate the graph before replay; in-place cache adjustments keep the generation stable and preserve the pointer. Runtime Truth reports generation fast-path and mismatch counters.

        The graph now applies the same ownership pattern to retrieval's torch routing cache. It stores the routing-cache generation captured from `retrieval.hnsw_index`, skips dirty-bit/pointer validation while that generation is unchanged, and falls back through `routing_tensor_cache()` plus fixed-pointer validation when retrieval reports a new generation. Runtime Truth reports route-cache generation fast-path, mismatch, rebuild-check, and clean-cache counters.

        Eligible graph ticks also keep that exact device routing cache coherent inside the in-place Triton transition. The transition writes the normalized next winner prototype into the prevalidated cache row, and the trainer skips the duplicate per-token HNSW winner/vector enqueue. The CPU routing-index store is a stale-capable slow mirror synchronized from live prototypes before retained index mutation. Runtime Truth exposes device updates, skipped buffering, host-mirror synchronization, and mirror freshness.

        After a successful persistent graph replay, candidate routing for the same token reuses the graph-prepared candidate buffer instead of immediately repeating routing-cache and graph-eligibility checks. The reuse is scoped by token count, increments `route_vote_prepared_graph_reuse_count`, and does not cross sensory/bootstrap/fallback boundaries.

        Trainer-stage profiling now splits persistent graph preparation into parameter staging, recent-row fill, input staging, and replay sub-buckets. These buckets are evaluation evidence only; they do not run when profiling is disabled. The current evidence shows parameter/control staging is larger than the actual input-buffer copy, so the next production-velocity boundary is device-owned modulator/control state inside a broader persistent executor.

        The graph now owns the previous-routing flag after capture. Python still stages the competitive modulator before replay, but it no longer copies the already graph-persisted `has_previous_routing_key` flag from host every token. Runtime Truth reports `previous_flag_device_owned_count`. This is a small host-control cleanup with parity evidence; it is not a broad throughput promotion.

        The graph also owns the competitive learning-rate counter for graph-backed text ticks. It computes `lr_initial / (1 + lr_decay * update_count)` inside the captured replay and increments the device update-count scalar after each replay. If a sensory/bootstrap/fallback tick advances Python `update_count` outside the graph, the next graph preparation resynchronizes the device scalar before replay. Runtime Truth reports `learning_rate_device_owned_count` and `learning_rate_host_resync_count`.

        The remaining host-staged competitive modulator now uses a `SurpriseMonitor.modulator_revision` cache. The graph copies the modulator scalar only when CPU-visible surprise state changes through error records, CPU neuromodulator updates, or graph host-truth mirror updates; intervening graph ticks reuse the already-staged device scalar. Runtime Truth reports `modulator_stage_copy_count` and `modulator_stage_skip_count`.

        The promoted graph specialization has live service evidence on an opt-in checkpoint: one 24-token source tick executed the graph-backed CUDA path 24 times with zero failures. Fresh-process hot-window evidence improved mean throughput from `176.24` to `264.46 ticks/sec`, but the source tick still took about `1.24 s`. The next training-owned optimization boundary is the remaining host orchestration and per-token stages outside the graph, without moving algorithms into service.

        Background semantic observation now uses the Sampled Batched Concept Observation boundary. `service.brain_runtime.BrainRuntime` schedules first/eighth/final samples, `service.operator_interaction.OperatorInteractionRuntime` adapts those samples into ConceptStore observations, and `semantics.concepts.ConceptStore` owns concept assignment plus structural maintenance. Structural growth/pruning maintenance runs once at the source-window boundary; service does not own concept algorithms.

        Runtime-source cache persistence is not trainer-owned cognition. The 2026-06-13 cache-material skip leaves deterministic source cache ownership in `service.runtime_sources.RuntimeSources` and removes identical cache rewrites from tick preparation before the trainer runs.

        Slow replay-memory archival now has its own trainer cadence. Every token still runs the promoted column transition, context, binding, cross-modal, surprise, and routing-index buffer policies, but expensive `DualMemoryStore.update()` admission and stream-text episode reconstruction run only on the first token, every `slow_memory_archive_interval_tokens` token, or high-surprise override tokens. Runtime Truth exposes archive count, skip count, interval, and last archive reason through `memory_hot_path`.

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
