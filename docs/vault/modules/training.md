---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.training

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

        `predictive_route_vote_mode=cuda_graph_text` widens that production boundary into a fixed-address text-tick island. It captures input normalization/projection, exact fresh reconstruction distance, fused route/vote, and the in-place transition after checkpoint restoration. Visual/audio ticks bypass graph pre-routing and retain ordinary routing. Runtime Truth reports capture latency, graph names, replay/bypass/failure counters, fixed-address status, and device evidence.

        The promoted graph specialization has live service evidence on an opt-in checkpoint: one 24-token source tick executed the graph-backed CUDA path 24 times with zero failures. Fresh-process hot-window evidence improved mean throughput from `176.24` to `264.46 ticks/sec`, but the source tick still took about `1.24 s`. The next training-owned optimization boundary is the remaining host orchestration and per-token stages outside the graph, without moving algorithms into service.

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
