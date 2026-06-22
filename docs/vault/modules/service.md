---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.service

        ## Responsibility

        FastAPI service, runtime composition, status projections, Runtime Truth, replay ledgers, persistence, and gated executors.

        ## Owns

        Operator-facing runtime boundaries and evidence/mutation orchestration.

        ## Should Not Own

        Low-level neural math that belongs in core or semantics.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        Status and Runtime Truth reads must stay read-only with respect to delayed-consequence maintenance. Cooling, compaction, split, and remerge work belongs in explicit runtime maintenance windows such as ticks, query consequence application, or operator-run maintenance paths; it must not run as a hidden side effect of `status()`.

        ## Language Plasticity Snapshot Boundary

        `snn_language_plasticity_runtime_state.v1` is a read-only service evidence projection. It may expose canonical aliases for legacy fields, but it must not train, replay, resize, prune, or write checkpoints while building status.

        The service exposes canonical SNN language surface, memory, consolidation, structural-plasticity, readout-capacity-mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning routes. They are orchestration over the existing ledger/executor paths, not new service-owned algorithms. Thought-era public route vocabulary is no longer part of this active boundary.

        Public SNN language/readout route payloads should use readout vocabulary. The readout-surface, readout-memory, readout-consolidation, readout-structural-plasticity, readout-capacity-mutation, and readout-newborn developmental chains are canonical end-to-end through service, facade, ledger, executor, schema, API routes, and checkpoint save state. Checkpoint load/save keeps only canonical readout-ledger fields and drops noncanonical readout-ledger state instead of maintaining old field aliases. The API mapper no longer translates readout-newborn payloads back into thought-era internals.

        `/checkpoints` and `/traces` are UI metadata reads, not runtime work. They should use cached summaries when the runtime lock is busy so SSE telemetry, background runtime activity, checkpoint saves, or replay/tool windows cannot make the control room appear dead while `/status` is healthy. Stale metadata is preferable to blocking the operator surface; checkpoint writes/restores and trace persistence remain explicit slow paths.

        Runtime Scope evidence is also a status sidecar rather than live cognition. `StatusReadModel` may reuse a deep-copied Runtime Scope projection for at most 500 ms across status and terminus polling, while exposing cache age and source/current token counts. Explicit fresh reads bypass the projection cache. The cache must never become model memory, a column-vote cache, mutation state, or CUDA speedup evidence.

        Terminus background tick execution evidence belongs to `RuntimeControl`. The controller may expose a read-only heartbeat with active request count, idle state, tick phase, source, target token count, and elapsed time so `/terminus` and the UI can show live first-tick progress while `/status` keeps its cached fallback semantics. This heartbeat must not become a scheduler, learning rule, or mutation authority.

        RuntimeControl may record bounded wall-clock stage timings and schedule the existing remote refill worker after consuming a source chunk so provider I/O overlaps trainer execution. It must not synchronize CUDA for telemetry, implement semantic assignment, or move source algorithms into service.

        RuntimeControl owns the Continuous Execution Quantum as host scheduling policy. It may group a bounded number of still-sequential trainer calls under one execution-lock acquisition, omit artificial yields, and check stop requests between quanta. Runtime Truth must expose the active quantum and yield. RuntimeControl must not batch neural learning, reorder tokens, or absorb trainer algorithms.

        Runtime sources may cache bounded encoded windows for deterministic local/file text sources using a file fingerprint in the cache key. Restored file-source queues reduce first-tick source collection without changing Subcortex state, replay, memory admission, or trainer semantics. Stale source files naturally miss the cache because size and modification timestamp are part of the key.

        Structural Mutation Application orchestrates the explicit binding-hub topology transaction. Service binds operator reason, target, method, edge budget, revision, and checkpoint path into reviewed hashes; verifies the full binding snapshot before and after mutation; publishes only a verified committed checkpoint; and restores binding state plus Runtime State revision on no-op, over-budget, tampered, or failed commits. The topology algorithm remains in `core`.

        `/terminus/subcortical-structural-plasticity/binding-growth-trial` is an explicit read-only review request. `StatusReadModel` reads server-owned repeated-failure candidates and delegates edge planning to core, then semantics binds the result to revision and promotion evidence. Service does not select edges or execute the trial.

        ## Key Files

        - [src/marulho/service](../../../src/marulho/service)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Truth](../concepts/runtime-truth.md), [Runtime Evidence](../concepts/runtime-evidence.md), [Replay Window](../concepts/replay-window.md), [Retired Path](../concepts/retired-path.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "service" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
