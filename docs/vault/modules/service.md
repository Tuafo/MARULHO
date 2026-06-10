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

        The service exposes canonical SNN language surface, memory, consolidation, structural-plasticity, capacity-mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning routes. They are orchestration over the existing ledger/executor paths, not new service-owned algorithms. Thought-era public route vocabulary is no longer part of this active boundary.

        Public SNN language/readout route payloads should use readout vocabulary. The surface and memory route boundary now translates older internal `autonomous_snn_language_thought_*` ledger keys to readout names for operators and UI clients, then translates canonical chained inputs back into the current ledger shape. This is a compatibility bridge for the current ledger shape, not a license to add new thought-era surfaces. Capacity, newborn-neuron, and pruning payloads still need the same cleanup.

        ## Key Files

        - [src/marulho/service](../../../src/marulho/service)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Truth](../concepts/runtime-truth.md), [Runtime Evidence](../concepts/runtime-evidence.md), [Replay Window](../concepts/replay-window.md), [Retired Path](../concepts/retired-path.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "service" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
