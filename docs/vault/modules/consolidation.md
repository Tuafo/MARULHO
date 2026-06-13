---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.consolidation

        ## Responsibility

        Archival memory store and consolidation records.

        ## Owns

        CPU-resident archival evidence and explicit memory records.

        ## Should Not Own

        Device-local replay computation or live plasticity application.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        Reservoir admission is decided before optional input-pattern and routing-key tensors are copied from CUDA into the CPU archival ledger. Assembly EMA/drift observation, STC state advance, reservoir probability, admitted payloads, and replay semantics remain unchanged. Device evidence reports update, admission, rejection, copied-payload, and avoided-copy counters.

        Capture tags, strong-tag flags, and local PRP values use contiguous Python numeric arrays. NumPy obtains zero-copy views over those buffers for exact in-place decay, avoiding three list-to-array/array-to-list conversions on every state advance. Checkpoint snapshots still serialize ordinary lists, and restore rebuilds the numeric buffers. Runtime Truth reports the storage mode, zero-copy decay status, and exact STC scalar-state bytes.

        Awake-ripple tagging keeps the scalar loop for small ledgers and switches to zero-copy NumPy scans only at the measured large-ledger crossover. Runtime Truth reports scalar/vector scan counts and the last scan mode under memory hot-path evidence.

        Bucket-consolidation tensor caches expose a monotonic `bucket_consolidation_cache_generation` for graph-safe pointer reuse. Explicit cache invalidation increments the generation; ordinary in-place cache adjustments preserve it. The Persistent Text Tick Executor uses this identity to fail closed before replay when the captured consolidation tensor is no longer valid, avoiding a per-tick cache lookup on graph-eligible warm-memory ticks.

        The archival boundary remains CPU-owned. A tested CUDA observation variant was slower (`75.69` versus `97.18 ticks/sec`), and bulk asynchronous staging was neutral in complete runs. Sampled replay tensors move to the model device only when replay computation consumes them.

        ## Key Files

        - [src/marulho/consolidation](../../../src/marulho/consolidation)
        - [tests](../../../tests)

        ## Related Concepts

        [Replay Window](../concepts/replay-window.md), [Runtime Evidence](../concepts/runtime-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "consolidation" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
