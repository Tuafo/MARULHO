---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.retrieval

        ## Responsibility

        Vector/routing indexes such as TurboQuant, IVF, HNSW, and decoder support.

        ## Owns

        Lookup and routing experiments with explicit performance/device evidence.

        ## Should Not Own

        Claiming CUDA acceleration without observed telemetry.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/retrieval](../../../src/marulho/retrieval)
        - [tests](../../../tests)

        ## Related Concepts

        [Hot Path](../concepts/hot-path.md), [Slow Path](../concepts/slow-path.md), [CUDA Evidence](../concepts/cuda-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "retrieval" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
