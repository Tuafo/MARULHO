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

        ## Key Files

        - [src/marulho/consolidation](../../../src/marulho/consolidation)
        - [tests](../../../tests)

        ## Related Concepts

        [Replay Window](../concepts/replay-window.md), [Runtime Evidence](../concepts/runtime-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "consolidation" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
