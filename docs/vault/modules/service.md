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

        ## Key Files

        - [src/marulho/service](../../../src/marulho/service)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Truth](../concepts/runtime-truth.md), [Runtime Evidence](../concepts/runtime-evidence.md), [Replay Window](../concepts/replay-window.md), [Retired Path](../concepts/retired-path.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "service" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
