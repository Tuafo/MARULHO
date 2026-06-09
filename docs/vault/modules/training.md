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

        Bootstrap, developmental, autonomy, consolidation, query, and long-run runners.

        ## Owns

        Offline or explicitly invoked training/evaluation workflows.

        ## Should Not Own

        Live runtime mutation authority without service gates.

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
