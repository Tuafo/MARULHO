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

        Offline or explicitly invoked training/evaluation workflows. Trainer checkpointing persists model-owned predictive column state, including prediction failure streaks used by the Column Runtime growth gate. The live trainer also owns delayed promotion points for candidate-scoped column metabolism: it keeps early learning all-column, then passes retrieved candidates into `CompetitiveColumnLayer.process()` and CPU predictive updates once stale/deep-sleep counters can exist. CUDA predictive updates currently remain dense and report a launch-bound fallback reason. The trainer also keeps adaptive context state live every token while applying dense context-weight plasticity on the configured four-token cadence; explicit replay/offline calls keep their own update policy.

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
