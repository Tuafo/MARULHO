---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.evaluation

        ## Responsibility

        Promotion gates, benchmarks, readiness checks, and validation harnesses.

        ## Owns

        Evidence standards for speed, readiness, CUDA placement, liveness, and promotion.
        Hot-window benchmark reports may now opt into measured-step trainer-stage profiling, and the persistent text-tick A/B runner summarizes reversed same-process stage deltas so route/index/graph-prep work can be judged by complete encoded ticks instead of noisy service endpoints.
        Continuous runtime stress reports include slow-path `velocity_environment.v1` snapshots so sustained CUDA throughput comparisons can distinguish architecture regressions from CPU/GPU contention.

        ## Should Not Own

        Runtime state mutation.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/evaluation](../../../src/marulho/evaluation)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Truth](../concepts/runtime-truth.md), [CUDA Evidence](../concepts/cuda-evidence.md), [Capability Claim](../concepts/capability-claim.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "evaluation" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
