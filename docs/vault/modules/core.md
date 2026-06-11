---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.core

        ## Responsibility

        Local SNN mechanisms: columns, context, binding, abstraction, topography, plasticity, surprise, and sparsity.

        ## Owns

        Tensor/state mechanisms and device-reportable substrate behavior.

        `core.column_runtime` now owns the report-only Column Runtime control plane: a bounded awake-column scheduler summary, cached vote evidence, disagreement, growth gate, and pruning/homeostasis evidence derived from existing competitive and predictive column tensors. It does not mutate topology or change execution scheduling yet.

        ## Should Not Own

        Operator HTTP surfaces, persistence policy, or language-facing claims.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/core](../../../src/marulho/core)
        - [tests](../../../tests)

        ## Related Concepts

        [Subcortex](../concepts/subcortex.md), [Metabolism](../concepts/metabolism.md), [Column Runtime](../concepts/column-runtime.md), [Plasticity Gate](../concepts/plasticity-gate.md), [Dynamic Growth](../concepts/dynamic-growth.md), [Pruning](../concepts/pruning.md), [CUDA Evidence](../concepts/cuda-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "core" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
