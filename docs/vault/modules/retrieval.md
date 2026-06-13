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

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path. Torch-backed routing owns `search_tensors()` for candidate ids and distances that stay on the routing device for live trainer competition and compiled CUDA kernels. Logically sharded torch indexes may use one exact merged cache to reduce launch count while retaining shard-owned updates; cache bytes, readiness, devices, and invalidation state are Runtime Scope evidence. Legacy list-returning `search()` remains a compatibility/control-plane surface and should not be used as the production-velocity path.

        `routing_tensor_cache()` exposes the current exact torch cache by reference for the checkpoint-opt-in fused text route/vote lifecycle. Retrieval remains responsible for invalidation and rebuild; training may refresh pointers but must not duplicate cache mutation policy.

        Same-shape, same-device torch-cache rebuilds copy into existing cache tensors instead of replacing them. This preserves exact rebuild semantics while keeping addresses stable for the checkpoint-opt-in CUDA Graph text-tick island. Shape or device changes may replace tensors and must disable the graph before mutation.

        ## Key Files

        - [src/marulho/retrieval](../../../src/marulho/retrieval)
        - [tests](../../../tests)

        ## Related Concepts

        [Hot Path](../concepts/hot-path.md), [Slow Path](../concepts/slow-path.md), [CUDA Evidence](../concepts/cuda-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "retrieval" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
