---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.data

        ## Responsibility

        Source loaders and sparse encoders for text, semantic, audio, event-camera, and multimodal evidence.

        ## Owns

        Input normalization and emitted encoder/device evidence.

        Terminus Source Bank defaults are data-plane configuration, not service algorithms. The maintained text bank now starts with `open_textbooks` from `izumi-lab/open-text-books` because Hugging Face Dataset Viewer evidence showed direct `text` rows with worked educational prose, letting MARULHO replace raw Wikipedia without adding a parser or hot-path work.

        ## Should Not Own

        Runtime Truth verdicts or promotion of observations into facts/actions.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/data](../../../src/marulho/data)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Evidence](../concepts/runtime-evidence.md), [CUDA Evidence](../concepts/cuda-evidence.md), [Capability Claim](../concepts/capability-claim.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "data" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
