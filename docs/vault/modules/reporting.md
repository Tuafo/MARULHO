---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.reporting

        ## Responsibility

        Reports, benchmark plots, validation summaries, and autonomy/readme report helpers.

        ## Owns

        Human-readable evidence summaries.

        ## Should Not Own

        Primary runtime truth or mutation decisions.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/reporting](../../../src/marulho/reporting)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Truth](../concepts/runtime-truth.md), [Capability Claim](../concepts/capability-claim.md), [CUDA Evidence](../concepts/cuda-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "reporting" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
