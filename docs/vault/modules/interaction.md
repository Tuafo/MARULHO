---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.interaction

        ## Responsibility

        Responder and operator-facing answer formation.

        ## Owns

        Grounded response shaping over evidence surfaces.

        ## Should Not Own

        Hidden thought loops or unsupported fluency claims.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/interaction](../../../src/marulho/interaction)
        - [tests](../../../tests)

        ## Related Concepts

        [Language from Spikes](../concepts/language-from-spikes.md), [Runtime Evidence](../concepts/runtime-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "interaction" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
