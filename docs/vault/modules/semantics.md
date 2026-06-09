---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.semantics

        ## Responsibility

        Grounded language/readout contracts, cognitive signal surfaces, decoder probes, and concept evidence.

        ## Owns

        Bounded readout artifacts and support/grounding diagnostics.

        ## Should Not Own

        Free-form cognition, fact promotion, action authority, or external checkpoint loading.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/semantics](../../../src/marulho/semantics)
        - [tests](../../../tests)

        ## Related Concepts

        [Spike Readout](../concepts/spike-readout.md), [Language from Spikes](../concepts/language-from-spikes.md), [Thought Trajectory](../concepts/thought-trajectory.md), [Capability Claim](../concepts/capability-claim.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "semantics" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
