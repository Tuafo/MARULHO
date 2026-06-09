---
type: map
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # MARULHO System Map

        A linked overview of runtime evidence, Subcortex, service gates, and operator-facing surfaces.


            ```mermaid
            flowchart LR
                Sources[Grounded sources] --> Data[data encoders]
                Data --> Subcortex[Subcortex runtime]
                Subcortex --> Semantics[semantics/readout]
                Subcortex --> Replay[replay and plasticity gates]
                Replay --> Truth[Runtime Truth]
                Semantics --> Truth
                Truth --> Operator[operator surfaces]
                Replay --> Checkpoints[checkpoint-backed executors]
                Checkpoints --> Subcortex
            ```


        ## Links

        - [Runtime Truth](../concepts/runtime-truth.md)
        - [Subcortex](../concepts/subcortex.md)
        - [Code organization](code-organization-map.md)
        - [Generated graph summary](../generated/graph-summary.md)
