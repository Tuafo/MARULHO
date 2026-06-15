---
type: paper
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # Structural plasticity

        ## Claim

        Supports bounded growth/pruning with explicit topology, device, usefulness/cost evidence, and rollback evidence.

        ## MARULHO Relevance

        Use this as research pressure only when it supports a MARULHO-owned mechanism, evidence gate, benchmark, or rejection note.

        ## Implementation Implication

        Do not import external runtime code or checkpoints unless a future ADR explicitly accepts that dependency. Prefer local probes, heldout gates, and rollback-aware experiments.

        Structural plasticity belongs behind repeated-surprise, usefulness, homeostasis, budget, isolated-evaluation, and checkpoint gates. It should grow or prune candidates only when evidence shows existing columns or synapses are not enough, and it must not run topology mutation inside the always-on tick.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
        - [Next Throughput Goal Map](../maps/next-throughput-goal-map.md)
