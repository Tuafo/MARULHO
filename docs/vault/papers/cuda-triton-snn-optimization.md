---
type: paper
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # CUDA/Triton SNN optimization

        ## Claim

        Supports sparse GPU execution only when observed runtime tensors prove placement.

        ## MARULHO Relevance

        Use this as research pressure only when it supports a MARULHO-owned mechanism, evidence gate, benchmark, or rejection note.

        ## Implementation Implication

        Do not import external runtime code or checkpoints unless a future ADR explicitly accepts that dependency. Prefer local probes, heldout gates, and rollback-aware experiments.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
