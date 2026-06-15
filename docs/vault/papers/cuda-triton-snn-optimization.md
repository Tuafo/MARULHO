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

        Supports sparse GPU execution only when observed runtime tensors prove placement and complete-runtime benchmarks prove the fused or persistent boundary is the real bottleneck.

        ## MARULHO Relevance

        Use this as research pressure only when it supports a MARULHO-owned mechanism, evidence gate, benchmark, or rejection note.

        ## Implementation Implication

        Do not import external runtime code or checkpoints unless a future ADR explicitly accepts that dependency. Prefer local probes, heldout gates, and rollback-aware experiments.

        For the current MARULHO speed path, local Triton kernels are no longer enough by themselves. Direct one-block route/vote fusion and partial-tail parent graphs were rejected by complete-runtime evidence. The next major implementation direction is a device-owned multi-tick or persistent sequence executor that reduces the actual per-token graph/kernel launch boundary while preserving sequential SNN state.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
        - [Next Throughput Goal Map](../maps/next-throughput-goal-map.md)
