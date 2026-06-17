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

        Current implementation implication: the isolated structural evaluator must bind a training-owned structural-review ticket to an exact baseline hash, candidate reason, cost/usefulness and latency/RAM/VRAM impact, Runtime Truth summary, rollback artifact, and no-mutation proof. This follows reward-modulated STDP and homeostatic-plasticity work for local evidence, sparse GPU structural-plasticity work for bounded sparse edits, event-based delay learning for timing/cost-aware evidence, and self-growing/growth-stability work for proving newborn structures before trust.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
        - [Next Throughput Goal Map](../maps/next-throughput-goal-map.md)
        - [GPU sparse structural plasticity](https://arxiv.org/abs/2510.19764)
        - [Three-factor learning in SNNs](https://arxiv.org/abs/2504.05341)
        - [Event-based delay learning](https://arxiv.org/abs/2501.07331)
        - [Self-motivated growing neural network](https://arxiv.org/abs/2512.12713)
        - [Stability of growth](https://arxiv.org/abs/2605.15435)
