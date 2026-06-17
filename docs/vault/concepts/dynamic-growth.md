---
type: concept
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # Dynamic Growth

        ## Definition

        Bounded structural addition driven by local evidence such as surprise, mismatch, replay failures, or concept pressure.

        Growth from one-shot surprise is blocked. A growth candidate must carry repeated-failure evidence through a checkpointed structural-review ticket, then prove usefulness and cost impact in isolated evaluation before any operator-reviewed transaction can apply topology changes. If the reviewed transaction is missing evidence, over budget, no-op, tampered, or fails checkpoint verification, the executor tombstones the candidate with provenance and rollback evidence instead of trusting or forgetting it.

        ## Relationships

        - [Subcortex](subcortex.md)
        - [Runtime Truth](runtime-truth.md)
        - [Runtime Evidence](runtime-evidence.md)

        ## Source Links

        - [CONTEXT.md](../../../CONTEXT.md)
        - [README.md](../../../README.md)
        - [Research notes](../../research-living-brain.md)

        ## Ambiguity

        Keep claims evidence-gated. Do not widen this term into a generic programming or biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
