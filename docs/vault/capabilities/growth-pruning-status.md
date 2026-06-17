---
type: capability
status: current
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

            # Growth Pruning Status

            Growth/pruning status is readiness evidence until checkpoint-backed mutation is confirmed.

            ## Evidence Rule

            Do not claim this capability as live unless linked Runtime Evidence or benchmark output supports it.

            The active path is now: training-owned `ColumnStructuralReviewQueue` ticket -> isolated structural-plasticity evaluation with checkpointed candidate gate -> operator-confirmed mutation design -> candidate-bound checkpoint preflight -> checkpoint-backed structural mutation executor. Candidate readiness requires baseline hash, candidate reason, cost/usefulness metrics, latency/RAM/VRAM impact, Runtime Truth summary, rollback artifact, and no-mutation proof. The executor recomputes the candidate-bound preflight hash, commits only through verified checkpoints, and emits candidate provenance plus tombstone/rollback artifacts for blocked, rejected, or retired candidates. Service may project or evaluate supplied evidence, but it must not select columns or mutate topology.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
