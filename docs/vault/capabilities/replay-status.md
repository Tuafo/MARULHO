---
type: capability
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
---

# Replay Status

Replay status is advisory until selection, review, artifact, permit, quality,
and executor gates align.

## Evidence Rule

Do not claim replay as a live cognition improvement unless linked Runtime
Evidence or benchmark output shows both bounded selection and a passed
prediction/grounding/reconstruction target.

Current bounded selection evidence:

- `bounded_replay_window_selection.v1` is emitted from `DualMemoryStore`.
- Deep sleep can select from column-anchor bucket ids without scoring unrelated
  memory entries.
- Zero-pressure global replay is retired with
  `fallback_reason=no_positive_global_scores`.
- `reports/bounded_replay_window_20260617/synthetic-selection.json` did not pass
  the reconstruction gate, so replay remains quality-open.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Replay Cost](../benchmarks/replay-cost.md)
