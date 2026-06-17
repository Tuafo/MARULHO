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
- `bounded_replay_window_recall.v1` is emitted from `DualMemoryStore` as a
  CPU slow-path, non-mutating associative recall report over selected replay
  windows.
- Deep sleep can select from column-anchor bucket ids without scoring unrelated
  memory entries.
- Zero-pressure global replay is retired with
  `fallback_reason=no_positive_global_scores`.
- Deep sleep blocks unanchored global replay mutation with
  `unscoped_global_fallback_retired=true`, so no-anchor and zero-pressure bucket
  cases apply `0` replay updates.
- `reports/bounded_replay_window_20260617/synthetic-selection.json` passes the
  bounded stored input-pattern recall gate for positive-pressure windows
  (`5.960464477539063e-08` mean distance against threshold `0.01`), but does
  not pass the prototype reconstruction gate, so consolidation promotion remains
  quality-open.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Replay Cost](../benchmarks/replay-cost.md)
