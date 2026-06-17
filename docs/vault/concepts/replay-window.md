---
type: concept
status: active
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
---

# Replay Window

## Definition

A bounded, measured set of memory entries selected for review or isolated sleep
replay before any consolidation or plasticity authority.

Current runtime surfaces: `bounded_replay_window_selection.v1` and
`bounded_replay_window_recall.v1`.

## Rules

- Replay-window selection must run in explicit slow-path windows, not the live
  tick.
- A column-anchored replay window must score only memory entries attached to
  the supplied bucket ids through the bucket index.
- A global slow-path fallback must report `global_slow_path_score_scan`; it must
  not hide a full-memory scorer.
- Zero-pressure replay is retired. No replay updates apply when the best score
  is zero.
- Replay-window recall is non-mutating: it can compare routing keys and stored
  input patterns inside the selected window, but it cannot apply plasticity or
  run from the live tick.
- Archival metadata stays CPU-resident; active replay tensors move to the model
  device only when replay is actually applied.

## Evidence

`reports/bounded_replay_window_20260617/synthetic-selection.json` proves the
selection/retirement guardrail and passes the bounded stored input-pattern
recall gate for positive-pressure windows (`5.960464477539063e-08` mean input
distance, threshold `0.01`). It still does not pass prototype reconstruction
quality. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json`
keeps the 65536-column live tick in band.

## Relationships

- [Subcortex](subcortex.md)
- [Runtime Truth](runtime-truth.md)
- [Runtime Evidence](runtime-evidence.md)
- [Replay Cost](../benchmarks/replay-cost.md)

## Source Links

- [CONTEXT.md](../../../CONTEXT.md)
- [README.md](../../../README.md)
- [Research notes](../../research-living-brain.md)

## Ambiguity

Keep claims evidence-gated. Do not widen this term into a generic programming or
biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
