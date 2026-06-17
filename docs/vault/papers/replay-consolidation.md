---
type: paper
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - https://arxiv.org/abs/2008.02217
  - https://pubmed.ncbi.nlm.nih.gov/7624455/
  - https://papers.neurips.cc/paper/8327-experience-replay-for-continual-learning
  - https://pubmed.ncbi.nlm.nih.gov/9020359/
  - https://arxiv.org/abs/1912.01100
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
---

# Replay/consolidation

## Claim

Replay and consolidation are slow-path mechanisms. MARULHO should select a
bounded replay window from explicit local evidence, run it only in sleep/replay
maintenance, and keep archival metadata CPU-resident unless active replay
computation benefits from CUDA.

## MARULHO Relevance

Modern Hopfield work is useful as an associative-memory operator, but in
MARULHO it must remain local: inside a column, a routed candidate set, or a
bounded replay window. Its attention equivalence is not permission to add a
transformer-like global mind or scan all memory in the live tick.

Complementary learning systems, continual-learning replay, latent/sparse replay,
and synaptic tagging/capture all point in the same engineering direction:
separate fast live plasticity from slower replay/consolidation; replay selected
compressed evidence rather than raw unbounded history; and promote memories only
when tags/PRP/replay pressure are positive enough to justify the cost.

## Implementation Implication

`DualMemoryStore.select_replay_window(...)` records
`bounded_replay_window_selection.v1`. When deep sleep has column anchors, the
selection scores only entries attached to those bucket ids through the
bucket-to-entry index. If no bucket scope is available, the report says
`global_slow_path_score_scan` so the full slow-memory scorer is not hidden.

`DualMemoryStore.recall_replay_window(...)` records
`bounded_replay_window_recall.v1`. It is a non-mutating slow-path local memory
operator over the selected replay window: routing keys and optional input
patterns stay CPU-normalized for archival recall evidence, `runs_live_tick=false`,
`mutates_runtime_state=false`, and no plasticity is applied.

Zero-pressure replay is now retired: if the global scorer finds no positive
consolidation/repair/maintenance pressure, it returns an empty selection with
`fallback_reason=no_positive_global_scores` instead of rehearsing arbitrary
zero-score entries.

The 2026-06-17 synthetic benchmark now separates stored-experience recall from
prototype repair. With positive anchor pressure, the bounded replay window
recalled stored Task-A input patterns with mean distance
`5.960464477539063e-08` under the `0.01` gate while scoring only bucket-indexed
entries. The zero-pressure guard applied `0` replay updates, and the global
control stayed unbounded for recall. The reconstruction/prototype consolidation
gate still failed, so the next replay slice must improve that target before
claiming consolidation promotion.

## Status

bounded slow-path selection and stored-experience recall implemented; prototype
consolidation promotion open

## Links

- [Research notes](../../research-living-brain.md)
- [Column Runtime](../concepts/column-runtime.md)
- [Replay Cost](../benchmarks/replay-cost.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)
