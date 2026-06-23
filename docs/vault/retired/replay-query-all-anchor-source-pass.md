---
type: retired
status: retired
related_code:
  - ../../../src/marulho/training/memory_consolidation_runner.py
  - ../../../src/marulho/training/replay_anchor_window.py
  - ../../../src/marulho/evaluation/replay_query_anchor_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json
---

# Replay Query All-Anchor Source Pass

## Retired Path

HF replay-query collection used to cap returned queries, but still passed every
checkpointed `column_anchors` bucket into
`DualMemoryStore.collect_replay_query_indices(...)`. That kept an all-anchor
source pass ahead of the bounded bucket-indexed query window.

## Replacement

`_collect_anchor_replay_queries(...)` now emits
`bounded_replay_query_anchor_bucket_source_window.v1` from the shared anchor
window helper and passes at most `16` reverse-recency anchor buckets into query
collection and HF recall. `_bounded_replay_recall_evaluation(...)` also re-caps
canonical inherited query-collection bucket scopes before recall.

The maintained path reports source/window counts, CPU archival and active recall
placement, no live tick, no every-token work, no global score/candidate scan, no
raw replay text, no hidden language reasoning, and
`anchor_source_full_scan=false`.

## Evidence

The historical comparison report
`reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json`
used `8192` anchors and `64` iterations. It reduced source-selection latency
from `16.414 ms` to `0.346 ms`, selected newest-anchor queries with hit rate
`1.0`, kept exact input recall, and used `0.0 MiB` CUDA allocation.

After that comparison was accepted, the benchmark-local all-anchor
implementation was removed. The maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\replay-query-anchor-maintained-only.json`
passed with `retired_all_anchor_source_absence.implementation_present=false`,
bounded recent-anchor hit rate `1.0`, exact input recall
`mean_input_pattern_distance=0.0`, inherited cap `4096->16`, `4080` truncated
inherited buckets, bounded collection mean `1.451 ms`, and `0.0 MiB` CUDA
allocation.

The same-checkpoint `524288`-token protection run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-inherited-query-cap-pinned-main-rerun.json`
processed `6162.974 tokens/sec`, kept route scoring bounded, and recorded zero
graph/native failures.

## Revisit Condition

Do not reintroduce all-anchor replay-query source construction as production
code. Reintroduce full-anchor comparisons only as a new diagnostic-only
benchmark module with explicit source-size accounting and no production import
path.
