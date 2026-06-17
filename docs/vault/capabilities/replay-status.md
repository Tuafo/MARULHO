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
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json
  - reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json
  - reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
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
- `bounded_replay_query_collection.v1` is emitted from `DualMemoryStore` as a
  CPU slow-path query collector over the same bucket-indexed candidate window;
  it collects query indices without scoring memory entries or walking all
  `slow_bucket_ids`.
- `bounded_query_memory_match.v1` is emitted from `query_runner` for explicit
  query/readout recall. It derives candidate buckets from routing, collects a
  capped bucket-indexed memory window, and computes similarity/replay-priority
  scores only for those candidate entries.
- `bounded_recent_memory_window.v1`, `bounded_recent_memory_tag.v1`, and
  `bounded_recent_anchor_capture.v1` are emitted from the recent replay setup
  path. They collect from a CPU recency index, cap by `max_recent_entries`,
  report selected indices and scan flags, and keep tag/anchor setup out of the
  live tick.
- Deep sleep can select from column-anchor bucket ids without scoring unrelated
  memory entries.
- Bucket-scoped selection now caps candidate entries before scoring and reports
  `candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
  `candidate_window_limit`, `candidate_index_available_count`, and scored
  `candidate_index_count`.
- Replay-priority scoring no longer has a public full-buffer helper. Callers
  must pass selected candidate indices to `replay_scores_for_indices(...)`.
- Zero-pressure global replay is retired with
  `fallback_reason=no_positive_global_scores`.
- Deep sleep blocks unanchored global replay mutation with
  `unscoped_global_fallback_retired=true`, so no-anchor and zero-pressure bucket
  cases apply `0` replay updates.
- Unscoped random replay is retired by default; explicit diagnostics must report
  `global_slow_path_candidate_scan`.
- `reports/bounded_replay_window_20260617/synthetic-selection.json` passes the
  bounded stored input-pattern recall gate for positive-pressure windows
  (`5.960464477539063e-08` mean distance against threshold `0.01`), but does
  not pass the prototype reconstruction gate, so consolidation promotion remains
  quality-open.
- `reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json`
  keeps the positive-pressure recall/prototype gates passing under the capped
  selector and reports CPU archival/scoring with no global score or candidate
  scan.
- `reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json`
  keeps HF stored-experience recall passing with `3` bounded Task-A anchor
  queries, `candidate_window_limit=16`, `score_count=0`, no global scans, and
  after-consolidation `mean_input_pattern_distance=0.0`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json`
  keeps the live tick protected at `6221.949 tokens/sec`, bounded
  `12/65536` route rows, flat `1848 MiB` GPU memory, and zero graph/native
  failures.
- `reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json`
  proves explicit query readout now reports `bounded_query_memory_match.v1`
  with `candidate_window_limit=192`, `1` candidate scored, no global scans, CPU
  archival placement, and `runs_live_tick=false`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json`
  keeps the live tick protected at `6137.185 tokens/sec`, bounded
  `12/65536` route rows, flat `1848 MiB` GPU memory, and zero graph/native
  failures.
- `reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json`
  keeps replay recall/prototype gates passing while recent tag and anchor setup
  use `candidate_window_limit=256`, `candidate_index_count=14`, no global
  scans, CPU archival storage, and `runs_live_tick=false`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json`
  keeps the live tick protected at `6228.243 tokens/sec`, bounded
  `12/65536` route rows, flat `1846 MiB` GPU memory, and zero graph/native
  failures.
- `reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json`
  keeps recall/prototype gates passing after deleting the full-buffer
  `replay_scores(...)` helper and moving tests to explicit candidate indices.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json`
  keeps the live tick protected at `6211.859 tokens/sec`, bounded
  `12/65536` route rows, flat `1852 MiB` GPU memory, and zero graph/native
  failures.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Replay Cost](../benchmarks/replay-cost.md)
