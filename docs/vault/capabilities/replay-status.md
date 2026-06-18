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
  - reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json
  - reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json
  - reports/bounded_replay_window_20260617/frontier-gap-bounded.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-frontier-gap-bounded.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json
  - reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json
  - reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json
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
- `bounded_concept_frontier_memory_metrics.v1` is emitted from autonomy source
  acquisition planning. It derives candidate buckets from the probe-bank
  signature, collects a capped bucket-indexed memory window, and computes
  novelty/uncertainty/support only over those selected entries.
- `bounded_concept_memory_signature_lookup.v1` is emitted from ConceptStore
  semantic observation. It resolves memory signatures only from
  already-selected evidence indices, caps each source at `8` unique indices,
  direct-indexes CPU archival arrays, and reports no archive list
  materialization or global candidate/score scan.
- `bounded_frontier_gap_selection.v1` is emitted from semantic frontier
  planning. It asks `DualMemoryStore.collect_frontier_gap_indices(...)` for a
  capped CPU recency or bucket candidate window, scores only selected raw-window
  payloads for explicit gap terms, and reports no global candidate/score scan
  or hidden language reasoning.
- `bounded_recent_memory_window.v1`, `bounded_recent_memory_tag.v1`, and
  `bounded_recent_anchor_capture.v1` are emitted from the recent replay setup
  path. They collect from a CPU recency index, cap by `max_recent_entries`,
  report selected indices and scan flags, and keep tag/anchor setup out of the
  live tick.
- `bounded_awake_ripple_tag.v1` is emitted from awake-ripple replay-priority
  tagging. Production tagging requires awake bucket scope, caps candidates
  through the CPU bucket/recency index, and marks `runs_every_token=false`;
  the old global scalar/vector scan is diagnostic-only.
- Deep sleep can select from column-anchor bucket ids without scoring unrelated
  memory entries.
- Bucket-scoped selection now caps candidate entries before scoring and reports
  `candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
  `candidate_window_limit`, `candidate_index_available_count`, and scored
  `candidate_index_count`.
- Replay-priority scoring no longer has a public full-buffer helper. Callers
  must pass selected candidate indices to `replay_scores_for_indices(...)`.
- The older public full-buffer score tensor helper family is also removed.
  Production selection scores only selected candidates; explicit global scoring
  exists only as a diagnostic branch inside `select_replay_window(...)`.
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
- `reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json`
  proves similarity-only query readout now materializes replay text only for
  returned matches. It preserves selected indices against the retired eager
  candidate-payload shape, drops raw text payload loads from `192` to `5`, and
  reduces mean readout latency from `33.612 ms` to `25.881 ms`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json`
  keeps the longer live tick protected at `6152.079 tokens/sec`, bounded
  `12/65536` route rows, flat GPU memory (`1874->1878 MiB`), no observed
  contention, and zero graph/native failures.
- `reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json`
  keeps concept-frontier acquisition metrics bounded at `64/8192` scored
  entries, preserves the diagnostic full-scan top-1, and reduces metric latency
  from `658.116 ms` to `5.040 ms` with no global score/candidate scan.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json`
  keeps the live tick protected at `6148.846 tokens/sec`, bounded
  `12/65536` route rows, flat `1805 MiB` GPU memory, no observed contention,
  and zero graph/native failures.
- `reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json`
  keeps ConceptStore signature lookup bounded to evidence-provided indices over
  `65536` archival entries, preserves diagnostic legacy signature quality
  (`min cosine=0.9999998212`), removes archive list materialization, and cuts
  mean lookup latency from `12.490 ms` to `1.454 ms`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json`
  keeps the live tick protected at `6143.768 tokens/sec`, bounded
  `12/65536` route rows, flat `1746 MiB` GPU memory, no observed contention,
  and zero graph/native failures.
- `reports/bounded_replay_window_20260617/frontier-gap-bounded.json` retires
  semantic frontier planning's archive-wide raw-window scan. It scored
  `192/65536` entries, preserved expected and diagnostic legacy frontier terms
  (`quality.min=1.0`), reduced mean latency from `221.554 ms` to `9.589 ms`
  (`23.105x`), and reported CPU archival/scoring with no global scans.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-frontier-gap-bounded.json`
  keeps the longer live tick protected at `6184.133 tokens/sec`, bounded
  `12/65536` route rows, flat GPU memory (`1884->1880 MiB`), no observed
  contention, and zero graph/native failures.
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
- `reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json`
  keeps recall/prototype gates passing after deleting the public
  `maintenance_scores(...)`, `consolidation_scores(...)`, `repair_scores(...)`,
  and `fragility_scores(...)` archive-wide tensor helpers.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json`
  keeps the live tick protected at `6151.952 tokens/sec`, bounded
  `12/65536` route rows, flat `1805 MiB` GPU memory, no observed contention,
  and zero graph/native failures.
- `reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json`
  proves wake-bucket scoped ripple tagging avoids global memory scans: scoped
  tagging used `0` scalar/vector scans, `256` awake-bucket scans, and averaged
  `1.091997 ms` versus `1.433332 ms` for the diagnostic global vector path.
- `reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json`
  keeps recall/prototype gates passing and records
  `last_awake_ripple_tag_report` with `global_candidate_scan=false`,
  `diagnostic_global_candidate_scan=false`, and `runs_every_token=false`.
- `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json`
  keeps the longer live tick protected at `6152.328 tokens/sec`, bounded
  `12/65536` route rows, flat `2013 MiB` GPU memory, no observed contention,
  and zero graph/native failures.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Replay Cost](../benchmarks/replay-cost.md)
