---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/training/replay_anchor_window.py
  - ../../../tests/test_memory_consolidation.py
related_benchmarks:
  - ../../../reports/bounded_replay_window_20260622/sleep-replay-anchor-nonreversible-fallback-retired.json
  - ../../../reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-dataset-history-anchor-retired.json
---

# Non-Reversible Anchor Source Materialization

The shared replay anchor-window helper no longer materializes `list(anchors)`
when a source cannot provide reverse-recency iteration. That fallback preserved
a full-source materialization shape beside bounded replay windows.

Replacement:

- reversible dict-backed anchors still read the newest `16` anchors;
- non-reversible sources fail closed with
  `fallback_reason=non_reversible_anchor_bucket_source`;
- the report records `anchor_bucket_source_read_count=0`,
  `anchor_bucket_source_materialized_count=0`, and
  `anchor_source_full_scan=false`.

Evidence:

- focused test:
  `test_sleep_replay_anchor_source_blocks_non_reversible_mapping_without_materializing`;
- benchmark:
  `sleep-replay-anchor-nonreversible-fallback-retired.json` passed over
  `8192` anchors with `16` bounded reads, `0` materialized entries,
  `1.0` newest-anchor hit rate, CPU placement, and `0.0 MiB` CUDA delta;
- long run:
  `6151.826 tokens/sec` over `524288` tokens with bounded route scoring.

Revisit only if a non-reversible anchor source exposes an indexed tail window
with a measured source budget and no full retained-anchor scan.
