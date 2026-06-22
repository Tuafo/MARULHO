---
type: retired
status: retired
related_code:
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/consolidation/memory_store.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/recent-anchor-capture-store-owned-row.json
---

# Recent Anchor-Capture Direct Bucket Reads

`capture_recent_memory_anchors(...)` used to select a bounded recent-memory
window, then read `memory_store.slow_bucket_ids[idx]` directly in trainer code.
The window was bounded, but the row access still lived outside
`DualMemoryStore`.

The maintained path is:

- `DualMemoryStore.collect_recent_entry_indices(...)` selects the bounded
  recent source window.
- `DualMemoryStore.recent_anchor_capture_row(...)` reads one selected anchor
  row under `bounded_recent_anchor_capture_row.v1`.
- Trainer anchor capture consumes the row surface and reports
  `direct_slow_memory_bucket_reads_retired=true`.

Evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260622\recent-anchor-capture-store-owned-row.json`
  passed with `64` captured rows, `anchor_row_read_count=64`, zero invalid
  anchor rows, CPU archival placement, no global scan, no live tick, no
  every-token work, no raw replay text, no hidden language reasoning, and
  `0.0 MiB` CUDA allocation delta.
- Mean capture latency was `1.743 ms` with p95 `2.071 ms`.
- The paired hot-path report
  `..\..\MARULHO_reports\bounded_replay_window_20260622\hotpath-active-pressure-65536-524288-i32-recent-anchor-capture-row.json`
  processed `524288` tokens at `5916.223 tokens/sec`, p95 tick
  `21.992 ms`, bounded `12/65536` route rows, RTX memory `1997->1998 MiB`,
  no observed contention, and zero graph/native/sequence failures.

Do not restore trainer-side direct reads of `slow_bucket_ids` for anchor
capture. New anchor/replay setup work must use store-owned row surfaces with
explicit source budgets, device placement, quality gates, and hot-path
protection evidence.
