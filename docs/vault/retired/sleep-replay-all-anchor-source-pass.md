---
type: retired
status: retired
related_code:
  - ../../../src/marulho/training/replay_anchor_window.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/evaluation/sleep_replay_anchor_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/sleep-replay-anchor-source-window-bounded.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-sleep-replay-anchor-source-window.json
---

# Sleep Replay All-Anchor Source Pass

## Retired Path

Trainer sleep replay used to sort every checkpointed `column_anchors` bucket
before calling `DualMemoryStore.select_replay_window(...)`. The store selector
was already bucket-windowed, but the trainer source bucket set still scaled with
retained anchor count before sleep replay selection.

## Replacement

`sleep_replay_anchor_bucket_source_window(...)` now emits
`bounded_sleep_replay_anchor_bucket_source_window.v1` from the shared
`replay_anchor_window.py` helper used by HF replay-query and sleep replay. The
maintained path passes at most `16` reverse-recency anchor buckets into sleep
replay selection and carries the source report into
`_last_sleep_replay_selection_report`.

The report records source/window counts, selected anchor metadata, CPU
archival/source/compute placement, no live tick, no every-token work, no raw
replay text, no hidden language reasoning, no global score/candidate scan, and
`anchor_source_full_scan=false`.

## Evidence

`reports/bounded_replay_window_20260622/sleep-replay-anchor-source-window-bounded.json`
passed over `8192` anchors and `64` iterations. Source latency fell from
`0.892263 ms` to `0.037825 ms`, full selection latency fell from
`7.797869 ms` to `0.104864 ms`, newest-anchor source and selected replay-bucket
hit rates were both `1.0`, and CUDA allocation delta was `0.0 MiB`.

The paired hot-path report
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-sleep-replay-anchor-source-window.json`
processed `524288` tokens at `6135.629 tokens/sec`, kept route scoring bounded
at `12/65536`, cached `65526` transition rows, kept RTX 3060 memory
`1615->1614 MiB`, and recorded zero graph/native sequence failures.

## Revisit Condition

Do not reintroduce all-anchor sleep source construction as production code.
Full-anchor comparisons belong only in benchmark-local diagnostics unless a
future indexed replay source proves better quality, keeps archival metadata
CPU-resident, reports no hidden language reasoning or live/every-token work,
and preserves the 6k-ish long-run hot-path band.
