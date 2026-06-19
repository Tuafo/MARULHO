---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/hot-path-latency.md
  - ../papers/replay-consolidation.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json
---

# Autonomous Confidence Use Broad Ledger Normalization

The retired production path called `_normalized_state()` before autonomous
calibrated confidence-use duplicate detection and event review. That normalized
every retained readout-ledger event family just to inspect
`autonomous_confidence_use_events`.

The maintained path is
`bounded_snn_autonomous_confidence_use_source_window.v1`. The hash-only
executor and read-only event review read only
`autonomous_confidence_use_events`. The focused writer persists only that event
family, total use count, and last-used timestamp.

The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json`
preserved event-hash parity while checking `128` confidence-use rows instead of
`2944` normalized ledger rows. Mean lookup latency moved from `350.647280 ms`
to `13.261960 ms` (`26.439331x`). The path reports CPU archival/lookup/write
placement, CUDA available but unused, no raw text payload, no language
reasoning, no live tick, no every-token cadence, and no plasticity.

The paired `524288`-token hot-path run stayed in the maintained 6k-ish band at
`5965.377 tokens/sec`, with bounded `12/65536` route rows, no observed
contention, GPU memory `2045->2047 MiB`, and zero graph/native sequence
failures. This is throughput-protection evidence, not a speed promotion.

Do not restore the broad normalized duplicate/review lookup as a compatibility
branch. Confidence use remains a bounded hash-only ledger path, not a text
reasoning or live replay operator.
