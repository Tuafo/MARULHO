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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-update-source-window.json
---

# Dense Label Calibration Update Broad Ledger Normalization

The retired production path called `_normalized_state()` before dense-label
calibration update application and application-review, then wrote applied
updates through `_store_state(...)`. That normalized every retained readout
ledger event family just to inspect `dense_label_calibration_update_events` and
the current calibration update. On mutation, the general writer also rewrote
unrelated event families.

The maintained path is
`bounded_snn_dense_label_calibration_update_source_window.v1`. Operator and
autonomous update executors, plus both application-review surfaces, read only
`dense_label_calibration_update_events` for duplicate/current lineage checks.
The focused writer persists only calibration update events, current update,
total update count, and last-applied timestamp.

The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json`
preserved event-hash and current-hash parity while checking `128` update rows
instead of `2944` normalized ledger rows. Mean lookup latency moved from
`245.671760 ms` to `11.647260 ms` (`21.092666x`). The path reports CPU
archival/lookup/write placement, CUDA available but unused, no raw text payload,
no language reasoning, no live tick, no every-token cadence, and no plasticity.

The paired `524288`-token hot-path run stayed in the maintained 6k-ish band at
`6009.497 tokens/sec`, with bounded `12/65536` route rows, GPU memory
`2045->2046 MiB`, and zero graph/native sequence failures. The velocity sample
observed GPU-side contention at `21%`, so this is throughput-protection
evidence rather than a speed promotion.

Do not restore the broad normalized update/current lookup as a compatibility
branch. Full-ledger normalization remains available only where the ledger
snapshot/store-state boundary is the explicit surface under test.
