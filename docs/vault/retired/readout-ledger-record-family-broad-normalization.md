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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-record-family-append.json
---

# Readout Ledger Record-Family Broad Normalization

The retired production shape called `_normalized_state()` or `snapshot(limit=0)`
before appending one readout-ledger record family. That normalized every retained
ledger event family just to check duplicates in one target family, then used the
general store path where focused field writes were enough.

The maintained path is
`bounded_snn_readout_ledger_record_family_source_window.v1`. Draft,
rollout-replay, emission-review, and dense-label candidate recorders read only
their target event family before duplicate detection. Accepted records persist
only that event family plus its total-count and timestamp fields.

The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json`
preserved latest-hash and total-count parity while checking `128` target-family
rows instead of `2944` normalized ledger rows. Mean append latency moved from
`883.251340 ms` to `57.255420 ms` (`15.426511x`). The path reports CPU
archival/lookup/write placement, CUDA available but unused, no raw text payload,
no language reasoning, no live tick, no every-token cadence, and no plasticity.

The paired `524288`-token hot-path run stayed in the maintained 6k-ish band at
`5966.765 tokens/sec`, with bounded `12/65536` route rows, no observed
contention, GPU memory `2046->2043 MiB`, and zero graph/native sequence
failures. This is throughput-protection evidence, not a speed promotion.

Do not restore broad normalized single-family appends as compatibility branches.
Ledger recorders must stay one-path: selected source family, bounded duplicate
check, focused CPU write.
