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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-autonomous-chain.json
---

# Autonomous Hash-Readout Broad Ledger Normalization

The retired production shape called `_normalized_state()` before autonomous
hash-readout binding append/review and bound-observation append/review. That
normalized every retained readout-ledger event family just to inspect one
hash-only event family.

The maintained path is the record-family source window:
`autonomous_hash_readout_binding_events` for binding execution/review and
`autonomous_bound_readout_observation_events` for observation execution/review.
Accepted writes persist only the target event family plus its count and
timestamp. Reviews read only the target event family before hash lineage checks.

The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json`
preserved binding and observation hash parity, review-match parity, and
total-count parity while checking `512` target-family rows instead of `11776`
normalized ledger rows. Mean chain latency moved from `2371.472400 ms` to
`110.685950 ms` (`21.425234x`). The path reports CPU archival/lookup/write
placement, CUDA available but unused, no raw text payload, no language
reasoning, no live tick, no every-token cadence, and no plasticity.

The paired `524288`-token hot-path run stayed in band at
`6272.156 tokens/sec`, with bounded `12/65536` route rows, no observed
contention, GPU memory `2044->2045 MiB`, and zero graph/native sequence
failures. This is throughput-protection evidence, not a speed promotion.

Do not restore broad normalized autonomous binding or observation append/review
as a compatibility branch. The chain remains hash-only, bounded to its event
families, and CPU-resident for archival ledger metadata.
