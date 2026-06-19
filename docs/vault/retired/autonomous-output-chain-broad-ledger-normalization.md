---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-output-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-output-chain.json
---

# Autonomous Output Chain Broad Ledger Normalization

The retired production shape called `_normalized_state()` before autonomous
language-output and decoded-output append/review. That normalized every retained
SNN readout-ledger event family just to inspect
`autonomous_language_output_events` or `autonomous_decoded_output_events`.

The maintained path is the record-family source window:
`execute_autonomous_language_output(...)` and
`autonomous_language_output_event_review(...)` read only
`autonomous_language_output_events`; `execute_autonomous_decoded_output(...)`
and `autonomous_decoded_output_event_review(...)` read only
`autonomous_decoded_output_events`. Accepted writes persist only the target
event family plus its count and timestamp. Reviews read only the target event
family before hash-lineage checks.

The retired broad-normalized comparison remains only inside
`snn_readout_ledger_normalization_source_window_benchmark` as diagnostic
evidence. The large report preserved hash, review-match, and count parity
across binding, observation, training, decoder probe, language output, and
decoded output while checking `1536` bounded target-family rows instead of
`35328` broad-normalized rows (`23x`) and reducing mean chain latency from
`6778.768800 ms` to `321.988933 ms` (`21.052801x`). The benchmark reports CPU
archival/lookup/write placement, CUDA available but unused for ledger metadata,
no raw text payload, no hidden language reasoning, no live tick, and no
every-token cadence.

The paired `524288`-token hot-path run stayed in band at
`6048.638 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, GPU memory `2046->2047 MiB`, and zero graph/native sequence
failures under observed GPU contention. This is throughput-protection evidence,
not a speed promotion.

Do not restore broad normalized autonomous output append/review as a
compatibility branch. The output chain remains hash-only, bounded to its event
families, and CPU-resident for archival ledger metadata.
