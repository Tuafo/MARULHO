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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-text-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-text-surface-chain.json
---

# Autonomous Text Surface Broad Ledger Normalization

The retired production shape called `_normalized_state()` before bounded
text-emission append/review and text-surface commit append/review. That
normalized every retained SNN readout-ledger event family just to inspect
`autonomous_bounded_text_emission_events` or
`autonomous_text_surface_commit_events`.

The maintained path is the record-family source window:
`execute_autonomous_bounded_text_emission(...)` and
`autonomous_bounded_text_emission_event_review(...)` read only
`autonomous_bounded_text_emission_events`; `execute_autonomous_text_surface_commit(...)`
and `autonomous_text_surface_commit_event_review(...)` read only
`autonomous_text_surface_commit_events`. Accepted commit writes persist only the
target event family plus count/timestamp fields and update the single
`current_text_surface_commit` pointer. Reviews read only the target event
family plus that current pointer before hash-lineage checks.

The retired broad-normalized comparison remains only inside
`snn_readout_ledger_normalization_source_window_benchmark` as diagnostic
evidence. The large report preserved hash, review-match, total-count, and
current-commit parity across binding, observation, training, decoder probe,
language output, decoded output, bounded text emission, and text-surface commit
while checking `2048` bounded target-family rows instead of `47104`
broad-normalized rows (`23x`) and reducing mean chain latency from
`9289.008333 ms` to `429.436800 ms` (`21.630676x`). The benchmark reports CPU
archival/lookup/write placement, CUDA available but unused for ledger metadata,
no raw text payload, no hidden language reasoning, no live tick, and no
every-token cadence.

The paired `524288`-token hot-path run stayed in band at
`5980.715 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, GPU memory `2045->2047 MiB`, and zero
graph/native sequence failures. This is throughput-protection evidence, not a
speed promotion.

Do not restore broad normalized bounded text-emission or text-surface commit
append/review as a compatibility branch. The text-surface chain remains
hash-only, bounded to its event families, and CPU-resident for archival ledger
metadata.
