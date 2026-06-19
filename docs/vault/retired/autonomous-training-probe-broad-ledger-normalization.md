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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-training-probe-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json
---

# Autonomous Training Probe Broad Ledger Normalization

The retired path normalized every retained SNN readout-ledger event family, then
used that broad snapshot to append or review one autonomous training-window or
decoder-probe event. That shape is not a valid production path for future
LLM-size memory: it scales with unrelated ledger families, can hide global
control-plane work behind a capped result, and weakens the rule that replay and
readout evidence must be selected before any associative lookup.

The active path uses `bounded_snn_readout_ledger_record_family_source_window.v1`.
`execute_autonomous_readout_training_window(...)` and
`autonomous_readout_training_window_event_review(...)` read only
`autonomous_readout_training_window_events`; `execute_autonomous_decoder_probe(...)`
and `autonomous_decoder_probe_event_review(...)` read only
`autonomous_decoder_probe_events`. Accepted writes persist only that event
family plus its count and timestamp fields.

The retired broad-normalized comparison remains only inside
`snn_readout_ledger_normalization_source_window_benchmark` as diagnostic
evidence. The large report preserved hash, review-match, and count parity while
checking `1024` bounded target-family rows instead of `23552` broad-normalized
rows (`23x`) and reducing mean chain latency from `4927.213200 ms` to
`197.573467 ms` (`24.938638x`). The hot-path protection run processed `524288`
tokens at `6057.953 tokens/sec` with bounded `12/65536` route rows and zero
graph/native sequence failures; GPU-side contention was observed, so the run is
protection evidence rather than a clean speed ceiling.
