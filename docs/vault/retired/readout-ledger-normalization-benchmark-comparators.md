---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ..\..\..\..\MARULHO_reports\bounded_replay_window_20260624\snn-readout-ledger-normalization-comparators-removed.json
  - ..\..\..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-readout-ledger-normalization-comparators-removed-default-nosample.json
---

# Readout-Ledger Normalization Benchmark Comparators

## Retired Shape

`snn_readout_ledger_normalization_source_window_benchmark.py` kept executable
full-materialized normalization/store comparators and broad-normalized
per-boundary comparators after production had already moved to bounded
readout-ledger source windows.

## Decision

The benchmark is maintained-only now. It calls the bounded ledger helpers,
checks seeded newest-first reconstruction directly, and records absence fields
for the removed comparator families instead of running old implementations.

## Evidence

`..\..\..\..\MARULHO_reports\bounded_replay_window_20260624\snn-readout-ledger-normalization-comparators-removed.json`
passed with `2944` bounded rows out of `47104` known source rows, `44160`
full-materialized rows removed, CPU archival/normalization/store/lookup
placement, no live tick, no every-token cadence, no global scan, no hidden
language reasoning, and `0.0 MiB` CUDA allocation/reservation.

The paired hot-path run
`..\..\..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-readout-ledger-normalization-comparators-removed-default-nosample.json`
processed `524288` tokens at `6507.349 tokens/sec`, p95 `19.722 ms`, route
scoring `12/65536`, cached `65526` transition rows, no observed contention,
flat RTX memory, and zero graph/native sequence failures.

## Rule

Do not reintroduce these comparators as repo-local diagnostics. Rejected
readout-ledger recall/replay shapes belong in retired docs, not executable side
paths, unless a new ADR reopens the trade-off with stronger quality evidence and
long-run throughput protection.
