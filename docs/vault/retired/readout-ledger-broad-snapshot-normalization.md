---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_snapshot_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
  - ../benchmarks/replay-cost.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/snn-readout-ledger-snapshot-comparator-removed.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-readout-ledger-snapshot-comparator-removed-default-nosample.json
---

# Readout Ledger Broad Snapshot Normalization

## Status

Retired from production code on 2026-06-20 and removed from benchmark code on
2026-06-24.

## Why

`SNNLanguageReadoutEvidenceLedger.snapshot(...)` called `_normalized_state()`,
which normalized every retained SNN readout-ledger event family before returning
only the requested display rows. That made a service/status snapshot pay a
checkpoint-style all-family retention budget, kept a broad control-plane read
path beside bounded replay/readout operators, and scaled poorly for future
LLM-size histories.

## Replacement

`snapshot(...)` now reads only the event families returned by the snapshot
through `bounded_snn_readout_ledger_snapshot_source_window.v1`. Each returned
family is capped by the requested snapshot limit and the ledger retention limit.
Retained summary counts remain available from known source counts; returned rows
and unique hashes are scoped to the snapshot window. The source-window report
keeps archival/source/snapshot placement on CPU and states no live tick, no
every-token cadence, no global candidate/score scan, no hidden language
reasoning, and no CUDA archive.

## Evidence

`reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window.json`
used `2048` rows per retained ledger family, `ledger_limit=128`, and
`snapshot_limit=20`. The bounded snapshot read `260` rows instead of `2944` in
the benchmark-local retired normalizer model, preserved newest-first display
quality and retained-count parity, and reduced mean latency from `393.040600 ms`
to `67.334088 ms` (`5.837171x`). Python traced peak allocation was
`0.575356 MiB`; CUDA allocation/reservation stayed `0.0 MiB`.

`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-readout-ledger-snapshot-comparator-removed-default-nosample.json`
processed `524288` tokens at `6069.794524 tokens/sec`, with
`train_compute=0.132884 ms/token`, `prepare_training=0.006725 ms/token`,
`finalize_total=0.006776 ms/token`, bounded `12/65536` route rows, `10` output
candidates, `65526` cached transition rows, zero graph/native sequence
failures, boundary `contention_observed` (`cpu max=64%`, `gpu max=20%`), and
flat RTX 3060 memory at `2191 MiB`.

The follow-up
`reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window-production-normalizer-retired-smoke.json`
confirms the snapshot benchmark no longer calls production `_normalized_state()`.
That historical benchmark-local all-family comparison preserved quality while
the active snapshot read `260` rows instead of `2944`, reducing mean latency
from `409.182080 ms` to `71.702280 ms` with `0.0 MiB` CUDA
allocation/reservation.

The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\snn-readout-ledger-snapshot-comparator-removed.json`
deletes the executable all-family snapshot comparator. It passes by verifying
returned-field-only source reads directly, reading `260` bounded CPU rows for
`13` snapshot fields, projecting `2684` all-family rows removed from `23`
retained ledger fields, averaging `77.561856 ms`, tracing `0.581556 MiB`
Python peak, and keeping CUDA allocation/reservation at `0.0 MiB`.

## Revisit Only If

A future snapshot/status surface proves a stronger measured grounding or review
target that cannot be satisfied inside the bounded snapshot source window, and
repeated long-run hot-path evidence shows the replacement does not reduce
sustained throughput or reintroduce broad ledger normalization, repo-local
benchmark baselines, replay authority, GPU-resident archival metadata, or
hidden language reasoning.
