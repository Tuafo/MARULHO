---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
related_docs:
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-synapse-provenance-map.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-synapse-provenance-map.json
---

# Synapse Provenance Broad Event-Map Normalization

## Status

Retired from production code on 2026-06-19.

## Why

`synapse_provenance_audit(...)` only needs readout evidence rows referenced by
the runtime `synapse_provenance_by_key` map. The old production shape called
`_normalized_state()` and materialized all retained readout-ledger families
before building a readout evidence event map. That broad normalization was a
side path that would scale poorly for future LLM-size readout and provenance
histories.

## Replacement

The audit now gathers requested readout hashes from `readout_evidence_hash` and
`readout_evidence_hashes`, reads only the `events` family through
`bounded_snn_readout_evidence_event_map_source_window.v1`, and records the
source-window report in the promotion gate. The lookup keeps archival metadata
and event-map work on CPU, does not run in the live tick, does not run every
token, and does not load replay text or perform hidden language reasoning.

## Evidence

`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-synapse-provenance-map.json`
passed with requested event-map hash parity. Checked rows dropped from `2944`
to `128` (`23x`), and mean event-map latency dropped from `319.823233 ms` to
`13.972533 ms` (`22.889424x`).

`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-synapse-provenance-map.json`
processed `524288` tokens at `5994.111 tokens/sec` with bounded `12/65536`
route rows, `10` output candidates, `65526` cached transition rows, CUDA
runtime on RTX 3060, no observed contention, GPU memory `1980->1976 MiB`, and
zero graph/native sequence failures.

## Revisit Only If

A future synapse audit proves a stronger measured quality target that cannot be
satisfied by requested-hash event lookup, and repeated long-run hot-path
evidence shows the replacement does not reduce sustained throughput or
reintroduce full-ledger scans, GPU-resident archival metadata, or hidden
language reasoning.
