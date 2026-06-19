---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-emission-history.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-history.json
---

# Emission Review History Broad Normalization

## Status

Retired from production code on 2026-06-19.

## Why

`emission_review_history(...)` only needs the reviewed emission display rows in
`emission_review_events`. The old production shape called `_normalized_state()`,
which normalized every retained SNN readout-ledger event family before returning
a capped operator display history. That contradicted the narrow display
contract and preserved a broad readout-ledger side path for future LLM-size
review histories.

## Replacement

Emission review history now reads only `emission_review_events` through
`bounded_snn_emission_review_history_source_window.v1`. The response carries the
source-window report, keeps archival/lookup metadata on CPU, and does not run
replay, mutate runtime state, write checkpoints, or perform hidden language
reasoning. Reviewed bounded text may still be exposed for operator display.

## Evidence

`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-emission-history.json`
passed with review-hash and text-hash parity. Checked rows dropped from `2944`
to `128` (`23x`), and mean display-history latency dropped from `345.815600 ms`
to `25.503433 ms` (`13.559570x`).

`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-history.json`
processed `524288` tokens at `6051.817 tokens/sec` with bounded `12/65536`
route rows, `10` output candidates, `65526` cached transition rows, CUDA
runtime on RTX 3060, no observed contention, flat GPU memory at `1972 MiB`, and
zero graph/native sequence failures.

## Revisit Only If

A future operator display surface proves a stronger measured review/grounding
target that cannot be satisfied inside the bounded review-history source
window, and repeated long-run hot-path evidence shows the replacement does not
reduce sustained throughput or reintroduce broad ledger scans, replay authority,
GPU-resident archival metadata, or hidden language reasoning.
