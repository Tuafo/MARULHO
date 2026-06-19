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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-surface-chain.json
---

# Autonomous Language Surface Broad Ledger Normalization

## Status

Retired from production code on 2026-06-19.

## Why

Text-surface materialization and bounded language-surface commit only need one
ledger family for duplicate checks and event review. The old production shape
called `_normalized_state()`, which normalized unrelated readout/replay ledger
families before looking up one materialization or language-surface commit event.
That preserved a broad scan-shaped side path and would scale poorly for
LLM-size language evidence histories.

## Replacement

`execute_autonomous_text_surface_materialization(...)` and
`autonomous_text_surface_materialization_event_review(...)` now use
`bounded_snn_readout_ledger_record_family_source_window.v1` on
`autonomous_text_surface_materialization_events`.

`execute_autonomous_bounded_language_surface_commit(...)` and
`autonomous_bounded_language_surface_commit_event_review(...)` use the same
bounded record-family helper on
`autonomous_bounded_language_surface_commit_events`.

Both paths keep archival/source/review metadata CPU-resident, avoid live-tick
and every-token work, avoid hidden language reasoning, and update only the
target event family plus the single current pointer.

## Evidence

`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json`
passed with hash, review-match, total-count, and current-pointer parity across
the ten-family autonomous language-surface chain. Checked source rows dropped
from `58880` to `2560` (`23x`), and mean chain latency dropped from
`11175.229267 ms` to `525.534133 ms` (`21.264517x`).

`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-surface-chain.json`
processed `524288` tokens at `5994.060 tokens/sec` with bounded `12/65536`
route rows, `65526` cached transition rows, CUDA runtime on RTX 3060, GPU
memory `2044->2059 MiB`, and zero graph/native sequence failures.

## Revisit Only If

A future language-surface mechanism proves a stronger quality target that cannot
be satisfied inside bounded event-family windows, and repeated long-run
hot-path evidence shows the replacement does not reduce sustained throughput or
reintroduce full-ledger scans, GPU-resident archival metadata, or hidden
language reasoning.
