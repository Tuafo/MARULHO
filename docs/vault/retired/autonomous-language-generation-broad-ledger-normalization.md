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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-generation-chain.json
---

# Autonomous Language Generation Broad Ledger Normalization

## Status

Retired from production code on 2026-06-19.

## Why

Bounded language-surface use and SNN language-generation only need one ledger
family for duplicate checks and event review. The old production shape called
`_normalized_state()`, which normalized unrelated readout/replay ledger
families before looking up one use or generation event. That preserved a broad
scan-shaped side path and would scale poorly for LLM-size language evidence
histories.

## Replacement

`execute_autonomous_bounded_language_surface_use(...)` and
`autonomous_bounded_language_surface_use_event_review(...)` now use
`bounded_snn_readout_ledger_record_family_source_window.v1` on
`autonomous_bounded_language_surface_use_events`.

`execute_autonomous_snn_language_generation(...)` and
`autonomous_snn_language_generation_event_review(...)` use the same bounded
record-family helper on `autonomous_snn_language_generation_events`.

Both paths keep archival/source/review metadata CPU-resident, avoid live-tick
and every-token work, avoid hidden language reasoning, and update only the
target event family count/timestamp.

## Evidence

`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json`
passed with hash, review-match, total-count, and current-pointer parity across
the expanded autonomous language-generation chain. Checked source rows dropped
from `70656` to `3072` (`23x`), and mean chain latency dropped from
`13505.919533 ms` to `631.221 ms` (`21.396499x`).

`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-generation-chain.json`
processed `524288` tokens at `6074.417 tokens/sec` with bounded `12/65536`
route rows, `65526` cached transition rows, CUDA runtime on RTX 3060, no
observed contention, GPU memory `2044->2047 MiB`, and zero graph/native
sequence failures.

## Revisit Only If

A future language-generation mechanism proves a stronger quality target that
cannot be satisfied inside bounded event-family windows, and repeated long-run
hot-path evidence shows the replacement does not reduce sustained throughput or
reintroduce full-ledger scans, GPU-resident archival metadata, or hidden
language reasoning.
