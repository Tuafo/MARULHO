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
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-thought-structural-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-thought-structural-chain-rerun.json
---

# Autonomous Language Thought Broad Ledger Normalization

## Status

Retired from production code on 2026-06-19.

## Why

SNN language decoding, readout-surface, thought-memory, thought-consolidation,
and thought-structural-plasticity only need one event family for duplicate
checks and review lookups. The old production shape called `_normalized_state()`
before each append or review, which normalized unrelated readout/replay ledger
families before looking up one downstream language/thought event. That preserved
a broad scan-shaped side path and would scale poorly for LLM-size language and
thought evidence histories.

## Replacement

The executor/review pairs for:

- `autonomous_snn_language_decoding_events`
- `snn_language_readout_surface_events`
- `autonomous_snn_language_thought_memory_events`
- `autonomous_snn_language_thought_consolidation_events`
- `autonomous_snn_language_thought_structural_plasticity_events`

now use `bounded_snn_readout_ledger_record_family_source_window.v1`. Each path
reads only the target event family, updates only the target count/timestamp on
accepted append, keeps archival/source/review metadata CPU-resident, avoids
live-tick and every-token work, and does not replay or reason through text.
The legacy `autonomous_snn_language_thought_surface_events` persisted field is
now a one-way checkpoint migration alias only.

## Evidence

`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-thought-structural-chain.json`
passed with hash, review-match, total-count, and current-pointer parity across
the expanded seventeen-component autonomous readout/language/thought chain.
Checked source rows dropped from `100096` to `4352` (`23x`), and mean chain
latency dropped from `19704.406867 ms` to `1046.241300 ms` (`18.833520x`).

`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-thought-structural-chain-rerun.json`
processed `524288` tokens at `6005.229 tokens/sec` with bounded `12/65536`
route rows, `10` output candidates, `65526` cached transition rows, CUDA runtime
on RTX 3060, no observed contention, GPU memory `1856->1857 MiB`, and zero
graph/native sequence failures.

The same-shape first run succeeded at `5921.867 tokens/sec` but is not primary
throughput evidence because the sampler observed GPU contention.

## Revisit Only If

A future downstream language/thought mechanism proves a stronger measured
quality target that cannot be satisfied inside bounded event-family windows,
and repeated long-run hot-path evidence shows the replacement does not reduce
sustained throughput or reintroduce full-ledger scans, GPU-resident archival
metadata, or hidden language reasoning.
