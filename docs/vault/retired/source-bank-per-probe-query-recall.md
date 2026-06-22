---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/semantics/frontier.py
  - ../../../src/marulho/evaluation/source_bank_memory_match_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
  - ../concepts/replay-window.md
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json
---

# Source-Bank Per-Probe Query Recall

## Status

Retired from production source-bank semantic recall.

## Why Retired

The old source-bank path sampled bounded probes but called query-memory recall
once per probe. That kept recall outside the live tick, but it still preserved
multiple recall passes, duplicated candidate collection, and kept a second
implementation shape beside the bank-level source-acquisition plan.

Future LLM-size memory needs one explicit source-bank recall boundary with a
single local candidate window, clear budgets, and device placement evidence.

## Replacement

`bank_memory_matches_with_report(...)` now unions sampled probe bucket ids,
collects one CPU bucket-indexed candidate window capped at `192`, vector-scores
the sampled probes against that local associative window, and loads raw text
only for returned matches. The report surface remains
`bounded_source_bank_memory_match.v1`, with `merged_probe_candidate_window=true`
and `per_probe_query_match_call_count=0`.

## Evidence

`reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json`
preserved selected indices against the retired per-probe diagnostic path
(`quality.min=1.0`), reduced raw text payload loads from `32` to `4`, and cut
mean latency from `560.177 ms` to `106.543 ms` (`5.258x`) over a `65536`-entry
archive. Archival storage and scoring stayed on CPU, CUDA archival allocation
and reservation stayed at `0.0 MiB`, and traced Python peak allocation was
`3.059 MiB`.

`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json`
kept the 524288-token live tick protected at `6129.933 tokens/sec`, with
bounded `12/65536` route rows, `65526` cached transition rows, mild GPU
contention observed (`21%` against a `20%` threshold), flat `1763 MiB` GPU
memory, and zero graph/native/sequence failures.

## Revisit Only If

A replacement must keep one bank-level bounded report, one explicit candidate
window, CPU-resident archival metadata unless active replay computation benefits
from CUDA, quality parity or better, and long-run evidence that no archive-wide
or every-token work enters the live tick.
