---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/status_read_model.py
  - ../../../src/marulho/evaluation/status_transition_memory_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
  - ../benchmarks/replay-cost.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window-rerun.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/status-transition-memory-comparison-surface-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-retired-comparison-surfaces-removed-default-nosample.json
---

# Status Transition Memory Broad Projection

## Status

Retired from production code on 2026-06-20.
Removed from repo-local benchmark code on 2026-06-24.

## Why

Capacity pressure, dense readout tensor integrity, applied-synapse provenance,
and rollout/server binding status each had enough direct access to retained
`sparse_transition_weights` and `synapse_provenance_by_key` data to become
archive-linear readiness checks. These were read-only control-plane paths, not
live replay, but they preserved broad transition-memory scans beside bounded
replay windows.

For future LLM-size histories, operator status must not become the place where
full-memory recall or hidden integrity reasoning happens. Exact integrity is
valid only when the bounded source window is complete; otherwise the evidence
must say the window was partial and block exact review readiness.

## Replacement

`StatusReadModel` now routes transition-memory status through one bounded
source-window helper. Capacity pressure, dense tensor integrity, applied
synapse provenance, and rollout/server binding each read at most `32`
`sparse_transition_weights` rows and `32` `synapse_provenance_by_key` rows.
The source-window evidence reports retained/source/truncated counts, CPU
archival/lookup placement, no global candidate or score scan, no raw text
payload, no hidden language reasoning, no replay, no mutation/plasticity, no
live tick, and no every-token cadence.

When retained state exceeds the window, exact resize, dense-integrity,
applied-synapse audit, and rollout/server review readiness are blocked instead
of computed from partial evidence.

## Evidence

`reports/bounded_replay_window_20260620/status-transition-memory-source-window.json`
used `2048` sparse-transition weights and `2048` provenance rows. The bounded
path read `256` rows across four projections instead of `10240` rows in the
benchmark-local retired repeated broad projection (`40x` less source work),
reduced mean latency from `89.558896 ms` to `11.162376 ms` (`8.023282x`),
kept Python peak allocation at `0.065983 MiB` versus `1.372842 MiB`, and used
`0.0 MiB` CUDA allocation/reservation.

The first `524288`-token protection run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window.json`
succeeded but is rejected as primary evidence: `5278.529 tokens/sec`,
`train_compute=0.153055 ms/token`, and velocity reported GPU contention at the
`20%` threshold.

The accepted rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window-rerun.json`
processed `524288` tokens at `6371.238 tokens/sec`, with
`train_compute=0.128035 ms/token`, `prepare_training=0.006444 ms/token`,
`finalize_total=0.006113 ms/token`, bounded `12/65536` route rows, `10`
output candidates, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures. Velocity still reported borderline GPU contention (`max_gpu=23%`,
memory utilization max `19%`), so this is accepted as same-band throughput
protection, not as a clean speed ceiling. RTX 3060 memory stayed flat at
`1986 MiB`.

The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\status-transition-memory-comparison-surface-removed.json`
removes the executable broad benchmark comparator. It passes by reading `256`
bounded CPU rows across the four projections, keeps the stale-first/recent-last
quality gate true, blocks exact reviews when truncated, averages
`11.302696 ms`, peaks at `0.066196 MiB` traced Python memory, and uses
`0.0 MiB` CUDA allocation.

The paired current-tree hot-path gate
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-retired-comparison-surfaces-removed-default-nosample.json`
processed `524288` tokens at `6259.398 tokens/sec`, p95 `20.244 ms`,
`train_compute=0.129477 ms/token`, route scoring `12/65536`, cached `65526`
transition rows, `state_transition_runs_all_columns=false`, no observed
contention, flat RTX 3060 memory at `1983 MiB`, and zero graph/native sequence
failures.

## Revisit Only If

A future status surface proves it needs exact transition-memory integrity inside
an explicit slow audit/replay window, not routine status projection, and
long-run evidence shows the selected window preserves the maintained 6k-ish
throughput band without GPU-resident archival metadata, hidden replay-text
reasoning, or any live-tick/full-memory scan. Do not restore the repo-local
benchmark comparator; external diagnostics must carry source-size accounting.
