---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_plasticity_executor.py
  - ../../../src/marulho/service/transition_memory_source_window.py
  - ../../../src/marulho/evaluation/plasticity_runtime_state_source_window_benchmark.py
  - ../../../tests/test_snn_language_plasticity_executor.py
  - ../../../tests/test_status_read_model.py
  - ../../../tests/test_snn_language_readout_ledger.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260621/plasticity-runtime-state-source-window.json
  - reports/bounded_replay_window_20260621/hotpath-active-pressure-65536-524288-i32-plasticity-runtime-state-source-window.json
  - reports/bounded_replay_window_20260621/hotpath-active-pressure-65536-524288-i32-plasticity-runtime-state-source-window-rerun.json
---

# Plasticity Runtime-State Full Snapshot

## Status

Retired from production code on 2026-06-21.

## Why

`SNNLanguagePlasticityApplicationExecutor.snapshot()` used to begin by
deep-copying the complete SNN language plasticity state. That made
`/terminus/snn-language-sequence/plasticity-runtime-state` a potential
full-retained transition-memory export for `sparse_transition_weights`,
`synapse_provenance_by_key`, critical-period by-synapse rows, and pruned
provenance rows.

The endpoint is read-only control-plane state, not live replay, but the shape
scaled with retained transition memory and invited consumers to treat returned
maps as complete. For future LLM-size memories, runtime-state projection must
not be a hidden archive scan, replay operator, or status-side integrity audit.

## Replacement

`snapshot()` now builds
`bounded_snn_language_plasticity_runtime_transition_memory_source_window.v1`
through `transition_memory_source_window.py`. It returns at most `64` newest
sparse-transition rows and `64` newest synapse-provenance rows from CPU
archival state, while preserving retained/source/truncated counts in
`transition_memory_source_window`.

Status and readout-ledger consumers read those retained counts instead of
treating bounded maps as complete. Exact audit/readiness paths block when the
source window is truncated and must move to explicit slow audit or replay
windows when exact integrity is required.

## Evidence

`reports/bounded_replay_window_20260621/plasticity-runtime-state-source-window.json`
passed on `65536` sparse weights and `65536` provenance rows. The active path
read `256` source records versus `262144` in the benchmark-local retired full
snapshot (`1024x` less source work), reduced mean latency from `752.314014 ms`
to `7.770271 ms` (`96.819528x`), used `0.110454 MiB` traced Python peak versus
`12.287186 MiB`, and kept CUDA allocation/reservation deltas at `0`.

Focused regression coverage proves the runtime snapshot does not fully scan a
counting mapping, status retained counts survive the bounded source window, and
readout synapse-provenance audits use the retained counts when the runtime
window is partial.

The paired `524288`-token protection runs succeeded with no observed
contention, bounded `12/65536` route scoring, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures. They measured `5642.888` and `5736.332 tokens/sec`, so this retired
path has live-tick protection evidence but not durable completion or speed
ceiling evidence.

## Revisit Only If

Do not restore a production full runtime-state transition-memory snapshot.
Exact all-retained transition-memory comparisons belong only in explicit
offline diagnostics or slow audit/replay windows with a declared source budget,
CPU archival placement, no hidden replay text or language reasoning, no
live-tick/every-token work, no GPU-resident archival metadata, and repeated
6k-ish hot-path protection evidence.
