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
  - reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json
---

# Status Applied Synapse Provenance Broad Key Scan

## Status

Retired from production code on 2026-06-20.

## Why

`StatusReadModel._snn_readout_applied_synapse_provenance()` scanned every
retained `sparse_transition_weights` key and every
`synapse_provenance_by_key` row before deciding whether applied
replay-regenerated synapses were ready for audit review. That made an operator
status projection pay an archive-linear provenance cost and left a broad
control-plane readiness check beside the bounded replay windows.

For future LLM-size histories, status must not become the place where hidden
full-memory recall happens. Exact integrity is valid only when the bounded
source window is complete; otherwise the evidence must say it saw only a
bounded window and must block exact audit readiness.

## Replacement

Status evidence now emits
`bounded_snn_status_applied_synapse_provenance_source_window.v1`. It reads at
most `32` sparse-weight keys and `32` provenance rows, reports retained,
source, and truncated counts, keeps archival metadata and lookup on CPU, and
states no live tick, no every-token cadence, no replay execution, no global
candidate or score scan, no raw text payload, no hidden language reasoning, no
runtime mutation, and no plasticity application.

If retained rows exceed the source window, `source_window_complete=false`,
`integrity_count_scope=bounded_source_window`, and
`eligible_for_readout_synapse_audit_review=false`.

## Evidence

`reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json`
used `2048` sparse weights and `2048` provenance rows. The bounded status path
read `64` rows instead of `4096` in the benchmark-local retired broad scan
model (`64x` less source work), reduced mean latency from `66.313336 ms` to
`3.242332 ms` (`20.452358x`), preserved bounded-window provenance health,
reported CPU archival/lookup placement, and used `0.0 MiB` CUDA
allocation/reservation.

The first `524288`-token protection run succeeded at `5875.245 tokens/sec` and
is kept as secondary variance. The accepted rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json`
processed `524288` tokens at `6350.288 tokens/sec`, with
`train_compute=0.128968 ms/token`, `prepare_training=0.006470 ms/token`,
`finalize_total=0.005945 ms/token`, bounded `12/65536` route rows, `10` output
candidates, `65526` cached transition rows, no observed contention, flat RTX
3060 memory at `1936 MiB`, and zero graph/native sequence failures.

## Revisit Only If

A future audit/status surface proves that exact applied-synapse integrity must
be computed in a selected slow audit window, not the status projection, and
long-run evidence shows the selected window preserves the maintained 6k-ish
throughput band without GPU-resident archival metadata, hidden replay-text
reasoning, or any live-tick/full-memory scan.
