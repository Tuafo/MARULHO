---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/synapse_provenance_audit_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
  - ../benchmarks/replay-cost.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/synapse-provenance-audit-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-source-window-rerun.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/synapse-provenance-audit-comparator-removed.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-comparator-removed-default-nosample.json
---

# Applied Synapse Provenance Audit Full Scan

## Status

Retired from production code on 2026-06-20 and removed from repo-local
benchmark code on 2026-06-24.

## Why

`synapse_provenance_audit(...)` copied and scanned all retained
`sparse_transition_weights` and `synapse_provenance_by_key` rows before
returning capped audit rows. That path was an explicit audit/control-plane
surface, not a live tick, but it still kept an archive-linear
applied-synapse integrity path beside bounded recall and replay windows.

For future LLM-size transition memory, exact audit review must start from a
selected source window. A full retained-history audit can exist only outside
the repo as an explicit diagnostic script/notebook, or as a separately
selected slow-audit policy with its own source budget and throughput proof.

## Replacement

The maintained audit emits
`bounded_snn_readout_synapse_provenance_audit_source_window.v1`. It reads at
most `64` applied sparse-weight/provenance rows from CPU archival state,
requests ledger evidence only for selected hashes, reports retained/source
counts and truncation, and blocks exact audit review when the source window is
incomplete.

The report states no global candidate or score scan, no raw text payload, no
hidden language reasoning, no live tick, no every-token work, no runtime
mutation, no plasticity application, and no GPU-resident archival metadata.

## Evidence

`reports/bounded_replay_window_20260620/synapse-provenance-audit-source-window.json`
used `2048` retained sparse weights and `2048` retained provenance rows. The
bounded production audit read `64` source rows while the benchmark-local
diagnostic full materializer touched `4096` records and materialized `2048`
rows (`32x` less source work by the report metric). Selected source keys
matched the diagnostic first source window, requested ledger hashes were
capped at `64`, truncated source windows blocked exact review, mean audit
latency was `75.262088 ms` versus `259.221928 ms`, traced Python peak
allocation was `1.909667 MiB`, CUDA was available, and production audit GPU
use was `false`.

The first `524288`-token protection run succeeded at `6441.926 tokens/sec` but
velocity reported contention, so it is secondary evidence. The accepted rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-source-window-rerun.json`
processed `524288` tokens at `6441.166 tokens/sec`, with
`tick_duration_ms.p95=19.527`, `train_compute=0.127184 ms/token`,
`prepare_training=0.006068 ms/token`, `finalize_total=0.005747 ms/token`,
bounded `12/65536` route rows, `10` output candidates, `65526` cached
transition rows, no observed contention, flat RTX 3060 memory at `1866 MiB`,
and zero graph/native sequence failures.

The maintained-only refresh
`..\..\MARULHO_reports\bounded_replay_window_20260624\synapse-provenance-audit-comparator-removed.json`
removes the executable full-scan comparator from the benchmark code. It uses
`seeded_bounded_applied_synapse_audit_source_window_reconstruction`, records
`retired_full_applied_synapse_audit_scan_absence.implementation_present=false`,
reads `64` sparse rows plus `64` provenance rows on CPU, reports `128`
bounded source rows for `2048` retained rows, projects `3968` removed
full-scan rows, averages `73.058288 ms` with p95 `84.848 ms`, traces
`0.254670 MiB` Python peak allocation, and keeps CUDA allocation/reservation
at `0.0 MiB`.

The current `524288`-token protection run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-comparator-removed-default-nosample.json`
processed `524288` tokens at `6101.308 tokens/sec`, with
`last_tick_duration_ms=17.730`, trainer profile total `0.126931 ms/token`,
prewarm `277.363 s`, bounded `12/65536` route rows, `10` output candidates,
`65526` cached transition rows, `state_transition_runs_all_columns=false`,
native sequence-loop and burst-replay failure counts `0`, no observed
before/after contention (`cpu max=42%`, `gpu max=10%`), and RTX 3060 memory
`2049->2050 MiB`.

## Revisit Only If

A future audit policy proves that a wider applied-synapse source window is
selected deliberately, keeps archival metadata CPU-resident, avoids hidden
replay-text reasoning and live-tick/every-token work, blocks exact review when
incomplete, and preserves repeated 6k-ish long-run throughput. Do not restore
repo-local executable full-scan comparators for this retired path.
