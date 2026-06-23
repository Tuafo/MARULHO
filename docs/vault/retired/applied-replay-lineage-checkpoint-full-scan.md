---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/applied_replay_lineage.py
  - ../../../src/marulho/service/persistence.py
  - ../../../src/marulho/service/snn_language_plasticity_executor.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/applied-replay-lineage-checkpoint-summary.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-applied-replay-lineage-checkpoint-summary.json
---

# Applied Replay Lineage Checkpoint Full Scan

## Retired Path

`RuntimePersistence._applied_replay_lineage_checkpoint_summary(...)` used to
derive replay-regenerated synapse lineage by scanning and materializing
`synapse_provenance_by_key` during checkpoint save. Restore validation repeated
that derivation against hydrated plasticity state.

## Replacement

Replay regeneration now maintains
`snn_applied_replay_lineage_incremental_summary.v1` as CPU mutation-time
evidence. Replay-regenerated synapses add one row hash; non-replay overwrites
and pruning clear the affected row. Checkpoint summary and restore validation
read the maintained counts/digest with `source_record_scan_count=0` and
`full_provenance_scan=false`. Missing incremental state blocks exact validation
instead of rebuilding from provenance, and the active report no longer emits a
legacy-source compatibility field for that case.

## Evidence

`reports/bounded_replay_window_20260620/applied-replay-lineage-checkpoint-summary.json`
passed over `65536` replay-lineage rows: active summary read `0` provenance
records and matched the benchmark-local retired full-scan diagnostic. Active
mean latency was `0.065529 ms`; retired mean latency was `6766.639043 ms`.
Active traced Python peak was `0.001343 MiB`; CUDA allocation/reservation stayed
`0.0 MiB`.

The paired hot-path report
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-applied-replay-lineage-checkpoint-summary.json`
processed `524288` tokens at `5993.011 tokens/sec`, kept route scoring bounded
at `12/65536`, cached `65526` transition rows, reported no observed contention,
and recorded zero graph/native sequence failures.

## Revisit Condition

Do not reintroduce this production scan. Full provenance comparisons belong only
in benchmark-local diagnostics unless a future architecture replaces them with
a bounded indexed validator that proves quality benefit, CPU archival placement,
no hidden language reasoning, no live/every-token work, and repeated 6k-ish
hot-path protection. Do not reintroduce legacy-source compatibility fields as a
substitute for exact incremental summary evidence.
