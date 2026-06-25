---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/runtime_evidence.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/living_status.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../reports/bounded_replay_window_20260620/replay-dataset-runtime-trace-export-summary-source-window.json
  - ../../../reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-runtime-trace-export-summary-source-window-rerun.json
---

# Runtime Trace Export Replay Summary Full History

## Status

Retired from production status/export paths.

## Why Retired

Runtime trace export, living status, feedback summary, and replay-sample summary
could still read full retained trace or replay-sample histories before trimming
or filtering. That kept an archive-wide control-plane read shape beside the
bounded replay-dataset preview path.

## Replacement

- `bounded_runtime_trace_export_source_window.v1` over at most `50` retained
  runtime episode traces before endpoint filtering and trace-state lookup.
- `bounded_replay_sample_summary_source_window.v1` over at most `64` retained
  replay sample records.
- Bounded recent trace/action windows for living status and feedback summary.
- Bounded export sanitization through iterator slicing instead of
  materialize-then-trim.

The benchmark
`reports/bounded_replay_window_20260620/replay-dataset-runtime-trace-export-summary-source-window.json`
passed with `50/64` trace-export rows, `64/256` replay-sample summary rows,
`64/256` replay-sample link rows, `1024/4096` selected-candidate link rows, and
exact `50/50` selected target/export ID parity against the diagnostic bounded
window. All source windows report CPU archival/source placement, no live tick,
no every-token cadence, no replay text or hidden language reasoning, no
mutation/plasticity/training, and no GPU-resident archival metadata.

The accepted hot-path rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-runtime-trace-export-summary-source-window-rerun.json`
processed `524288` tokens at `6047.311 tokens/sec`, kept route scoring at
`12/65536` input rows with `65526` cached transition rows, recorded
`state_transition_runs_all_columns=false`, no observed contention, flat RTX 3060
memory at `1911 MiB`, and zero graph/native sequence failures.

## Revisit Condition

Only as an external/offline diagnostic with explicit source-size accounting, or
if a stronger indexed export/summary path improves operator evidence while
preserving source-window reports and long-run hot-path throughput.
