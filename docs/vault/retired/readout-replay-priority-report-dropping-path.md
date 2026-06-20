---
type: retired
status: active
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/runtime_facade.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json
---

# Readout Replay-Priority Report-Dropping Path

The evaluated replay artifact path no longer accepts a replay window selected
by `replay_priority(...)` without the priority selector's bounded source-window
report. `transition_memory_replay_artifact_proposal(...)` now carries
`replay_priority_source_window` and `replay_priority_source_window_hash`.

`ReplayController.record_evaluated_snn_transition_memory_replay_artifact(...)`
requires that source window to be bounded, CPU-resident for archival/scoring
metadata, free of global candidate/score scans, raw text, language reasoning,
live-tick work, every-token work, mutation, plasticity, and CUDA archival use.
Artifact, permit, and rollout-review hash recomputation includes
`replay_priority_source_window_hash`, so callers cannot drop or spoof the
selection report after proposal time.

The focused benchmark
`reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json`
passed with replay-priority source window `1/32`, persisted
`replay_priority_source_window_hash`, `0.421992 ms` mean verification latency,
`0.719700 ms` p95, `0.014385 MiB` traced Python peak, CPU archival/score
placement, and `0.0 MiB` CUDA allocation/reservation.

The first long protection run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding.json`
is rejected as primary evidence: `4662.031 tokens/sec` with observed GPU
contention. The accepted no-contention rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json`
processed `524288` tokens at `5937.908 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1943 MiB` RTX 3060 memory,
and zero graph/native sequence failures. This is same-band protection for
deleting a slow-path bypass, not a speed promotion or live-tick replay path.
