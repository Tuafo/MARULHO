---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/slow_memory_fixed_cadence_retirement_benchmark.py
  - ../../../src/marulho/evaluation/strong_capture_admission_cadence_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json
  - reports/bounded_replay_window_20260620/strong-capture-admission-cadence-after-fixed-cadence-retirement.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json
---

# Fixed-Cadence Slow-Memory Admission

Fixed-cadence slow-memory admission is retired as an executable archive-write
path. Retained `train_step` no longer archives ordinary
`slow_memory_archive_interval_tokens` hits. It records `cadence_deferred`
through the cognitive boundary controller instead.

The maintained live archive writes are:

- first-token retained/fallback admission
- selected strong-capture admission bounded by
  `slow_memory_archive_strong_capture_min_interval_tokens`

The retirement benchmark kept `1` first-token archive and removed `17`
projected fixed-cadence writes over `256` tokens. The refreshed strong-capture
benchmark still kept `17` selected strong archives versus `256` retired
every-strong writes, so useful STC-like selection remains without fixed cadence.

The accepted `524288`-token protection rerun stayed in the maintained band at
`6043.321 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, `2048` deferred cadence hits, zero graph/native sequence
failures, and flat RTX 3060 memory at `1958 MiB`. The first same-code run is
retained only as below-band variance evidence at `5758.051 tokens/sec`.

Reopen this path only if a selected replay/admission design improves measured
recall, grounding, or reconstruction and repeated long-run evidence proves the
live tick remains protected without scans, every-token admission, GPU-resident
archival metadata, or hidden language reasoning.
