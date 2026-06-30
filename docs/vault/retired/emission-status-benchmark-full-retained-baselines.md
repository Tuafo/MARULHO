---
type: retired
status: active
related_code:
  - ../../../src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py
  - ../../../src/marulho/evaluation/status_replay_path_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks: []
---

# Emission/Status Benchmark Full-Retained Baselines

## Retired

The benchmark-local full-retained emission-review policy/design comparator and
full-retained Runtime Truth replay-path status projection comparators are
removed from executable benchmark code.

## Replacement

Active benchmark reports assert the maintained bounded source-window paths:

- emission-review policy/design uses `16` reviewed emissions plus `16`
  internal readout rows, CPU archival/score placement, seeded-top quality, and
  `retired_full_retained_emission_review_policy_absence`.
- status replay projection uses the three maintained status source windows,
  seeded latest-hash quality, and
  `retired_full_retained_status_projection_absence`.

The full-retained comparators remain documented as retired history only; no
repo-local retirement-only guard test remains.

## Evidence

External local reports under
`..\..\MARULHO_reports\bounded_replay_window_20260623\` passed:

- `snn-emission-review-replay-policy-legacy-baseline-removed.json`: `1.427000 ms`
  mean, `2.192300 ms` p95, `8` candidates, `128x` source-work estimate,
  `0.045339 MiB` traced Python peak, CUDA available but unused for archival
  metadata.
- `status-replay-path-legacy-baseline-removed.json`: `1.032916 ms` combined
  mean, `80` bounded rows instead of the removed `10240`-row estimate, `128x`
  source-work estimate, `0.086983 MiB` traced Python peak, CUDA available but
  unused for archival metadata.
- `hotpath-active-pressure-65536-524288-i32-emission-status-legacy-baselines-removed-default-nosample.json`:
  `6518.530 tokens/sec`, `train_compute=0.124903 ms/token`, bounded
  `12/65536` route rows, `65526` cached rows, flat `1747 MiB` RTX 3060 memory,
  no observed contention, no in-window environment sampling, and zero
  graph/native sequence failures.

## Reopen Gate

Do not reintroduce these comparators into repo-local benchmark modules. If a
future paper or migration needs a full-retained control, keep it in an external
diagnostic script or notebook and require explicit source budget, device
placement, quality target, and long-run hot-path protection evidence.
