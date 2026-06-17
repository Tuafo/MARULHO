---
type: benchmark
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
---

# Replay Cost

Replay selection, rehearsal, and artifact-review cost checks.

## Commands

- Focused tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_global_replay_selection_retires_zero_pressure_window tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_uses_anchor_bucket_replay_window_report tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_sleep_replay_selection_report -q`
- Synthetic replay selector:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-selection.json`
- Hot-path protection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 360 --sample-interval-seconds 0.25 --host-truth-sync-interval-tokens 32`

## Latest Known Result

2026-06-17 bounded replay-window selection adds a measured sleep/replay
selector, not an always-on memory path. `DualMemoryStore` keeps archival storage
and selection scoring on CPU, reports `runs_live_tick=false`, and exposes
whether a replay selection used a bucket-indexed candidate window or the
explicit global slow-path scorer.

The synthetic selector report at
`reports/bounded_replay_window_20260617/synthetic-selection.json` produced:

| Trial | Replay updates | Bounded cycles | Global fallback cycles | Reconstruction gate | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `bounded_positive_pressure` | `1` | `1` | `3` | failed | bounded pressure can select, but no quality promotion |
| `bounded_zero_pressure_guard` | `0` | `0` | `4` | failed | zero-pressure replay retired |
| `global_control` | `0` | `0` | `4` | failed | zero-pressure global replay retired |

The retired behavior was replaying an arbitrary zero-score entry after a global
slow-path scan. The current selector returns an empty report with
`fallback_reason=no_positive_global_scores`, so slow replay work is skipped
unless positive tag/PRP/consolidation/repair pressure exists.

The current hot-path check used the 65536-column active-pressure checkpoint and
processed `131072` tokens at `6275.851 tokens/sec`, with
`train_compute=0.130522 ms/token`, `prepare_training=0.006131 ms/token`,
`finalize_total=0.006080 ms/token`, `tick_duration_ms.p95=19.927`, CUDA RTX
3060 selected, no observed contention, `route_input_rows_scored=12/65536`,
`state_transition_cached_count=65526`, and zero graph/native/sequence failures.
This keeps live runtime throughput in the existing same-day band while replay
selection remains slow-path only.

Next gate: improve reconstruction/grounding under bounded positive-pressure
replay without reintroducing zero-pressure rehearsal or live-tick memory scans.
