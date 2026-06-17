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
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_recall_uses_bucket_routing_keys tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_global_replay_selection_retires_zero_pressure_window tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_uses_anchor_bucket_replay_window_report tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_without_anchors_blocks_global_replay_mutation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_anchor_zero_pressure_blocks_global_replay_mutation tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_sleep_replay_selection_report tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report -q`
- Synthetic replay selector:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-selection.json`
- Hot-path protection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 360 --sample-interval-seconds 0.25 --host-truth-sync-interval-tokens 32`

## Latest Known Result

2026-06-17 bounded replay-window selection and recall add measured sleep/replay
surfaces, not an always-on memory path. `DualMemoryStore` keeps archival storage,
selection scoring, and associative recall evidence on CPU, reports
`runs_live_tick=false`, and exposes whether replay selection/recall used a
bucket-indexed candidate window or the explicit global slow-path scorer. Deep
sleep consolidation now requires an anchor-bucket scope; no-anchor or
zero-pressure bucket windows block global replay mutation instead of falling
back to a full slow-memory score scan.

The synthetic selector report at
`reports/bounded_replay_window_20260617/synthetic-selection.json` produced:

| Trial | Replay updates | Bounded cycles | Global fallback cycles | Recall gate | Prototype gate | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `bounded_positive_pressure` | `64` | `4` | `0` | passed, input distance `5.960464477539063e-08` | failed | bounded stored-experience recall passes; prototype repair open |
| `bounded_zero_pressure_guard` | `0` | `4` | `0` | failed | failed | zero-pressure replay retired; global fallback blocked |
| `global_control` | `0` | `4` | `0` | failed | failed | no-anchor deep replay retired; global mutation blocked |

The retired behavior was replaying an arbitrary zero-score entry after a global
slow-path scan, or applying deep replay from an unanchored global-control path.
The current trainer records `unscoped_global_fallback_retired=true` with
`global_fallback_blocked_reason=bucket_window_zero_positive_replay_pressure` or
`no_anchor_bucket_scope_for_deep_replay`, so deep replay work is skipped unless
positive tag/PRP/consolidation pressure exists inside a bounded bucket window.

The current hot-path check used the 65536-column active-pressure checkpoint and
processed `131072` tokens at `6277.353 tokens/sec`, with
`train_compute=0.130380 ms/token`, `prepare_training=0.006136 ms/token`,
`finalize_total=0.006208 ms/token`, `tick_duration_ms.p95=20.123`, CUDA RTX
3060 selected, no observed contention, `route_input_rows_scored=12/65536`,
`state_transition_cached_count=65526`, and zero graph/native/sequence failures.
CPU max was `52%`, GPU utilization max `10%`, GPU memory utilization max `11%`,
and GPU memory stayed `1759 MiB` before/after the measured run. This keeps live
runtime throughput in
the existing same-day band while replay selection and recall remain slow-path
only.

Next gate: improve prototype reconstruction/grounding under bounded
positive-pressure replay without reintroducing zero-pressure rehearsal or
live-tick memory scans.
