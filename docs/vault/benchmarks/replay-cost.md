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
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json
  - reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json
---

# Replay Cost

Replay selection, rehearsal, and artifact-review cost checks.

## Commands

- Focused tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_recall_uses_bucket_routing_keys tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_global_replay_selection_retires_zero_pressure_window tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_uses_anchor_bucket_replay_window_report tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_without_anchors_blocks_global_replay_mutation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_anchor_zero_pressure_blocks_global_replay_mutation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_micro_sleep_refreshes_tags_without_weight_commit tests\test_memory_consolidation.py::MemoryConsolidationTests::test_micro_sleep_without_anchors_blocks_global_maintenance_refresh tests\test_memory_consolidation.py::MemoryConsolidationTests::test_repair_sleep_reanchors_prototypes_without_consolidation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_repair_sleep_without_anchors_blocks_global_repair_mutation tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_sleep_replay_selection_report tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report -q`
- Synthetic replay selector:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-selection-candidate-repair-bounded-micro.json`
- HF-backed replay recall:
  `PYTHONPATH=src python -m marulho.training.memory_consolidation_runner --task-a-train-tokens 512 --task-b-train-tokens 512 --eval-tokens 128 --n-columns 64 --column-latent-dim 64 --memory-capacity 512 --deep-sleep-replay-steps 32 --deep-sleep-candidate-pool 32 --task-boundary-consolidation-cycles 2 --consolidation-cycles 3 --no-plots --output-dir reports\bounded_replay_window_20260617\hf-recall-bounded-window`
- Hot-path protection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-bounded-micro.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.25 --host-truth-sync-interval-tokens 32`

## Latest Known Result

2026-06-17 bounded replay-window selection, recall, and candidate repair add
measured sleep/replay surfaces, not an always-on memory path. `DualMemoryStore`
keeps archival storage, selection scoring, and associative recall evidence on
CPU, reports `runs_live_tick=false`, and exposes whether replay
selection/recall used a bucket-indexed candidate window or the explicit global
slow-path scorer. Deep sleep consolidation now requires an anchor-bucket scope;
no-anchor or zero-pressure bucket windows block global replay mutation instead
of falling back to a full slow-memory score scan.

Positive-pressure deep replay commits through
`bounded_reconstruction_gated_candidate_repair`: selected replay traces are
de-duplicated, candidate columns are bounded route candidates plus explicit
stored-bucket fallback candidates, and prototype repair commits only when the
local replay-window reconstruction metric improves. The old stored-bucket
modulated replay mutation is retired for deep replay.

Emergency repair replay is now bounded by the same anchor-bucket scope.
Anchored repair reports `bounded_repair_reanchor`; no-anchor repair records
`global_fallback_blocked_reason=no_anchor_bucket_scope_for_repair_replay` and
leaves prototypes unchanged instead of using the global repair scorer.

Micro maintenance is now bounded by the same anchor-bucket scope and no longer
calls the competitive replay/plasticity path with zero learning rates. Anchored
micro refresh reports `bounded_micro_maintenance_refresh`, selects through
`bucket_indexed_candidate_window`, updates CPU metadata only, and sets
`sleep_replay_bypasses_competitive_process=true`; no-anchor micro refresh
records `global_fallback_blocked_reason=no_anchor_bucket_scope_for_micro_replay`
and leaves replay counters, tags, prototypes, and input weights unchanged.

The synthetic selector report at
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json`
produced:

| Trial | Replay commits | Bounded cycles | Global fallback cycles | Recall gate | Prototype gate | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `bounded_positive_pressure` | `6` | `4` | `0` | passed, input distance `5.960464477539063e-08` | passed, relative degradation `0.0467838377` | bounded recall plus candidate repair improves reconstruction |
| `bounded_zero_pressure_guard` | `0` | `4` | `0` | failed | failed | zero-pressure replay retired; global fallback blocked |
| `global_control` | `0` | `4` | `0` | failed | failed | no-anchor deep replay retired; global mutation blocked |

The positive arm committed `6` repairs across `3` mutating cycles, rejected
`14` non-improving commits, tried `189` candidate-column repairs, and kept the
maximum candidate union to `11` columns for `5` unique replay traces. The local
quality metric was
`mean_one_minus_best_similarity_over_selected_replay_routing_keys` over
`selected_replay_window_candidate_columns`; total quality delta was
`0.00038640499114990235`. Task-A reconstruction moved from `0.0052170157`
after Task B to `0.0034434795` after consolidation, and overlap was
`0.8981397152`.

The retired behavior was replaying an arbitrary zero-score entry after a global
slow-path scan, or applying deep replay from an unanchored global-control path.
The current trainer records `unscoped_global_fallback_retired=true` with
`global_fallback_blocked_reason=bucket_window_zero_positive_replay_pressure` or
`no_anchor_bucket_scope_for_deep_replay`, so deep replay work is skipped unless
positive tag/PRP/consolidation pressure exists inside a bounded bucket window.

The current longer hot-path check used the 65536-column active-pressure
checkpoint and processed `262144` tokens at `6306.507 tokens/sec`, with
`train_compute=0.129511 ms/token`, `prepare_training=0.006408 ms/token`,
`finalize_total=0.006255 ms/token`, `tick_duration_ms.p95=20.176`, CUDA RTX
3060 selected, no observed contention, `route_input_rows_scored=12/65536`,
`state_transition_cached_count=65526`, and zero graph/native/sequence failures.
CPU max was `18%`, GPU utilization max `11%`, GPU memory utilization max `11%`,
and GPU memory stayed `1871 MiB` before/after the measured run. This keeps live
runtime throughput in the existing same-day band while replay selection, recall,
and candidate repair remain slow-path only.

After bounding emergency repair mode, the current-tree hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json`
processed `131072` tokens at `6252.073 tokens/sec`, with
`train_compute=0.129794 ms/token`, `prepare_training=0.006361 ms/token`,
`finalize_total=0.006213 ms/token`, `tick_duration_ms.p95=20.060`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, and no observed contention.

After bounding micro maintenance and removing the zero-LR competitive refresh,
the longer current-tree hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json`
processed `262144` tokens at `6332.439 tokens/sec`, with
`train_compute=0.129001 ms/token`, `prepare_training=0.006330 ms/token`,
`finalize_total=0.006198 ms/token`, `tick_duration_ms.p95=20.048`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, and no observed contention. CPU max was
`28%`, GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU
memory stayed `1881 MiB` before/after measurement.

The HF-backed recall report
`reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json`
adds less-synthetic stored-experience recall evidence to the consolidation
runner. `bounded_replay_window_hf_recall_summary.v1` snapshots stored Task-A
anchor-window patterns and evaluates recall after Task B and after consolidation
without using replay text as hidden reasoning. The after-consolidation recall
gate passed with `3` queries, `3` candidate buckets, `3` scored CPU entries,
`max_candidates=32`, `mean_input_pattern_distance=0.0`,
`mean_routing_key_distance=1.9868214925130207e-08`, `runs_live_tick=false`, and
`mutates_runtime_state=false`; observed per-query recall latency was about
`0.82-1.04 ms`. The overall memory-consolidation gate remained false because
Task-A reconstruction worsened from `0.0137995831` after Task B to
`0.0200071791` after consolidation (`task_a_recovery_nonnegative=false`), even
though it stayed better than the Task-A-after-A baseline `0.0571274995` and
overlap remained `0.9909951999`.

Next gate: repeat quality on a less synthetic grounding/prediction target and
turn the HF recall evidence into a reconstruction/grounding improvement without
widening live-tick memory work.
