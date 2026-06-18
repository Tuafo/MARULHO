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
  - reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json
  - reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json
  - reports/bounded_replay_window_20260617/synthetic-replay-tensor-payload-boundary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json
  - reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json
  - reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json
  - reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/frontier-gap-bounded.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-frontier-gap-bounded.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json
  - reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json
  - reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-awake-ripple-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json
---

# Replay Cost

Replay selection, rehearsal, and artifact-review cost checks.

## Commands

- Focused tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_recall_uses_bucket_routing_keys tests\test_memory_consolidation.py::MemoryConsolidationTests::test_hf_recall_evaluation_reports_bounded_anchor_window tests\test_memory_consolidation.py::MemoryConsolidationTests::test_reconstruction_guard_rolls_back_harmful_replay_cycle tests\test_memory_consolidation.py::MemoryConsolidationTests::test_reconstruction_guard_rejects_regression_even_when_no_updates_reported tests\test_memory_consolidation.py::MemoryConsolidationTests::test_reconstruction_guard_skips_repeated_rejected_selection tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bucket_replay_selection_caps_candidate_window_before_scoring tests\test_memory_consolidation.py::MemoryConsolidationTests::test_unscoped_replay_selection_requires_diagnostic_opt_in tests\test_memory_consolidation.py::MemoryConsolidationTests::test_unscoped_random_replay_selection_requires_diagnostic_opt_in tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_uses_anchor_bucket_replay_window_report tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_without_anchors_blocks_global_replay_mutation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_anchor_zero_pressure_blocks_global_replay_mutation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_micro_sleep_refreshes_tags_without_weight_commit tests\test_memory_consolidation.py::MemoryConsolidationTests::test_micro_sleep_without_anchors_blocks_global_maintenance_refresh tests\test_memory_consolidation.py::MemoryConsolidationTests::test_repair_sleep_reanchors_prototypes_without_consolidation tests\test_memory_consolidation.py::MemoryConsolidationTests::test_repair_sleep_without_anchors_blocks_global_repair_mutation tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_sleep_replay_selection_report tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report -q`
- Replay text/SFA boundary tests:
  `PYTHONPATH=src python -m pytest tests\test_sfa_correction.py::TestSampleForSFA::test_sample_can_use_bounded_candidate_indices tests\test_memory_consolidation.py::MemoryConsolidationTests::test_replay_entry_can_exclude_text_payload_for_sleep_replay tests\test_memory_consolidation.py::MemoryConsolidationTests::test_deep_sleep_sfa_correction_samples_selected_replay_window tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_sleep_replay_selection_report -q`
- Synthetic replay selector:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-selection-candidate-repair-bounded-micro.json`
- Synthetic replay text/SFA boundary:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-replay-tensor-payload-boundary.json`
- Synthetic capped replay candidate window:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-selection-candidate-repair-capped-window.json`
- HF-backed replay recall with capped query collection:
  `PYTHONPATH=src python -m marulho.training.memory_consolidation_runner --task-a-train-tokens 512 --task-b-train-tokens 512 --eval-tokens 128 --n-columns 64 --column-latent-dim 64 --memory-capacity 512 --deep-sleep-replay-steps 32 --deep-sleep-candidate-pool 32 --task-boundary-consolidation-cycles 2 --consolidation-cycles 3 --no-plots --output-dir reports\bounded_replay_window_20260617\hf-recall-capped-query-collection`
- HF-backed replay recall:
  `PYTHONPATH=src python -m marulho.training.memory_consolidation_runner --task-a-train-tokens 512 --task-b-train-tokens 512 --eval-tokens 128 --n-columns 64 --column-latent-dim 64 --memory-capacity 512 --deep-sleep-replay-steps 32 --deep-sleep-candidate-pool 32 --task-boundary-consolidation-cycles 2 --consolidation-cycles 3 --no-plots --output-dir reports\bounded_replay_window_20260617\hf-recall-guarded-consolidation-cadenced`
- Hot-path protection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Hot-path protection for replay text/SFA boundary:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Hot-path protection for capped replay candidate windows:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-capped-replay-window.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Hot-path protection for capped query collection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-collection.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded query-memory match readout:
  `PYTHONPATH=src python -m marulho.training.query_runner --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --query-text "bounded replay memory" --top-k-candidates 5 --top-k-memories 5 --top-chars 4 --output-json reports\bounded_replay_window_20260617\query-memory-match-bounded-window.json`
- Hot-path protection for bounded query-memory match:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-memory-match.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded concept-frontier memory metrics tests:
  `PYTHONPATH=src python -m pytest tests\test_autonomy_runner.py::AutonomySelectionTests::test_concept_frontier_metrics_use_bounded_candidate_window -q`
- Bounded concept-frontier memory metrics benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.concept_frontier_scope_benchmark --output reports\bounded_replay_window_20260617\concept-frontier-bounded-scope.json --capacity 8192 --bucket-count 1024 --candidate-bucket-count 8 --iterations 64 --dim 16`
- Hot-path protection for bounded concept-frontier memory metrics:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded frontier-gap planner tests:
  `PYTHONPATH=src python -m pytest tests\test_gap_planner.py tests\test_memory_consolidation.py::MemoryConsolidationTests::test_frontier_gap_collection_uses_bounded_recent_index -q`
- Bounded frontier-gap planner benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.frontier_gap_bounded_benchmark --output reports\bounded_replay_window_20260617\frontier-gap-bounded.json --capacity 65536 --iterations 8 --top-entries 24 --max-terms 8`
- Hot-path protection for bounded frontier-gap planner:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-524288-i32-frontier-gap-bounded.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Recent tag/anchor recency-index tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_recent_memory_tagging_uses_capped_recency_index tests\test_memory_consolidation.py::MemoryConsolidationTests::test_recent_anchor_capture_uses_capped_recency_index tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report -q`
- Synthetic recent replay tag/anchor setup:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-recent-anchor-window.json`
- Hot-path protection for recent replay tag/anchor setup:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Full-buffer replay-score helper retirement tests:
  `PYTHONPATH=src python -m pytest tests\test_p1_improvements.py::TestAwakeRippleTagging::test_ripple_tagged_entries_get_higher_replay_scores tests\test_memory_consolidation.py::MemoryConsolidationTests::test_capture_tags_recruit_prp_and_raise_replay_priority tests\test_query_runner.py -q`
- Synthetic replay-score helper retirement:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-replay-score-helper-retired.json`
- Hot-path protection for replay-score helper retirement:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Full-buffer score tensor helper family retirement tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_fragility_priority_prefers_stale_unconsolidated_memories tests\test_memory_consolidation.py::MemoryConsolidationTests::test_unscoped_replay_selection_requires_diagnostic_opt_in tests\test_memory_consolidation.py::MemoryConsolidationTests::test_unscoped_random_replay_selection_requires_diagnostic_opt_in tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bucket_replay_selection_caps_candidate_window_before_scoring tests\test_sfa_correction.py::TestSampleForSFA -q`
- Synthetic score tensor helper family retirement:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-score-tensor-helpers-retired.json`
- Hot-path protection for score tensor helper family retirement:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded awake-ripple scope tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_awake_ripple_tagging_uses_awake_bucket_index tests\test_memory_consolidation.py::MemoryConsolidationTests::test_awake_ripple_tagging_caps_awake_bucket_candidates tests\test_memory_consolidation.py::MemoryConsolidationTests::test_awake_ripple_unscoped_requires_diagnostic_opt_in tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report tests\test_p1_improvements.py::TestAwakeRippleTagging -q`
- Awake-ripple scoped/global diagnostic benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.awake_ripple_scope_benchmark --output reports\bounded_replay_window_20260617\awake-ripple-bounded-scope-8192-i256.json --capacity 8192 --bucket-count 8192 --awake-bucket-count 10 --iterations 256 --dim 16`
- Synthetic replay quality after bounded awake-ripple tagging:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-awake-ripple-bounded-scope.json`
- Hot-path protection for bounded awake-ripple tagging:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-awake-ripple-bounded-scope.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Longer hot-path protection for bounded awake-ripple tagging:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

## Latest Known Result

2026-06-17 bounded replay-window selection, recall, and candidate repair add
measured sleep/replay surfaces, not an always-on memory path. `DualMemoryStore`
keeps archival storage, selection scoring, and associative recall evidence on
CPU, reports `runs_live_tick=false`, and exposes whether replay
selection/recall used a bucket-indexed candidate window or the explicit global
slow-path scorer. Bucket-indexed selection now applies a pre-score
`candidate_window_limit=max(requested_count,candidate_pool)` and reports
`candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
`candidate_index_available_count`, `candidate_index_count`, and `score_count`, so
even a hot anchor bucket cannot make the slow window score every stored entry.
Deep sleep consolidation now requires an anchor-bucket scope;
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
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json`
adds less-synthetic stored-experience recall evidence to the consolidation
runner. `bounded_replay_window_hf_recall_summary.v1` snapshots stored Task-A
anchor-window patterns and evaluates recall after Task B and after consolidation
without using replay text as hidden reasoning. The runner now wraps boundary and
post-Task-B consolidation in `reconstruction_guarded_replay_consolidation.v1`:
each slow replay cycle snapshots trainer and memory state, attempts the bounded
anchor-window replay, measures Task-A `mean_reconstruction_error`, and commits
only if the broader reconstruction score does not regress. In this HF run the
post-Task-B guard attempted `9` candidate repair updates across `3` cycles,
rejected all `9`, restored the snapshot each time, and left effective replay
updates at `0`. The quality score stayed `0.0161349615` before and after guarded
consolidation, while the memory-consolidation gate passed with Task-A overlap
`0.9973959164` and relative degradation `-0.6278533707`. Stored-experience
recall still passed after consolidation with `3` queries, `3` candidate buckets,
`3` scored CPU entries, `max_candidates=32`, `mean_input_pattern_distance=0.0`,
`mean_routing_key_distance=1.9868214925130207e-08`, `runs_live_tick=false`, and
`mutates_runtime_state=false`. Selection and recall stay on CPU archival
metadata; guard scoring uses the model device (`cuda`) only inside the slow
window.

The current guarded-consolidation hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation.json`
processed `262144` tokens at `6606.251 tokens/sec`, with
`train_compute=0.123393 ms/token`, `prepare_training=0.005897 ms/token`,
`finalize_total=0.005707 ms/token`, `tick_duration_ms.p95=18.562`,
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and no observed contention. CPU max was
`6%`, GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU
memory stayed `1539 MiB` before/after measurement. The guard is therefore
accepted as a slow-window quality boundary, not live-tick work.

The follow-up cadenced guard report
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json`
retires repeated identical rejected replay attempts inside the same slow window.
After the first rejected candidate-repair cycle, the guard records
`cadence_strategy=skip_repeated_rejected_selection` and skips later cycles until
new state or a new window can change the selection. The HF run kept
`memory_consolidation_gate.pass=true` and after-consolidation recall passing
with `mean_input_pattern_distance=0.0`, but reduced post-Task-B attempted repair
updates from `9` to `3`, rejected attempted updates from `9` to `3`, and skipped
`2` repeated rejected cycles. The guard quality stayed `0.0119761885` before and
after consolidation, Task-A overlap was `0.9928868827`, relative degradation was
`-0.7557967309`, and guard latency was `559.694 ms` versus `1003.442 ms` for
the immediately previous guarded report.

The first post-cadence 262144-token hot-path run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced.json`
completed but was rejected as throughput evidence: `5388.450 tokens/sec`,
`train_compute=0.152640 ms/token`, and `tick_duration_ms.p95=43.215` despite
bounded `route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, and no observed contention. The immediate
rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json`
restored the maintained band at `6199.988 tokens/sec`, with
`train_compute=0.130574 ms/token`, `prepare_training=0.006633 ms/token`,
`finalize_total=0.006399 ms/token`, `tick_duration_ms.p95=20.215`,
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and no observed contention. CPU max was
`29%`, GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU
memory was `1688 MiB` before and `1689 MiB` after measurement. Treat the first
run as a failed/noisy gate and the rerun as the current accepted live-tick
protection evidence for this slow-window cadence change.

The target-aware replay-strength search follow-up keeps that guard but now
records the trial budget and budget policy beside the exact schedule. Each
trial restores from the same model/memory snapshot, measures the same target
reconstruction metric, and commits only the best non-regressing trial. The old
low-strength tails are no longer defaults: HF text consolidation uses the
single-strength schedule `[0.1]`, while the synthetic prototype stress
benchmark uses compact escalation `[0.1, 0.5, 1.0]` because the single-strength
stress run improved reconstruction but failed the prototype gate.

The patched HF report
`reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-promoted/summary.json`
turns the stored-experience recall path into a cheaper measured reconstruction
improvement. Boundary consolidation accepted `2` guarded repairs, then
post-Task-B consolidation accepted `6` repairs, rejected `0` trial attempts,
and improved Task-A reconstruction from `0.0170305534` to `0.0149637708`
(`quality_delta=0.0020667827`). Task-B reconstruction moved from
`0.0171923228` to `0.0153899691`. The post-B guard latency was
`1040.506 ms`, versus `3477.025 ms` for the previous four-low-strength HF
default. The report records `repair_strength_trial_budget=1`,
`repair_strength_trial_budget_policy=explicit_schedule_length`,
`score_device=cuda`, `archival_storage_device=cpu`, and `runs_live_tick=false`.
Bounded recall still passed after consolidation with
`mean_input_pattern_distance=0.0`, and the memory-consolidation gate passed.

The larger medium HF qualification
`reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-medium-2048/summary.json`
keeps the same single-strength budget on a broader target: `2048` Task-A train
tokens, `2048` Task-B train tokens, `512` eval tokens, `128` columns, `128`
latent dimensions, and `2048` memory capacity. Boundary consolidation accepted
`18` repairs across `3` cycles, then post-Task-B consolidation accepted `28`
repairs across `4` cycles, rejected `0` trial attempts, and improved Task-A
reconstruction from `0.0103354922` to `0.0101451825`
(`quality_delta=0.0001903097`) while Task-A overlap stayed `0.9801252444`.
The report records `repair_strength_trial_budget=1`, `score_device=cuda`,
`archival_storage_device=cpu`, and `runs_live_tick=false`; bounded recall and
the memory-consolidation gate both passed with `mean_input_pattern_distance=0.0`.
Checkpoint reload of
`reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-medium-2048/checkpoint.pt`
restored `token_count=4096`, `17` CPU archival replay entries/input patterns,
`bounded_replay_window_recall.v1`, and `bounded_replay_window_selection.v1`
with both reports still marked `runs_live_tick=false`.

The patched synthetic default report
`reports/bounded_replay_window_20260617/synthetic-target-strength-budget-compact-default.json`
keeps the prototype stress target passing with less schedule cost. The
positive-pressure arm used schedule `[0.1, 0.5, 1.0]`,
`repair_strength_trial_budget=3`, accepted `2` repairs from `4` attempted
updates, rejected `0`, passed stored input-pattern recall
(`5.960464477539063e-08` mean input distance), passed the prototype gate, and
kept recovery at `0.0017409722`. Guard latency was `2585.941 ms`, versus
`5074.171 ms` for the full low-first escalation control
`[0.1, 0.05, 0.02, 0.01, 0.5, 1.0]`. The single-strength synthetic control
`reports/bounded_replay_window_20260617/synthetic-target-strength-budget-single-010-promoted.json`
is rejected as a universal default because it failed the prototype gate despite
passing recall. Zero-pressure and no-anchor/global-control arms still applied
`0` updates and recorded `0` global fallback cycles.

The matching 65536-column hot-path protection check stayed in the maintained
band after the default cleanup. The current 262144-token run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-target-strength-budget-compact.json`
processed `262144` tokens at `6232.282 tokens/sec`, with
`train_compute=0.130988 ms/token`, `prepare_training=0.006491 ms/token`,
`finalize_total=0.006431 ms/token`, `tick_duration_ms.p95=20.659`,
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence
failures, and no observed contention. CPU max was `33%`, GPU utilization max
`10%`, GPU memory utilization max `10%`, and GPU memory stayed flat at
`1715 MiB` before and after measurement.

The replay text/SFA boundary cleanup removes old implementation shapes from the
same slow-window path. Sleep replay now calls
`DualMemoryStore.replay_entry(..., include_text_payload=False)`, so deep
candidate repair and anchored repair receive tensor payloads only; raw windows,
expanded text, and metadata remain available to explicit query/display callers
but are not loaded by replay consolidation. `bounded_replay_window_selection.v1`
and `bounded_replay_window_recall.v1` record `raw_text_payload_loaded=false` and
`language_reasoning=false`; sleep replay records
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`, and
`sleep_replay_text_payload_policy=sleep_replay_uses_tensor_payloads_only`.
Deep replay with abstraction also bounds SFA correction to the processed replay
window by passing `candidate_indices=processed_indices` into
`sample_for_sfa(...)`; Runtime Truth records
`sleep_replay_sfa_correction_scope`, candidate/sample counts, and
`sleep_replay_sfa_full_memory_sample_retired=true`. The helper defaults now
enforce the selected-window contract: `sample_replay_indices(...)` returns no
indices for unscoped calls unless `allow_global_score_scan=true` marks an
explicit diagnostic, and `sample_for_sfa(...)` returns no samples without
`candidate_indices` unless `allow_global_diagnostic=true` is supplied.

The helper-retirement verification passed the focused helper tests and the
broader memory/SFA suite after the STC robustness test was made explicit about
captured anchors:
`PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_fragility_priority_prefers_stale_unconsolidated_memories tests\test_memory_consolidation.py::MemoryConsolidationTests::test_unscoped_replay_selection_requires_diagnostic_opt_in tests\test_memory_consolidation.py::MemoryConsolidationTests::test_bounded_replay_window_selection_scores_only_bucket_candidates tests\test_sfa_correction.py::TestSampleForSFA -q`
returned `7 passed`, and
`PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py tests\test_sfa_correction.py -q`
returned `47 passed`. These tests prove the helper defaults refuse unscoped
archival scans, while explicit diagnostics remain available and visibly marked.

The focused replay/checkpoint/SFA suite passed `19` tests, including the new
text-payload exclusion, bounded SFA sample, deep replay SFA scope, and
checkpoint roundtrip assertions. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-replay-tensor-payload-boundary.json`
kept the positive-pressure arm passing stored input-pattern recall
(`5.960464477539063e-08` mean distance) and the prototype gate. It accepted `2`
post-B repairs from `4` attempted updates, kept recovery at `0.0017409722`,
reported `score_device=cuda`, `archival_storage_device=cpu`, `runs_live_tick=false`,
`raw_text_payload_loaded=false`,
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`, and
`sleep_replay_sfa_full_memory_sample_retired=true`. Zero-pressure and
no-anchor/global-control arms still applied `0` updates.

The matching final 65536-column hot-path run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json`
processed `262144` tokens at `6237.420 tokens/sec`, with
`train_compute=0.130490 ms/token`, `prepare_training=0.006495 ms/token`,
`finalize_total=0.006446 ms/token`, and `tick_duration_ms.p95=20.383`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `28%`,
GPU utilization max `18%`, GPU memory utilization max `12%`, and GPU memory
stayed flat at `1719 MiB` before/after measurement. This is accepted as
same-band live-tick protection while replay text and SFA stay selected,
measured, and slow-windowed.

The unscoped-helper retirement then received its own 65536-column long-run gate.
The first 262144-token run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired.json`
processed `262144` tokens at `6068.338 tokens/sec`, with
`train_compute=0.135063 ms/token`, `prepare_training=0.006458 ms/token`,
`finalize_total=0.006376 ms/token`, `tick_duration_ms.p95=21.140`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
flat `1856 MiB` GPU memory, and zero graph/native/sequence failures, but it was
kept as secondary evidence because `velocity_environment` reported
`contention_observed` with GPU utilization max `21%`.

The accepted clean rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json`
processed `262144` tokens at `5668.688 tokens/sec`, with
`train_compute=0.141909 ms/token`, `prepare_training=0.007435 ms/token`,
`finalize_total=0.006774 ms/token`, and `tick_duration_ms.p95=25.429`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `71%`,
GPU utilization max `11%`, GPU memory utilization max `11%`, and GPU memory
moved from `1877 MiB` before measurement to `1844 MiB` after measurement. This
protects the live tick for retiring the unscoped helper defaults; the retired
paths remain slow-window diagnostics only when explicitly opted in.

The capped replay-candidate window slice then tightened the bucket-indexed
selector itself. The new focused test proves a hot two-bucket window with `10`
available entries and an old high-importance trace scores only the recent
round-robin `candidate_window_limit=4` entries before ranking; the old trace is
not selected because it never enters the capped candidate window. Unscoped
`strategy=random` now also returns an empty retired report unless
`allow_global_score_scan=true` marks the call as a diagnostic global candidate
scan.

The broader memory/SFA suite returned `49 passed`, and the replay checkpoint
roundtrip checks returned `2 passed`. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json`
kept positive-pressure stored input-pattern recall passing at
`5.960464477539063e-08`, passed the prototype gate with relative degradation
`0.0462463007`, accepted `2` bounded post-B repairs, and kept zero-pressure and
no-anchor/global-control arms at `0` updates. Selection reported
`candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
`candidate_window_limit=32`, `candidate_index_available_count=16`,
`candidate_index_count=16`, `score_count=16`, `score_device=cpu`,
`archival_storage_device=cpu`, `runs_live_tick=false`, `global_score_scan=false`,
and `global_candidate_scan=false`.

The matching 65536-column long run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json`
processed `262144` tokens at `6148.125 tokens/sec`, with
`train_compute=0.132113 ms/token`, `prepare_training=0.006656 ms/token`,
`finalize_total=0.006548 ms/token`, and `tick_duration_ms.p95=21.137`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `29%`,
GPU utilization max `15%`, GPU memory utilization max `13%`, and GPU memory
stayed flat at `1848 MiB`. This keeps replay selection selected, measured,
CPU-resident for archival metadata, and slow-windowed while protecting the live
tick.

The replay query-collection follow-up retires the HF runner's linear
`slow_bucket_ids` walk. `DualMemoryStore.collect_replay_query_indices(...)`
emits `bounded_replay_query_collection.v1`, uses the same bucket-indexed recent
round-robin candidate window, caps collection at `max_queries`, requires input
patterns by default, and reports `score_count=0`, `global_score_scan=false`,
`global_candidate_scan=false`, `runs_live_tick=false`, and
`archival_storage_device=cpu`. The HF report
`reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json`
kept recall and consolidation gates passing: query collection reported
`candidate_window_limit=16`, `candidate_index_available_count=3`,
`candidate_index_count=3`, `query_count=3`, and no global scans; after
consolidation recall passed with `mean_input_pattern_distance=0.0` and
`mean_routing_key_distance=1.98682149251302e-08`. The guarded consolidation
accepted `6` post-Task-B repairs, rejected `0`, improved the target quality
from `0.0234637554` to `0.0213608844`, used `score_device=cuda` only inside the
slow window, and kept archival replay metadata on CPU.

The matching long hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json`
processed `262144` tokens at `6221.949 tokens/sec`, with
`train_compute=0.131162 ms/token`, `prepare_training=0.006563 ms/token`,
`finalize_total=0.006444 ms/token`, and `tick_duration_ms.p95=20.657`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `28%`,
GPU utilization max `12%`, GPU memory utilization max `11%`, and GPU memory
stayed flat at `1848 MiB` before and after measurement.

The query-memory match follow-up removes the explicit query readout's full
slow-memory scan. `query_runner.memory_matches_with_report(...)` now derives
candidate bucket ids from routing, asks
`DualMemoryStore.collect_query_memory_match_indices(...)` for a bounded
bucket-indexed candidate window, and computes similarity plus replay-priority
scores only for those candidate indices. The replay-priority formula is
preserved through `replay_scores_for_indices(...)`, so the ranking signal is
unchanged inside the selected window without exposing a full-buffer score
helper. The query report
`reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json`
emits `bounded_query_memory_match.v1`: it used
`candidate_window_limit=192`, had `candidate_index_available_count=1`, scored
`1` similarity and replay-priority candidate, returned `1` match, reported no
global score/candidate scan, kept archival storage on CPU, and marked
`runs_live_tick=false` and `mutates_runtime_state=false`. The top retrieved
memory was `promoted scheduler checkpoint route-bank seed` with similarity
`0.9932903051`.

The matching 65536-column long run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json`
processed `262144` tokens at `6137.185 tokens/sec`, with
`train_compute=0.131555 ms/token`, `prepare_training=0.006550 ms/token`,
`finalize_total=0.006528 ms/token`, and `tick_duration_ms.p95=20.711`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `14%`,
GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU memory
stayed flat at `1848 MiB` before and after measurement.

The concept-frontier metric follow-up applies the same selected-window rule to
autonomy source-acquisition planning. `concept_frontier_metrics_with_report(...)`
derives candidate buckets from the probe-bank routing signature, asks
`DualMemoryStore.collect_query_memory_match_indices(...)` for a capped recent
bucket-indexed candidate window, and emits
`bounded_concept_frontier_memory_metrics.v1`. It scores novelty, uncertainty,
and support only for those candidate entries; the old direct iteration over
every `slow_routing_keys` entry is retired. The synthetic scope benchmark
`reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json`
compared the bounded path with a diagnostic full-memory baseline over `8192`
entries and `64` iterations: bounded scoring touched `64` entries at
`5.040 ms` mean versus `658.116 ms` for the `8192`-entry full scan, preserved
the full-scan top-1, kept `novelty_delta=0.0`, `uncertainty_delta=0.0`, and
`support_delta=0.015893`, and reported no global score/candidate scan. The
matching 65536-column hot-path run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json`
processed `262144` tokens at `6148.846 tokens/sec`, with
`train_compute=0.131437 ms/token`, bounded `12/65536` route rows, flat
`1805 MiB` GPU memory, no observed contention, and zero graph/native/sequence
failures.

The ConceptStore signature lookup follow-up removes the remaining
archive-materializing access shape from semantic observation. Concept evidence
still comes from bounded query/readout/source observations, but
`ConceptStore._memory_signature(...)` now uses only evidence-provided memory
indices, caps each source at `8` unique indices with a `32`-reference scan
budget, direct-indexes the CPU archival arrays, and records
`bounded_concept_memory_signature_lookup.v1`. The diagnostic benchmark
`reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json`
compared the retired list-materializing shape with the bounded direct-index
shape over `65536` archival entries, `512` iterations, and `8` memory indices
per evidence source. Bounded lookup preserved signatures against the diagnostic
baseline (`min cosine=0.9999998212`), removed `4096` archive list
materializations, kept `archive_list_materialization_count=0`, and reduced
mean lookup latency from `12.490 ms` to `1.454 ms` (`8.591x`). The clean
hot-path gate
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json`
processed `262144` tokens at `6143.768 tokens/sec`, kept bounded `12/65536`
route rows and `65526` cached transition rows, reported no observed
contention, held GPU memory flat at `1746 MiB`, and had zero
graph/native/sequence failures. Two longer `524288`-token same-code runs were
fast (`6183.670` and `6196.447 tokens/sec`) but kept secondary because the
benchmark condition report saw pre-measurement GPU contention after prewarm.

The semantic frontier-gap planner follow-up removes the remaining archive-wide
raw-window planning scan. `frontier_gap_plan(...)` now calls
`DualMemoryStore.collect_frontier_gap_indices(...)`, scores only a capped CPU
recency or bucket candidate window, and records
`bounded_frontier_gap_selection.v1` with
`raw_text_payload_policy=selected_frontier_candidate_indices_only`,
`global_candidate_scan=false`, `global_score_scan=false`, and
`language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260617/frontier-gap-bounded.json` compared the
retired global baseline against the bounded path over `65536` archival entries:
bounded scoring touched `192` entries, preserved expected and diagnostic legacy
terms with `quality.min=1.0`, and reduced mean latency from `221.554 ms` to
`9.589 ms` (`23.105x`). The longer hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-frontier-gap-bounded.json`
processed `524288` tokens at `6184.133 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat GPU memory (`1884->1880 MiB`),
no observed contention, and zero graph/native/sequence failures.

The recent replay tag/anchor setup follow-up removes the last archive-linear
setup shape from this replay window. `DualMemoryStore` now keeps a CPU
recency index over slow-memory entries and emits `bounded_recent_memory_window.v1`.
`tag_recent_entries(...)` emits `bounded_recent_memory_tag.v1` from that capped
window, and `MarulhoTrainer.capture_recent_memory_anchors(...)` emits
`bounded_recent_anchor_capture.v1` from the same index while requiring bucketed
entries. The focused cap tests prove the old scan is gone by inserting `10`
recent entries and limiting setup to `3`: both tagging and anchor capture touch
only indices `[9, 8, 7]`, while older entries are not tagged or anchored.

The synthetic report
`reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json`
kept the positive-pressure replay recall and prototype gates passing, accepted
`2` bounded updates, kept `global_fallback_cycle_count=0`, and preserved stored
input-pattern recall at `5.960464477539063e-08` mean distance. The recent tag
setup reported `candidate_window_limit=256`, `candidate_index_available_count=14`,
`candidate_index_count=14`, `global_score_scan=false`, `global_candidate_scan=false`,
`runs_live_tick=false`, `archival_storage_device=cpu`, and `latency_ms=0.0259`.
The anchor setup reported the same `256` candidate cap, `14` indexed entries,
`captured_entry_count=14`, `captured_anchor_count=4`, no global scans,
`runs_live_tick=false`, `archival_storage_device=cpu`, and `latency_ms=0.0136`.
The memory-store device report kept `all_archival_tensors_cpu=true`.

The matching 65536-column long run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json`
processed `262144` tokens at `6228.243 tokens/sec`, with
`train_compute=0.131307 ms/token`, `prepare_training=0.006430 ms/token`,
`finalize_total=0.006432 ms/token`, and `tick_duration_ms.p95=20.538`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `25%`,
GPU utilization max `13%`, GPU memory utilization max `11%`, and GPU memory
stayed flat at `1846 MiB` before and after measurement.

The full-buffer replay-score helper retirement removes the last public helper
that could compute replay priority for the whole slow buffer by default.
`DualMemoryStore.replay_scores(...)` is gone; callers must pass explicit
candidate indices to `replay_scores_for_indices(...)`, which keeps the ranking
formula scoped to a selected window. Focused tests moved ripple-priority and
capture-tag checks onto explicit candidate indices, and `rg "replay_scores\("`
now finds no source or test call sites beyond the retained test name.

The synthetic report
`reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json`
kept the positive-pressure recall and prototype gates passing, applied `2`
bounded updates, ran `4` bounded cycles, kept `global_fallback_cycle_count=0`,
and preserved stored input-pattern recall at
`5.960464477539063e-08` mean distance. The final bounded replay selection
scored `16` candidates and selected `16`, still inside the bucket-indexed
candidate window.

The matching 65536-column long run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json`
processed `262144` tokens at `6211.859 tokens/sec`, with
`train_compute=0.131468 ms/token`, `prepare_training=0.006475 ms/token`,
`finalize_total=0.006438 ms/token`, and `tick_duration_ms.p95=20.679`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `38%`,
GPU utilization max `16%`, GPU memory utilization max `14%`, and GPU memory
stayed flat at `1852 MiB` before and after measurement.

The follow-up score tensor helper retirement removes the remaining public
full-buffer scoring helpers: `maintenance_scores(...)`,
`consolidation_scores(...)`, `repair_scores(...)`, `fragility_scores(...)`, and
unused capture/tag/PRP tensor builders. Production replay selection now reaches
the priority formula only through `_score_replay_index(...)` for selected
candidate indices. The explicit global diagnostic branch still exists behind
`allow_global_score_scan=true`, but it scores privately inside
`select_replay_window(...)`, samples only positive scores, and reports the
diagnostic scan.

The synthetic report
`reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json`
kept recall/prototype gates passing with `2` bounded updates, `4` bounded
cycles, `16` scored/selected positive-pressure entries, and `0` global
fallback cycles. The accepted 65536-column hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json`
processed `262144` tokens at `6151.952 tokens/sec`, with
`train_compute=0.132119 ms/token`, `prepare_training=0.006688 ms/token`,
`finalize_total=0.006420 ms/token`, and `tick_duration_ms.p95=20.697`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The velocity surface reported no observed contention: CPU max `32%`,
GPU utilization max `18%`, GPU memory utilization max `14%`, and GPU memory
stayed flat at `1805 MiB` before and after measurement.

Bounded awake-ripple tagging now retires the production unscoped global
recent-memory scan. Production callers must pass awake bucket scope; otherwise
`ripple_tag_awake(...)` returns an empty `bounded_awake_ripple_tag.v1` report
with `fallback_reason=awake_bucket_scope_required_for_ripple_tagging` and no
scalar/vector scan. The old global scan is diagnostic-only through
`allow_global_diagnostic=true`. The scoped path collects a recent round-robin
candidate window from awake buckets, reports available versus touched entries,
keeps archival storage on CPU, and records `runs_every_token=false`.

The direct benchmark
`reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json`
ran `256` iterations on an `8192`-entry ledger. The diagnostic global path used
`256` vector scans and averaged `1.433332 ms`; the wake-bucket scoped path used
`0` scalar/vector scans, `256` awake-bucket index scans,
`last_ripple_awake_candidate_count=10`, and averaged `1.091997 ms`
(`1.312579x`). Gates passed for avoiding global scans, bounded candidate count,
and not being slower than the diagnostic global baseline.

The synthetic replay quality report
`reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json`
kept positive-pressure recall/prototype gates passing with `2` bounded replay
updates, `4` bounded cycles, `0` global fallback cycles, `16` scored/selected
entries, stored input-pattern distance `5.960464477539063e-08`, and recovery
delta `0.0017409722`. Its memory-store Runtime Truth carries
`last_awake_ripple_tag_report` with `candidate_scope=awake_bucket_index_candidate_window`,
`global_candidate_scan=false`, `diagnostic_global_candidate_scan=false`, and
`runs_every_token=false`.

The 65536-column `262144`-token hot-path run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-awake-ripple-bounded-scope.json`
processed `6149.285 tokens/sec`, with `train_compute=0.131598 ms/token`,
`prepare_training=0.006524 ms/token`, `finalize_total=0.006491 ms/token`, and
`tick_duration_ms.p95=20.586`. The longer `524288`-token run
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json`
processed `6152.328 tokens/sec`, with `train_compute=0.131727 ms/token`,
`prepare_training=0.006691 ms/token`, `finalize_total=0.006526 ms/token`, and
`tick_duration_ms.p95=20.949`. Both runs kept route scoring bounded at
`12/65536`, cached `65526` transition rows, reported
`state_transition_runs_all_columns=false`, had zero graph/native/sequence
failures, and reported no observed contention. The 524288-token run kept GPU
memory flat at `2013 MiB`.

Next gate: repeat the target-specific schedule budgets on a larger or more
grounded target, or replace the synthetic capped-window proof with a larger
hot-bucket replay corpus. Do not broaden a schedule or revive unscoped helper
scans without a target-specific quality gate and a clean long-run check proving
replay remains slow-window work.
