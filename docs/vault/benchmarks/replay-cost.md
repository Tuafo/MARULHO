---
type: benchmark
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/retrieval/routing_index.py
  - ../../../src/marulho/evaluation/source_bank_memory_match_benchmark.py
  - ../../../src/marulho/evaluation/replay_query_anchor_source_window_benchmark.py
  - ../../../src/marulho/evaluation/sleep_replay_anchor_source_window_benchmark.py
  - ../../../src/marulho/evaluation/artifact_io.py
  - ../../../src/marulho/evaluation/bucket_candidate_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_replay_priority_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py
  - ../../../src/marulho/evaluation/emission_replay_context_review_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_replay_evaluation_context_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_rollout_rehearsal_source_window_benchmark.py
  - ../../../src/marulho/evaluation/status_replay_path_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_snapshot_source_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_replay_target_window_benchmark.py
  - ../../../src/marulho/evaluation/language_plasticity_replay_window_benchmark.py
  - ../../../src/marulho/evaluation/language_application_synapse_window_benchmark.py
  - ../../../src/marulho/evaluation/dense_readout_training_transition_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_ledger_rollout_candidate_window_benchmark.py
  - ../../../src/marulho/evaluation/strong_capture_admission_cadence_benchmark.py
  - ../../../src/marulho/evaluation/slow_memory_fixed_cadence_retirement_benchmark.py
  - ../../../src/marulho/evaluation/source_tick_sleep_deferral_benchmark.py
  - ../../../src/marulho/evaluation/live_memory_summary_projection_benchmark.py
  - ../../../src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py
  - ../../../src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py
  - ../../../src/marulho/evaluation/sleep_plasticity_ticket_queue_source_window_benchmark.py
  - ../../../src/marulho/evaluation/status_transition_memory_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_replay_artifact_provenance_source_window_benchmark.py
  - ../../../src/marulho/evaluation/replay_restore_source_window_benchmark.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/manager.py
  - ../../../src/marulho/service/persistence.py
  - ../../../src/marulho/service/snn_language_plasticity_executor.py
  - ../../../src/marulho/service/status_read_model.py
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
  - reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json
  - reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260618/source-bank-memory-match-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json
  - reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json
  - reports/bounded_replay_window_20260617/frontier-gap-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json
  - reports/bounded_replay_window_20260622/sleep-replay-anchor-source-window-bounded.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-sleep-replay-anchor-source-window.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-adapter-stack-retired.json
  - reports/bounded_replay_window_20260618/bucket-candidate-source-window-bounded.json
  - reports/bounded_replay_window_20260618/synthetic-bucket-source-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-bucket-candidate-source-window.json
  - reports/bounded_replay_window_20260618/snn-readout-replay-priority-source-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-readout-replay-priority-source-window.json
  - reports/bounded_replay_window_20260618/snn-emission-review-replay-policy-source-window.json
  - reports/bounded_replay_window_20260619/emission-replay-context-review-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-profile-rerun.json
  - reports/bounded_replay_window_20260618/snn-rollout-rehearsal-source-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-rollout-rehearsal-source-window.json
  - reports/bounded_replay_window_20260618/status-replay-path-source-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-profile.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json
  - reports/bounded_replay_window_20260620/status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window-rerun.json
  - reports/bounded_replay_window_20260620/snn-readout-ledger-normalization-production-normalizer-retired.json
  - reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window-production-normalizer-retired-smoke.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-readout-ledger-production-normalizer-retired.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json
  - reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json
  - reports/bounded_replay_window_20260620/strong-capture-admission-cadence-after-fixed-cadence-retirement.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json
  - reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery-sharded.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json
  - reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-plasticity-ticket-queue-source-window.json
  - reports/bounded_replay_window_20260620/replay-restore-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json
  - reports/bounded_replay_window_20260619/readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/language-plasticity-replay-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json
  - reports/bounded_replay_window_20260619/language-application-synapse-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-application-synapse-window.json
  - reports/bounded_replay_window_20260619/dense-readout-training-transition-window.json
  - reports/bounded_replay_window_20260619/readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-readout-training-transition-window.json
  - reports/bounded_replay_window_20260618/strong-capture-admission-cadence.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-strong-capture-admission-cadence.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-strong-capture-admission-cadence-rerun.json
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
- Hot-bucket candidate source window:
  `PYTHONPATH=src python -m marulho.evaluation.bucket_candidate_source_window_benchmark --output reports\bounded_replay_window_20260618\bucket-candidate-source-window-bounded.json --archive-size 65536 --candidate-limit 32 --iterations 64`
- Synthetic replay quality for bucket source windows:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260618\synthetic-bucket-source-window.json`
- SNN readout replay-priority source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_replay_priority_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-readout-replay-priority-source-window.json`
- SNN emission-review replay-policy source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_emission_review_replay_policy_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-emission-review-replay-policy-source-window.json`
- SNN rollout rehearsal source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_rollout_rehearsal_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-rollout-rehearsal-source-window.json`
- SNN status replay-path source windows:
  `PYTHONPATH=src python -m marulho.evaluation.status_replay_path_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260618\status-replay-path-source-window.json`
- SNN status transition-memory source window:
  `PYTHONPATH=src python -m marulho.evaluation.status_transition_memory_source_window_benchmark --entry-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\status-transition-memory-source-window.json`
- SNN readout-ledger normalization/store-state source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-source-window.json`
- SNN readout-ledger production normalizer retirement:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260620\snn-readout-ledger-normalization-production-normalizer-retired.json`
- SNN readout-ledger snapshot smoke after production normalizer retirement:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_snapshot_source_window_benchmark --retention-count 2048 --ledger-limit 128 --snapshot-limit 20 --runs 5 --output reports\bounded_replay_window_20260620\snn-readout-ledger-snapshot-source-window-production-normalizer-retired-smoke.json`
- SNN readout-ledger normalization/store-state/known-hash source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-source-window.json`
- SNN replay artifact known-readout source-window binding:
  `PYTHONPATH=src python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-known-readout-source-window.json`
- SNN replay artifact readout-priority source-window binding:
  `PYTHONPATH=src python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-readout-priority-source-window.json`
- SNN replay artifact raw caller-window recorder retirement:
  `PYTHONPATH=src python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-raw-recorder-retired.json`
- Sleep-plasticity ticket queue source windows:
  `PYTHONPATH=src python -m marulho.evaluation.sleep_plasticity_ticket_queue_source_window_benchmark --retained-count 64 --runs 25 --output reports\bounded_replay_window_20260620\sleep-plasticity-ticket-queue-source-window.json`
- SNN readout-ledger dense-label calibration source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json`
- SNN readout-ledger dense-label evaluation source window:
  `PYTHONPATH=src python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json`
- SNN readout replay target payload windows:
  `PYTHONPATH=src python -m marulho.evaluation.readout_replay_target_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\readout-replay-target-window.json`
- Strong-capture slow-memory admission cadence:
  `PYTHONPATH=src python -m marulho.evaluation.strong_capture_admission_cadence_benchmark --tokens 256 --min-interval-tokens 16 --runs 10 --output reports\bounded_replay_window_20260618\strong-capture-admission-cadence.json`
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
- Hot-path protection for bucket candidate source windows:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-bucket-candidate-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- Hot-path protection for SNN readout replay-priority source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-readout-replay-priority-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- Replacement promoted scheduler checkpoint after local report cleanup:
  `PYTHONPATH=src python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --report reports\column_scheduler_20260618\active-pressure-scheduler-65536-checkpoint.json --n-columns 65536 --column-latent-dim 64 --k-routing 10 --seed 20260617 --device cuda --active-pressure-filter-count 2 --candidate-memory-pressure-filter-start-tokens 0`
- Hot-path protection for SNN emission-review replay-policy source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-profile-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- Hot-path protection for SNN rollout rehearsal source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-rollout-rehearsal-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- Hot-path protection for SNN status replay-path source windows:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-profile.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- No-profile hot-path rerun for SNN status replay-path source windows:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-noprofile-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05`
- Hot-path protection for SNN readout-ledger normalization/store-state source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN readout known-hash source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN replay artifact known-readout source-window binding:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN replay artifact readout-priority source-window binding:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for raw replay-artifact recorder retirement:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN readout dense-label calibration source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN readout dense-label evaluation source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for SNN readout replay target payload windows:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`
- Hot-path protection for capped query collection:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-collection.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded query-memory match readout:
  `PYTHONPATH=src python -m marulho.training.query_runner --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --query-text "bounded replay memory" --top-k-candidates 5 --top-k-memories 5 --top-chars 4 --output-json reports\bounded_replay_window_20260617\query-memory-match-bounded-window.json`
- Hot-path protection for bounded query-memory match:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-memory-match.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded query-memory returned payload benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.query_memory_payload_benchmark --output reports\bounded_replay_window_20260617\query-memory-payload-returned-only.json --capacity 65536 --bucket-count 16 --candidate-limit 192 --top-k 5 --iterations 16`
- Hot-path protection for query-memory returned payloads:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-524288-i32-query-memory-payload.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded concept-frontier memory metrics tests:
  `PYTHONPATH=src python -m pytest tests\test_autonomy_runner.py::AutonomySelectionTests::test_concept_frontier_metrics_use_bounded_candidate_window -q`
- Bounded concept-frontier memory metrics benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.concept_frontier_scope_benchmark --output reports\bounded_replay_window_20260617\concept-frontier-bounded-scope.json --capacity 8192 --bucket-count 1024 --candidate-bucket-count 8 --iterations 64 --dim 16`
- Hot-path protection for bounded concept-frontier memory metrics:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded source-bank memory match benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.source_bank_memory_match_benchmark --output reports\bounded_replay_window_20260618\source-bank-memory-match-bounded.json --capacity 65536 --bucket-count 16 --probe-samples 8 --memories-per-probe 4 --max-matches 16 --payload-repeats 24 --iterations 16`
- Hot-path protection for bounded source-bank memory match:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Merged source-bank memory match benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.source_bank_memory_match_benchmark --output reports\bounded_replay_window_20260622\source-bank-merged-probe-window.json --capacity 65536 --bucket-count 16 --probe-samples 8 --memories-per-probe 4 --max-matches 16 --payload-repeats 24 --iterations 32`
- Hot-path protection for merged source-bank memory match:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260622\hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`
- Bounded frontier-gap planner tests:
  `PYTHONPATH=src python -m pytest tests\test_gap_planner.py tests\test_memory_consolidation.py::MemoryConsolidationTests::test_frontier_gap_collection_uses_bounded_recent_index -q`
- Bounded frontier-gap planner benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.frontier_gap_bounded_benchmark --output reports\bounded_replay_window_20260617\frontier-gap-bounded.json --capacity 65536 --iterations 8 --top-entries 24 --max-terms 8`
- Hot-path protection for bounded frontier-gap planner:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
- Recent tag/anchor recency-index tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_recent_memory_tagging_uses_capped_recency_index tests\test_memory_consolidation.py::MemoryConsolidationTests::test_recent_anchor_capture_uses_capped_recency_index tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_replay_window_recall_report -q`
- Synthetic recent replay tag/anchor setup:
  `PYTHONPATH=src python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260617\synthetic-recent-anchor-window.json`
- Hot-path protection for recent replay tag/anchor setup:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Bounded replay-query anchor source window tests:
  `PYTHONPATH=src python -m pytest tests\test_memory_consolidation.py::MemoryConsolidationTests::test_hf_recall_evaluation_reports_bounded_anchor_window tests\test_memory_consolidation.py::MemoryConsolidationTests::test_hf_replay_query_collection_caps_anchor_bucket_source_window tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_column_anchor_recency_metadata -q`
- Bounded replay-query anchor source window benchmark:
  `PYTHONPATH=src python -m marulho.evaluation.replay_query_anchor_source_window_benchmark --output reports\bounded_replay_window_20260618\replay-query-anchor-source-window-bounded.json --anchor-count 8192 --column-latent-dim 32 --max-queries 16 --max-candidates 32 --iterations 64`
- Hot-path protection for bounded replay-query anchor source window:
  `PYTHONPATH=src python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`
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

SNN readout replay-priority scoring is now bounded before it can feed the Replay
Controller priority queue. `snn_language_readout_replay_priority.v1` reads a
recent `32`-event CPU source window and reports
`bounded_snn_readout_replay_priority_source_window.v1`; it does not scan all
retained readout events, load raw replay text, run language reasoning, use GPU
archival metadata, run live tick, or run every token. The source benchmark
`reports/bounded_replay_window_20260618/snn-readout-replay-priority-source-window.json`
matched the diagnostic full-retained scorer's top high-signal readout while
scoring `32/2048` retained events, averaging `1.424948 ms` versus
`51.002932 ms` (`35.792837x`) with no CUDA allocation. The paired `524288`-token
hot-path check stayed in band at `6284.379 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `1852->1858 MiB`, no
observed contention, and zero graph/native/sequence failures.

SNN emission-review replay policy is now bounded before reviewed emissions can
become replay-context seeds. `snn_language_readout_emission_replay_evaluation_policy.v1`
and its design verifier read `16` recent reviewed emissions plus `16` recent
internal readout events and report
`bounded_snn_emission_review_replay_policy_source_window.v1`; they do not scan
all retained review/readout records, load raw reviewed text, run language
reasoning, use GPU archival metadata, run live tick, or run every token. The
source benchmark
`reports/bounded_replay_window_20260618/snn-emission-review-replay-policy-source-window.json`
matched the diagnostic full-retained policy/design top candidate while checking
`32` source events instead of `4096` retained review/readout records, averaging
`2.476164 ms` versus `166.924984 ms` (`67.412734x`) with no CUDA allocation. A
replacement active-pressure checkpoint was generated at
`reports/column_scheduler_20260618/checkpoints/active-pressure-scheduler-65536-seeded.pt`
after the local report cleanup; restore verification kept
`state_transition_runs_all_columns=false`. The clean profiled `524288`-token
hot-path rerun stayed in band at `6376.714 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `2122->2123 MiB`, no
observed contention, and zero graph/native/sequence failures. A same-code
no-profile rerun reached `6392.672 tokens/sec`; an earlier profiled run is
rejected as contended external-load evidence.

The emission replay-context review bridge now keeps that same bounded rule
between operator-reviewed replay design and Replay Controller context
recording. `snn_language_readout_emission_replay_context_review(...)` windows
caller-supplied `selected_replay_context_seeds` and `observed_readout_slots`
through the shared readout replay source-window budget
`SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32`, and requires both source payloads
to be bounded, untruncated, and well formed before mismatch, pressure, or
context recording can run. The old full-payload facade bridge is retired, not
kept as a side implementation.

Focused quality benchmark:

`python -m marulho.evaluation.emission_replay_context_review_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\emission-replay-context-review-window.json`

It passed with exact `32/32` seed and observed-slot windows recording one
context, while oversized seeds and observed slots both blocked at `32/2048`.
Blocked payloads made no mismatch, pressure, or Replay Controller calls. The
report records no global candidate/score scan, no raw text replay payload, no
hidden language reasoning, no live tick, no every-token cadence, CPU
archival/source-window/gate placement, `64x` projected source-work reduction,
`0.0 MiB` CUDA allocation/reservation, and `1.832774 MiB` traced Python peak
allocation.

The first clean hot-path run for this slice
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window.json`
finished at `5877.891 tokens/sec`, so it is retained as below-band variance
evidence. The rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json`
processed `524288` tokens at `5990.908 tokens/sec` with
`train_compute=0.135901 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, GPU memory `2032->2031 MiB`,
and zero graph/native sequence failures. That keeps the bridge out of the live
tick while preserving the maintained 6k-ish band.

The generic SNN replay evaluation-context facade now follows the same rule for
observed sparse slots. `snn_replay_evaluation_context(...)` windows
caller-supplied `observed_readout_slots` through
`SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32`, requires the source payload to be
bounded, untruncated, and well formed before mismatch, pressure, or context
recording can run, and stores the source-window report in the recorded context
metadata. The old generic full-payload observed-slot bridge is retired as an
active shape; the route remains only as a bounded server-recomputed evidence
gate.

Focused quality benchmark:

`python -m marulho.evaluation.snn_replay_evaluation_context_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\snn-replay-evaluation-context-window.json`

It passed with an exact `32/32` observed-slot window recording one context,
while oversized observed slots blocked at `32/2048`. Blocked payloads made no
mismatch, pressure, or Replay Controller calls, and the accepted context carried
the observed-slot source-window metadata. The report records no global
candidate/score scan, no raw text replay payload, no hidden language reasoning,
no live tick, no every-token cadence, CPU archival/source-window/gate
placement, `64x` projected source-work reduction, `0.0 MiB` CUDA
allocation/reservation, `0.656714 MiB` traced Python peak allocation,
`1.276744 ms` exact-path mean latency, and `8.440372 ms` oversized-block mean
latency.

The long hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json`
processed `524288` tokens at `6009.932 tokens/sec` with
`train_compute=0.135671 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, GPU memory `2031->2045 MiB`, and zero graph/native
sequence failures. Sampled GPU contention reached `22%`, so this is maintained
throughput-band evidence rather than contention-free evidence.

SNN rollout rehearsal promotion is bounded before it can feed rehearsal,
consolidation, or regeneration review surfaces. `snn_language_readout_rollout_rehearsal_promotion_policy.v1`
reads a recent `16`-event CPU source window with a `32`-target cap per event and
reports `bounded_snn_readout_rollout_rehearsal_source_window.v1`; it does not
scan all retained rollout events, load raw replay text, run language reasoning,
use GPU archival metadata, run live tick, or run every token. The source
benchmark
`reports/bounded_replay_window_20260618/snn-rollout-rehearsal-source-window.json`
matched the diagnostic full-retained scorer's top high-signal rollout while
scoring `16/2048` retained events, averaging `2.090592 ms` versus
`309.922768 ms` (`148.246414x`) with no CUDA allocation. The paired
`524288`-token hot-path check stayed in band at `6339.682 tokens/sec`, with
bounded `12/65536` route rows, `65526` cached transition rows, GPU memory
`1867->1865 MiB`, and zero graph/native/sequence failures. The velocity sampler
observed GPU contention at `22%`, so this is throughput-protection evidence,
not contention-free hardware evidence.

Runtime Truth replay-path projection is bounded before it summarizes those
slow/control-plane replay surfaces. `StatusReadModel` now emits
`bounded_snn_status_emission_review_history_source_window.v1`,
`bounded_snn_status_emission_replay_design_path_source_window.v1` and
`bounded_snn_status_rollout_consolidation_path_source_window.v1`, each capped to
`16` recent source rows from the relevant status ledgers. The status projection
keeps retained counts and truncation visible, but it no longer materializes all
retained readout, emission-review, or rollout events just to compute readiness.
All three reports state CPU archival/score placement, no live tick, no every-token
work, no raw text payload, no hidden language reasoning, no global
candidate/score scan, and no CUDA/VRAM use. The benchmark
`reports/bounded_replay_window_20260618/status-replay-path-source-window.json`
matched diagnostic full-retained latest history, emission, and rollout evidence while
checking `80` source rows instead of `10240` retained rows, reducing combined
mean projection latency from `102.831789 ms` to `1.309999 ms` (`78.497629x`).
The profiled `524288`-token protection run stayed bounded and contention-free
at `6081.034 tokens/sec`; the no-profile rerun reached `6408.252 tokens/sec`
with bounded `12/65536` route rows, `65526` cached transition rows, and zero
graph/native/sequence failures, but had observed GPU-side contention. This
closes the status/export scan shape without promoting status projection into
replay execution.

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
window. The initial cleanup passed `candidate_indices=processed_indices` into
`sample_for_sfa(...)`; the current reported path calls
`sample_for_sfa_with_report(...)` and embeds `bounded_sfa_sample.v1`. Runtime
Truth records
`sleep_replay_sfa_correction_scope`, candidate/sample counts, and
`sleep_replay_sfa_full_memory_sample_retired=true`. The former list-only
`sample_replay_indices(...)` and `sample_for_sfa(...)` helper APIs are now
removed; callers use reported selection/sampling APIs instead.

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

Repair replay now has its own bounded input-preparation report. The old repair
branch called `assembly_from_input(...)` for every selected replay input even
when the replay entry already had a stored routing key and bucket. The new path
uses `prepare_input_for_candidate_routing(...)` for stored-key entries, clears
stale dense caches when an input trace is absent, and reports
`sleep_replay_unconditional_dense_input_assembly_retired=true` plus dense
fallback counts. The 65536-column benchmark
`reports/bounded_replay_window_20260618/sleep-repair-replay-bounded-input-prepare.json`
selected and repaired `32/32` anchored replay entries, improved mean anchor
distance from `0.508855` to `0.360171`, reduced selected input-prep latency from
`61.351 ms` to `32.613 ms` (`1.881x`), made `0` dense assembly calls during
repair, kept archival tensors on CPU, and used CUDA only for active repair
computation. The focused replay/checkpoint suite passed `15` tests, including
repair replay, selected-window SFA correction, and checkpoint roundtrip
coverage. The paired hot-path run processed `524288` tokens at
`6302.207 tokens/sec`, bounded route scoring at `12/65536`, cached `65526`
transition rows, reported no observed contention, and kept GPU memory
`1805->1806 MiB`.

The follow-up retires the remaining missing-key repair fallback entirely.
Selected repair entries without stored routing keys no longer rebuild keys from
input patterns or project stored assemblies; they are deferred and reported as
`sleep_replay_missing_routing_key_deferred_count`. The mixed-key benchmark
`reports/bounded_replay_window_20260620/sleep-repair-replay-missing-routing-key-deferred.json`
used `32` selected anchored repair entries, updated the `16` entries with
stored routing keys, deferred `16` missing-key entries, made `0` dense
input-assembly calls, removed the stored-assembly projection fallback field,
improved stored-key repair quality by `0.149600`, and kept selected input-prep
speedup at `1.869720x`. The long hot-path check processed `524288` tokens at
`5988.223 tokens/sec`, with `train_compute=0.135762 ms/token`, bounded
`12/65536` route rows, `65526` cached transition rows, GPU memory
`1877->1878 MiB`, and zero graph/native/sequence failures. Velocity observed a
GPU-utilization sample at `31%`, so this is protection evidence, not a speed
ceiling.

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
`strategy=random` now returns an empty `bucket_index_scope_required` report with
no diagnostic global candidate scan.

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

The query-memory payload follow-up keeps explicit query readout bounded after
candidate selection. Similarity-only readout now scores the selected candidate
window tensor-first and materializes replay text only for returned matches,
while term/focus ranking still reports the intentional bounded candidate-window
text ranking policy. `bounded_query_memory_match.v1` now records
`raw_text_payload_loaded`, `raw_text_payload_count`,
`raw_text_payload_policy`, and `language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json`
compared the retired eager candidate-payload shape with the returned-only path
over `65536` archival entries, a `192`-entry candidate window, and `16`
iterations: selected indices matched exactly, raw text payload loads dropped
from `192` to `5`, and mean latency moved from `33.612 ms` to `25.881 ms`
(`1.299x`). The longer hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json`
processed `524288` tokens at `6152.079 tokens/sec`, with
`train_compute=0.132199 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, flat GPU memory (`1874->1878 MiB`), no observed
contention, and zero graph/native/sequence failures.

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

The source-bank probe-signature follow-up bounds the same planner before bucket
selection. `concept_frontier_metrics_with_report(...)` and
`candidate_semantic_signature(...)` now sample an evenly spaced `16`-probe
source window, report the source-probe budget and selected indices, then score
only the capped bucket-indexed memory candidates. The direct report
`reports/bounded_replay_window_20260618/concept-frontier-source-probe-window-bounded.json`
sampled `16/64` probes, scored `64/16384` memory entries, preserved the
diagnostic full-scan top-1, and reduced mean latency from `1556.602 ms` to
`7.637 ms` (`203.829x`). The paired 524288-token protection check kept the
current tree in the same band as the committed baseline (`6303.548` versus
`6307.437 tokens/sec`), with bounded `12/65536` route rows, `65526` cached
transition rows, flat `1789 MiB` GPU memory, no observed contention, and zero
graph/native/sequence failures.

The source-bank memory-match follow-up now uses one bank-level merged candidate
window instead of delegating each sampled probe to the query-memory matcher.
`bank_memory_matches_with_report(...)` samples bounded source-bank probes,
unions routing-index bucket ids, asks the memory store for one CPU
bucket-indexed candidate window capped at `192`, vector-scores sampled probes
against that local window, and loads raw text only for returned matches. It
records `bounded_source_bank_memory_match.v1` with `merged_probe_candidate_window=true`,
`per_probe_query_match_call_count=0`, `retired_per_probe_query_match_call_count`,
candidate/window budgets, CPU archival/score placement, no global scans,
`runs_live_tick=false`, `runs_every_token=false`, and
`language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json`
compared the merged path with the retired per-probe diagnostic aggregation over
`65536` archival entries and `8` probes: selected indices still matched
(`quality.min=1.0`), the bounded path scored `192` candidates and `1536`
probe-candidate similarities, raw text payload loads were `4` returned matches
instead of `32`, and mean latency improved from `560.177 ms` to `106.543 ms`
(`5.258x`). Archival storage and scoring remained CPU-resident, CUDA was
available but archival recall allocated/reserved `0.0 MiB`, and traced Python
peak allocation was `3.059 MiB`. The 524288-token protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json`
stayed in the same 6k-ish band at `6129.933 tokens/sec`, with
`train_compute=0.132126 ms/token`, `prepare_training=0.006695 ms/token`,
`finalize_total=0.006421 ms/token`, `tick_duration_ms.p95=20.698`, bounded
`12/65536` route rows, `65526` cached transition rows, mild GPU contention
observed (`21%` against a `20%` threshold), flat `1763 MiB` GPU memory, and
zero graph/native/sequence failures.

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
terms with `quality.min=1.0`, and reduced mean latency from `217.530 ms` to
`9.073 ms` (`23.975x`). The same report now includes a missing-collector gate
showing that stores without `collect_frontier_gap_indices(...)` produce zero
candidates, zero text payloads, and no global scans instead of using the old
compatibility prefix read. The longer hot-path report
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json`
processed `524288` tokens at `6233.085 tokens/sec`, with
`train_compute=0.131067 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, GPU memory `1844->1840 MiB`, no observed contention,
and zero graph/native/sequence failures.

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
candidate indices. The later runtime hook cleanup removes the remaining
private global diagnostic branch from `select_replay_window(...)`; retired
full-scan comparisons are benchmark-local only.

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
scalar/vector scan. The old global scan has no runtime hook; benchmark-local
retired baselines carry the comparison. The scoped path collects a recent round-robin
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

Runtime concept memory lookup now has an explicit cost envelope instead of
service-owned direct archive reads. The service passes cadenced train-step
observations to `DualMemoryStore.resolve_runtime_concept_memory_matches(...)`;
the store accepts only explicit `memory_index` evidence, caps the batch, caches
duplicate text payloads, and records
`bounded_runtime_concept_memory_lookup.v1`. The benchmark
`reports/bounded_replay_window_20260618/runtime-concept-memory-lookup-bounded.json`
used `512` observations over a `65536`-entry archive with `64` unique memory
indices. It preserved selected-index parity, reduced raw payload reads from
`512` to `64`, recorded `448` cache hits, and reduced mean lookup latency from
`47.156 ms` to `6.380 ms`. It reports CPU archival/score placement, no global
candidate/score scan, `runs_live_tick=true`, `runs_every_token=false`, and
`language_reasoning=false`.

The paired 65536-column `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-runtime-concept-memory-lookup.json`
processed `6237.075 tokens/sec`, with `train_compute=0.131104 ms/token`,
`prepare_training=0.006390 ms/token`, `finalize_total=0.006141 ms/token`, and
`concept_observation=0.000474 ms/token`. It kept route scoring bounded at
`12/65536`, cached `65526` transition rows, reported
`state_transition_runs_all_columns=false`, had zero graph/native/sequence
failures, GPU memory `1809->1861 MiB`, and no observed contention.

Context comparison now reports its bounded query-memory cost instead of
calling the deleted report-dropping `query_runner.memory_matches(...)`
compatibility wrapper. The comparison path calls
`memory_matches_with_report(...)` once per context, shares one returned
replay-entry payload cache, and emits
`bounded_context_comparison_memory_match.v1` with the per-context reports
attached. The benchmark
`reports/bounded_replay_window_20260618/context-memory-match-bounded.json`
used two contexts over a `65536`-entry archive, a `192`-entry per-context
candidate window, and `top_k=8`. It preserved selected-index parity,
collapsed duplicate raw payload reads from `16` to `8` with `8` cache hits,
and reduced mean readout latency from `71.927 ms` to `70.550 ms`. It reports
CPU archival/score placement, no global candidate/score scan,
`runs_live_tick=false`, `runs_every_token=false`, and
`language_reasoning=false`.

The paired 65536-column `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-context-memory-match.json`
processed `6065.987 tokens/sec`, with `train_compute=0.135179 ms/token`,
`prepare_training=0.006512 ms/token`, `finalize_total=0.006339 ms/token`, and
`concept_observation=0.000474 ms/token`. It kept route scoring bounded at
`12/65536`, cached `65526` transition rows, reported
`state_transition_runs_all_columns=false`, had zero graph/native/sequence
failures, GPU memory `1839->1845 MiB`, and no observed contention.

SFA correction sampling now has its own bounded cost report instead of hiding
behind a list-returning helper. `sample_for_sfa_with_report(...)` samples only
selected replay-window indices and records `bounded_sfa_sample.v1`; the
unreported `sample_for_sfa(...)` helper and `sample_replay_indices(...)` are
removed from active code. The benchmark
`reports/bounded_replay_window_20260618/sfa-sample-bounded-window.json` used a
`65536`-entry archive, a `192`-entry selected replay-window candidate set, and
`64` requested samples over `32` iterations. The retired full-buffer sampler
had selected-window sample purity `0.00439453125` and mean latency
`1.451 ms`; the bounded sampler had purity `1.0`, mean latency `0.656 ms`,
and `2.210x` speedup. The report keeps archival storage and samples on CPU,
sets `global_candidate_scan=false`, `runs_live_tick=false`,
`runs_every_token=false`, and `language_reasoning=false`. The accepted
`524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-reported-sfa-sampler-noprofile-rerun2.json`
processed `6127.490 tokens/sec`, with last-tick `train_compute=17.738 ms`
(`0.138579 ms/token` over the 128-token tick), bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`1840->1861 MiB`, and zero graph/native/sequence failures.

The query episode readout benchmark
`reports/bounded_replay_window_20260618/query-episode-readout-bounded.json`
used a `65536`-entry synthetic archive, four returned fragment matches, and a
selected-neighbor radius of `3`. Fragment-only readout missed the target top
episode (`els safe.`); reported selected-neighbor readout recovered
`a cat purrs when it feels safe.` while reading `10` direct neighbor windows
under a `28`-entry budget. Mean latency increased from `0.490 ms` to
`0.936 ms`, so this is a measured explicit-query readout cost rather than a
hot-path optimization. The report keeps archival storage/readout on CPU, sets
`global_candidate_scan=false`, `global_score_scan=false`,
`runs_live_tick=false`, `runs_every_token=false`, and
`language_reasoning=false`. The paired `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-query-episode-readout.json`
processed `6219.926 tokens/sec`, with `train_compute=0.130647 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, no observed
contention, GPU memory `1810->1811 MiB`, and zero graph/native/sequence
failures.

The source-episode admission benchmark
`reports/bounded_replay_window_20260618/source-episode-admission-bounded.json`
compares explicit feed with source admission disabled against bounded source
admission. The disabled arm passed `1/4` simple-animals grounded queries
(`0.25` pass rate); bounded admission passed `4/4` (`1.0` pass rate) by
admitting `5` deduplicated source episodes under the `32`-episode,
`240`-char budget. Admission reports no live tick, no every-token work, no
global candidate/score scan, no language reasoning, and CPU archival storage
for slow buffers, input patterns, and routing keys. Explicit feed latency rose
from `102843.415 ms` to `120136.642 ms`; mean query readout latency improved
from `723.239 ms` to `678.412 ms`. The paired long protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-episode-admission.json`
processed `524288` tokens at `6702.362 tokens/sec`, with
`train_compute=0.121727 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, flat `1808 MiB` GPU memory,
and zero graph/native/sequence failures.

The v2 rerun retires dense source-admission assembly after source candidates
are selected. `bounded_feed_source_episode_admission.v1` now reports
`assembly_policy=bounded_offline_competition_winner_assembly` and
`dense_source_admission_assembly_retired=true`, with one bounded offline
competition returning the winner, assembly, and routing key before CPU archival
storage. `reports/bounded_replay_window_20260618/source-episode-admission-bounded-v2.json`
kept the quality gate at `0.25 -> 1.0`, admitted `5/5` source episodes,
reported `2725.253 ms` admission latency, kept all archival tensors on CPU,
used `cuda:0` only for active assembly computation, and measured a
`46.234 ms` feed-latency delta with mean query latency improving by
`16.968 ms`. The paired v2 hot-path run processed `524288` tokens at
`6412.209 tokens/sec`, bounded route scoring at `12/65536`, cached `65526`
transition rows, and reported zero runtime failures; sampler telemetry observed
GPU-side contention, so this is hot-path protection evidence rather than a
speedup claim.

HF replay query collection now bounds the retained column-anchor source window
before asking the memory store for replay-query indices. The focused benchmark
was:

`python -m marulho.evaluation.replay_query_anchor_source_window_benchmark --output reports\bounded_replay_window_20260618\replay-query-anchor-source-window-bounded.json --anchor-count 8192 --column-latent-dim 32 --max-queries 16 --max-candidates 32 --iterations 64`

It compared the retired all-anchor source pass against
`bounded_replay_query_anchor_bucket_source_window.v1`. The old source pass
handed `8192` anchor buckets to `collect_replay_query_indices(...)` and
averaged `16.414 ms`; the bounded path handed `16` reverse-recency buckets,
averaged `0.346 ms`, and improved mean latency by `47.373x`. Quality improved
for the intended Task-A anchor query source: newest-anchor query hit rate was
`1.0` for the bounded path versus `0.0` for the all-anchor bucket order, while
HF replay recall stayed exact with `mean_input_pattern_distance=0.0`. The
benchmark pinned the trainer to CPU for replay-query evidence and reported
`archival_storage_device=cpu`, `active_replay_compute_device=cpu`, and
`cuda_memory_delta_mib=0.0`.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6376.873 tokens/sec`, with
`train_compute=0.128288 ms/token`, `prepare_training=0.006247 ms/token`,
`finalize_total=0.005964 ms/token`, and `tick_duration_ms.p95=20.160`. Runtime
Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph, selection, native burst, and
native sequence failures were all `0`. GPU memory stayed flat at `1787 MiB`.
The velocity sampler observed borderline GPU contention at `20%`, so this is
accepted as hot-path protection and same-band throughput evidence, not a clean
contention-free ceiling.

Trainer sleep replay now uses the same anchor-source window before calling
`DualMemoryStore.select_replay_window(...)`. The retired helper sorted every
checkpointed `column_anchors` bucket, so the store selector was bucket-bounded
but the source bucket set still scaled with retained anchor count. The focused
benchmark was:

`python -m marulho.evaluation.sleep_replay_anchor_source_window_benchmark --output reports\bounded_replay_window_20260622\sleep-replay-anchor-source-window-bounded.json --anchor-count 8192 --column-latent-dim 32 --replay-steps 16 --candidate-pool 32 --iterations 64 --seed 23`

It compared `sorted_all_column_anchors` against
`bounded_sleep_replay_anchor_bucket_source_window.v1`. The old source pass
averaged `0.892263 ms`; the bounded path read `16/8192` reverse-recency anchor
buckets, averaged `0.037825 ms`, and improved source latency by `23.589x`.
Full sleep-window selection improved from `7.797869 ms` to `0.104864 ms`.
Newest-anchor source hit rate and selected replay-bucket hit rate were both
`1.0`; CUDA was available but unused for archival/source work with
`cuda_memory_delta_mib=0.0`.

The matching protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260622\hotpath-active-pressure-65536-524288-i32-sleep-replay-anchor-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

It processed `524288` tokens at `6135.629 tokens/sec`, with
`train_compute=0.132272 ms/token`, `prepare_training=0.006595 ms/token`, and
`finalize_total=0.006411 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`, and RTX memory moved from `1615 MiB` to `1614 MiB`. GPU utilization touched
the `20%` contention threshold, so this is same-band protection and all-anchor
source retirement evidence, not a speed ceiling.

The replay adapter experiment stack is retired rather than benchmarked as a
new replay mechanism. The deleted stack included dry-run training approval,
dry-run plan creation, metadata-only replay adapter output, experimental
promotion gate, and replay-to-adaptation experiment evidence. It was isolated
from runtime mutation, but it kept an active-looking replay adaptation branch in
`src` and exposed adapter report kinds through service validation summaries.
Active evaluation modules now import JSON/hash helpers from
`marulho.evaluation.artifact_io`; the replay approval/plan/adapter/promotion
modules and their old tests are removed. The retained evidence is the absence
guard in `tests/test_replay_adapter_stack_retired.py`: deleted modules are no
longer importable, service report kinds are absent, and neutral artifact I/O
still preserves JSON object loading plus canonical SHA-256 hashing.
The paired protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-adapter-stack-retired.json`
processed `524288` tokens at `6167.298 tokens/sec`, kept route scoring bounded
at `12/65536`, cached `65526` transition rows, recorded zero graph/native
sequence failures, and kept RTX memory nearly flat at `1728->1729 MiB`; GPU
utilization touched the `20%` contention threshold, so this is same-band
protection evidence rather than a speed ceiling.

The bucket candidate source-window follow-up closes a lower-level source-cost
gap in the shared store helper. The maintained collector now uses
tail-indexed round-robin cursors and reports
`candidate_source_window_policy=tail_indexed_bucket_round_robin_no_full_bucket_materialization`.
`reports/bounded_replay_window_20260618/bucket-candidate-source-window-bounded.json`
kept newest-candidate parity on a `65536`-entry hot bucket, read `32` source
entries, materialized `0`, used CPU source/archival placement with
`cuda_memory_delta_mib=0.0`, and reduced mean source latency from
`0.416944 ms` to `0.060931 ms` (`6.843x`). The replay-quality rerun
`reports/bounded_replay_window_20260618/synthetic-bucket-source-window.json`
kept the positive-pressure consolidation and recall gates passing with
`mean_input_pattern_distance=5.96046447753906e-08`, while the `524288`-token
protection run stayed in band at `6290.744 tokens/sec`, with bounded
`12/65536` route rows, flat `1788 MiB` RTX 3060 memory, and no observed
contention. This keeps source construction selected and CPU-resident before
any bounded replay/query/frontier/ripple operator runs.

The SNN readout-ledger normalization/store follow-up retires the remaining broad
full-materialize-then-cap shapes inside `SNNLanguageReadoutEvidenceLedger`.
`bounded_snn_readout_ledger_normalization_source_window.v1` caps each retained
event family to the newest `128` records before deepcopy/review, and the
store-state persistence boundary now uses the same bounded event-field helper
instead of a second hand-written `list(... )[:limit]` copy path. The refreshed
report includes `bounded_snn_readout_ledger_store_state_source_window.v1` with
CPU archival/normalization/store placement, no live tick, no every-token cadence,
no global candidate/score scan, no hidden language reasoning, and no CUDA archive.
The focused benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-source-window.json`

It used `23` event families with `2048` records each. The bounded normalizer
read `2944` rows under the `max_records_total=2944` budget instead of the
retired path's `47104` rows (`16x` less source work), preserved newest-first
retention (`bounded_recent_retention_rate=1.0` versus `0.0`), and reduced mean
latency from `2415.385992 ms` to `159.388156 ms`. The store-state boundary also
read `2944` rows instead of `47104`, preserved newest-first store-window parity
with the retired list-slice shape, and measured `159.156636 ms` versus
`169.042904 ms` (`1.062117x`). Python traced peak allocation was `6.514462 MiB`;
CUDA allocation/reservation stayed `0.0 MiB` on RTX 3060. A follow-up
replay-priority benchmark still matched the full-retained top candidate while
scoring `32/2048` readout events at `1.253520 ms`, and rollout rehearsal still
matched top quality while scoring `16/2048` events at `2.705792 ms`.

The 65536-column no-profile protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6044.412 tokens/sec`, with
`train_compute=0.134651 ms/token`, `prepare_training=0.007100 ms/token`,
`finalize_total=0.006343 ms/token`, `tick_duration_ms.p95=21.680`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; `velocity_environment.v1` reported no observed
contention, CPU max `25%`, GPU max `13%`, GPU memory-util max `18%`, and RTX
3060 memory `2029->2032 MiB`. The paired profiled pass succeeded but reached
`5953.828 tokens/sec`, so it is retained as secondary stage-profile evidence
rather than the primary throughput gate.

The 2026-06-20 cleanup removes the production all-family normalizer callable
instead of retaining it as dead code. `SNNLanguageReadoutEvidenceLedger` no
longer exposes `_normalized_state()`; all-family normalization is now only a
benchmark-local retired comparison. The focused benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260620\snn-readout-ledger-normalization-production-normalizer-retired.json`

It passed all checks, marked the comparison `production_callable=false` and
`benchmark_local_only=true`, kept bounded all-family source work to `2944` rows
instead of the full-materialized legacy model's `47104` rows (`16x`), preserved
newest-first retention, and reduced mean normalization latency from
`5807.281164 ms` to `379.736352 ms`. The autonomous chain boundary preserved
hash/review/count/current-pointer parity while checking `4352` target-family
rows instead of `100096` broad-normalized rows (`23x`) and reducing mean chain
latency from `18942.731124 ms` to `930.299448 ms` (`20.361972x`). The snapshot
smoke after deletion preserved quality while reading `260` rows instead of
`2944` benchmark-local all-family rows and reducing mean snapshot latency from
`409.182080 ms` to `71.702280 ms`. CUDA allocation/reservation stayed
`0.0 MiB`; archival/normalization/snapshot placement stayed CPU-resident. The
paired `524288`-token protection run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-readout-ledger-production-normalizer-retired.json`
stayed in band at `6224.717 tokens/sec`, with
`train_compute=0.130755 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, GPU memory `1988->1987 MiB`, and zero graph/native
sequence failures. Borderline `21%` GPU contention makes it same-band
protection, not a speed ceiling.

The known-readout-evidence hash provenance helper now follows the same source
window instead of invoking all-family ledger normalization before replay design,
dry-run, preflight, or bridge provenance checks. The active helper emits
`bounded_snn_readout_known_evidence_hash_source_window.v1` from
`snn_readout_ledger.events`, reports CPU archival/lookup placement, no live
tick, no every-token cadence, no raw text payload, no hidden language reasoning,
no mutation/plasticity, and no CUDA archive. The focused benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-source-window.json`

It preserved known-hash set parity with the retired broad-normalized lookup,
returned `128` hashes, checked `128` `events` rows instead of `2944` normalized
ledger rows (`23x` less source work), and reduced mean lookup latency from
`156.192384 ms` to `6.792628 ms` (`22.994397x`). Python traced peak allocation
was `6.538644 MiB`; CUDA allocation/reservation stayed `0.0 MiB` on RTX 3060.

The same-shape 65536-column no-profile rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `5938.461 tokens/sec`, with
`train_compute=0.136714 ms/token`, `prepare_training=0.007355 ms/token`,
`finalize_total=0.006530 ms/token`, `tick_duration_ms.p95=22.329`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; `velocity_environment.v1` reported no observed
contention, CPU max `51%`, GPU max `13%`, GPU memory-util max `18%`, and RTX
3060 memory `2032->2031 MiB`. The first same-code run reached
`5871.364 tokens/sec`, so the rerun is the primary same-band protection sample,
not a new throughput ceiling.

The 2026-06-20 follow-up removes the set-only known-readout evidence API rather
than leaving it as a side path. Replay design, dry-run, plasticity preflight,
plasticity bridge, and evaluated replay-artifact recording now call
`known_readout_evidence_hashes_with_report()` and carry
`bounded_snn_readout_known_evidence_hash_source_window.v1` into required
evidence and artifact hashes. `ReplayController` rejects evaluated transition
memory replay artifacts unless that report is bounded, CPU-resident, non-live,
non-every-token, no raw text, no language reasoning, no mutation/plasticity, and
no CUDA archive. The focused benchmark was:

`python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-known-readout-source-window.json`

It passed with known-readout source window `1/8`, persisted
`readout_evidence_source_window_hash`, CPU archival placement, no global
candidate/score scan, no raw text payload, no hidden language reasoning, no
live tick, and no every-token work. Indexed artifact provenance verification
checked `4` bounded records instead of `256` worst-case retained records
(`64x` less source work). Mean latency was `0.365884 ms`, median `0.320000 ms`,
and p95 `0.629300 ms`; Python traced peak allocation was `0.014095 MiB`; CUDA
allocation/reservation stayed `0.0 MiB` on RTX 3060.

The same-shape 65536-column rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6007.228 tokens/sec`, with
`train_compute=0.134831 ms/token`, `prepare_training=0.007471 ms/token`,
`finalize_total=0.006545 ms/token`, `tick_duration_ms.p95=22.165`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; `velocity_environment.v1` reported GPU
contention, CPU max `49%`, GPU max `30%`, GPU memory-util max `23%`, and RTX
3060 memory stayed flat at `1986 MiB`. This is same-band protection under
contention for a retired bypass, not a speed ceiling.

The replay-priority binding follow-up closes the next report-dropping artifact
shape. `transition_memory_replay_artifact_proposal(...)` already chose its
replay window from `replay_priority(...)`, but the evaluated replay artifact did
not have to carry the priority selector's own source-window report. The
maintained path now stores `replay_priority_source_window` plus
`replay_priority_source_window_hash` in the proposal, requires that report when
recording evaluated replay artifacts, and includes the hash in artifact and
rollout-review recomputation.

Focused quality and latency benchmark:

`python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-readout-priority-source-window.json`

It passed with replay-priority source window `1/32`, persisted
`replay_priority_source_window_hash`, CPU archival and scoring placement, no
global candidate/score scan, no raw text payload, no language reasoning, no
live-tick or every-token work, CUDA available but unused, and `0.0 MiB` CUDA
allocation/reservation. The indexed replay-artifact provenance check still
verified the oldest retained permit every run with `4` bounded index lookups
instead of `256` projected retained-record checks (`64x`). Mean verification
latency was `0.421992 ms`, p95 was `0.719700 ms`, and traced Python peak was
`0.014385 MiB`.

Long protection runs:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The first run succeeded but is rejected as primary evidence: `4662.031
tokens/sec`, `tick_duration_ms.p95=46.504`, and `train_compute=0.173596
ms/token` with observed GPU contention (`gpu max=27%`, GPU memory-util max
`25%`).

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The accepted rerun processed `524288` tokens at `5937.908 tokens/sec`, with
`train_compute=0.136165 ms/token`, `prepare_training=0.007577 ms/token`,
`finalize_total=0.006686 ms/token`, `tick_duration_ms.p95=22.814`, and prewarm
`284.026 s`. Runtime Truth kept route scoring bounded at `12/65536` input rows
and `10` output candidates, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, and recorded zero graph/native
sequence failures. Contention was `not_observed` (`cpu max=22%`, `gpu max=13%`,
GPU memory-util max `18%`), and RTX 3060 memory stayed flat at `1943 MiB`.
This is same-band hot-path protection for deleting a report-dropping slow-path
artifact shape, not a speed promotion.

The raw replay-artifact recorder retirement closes the remaining production
caller-window artifact path. The public raw recorder is absent, controller load
drops raw caller-window artifacts, and only the evaluated internal-ledger
artifact recorder can persist replay artifacts with known-readout,
replay-priority, and provenance source-window hashes. The focused benchmark was:

`python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\snn-replay-artifact-raw-recorder-retired.json`

It passed with `public_raw_recorder_callable=false`,
`raw_loaded_artifact_count=0`, `raw_artifact_index_hit=false`, persisted
source-window hashes, CPU archival placement, no global scan, no raw text
payload, no language reasoning, no live-tick or every-token work, and `0.0 MiB`
CUDA allocation/reservation. Indexed artifact provenance verification checked
`4` bounded records instead of `256` worst-case retained records (`64x`). Mean
verification latency was `0.538460 ms`, p95 was `1.659900 ms`, and traced
Python peak was `0.017773 MiB`.

The long protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6004.719 tokens/sec`, with
`train_compute=0.135314 ms/token`, `prepare_training=0.007010 ms/token`,
`finalize_total=0.006575 ms/token`, `tick_duration_ms.p95=21.825`, and prewarm
`278.046 s`. Runtime Truth kept route scoring bounded at `12/65536` input rows,
`10` output candidates, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, and recorded zero graph/native
sequence failures. The velocity sampler observed contention (`cpu max=32%`,
`gpu max=28%`, GPU memory-util max `22%`), and RTX 3060 memory moved only
`1863->1865 MiB`; this is same-band protection, not a clean speed ceiling.

Dense-label candidate history and calibration policy now use the same one-family
source-window boundary. The active path emits
`bounded_snn_dense_label_candidate_calibration_source_window.v1` from
`snn_readout_ledger.dense_label_candidate_events`, reports CPU archival/lookup
placement, no live tick, no every-token cadence, no raw text payload, no hidden
language reasoning, no mutation/plasticity, and no CUDA archive. The focused
benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json`

It preserved history-hash, policy-hash, and ready-candidate parity with the
retired broad-normalized path, returned `128` candidates, checked `128`
dense-label rows instead of `2944` normalized ledger rows (`23x` less source
work), and reduced mean history+policy latency from `222.453668 ms` to
`49.093244 ms` (`4.531248x`). Python traced peak allocation was `9.320584 MiB`;
CUDA allocation/reservation stayed `0.0 MiB` on RTX 3060.

The same-shape 65536-column no-profile run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6018.915 tokens/sec`, with
`train_compute=0.135317 ms/token`, `prepare_training=0.006959 ms/token`,
`finalize_total=0.006455 ms/token`, `tick_duration_ms.p95=21.517`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; `velocity_environment.v1` reported no observed
contention, CPU max `64%`, GPU max `16%`, GPU memory-util max `20%`, and RTX
3060 memory `2030->2029 MiB`.

Dense-label calibration evaluation now resolves only preflight-selected
candidate hashes through the same dense-label event family. The active path
emits
`bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1`,
reports CPU archival/lookup/evaluation placement, no global candidate/score
scan, no raw text payload, no hidden language reasoning, no live tick, no
every-token cadence, no mutation/plasticity, and no CUDA archive. The focused
benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 25 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json`

It preserved sample-hash and metric parity with the retired broad-normalized
evaluation path, evaluated `8` selected samples, checked `128` dense-label rows
instead of `2944` normalized ledger rows (`23x` less source work), and reduced
mean evaluation latency from `225.545020 ms` to `12.673884 ms`
(`17.796046x`). Python traced peak allocation was `9.320584 MiB`; CUDA
allocation/reservation stayed `0.0 MiB` on RTX 3060.

The first same-shape `524288`-token protection run reached
`5906.886 tokens/sec`, so it is retained as below-band variance evidence. The
accepted rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6116.710 tokens/sec`, with
`train_compute=0.133135 ms/token`, `prepare_training=0.006912 ms/token`,
`finalize_total=0.006197 ms/token`, `tick_duration_ms.p95=21.705`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; `velocity_environment.v1` reported no observed
contention, CPU max `41%`, GPU max `13%`, GPU memory-util max `18%`, and RTX
3060 memory `2030->2030 MiB`.

The SNN readout replay dry-run/preflight/bridge path now bounds caller-supplied
payloads before tensor materialization. `replay_dry_run(...)` windows
`selected_replay_targets`, `plasticity_preflight(...)` windows the dry-run
ephemeral replay trace, and `plasticity_replay_bridge(...)` windows
`candidate_replay_sequences` with `SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32`.
The three surfaces report
`bounded_snn_readout_replay_dry_run_target_window.v1`,
`bounded_snn_readout_plasticity_preflight_trace_window.v1`, and
`bounded_snn_readout_plasticity_bridge_sequence_window.v1` with CPU archival
placement, no global candidate/score scan, no live tick, no every-token cadence,
no raw replay text, and no hidden language reasoning. The focused benchmark was:

`python -m marulho.evaluation.readout_replay_target_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\readout-replay-target-window.json`

It passed while cutting both dry-run targets and bridge sequences from
`2048` supplied records to `32` materialized records (`64x` less source work),
with `2016` truncated in each surface. Mean dry-run latency was `6.061784 ms`;
mean bridge latency was `1.328924 ms`. Archival storage, source selection, and
active replay computation stayed on CPU, `cuda_memory_allocated_before/after`
and `cuda_memory_reserved_before/after` remained `0.0 MiB`, and the report marks
the old full-payload shape as a projection, not an executable side path.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6109.000 tokens/sec`, with
`train_compute=0.133186 ms/token`, `prepare_training=0.006965 ms/token`,
`finalize_total=0.006289 ms/token`, and `tick_duration_ms.p95=21.677`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph and native sequence failures
were `0`; `velocity_environment.v1` reported no observed contention, CPU max
`31%`, GPU max `13%`, GPU memory-util max `18%`, and RTX 3060 memory
`2020->2018 MiB`. This is same-band hot-path protection for a slow/control-plane
replay cleanup, not a live-tick replay promotion.

The exported SNN language plasticity replay functions now carry the same
runtime-boundary rule instead of relying only on API schema limits.
`evaluate_spike_language_plasticity_replay(...)`,
`run_spike_language_plasticity_replay_experiment(...)`, and
`build_spike_language_plasticity_shadow_delta(...)` cap caller-supplied replay
records at `32`; the shadow-delta path also caps `pre_indices`,
`post_indices`, and fallback `active_indices` to `16` per side before the
`pre x post` sparse pair loop. The focused benchmark was:

`python -m marulho.evaluation.language_plasticity_replay_window_benchmark --payload-count 2048 --index-count 256 --runs 25 --output reports\bounded_replay_window_20260619\language-plasticity-replay-window.json`

It passed with replay evaluation `32/2048`, replay experiment `32/2048`, and
shadow-delta pair checks `8192/134217728`, a `64x` record-work reduction and
`16384x` pair-work reduction versus the retired full-payload projection. Mean
latencies were `11.024580 ms` for replay evaluation, `8.622980 ms` for replay
experiment, and `297.890092 ms` for shadow delta. Archival storage, source
selection, and active replay computation stayed CPU-resident, traced Python peak
allocation was `14.474813 MiB`, CUDA allocation/reservation stayed `0.0 MiB`,
and the reports state no global candidate/score scan, no live tick, no
every-token cadence, no raw replay text, no hidden language reasoning, and no
runtime mutation. The API schemas now reuse
`SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT` so the validation cap and semantics
runtime cap have one named budget.

The longer protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `5999.398 tokens/sec`, with
`train_compute=0.135445 ms/token`, `prepare_training=0.007140 ms/token`,
`finalize_total=0.006422 ms/token`, and `tick_duration_ms.p95=22.016`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. The sampler reported GPU-side contention at the threshold
(`max_gpu_utilization_percent=22`, memory-util max `23`), CPU max `38%`, and
RTX 3060 memory `2018->2023 MiB`, so this is same-band protection evidence under
observed contention rather than a clean speed-ceiling run. The preceding same
command without the `-rerun` suffix also succeeded at `6025.620 tokens/sec` and
was likewise marked GPU-contended.

The checkpointed language-application boundary now enforces that the final
mutation payload is itself bounded and untruncated. This closes the downstream
side entrance after replay/shadow selection: `apply_live_application(...)`
windows `shadow_delta.bounded_synapses`, and
`regenerate_transition_memory(...)` windows
`regeneration_design.candidate_synapses`, both with
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32`. Oversized payloads are
blocked before checkpoint writes or runtime mutation rather than silently
truncated into a partial structural write.

Focused quality benchmark:

`python -m marulho.evaluation.language_application_synapse_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\language-application-synapse-window.json`

It passed with oversized live-application and transition-regeneration payloads
blocked at `32/2048`, `source_probe_count=33`, `source_truncated_count=2016`,
zero checkpoint calls, zero state mutation, no global candidate/score scan, no
raw text payload, and no hidden language reasoning. Exact-window payloads still
worked: live application applied `32` synapses and regeneration added `32`
synapses through two checkpoint calls each. Mean latencies were `1.827460 ms`
for oversized live blocking, `1.840956 ms` for oversized regeneration blocking,
`15.976332 ms` for exact live application, and `34.346436 ms` for exact
regeneration. Archival storage, source-window selection, and active application
stayed CPU-resident; traced Python peak allocation was `1.982166 MiB`, CUDA
allocation/reservation stayed `0.0 MiB`, and the retired full-payload work is
projected from source counts only (`64x` source-work reduction).

Clean long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-application-synapse-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6039.734 tokens/sec`, with
`train_compute=0.134728 ms/token`, `prepare_training=0.006949 ms/token`,
`finalize_total=0.006436 ms/token`, and `tick_duration_ms.p95=21.511`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`. `velocity_environment.v1` reported no observed contention, CPU max `29%`,
GPU max `16%`, GPU memory-util max `19%`, and RTX 3060 memory `2020->2034 MiB`.
This is throughput protection for a checkpointed slow-path boundary, not a new
live-tick replay operator.

The dense-readout training transition boundary now uses the same single
bounded slow-path contract. `apply_dense_readout_training_loop(...)` no longer
copies every caller-supplied training transition or every caller-sized
`pre_indices`/`post_indices` list before slicing. Instead, design, schema,
preflight, and executor all share
`SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT=32` and
`SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT=32`. Oversized
transition or sparse-index payloads are blocked before checkpoint writes or
runtime mutation.

Focused quality benchmark:

`python -m marulho.evaluation.dense_readout_training_transition_window_benchmark --payload-count 2048 --index-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\dense-readout-training-transition-window.json`

It passed with oversized transition payloads blocked at `32/2048`, oversized
index payloads blocked at `32/2048`, zero checkpoint calls, zero state mutation,
no global candidate/score scan, no raw text payload, and no hidden language
reasoning. Exact-window training still committed `32` dense/sparse updates
through two checkpoint calls. Mean latencies were `43.082416 ms` for oversized
transition blocking, `6.215512 ms` for oversized index blocking, and
`115.425860 ms` for exact-window checkpointed training. Archival storage and
source-window selection stayed CPU-resident, active benchmark training stayed on
CPU, traced Python peak allocation was `5.696876 MiB`, CUDA
allocation/reservation stayed `0.0 MiB`, and the retired full-payload transition
work is projected from source counts only (`64x` source-work reduction).

Clean long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-readout-training-transition-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

It processed `524288` tokens at `6028.820 tokens/sec`, with
`train_compute=0.135088 ms/token`, `prepare_training=0.007078 ms/token`,
`finalize_total=0.006280 ms/token`, and `tick_duration_ms.p95=21.702`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`. `velocity_environment.v1` reported no observed contention, CPU max `54%`,
GPU max `15%`, GPU memory-util max `18%`, and RTX 3060 memory `2029->2028 MiB`.
This retires the caller-sized checkpointed training side path while preserving
the maintained 6k-ish live-tick band.

The readout-ledger rollout consolidation/regeneration chain now uses that same
single application-synapse source-window operator before every structural
candidate review step. `rollout_consolidation_design(...)`,
`rollout_consolidation_shadow_delta(...)`,
`rollout_developmental_plasticity_review(...)`,
`rollout_regeneration_proposal_adapter(...)`,
`rollout_regeneration_replay_artifact_review(...)`, and direct Replay
Controller regeneration-design normalization all cap candidate inputs at
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32` and require untruncated
source payloads before permit-preview or permit hashing. The old full-list
ledger/controller materialization path is retired, not kept as a side
implementation.

Focused quality benchmark:

`python -m marulho.evaluation.readout_ledger_rollout_candidate_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\readout-ledger-rollout-candidate-window.json`

It passed with exact-window rollout evidence reaching the permit-preview gate
with `32/32` candidates, while oversized design, shadow, developmental,
adapter, replay-artifact review, and direct replay-controller normalization
all blocked at `32/2048`. The report records no global candidate/score scan,
no raw text payload, no hidden language reasoning, no live tick, no every-token
cadence, CPU archival/source-window/gate placement, `64x` projected source-work
reduction, `0.0 MiB` CUDA allocation/reservation, and `9.073439 MiB` traced
Python peak allocation.

The matching clean hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json`
processed `524288` tokens at `6075.293 tokens/sec` with
`train_compute=0.134312 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, GPU memory `2031->2043 MiB`,
and zero graph/native sequence failures. That keeps the slow-path candidate
window out of the live tick while preserving the maintained sustained band.

The rollout-regeneration facade now uses the same single candidate-window
operator before permit issuance, application preflight, and checkpoint-backed
application. The old facade path built full `regeneration_design.candidate_synapses`
lists before the bounded executor could reject them. The maintained path caps
each facade gate at `SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32`, forwards
only the bounded regeneration design, and requires untruncated candidate
payloads before calling the replay controller or executor.

Focused quality benchmark:

`python -m marulho.evaluation.rollout_regeneration_facade_candidate_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\rollout-regeneration-facade-candidate-window.json`

It passed with oversized permit, preflight, and application payloads blocked at
`32/2048`, zero replay-controller calls for oversized permits, zero executor
calls and zero checkpoint writes for oversized applications, no global
candidate/score scan, no raw text payload, and no hidden language reasoning.
The exact `32`-candidate flow still issued one permit, produced one ready
preflight proposal, and reached the single executor path with `32` candidates.
Mean latencies were `0.728800 ms` for oversized permit blocking,
`0.735856 ms` for oversized preflight blocking, `0.742636 ms` for oversized
application blocking, and `2.277792 ms` for the exact permit/preflight/application
flow. Archival storage, source-window selection, facade gating, and active
application placement stayed CPU-resident, traced Python peak allocation was
`1.852119 MiB`, CUDA allocation/reservation stayed `0.0 MiB`, and the retired
facade full-payload work is projected from source counts only (`64x`
source-work reduction).

Long protection runs:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-rollout-regeneration-facade-candidate-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

The first same-code run succeeded but is retained as below-band variance:
`5938.820 tokens/sec`, `train_compute=0.137347 ms/token`, bounded
`12/65536` route rows, `65526` cached rows, no observed contention, GPU memory
`2033->2031 MiB`, and zero graph/native sequence failures.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-rollout-regeneration-facade-candidate-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

The accepted rerun processed `524288` tokens at `6121.143 tokens/sec`, with
`train_compute=0.133293 ms/token`, `prepare_training=0.006856 ms/token`,
`finalize_total=0.006270 ms/token`, and `tick_duration_ms.p95=21.611`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`. The sampler observed light GPU contention at `23%`, CPU max was `36%`,
GPU memory-util max was `23%`, and RTX 3060 memory stayed flat at `2031 MiB`.
This is protection evidence that the facade cleanup did not add live-tick work,
not a clean speed-ceiling claim.

The strong-capture admission cadence follow-up closes the remaining
every-strong slow-memory write shape. Strong-event evidence can still be
emitted by the device path on every threshold crossing, but `DualMemoryStore`
admission now archives at most one strong capture per
`slow_memory_archive_strong_capture_min_interval_tokens` window. The production
config default is `16`, values `<=1` are invalid, and Runtime Truth exposes the
min interval, strong archive count, refractory skip count, and last archived
strong-capture token.

The focused benchmark was:

`python -m marulho.evaluation.strong_capture_admission_cadence_benchmark --tokens 256 --min-interval-tokens 16 --runs 10 --output reports\bounded_replay_window_20260618\strong-capture-admission-cadence.json`

It forced every token to qualify as a strong candidate while disabling cadence
admission with `slow_memory_archive_interval_tokens=1000000000`. The bounded
path archived `17` records, skipped `239`, and selected `16` strong captures
under a max selected gap of `16` tokens with a final gap of `14`. The retired
every-strong path is not executable in the benchmark; it is projected from the
forced-strong candidate count as `256` archive writes, giving a `15.058824x`
write reduction. Bounded mean latency was `1172.027720 ms` over `10` runs.
Archival storage stayed CPU-resident, active replay computation was `none`,
CUDA allocation/reservation stayed `0.0 MiB`, and the report states no global
candidate scan, no global score scan, no raw text loading except archived
entries, and no language reasoning.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-strong-capture-admission-cadence.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6100.415 tokens/sec`, with
`train_compute=0.133405 ms/token`, `prepare_training=0.007070 ms/token`,
`finalize_total=0.006437 ms/token`, `tick_duration_ms.p95=21.328`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`, and
`state_transition_cached_count=65526`. Strong-capture archive/refractory counts
remained `0` in the ordinary live tick, graph/native/sequence failures were
all `0`, and RTX 3060 memory stayed flat at `2390 MiB`. The rerun also
succeeded at `5326.602 tokens/sec`, but included a `12435 ms` max tick outlier
and observed GPU contention, so it is retained as variance evidence rather
than promotion evidence.

The fixed-cadence slow-memory admission follow-up removes the remaining
per-token fallback archive trigger. Fixed cadence now records deferred
maintenance evidence through the cognitive boundary controller; it does not
call `DualMemoryStore.update(...)`, does not tag awake ripple entries, and does
not materialize raw replay text. First-token retained/fallback admission and
bounded strong-capture admission remain the only live archive writes.

The focused benchmark was:

`python -m marulho.evaluation.slow_memory_fixed_cadence_retirement_benchmark --tokens 256 --archive-interval-tokens 16 --runs 10 --output reports\bounded_replay_window_20260620\slow-memory-fixed-cadence-admission-retired.json`

It passed with `1` bounded archive versus `17` retired fixed-cadence writes
over `256` tokens, a `17x` archival-write reduction. First-token retention was
intact, strong-capture counters stayed zero in the non-strong case, deferred
cadence count was `16`, and the fixed-cadence execution gate was closed.
Archival placement stayed CPU-resident, active replay computation was `none`,
CUDA allocation/reservation stayed `0.0 MiB`, and the report states no global
candidate scan, no global score scan, no every-token slow-memory admission, and
no hidden language reasoning.

The refreshed strong-capture regression benchmark was:

`python -m marulho.evaluation.strong_capture_admission_cadence_benchmark --tokens 256 --min-interval-tokens 16 --runs 10 --output reports\bounded_replay_window_20260620\strong-capture-admission-cadence-after-fixed-cadence-retirement.json`

It still passed with `17` bounded strong-capture archives versus `256` retired
every-strong writes (`15.058824x` write reduction), proving the selected
STC-like path remains after fixed cadence was removed.

The first 65536-column protection run after the change was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It succeeded at `5758.051 tokens/sec` with no observed contention, but is
rejected as primary throughput evidence because it fell below the maintained
6k-ish band. Stage costs were still comparable
(`train_compute=0.140847 ms/token`, `prepare_training=0.007262 ms/token`,
`finalize_total=0.006773 ms/token`), bounded route scoring stayed at
`12/65536`, cached transition rows stayed at `65526`, and graph/native
sequence failures were `0`.

The accepted rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6043.321 tokens/sec`, with
`train_compute=0.134537 ms/token`, `prepare_training=0.007011 ms/token`,
`finalize_total=0.006472 ms/token`, and `tick_duration_ms.p95=21.354`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. The run recorded
`slow_memory_cadence_deferred_count=2048`,
`last_slow_memory_cadence_token=524288`,
`slow_memory_cadence_execution_gate=false`, no graph/native sequence failures,
and flat RTX 3060 memory at `1958 MiB`. The sampler observed borderline GPU
contention at the configured threshold, so this is same-band hot-path
protection evidence rather than a clean speed-ceiling claim.

The source tick sleep replay deferral follow-up removes the remaining service
fallback route into automatic sleep replay. Background source ticks may still
fall back to per-token `train_step` for metrics or unsupported burst execution,
but that fallback now passes `allow_sleep_maintenance=False`; due sleep is
reported as deferred maintenance and explicit trainer sleep windows remain the
only replay execution path.

The focused benchmark was:

`python -m marulho.evaluation.source_tick_sleep_deferral_benchmark --runs 10 --sleep-cost-ms 3.0 --output reports\bounded_replay_window_20260620\source-tick-sleep-replay-deferred.json`

It passed with service fallback sleep calls `0`, explicit allowed slow-path
projection sleep calls `1`, visible `sleep_maintenance_deferred=1`, and mean
controlled service latency `4.551660 ms` versus `7.446660 ms` for the allowed
projection. The report keeps archival storage on CPU, active replay computation
deferred to slow path, no global candidate/score scan, and no hidden language
reasoning.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-source-tick-sleep-replay-deferred.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `5993.959 tokens/sec`, with
`train_compute=0.135624 ms/token`, `prepare_training=0.007083 ms/token`,
`finalize_total=0.006522 ms/token`, and `tick_duration_ms.p95=21.728`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`. The velocity sampler reported no observed contention, CPU max `56%`, GPU
max `13%`, GPU memory-util max `18%`, and RTX 3060 memory stayed flat at
`1959 MiB`. This is same-band hot-path protection for deleting source tick
sleep replay execution, not a speed-ceiling claim.

The live memory-summary projection follow-up removes the remaining
trainer/service/status call shape that used full slow-memory summaries as
display telemetry. `DualMemoryStore.summary_stats()` still exists for explicit
offline consolidation and quality windows, but live projections now call
`DualMemoryStore.live_summary_stats()`, which reports fill/counter/report
fields without advancing STC decay, creating all-entry tensors, or scanning
archival metadata.

The focused benchmark was:

`python -m marulho.evaluation.live_memory_summary_projection_benchmark --entries 65536 --dim 16 --runs 5 --output reports\bounded_replay_window_20260620\live-memory-summary-projection.json`

It passed with scalar fill/report parity, `summary_full_memory_scan=false`,
`summary_scan_entry_count=0`, `summary_projection_read_only=true`, CPU
projection/archive placement, and no hidden language reasoning. The retired
full-summary path scanned `65536` entries at `658.789240 ms` mean, while the
bounded live projection averaged `0.149500 ms`.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-live-memory-summary-projection.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6024.783 tokens/sec`, with
`train_compute=0.135003 ms/token`, `prepare_training=0.006861 ms/token`,
`finalize_total=0.006603 ms/token`, and `tick_duration_ms.p95=21.265`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
`0`. The velocity sampler reported no observed contention, CPU max `22%`, GPU
max `13%`, GPU memory-util max `18%`, and RTX 3060 memory moved
`1959->1958 MiB`. This is same-band hot-path protection for deleting full
memory-summary scans from live/status projection, not a speed-ceiling claim.

The readout-ledger snapshot follow-up retires the broad
snapshot-through-normalizer shape. `SNNLanguageReadoutEvidenceLedger.snapshot`
now reads only the event families it returns through
`bounded_snn_readout_ledger_snapshot_source_window.v1`, capping each family at
the requested snapshot limit and ledger retention limit. The snapshot source
window reports CPU archival/source/snapshot placement, no global candidate or
score scan, no live tick, no every-token cadence, no hidden language reasoning,
and no CUDA archive.

The focused benchmark was:

`python -m marulho.evaluation.snn_readout_ledger_snapshot_source_window_benchmark --retention-count 2048 --ledger-limit 128 --snapshot-limit 20 --runs 25 --output reports\bounded_replay_window_20260620\snn-readout-ledger-snapshot-source-window.json`

It preserved newest-first display quality and retained-count parity, read `260`
rows instead of `2944` in the benchmark-local retired normalizer model
(`11.323077x` less source work), and reduced mean snapshot latency from
`393.040600 ms` to `67.334088 ms` (`5.837171x`). Traced Python peak allocation
was `0.575356 MiB`; CUDA allocation/reservation stayed `0.0 MiB`.

The `524288`-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-ledger-snapshot-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6443.960 tokens/sec`, with
`train_compute=0.127084 ms/token`, `prepare_training=0.006251 ms/token`,
`finalize_total=0.005922 ms/token`, `tick_duration_ms.p95=19.655`, bounded
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`, and
`state_transition_cached_count=65526`. Graph/native/sequence failures were
all `0`, no contention was observed, and RTX 3060 memory stayed flat at
`1899 MiB`.

The applied-synapse provenance status follow-up retires the broad readiness
scan over all applied sparse weights and provenance rows. The status projection
now emits `bounded_snn_status_applied_synapse_provenance_source_window.v1`,
reads at most `32` `sparse_transition_weights` keys and `32`
`synapse_provenance_by_key` rows, and blocks exact audit readiness when the
source window is truncated. This keeps exact integrity review in an explicit
audit/slow window instead of letting operator status become an archive-wide
recall pass.

Historical applied-only quality report:

`reports\bounded_replay_window_20260620\status-applied-synapse-provenance-source-window.json`

It passed with `64` bounded source rows instead of `4096` retired broad-scan
rows (`64x` less source work), reduced mean latency from `66.313336 ms` to
`3.242332 ms` (`20.452358x`), preserved bounded-window replay-regeneration
provenance health, and correctly reported `source_window_complete=false`,
`integrity_count_scope=bounded_source_window`, and
`eligible_for_readout_synapse_audit_review=false` for truncated source state.
Archival metadata and lookup stayed CPU-resident, CUDA allocation/reservation
stayed `0.0 MiB`, and the report states no global candidate/score scan, no
raw text payload, no hidden language reasoning, no replay, no live tick, and no
every-token work.

Long protection runs:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The first run succeeded at `5875.245 tokens/sec` with no observed contention,
bounded `12/65536` route rows, and zero graph/native sequence failures, but it
is retained as secondary variance rather than primary evidence.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The accepted rerun processed `524288` tokens at `6350.288 tokens/sec`, with
`train_compute=0.128968 ms/token`, `prepare_training=0.006470 ms/token`,
`finalize_total=0.005945 ms/token`, `tick_duration_ms.p95=20.090`, and prewarm
`241.741 s`. Runtime Truth kept route scoring bounded at `12/65536` input rows
and `10` output candidates, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, and recorded zero graph/native
sequence failures. Contention was `not_observed` (`cpu max=25%`, `gpu max=8%`,
GPU memory utilization max `9%`), and RTX 3060 memory stayed flat at
`1936 MiB`.

The exact applied-synapse audit now also has a bounded source window instead
of copying all applied sparse weights and provenance rows before returning
capped audit rows. `synapse_provenance_audit(...)` emits
`bounded_snn_readout_synapse_provenance_audit_source_window.v1`, selects at
most `64` applied sparse-weight/provenance rows from CPU archival state, and
requests ledger evidence only for the selected hashes. If retained applied
synapses exceed the source window, exact audit review is blocked until a
separate explicit slow-audit policy widens the selected source.

Quality and latency report:

`reports\bounded_replay_window_20260620\synapse-provenance-audit-source-window.json`

It passed with `2048` retained sparse weights and `2048` retained provenance
rows. The bounded production audit read `64` source rows while the
benchmark-local diagnostic full scan touched `4096` records and materialized
`2048` audit rows (`32x` less source work by the benchmark's comparison
metric). Selected source keys matched the diagnostic first source window,
requested ledger hashes were capped at `64`, truncated source windows blocked
exact review, and the audit reported CPU archival/lookup placement, no global
candidate or score scan, no raw text payload, no hidden language reasoning, no
live tick, no every-token work, no mutation/plasticity, and no GPU-resident
archival metadata. Mean latency was `75.262088 ms` versus `259.221928 ms` for
the diagnostic full materializer, traced Python peak allocation was
`1.909667 MiB`, CUDA was available, and production audit GPU use was `false`.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Accepted rerun: `success=true`, `524288` tokens in `81.396445 s`,
`6441.166 tokens/sec`, `tick_duration_ms.p95=19.527`,
`train_compute=0.127184 ms/token`, `prepare_training=0.006068 ms/token`, and
`finalize_total=0.005747 ms/token`. Prewarm took `239.005 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. Contention was `not_observed`
(`cpu max=20%`, `gpu max=13%`, GPU memory utilization max `18%`), and RTX 3060
memory stayed flat at `1866 MiB`.

## Runtime Trace Export And Replay Sample Summary Windows

Runtime trace export and replay-sample summary now follow the same selected
slow-window rule as replay-dataset preview. Trace export emits
`bounded_runtime_trace_export_source_window.v1` before endpoint filtering or
trace-state lookup. Replay-sample summary emits
`bounded_replay_sample_summary_source_window.v1`, scans at most `64` recent
sample records for mode/status/latest-item summary, and keeps the retained
sample count as O(1) deque length. Living status and feedback summary use
bounded recent trace/action windows instead of asking for all retained runtime
episode traces.

Focused quality and source-budget benchmark:

`python -m marulho.evaluation.replay_dataset_source_window_benchmark --output reports\bounded_replay_window_20260620\replay-dataset-runtime-trace-export-summary-source-window.json --trace-count 64 --sample-count 256 --selected-candidates-per-sample 16 --limit 50 --endpoint respond --runs 15`

Result: `pass=true`; trace export read `50/64` retained traces, replay-sample
summary read `64/256` retained sample records, replay-sample links read
`64/256`, and selected-candidate links read `1024/4096`. Selected target IDs
and trace-export IDs both matched the diagnostic bounded window (`50/50`).
All source windows report CPU archival/source placement, no global
candidate/score scan, no raw replay text, no hidden language reasoning, no
live tick, no every-token cadence, no mutation/plasticity/training, and no
GPU-resident archival metadata. Direct bounded replay-sample summary cost was
`7.440207 ms` on this small retained fixture; that is source-budget evidence,
not a latency promotion.

The accepted `524288`-token hot-path rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-runtime-trace-export-summary-source-window-rerun.json`
kept the live tick protected at `6047.311 tokens/sec`,
`tick_duration_ms.p95=21.357`, `train_compute=0.134598 ms/token`,
`prepare_training=0.007216 ms/token`, `finalize_total=0.006447 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, no observed contention, flat RTX 3060
memory at `1911 MiB`, and zero graph/native sequence failures.

## Replay Dataset Candidates Endpoint Retirement

The replay-dataset candidates preview endpoint is retired so candidate review
has one public path: `/terminus/replay-plan`. Dataset preview and bundle keep
candidate context inside the bounded preview source window instead of exposing
`/terminus/replay-dataset/candidates`.

Focused quality and source-budget benchmark:

`python -m marulho.evaluation.replay_dataset_source_window_benchmark --output reports\bounded_replay_window_20260622\replay-dataset-candidates-retired-source-window.json --trace-count 64 --sample-count 256 --selected-candidates-per-sample 16 --limit 50 --endpoint respond --runs 15`

Result: `pass=true`. The preview source window reports
`candidate_context_source=replay_plan_summary_inside_dataset_preview`,
`retired_public_candidate_preview_endpoint=/terminus/replay-dataset/candidates`,
and `replacement_candidate_endpoint=/terminus/replay-plan`. Bounded preview
kept `50/50` selected target IDs and replay-link coverage against the
diagnostic window. Runtime trace work was `50/64` (`1.28x` reduction);
replay-sample, replay-sample summary, and selected-candidate work each reduced
`4x`. Mean bounded preview latency was `1851.868400 ms`; trace export mean was
`2415.486627 ms`; replay-sample summary mean was `7.221433 ms`. Python traced
peak was `3.53 MiB`; CUDA was available on the RTX 3060 but unused by this
CPU archival/source/link path (`gpu_used=false`, `0.23 MiB` allocated,
`2.0 MiB` reserved).

## Replay Dataset History Wrapper Retirement

`GET /terminus/replay-dataset/history` is retired because replay-sample history
already has one canonical path: `/terminus/replay-sample/history`. Dataset
preview and bundle remain the only dataset/export surfaces.

Service endpoint benchmark:

`python -m marulho.evaluation.service_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260622\service-benchmark-replay-dataset-history-retired.json --export-limit 5`

Result: `success=true`; endpoint timings contain no
`replay_dataset_history`, the report contains no
`replay_dataset_history_summary`, slow-path endpoint count is `5`, and the
hot-path budget passes with `p95=340.668 ms` and total `664.890 ms`.

Refreshed dataset source-window benchmark:

`python -m marulho.evaluation.replay_dataset_source_window_benchmark --output reports\bounded_replay_window_20260622\replay-dataset-history-retired-source-window.json --trace-count 64 --sample-count 256 --selected-candidates-per-sample 16 --limit 50 --endpoint respond --runs 15`

Result: `pass=true`; selected target and replay-link parity stayed `50/50`,
trace work stayed `50/64`, replay-sample and selected-candidate work stayed
`4x` reduced, mean bounded preview cost was `1956.391633 ms`, replay-sample
summary mean was `7.152813 ms`, Python traced peak was `3.53 MiB`, CUDA was
available but unused, and archival/source/link placement stayed CPU-resident.

## Non-Reversible Anchor Source Fallback Retirement

The shared replay anchor-window helper no longer materializes non-reversible
anchor sources with `list(anchors)`. Non-reversible sources now block before
selection with explicit source-window evidence; reversible dict-backed anchors
still read the newest `16` anchors.

Focused test:

`python -m pytest tests/test_memory_consolidation.py -k "anchor_source or anchor_window or replay_window_recall_report or sleep_replay_selection_report"`

Benchmark:

`python -m marulho.evaluation.sleep_replay_anchor_source_window_benchmark --output reports\bounded_replay_window_20260622\sleep-replay-anchor-nonreversible-fallback-retired.json --anchor-count 8192 --replay-steps 4 --candidate-pool 8 --iterations 15 --seed 7`

Result: `status=passed`; the maintained path read `16/8192` anchors,
materialized `0` entries, kept `anchor_source_full_scan=false`, kept newest
anchor and selected replay-bucket hit rates at `1.0`, improved source latency
from `1.046780 ms` retired mean to `0.044360 ms` (`23.597x`), used CPU
archival/source placement, and had `0.0 MiB` CUDA memory delta. The
non-reversible regression test covers the blocked branch with
`fallback_reason=non_reversible_anchor_bucket_source` and zero source reads.

## Status Transition Memory Source Window

Capacity pressure, dense readout tensor integrity, applied-synapse provenance,
and rollout/server binding status now share the same bounded transition-memory
source-window helper. Each projection reads at most `32`
`sparse_transition_weights` rows and `32` `synapse_provenance_by_key` rows,
keeps archival metadata and lookup CPU-resident, and reports no replay,
mutation, plasticity, raw text payload, hidden language reasoning, live tick,
or every-token work. Truncated windows block exact review readiness for resize,
dense tensor integrity, applied-synapse audit, and rollout/server binding.

Focused quality and latency benchmark:

`python -m marulho.evaluation.status_transition_memory_source_window_benchmark --entry-count 2048 --runs 25 --output reports\bounded_replay_window_20260620\status-transition-memory-source-window.json`

Result: `pass=true`; bounded source reads were `256` rows across the four
status projections versus `10240` rows in the benchmark-local retired repeated
broad projection (`40x` less source work). Mean status latency fell from
`89.558896 ms` to `11.162376 ms` (`8.023282x`). The path preserved retained
counts, reported `source_window_complete=false`, and blocked exact readiness:
`capacity_promotion_status=waiting_for_complete_language_capacity_source_window`,
`dense_ready=false`,
`applied_promotion_status=waiting_for_complete_applied_synapse_provenance`,
and
`binding_promotion_status=waiting_for_complete_server_transition_memory_source_window`.
Python traced peak allocation was `0.065983 MiB`; CUDA allocation/reservation
stayed `0.0 MiB`.

Long protection runs:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The first run succeeded but is rejected as primary evidence: `524288` tokens at
`5278.529 tokens/sec`, `tick_duration_ms.p95=34.919`,
`train_compute=0.153055 ms/token`, `prepare_training=0.008041 ms/token`, and
`finalize_total=0.007574 ms/token`. Runtime Truth still kept route scoring at
`12/65536`, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, and recorded zero graph/native
sequence failures, but velocity reported GPU contention at the `20%` threshold.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Accepted rerun: `success=true`, `524288` tokens in `82.289821 s`,
`6371.238 tokens/sec`, `tick_duration_ms.p95=19.949`,
`train_compute=0.128035 ms/token`, `prepare_training=0.006444 ms/token`, and
`finalize_total=0.006113 ms/token`. Prewarm took `268.296 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. Velocity still reported
borderline GPU contention (`cpu max=17%`, `gpu max=23%`, GPU memory utilization
max `19%`), so this is same-band throughput protection rather than a clean
ceiling claim. RTX 3060 memory stayed flat at `1986 MiB`.

## Sleep Replay Routing-Index Refresh

Selected deep/repair replay no longer performs a full routing-index rebuild
after it updates a small set of prototypes, and missing replay rows no longer
revive that rebuild as a recovery fallback. The active path is
`routing_index_existing_row_refresh.v1`: the trainer passes the replay-updated
prototype IDs, the routing index refreshes only existing cache rows through a
CPU ID-to-row map, and the report states direct update count, row lookup mode,
missing-ID count, dirty-cache state, skipped-update count, and deferred recovery
status. Full rebuild is retained only for checkpoint restore, bootstrap,
explicit offline repair, or benchmark-local diagnostics, not as the normal
selected-replay path.

Focused quality and latency benchmark:

`python -m marulho.evaluation.sleep_replay_routing_index_refresh_benchmark --entries 65536 --dim 64 --update-count 16 --missing-update-count 1 --runs 10 --device auto --output reports\bounded_replay_window_20260620\sleep-replay-routing-index-deferred-recovery.json`

Result: `pass=true`; the bounded path updated `16/65536` selected rows, deferred
`1` missing row without inserting it, used `row_lookup_mode=host_id_row_map`,
avoided a rebuild, left the cache clean, and returned exact top-1 IDs for the
updated vectors. Mean latency fell from `118.414640 ms` for the benchmark-local
retired `add()+rebuild()` path to `4.171690 ms` for selected-row refresh. The
sharded companion report passed at `13.348040 ms` versus `140.566380 ms`.
Archival storage and row-lookup metadata stayed CPU-resident; the active
routing tensor cache stayed on the selected routing device.

Hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `88.211824 s`,
`5943.512 tokens/sec`, `tick_duration_ms.p95=22.097`,
`train_compute=0.136627 ms/token`, `prepare_training=0.007218 ms/token`, and
`finalize_total=0.006603 ms/token`. Prewarm took `325.352 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. CPU max was `28%`; GPU max was
`19%`, velocity marked contention not observed, RTX 3060 memory stayed flat at
`1878 MiB`, and throughput remained in the maintained 6k-ish band. A first
same-slice run at `5688.783 tokens/sec` is rejected as primary evidence because
velocity observed CPU contention at `99%`.

## Bucket Consolidation Cache Lookup

The live scalar bucket-consolidation lookup no longer scans every slow-memory
entry to answer one winner/bucket consolidation metric. Production scalar reads
use the maintained CPU bucket consolidation cache and report
`bucket_consolidation_level_cache_lookup.v1` with `full_memory_scan=false` and
`scan_entry_count=0`. A missing cache is reported as a no-scan miss; explicit
tensor rebuilds remain load/capture/offline, explicit tensor request,
checkpoint load, graph capture/prewarm, or benchmark-local work.

Focused quality and latency benchmark:

`python -m marulho.evaluation.bucket_consolidation_cache_lookup_benchmark --entries 65536 --buckets 65536 --bucket-id 37 --runs 25 --output reports\bounded_replay_window_20260620\bucket-consolidation-cache-lookup.json`

Result: `pass=true`; cached scalar lookup matched the retired full scan within
`1e-6`, reported `cache_hit`, and scanned `0` entries. Mean latency fell from
`12.999192 ms` for the retired scalar scan to `0.016260 ms` for the cached
lookup. The cache is CPU metadata; CUDA was available but unused by the
benchmark.

Hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `87.860658 s`,
`5967.267 tokens/sec`, `tick_duration_ms.p95=22.005`,
`train_compute=0.135870 ms/token`, `prepare_training=0.007225 ms/token`, and
`finalize_total=0.006671 ms/token`. Prewarm took `325.285 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. CPU max was `28%`; GPU max was
`25%`, so velocity marked contention observed, but RTX 3060 memory stayed flat
at `1963->1964 MiB` and throughput remained in the maintained 6k-ish band.

Next gate: repeat the target-specific schedule budgets on a larger or more
grounded target, or replace the synthetic capped-window/readout-payload proof
with a larger grounded replay corpus. Do not broaden a schedule or revive
unscoped helper scans without a target-specific quality gate and a clean
long-run check proving replay remains slow-window work.

## Explicit Replay Text Payload Opt-In

The implicit raw-text default on `DualMemoryStore.replay_entry(...)` is retired.
The maintained path returns tensor and replay metadata by default; raw text
payloads require `include_text_payload=True` after a bounded query/source-bank
or context readout window selects entries.

Focused benchmark:

`python -m marulho.evaluation.replay_entry_text_payload_opt_in_benchmark --entries 65536 --buckets 16 --read-count 192 --candidate-limit 192 --top-k 5 --runs 25 --output reports\bounded_replay_window_20260620\replay-entry-text-payload-opt-in.json`

Result: `passed=true`; default replay-entry reads loaded `0/192` raw text
payloads, explicit opt-in loaded `192/192`, and bounded query readout loaded
`5` returned-match payloads. Runtime Truth fields report no global
candidate/score scan, no live tick, no every-token cadence, CPU archival
placement, and `language_reasoning=false`.

Hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-replay-entry-text-payload-opt-in.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `87.470800 s`,
`5993.863 tokens/sec`, `tick_duration_ms.p95=21.555`,
`train_compute=0.135543 ms/token`, `prepare_training=0.006947 ms/token`, and
`finalize_total=0.006613 ms/token`. Prewarm took `320.045 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. CPU max was `33%`; GPU max was
`15%`, so velocity reported no observed contention. RTX 3060 memory stayed flat
at `1878->1879 MiB`.

## Replay Sample Single Path

The audit-only replay executor alias is retired. `POST /terminus/replay-sample`
and `GET /terminus/replay-sample/history` are the only maintained sampler
surfaces; `mode="execute"`, `execution_id`, and `replay_executor_summary` are
removed from production output.

Service report:
`reports/bounded_replay_window_20260620/replay-sample-single-path-service-benchmark.json`
passed with no `replay_executor_summary`, replay-sample history latency
`4.798 ms`, mode counts limited to `dry_run` and `sample`, CPU summary
placement, no raw replay text, no hidden language reasoning, no live tick, no
every-token work, and no training/plasticity/action side effects.

Replay-dataset source-window report:
`reports/bounded_replay_window_20260620/replay-dataset-source-window-replay-sample-single-path.json`
passed with canonical `sample` records, `50/50` target/link parity, `64/256`
bounded replay-sample summary rows, `64/256` replay link rows, CPU
archival/source placement, no GPU-resident archival metadata, and no
live/every-token work.

Hot-path protection:
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-sample-single-path.json`
processed `524288` tokens in `88.089265 s` at `5951.781 tokens/sec`, p95
`21.962 ms`, `train_compute=0.136320 ms/token`,
`prepare_training=0.007157 ms/token`, `finalize_total=0.006694 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, zero
graph/native sequence failures, no observed contention, CPU max `38%`, GPU max
`14%`, and RTX 3060 memory `1894->1881 MiB`.

## Sleep Plasticity Ticket Queue Source Windows

Sleep-plasticity review-ticket and scheduler-design-review-ticket queues now
inspect bounded source windows instead of the full retained ticket history
before autonomy, scheduler-design, or installation proposals.

Focused benchmark:
`reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json`
passed with both queues inspecting `16/64` retained records and matching the
diagnostic latest verified ticket. Source work fell `4x`; mean bounded latency
was `3.326372 ms` for the sleep-review queue and `201.999864 ms` for the
scheduler-design queue, versus diagnostic full-retained means of
`14.464224 ms` and `681.522452 ms`. Runtime Truth fields report CPU
archival/source/score placement, no global candidate/score scan, no raw replay
text, no hidden language reasoning, no live tick, no every-token cadence, no
scheduler install, no mutation/plasticity, CUDA available but unused, and
`0.072 MiB` traced Python peak.

Hot-path protection:
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-plasticity-ticket-queue-source-window.json`
processed `524288` tokens in `87.414632 s` at `5997.714 tokens/sec`, p95
`21.621 ms`, `train_compute=0.135466 ms/token`,
`prepare_training=0.006990 ms/token`, `finalize_total=0.006604 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, zero
graph/native sequence failures, CPU max `32%`, GPU max `20%`, and RTX 3060
memory `1779->1780 MiB`.

## Selected Replay Consolidation Cache Recovery

Selected replay consolidation no longer rebuilds the full bucket-consolidation
cache when cache metadata is missing. `DualMemoryStore.consolidate_replay(...)`
now reports `bounded_selected_replay_consolidation.v1`: selected replay entries
still update replay counts, capture tags, consolidation levels/events, and
global/local EMAs; selected-bucket cache delta updates run only when cache
metadata already exists; and missing metadata records
`cache_missing_deferred_no_full_rebuild`.

Focused benchmark:

`python -m marulho.evaluation.selected_replay_consolidation_cache_benchmark --entries 65536 --buckets 65536 --selected-count 16 --runs 7 --output reports\bounded_replay_window_20260620\selected-replay-consolidation-cache.json`

Result: `pass=true`. The bounded path matched selected-entry consolidation
levels, replay counts, consolidation events, capture tags, and fast EMA against
the benchmark-local retired diagnostic that rebuilt the full cache first. It
scanned `0` cache-rebuild entries, reduced source work `4096x`, and averaged
`2.291943 ms` versus `2979.156029 ms` for the retired diagnostic. Runtime Truth
fields report no full-memory scan, no global candidate scan, no live tick, no
every-token cadence, no raw replay text, no hidden language reasoning, CPU
archival/cache metadata placement, CUDA available but unused with `0.0 MiB`
allocated/reserved, and `0.510799 MiB` traced Python peak.

Hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-selected-replay-consolidation-cache-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `87.775633 s`,
`5973.047 tokens/sec`, p95 tick `21.749 ms`,
`train_compute=0.135713 ms/token`, `prepare_training=0.007071 ms/token`, and
`finalize_total=0.006787 ms/token`. Prewarm took `339.026 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
recorded zero graph/native sequence failures. Velocity reported no observed
contention, CPU max `27%`, GPU max `13%`, and RTX 3060 memory `2039->2041 MiB`.
The first same-slice run at `5879.905 tokens/sec` is treated as contended
variance because velocity observed GPU contention at `21%`.

## Replay Restore Source Window

Replay checkpoint restore no longer normalizes full retained replay-controller
histories before applying retention budgets. The maintained path reports
`bounded_replay_restore_source_window.v1` and slices replay sample history,
regeneration permits, replay-evaluation contexts, review tickets, scheduler
installations, and transition-memory replay artifacts before normalization,
indexing, or evaluated-artifact validation.

Focused quality and latency report:

`python -m marulho.evaluation.replay_restore_source_window_benchmark --entry-count 65536 --runs 7 --output reports\bounded_replay_window_20260620\replay-restore-source-window.json`

Result: `pass=true`; latest-window state matched the benchmark-local retired
full-materialized restore diagnostic, including `64` valid evaluated artifacts.
The bounded path inspected `656` records instead of `524288`
(`799.219512x` less source work). Mean bounded restore latency was
`15.600729 ms`; the retired diagnostic averaged `6605.339529 ms`
(`423.399426x`). Runtime Truth reported CPU archival/source placement,
`full_retained_materialization=false`, no live tick, no every-token work, no raw
replay text, no hidden language reasoning, no mutation/plasticity, and no
GPU-resident archival metadata. CUDA was available but unused with `0.0 MiB`
allocated/reserved, and Python traced peak allocation was `0.581783 MiB`.

Accepted hot-path protection rerun:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `88.181184 s`,
`5945.577 tokens/sec`, p95 tick `22.062 ms`,
`train_compute=0.136201 ms/token`, `prepare_training=0.007152 ms/token`, and
`finalize_total=0.006825 ms/token`. Prewarm took `345.053 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`,
selected CUDA on the RTX 3060, and recorded zero graph/native sequence
failures. Velocity reported no observed contention, CPU max `30%`, GPU max
`13%`, GPU memory utilization max `18%`, and RTX memory `2061->2062 MiB`. A
first same-shape run at `5904.020 tokens/sec` is retained as secondary evidence
because velocity observed GPU contention at `22%`.

## Applied Replay Lineage Checkpoint Summary

Checkpoint save/restore no longer derives applied replay-lineage integrity by
copying and scanning the full `synapse_provenance_by_key` archive. The active
path maintains `snn_applied_replay_lineage_incremental_summary.v1` as mutation
evidence: replay regeneration adds a row hash for each replay-regenerated
synapse, and non-replay overwrites or pruning remove that row. Checkpoint
summary and restore validation read only the maintained CPU counts/digests and
report `full_provenance_scan=false` plus `source_record_scan_count=0`.

Focused quality and latency report:

`python -m marulho.evaluation.applied_replay_lineage_checkpoint_summary_benchmark --entry-count 65536 --runs 7 --output reports\bounded_replay_window_20260620\applied-replay-lineage-checkpoint-summary.json`

Result: `pass=true`; the active summary matched the benchmark-local retired
full-scan diagnostic on lineage counts and digest over `65536` replay-lineage
rows while reading `0` provenance source records. Retired diagnostic source
reads were `196608`. Active mean latency was `0.065529 ms`; retired full-scan
mean latency was `6766.639043 ms` (`103261.747364x`). Active traced Python peak
was `0.001343 MiB`; retired traced peak was `24.036118 MiB`. CUDA was
available on the RTX 3060 but unused with `0.0 MiB` allocated/reserved.

Accepted hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260620\hotpath-active-pressure-65536-524288-i32-applied-replay-lineage-checkpoint-summary.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

Result: `success=true`, `524288` tokens in `87.483241 s`,
`5993.011 tokens/sec`, p95 tick `21.608 ms`,
`train_compute=0.135253 ms/token`, `prepare_training=0.007078 ms/token`, and
`finalize_total=0.006754 ms/token`. Prewarm took `335.271 s`. Runtime Truth
kept route scoring at `12/65536` input rows and `10` output candidates, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`,
selected CUDA on the RTX 3060, and recorded zero graph/native sequence
failures. Velocity reported no observed contention, CPU max `15%`, GPU max
`13%`, GPU memory utilization max `18%`, and RTX memory `2082->2084 MiB`.

## Sleep Replay Associative Recall And Sequence Deferral

Trainer deep sleep now measures bounded associative recall inside the selected
sleep replay window. The operator is not a new archive scan: it uses at most
`4` selected replay entries as queries, loads replay tensors without raw text,
and recalls only within the selected bucket-indexed candidate window. Runtime
Truth reports CPU archival/source/score placement, no live tick, no every-token
cadence, no hidden language reasoning, no mutation authority, and no plasticity
authority.

Focused recall report:

`python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260622\sleep-replay-associative-recall-window.json`

Result: the positive-pressure arm passed
`bounded_sleep_replay_associative_recall.v1` with `4` bounded queries and mean
best input-pattern distance `5.96046447753906e-08`. Zero-pressure and no-anchor
controls ran `0` queries and made no quality claim. Prototype repair still did
not claim improvement in this report.

Focused source-tick deferral report:

`python -m marulho.evaluation.source_tick_sleep_deferral_benchmark --runs 10 --sleep-cost-ms 3.0 --output reports\bounded_replay_window_20260622\source-tick-sequence-sleep-deferral.json`

Result: `pass=true`; service fallback sleep calls `0`, delegated sequence
fallback sleep calls `0`, explicit slow-path projection sleep calls `1`,
service mean `5.411860 ms`, sequence mean `64.898510 ms`, and allowed slow-path
mean `6.603540 ms`. Sequence fallback exposed `allow_sleep_maintenance=false`,
fallback train-step count, and deferred sleep-maintenance count instead of
running replay in the source tick.

Hot-path protection:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260622\hotpath-active-pressure-65536-524288-i32-sleep-replay-associative-recall-source-sequence-deferral.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

Result: `success=true`, `524288` tokens in `80.817236 s`,
`6487.329 tokens/sec`, p95 tick `19.023 ms`,
`train_compute=0.125633 ms/token`, `prepare_training=0.005922 ms/token`, and
`finalize_total=0.005852 ms/token`. Prewarm took `304.955 s`. Runtime Truth
kept route scoring bounded at `12/65536`, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, selected CUDA on the RTX 3060, and
recorded zero graph/native sequence failures. Velocity reported no observed
contention, CPU max `12%`, GPU max `19%`, GPU memory utilization max `22%`, and
RTX memory `1709->1707 MiB`.
