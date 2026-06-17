---
type: benchmark
status: current
related_code:
  - ../../../src/marulho/training/column_structural_review.py
  - ../../../src/marulho/semantics/language_surface.py
related_docs:
  - ../concepts/column-runtime.md
  - ../concepts/plasticity-gate.md
  - ../concepts/pruning.md
related_papers:
  - ../papers/structural-plasticity.md
related_benchmarks: []
---

# Growth Pruning Cost

Structural evaluation and mutation-preflight cost checks.

## Commands

- Focused tests: `python -m pytest tests\test_column_structural_review.py tests\test_checkpointing.py::CheckpointDevicePlacementTests::test_checkpoint_roundtrip_preserves_column_structural_review_queue tests\test_language_surface.py::SubcorticalStructuralPlasticitySurfaceTests tests\test_structural_mutation_executor.py tests\test_status_read_model.py::StatusReadModelCognitiveSignalStateTests::test_structural_plasticity_isolated_evaluation_does_not_advance_revision tests\test_status_read_model.py::StatusReadModelCognitiveSignalStateTests::test_structural_mutation_design_does_not_advance_revision tests\test_status_read_model.py::StatusReadModelCognitiveSignalStateTests::test_structural_mutation_preflight_does_not_advance_revision`
- Service read-only gate: `python -m pytest tests\test_service_api.py -k subcortical_structural_plasticity_endpoint_is_read_only_gate`
- Accepted long throughput run: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\growth_pruning_20260617\checkpointed-candidate-gate-65536-262144-i32-clean2.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 1200 --sample-interval-seconds 0.5`
- Contended rejection sample: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\growth_pruning_20260617\checkpointed-candidate-gate-65536-262144-i32-rerun.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 1200 --sample-interval-seconds 0.5`

## Latest Known Result

2026-06-17 checkpointed candidate structural gate check, using `reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt`.

Accepted local evidence: `reports\growth_pruning_20260617\checkpointed-candidate-gate-65536-262144-i32-clean2.json` processed `262144` tokens at `6019.609 tokens/sec` on `NVIDIA GeForce RTX 3060`, with `velocity_environment.contention.verdict=not_observed`, `train_compute=0.133862 ms/token`, `prepare_training=0.006707 ms/token`, `finalize_total=0.006684 ms/token`, and `tick_duration_ms.p95=21.613`. Runtime Truth stayed on the maintained bounded path: `route_input_rows_scored=12/65536`, `route_output_candidate_count=10`, `state_transition_cached_count=65526`, `state_transition_runs_all_columns=false`, `observed_filtered_deep_sleep_total=510`, `observed_filtered_memory_pressure_total=2`, `observed_fallback_count=0`, `native_sequence_loop_token_count=262128`, and `native_sequence_loop_failure_count=0`. This preserves the current 6k-ish complete-runtime band; it is not a speedup claim.

Same-slice shorter comparison: `reports\growth_pruning_20260617\checkpointed-candidate-gate-65536-131072-i32.json` processed `131072` tokens at `6136.665 tokens/sec`, `train_compute=0.130686 ms/token`, `prepare_training=0.006236 ms/token`, `finalize_total=0.006219 ms/token`, and `tick_duration_ms.p95=20.147`. The earlier same-day active-pressure baseline at `reports\column_scheduler_20260617\active-pressure-scheduler-65536-131072-i32.json` was `6297.455 tokens/sec`, `train_compute=0.130524 ms/token`, `prepare_training=0.006053 ms/token`, `finalize_total=0.006145 ms/token`, and `tick_duration_ms.p95=20.115`.

Rejected long-run sample: `reports\growth_pruning_20260617\checkpointed-candidate-gate-65536-262144-i32-rerun.json` also completed `262144` tokens but fell to `3089.638 tokens/sec` under `velocity_environment.contention.verdict=contention_observed`, with `cpu_before=94`, `cpu_max=100`, `gpu_max=12`, `train_compute=0.248358 ms/token`, and `tick_duration_ms.p95=83.896`. The Runtime Truth route and mutation boundaries were still correct (`route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`, `observed_fallback_count=0`), so the sample rejects the run condition, not the gate.

The growth/pruning gate itself is read-only until the reviewed structural mutation executor runs: candidate tickets carry baseline and evidence hashes, isolated evaluation reports rollback and cost/usefulness evidence, and service endpoints project the gate without selecting edges or mutating structure.
