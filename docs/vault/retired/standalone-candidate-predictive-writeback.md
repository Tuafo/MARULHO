---
type: retired
status: rejected
related_code:
  - ../../../src/marulho/core/inplace_column_cuda.py
  - ../../../src/marulho/evaluation/predictive_transition_benchmark.py
  - ../../../src/marulho/training/column_transition_runtime.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/hot-path-latency.md
related_papers: []
related_benchmarks:
  - ../../../reports/column_scheduler_20260615/cuda-8192-predictive-writeback-scope-triton-experiment.json
  - ../../../reports/column_scheduler_20260615/cuda-direct-inplace-predictive-scope-ab.json
---

# Standalone Candidate Predictive Writeback

The standalone candidate predictive writeback side path is removed.

The isolated Triton writeback proved candidate-row parity and very low launch
latency, but wiring it as a separate runtime side path added materialization and
orchestration cost. The valid direct 1024-column complete-step A/B regressed
mean latency from `8.607221875 ms` to `27.7788925 ms`.

The maintained CUDA path is `ColumnTransitionRuntime` calling
`inplace_column_transition_cuda`, where candidate predictive state updates
inside the existing fused transition launch and Runtime Truth reports
`candidate_predictive_transition_mode=fused_inplace`. The dense predictive
transition benchmark remains fallback/evidence tooling, not a second scheduler
mutation boundary.
