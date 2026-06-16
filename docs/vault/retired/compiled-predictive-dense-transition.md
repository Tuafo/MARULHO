---
type: retired
status: rejected
related_code:
  - ../../../src/marulho/core/predictive_columns.py
  - ../../../src/marulho/training/column_transition_runtime.py
  - ../../../src/marulho/training/checkpointing.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../benchmarks/hot-path-latency.md
related_papers: []
related_benchmarks:
  - ../../../reports/predictive_transition_cuda_1024/predictive-transition-benchmark.json
---

# Compiled Predictive Dense Transition

The configurable `compiled` predictive dense transition mode is removed.

It was useful as isolated evidence that torch-compiled predictive state math can
be fast, but it became stale once `ColumnTransitionRuntime` and the fused
in-place/graph transition owned the CUDA scheduler boundary. Keeping `compiled`
as a default or selectable runtime path preserved a second mutation path with
hidden compile warmup and weaker Runtime Truth.

New configs default to `inplace_triton`, old checkpoints carrying `compiled`
migrate to `inplace_triton`, and failed in-place eligibility falls back to dense
eager tensor semantics with the concrete in-place fallback reason.
