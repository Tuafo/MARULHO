---
type: retired
status: rejected
related_code:
  - ../../../src/marulho/training/column_transition_runtime.py
  - ../../../src/marulho/core/inplace_column_cuda.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/hot-path-latency.md
related_papers: []
related_benchmarks:
  - ../../../reports/steady_state_column_transition_cuda_1024/steady-state-column-transition-benchmark.json
  - ../../../reports/inplace_column_cuda_1024/inplace-column-cuda-benchmark.json
---

# Functional Full-State Transition

The functional full-state steady-state column transition is rejected and removed.

The old implementation returned full replacement tensors for prototypes,
thresholds, homeostasis state, and predictive state, then copied them back into
runtime-owned buffers. It was useful as isolated evidence, but complete
hot-window runs did not beat the retained runtime and first-use compile cost was
large.

The maintained path is `ColumnTransitionRuntime` plus the in-place CUDA/Triton
and CUDA Graph transition lifecycle. Parity tests compare that promoted kernel
against retained module semantics without keeping a rejected full-state core
implementation or its standalone benchmark runner.

Revisit only with a new evaluator that does not preserve full-state
output/writeback as the architectural shape and wins complete-runtime evidence.
