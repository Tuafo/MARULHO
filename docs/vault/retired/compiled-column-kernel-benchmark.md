---
type: retired
status: retired
related_code:
  - ../../../src/marulho/evaluation/compiled_hot_path_kernel_benchmark.py
  - ../../../src/marulho/training/column_transition_runtime.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/hot-path-latency.md
related_papers: []
related_benchmarks:
  - ../../../reports/compiled_column_kernel_cuda_1024/compiled-column-kernel-benchmark.json
---

# Compiled Column Kernel Benchmark

The competition-only compiled column-kernel benchmark runner is removed.

It measured candidate-scoped competition after removing projection, retrieval,
predictive state, runtime mutation, trainer orchestration, memory, binding,
grounding, checkpointing, and service cost. That was useful early evidence, but
the wider compiled hot-path benchmark and the promoted route/vote/transition
lifecycle now cover the meaningful direction.

Historical numbers remain in the hot-path benchmark note. Future promotion work
should use wider fusion probes or complete runtime stress gates rather than
reviving a competition-only runner.
