---
type: retired-path
status: rejected
related_code:
  - ../../../src/marulho/training/cuda_graph_route_transition.py
related_docs:
  - ../../retired-paths.md
  - ../../adr/0006-persistent-text-tick-executor.md
related_papers:
  - ../../research-living-brain.md
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
---

# Partial Native Parent Graph

Defaulting native repeated-child CUDA Graph parent launches for non-eight-token
burst tails is rejected for now. The opt-in path preserved CUDA parity and
proved `[2, 8]` parent graph token-count coverage, but the complete
16640-token `tick_tokens=130` stress run was slower than leaving partial tails
on the retained Python replay fallback.

The important evidence was not only the small partial replay result. The same
unaligned 130-token tick shape forced `384` `host_truth_boundary` burst
fallbacks, so the maintained fast path remains aligned 128-token source ticks,
16-token execution quanta, exact eight-token native parent graphs, and 32-token
truth/event cadence.

Revisit only through a startup-warmed or lower-level device-owned multi-tick
executor that avoids hot-path compile, preserves bounded Runtime Truth
freshness, and wins long complete-runtime evidence.
