---
type: retired
status: rejected
related_code:
  - ../../../src/marulho/core/fused_route_vote_cuda.py
  - ../../../src/marulho/training/column_transition_runtime.py
  - ../../../src/marulho/training/cuda_graph_route_transition.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/hot-path-latency.md
related_papers: []
related_benchmarks:
  - ../../../reports/direct_route_vote_20260614/stress-8192-profile-direct-interval32.json
  - ../../../reports/direct_route_vote_20260614/stress-32768-clean-direct-interval32.json
  - ../../../reports/direct_route_vote_20260614/stress-32768-clean-retained-interval32.json
---

# Direct Route/Vote Fusion

The direct one-block Triton route/vote fusion for the live `1024 x 64`
exact-cache route tensor is rejected.

The candidate folded route-score computation, top-k selection, and predictive
vote into one Triton program. It passed CUDA parity, but complete runtime
evidence regressed: the 8192-token profiled direct run measured
`2381.587 tokens/sec` versus `2408.630` before direct selection, and the
32768-token clean direct run measured `2266.882` versus `2359.929` after
reverting to the retained two-stage path.

The maintained runtime keeps `two_stage_route_vote` and exposes
`route_vote_kernel_variant` through Runtime Truth. Revisit this only inside a
lower-level device-owned multi-tick executor or persistent sequence kernel that
reduces the real per-token graph/kernel launch boundary and wins long
complete-runtime evidence.
