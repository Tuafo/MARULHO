---
type: concept
status: draft
related_code:
  - ../../../src/marulho/core/column_runtime.py
  - ../../../src/marulho/training/model.py
related_docs:
  - ../modules/core.md
  - ../concepts/runtime-truth.md
related_papers:
  - ../papers/predictive-coding.md
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
---

# Column Runtime

Column Runtime is the MARULHO control-plane direction where many small predictive columns can exist, but only a bounded subset should wake on a tick.

The current implementation is report-only. `core.column_runtime` reads existing predictive-column and competitive-column tensors, then reports:

- total columns and awake budget
- awake, cached-vote, sleeping, and deep-sleeping counts
- bounded column votes with confidence, prediction error, usefulness, cost, disagreement, and wake reason
- growth-gate evidence from repeated prediction error
- pruning/homeostasis evidence from weak, idle, or redundant columns

It does not schedule execution yet, grow columns, prune columns, write checkpoints, mutate topology, or claim Thousand-Brains completeness. Runtime Truth exposes a compact projection so operators can see whether awake columns remain small before any execution scheduler is promoted.

## Latest Local Evidence

On 2026-06-10, live Runtime Truth after local backend restart reported `total_columns=1024`, `awake_budget=10`, `awake_count=10`, `runs_all_columns=false`, `vote_count=10`, growth gate `ready=false`, pruning/homeostasis `ready=false`, and claim boundary `column_scheduler_evidence_only_not_sparse_execution_promotion`.

The explicit service benchmark at ignored `reports/service_benchmark_column_runtime/service-benchmark.json` also captured the compact Runtime Truth evidence. That run succeeded, but CUDA/status-sidecar latency remained weak: hot-path p95 `2367.951 ms`, hot-path total `4017.128 ms`, and `/status` latency `30161.935 ms`. This means the column evidence is visible, but execution scheduling and status-scope cost still need benchmarked improvement before promotion.

## Next Gate

Promotion from report-only evidence to execution scheduling needs a benchmark that proves fewer awake columns reduce hot-path cost without degrading prediction, grounding, Runtime Truth, or rollback guarantees.

## Links

- [Runtime Truth](runtime-truth.md)
- [Metabolism](metabolism.md)
- [Hot Path](hot-path.md)
- [Dynamic Growth](dynamic-growth.md)
- [Pruning](pruning.md)
- [Core module](../modules/core.md)
