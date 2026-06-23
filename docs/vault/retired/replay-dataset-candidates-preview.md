---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/runtime_evidence.py
  - ../../../src/marulho/evaluation/service_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/replay-dataset-candidates-retired-source-window.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-dataset-candidates-retired.json
---

# Replay Dataset Candidates Preview

`GET /terminus/replay-dataset/candidates` is retired and removed. It was a
second public replay-plan candidate preview beside `/terminus/replay-plan` and
the bounded replay-dataset preview/bundle export path, but it did not add
dataset source-window evidence.

Superseded replacement:

- The whole service advisory replay lane is now retired: `/terminus/replay-plan`,
  `/terminus/replay-dataset/preview`, and `/terminus/replay-dataset/bundle`
  are deleted too.
- Use trainer/SNN slow-window evidence and ReplayController artifacts,
  regeneration permits, sleep-plasticity review tickets, scheduler tickets, and
  transition-memory replay artifacts for bounded replay/consolidation review.
- Use `/terminus/runtime-traces/export` only for trace-only export.

Current evidence:

- Focused API coverage verifies the retired route returns `404`.
- `bounded_replay_dataset_preview_source_window.v1` now records
  `candidate_context_source=replay_plan_summary_inside_dataset_preview`,
  `retired_public_candidate_preview_endpoint=/terminus/replay-dataset/candidates`,
  and `replacement_candidate_endpoint=/terminus/replay-plan`.
- `reports/bounded_replay_window_20260622/replay-dataset-candidates-retired-source-window.json`
  passed with `50/50` target/link parity, CPU archival/source/link placement,
  no live-tick or every-token work, no replay text, no hidden language
  reasoning, and CUDA available but unused.
- The paired `524288`-token run stayed in band at `6131.415 tokens/sec` with
  bounded `12/65536` route scoring, `65526` cached rows, no observed
  contention, flat RTX 3060 memory, and zero graph/native/sequence failures.

Revisit only with a measured candidate-export contract that proves unique value
over the trainer/SNN slow-window path, carries explicit source-window evidence,
and preserves repeated long-run throughput.
