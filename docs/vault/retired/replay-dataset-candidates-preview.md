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

Replacement:

- Use `/terminus/replay-plan` for replay-plan candidates.
- Use `/terminus/replay-dataset/preview` for dataset/export source-window
  evidence.
- Use `/terminus/replay-dataset/bundle` for operator-approved package previews.

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
over `/terminus/replay-plan`, carries explicit source-window evidence, and
preserves repeated long-run throughput.
