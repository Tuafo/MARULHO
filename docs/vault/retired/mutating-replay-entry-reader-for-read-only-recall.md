---
type: retired
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/training/memory_consolidation_runner.py
related_docs:
  - ../concepts/replay-window.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/read-only-recall-row-telemetry-retired.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hf-query-row-reader-retired/summary.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-read-only-recall-row.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-read-only-recall-row-rerun.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-row-reader-telemetry-retired-noprofile.json
---

# Mutating Replay Entry Reader And Direct Runner Rows For Read-Only Recall

Sleep associative recall used to prepare query tensors with
`DualMemoryStore.replay_entry(...)`. That method intentionally advances STC
state for mutating replay/consolidation work, so it contradicted the
`bounded_sleep_replay_associative_recall.v1` claim that recall was read-only.
The recall window also used the default replay selector, which could decay STC
state while reporting no recall mutation. HF replay-query collection also
selected bounded indices and then read `slow_input_patterns` directly inside the
runner, leaving a second row path outside the store-owned recall reader.

The maintained read-only path is:

- `DualMemoryStore.replay_recall_row(...)` for selected query tensor rows under
  `bounded_replay_recall_row.v1`.
- `DualMemoryStore.recall_replay_window(...)` with
  `select_replay_window(..., advance_stc_state=false)`.
- `MarulhoTrainer._sleep_replay_associative_recall(...)` reports row-reader and
  selector state-advance counts.
- `memory_consolidation_runner._collect_anchor_replay_queries(...)` uses
  `DualMemoryStore.replay_recall_row(...)` after bounded anchor-bucket query
  selection and reports `direct_slow_memory_input_pattern_reads_retired=true`.
- Active report schemas do not keep compatibility fields for the deleted
  generic replay row reader.

Evidence:

- Focused tests prove associative recall does not call `replay_entry` and does
  not advance `_state_token`, capture tags, local PRP, global PRP, or bucket
  PRP. HF query-collection tests prove selected query payloads use the
  store-owned read-only row reader and do not directly index slow-memory input
  patterns in the runner.
- The external local replay report
  `..\..\MARULHO_reports\bounded_replay_window_20260622\read-only-recall-row-telemetry-retired.json`
  passed positive-pressure recall with `1` query, mean best input-pattern
  distance `5.96046447753906e-08`,
  `query_row_reader=DualMemoryStore.replay_recall_row`,
  `query_row_state_advance_count=0`,
  `recall_selection_state_advance_count=0`, `read_only_replay_row=true`,
  `recall_selection_read_only=true`, and `mutates_runtime_state=false`.
- `..\..\MARULHO_reports\bounded_replay_window_20260622\hf-query-row-reader-retired\summary.json`
  kept the memory-consolidation gate passing while proving HF query collection
  uses the same row reader with `query_row_read_count=1`; its bounded recall
  gate remained false, so it is not quality-promotion evidence.
- The paired `524288`-token protection runs stayed same-band at `5872.559` and
  `5943.110 tokens/sec`, kept route scoring at `12/65536`, cached `65526`
  transition rows, used CUDA on the RTX 3060, and recorded zero
  graph/native/sequence failures.
- The telemetry-retirement `524288`-token rerun stayed same-band at
  `5819.770 tokens/sec`, skipped `524288` graph consolidation lookups, kept
  route scoring bounded to `12/65536`, selected CUDA on the RTX 3060, and had
  zero native sequence-loop fallback.

Do not reintroduce `replay_entry(...)` or direct `slow_*` runner reads as
read-only recall query readers. Mutating replay/consolidation must use named
mutating row APIs such as `sleep_repair_replay_row(...)` when the report says
mutation/plasticity authority is present.
