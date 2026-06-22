---
type: retired
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
related_docs:
  - ../concepts/replay-window.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/sleep-replay-read-only-recall-row.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-read-only-recall-row.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-read-only-recall-row-rerun.json
---

# Mutating Replay Entry Reader For Read-Only Recall

Sleep associative recall used to prepare query tensors with
`DualMemoryStore.replay_entry(...)`. That method intentionally advances STC
state for mutating replay/consolidation work, so it contradicted the
`bounded_sleep_replay_associative_recall.v1` claim that recall was read-only.
The recall window also used the default replay selector, which could decay STC
state while reporting no recall mutation.

The maintained read-only path is:

- `DualMemoryStore.replay_recall_row(...)` for selected query tensor rows under
  `bounded_replay_recall_row.v1`.
- `DualMemoryStore.recall_replay_window(...)` with
  `select_replay_window(..., advance_stc_state=false)`.
- `MarulhoTrainer._sleep_replay_associative_recall(...)` reports row-reader and
  selector state-advance counts.

Evidence:

- Focused tests prove associative recall does not call `replay_entry` and does
  not advance `_state_token`, capture tags, local PRP, global PRP, or bucket
  PRP.
- The external local replay report
  `..\..\MARULHO_reports\bounded_replay_window_20260622\sleep-replay-read-only-recall-row.json`
  passed positive-pressure recall with `1` query, mean best input-pattern
  distance `5.96046447753906e-08`, `query_row_state_advance_count=0`,
  `recall_selection_state_advance_count=0`, `read_only_replay_row=true`,
  `recall_selection_read_only=true`, `replay_entry_reader_used=false`, and
  `mutates_runtime_state=false`.
- The paired `524288`-token protection runs stayed same-band at `5872.559` and
  `5943.110 tokens/sec`, kept route scoring at `12/65536`, cached `65526`
  transition rows, used CUDA on the RTX 3060, and recorded zero
  graph/native/sequence failures.

Do not reintroduce `replay_entry(...)` as a read-only recall query reader.
Mutating replay/consolidation may still use explicit replay-entry paths when
the report says mutation/plasticity authority is present.
