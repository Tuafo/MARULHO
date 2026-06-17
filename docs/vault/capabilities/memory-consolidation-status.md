---
type: capability
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/memory_consolidation_runner.py
  - ../../../src/marulho/training/trainer.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
---

# Memory Consolidation Status

Memory/consolidation status separates CPU archival records from device-local
replay computation.

## Evidence Rule

Do not claim consolidation as improved unless the linked quality gate improves
prediction, grounding, or reconstruction while the hot path stays protected.

Current status:

- Archival memory, tags, PRP, timestamps, bucket ids, and replay-selection
  scoring stay CPU-resident.
- Active replay still moves tensors to the model device only when sleep replay
  actually applies plasticity.
- Checkpoints preserve `last_sleep_replay_selection_report` so replay-window
  evidence survives restore.
- The 2026-06-17 benchmark retired zero-pressure replay but did not pass the
  reconstruction quality gate.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Replay Cost](../benchmarks/replay-cost.md)
