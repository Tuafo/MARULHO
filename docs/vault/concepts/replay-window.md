---
type: concept
status: active
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json
  - reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json
---

# Replay Window

## Definition

A bounded, measured set of memory entries selected for review or isolated sleep
replay before any consolidation or plasticity authority.

Current runtime surfaces: `bounded_replay_window_selection.v1` and
`bounded_replay_window_recall.v1`.

## Rules

- Replay-window selection must run in explicit slow-path windows, not the live
  tick.
- A column-anchored replay window must score only memory entries attached to
  the supplied bucket ids through the bucket index.
- A global slow-path scorer must report `global_slow_path_score_scan`; it must
  not hide a full-memory scorer. Production deep replay cannot mutate from that
  unscoped path.
- Emergency repair replay follows the same anchor-bucket rule. Without anchor
  buckets it must report `no_anchor_bucket_scope_for_repair_replay` and apply
  no mutation.
- Micro maintenance follows the same anchor-bucket rule. Without anchor buckets
  it must report `no_anchor_bucket_scope_for_micro_replay` and apply no refresh.
  Anchored micro refresh updates CPU metadata only; it must not call the live
  competitive plasticity path.
- Zero-pressure replay is retired. No replay updates apply when the best score
  is zero.
- Replay-window recall is non-mutating: it can compare routing keys and stored
  input patterns inside the selected window, but it cannot apply plasticity or
  run from the live tick.
- Positive-pressure deep replay may apply bounded candidate repair only after a
  local reconstruction gate improves over selected replay-window routing keys.
  Candidate columns come from bounded route candidates plus explicit stored
  bucket fallback candidates; rejected commits are evidence, not silent success.
- Archival metadata stays CPU-resident; active replay tensors move to the model
  device only when replay is actually applied.

## Evidence

`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair.json`
proves the selection/retirement guardrail, passes the bounded stored
input-pattern recall gate for positive-pressure windows
(`5.960464477539063e-08` mean input distance, threshold `0.01`), and passes the
prototype reconstruction gate after bounded candidate repair. The positive arm
committed `6` repairs, rejected `14` non-improving commits, scored at most `11`
candidate columns for `5` unique traces, and moved Task-A reconstruction from
`0.0052170157` after Task B to `0.0034434795` after consolidation. It also
proves deep replay blocks global mutation when there are no anchor buckets or
the anchored bucket window has zero positive pressure. The matching longer
hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json`
keeps the 65536-column live tick in band at `6306.507 tokens/sec`.

`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-repair.json`
reconfirms the same recall/prototype result after emergency repair mode was
bounded to anchor buckets. The focused repair test proves an anchored repair can
still reanchor a disturbed prototype, while no-anchor repair records
`no_anchor_bucket_scope_for_repair_replay` and leaves prototypes unchanged. The
current-tree hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json`
kept the live tick in band at `6252.073 tokens/sec`.

`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json`
keeps the same positive-pressure recall/prototype result after removing the old
micro-maintenance zero-LR competitive refresh. The focused micro tests prove
anchored micro refresh uses `bucket_indexed_candidate_window`, reports
`bounded_micro_maintenance_refresh`, bypasses `competitive.process(...)`, and
leaves prototypes/input weights unchanged; no-anchor micro refresh records
`no_anchor_bucket_scope_for_micro_replay` and leaves replay counters/tags
unchanged. The 262144-token hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json`
kept the live tick in band at `6332.439 tokens/sec`.

The less-synthetic HF-backed report
`reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json`
adds `bounded_replay_window_hf_recall_summary.v1` to the consolidation runner.
It snapshots stored Task-A anchor-window input patterns and measures recall
after Task B and after consolidation without replaying text or mutating runtime
state. The after-consolidation recall gate passed over `3` queries from `3`
anchor buckets, scored `3` CPU entries, used `max_candidates=32`, reached
`0.0` mean input-pattern distance and `1.9868214925130207e-08` mean routing-key
distance, and kept per-query latency around `0.82-1.04 ms`. The broader
reconstruction recovery gate still failed on this small HF run, so this is
bounded stored-experience recall evidence, not a consolidation-quality
promotion.

## Relationships

- [Subcortex](subcortex.md)
- [Runtime Truth](runtime-truth.md)
- [Runtime Evidence](runtime-evidence.md)
- [Replay Cost](../benchmarks/replay-cost.md)

## Source Links

- [CONTEXT.md](../../../CONTEXT.md)
- [README.md](../../../README.md)
- [Research notes](../../research-living-brain.md)

## Ambiguity

Keep claims evidence-gated. Do not widen this term into a generic programming or
biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
