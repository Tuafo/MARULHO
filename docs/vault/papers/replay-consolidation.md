---
type: paper
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - https://arxiv.org/abs/2008.02217
  - https://pubmed.ncbi.nlm.nih.gov/7624455/
  - https://papers.neurips.cc/paper/8327-experience-replay-for-continual-learning
  - https://pubmed.ncbi.nlm.nih.gov/9020359/
  - https://arxiv.org/abs/1912.01100
related_benchmarks:
  - reports/bounded_replay_window_20260617/synthetic-selection.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-repair.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json
  - reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json
---

# Replay/consolidation

## Claim

Replay and consolidation are slow-path mechanisms. MARULHO should select a
bounded replay window from explicit local evidence, run it only in sleep/replay
maintenance, and keep archival metadata CPU-resident unless active replay
computation benefits from CUDA.

## MARULHO Relevance

Modern Hopfield work is useful as an associative-memory operator, but in
MARULHO it must remain local: inside a column, a routed candidate set, or a
bounded replay window. Its attention equivalence is not permission to add a
transformer-like global mind or scan all memory in the live tick.

Complementary learning systems, continual-learning replay, latent/sparse replay,
and synaptic tagging/capture all point in the same engineering direction:
separate fast live plasticity from slower replay/consolidation; replay selected
compressed evidence rather than raw unbounded history; and promote memories only
when tags/PRP/replay pressure are positive enough to justify the cost.

## Implementation Implication

`DualMemoryStore.select_replay_window(...)` records
`bounded_replay_window_selection.v1`. When deep sleep has column anchors, the
selection scores only entries attached to those bucket ids through the
bucket-to-entry index. If no bucket scope is available, the report says
`global_slow_path_score_scan` so the full slow-memory scorer is not hidden.

`DualMemoryStore.recall_replay_window(...)` records
`bounded_replay_window_recall.v1`. It is a non-mutating slow-path local memory
operator over the selected replay window: routing keys and optional input
patterns stay CPU-normalized for archival recall evidence, `runs_live_tick=false`,
`mutates_runtime_state=false`, and no plasticity is applied.

Zero-pressure replay is now retired: if the global scorer finds no positive
consolidation/repair/maintenance pressure, it returns an empty selection with
`fallback_reason=no_positive_global_scores` instead of rehearsing arbitrary
zero-score entries.

Deep-sleep consolidation no longer uses the global scorer as a production
mutation fallback. When no anchor buckets exist, or when the anchor-bucket
window has no positive replay pressure, the trainer records
`unscoped_global_fallback_retired=true`, leaves `sleep_replay_applied_count=0`,
and does not apply plasticity.

Emergency repair sleep now follows the same anchor-bucket boundary. Repair mode
uses `bounded_repair_reanchor` only when anchors provide a bucket-indexed replay
window; without anchors it records
`global_fallback_blocked_reason=no_anchor_bucket_scope_for_repair_replay` and
applies no mutation. This retires the remaining unscoped repair-global mutation
path while preserving anchored repair as an explicit slow-path operation.

Micro maintenance now follows the same anchor-bucket boundary. Unanchored micro
refresh records
`global_fallback_blocked_reason=no_anchor_bucket_scope_for_micro_replay` and
applies no tag/replay-count refresh. Anchored micro refresh reports
`bounded_micro_maintenance_refresh`, selects through the bucket index, updates
CPU memory metadata only, and bypasses the old zero-LR
`CompetitiveColumnLayer.process(...)` call.

Positive-pressure deep replay no longer commits the old stored-bucket
`CompetitiveColumnLayer.process(...)` mutation. The promoted slow-window commit
path is `bounded_reconstruction_gated_candidate_repair`: selected replay entries
are de-duplicated into local traces, candidate columns come from bounded routing
candidates plus an explicit stored-bucket fallback candidate, and a temporary
prototype repair is committed only when it improves
`mean_one_minus_best_similarity_over_selected_replay_routing_keys` inside the
selected replay-window candidate columns. The report exposes candidate-column
budget, trial count, rejected commits, updated columns, quality before/after,
CPU score device, CPU archival storage, and `runs_live_tick=false`.

The 2026-06-17 synthetic candidate-repair benchmark
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair.json`
now separates stored-experience recall from bounded repair and passes both gates
for the positive-pressure arm. It recalled stored Task-A input patterns with
mean distance `5.960464477539063e-08` under the `0.01` gate, committed `6`
bounded candidate repairs across `4` consolidation cycles, rejected `14`
non-improving commits, and improved Task-A reconstruction from `0.0052170157`
after Task B to `0.0034434795` after consolidation. The prototype gate passed
with relative degradation `0.0467838377` under the `0.05` threshold and overlap
`0.8981397152`. The zero-pressure guard and no-anchor global-control arms still
applied `0` updates.

The matching longer hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json`
processed `262144` tokens at `6306.507 tokens/sec`, with
`train_compute=0.129511 ms/token`, `route_input_rows_scored=12/65536`,
`state_transition_cached_count=65526`, zero graph/native/sequence failures, and
no observed contention. Replay repair remains an explicit sleep/replay window;
it is not every-token background work.

After the micro-maintenance cleanup, the current synthetic report
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json`
kept the same recall/prototype gates and the 262144-token hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json`
stayed in band at `6332.439 tokens/sec` with zero graph/native/sequence
failures. Micro refresh is now bounded CPU metadata maintenance, not hidden
competitive replay.

The less-synthetic HF-backed consolidation runner now records
`bounded_replay_window_hf_recall_summary.v1`. It snapshots stored Task-A
anchor-window input patterns and recalls them after Task B and after
consolidation through the same CPU bucket-index operator, without replaying text
or mutating runtime state. The report
`reports/bounded_replay_window_20260617/hf-recall-bounded-window/summary.json`
passed the after-consolidation recall gate over `3` Task-A replay queries from
`3` anchor buckets with `mean_input_pattern_distance=0.0`, but the broader
reconstruction recovery gate remained false. That result supports bounded local
stored-experience recall, not a claim that consolidation has solved HF
forgetting/reconstruction.

## Status

bounded slow-path selection, stored-experience recall, and reconstruction-gated
candidate repair implemented; long-run hot-path protection remains required for
future larger replay windows

## Links

- [Research notes](../../research-living-brain.md)
- [Column Runtime](../concepts/column-runtime.md)
- [Replay Cost](../benchmarks/replay-cost.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)
