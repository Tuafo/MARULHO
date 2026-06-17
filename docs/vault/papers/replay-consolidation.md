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
  - reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json
  - reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json
  - reports/bounded_replay_window_20260617/synthetic-replay-tensor-payload-boundary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json
  - reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json
  - reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json
  - reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json
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
bucket-to-entry index, and the bucket-indexed path now caps the candidate window
before scoring. The active policy reports
`candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
`candidate_window_limit=max(requested_count,candidate_pool)`,
`candidate_index_available_count`, and the scored `candidate_index_count`, so a
hot local bucket remains bounded. If no bucket scope is available, selection now
returns empty by default with
`fallback_reason=global_score_scan_requires_explicit_diagnostic_opt_in`.
The full slow-memory scorer is available only when a caller explicitly sets
`allow_global_score_scan=true`, and that diagnostic report must say
`global_slow_path_score_scan`. Unscoped random candidate scans are also retired
by default and can run only as explicit diagnostics that report
`global_slow_path_candidate_scan`.

`DualMemoryStore.recall_replay_window(...)` records
`bounded_replay_window_recall.v1`. It is a non-mutating slow-path local memory
operator over the selected replay window: routing keys and optional input
patterns stay CPU-normalized for archival recall evidence, `runs_live_tick=false`,
`mutates_runtime_state=false`, and no plasticity is applied.

`DualMemoryStore.collect_replay_query_indices(...)` records
`bounded_replay_query_collection.v1`. HF replay recall now collects Task-A
anchor queries through the same bucket-indexed recent round-robin candidate
window instead of walking `slow_bucket_ids` linearly until enough anchors are
found. The collector caps the candidate window at `max_queries`, requires input
patterns by default, reports available versus collected query indices, and
records `score_count=0`, no global scans, CPU archival placement, and
`runs_live_tick=false`.

Explicit query/readout recall follows the same literature boundary. Modern
Hopfield-style matching is useful only as a bounded local associative operator,
so `query_runner.memory_matches_with_report(...)` records
`bounded_query_memory_match.v1`: routing supplies candidate bucket ids,
`DualMemoryStore.collect_query_memory_match_indices(...)` returns a capped
bucket-indexed memory window, and the query runner computes similarity plus
replay-priority scores only for those entries. This keeps readout recall from
becoming an archive-wide hidden reasoning pass.

Awake-ripple replay tagging is treated as selected synaptic tagging/capture
metadata, not as a global recent-memory operator. `ripple_tag_awake(...)` now
requires awake bucket scope for production tagging, caps candidates through the
CPU bucket/recency index, and records `bounded_awake_ripple_tag.v1` with
candidate budget, scan flags, device placement, and `runs_every_token=false`.
If awake bucket scope is absent, it returns an empty retired report instead of
scanning all memory. The retained scalar/vector recent-memory scan can run only
as an explicit diagnostic baseline through `allow_global_diagnostic=true`, where
it records `diagnostic_awake_ripple_global_tag.v1`.

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
or mutating runtime state. The guarded report
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json`
also records `reconstruction_guarded_replay_consolidation.v1`: sleep replay is
selected from the bounded anchor window, but each cycle is accepted only if a
Task-A reconstruction score does not regress after the attempted repair. The
2026-06-17 HF run attempted `9` bounded repair updates across `3` post-Task-B
cycles, rejected all `9`, rolled the model/memory snapshot back each time, and
kept effective replay updates at `0`. The memory-consolidation gate then passed,
while after-consolidation stored-experience recall still passed over `3` Task-A
queries from `3` anchor buckets with `mean_input_pattern_distance=0.0`. That
result supports bounded local stored-experience recall plus quality-gated replay
acceptance, not a claim that replay should run continuously or reason through
text.

The matching current-tree hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation.json`
processed `262144` tokens at `6606.251 tokens/sec` with
`train_compute=0.123393 ms/token`, bounded route scoring at `12/65536`,
`state_transition_cached_count=65526`, zero graph/native/sequence failures, no
observed contention, and flat `1539 MiB` GPU memory. Replay guard scoring uses
the model device inside explicit slow windows; archival replay metadata remains
CPU-resident.

The cadenced follow-up keeps the same slow-window acceptance boundary but stops
retrying an identical rejected selection after rollback. The report
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json`
records `cadence_strategy=skip_repeated_rejected_selection`: the first
post-Task-B replay cycle rejected `3` attempted repairs, then skipped `2`
repeated rejected cycles, keeping effective updates at `0` while recall and the
memory-consolidation gate stayed passing. The clean hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json`
stayed in the maintained band at `6199.988 tokens/sec`,
`train_compute=0.130574 ms/token`, bounded route scoring at `12/65536`, `65526`
cached transition rows, zero graph/native/sequence failures, and no observed
contention. This retires repeated dead replay attempts inside a slow window; it
does not make replay continuous or live-tick work.

The target-aware replay-strength slice keeps replay under the same guard but
lets the slow window test a bounded schedule from one snapshot before commit.
`reconstruction_guarded_replay_consolidation.v1` now records the
repair-strength strategy, schedule, trial budget, budget policy, per-strength
trial reports, selected strength, attempted/effective update counts, rejected
trial attempts, and cadence skips. The patched HF runner uses `[0.1]`; the
report
`reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-promoted/summary.json`
accepted `6` post-Task-B repairs, rejected `0` trial attempts, and improved
Task-A reconstruction from `0.0170305534` to `0.0149637708` while preserving
exact stored-experience recall and the memory-consolidation gate. This cut
post-B guard latency to `1040.506 ms` from the old four-low-strength
`3477.025 ms` run. A larger medium HF qualification at
`reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-medium-2048/summary.json`
kept the same budget on `2048/2048` train tokens, `512` eval tokens, `128`
columns, and `2048` memory capacity: post-Task-B consolidation accepted `28`
repairs, rejected `0` trial attempts, improved Task-A reconstruction from
`0.0103354922` to `0.0101451825`, passed bounded recall with
`mean_input_pattern_distance=0.0`, and passed the consolidation gate. Checkpoint
reload restored the bounded recall and selection reports with
`runs_live_tick=false`, keeping replay evidence in the slow path after save/load.
The synthetic stress benchmark now defaults to compact
escalation `[0.1, 0.5, 1.0]`: `reports/bounded_replay_window_20260617/synthetic-target-strength-budget-compact-default.json`
passes recall and prototype gates with `repair_strength_trial_budget=3`, while
the single-strength synthetic control is rejected as a universal default
because it failed the prototype gate. The clean hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-target-strength-budget-compact.json`
processed `262144` tokens at `6232.282 tokens/sec`, kept route scoring bounded
at `12/65536`, cached `65526` state-transition rows, had zero
graph/native/sequence failures, and reported no observed contention.

The replay text/SFA boundary cleanup makes the "no hidden language reasoning"
claim concrete. Sleep replay now asks `DualMemoryStore.replay_entry(...)` for
tensor payloads only by passing `include_text_payload=false`; deep candidate
repair and anchored repair do not receive `raw_window`, expanded text, or
metadata. Selection and recall reports expose `raw_text_payload_loaded=false`
and `language_reasoning=false`, while the sleep replay report exposes
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`, and
`sleep_replay_text_payload_policy=sleep_replay_uses_tensor_payloads_only`.
Deep sleep with an abstraction layer also samples SFA correction from the
processed replay indices instead of the whole slow buffer, with
`sleep_replay_sfa_correction_scope=selected_replay_window` and
`sleep_replay_sfa_full_memory_sample_retired=true`. The helper defaults enforce
the same rule: `sample_replay_indices(...)` now returns no indices for unscoped
calls unless the caller explicitly sets `allow_global_score_scan=true`, and
`sample_for_sfa(...)` returns no samples without selected candidate indices
unless `allow_global_diagnostic=true` marks the call as diagnostic. The
synthetic boundary report
`reports/bounded_replay_window_20260617/synthetic-replay-tensor-payload-boundary.json`
kept bounded recall and prototype gates passing with `2` accepted post-B
repairs and `0` updates in the zero-pressure/no-anchor controls. The matching
262144-token hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json`
stayed in band at `6237.420 tokens/sec`, scored `12/65536` route rows, cached
`65526` transition rows, reported no observed contention, kept flat `1719 MiB`
GPU memory, and had zero graph/native/sequence failures. Replay therefore
remains bounded associative memory inside explicit slow windows, not a text
reasoning loop. The follow-up helper-default retirement gate confirmed the
live tick still stayed protected after unscoped helper defaults were removed:
the clean 262144-token active-pressure rerun reached `5668.688 tokens/sec`,
`train_compute=0.141909 ms/token`, bounded route rows at `12/65536`, cached
`65526` transition rows, no observed contention, and zero graph/native/sequence
failures.

The capped replay-candidate window follow-up tightens the selected-window
boundary for future larger memory. The store now keeps per-bucket entry indices
in recency order and collects recent entries round-robin across anchor buckets
until the candidate window limit is reached, before any maintenance,
consolidation, or repair scores are computed. A focused hot-bucket test shows
`10` available entries cut to `candidate_window_limit=4` and `score_count=4`;
the older high-importance entry is not selected because it never enters the
bounded candidate window. The synthetic replay benchmark
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json`
kept the positive-pressure recall/prototype gates passing with CPU archival and
CPU selection scoring, no global score/candidate scan, and `0` updates in the
zero-pressure/no-anchor controls. The 262144-token hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json`
stayed in band at `6148.125 tokens/sec`, scored only `12/65536` route rows,
cached `65526` transition rows, reported no observed contention, kept GPU memory
flat at `1848 MiB`, and had zero graph/native/sequence failures.

The capped replay-query collection follow-up removes another old scan shape
from the HF recall runner. The report
`reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json`
collected `3` Task-A anchor queries from `3` available bucket-indexed entries
under `candidate_window_limit=16`, scored `0` entries during collection,
reported no global score/candidate scan, and kept query collection on CPU with
`runs_live_tick=false`. After-consolidation stored-experience recall passed
with `mean_input_pattern_distance=0.0` and
`mean_routing_key_distance=1.98682149251302e-08`; guarded consolidation accepted
`6` post-Task-B repairs, rejected `0`, and improved target reconstruction
quality from `0.0234637554` to `0.0213608844`. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json`
stayed in band at `6221.949 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, flat `1848 MiB` GPU memory, no observed
contention, and zero graph/native/sequence failures.

The query-memory match follow-up removes the full slow-buffer query readout
scan. The report
`reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json`
emits `bounded_query_memory_match.v1`, used `candidate_window_limit=192`, had
`1` available bucket-indexed candidate, computed `1` similarity score and `1`
bounded replay-priority score, returned `1` match, reported no global
score/candidate scan, and kept archival placement on CPU with
`runs_live_tick=false`. The top memory was
`promoted scheduler checkpoint route-bank seed` with similarity
`0.9932903051`. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json`
processed `262144` tokens at `6137.185 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1848 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

The recent replay tag/anchor setup follow-up applies the same literature
boundary to STC/PRP setup itself: tags and anchors are useful only when selected,
bounded, and cadenced. `DualMemoryStore.collect_recent_entry_indices(...)`
maintains a CPU recency index and records `bounded_recent_memory_window.v1`.
`tag_recent_entries(...)` records `bounded_recent_memory_tag.v1`, while
`capture_recent_memory_anchors(...)` records `bounded_recent_anchor_capture.v1`
and requires bucketed entries before creating column anchors. This retires the
old archive-linear timestamp/bucket walk for recent replay setup without moving
archival metadata to CUDA. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json`
kept recall and prototype gates passing, used `candidate_window_limit=256` for
both recent tagging and anchor capture, touched `14` indexed entries, captured
`4` anchors, reported no global score/candidate scan, and kept
`runs_live_tick=false` with CPU archival storage. The matching 65536-column
hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json`
processed `262144` tokens at `6228.243 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1846 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

The bounded awake-ripple follow-up applies the same rule to ripple priority
tagging. The direct benchmark
`reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json`
compared the diagnostic global scan with a wake-bucket candidate window over
`256` iterations on an `8192`-entry ledger: diagnostic global tagging averaged
`1.433332 ms` with `256` vector scans, while scoped tagging averaged
`1.091997 ms`, used `0` scalar/vector scans, used `256` awake-bucket scans, and
touched `10` final candidate entries (`1.312579x`). The synthetic replay report
`reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json`
kept recall/prototype gates passing and recorded
`last_awake_ripple_tag_report` with `global_candidate_scan=false` and
`runs_every_token=false`. The longer 65536-column hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json`
processed `524288` tokens at `6152.328 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `2013 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

The replay-score helper cleanup removes a leftover archive-wide scoring API.
The replay-priority formula remains, but callers must now use
`replay_scores_for_indices(...)` with explicit candidate indices. That keeps
priority ranking inside selected replay/query windows and avoids leaving a
public full-buffer scorer beside the bounded selector. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json`
kept the positive-pressure recall/prototype gates passing with `2` bounded
updates and `0` global fallback cycles. The matching 65536-column hot-path
report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json`
processed `262144` tokens at `6211.859 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1852 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

## Status

bounded slow-path selection, stored-experience recall, reconstruction-gated
candidate repair, reconstruction-guarded HF replay acceptance, skipped repeated
rejected replay attempts, target-specific repair-strength budgets, tensor-only
sleep replay payloads, selected-window SFA correction, capped pre-score replay
candidate windows, capped replay query collection, bounded query-memory
readout, bounded recent tag/anchor setup, bounded awake-ripple tagging, and
retired unscoped random replay defaults plus the full-buffer replay-score helper
implemented; future larger replay windows still require repeated long-run
hot-path and grounding checks

## Links

- [Research notes](../../research-living-brain.md)
- [Column Runtime](../concepts/column-runtime.md)
- [Replay Cost](../benchmarks/replay-cost.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)
