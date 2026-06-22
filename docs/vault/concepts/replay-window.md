---
type: concept
status: active
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/evaluation/source_bank_memory_match_benchmark.py
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
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json
  - reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json
  - reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json
  - reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json
  - reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json
  - reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260618/source-bank-memory-match-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json
  - reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json
  - reports/bounded_replay_window_20260617/frontier-gap-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json
  - reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json
  - reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json
  - reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json
---

# Replay Window

## Definition

A bounded, measured set of memory entries selected for review or isolated sleep
replay before any consolidation or plasticity authority.

Current runtime surfaces: `bounded_replay_window_selection.v1`,
`bounded_replay_window_recall.v1`, and
`bounded_replay_query_collection.v1`. Explicit query/readout recall also uses
`bounded_query_memory_match.v1`, and autonomy source-acquisition planning uses
`bounded_concept_frontier_memory_metrics.v1`. ConceptStore semantic observation
uses `bounded_concept_memory_signature_lookup.v1` when it resolves memory
signatures from already-selected evidence. Recent replay setup uses
`bounded_recent_memory_window.v1`, `bounded_recent_memory_tag.v1`, and
`bounded_recent_anchor_capture.v1`. Semantic frontier planning uses
`bounded_frontier_gap_selection.v1`. Awake replay-priority tagging uses
`bounded_awake_ripple_tag.v1`. Source-bank semantic recall uses
`bounded_source_bank_memory_match.v1`.

## Rules

- Replay-window selection must run in explicit slow-path windows, not the live
  tick.
- A column-anchored replay window must score only memory entries attached to
  the supplied bucket ids through the bucket index.
- Bucket-indexed selection must cap candidates before scoring. The current
  policy is `recent_bucket_round_robin_candidate_pool` with
  `candidate_window_limit=max(requested_count,candidate_pool)`, and reports both
  `candidate_index_available_count` and the actually scored
  `candidate_index_count`.
- A missing bucket scope must return `bucket_index_scope_required` with no
  mutation and no global scan. Retired full-memory scorer comparisons are
  benchmark-local only and cannot be requested through the runtime store.
- Emergency repair replay follows the same anchor-bucket rule. Without anchor
  buckets it must report `no_anchor_bucket_scope_for_repair_replay` and apply
  no mutation.
- Repair replay must not rebuild dense input assemblies or project stored
  assemblies for selected replay entries missing routing keys. Stored routing
  keys are required for repair mutation; missing keys are deferred and reported
  through `sleep_replay_missing_routing_key_deferred_count` while
  `sleep_replay_dense_input_assembly_fallback_count=0` remains true. The
  mixed-key benchmark
  `reports/bounded_replay_window_20260620/sleep-repair-replay-missing-routing-key-deferred.json`
  updated `16` stored-key entries, deferred `16` missing-key entries, made `0`
  dense input-assembly calls, and the 524288-token hot-path run stayed in band
  at `5988.223 tokens/sec`.
- Micro maintenance follows the same anchor-bucket rule. Without anchor buckets
  it must report `no_anchor_bucket_scope_for_micro_replay` and apply no refresh.
  Anchored micro refresh updates CPU metadata only; it must not call the live
  competitive plasticity path.
- Zero-pressure replay is retired. No replay updates apply when the best score
  is zero.
- Replay-window recall is non-mutating: it can compare routing keys and stored
  input patterns inside the selected window, but it cannot apply plasticity or
  run from the live tick.
- Replay query collection must use the same bucket-indexed candidate window. It
  must not walk `slow_bucket_ids` linearly to find anchors, must report
  available versus collected query indices, and must keep `score_count=0`
  because it is collection, not another scorer.
- ConceptStore signature lookup may read only evidence-provided memory indices
  from bounded query/source/concept observations. It must cap each source,
  direct-index CPU archival arrays, report `archive_list_materialization_count=0`,
  and never turn semantic observation into a global memory scan.
- Awake-ripple tagging must use awake bucket scope from the scheduler, cap its
  candidate window before mutation, and report `runs_every_token=false`.
  Unscoped production calls must return an empty retired report; the old global
  scalar/vector scan is diagnostic-only.
- Positive-pressure deep replay may apply bounded candidate repair only after a
  local reconstruction gate improves over selected replay-window routing keys.
  Candidate columns come from bounded route candidates plus explicit stored
  bucket fallback candidates; rejected commits are evidence, not silent success.
- Archival metadata stays CPU-resident; active replay tensors move to the model
  device only when replay is actually applied.
- Replay/SFA helper APIs must not drop reports. `sample_replay_indices(...)`
  and `sample_for_sfa(...)` are removed; callers use
  `select_replay_window(...)` or `sample_for_sfa_with_report(...)` and keep the
  returned bounded report.
- Query/readout memory matching must also default to bounded inputs. It derives
  candidate bucket ids from routing, collects a capped bucket-indexed memory
  window, and scores only those entries; it must not compute similarity or
  replay priority over the whole slow buffer.
- Concept-frontier source acquisition metrics must follow the same candidate
  window. They can compare probe-bank signatures against selected memory
  routing keys for novelty/uncertainty/support, but they must not iterate every
  `slow_routing_keys` entry.
- Source-bank semantic recall must merge sampled probe bucket ids into one
  bounded bank-level candidate window before scoring. It may sample source-bank
  probe patterns, but production recall must not call the query-memory matcher
  once per probe; it must collect one capped bucket-indexed window, report
  `merged_probe_candidate_window=true`, `per_probe_query_match_call_count=0`,
  candidate/window budgets, CPU archival/score placement, and keep
  `runs_live_tick=false`, `runs_every_token=false`, and
  `language_reasoning=false`.
- Semantic frontier-gap planning must also follow a selected candidate window.
  It may load raw text only for bounded candidates returned by
  `collect_frontier_gap_indices(...)`; it must not materialize
  `slow_raw_windows` or score all archival windows.
- Recent replay setup must also default to bounded inputs. Tagging and anchor
  capture collect from the CPU recency index, cap by `max_recent_entries`, and
  report candidate availability, selected indices, CPU archival placement,
  global scan flags, and `runs_live_tick=false`. Anchor capture additionally
  requires bucketed entries so slow-window replay remains column-local.
- Replay-priority scoring must be called with explicit candidate indices.
  The old public full-buffer helper is retired; selected windows should use
  `replay_scores_for_indices(...)`.

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

The unscoped-helper retirement preserves the same rule at the helper API
boundary. The list-only `sample_replay_indices(...)` and `sample_for_sfa(...)`
helpers are now deleted, while `sample_for_sfa_with_report(...)` emits
`bounded_sfa_sample.v1` with selected-window candidate/sample indices, CPU
placement, no global candidate scan, and no live-tick or language reasoning
claim. The clean 262144-token active-pressure hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json`
processed `262144` tokens at `5668.688 tokens/sec`, kept route scoring bounded
at `12/65536`, cached `65526` transition rows, reported no observed contention,
and had zero graph/native/sequence failures.

The reported SFA sample benchmark
`reports/bounded_replay_window_20260618/sfa-sample-bounded-window.json` used a
`65536`-entry archive, `192` selected replay-window candidates, and `64`
requested samples. Selected-window sample purity improved from
`0.00439453125` for the retired full-buffer sampler to `1.0`, with mean
latency `0.656 ms` versus `1.451 ms` (`2.210x`).

Query memory episodes now have their own bounded readout report. The deleted
`build_memory_episodes(...)` helper is replaced by
`build_memory_episodes_with_report(...)`, which records
`bounded_query_memory_episode_readout.v1`, selected match count, neighbor
radius, direct neighbor-window payload count, CPU readout placement, no global
candidate/score scan, no live tick, no every-token work, and no language
reasoning. The benchmark
`reports/bounded_replay_window_20260618/query-episode-readout-bounded.json`
recovered `a cat purrs when it feels safe.` from four selected fragments while
fragment-only readout returned `els safe.`; the bounded readout cost was
`0.936 ms` mean versus `0.490 ms` for fragment-only readout, with `10` direct
neighbor payloads under a `28`-entry budget.

The capped replay-candidate window follow-up makes the bucket-index rule
scalable for hot buckets. `DualMemoryStore` keeps bucket entry lists in recency
order and collects candidates by recent round-robin across anchor buckets before
scoring. The focused test proves `10` available entries can be cut to
`candidate_window_limit=4`, with `score_count=4`, before importance ranking; an
older high-importance entry outside the capped window is not selected. The fresh
synthetic report
`reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-capped-window.json`
kept positive-pressure recall/prototype gates passing and reported
`candidate_window_policy=recent_bucket_round_robin_candidate_pool`,
`candidate_window_limit=32`, `candidate_index_available_count=16`,
`candidate_index_count=16`, `score_count=16`, `score_device=cpu`,
`archival_storage_device=cpu`, and no global score/candidate scan. The matching
262144-token hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json`
stayed in band at `6148.125 tokens/sec` with `12/65536` route rows, `65526`
cached transition rows, zero graph/native/sequence failures, flat `1848 MiB`
GPU memory, and no observed contention.

The hot-bucket source construction below that capped scorer is now bounded too.
`_candidate_indices_for_bucket_ids(...)` used to build `list(reversed(...))`
for every selected bucket before returning the capped candidate list. It now
uses tail-indexed cursors and reports
`candidate_source_window_policy=tail_indexed_bucket_round_robin_no_full_bucket_materialization`,
`candidate_source_entry_read_count`, materialization counts, CPU source device,
and no full-bucket scan/materialization. The diagnostic benchmark
`reports/bounded_replay_window_20260618/bucket-candidate-source-window-bounded.json`
kept newest-candidate parity on a `65536`-entry bucket, read `32` source
indices within a `32`-entry source-read budget, materialized `0`, used
`0.0 MiB` CUDA allocation, and cut mean source latency from `0.416944 ms` to
`0.060931 ms` (`6.843x`). This source window feeds
replay selection, replay-query collection, query readout, frontier planning,
and awake ripple tagging.

The replay query-collection follow-up applies the same cap before HF recall
queries are loaded. `DualMemoryStore.collect_replay_query_indices(...)` emits
`bounded_replay_query_collection.v1`, returns recent bucket-indexed query
indices up to `max_queries`, requires stored input patterns by default, records
`candidate_index_available_count`, `candidate_index_count`, `query_indices`,
`query_count`, and `skipped_missing_input_pattern_count`, and reports
`score_count=0`, no global scans, CPU archival placement, and
`runs_live_tick=false`. The HF report
`reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json`
collected `3` Task-A anchor queries through a `candidate_window_limit=16` with
no global score/candidate scan, kept after-consolidation recall passing at
`mean_input_pattern_distance=0.0`, accepted `6` guarded repairs, and passed the
memory-consolidation gate. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json`
processed `262144` tokens at `6221.949 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, no observed contention, flat
`1848 MiB` GPU memory, and zero graph/native/sequence failures.

Explicit query/readout recall now follows the same selected-window contract.
`query_runner.memory_matches_with_report(...)` emits
`bounded_query_memory_match.v1`, while
`DualMemoryStore.collect_query_memory_match_indices(...)` emits
`bounded_query_memory_match_candidates.v1`. The query runner gets candidate
bucket ids from routing, collects recent bucket-indexed memory indices up to a
candidate limit, computes similarity and replay-priority scores only for those
indices, and records no global score/candidate scan, CPU archival placement,
`runs_live_tick=false`, and `mutates_runtime_state=false`. The report
`reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json`
used `candidate_window_limit=192`, scored `1` candidate, returned `1` memory
match, and retrieved `promoted scheduler checkpoint route-bank seed` with
similarity `0.9932903051`. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json`
processed `262144` tokens at `6137.185 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `1848 MiB`, and had zero graph/native/sequence failures.
For similarity-only query readout, the same surface now delays replay text
payload construction until after the returned set is known. The benchmark
`reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json`
kept selected indices identical to the retired eager candidate-payload shape,
loaded raw text for `5` returned matches instead of all `192` candidates, and
reduced mean latency from `33.612 ms` to `25.881 ms`. The report records
`raw_text_payload_policy=returned_similarity_matches_only` and
`language_reasoning=false`. The 524288-token hot-path check stayed in band at
`6152.079 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, flat GPU memory (`1874->1878 MiB`), no observed contention,
and zero graph/native/sequence failures.

Concept-frontier source acquisition now uses that selected-window contract too.
`concept_frontier_metrics_with_report(...)` emits
`bounded_concept_frontier_memory_metrics.v1`: routing supplies candidate bucket
ids from the probe-bank signature, the memory store returns a capped
bucket-indexed candidate window, and the frontier metric scores novelty,
uncertainty, and support only for those entries. The report
`reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json`
scored `64/8192` entries, preserved the diagnostic full-scan top-1, kept
frontier metric deltas within gate (`novelty_delta=0.0`,
`uncertainty_delta=0.0`, `support_delta=0.015893`), and reduced mean metric
latency from `658.116 ms` to `5.040 ms`. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json`
processed `262144` tokens at `6148.846 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `1805 MiB`, and had zero graph/native/sequence failures.

The source-bank signature feeding that frontier metric is now capped before
candidate-bucket lookup. `concept_frontier_metrics_with_report(...)` and
`candidate_semantic_signature(...)` sample an evenly spaced `16`-probe source
window and report `source_probe_count`, `source_probe_window_limit`,
`source_probe_indices`, and selection-budget fields. The direct report
`reports/bounded_replay_window_20260618/concept-frontier-source-probe-window-bounded.json`
sampled `16/64` probes, scored `64/16384` memory entries, preserved top-1, and
reduced mean latency from `1556.602 ms` to `7.637 ms`. The paired hot-path
check stayed in the same band as the committed baseline (`6303.548` versus
`6307.437 tokens/sec`) with bounded `12/65536` route rows, cached `65526`
transition rows, no observed contention, flat `1789 MiB` GPU memory, and zero
graph/native/sequence failures.

Source-bank semantic recall now records the same selected-window contract at
the bank-planning layer. `bank_memory_matches_with_report(...)` samples a
capped probe set, unions routing-index bucket ids, collects one CPU candidate
window capped at `192`, and scores sampled probes against that local window
before loading raw text for returned matches. It emits
`bounded_source_bank_memory_match.v1` with merged-window truth and zero
per-probe query-matcher calls. The benchmark
`reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json`
used `8` probes over a `65536`-entry store, preserved selected indices against
the retired per-probe diagnostic path (`quality.min=1.0`), reduced raw text
payload loads from `32` to `4`, and reduced mean latency from `560.177 ms` to
`106.543 ms`. The matching 524288-token hot-path run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json`
processed `6129.933 tokens/sec`, kept bounded `12/65536` route rows, cached
`65526` transition rows, reported mild GPU contention (`21%` against a `20%`
threshold), kept archival recall metadata on CPU, and had zero
graph/native/sequence failures.

ConceptStore memory-signature lookup now follows the same evidence-window rule.
`ConceptStore.observe(...)` emits `bounded_concept_memory_signature_lookup.v1`
inside its concept summary. It accepts the memory indices already selected by
query/readout/source evidence, caps each source at `8` unique indices with a
`32`-reference scan budget, direct-indexes `slow_routing_keys`,
`slow_input_patterns`, and `slow_buffer`, and reports no archive list
materialization or global candidate/score scan. The benchmark
`reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json`
used `65536` archival entries, preserved legacy signatures
(`min cosine=0.9999998212`), and reduced mean lookup latency from `12.490 ms`
to `1.454 ms`. The clean hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json`
processed `262144` tokens at `6143.768 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `1746 MiB`, and had zero graph/native/sequence
failures.

Semantic frontier-gap planning now follows the selected-window rule too.
`frontier_gap_plan(...)` no longer materializes `slow_raw_windows` or rebuilds
archive-side lists while ranking gap terms. It calls
`DualMemoryStore.collect_frontier_gap_indices(...)`, which emits
`bounded_frontier_gap_candidates.v1` from a capped CPU recency/bucket index, and
then returns `bounded_frontier_gap_selection.v1` after scoring only those
candidate raw-window payloads. The benchmark
`reports/bounded_replay_window_20260617/frontier-gap-bounded.json` used
`65536` archival entries, scored `192` bounded candidates, preserved expected
and diagnostic legacy top terms (`quality.min=1.0`), and reduced mean latency
from `217.530 ms` to `9.073 ms` (`23.975x`). The refreshed report also passes
the missing-collector retirement gate: without
`collect_frontier_gap_indices(...)`, planning returns zero candidates and zero
text payloads with no global scans. The longer hot-path report
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json`
processed `524288` tokens at `6233.085 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory at `1844->1840 MiB`, and had zero graph/native/sequence
failures.

Recent replay tag and anchor setup now use the same bounded-window discipline.
`DualMemoryStore.collect_recent_entry_indices(...)` emits
`bounded_recent_memory_window.v1`, `tag_recent_entries(...)` emits
`bounded_recent_memory_tag.v1`, and
`MarulhoTrainer.capture_recent_memory_anchors(...)` emits
`bounded_recent_anchor_capture.v1`. Focused tests cap a `10`-entry recency index
to `[9, 8, 7]` for both tagging and anchor capture, proving older entries are
not reached by a hidden archive walk. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json`
kept positive-pressure recall/prototype gates passing while the tag and anchor
reports used `candidate_window_limit=256`, `candidate_index_count=14`, no global
score/candidate scan, CPU archival storage, and `runs_live_tick=false`. The
matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json`
processed `262144` tokens at `6228.243 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `1846 MiB`, and had zero graph/native/sequence failures.

Awake-ripple replay-priority tagging now follows the same bounded-window
discipline. Production calls to `ripple_tag_awake(...)` require awake bucket
ids, collect recent bucket candidates up to `max_candidate_entries`, and emit
`bounded_awake_ripple_tag.v1`; no-scope calls return an empty retired report
with `awake_bucket_scope_required_for_ripple_tagging`. The diagnostic-only
global scan has no runtime hook; benchmark-local retired baselines carry the
comparison. The direct benchmark
`reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json`
averaged `1.091997 ms` on the wake-bucket path versus `1.433332 ms` on the
diagnostic global vector path, with scoped `0` scalar/vector scans, `256`
awake-bucket scans, and `last_ripple_awake_candidate_count=10`. The synthetic
report
`reports/bounded_replay_window_20260617/synthetic-awake-ripple-bounded-scope.json`
kept recall/prototype gates passing and recorded no global candidate scan. The
longer 65536-column hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json`
processed `524288` tokens at `6152.328 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `2013 MiB`, and had zero graph/native/sequence
failures.

The replay-score helper retirement removes the old public full-buffer priority
scorer. The formula is still available through `replay_scores_for_indices(...)`
when a caller has already selected candidate indices. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json`
kept recall/prototype gates passing with `2` bounded updates, `4` bounded
cycles, and `0` global fallback cycles. The matching hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json`
processed `262144` tokens at `6211.859 tokens/sec`, kept bounded `12/65536`
route rows, cached `65526` transition rows, reported no observed contention,
kept GPU memory flat at `1852 MiB`, and had zero graph/native/sequence failures.

The score tensor helper follow-up removes the remaining public archive-wide
score tensors (`maintenance_scores(...)`, `consolidation_scores(...)`,
`repair_scores(...)`, `fragility_scores(...)`, and unused capture/tag/PRP tensor
builders). Production replay now scores only selected candidate indices through
`_score_replay_index(...)`. The later runtime hook cleanup removes the remaining
private global scoring branch from `select_replay_window(...)`, so there is no
reusable production helper or runtime flag for full-buffer replay tensors. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json`
kept recall/prototype gates passing, and the accepted 65536-column hot-path
rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json`
processed `262144` tokens at `6151.952 tokens/sec`, with bounded `12/65536`
route rows, flat `1805 MiB` GPU memory, no observed contention, and zero
graph/native/sequence failures.

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

The HF replay-query source path is now bounded before query collection. Anchor
capture records recency metadata and refreshes anchor dict recency; checkpoints
preserve that ordering evidence. `_collect_anchor_replay_queries(...)` emits
`bounded_replay_query_anchor_bucket_source_window.v1`, takes at most `16`
reverse-recency anchor buckets, and passes that same bucket window into the
store collector and HF recall evaluator. The 8192-anchor benchmark
`reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json`
reduced mean source latency from `16.414 ms` to `0.346 ms`, selected newest
anchor queries with hit rate `1.0`, kept exact input recall, and used CPU-only
replay-query placement with no CUDA allocation. The paired 524288-token
hot-path run stayed in band at `6376.873 tokens/sec` with bounded route rows,
flat `1787 MiB` GPU memory, and zero runtime failures, while noting borderline
sampled GPU contention.

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
