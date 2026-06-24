---
type: paper
status: draft
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/retrieval/routing_index.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/training/query_runner.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/manager.py
  - ../../../src/marulho/service/persistence.py
  - ../../../src/marulho/service/applied_replay_lineage.py
  - ../../../src/marulho/service/brain_runtime.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_replay_artifact_provenance_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py
  - ../../../src/marulho/evaluation/emission_replay_context_review_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_replay_evaluation_context_window_benchmark.py
  - ../../../src/marulho/evaluation/status_replay_path_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_snapshot_source_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_replay_target_window_benchmark.py
  - ../../../src/marulho/evaluation/language_plasticity_replay_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_ledger_rollout_candidate_window_benchmark.py
  - ../../../src/marulho/evaluation/strong_capture_admission_cadence_benchmark.py
  - ../../../src/marulho/evaluation/slow_memory_fixed_cadence_retirement_benchmark.py
  - ../../../src/marulho/evaluation/source_tick_sleep_deferral_benchmark.py
  - ../../../src/marulho/evaluation/live_memory_summary_projection_benchmark.py
  - ../../../src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py
  - ../../../src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py
  - ../../../src/marulho/evaluation/sleep_plasticity_ticket_queue_source_window_benchmark.py
  - ../../../src/marulho/evaluation/status_transition_memory_source_window_benchmark.py
  - ../../../src/marulho/evaluation/plasticity_runtime_state_source_window_benchmark.py
  - ../../../src/marulho/evaluation/applied_replay_lineage_checkpoint_summary_benchmark.py
  - ../../../src/marulho/evaluation/query_recent_fallback_retirement_benchmark.py
  - ../../../src/marulho/service/status_read_model.py
  - ../../../src/marulho/service/transition_memory_source_window.py
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
  - reports/bounded_replay_window_20260618/snn-emission-review-replay-policy-source-window.json
  - reports/bounded_replay_window_20260619/emission-replay-context-review-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-profile-rerun.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json
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
  - reports/bounded_replay_window_20260622/query-recent-fallback-retired-bucket-only.json
  - reports/bounded_replay_window_20260622/replay-dataset-bundle-source-window-query-fallback-retired.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-fallback-retired-bundle-source-window.json
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
  - reports/bounded_replay_window_20260618/replay-plan-source-window-bounded.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-plan-source-window.json
  - reports/bounded_replay_window_20260618/snn-replay-artifact-provenance-source-window.json
  - reports/bounded_replay_window_20260618/status-replay-path-source-window.json
  - reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json
  - reports/bounded_replay_window_20260620/status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-transition-memory-source-window-rerun.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-profile.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-priority-source-window-binding-rerun.json
  - reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json
  - reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json
  - reports/bounded_replay_window_20260620/strong-capture-admission-cadence-after-fixed-cadence-retirement.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json
  - reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery-sharded.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json
  - reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-plasticity-ticket-queue-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-update-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-record-family-append.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-autonomous-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-training-probe-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-output-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-output-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-text-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-text-surface-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-surface-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-generation-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-thought-structural-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-thought-structural-chain-rerun.json
  - reports/bounded_replay_window_20260619/readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/language-plasticity-replay-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json
  - reports/bounded_replay_window_20260619/readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260620/replay-restore-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json
  - reports/bounded_replay_window_20260620/applied-replay-lineage-checkpoint-summary.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-applied-replay-lineage-checkpoint-summary.json
  - reports/bounded_replay_window_20260621/plasticity-runtime-state-source-window.json
  - reports/bounded_replay_window_20260621/hotpath-active-pressure-65536-524288-i32-plasticity-runtime-state-source-window.json
  - reports/bounded_replay_window_20260621/hotpath-active-pressure-65536-524288-i32-plasticity-runtime-state-source-window-rerun.json
  - reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-consolidation-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-consolidation-canonical.json
  - reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-structural-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json
---

# Replay/consolidation

## Current MARULHO Boundary

The service advisory replay-plan/sample/dataset lane is retired and removed.
Replay research remains relevant only as bounded local recall or selected
slow-window consolidation inside trainer/SNN-owned paths. The maintained code
keeps `ReplayController` for SNN replay artifacts, regeneration permits,
sleep-plasticity review tickets, scheduler tickets, and transition-memory
replay artifacts. Runtime trace export is trace-only. Deleted service routes,
schemas, persisted replay-sample history, replay dataset packagers/runners, and
living-loop replay planning are not compatibility surfaces.

This retirement is a hot-path absence/protection claim, not a new replay
promotion claim. The first local `524288`-token reports under
`..\..\MARULHO_reports\bounded_replay_window_20260623\` are retained as
rejected/noisy evidence because contention was observed and throughput stayed at
`3982.916` and `4559.668 tokens/sec`, even though the reports preserved bounded
route shape and did not poll status snapshots during measurement. The later
same-session A/B that measured clean HEAD `c7a07c56` at `5067.347 tokens/sec`
and current worktree at `4859.400 tokens/sec` is also rejected as an unpinned
control. The diagnostic found the Python environment imported the editable main
checkout from the throwaway worktree unless `PYTHONPATH` was pinned; future
worktree comparisons must verify `marulho.__file__`.

The corrected pinned-import evidence returns the retirement slice to the
maintained live-tick band. The full `524288`-token current HEAD run reached
`5865.705 tokens/sec`, `tick_duration_ms.p95=22.378`, and
`train_compute=0.136846 ms/token`; pinned old known-good `fbb788de` reached
`5880.863 tokens/sec`, `tick_duration_ms.p95=23.166`, and
`train_compute=0.137990 ms/token`. Both arms recorded no contention, bounded
`12/65536` route rows, no all-column transition, no measurement-window status
polling, RTX 3060 CUDA placement, and zero graph/native sequence failures.

This follows the paper trail: CLS and hippocampal replay argue for separated
fast/slow learning windows; continual replay argues for selected rehearsal, not
always-on background work; synaptic tagging/capture argues for local tags and
capture windows; sparse replay argues for bounded candidate sets; modern
Hopfield-style attention-like recall is treated here as a local associative
memory operator, not as a transformer-like mind.

The 2026-06-23 maintained-only cleanup applies that paper boundary to the
benchmark layer itself. The repo no longer carries executable full-retained
emission-review or status replay comparators as side baselines; the active
benchmarks assert bounded source-window quality against seeded expectations and
report retired-path absence. The evidence lives in external local reports under
`..\..\MARULHO_reports\bounded_replay_window_20260623\` and the long
`524288`-token gate stayed protected at `6518.530 tokens/sec` with no observed
contention, no in-window environment sampling, bounded `12/65536` route rows,
and zero graph/native sequence failures.

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
returns empty with `candidate_scope=bucket_index_scope_required` and
`fallback_reason=candidate_bucket_scope_required_for_replay_window`. The full
slow-memory scorer is no longer callable through the runtime store; retired
full-scan comparisons live in benchmark-local harnesses only.

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

Inherited HF replay-query reports are bounded again before recall. When
`_bounded_replay_recall_evaluation(...)` receives a caller-supplied query
collection report, it accepts `candidate_bucket_ids` only if the report carries
the canonical `bounded_replay_query_anchor_bucket_source_window.v1` source
window. It then de-duplicates and caps the inherited buckets to
`REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT=16` before calling
`DualMemoryStore.recall_replay_window(...)`; noncanonical reports rebuild the
bounded source window from current anchors. This keeps checkpoint/restored
report metadata from widening modern-Hopfield-style local recall into an
archive-scale scope.

Explicit query/readout recall follows the same literature boundary. Modern
Hopfield-style matching is useful only as a bounded local associative operator,
so `query_runner.memory_matches_with_report(...)` records
`bounded_query_memory_match.v1`: routing supplies candidate bucket ids,
`DualMemoryStore.collect_query_memory_match_indices(...)` returns a capped
bucket-indexed memory window, and the query runner computes similarity plus
replay-priority scores only for those entries. This keeps readout recall from
becoming an archive-wide hidden reasoning pass.

Explicit query text payloads follow the same selected-window rule. Similarity
ranking can stay tensor-only until the returned set is known; raw replay text
is then materialized only for returned evidence. Term/focus ranking may still
inspect text inside the bounded candidate window, but the report must state
that policy explicitly and keep `language_reasoning=false`.

Sleep-plasticity review and scheduler-design review queues now follow the same
slow-window contract. `ReplayController` exposes
`bounded_snn_sleep_plasticity_review_ticket_queue_source_window.v1` and
`bounded_snn_sleep_plasticity_scheduler_design_review_ticket_queue_source_window.v1`
before autonomy, scheduler-design, or installation proposals. Each queue
inspects at most `16` newest retained tickets, reports `retained_count` without
using it as a scan budget, and keeps source selection and scoring on CPU. A
malformed newest record blocks the bounded window instead of widening the scan
to older retained records. The benchmark
`reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json`
matched diagnostic latest-verified quality while reading `16/64` retained
records on each queue (`4x` less source work), with no global candidate/score
scan, no raw replay text, no hidden language reasoning, no live tick, no
every-token cadence, no scheduler install, no mutation/plasticity, CUDA unused,
and `0.072 MiB` traced Python peak allocation. The maintained-only cleanup
`..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-plasticity-ticket-queue-legacy-baseline-removed.json`
removes the executable full-retained benchmark verifier too: deterministic
seeded newest-ticket quality still passes on `16/64` CPU source windows, the
sleep queue averages `2.836412 ms`, the scheduler-design queue averages
`273.395856 ms`, the report carries
`retired_full_retained_ticket_queue_absence.implementation_present=false`, and
CUDA archival allocation remains `0.0 MiB`. The paired
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-sleep-ticket-legacy-baseline-removed-default-nosample.json`
run kept the live tick protected at `6040.427 tokens/sec`, p95 `21.752 ms`,
`train_compute=0.133547 ms/token`, bounded `12/65536` route rows, no observed
contention, RTX 3060 memory `1791->1795 MiB`, and zero graph/native sequence
failures.

ConceptStore signature lookup is now treated as an evidence-window lookup, not
as general archive traversal. CLS and continual-replay work argue for separating
fast online traces from slower selected consolidation, and modern
Hopfield-style recall remains useful here only after a local candidate set
exists. `ConceptStore.observe(...)` therefore resolves memory signatures only
from already-selected query/source/concept evidence, caps each source at `8`
unique indices with a `32`-reference scan budget, direct-indexes CPU archival
arrays, and reports `bounded_concept_memory_signature_lookup.v1`. The
benchmark `reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json`
kept diagnostic legacy signature quality (`min cosine=0.9999998212`) while
removing `4096` archive list materializations and reducing mean lookup latency
from `12.490 ms` to `1.454 ms` over `65536` entries. This keeps semantic
observation tied to selected evidence rather than turning it into hidden global
memory recall.

Semantic frontier-gap planning follows the same rule. A modern Hopfield-style
match can rank a local candidate window, but complementary learning systems,
continual replay, synaptic tagging/capture, and sparse replay all argue against
letting planning read the whole archive as a background language loop.
`frontier_gap_plan(...)` therefore asks
`DualMemoryStore.collect_frontier_gap_indices(...)` for a capped CPU candidate
window and loads raw text only for those selected candidates. The report
surface `bounded_frontier_gap_selection.v1` records the candidate budget,
global scan flags, archival CPU placement, `runs_live_tick=false`, and
`language_reasoning=false`.

Source-bank semantic recall now has one explicit merged candidate-window
boundary. `bank_memory_matches_with_report(...)` samples a capped set of
source-bank probes, unions their routing-index bucket ids, collects one CPU
bucket-indexed candidate window capped at `192`, vector-scores probes against
that local associative window, and emits `bounded_source_bank_memory_match.v1`
with `merged_probe_candidate_window=true`,
`per_probe_query_match_call_count=0`, candidate/window budgets, raw text loaded
only for returned matches, CPU archival and score placement, no global scans,
`runs_live_tick=false`, `runs_every_token=false`, and
`language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260622/source-bank-merged-probe-window.json`
kept selected indices identical to the retired per-probe diagnostic path
(`quality.min=1.0`), scored `192` candidates and `1536` local similarities,
loaded `4` returned raw text payloads instead of `32`, and reduced mean
latency from `560.177 ms` to `106.543 ms` (`5.258x`) over a `65536`-entry
archive. CUDA was available but archival recall used `0.0 MiB` CUDA
allocation/reservation. The refreshed 524288-token hot-path run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-source-bank-merged-probe-window.json`
stayed in band at `6129.933 tokens/sec`, with `train_compute=0.132126 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, no observed
CPU contention, mild GPU contention (`21%` against a `20%` threshold), and flat
GPU memory at `1763 MiB`.

The 2026-06-23 maintained-only cleanup removes the benchmark-local executable
per-probe diagnostic baseline and keeps the quality gate on seeded expected
selection plus source-window budgets. The external report
`..\..\MARULHO_reports\bounded_replay_window_20260623\source-bank-legacy-baseline-removed.json`
passed with `192` bounded candidates, `1536` local similarities, `4` returned
text payloads, `134.160628 ms` bounded mean latency, CPU archival/score
placement, no CUDA archival allocation, and the retired active report field
absent.
The clean 524288-token protection gate
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-legacy-benchmark-baselines-removed-default-nosample.json`
passed at `6098.818 tokens/sec`, with `train_compute=0.132657 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, CUDA active on
RTX 3060, GPU memory `2069->2068 MiB`, no observed contention, and zero
graph/native/sequence failures.

Awake-ripple replay tagging is treated as selected synaptic tagging/capture
metadata, not as a global recent-memory operator. `ripple_tag_awake(...)` now
requires awake bucket scope for production tagging, caps candidates through the
CPU bucket/recency index, and records `bounded_awake_ripple_tag.v1` with
candidate budget, scan flags, device placement, and `runs_every_token=false`.
If awake bucket scope is absent, it returns an empty retired report instead of
scanning all memory. The retained scalar/vector recent-memory scan has no
runtime hook; benchmark-local retired baselines carry any full-scan comparison.

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

Post-replay routing maintenance now stays inside the same selected-window
budget. Deep/repair replay returns the prototype IDs it actually updated;
`MarulhoTrainer._refresh_sleep_replay_routing_index(...)` then calls
`routing_index_existing_row_refresh.v1` to update only those existing rows in
the tensor routing cache through a CPU ID-to-row map. Selection criteria are:
nonempty replay-updated prototype IDs, existing routing IDs, and a ready cache.
Missing IDs, missing row-update APIs, or dirty caches become deferred recovery
evidence and do not call `add()+rebuild()` inside selected replay. The refreshed
benchmark
`..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-replay-routing-index-legacy-baseline-removed-rerun.json`
now carries only the maintained path and records
`retired_full_rebuild_absence.implementation_present=false`. It updated
`16/65536` rows, deferred `1` missing row without inserting it, preserved exact
top-1 recall for updated rows, used CPU row-lookup metadata, and measured
`9.892320 ms` mean bounded latency across `25` runs (`median=5.693200 ms`). The
sharded variant
`..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-replay-routing-index-legacy-baseline-removed-sharded.json`
passed at `19.582580 ms` mean latency with `16` direct shard and merged updates.
Python trace-memory peak stayed below `0.085 MiB`; CUDA allocation/reservation
ended at `24.625/34.0 MiB` for single-index and `41.125/50.0 MiB` for sharded.
The current `524288`-token hot-path run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-routing-index-legacy-baseline-removed-default-nosample.json`
stayed in band at `6097.811 tokens/sec`, `tick_duration_ms.p95=21.193`,
`train_compute=0.132696 ms/token`, bounded `12/65536` route rows, cached
`65526` transition rows, zero graph/native sequence failures, no observed
contention, CPU max `32%`, GPU max `15%`, and RTX 3060 memory
`1825->1824 MiB`.

Bucket-level consolidation pressure now uses the same bounded accounting.
`DualMemoryStore.bucket_consolidation_level(...)` no longer recomputes a single
winner bucket by scanning every slow-memory entry. Live scalar reads use the
maintained CPU bucket cache and report
`bucket_consolidation_level_cache_lookup.v1` with `full_memory_scan=false` and
`scan_entry_count=0`; if the cache is absent, the scalar API returns a no-scan
miss instead of rebuilding. Explicit `bucket_consolidation_tensor(...)`
rebuilds remain load/capture/offline, explicit tensor request, checkpoint
load, graph capture/prewarm, or offline work. The maintained-only benchmark
`..\..\MARULHO_reports\bounded_replay_window_20260623\bucket-consolidation-cache-legacy-baseline-removed.json`
removes the executable retired scalar-scan comparator and records
`retired_full_bucket_scan_absence.implementation_present=false`. It matched the
seeded bucket expectation exactly for a `65536`-entry store, reported
`scan_entry_count=0`, averaged `0.017516 ms`, kept Python trace-memory peak to
`0.002090 MiB`, and used `0.0 MiB` CUDA allocation. The paired `524288`-token
hot-path run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-bucket-cache-legacy-baseline-removed-default-nosample.json`
stayed same-band at `6461.135 tokens/sec`, `tick_duration_ms.p95=19.207`,
`train_compute=0.125097 ms/token`, bounded `12/65536` route rows, cached
`65526` transition rows, zero graph/native sequence failures, no observed
contention, and RTX 3060 memory `1929->1928 MiB`.

Selected replay consolidation follows the same local-window rule. A selected
replay window can update selected memory STC/replay state, but it must not
repair missing global bucket-cache metadata by scanning every retained
slow-memory entry before consolidation. `DualMemoryStore.consolidate_replay`
now emits `bounded_selected_replay_consolidation.v1`: selected replay counts,
capture tags, consolidation levels/events, and EMAs are validated directly
against seeded maintained-path expectations; cache delta updates run only when
cache metadata is already present; and missing cache metadata records
`cache_missing_deferred_no_full_rebuild`. The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\selected-replay-consolidation-cache-diagnostic-removed.json`
removes the executable full-cache rebuild diagnostic, reads `16/65536` selected
CPU entries, scans `0` cache-rebuild entries, projects `65536` removed rebuild
entries (`4096x` source-work avoidance), averages `1.957360 ms`, uses
`0.510799 MiB` traced Python peak, and keeps CUDA allocation/reservation at
`0.0 MiB`. The current `524288`-token protection run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-selected-replay-cache-diagnostic-removed-default-nosample.json`
stayed in band at `6209.501 tokens/sec`, p95 `20.916 ms`,
`train_compute=0.129968 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, native sequence-loop and burst-replay failures `0`, and
RTX 3060 memory flat at `2026 MiB`; a before-sample GPU reading of `21%` makes
this throughput protection under borderline contention rather than a clean speed
ceiling.

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

The SNN replay-priority queue now follows the same bounded-source rule before a
due-cycle review can nominate a context. `snn_replay_consolidation_priority_queue.v1`
no longer verifies every retained Replay Controller context before applying its
output limit. It uses a controller-owned replay-evaluation-context ID index, a
`16`-entry recent context source window, and up to `16` explicit readout-target
context IDs. The resulting `bounded_snn_replay_priority_source_window.v1`
reports source counts, verified counts, CPU archival/score placement, no global
candidate or score scan, no raw text payload, `language_reasoning=false`,
`runs_live_tick=false`, and `gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-replay-priority-source-window.json`
kept an old readout-targeted high-signal context selectable outside the recent
window while verifying `17` contexts instead of all `64` retained contexts,
averaging `1.825268 ms` with `0.050346 MiB` traced peak allocation and no CUDA
allocation. The matching 65536-column `524288`-token hot-path check stayed in
band at `6298.310 tokens/sec`, `train_compute=0.129349 ms/token`, bounded
`12/65536` route rows, `65526` cached transition rows, `1799->1800 MiB` GPU
memory, no observed contention, and zero graph/native/sequence failures.

The upstream SNN readout priority report now obeys the same selected-source
rule before it can feed the Replay Controller. `snn_language_readout_replay_priority.v1`
no longer scores every retained readout event before applying its output limit.
It reads a CPU-resident `32`-event recent source window, scores provenance,
label repetition, transition reuse, and recency only inside that window, and
emits `bounded_snn_readout_replay_priority_source_window.v1` with no global
candidate or score scan, no raw text payload, `language_reasoning=false`,
`runs_live_tick=false`, `runs_every_token=false`, and `gpu_used=false`. The
benchmark
`reports/bounded_replay_window_20260618/snn-readout-replay-priority-source-window.json`
matched the diagnostic full-retained scorer's top high-signal readout while
scoring `32` of `2048` retained events, averaging `1.424948 ms` versus
`51.002932 ms` (`35.792837x`) with `0.065639 MiB` traced peak allocation and no
CUDA allocation. The paired `524288`-token hot-path check stayed in band at
`6284.379 tokens/sec`, `train_compute=0.129905 ms/token`, bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `1852->1858 MiB`, no
observed contention, and zero graph/native/sequence failures. This treats
attention-like readout recall as a bounded local replay-priority operator, not
as a transformer-style text reasoning pass.

The 2026-06-23 maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\snn-readout-replay-priority-legacy-baseline-removed.json`
removes the executable benchmark-local full-retained scorer from this evidence
path. It selected the seeded recent high-signal readout from a `32/2048` CPU
source window, returned `8` candidates, recorded the retired callable absent,
averaged `0.911452 ms`, and used no CUDA archival allocation.

Emission-review replay policy now follows the same selected-source rule before
reviewed emissions can become replay-context seeds. The previous
`snn_language_readout_emission_replay_evaluation_policy.v1` capped returned
candidates but still matched reviewed emissions against all retained readout
events, and the design step reopened all retained readouts for verification.
Both paths now use
`bounded_snn_emission_review_replay_policy_source_window.v1`: `16` recent
reviewed emissions, `16` recent internal readout events, hash-only candidates,
CPU archival/score placement, no global candidate/score scan, no raw reviewed
text payload, `language_reasoning=false`, `runs_live_tick=false`,
`runs_every_token=false`, and `gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-emission-review-replay-policy-source-window.json`
matched the diagnostic full-retained policy/design top candidate while checking
`32` source events instead of `4096` retained review/readout records, averaging
`2.476164 ms` versus `166.924984 ms` (`67.412734x`) with `0.046277 MiB` traced
peak allocation and no CUDA allocation. A clean profiled `524288`-token hot-path
rerun, made after rejecting an externally contended run, stayed in band at
`6376.714 tokens/sec`, `train_compute=0.128297 ms/token`, bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `2122->2123 MiB`, no
observed contention, and zero graph/native/sequence failures.

The emission replay-context bridge now keeps that selected-source boundary
when a reviewed-emission design becomes a Replay Controller context. Hash-only
replay-context seeds and observed sparse slots are local replay evidence, not
a text payload or a caller-sized replay batch. The facade now applies the
shared readout replay source-window budget
`SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32` to
`selected_replay_context_seeds` and `observed_readout_slots`, then requires
both windows to be bounded, untruncated, and well formed before recomputing
mismatch/pressure or recording a context.
`reports/bounded_replay_window_20260619/emission-replay-context-review-window.json`
passed with exact `32/32` seed and observed-slot windows recording one context,
oversized seeds and observed slots blocked at `32/2048`, and blocked payloads
making no mismatch, pressure, or Replay Controller calls. The report records
CPU archival/source/gate placement, no global candidate/score scan, no raw text
payload, no hidden language reasoning, no live tick, no every-token cadence,
`64x` projected source-work reduction, `1.832774 MiB` traced Python peak, and
`0.0 MiB` CUDA allocation/reservation. The first clean hot-path run finished
below band at `5877.891 tokens/sec`; the rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json`
processed `524288` tokens at `5990.908 tokens/sec` with bounded `12/65536`
route rows, `65526` cached rows, no observed contention, GPU memory
`2032->2031 MiB`, and zero graph/native sequence failures. This retires the
full-payload facade bridge rather than preserving a second context-recording
route.

The generic Replay Controller context facade now follows the same selected
observed-slot boundary. `snn_replay_evaluation_context(...)` applies
`SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32` to caller-supplied
`observed_readout_slots`, requires the observed-slot window to be bounded,
untruncated, and well formed before mismatch/pressure or context recording can
run, and stores the source-window report in the recorded context metadata. This
keeps the route as a bounded server-recomputed evidence gate instead of a
second caller-sized context-recording path.
`reports/bounded_replay_window_20260619/snn-replay-evaluation-context-window.json`
passed with an exact `32/32` observed-slot context, oversized slots blocked at
`32/2048`, and blocked payloads making no mismatch, pressure, or Replay
Controller calls. The report records CPU archival/source/gate placement, no
global candidate/score scan, no raw text payload, no hidden language reasoning,
no live tick, no every-token cadence, source-window metadata on the accepted
context, `64x` projected source-work reduction, `0.656714 MiB` traced Python
peak, and `0.0 MiB` CUDA allocation/reservation. The hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json`
processed `524288` tokens at `6009.932 tokens/sec` with bounded `12/65536`
route rows, `65526` cached rows, GPU memory `2031->2045 MiB`, and zero
graph/native sequence failures. Sampled GPU contention reached `22%`, so this
is maintained-throughput evidence, not contention-free evidence. The generic
full-payload observed-slot facade shape is retired.

The rollout rehearsal promotion policy now applies the same source-window rule
to sparse trajectory evidence before rehearsal or consolidation review. The
previous `snn_language_readout_rollout_rehearsal_promotion_policy.v1` capped
returned candidates but verified and scored every retained rollout event. It
now reads a CPU-resident `16`-event recent source window, normalizes at most
`32` replay targets per event, and emits
`bounded_snn_readout_rollout_rehearsal_source_window.v1` with no global
candidate or score scan, no raw text payload, `language_reasoning=false`,
`runs_live_tick=false`, `runs_every_token=false`, and `gpu_used=false`. The
benchmark
`reports/bounded_replay_window_20260618/snn-rollout-rehearsal-source-window.json`
matched the diagnostic full-retained top rollout while scoring `16` of `2048`
retained events, averaging `2.090592 ms` versus `309.922768 ms`
(`148.246414x`) with `0.066692 MiB` traced peak allocation and no CUDA
allocation. The paired `524288`-token hot-path check stayed in band at
`6339.682 tokens/sec`, `train_compute=0.129022 ms/token`, bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `1867->1865 MiB`, and
zero graph/native/sequence failures; GPU contention reached `22%`, so the
evidence supports protected throughput, not a contention-free environment.

The 2026-06-23 maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\snn-rollout-rehearsal-legacy-baseline-removed.json`
removes the executable benchmark-local full-retained rollout scorer from this
evidence path. It selected the seeded recent high-signal rollout from a
`16/2048` CPU source window, returned `8` candidates, recorded the retired
callable absent, averaged `3.282180 ms`, and used no CUDA archival allocation.

Status projection now obeys that same replay boundary. `StatusReadModel` no
longer scans all retained readout, emission-review, and rollout ledgers when it
publishes emission history, replay-design, or consolidation readiness. It emits
`bounded_snn_status_emission_review_history_source_window.v1` from `16` recent
reviewed emissions,
`bounded_snn_status_emission_replay_design_path_source_window.v1` from `16`
recent reviewed emissions plus `16` recent internal readouts, and
`bounded_snn_status_rollout_consolidation_path_source_window.v1` from `16`
recent rollout events plus `16` recent internal readouts. These are
control-plane reports only: CPU archival/score placement, retained/truncated
counts, no global candidate or score scan, no raw replay text, no hidden
language reasoning, no live tick, no every-token cadence, and no CUDA archival
metadata. The benchmark
`reports/bounded_replay_window_20260618/status-replay-path-source-window.json`
kept the same latest history, emission, and rollout evidence as the diagnostic
full-retained projection while checking `80` source rows instead of `10240`
retained rows, reducing combined projection latency from `102.831789 ms` to
`1.309999 ms`. The profiled `524288`-token hot-path run stayed protected and
contention-free at `6081.034 tokens/sec`; the no-profile rerun reached
`6408.252 tokens/sec` with bounded `12/65536` route rows and zero
graph/native/sequence failures, but had observed GPU-side contention. The
research implication is practical: even observability must not become a hidden
complementary-memory scan as histories scale.

The replay-artifact provenance path now follows that same selected-source
boundary after nomination. Artifact review tickets, evaluated transition-memory
replay artifacts, regeneration permits, sleep review tickets, and scheduler
design review tickets use controller-owned ID indexes instead of retained-deque
linear lookups. New evaluated artifacts and permits carry
`bounded_snn_replay_artifact_provenance_source_window.v1`, capped to context,
ticket, artifact, and permit IDs, with CPU archival placement, no global
candidate/score scan, no raw replay text, `language_reasoning=false`,
`runs_live_tick=false`, and `gpu_used=false`. The focused benchmark
`reports/bounded_replay_window_20260618/snn-replay-artifact-provenance-source-window.json`
kept the oldest retained provenance chain verifiable at the retention tail,
used `4` indexed lookups instead of the old worst-case `256` retained-record
checks, averaged `0.348376 ms`, used `0.012636 MiB` traced peak allocation, and
allocated no CUDA memory. The accepted 65536-column `524288`-token hot-path
rerun stayed in band at `6286.248 tokens/sec` with `train_compute=0.129585
ms/token`, bounded `12/65536` route rows, `65526` cached transition rows, flat
GPU memory behavior, no observed contention, and zero graph/native/sequence
failures after one same-code run at `5849.047 tokens/sec` was rejected as below
the maintained band. This is not live replay; it keeps structural-write consent
tied to bounded replay evidence without a retained-history scan.

The 2026-06-24 maintained-only cleanup applies the same rule to benchmark
surfaces. `snn_replay_priority_source_window_benchmark.py` and
`snn_replay_artifact_provenance_source_window_benchmark.py` no longer emit
`retired_path_comparison`, and
`status_transition_memory_source_window_benchmark.py` no longer executes the
broad transition-memory projection comparator. The current reports under
`..\..\MARULHO_reports\bounded_replay_window_20260624\` pass with bounded
replay-priority selection (`17/64` verified contexts, `1.581416 ms` mean),
indexed artifact provenance (`4/4` ID lookups, `0.398844 ms` mean), and bounded
status transition-memory projection (`256` CPU source rows, `11.302696 ms`
mean). CUDA archive allocation remains `0.0 MiB`, and all reports keep
no-live-tick, no-every-token, no-global-scan, and no-hidden-language-reasoning
flags. The paired `524288`-token hot-path gate stays in band at
`6259.398 tokens/sec` with bounded `12/65536` route rows and zero graph/native
sequence failures. Benchmark code therefore carries the maintained source-window
operators only; historical broad scans remain documentation, not executable
side implementations.

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
claim concrete. Sleep replay now uses `DualMemoryStore.sleep_repair_replay_row(...)`
for mutating tensor repair and `DualMemoryStore.replay_recall_row(...)` for
read-only recall; neither path exposes `raw_window`, expanded text, or metadata.
Selection and recall reports expose `raw_text_payload_loaded=false`
and `language_reasoning=false`, while the sleep replay report exposes
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`, and
`sleep_replay_text_payload_policy=sleep_replay_uses_tensor_payloads_only`.
Deep sleep with an abstraction layer also samples SFA correction from the
processed replay indices instead of the whole slow buffer, with
`sleep_replay_sfa_correction_scope=selected_replay_window` and
`sleep_replay_sfa_full_memory_sample_retired=true`. The initial helper defaults
blocked unscoped replay/SFA calls; the current code removes the list-only
`sample_replay_indices(...)` and `sample_for_sfa(...)` helpers and uses
reported selection/sampling APIs instead. The
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

The query-memory payload follow-up tightens that readout boundary. The
similarity-only path no longer builds text payloads for every candidate before
sorting; it scores the bounded candidate window first, then loads replay text
only for returned matches. The benchmark
`reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json`
preserved selected indices against the retired eager candidate-payload shape,
reduced raw text payload loads from `192` to `5`, and reduced mean latency from
`33.612 ms` to `25.881 ms` (`1.299x`). The 524288-token hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json`
processed `6152.079 tokens/sec`, with bounded `12/65536` route rows, `65526`
cached transition rows, flat GPU memory (`1874->1878 MiB`), no observed
contention, and zero graph/native/sequence failures.

The concept-frontier follow-up applies the same selected-memory rule to source
acquisition planning. `concept_frontier_metrics_with_report(...)` derives
candidate buckets from the probe-bank routing signature and uses
`DualMemoryStore.collect_query_memory_match_indices(...)` before scoring
novelty, uncertainty, and support. This retires direct iteration over every
`slow_routing_keys` entry without changing the paper boundary: modern
Hopfield-style matching can be a local associative metric, not an archive-wide
mind. The benchmark
`reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json`
scored `64/8192` entries, preserved the diagnostic full-scan top-1, kept
`novelty_delta=0.0`, `uncertainty_delta=0.0`, and
`support_delta=0.015893`, and reduced mean latency from `658.116 ms` to
`5.040 ms`. The matching 65536-column hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json`
processed `262144` tokens at `6148.846 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1805 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

The source-bank probe-signature follow-up bounds the recall operator one step
earlier. A source bank may contain many probe patterns, but the maintained
frontier path now treats the bank signature as a local associative window:
`concept_frontier_metrics_with_report(...)` and
`candidate_semantic_signature(...)` sample `16` evenly spaced probes, report the
source-probe budget and selected probe indices, and only then ask the routing
index for candidate buckets. This keeps modern Hopfield-style matching local
and keeps CLS/continual-replay/STC constraints intact: no full archive scan, no
every-token admission, no GPU-resident archival metadata, and no hidden replay
text reasoning. `reports/bounded_replay_window_20260618/concept-frontier-source-probe-window-bounded.json`
used `64` source probes and a `16384`-entry archive, sampled `16` probes,
scored `64` memory entries instead of `16384`, preserved top-1, and reduced
mean latency from `1556.602 ms` to `7.637 ms` (`203.829x`). The paired
524288-token protection check stayed at the baseline band: committed baseline
`6307.437 tokens/sec`, current source-probe tree `6303.548 tokens/sec`, bounded
`12/65536` route rows, `65526` cached transition rows, flat `1789 MiB` current
GPU memory, no observed contention, and zero graph/native/sequence failures.

The semantic frontier-gap planner follow-up applies that local-memory boundary
to gap-term planning. The old planner materialized the whole `slow_raw_windows`
archive and rebuilt side lists while ranking terms; the new path collects a
capped CPU recency or bucket candidate window, scores only selected entries,
and emits `bounded_frontier_gap_selection.v1` with no global candidate/score
scan and no hidden language reasoning. The benchmark
`reports/bounded_replay_window_20260617/frontier-gap-bounded.json` scored
`192/65536` entries, preserved expected and diagnostic legacy terms with
`quality.min=1.0`, reduced mean latency from `217.530 ms` to `9.073 ms`
(`23.975x`), and passed a missing-collector gate with zero candidates, zero
text payloads, and no global scans. The report-dropping
`frontier_gap_terms(...)` helper is now deleted, so callers must use
`frontier_gap_plan(...)` when they need terms plus bounded evidence. The longer
65536-column hot-path report
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json`
processed `524288` tokens at `6233.085 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, GPU memory `1844->1840 MiB`, no
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

The runtime concept memory lookup follow-up applies the same boundary to
cadenced source/feed concept observation. `OperatorInteractionRuntime` no
longer reaches into `slow_routing_keys`, `slow_texts`, `slow_raw_windows`, or
STC arrays directly. It asks `DualMemoryStore.resolve_runtime_concept_memory_matches(...)`
to resolve only trainer-provided `memory_index` evidence, cap the observation
batch, cache duplicate text payloads, and emit
`bounded_runtime_concept_memory_lookup.v1`. This lookup can occur in the live
runtime observation cadence, so it reports `runs_live_tick=true`; the important
guard is that it also reports `runs_every_token=false`, no global score or
candidate scan, CPU archival/score placement, and `language_reasoning=false`.
The benchmark
`reports/bounded_replay_window_20260618/runtime-concept-memory-lookup-bounded.json`
preserved selected-index parity over `512` observations on a `65536`-entry
archive, reduced payload reads from `512` to `64` with `448` cache hits, and
cut mean lookup latency from `47.156 ms` to `6.380 ms` (`7.391x`). The paired
`524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-runtime-concept-memory-lookup.json`
stayed in band at `6237.075 tokens/sec`, with
`concept_observation=0.000474 ms/token`, bounded `12/65536` route rows,
`65526` cached transition rows, GPU memory `1809->1861 MiB`, no observed
contention, and zero graph/native/sequence failures.

The context-comparison memory follow-up closes the remaining report-dropping
query readout path. Context A/B comparison now calls
`memory_matches_with_report(...)` for each context, shares one returned
replay-entry payload cache across contexts, returns per-context reports, and
emits `bounded_context_comparison_memory_match.v1`. The old
`query_runner.memory_matches(...)` compatibility wrapper is removed, so new
callers cannot accidentally drop bounded recall evidence. The report keeps
context comparison in explicit slow readout (`runs_live_tick=false`,
`runs_every_token=false`), records CPU archival/score placement, and reports no
global score/candidate scan or hidden language reasoning. The benchmark
`reports/bounded_replay_window_20260618/context-memory-match-bounded.json`
preserved selected indices for both contexts, reduced payload reads from `16`
to `8` with `8` cache hits, and reduced mean latency from `71.927 ms` to
`70.550 ms`. The paired `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-context-memory-match.json`
stayed in the maintained band at `6065.987 tokens/sec`, with bounded
`12/65536` route rows, `65526` cached transition rows, GPU memory
`1839->1845 MiB`, no observed contention, and zero graph/native/sequence
failures.

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

The score tensor helper cleanup removes the remaining public archive-wide score
tensor family. `maintenance_scores(...)`, `consolidation_scores(...)`,
`repair_scores(...)`, `fragility_scores(...)`, and unused capture/tag/PRP tensor
builders are gone, so selected replay/query windows no longer sit beside
production-looking full-buffer helper APIs. The later runtime hook cleanup also
removes the private global-score escape hatch from `select_replay_window(...)`;
retired full-scan comparisons are benchmark-local baselines only. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json`
kept recall/prototype gates passing with `2` bounded updates and `0` global
fallback cycles. The accepted 65536-column hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json`
processed `262144` tokens at `6151.952 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, flat `1805 MiB` GPU memory, no
observed contention, and zero graph/native/sequence failures.

The runtime global-scan hook cleanup removes the final callable full-archive
branches from the replay/consolidation store. Awake ripple tagging now blocks
without awake-bucket scope, replay-window selection blocks without bucket scope,
and SFA sampling blocks without selected replay indices. All three reports keep
CPU archival placement, no raw text payload, no hidden language reasoning, and
no mutation when scope is missing. Benchmark-local retired baselines still
measure the old cost: `awake-ripple-runtime-global-hooks-retired.json` measured
`1.285064 ms` for the retired full scan versus `1.082768 ms` for the scoped
10-bucket path, and `sfa-runtime-global-hooks-retired.json` improved
selected-window sample purity from `0.00439453125` to `1.0` while reducing mean
latency from `1.740475 ms` to `0.622956 ms` (`2.793896x`). The paired
`524288`-token protection run processed `6342.218 tokens/sec` with
`train_compute=0.128534 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, GPU memory `1801->1802 MiB`, and zero graph/native
sequence failures; GPU-side contention was observed at `23%` max utilization,
so this evidence supports throughput protection, not a contention-free claim.

The reported SFA sampling cleanup closes the remaining list-only replay/SFA
helper family. `sample_replay_indices(...)` is removed; callers must use
`select_replay_window(...)` and keep `bounded_replay_window_selection.v1`.
`sample_for_sfa(...)` is removed; deep sleep now calls
`sample_for_sfa_with_report(...)` and embeds the returned
`bounded_sfa_sample.v1` under the sleep replay report. The sampler records
selected candidate indices, sample indices, sample count, CPU archival/sample
placement, no global candidate scan, `runs_live_tick=false`,
`runs_every_token=false`, and `language_reasoning=false`; the report is also
kept in memory-store summaries and checkpoints. The benchmark
`reports/bounded_replay_window_20260618/sfa-sample-bounded-window.json` used a
`65536`-entry archive, `192` selected replay-window candidates, and `64` SFA
samples. Selected-window sample purity improved from `0.00439453125` for the
retired full-buffer sampler to `1.0`, and mean latency improved from
`1.451 ms` to `0.656 ms` (`2.210x`). The source-bank list wrapper
`bank_memory_matches(...)` is also removed so source-bank recall cannot drop
`bounded_source_bank_memory_match.v1`. The accepted `524288`-token protection
run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-reported-sfa-sampler-noprofile-rerun2.json`
processed `6127.490 tokens/sec`, with last-tick `train_compute=17.738 ms`
(`0.138579 ms/token` over the 128-token tick), bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`1840->1861 MiB`, and zero graph/native/sequence failures.

Repair replay now closes the adjacent dense-input preparation gap. After an
anchored replay window has selected stored entries, normal repair entries carry
stored routing keys, so the old unconditional `assembly_from_input(...)` call
was scoring all columns only to refresh competitive input state. The repair path
now calls `prepare_input_for_candidate_routing(...)` when a stored routing key
exists, clears stale dense caches when no input trace exists, and reports
`sleep_replay_unconditional_dense_input_assembly_retired=true` plus dense
fallback counts for legacy entries. The historical benchmark
`reports/bounded_replay_window_20260618/sleep-repair-replay-bounded-input-prepare.json`
selected and repaired `32/32` anchored replay entries, improved mean anchor
distance from `0.508855` to `0.360171`, reduced selected input-prep latency from
`61.351 ms` to `32.613 ms` (`1.881x`), made `0` dense assembly calls during
repair, and kept archival tensors on CPU while active repair computation used
CUDA. The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\sleep-repair-replay-dense-prepare-comparator-removed.json`
removes the executable dense-prepare comparator, selects `32` entries with `16`
stored routing keys and `16` missing keys, applies `8` repair updates, defers
`8` missing-key selected rows in the repair window, improves stored-key quality
by `0.076463`, measures bounded prepare mean `44.895575 ms` under a `100 ms`
budget, makes `0` dense assembly calls, keeps archival tensors on CPU, and runs
active repair computation on CUDA. The current hot-path run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-sleep-repair-dense-prepare-comparator-removed-default-nosample.json`
processed `524288` tokens at `6410.861 tokens/sec`, p95 `20.195 ms`,
`train_compute=0.126774 ms/token`, bounded route scoring at `12/65536`, cached
`65526` transition rows, reported no observed contention, and kept RTX memory
flat at `2190 MiB`.

The missing-key cleanup removes the remaining legacy branch for selected repair
entries that lack stored routing keys. Instead of rebuilding a routing key from
the input pattern or projecting the stored assembly trace, repair replay and
deep candidate repair now require stored routing keys and defer missing-key
entries with `sleep_replay_missing_routing_key_deferred_count`. The mixed-key
benchmark
`reports/bounded_replay_window_20260620/sleep-repair-replay-missing-routing-key-deferred.json`
dropped routing keys for `16/32` anchored repair entries, updated the `16`
stored-key entries, deferred the `16` missing-key entries, made `0` dense
input-assembly calls, removed the assembly-projection fallback field, and
improved stored-key repair quality by `0.149600`. The current maintained-only
repair report above keeps missing-key deferral while removing the dense-prepare
comparator from benchmark code and records
`retired_dense_prepare_comparator_absence.implementation_present=false`. The
524288-token protection run stayed in band at
`5988.223 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, GPU memory `1877->1878 MiB`, and zero graph/native/sequence
failures; a GPU-utilization sample crossed the contention threshold, so this is
not a speed-ceiling claim.

The query-memory episode readout cleanup makes returned text evidence explicit
without turning it into hidden replay reasoning. The list-only
`build_memory_episodes(...)` helper is removed; query result construction now
uses `build_memory_episodes_with_report(...)` and exposes
`bounded_query_memory_episode_readout.v1` beside `memory_episodes`. The report
records the returned match budget, selected neighbor radius, direct indexed
neighbor-window reads, CPU archival/readout placement, no global scans,
`runs_live_tick=false`, `runs_every_token=false`, and
`language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260618/query-episode-readout-bounded.json`
used a `65536`-entry archive and four selected fragment matches: fragment-only
readout missed the target top episode (`els safe.`), while bounded
selected-neighbor readout recovered `a cat purrs when it feels safe.` with
`10` direct neighbor payloads under a `28`-entry budget. Mean latency rose from
`0.490 ms` to `0.936 ms`, which is accepted as explicit query readout cost. The
paired `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-query-episode-readout.json`
processed `6219.926 tokens/sec`, with `train_compute=0.130647 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, no observed
contention, GPU memory `1810->1811 MiB`, and zero graph/native/sequence
failures.

The source-episode admission follow-up closes the gap where explicit feed
training preserved only cadence fragments in slow memory. `feed_text(...)` now
records `bounded_feed_source_episode_admission.v1` after explicit query-runner
feed, deduplicating source sentences under a `32`-episode, `240`-char payload
budget. This is a slow-path admission operator, not live runtime replay:
reports state no live tick, no every-token work, no global memory scan, CPU
archival storage, and no language reasoning. The readout side also retires the
mixed-provenance stitching assumption: complete admitted source episodes remain
whole, zero-support episodes are filtered when query-supported evidence exists,
and raw neighbor stitching requires character overlap so cadence fragments
cannot be concatenated across source-admission boundaries. The benchmark
`reports/bounded_replay_window_20260618/source-episode-admission-bounded.json`
improved the simple-animals grounding gate from `0.25` to `1.0`, admitted `5`
source episodes, kept slow buffers/input patterns/routing keys on CPU, and
accepted the explicit-feed cost increase (`+17293.226 ms`) because the hot path
does not call the admission operator. The long protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-episode-admission.json`
processed `524288` tokens at `6702.362 tokens/sec`, with
`train_compute=0.121727 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, GPU memory `1808->1808 MiB`,
and zero graph/native/sequence failures.

The next source-admission iteration retires the dense assembly step that still
sat after candidate selection. Admission now obtains the winner, assembly, and
routing key from one bounded offline competition
(`assembly_policy=bounded_offline_competition_winner_assembly`) rather than
calling the dense `assembly_for_pattern(...)` helper from this path. The v2
benchmark
`reports/bounded_replay_window_20260618/source-episode-admission-bounded-v2.json`
kept the simple-animals grounding gate at `0.25 -> 1.0`, admitted `5/5`
selected source episodes, reported admission latency `2725.253 ms`, kept CPU
archival storage, and used `cuda:0` only for active assembly computation. The
explicit-feed delta fell to `46.234 ms`, mean query latency improved by
`16.968 ms`, and the v2 hot-path check
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-episode-admission-v2.json`
processed `524288` tokens at `6412.209 tokens/sec` with bounded `12/65536`
route rows, `65526` cached rows, zero runtime failures, and GPU memory
`1812->1866 MiB` while sampler telemetry observed GPU-side contention.

Service replay planning now follows the same selected-window rule. The old
planner output was capped, but candidate construction still materialized all
runtime episodes, actions, predictions, uncertain domains, and recent feedback
before ranking. `build_replay_plan(...)` now records
`bounded_replay_plan_source_window.v1`: source windows are capped to `64` rows
per stream, recent feedback is capped and indexed at `128`, and up to `32`
feedback-target stubs preserve high-signal contradiction/correction evidence
outside the recent source tail. The ranking remains advisory, CPU-only, and
non-mutating with `runs_live_tick=false`, `gpu_used=false`, and no hidden
language reasoning. The benchmark
`reports/bounded_replay_window_20260618/replay-plan-source-window-bounded.json`
used `20000` episodes, actions, and predictions plus `2000` domains and `128`
feedback rows, still returned the old contradicted target `ep-42` as the top
candidate, and reduced mean plan latency from the pre-change unbounded
`6860.919 ms` to `14.684 ms` (`467.225x`) with traced peak allocation
`0.519 MiB` and zero CUDA/VRAM use. The paired 65536-column `524288`-token
protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-plan-source-window.json`
stayed in band at `6344.404 tokens/sec`, with
`train_compute=0.128679 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, flat `1799 MiB` GPU memory, and
zero graph/native/sequence failures. This retires the input-unbounded service
replay-plan shape while keeping replay selection inspectable and local to an
operator-reviewed slow/service window.

The shared bucket candidate source now has the same bounded-source contract as
the replay/query callers that consume it. The previous helper returned a capped
candidate list, but built `list(reversed(...))` for each selected bucket first,
so one hot bucket could still create source work proportional to bucket size.
`DualMemoryStore._candidate_indices_for_bucket_ids(...)` now requires an
explicit source budget, walks bucket tails with per-bucket cursors, round-robins
until the requested window is full, and reports
`candidate_source_window_policy=tail_indexed_bucket_round_robin_no_full_bucket_materialization`
plus source-read/materialization counts. The diagnostic benchmark
`reports/bounded_replay_window_20260618/bucket-candidate-source-window-bounded.json`
used a `65536`-entry hot bucket, preserved newest-candidate parity, read `32`
source entries within a `32`-entry source-read budget, materialized `0`, used
CPU archival/source placement with `0.0 MiB` CUDA allocation, and reduced mean
source latency from `0.416944 ms` to `0.060931 ms` (`6.843x`). This is not a
transformer-style memory scan; it is the
local source-window setup required before bounded associative recall, query
readout, frontier planning, or awake ripple tagging. The 2026-06-23 refresh
`..\..\MARULHO_reports\bounded_replay_window_20260623\bucket-candidate-source-window-explicit-budget.json`
removes the leftover optional full-bucket branch and still passes with `32`
source reads from a `65536` hot bucket, `32` scored replay candidates, `0`
materialized entries, `candidate_source_full_bucket_scan=false`, CPU archival
placement, `0.0 MiB` CUDA allocation, and `14.770743x` mean source-latency
improvement.
The paired long-run gate
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-bucket-candidate-explicit-budget-default-nosample.json`
reached `6047.414 tokens/sec`, p95 `21.647 ms`,
`train_compute=0.133284 ms/token`, bounded `12/65536` route rows, no observed
contention, RTX 3060 memory `1787->1788 MiB`, and zero graph/native sequence
failures.

HF replay query collection now applies the same bounded-source rule to retained
column anchors. The old runner capped returned query indices but still passed
every retained `column_anchors` bucket into the store collector, so
`_candidate_indices_for_bucket_ids(...)` built an all-anchor source pass before
the bounded query window. Anchor capture now records durable recency metadata,
refreshes dict recency when a bucket is recaptured, and checkpoints the recency
fields. `_collect_anchor_replay_queries(...)` emits
`bounded_replay_query_anchor_bucket_source_window.v1` before calling
`collect_replay_query_indices(...)`, passing at most `16` reverse-recency anchor
buckets and carrying that same candidate bucket window into
`_bounded_replay_recall_evaluation(...)`. The report states CPU archival and
active replay-query placement, no live tick, no every-token work, no raw replay
text, no hidden language reasoning, no global score/candidate scan, and
`anchor_source_full_scan=false`.

The benchmark
`reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json`
used `8192` retained anchors, `16` query budget, `32` recall candidates, and
`64` iterations. The retired all-anchor source pass averaged `16.414 ms`; the
bounded source path averaged `0.346 ms` (`47.373x`), used `16/8192` anchor
buckets, selected the newest anchor query indices with hit rate `1.0` versus
`0.0` for the retired pass, and passed exact input recall with
`mean_input_pattern_distance=0.0`. The benchmark pinned the trainer to CPU for
replay-query evidence and reported `0.0 MiB` CUDA allocation.

The inherited-scope follow-up
`..\..\MARULHO_reports\bounded_replay_window_20260623\replay-query-inherited-bucket-cap.json`
passed with an oversized `4096`-bucket inherited report capped to `16` buckets,
`4080` inherited buckets truncated, exact input recall
(`mean_input_pattern_distance=0.0`), bounded recent-anchor hit rate `1.0`, CPU
archival/active recall placement, no live tick, no every-token work, no global
candidate scan, no hidden language reasoning, and `0.0 MiB` CUDA allocation.
The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\replay-query-anchor-maintained-only.json`
keeps the all-anchor benchmark implementation absent while preserving the same
bounded hit rate, exact recall, inherited cap, CPU placement, and `0.0 MiB`
CUDA allocation.

The long hot-path protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json`
processed `524288` tokens at `6376.873 tokens/sec`, with
`train_compute=0.128288 ms/token`, `prepare_training=0.006247 ms/token`,
`finalize_total=0.005964 ms/token`, and `tick_duration_ms.p95=20.160`. Runtime
Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph, selection, native burst, and
native sequence failures were all `0`. GPU memory stayed flat at `1787 MiB`.
The velocity sampler observed borderline GPU contention at `20%`, so this is
accepted as hot-path protection and same-band throughput evidence, not a clean
hardware ceiling.

The same-checkpoint current rerun for the inherited cap
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-inherited-query-cap-pinned-main-rerun.json`
processed `524288` tokens at `6162.974 tokens/sec`,
`train_compute=0.130390 ms/token`, `prepare_training=0.006546 ms/token`,
`finalize_total=0.006625 ms/token`, and `tick_duration_ms.p95=20.644`. Runtime
Truth again stayed bounded at `route_input_rows_scored=12/65536`, no all-column
transition, no measurement-window polling, and zero graph/native sequence
failures. The sampler still marked GPU-side contention because the before-run
GPU sample was `22%`; use this as live-tick protection and same-band evidence,
not a claim that visible GPU utilization must reach a fixed percentage.

Sleep replay now consumes the same shared anchor-source operator before
calling `DualMemoryStore.select_replay_window(...)`. The previous trainer path
constructed a sorted list of every checkpointed `column_anchors` bucket, so the
store selector no longer scanned all memory entries but the source bucket set
still scaled with retained anchors. `replay_anchor_window.py` now owns that
logic for both HF replay-query and trainer sleep replay. The sleep surface is
`bounded_sleep_replay_anchor_bucket_source_window.v1`: it takes at most `16`
reverse-recency anchor buckets, records total/source/window counts and selected
anchor metadata, keeps archival metadata and source selection on CPU, reports no
live tick, no every-token work, no global score/candidate scan, no raw replay
text, no hidden language reasoning, and `anchor_source_full_scan=false`.

The focused benchmark
`reports/bounded_replay_window_20260622/sleep-replay-anchor-source-window-bounded.json`
used `8192` retained anchors and `64` iterations. The retired sorted
all-anchor source averaged `0.892263 ms`; the bounded source averaged
`0.037825 ms` (`23.589x`), read `16/8192` anchors, selected the newest anchor
source window with hit rate `1.0`, and the follow-up sleep-window selector chose
positive replay entries from those newest anchors with hit rate `1.0`. Full
selection latency moved from `7.797869 ms` with the retired all-anchor source to
`0.104864 ms` with the bounded source. CUDA was available but unused for
archival/source selection, with `0.0 MiB` allocation delta.
The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-replay-anchor-maintained-only.json`
keeps the all-anchor benchmark implementation absent while preserving bounded
source and selected-bucket hit rates at `1.0`, `16` selected replay rows, source
mean `0.039967 ms`, selection mean `0.101736 ms`, and `0.0 MiB` CUDA allocation.

The paired `524288`-token protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-sleep-replay-anchor-source-window.json`
processed `524288` tokens at `6135.629 tokens/sec`, with
`train_compute=0.132272 ms/token`, `prepare_training=0.006595 ms/token`,
`finalize_total=0.006411 ms/token`, bounded `12/65536` route rows, `10` output
candidates, `65526` cached transition rows, and
`state_transition_runs_all_columns=false`. Graph and native sequence failures
were `0`; RTX 3060 memory moved from `1615 MiB` to `1614 MiB`. The environment
sampler observed GPU utilization at the `20%` contention threshold, so this is
accepted as same-band live-tick protection and retirement evidence for the
all-anchor sleep source pass, not as a new speed ceiling.

The isolated replay-adapter experiment stack is also retired and removed. It
kept dry-run training approval, dry-run plan generation, metadata-only adapter
artifacts, an experimental promotion gate, and replay-to-adaptation experiment
evidence beside the maintained replay/sample/readout/sleep paths. That stack
did not mutate the runtime, but it made replay adaptation look service-visible:
`REPORT_SUMMARY_KINDS` exposed adapter report kinds and the plan included a
stale executable-looking adapter command. MARULHO now deletes those modules and
old tests, moves generic JSON/hash helpers to `marulho.evaluation.artifact_io`,
and adds `tests/test_replay_adapter_stack_retired.py` to keep the old modules
and service report kinds absent. Future adapter experiments must begin as
bounded offline proposals with explicit quality evidence, no hidden replay-text
reasoning, no live/every-token work, no runtime mutation, and hot-path
protection evidence before any active exposure. The paired protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-adapter-stack-retired.json`
processed `524288` tokens at `6167.298 tokens/sec`, with
`train_compute=0.131572 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, zero graph/native sequence failures, and RTX memory
`1728->1729 MiB`; GPU utilization touched the `20%` contention threshold, so
this is same-band protection evidence rather than a speed ceiling.

SNN Readout Evidence Ledger normalization now follows the same selected-source
rule before replay/readout review methods or ledger status snapshots touch
retained event history. The old normalizer built `list(...)` for every retained
ledger event family and then re-wrapped those lists in capped deques. The active
`bounded_snn_readout_ledger_normalization_source_window.v1` reads only the
newest `128` records per retained event family before deepcopy/review, reports
`recent_ledger_event_field_source_window_v1`, CPU archival/normalization
placement, `max_records_total=2944`, no global candidate/score scan, no live
tick, no every-token cadence, no hidden language reasoning, and no CUDA archive.
The refreshed benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-source-window.json`
uses the same `23` event families with `2048` records each and now covers both
normalization and store-state persistence. The bounded normalizer read `2944`
source rows instead of `47104`, preserved newest-first recall
(`bounded_recent_retention_rate=1.0` versus `0.0` for the retired
full-materialize-then-maxlen shape), and reduced mean normalization latency from
`2415.385992 ms` to `159.388156 ms`. The remaining hand-written `_store_state`
copy path is also retired: `bounded_snn_readout_ledger_store_state_source_window.v1`
uses the same event-field helper, reads `2944` rows instead of `47104`, preserves
newest-first store-window parity with the retired list-slice shape, and measured
`159.156636 ms` versus `169.042904 ms` (`1.062117x`). The report used
`6.514462 MiB` traced Python peak allocation and `0.0 MiB` CUDA
allocation/reservation on RTX 3060. A follow-up readout-priority benchmark still
matched the full-retained top candidate while scoring `32/2048` events at
`1.253520 ms`, and rollout rehearsal still matched top quality while scoring
`16/2048` events at `2.705792 ms`. The 65536-column `524288`-token no-profile
protection rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json`
stayed in band at `6044.412 tokens/sec`, with bounded `12/65536` route rows, no
observed contention, GPU memory `2029->2032 MiB`, and zero graph/native sequence
failures. This retires broad retained-ledger copy shapes without promoting
ledger normalization or checkpoint-style persistence into live replay. The
remaining replay-provenance helper that checks known readout evidence hashes now
uses a single `events` source window instead of calling the all-family
normalizer. The same benchmark report records
`bounded_snn_readout_known_evidence_hash_source_window.v1`, preserves known-hash
set parity, checks `128` `events` rows instead of `2944` normalized ledger rows
(`23x` less source work), and reduces mean lookup latency from `156.192384 ms`
to `6.792628 ms` (`22.994397x`) with CPU lookup placement, `6.538644 MiB`
traced Python peak, and no CUDA allocation/reservation. The paired 65536-column
`524288`-token rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json`
stayed in the same sustained band at `5938.461 tokens/sec` with bounded
`12/65536` route rows, `65526` cached transition rows, no observed contention,
GPU memory `2032->2031 MiB`, and zero graph/native sequence failures. This is
throughput protection for a slow replay/readout gate, not a speed promotion.

The 2026-06-20 source-window binding follow-up removes the remaining hash-only
production bypass. The set-only `_known_readout_evidence_hashes()` and
`known_readout_evidence_hashes()` helpers are gone; replay design, dry-run,
plasticity preflight, plasticity bridge, and evaluated replay-artifact recording
must carry the bounded known-readout source-window report with the selected
hashes. This matches the research boundary: associative recall can behave like
an attention/Hopfield-style local memory operator only after a bounded source
set is selected, and consolidation/replay gates must keep selection evidence
visible rather than passing bare identities. The report
`reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json`
passed with source window `1/8`, persisted
`readout_evidence_source_window_hash`, no global scan, no raw text, no language
reasoning, no live-tick/every-token work, CPU archival placement, `0.014095 MiB`
traced Python peak, and `0.0 MiB` CUDA allocation/reservation. Indexed
provenance verification reduced worst-case retained lookup checks from `256` to
`4` (`64x`). The paired `524288`-token rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json`
stayed in the 6k-ish band at `6007.228 tokens/sec` with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native sequence
failures, flat RTX 3060 memory, and observed GPU contention. This is deletion
of a report-dropping path, not a broader replay cadence.

The replay-priority selector now follows the same rule. A transition-memory
replay artifact proposal may only be treated as operator-review-ready when the
readout replay-priority report's bounded source window is carried forward and
hashed. This prevents the artifact chain from trusting a replay window selected
by `replay_priority(...)` while dropping the selector budget, CPU placement,
and no-hidden-work evidence. The report
`reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json`
passed with replay-priority source window `1/32`, persisted
`replay_priority_source_window_hash`, CPU archival and score placement, no
global scan, no raw text, no language reasoning, no live-tick/every-token work,
CUDA available but unused, `0.014385 MiB` traced Python peak, and `0.421992 ms`
mean permit-verification latency. The first long protection run was rejected at
`4662.031 tokens/sec` under observed GPU contention; the no-contention rerun
stayed same-band at `5937.908 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, flat RTX 3060 memory, and zero graph/native
sequence failures. This retires another report-dropping artifact shape rather
than adding a second replay-priority path.

Raw caller-window transition-memory replay artifact recording is now retired
from production instead of kept as a compatibility path. Replay artifacts are
recorded only through
`record_evaluated_snn_transition_memory_replay_artifact(...)`, where the
artifact is derived from a verified internal-ledger proposal, review ticket,
known-readout source window, replay-priority source window, and provenance
source window. Controller load drops raw caller-window artifacts, and permit
verification recomputes all three persisted source-window hashes before an
artifact can authorize regeneration. The report
`reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json`
passed with `public_raw_recorder_callable=false`,
`raw_loaded_artifact_count=0`, `raw_artifact_index_hit=false`, mean
verification latency `0.538460 ms`, traced Python peak `0.017773 MiB`, no CUDA
allocation/reservation, and indexed provenance verification at `4` bounded
lookups instead of `256` retained-record checks (`64x`). The paired
`524288`-token run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json`
stayed in the maintained band at `6004.719 tokens/sec`, with bounded
`12/65536` route rows, `65526` cached transition rows, zero graph/native
sequence failures, and RTX 3060 memory `1863->1865 MiB`; velocity observed GPU
contention, so this is same-band protection, not a new speed ceiling.

The 2026-06-20 follow-up removes the remaining production all-family normalizer
callable instead of keeping it as dead code. `SNNLanguageReadoutEvidenceLedger`
no longer exposes `_normalized_state()`, and the normalization policy string is
owned by benchmark-local retired comparisons only. The report
`reports/bounded_replay_window_20260620/snn-readout-ledger-normalization-production-normalizer-retired.json`
marks the all-family comparison `production_callable=false` and
`benchmark_local_only=true`, passes all quality checks, keeps bounded
all-family comparison at `2944` rows versus `47104` for the full-materialized
legacy model (`16x`), preserves newest-first retention, and uses `0.0 MiB`
CUDA allocation/reservation with CPU archival/normalization placement. The
matching hot-path run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-readout-ledger-production-normalizer-retired.json`
stayed in the same sustained band at `6224.717 tokens/sec`, bounded route
scoring at `12/65536`, cached `65526` transition rows, and zero graph/native
sequence failures; the `21%` GPU sample is recorded as borderline contention,
so this is a no-dead-path throughput-protection result rather than a new speed
claim.

Dense-label history and calibration policy now follow the same one-family rule:
`bounded_snn_dense_label_candidate_calibration_source_window.v1` reads only
`dense_label_candidate_events` before operator history or calibration ranking.
The follow-up report
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json`
preserved history-hash, policy-hash, and ready-candidate parity while checking
`128` dense-label rows instead of `2944` normalized ledger rows (`23x` less
source work), reducing mean history+policy latency from `222.453668 ms` to
`49.093244 ms` (`4.531248x`), and using CPU archival/lookup placement with
`9.320584 MiB` traced Python peak and no CUDA allocation/reservation. The
paired `524288`-token protection run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json`
stayed in band at `6018.915 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`2030->2029 MiB`, and zero graph/native sequence failures.
The evaluation gate is closed over the same one-family evidence: after preflight
selection, `bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1`
resolves only the selected dense-label hashes from
`dense_label_candidate_events`, keeps CPU archival/lookup/evaluation placement,
and reports no global scan, no raw text payload, no hidden language reasoning,
no live tick, no every-token cadence, no mutation/plasticity, and no CUDA
archive. The report
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json`
preserved sample-hash and calibration-metric parity for `8` selected samples
while checking `128` dense-label rows instead of `2944` normalized ledger rows
(`23x` less source work), reducing mean evaluation latency from
`225.545020 ms` to `12.673884 ms` (`17.796046x`) with `9.320584 MiB` traced
Python peak and no CUDA allocation/reservation. The first `524288`-token run at
`5906.886 tokens/sec` is retained as below-band variance evidence; the rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json`
stayed in band at `6116.710 tokens/sec`, `train_compute=0.133135 ms/token`,
bounded `12/65536` route rows, no observed contention, GPU memory
`2030->2030 MiB`, and zero graph/native sequence failures.

Dense-label calibration update application and application-review now share the
same one-family rule. Operator and autonomous update executors, plus their
read-only application reviews, read only
`dense_label_calibration_update_events` through
`bounded_snn_dense_label_calibration_update_source_window.v1` before duplicate
or current-update lineage checks. The mutation side stores only that update
event family, current update, total count, and last-applied timestamp, so
unrelated ledger families are not rewritten through `_store_state(...)`. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json`
preserved event-hash and current-hash parity while checking `128` update rows
instead of `2944` normalized ledger rows (`23x` less source work), reducing
mean lookup latency from `245.671760 ms` to `11.647260 ms` (`21.092666x`),
with CPU archival/lookup/write placement, `9.329722 MiB` traced Python peak,
CUDA available but unused, no raw text payload, no hidden language reasoning,
no live tick, and no every-token cadence. The paired long protection run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-update-source-window.json`
processed `524288` tokens at `6009.497 tokens/sec`,
`train_compute=0.134959 ms/token`, bounded `12/65536` route rows, GPU memory
`2045->2046 MiB`, and zero graph/native sequence failures. Its velocity sample
reported GPU-side contention at `21%`, so this is same-band protection
evidence, not a new speed claim.

Autonomous calibrated confidence-use now follows the same one-family rule. The
hash-only executor and its read-only event review read only
`autonomous_confidence_use_events` through
`bounded_snn_autonomous_confidence_use_source_window.v1`; the mutation side
stores only that event family, total use count, and last-used timestamp. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json`
preserved event-hash parity while checking `128` confidence-use rows instead of
`2944` normalized ledger rows (`23x` less source work), reducing mean lookup
latency from `350.647280 ms` to `13.261960 ms` (`26.439331x`). The report keeps
archival/lookup/write placement on CPU, uses no CUDA archive, loads no raw text,
does no hidden language reasoning, and does not run in the live tick or every
token. The paired `524288`-token protection run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json`
processed `524288` tokens at `5965.377 tokens/sec`,
`train_compute=0.136087 ms/token`, bounded `12/65536` route rows, GPU memory
`2045->2047 MiB`, no observed contention, and zero graph/native sequence
failures. The old broad-normalized duplicate/review lookup is retired as
benchmark-only evidence.

Readout-ledger recorders now use the same one-family rule for writes, not only
for advisory lookups. `record_readout_draft(...)`,
`record_readout_rollout_replay_evaluation(...)`,
`record_readout_emission_review(...)`, and
`record_dense_readout_label_candidate_review(...)` append through
`bounded_snn_readout_ledger_record_family_source_window.v1`: each recorder
reads only its target event family for duplicate detection, then persists only
that event family plus its total-count and timestamp fields. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json`
preserved latest-hash and total-count parity while checking `128` `events` rows
instead of `2944` normalized ledger rows (`23x` less source work), reducing mean
append latency from `883.251340 ms` to `57.255420 ms` (`15.426511x`). The report
keeps archival/lookup/write placement on CPU, uses no CUDA archive, loads no raw
text, performs no hidden language reasoning, and does not run in the live tick or
every token. The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-record-family-append.json`
processed `524288` tokens at `5966.765 tokens/sec`,
`train_compute=0.136141 ms/token`, bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, GPU memory `2046->2043 MiB`, and zero
graph/native sequence failures. The old broad-normalized single-family record
append shape is retired as benchmark-only evidence.

The autonomous hash-readout binding and bound-observation chain now uses the
same event-family source window on both write and review. Binding execution and
review read only `autonomous_hash_readout_binding_events`; observation execution
and review read only `autonomous_bound_readout_observation_events`. The active
chain reports
`bounded_snn_autonomous_hash_readout_event_family_chain_source_window.v1`, keeps
CPU archival/lookup/write placement, and forbids raw text, hidden language
reasoning, live-tick replay, every-token cadence, CUDA archival metadata, and
plasticity. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json`
preserved binding/observation hash, review-match, and total-count parity while
checking `512` target-family rows instead of `11776` normalized rows (`23x` less
source work), reducing mean chain latency from `2371.472400 ms` to
`110.685950 ms` (`21.425234x`). The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-autonomous-chain.json`
processed `524288` tokens at `6272.156 tokens/sec`,
`train_compute=0.130202 ms/token`, bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, GPU memory `2044->2045 MiB`, and zero
graph/native sequence failures. The old broad-normalized binding/observation
append and review shape is retired as benchmark-only evidence.

The downstream autonomous training-window and decoder-probe chain now shares
that same one-family rule instead of reopening the whole ledger after
observation review. Training execution/review read only
`autonomous_readout_training_window_events`; decoder-probe execution/review read
only `autonomous_decoder_probe_events`. The expanded report
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-training-probe-chain.json`
preserved binding, observation, training, and decoder hash/review/count parity
while checking `1024` target-family rows instead of `23552` normalized rows
(`23x` less source work), reducing mean chain latency from `4927.213200 ms` to
`197.573467 ms` (`24.938638x`). It keeps archival/lookup/write placement on
CPU, uses no CUDA archive (`0 MiB` allocation/reservation despite RTX 3060
availability), loads no raw text, performs no hidden language reasoning, and
does not run in the live tick or every token. The paired `524288`-token
hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json`
processed `524288` tokens at `6057.953 tokens/sec`,
`train_compute=0.134322 ms/token`, bounded `12/65536` route rows, `65526` cached
transition rows, GPU memory `2046->2064 MiB`, and zero graph/native sequence
failures. GPU-side contention was observed at `24%`, so this is throughput
protection evidence under contention, not a clean speed ceiling. The old
broad-normalized autonomous training/probe append and review shape is retired
as benchmark-only evidence.

SNN readout replay dry-run and plasticity bridge payloads now obey the same
bounded-source rule after replay design has selected candidates. The active
path caps caller-supplied `selected_replay_targets`, dry-run
`ephemeral_replay.trace`, and `candidate_replay_sequences` to `32` records
before tensor work or bridge canonicalization, and emits
`bounded_snn_readout_replay_dry_run_target_window.v1`,
`bounded_snn_readout_plasticity_preflight_trace_window.v1`, and
`bounded_snn_readout_plasticity_bridge_sequence_window.v1`. The reports state
CPU archival placement, CPU active replay computation for this benchmark, no
global candidate/score scan, no live tick, no every-token cadence, no raw replay
text, and no hidden language reasoning. The benchmark
`reports/bounded_replay_window_20260619/readout-replay-target-window.json`
passed with dry-run `32/2048`, bridge `32/2048`, `64x` source-work reduction,
mean dry-run latency `6.061784 ms`, mean bridge latency `1.328924 ms`, and
`0.0 MiB` CUDA allocation/reservation. The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json`
stayed in the maintained band at `6109.000 tokens/sec`,
`train_compute=0.133186 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, GPU memory `2020->2018 MiB`,
and zero graph/native sequence failures. This retires caller-supplied
full-payload replay materialization without adding a second implementation
path.

The same replay/consolidation boundary now applies inside the exported SNN
language plasticity semantics path, not only the readout-ledger adapter. A
modern-Hopfield-style association is acceptable only after a local replay window
exists; CLS, continual replay, synaptic tagging/capture, latent replay, and
sparse replay do not justify caller-sized replay records or sparse-index lists
inside the plasticity review surface. MARULHO therefore bounds
`evaluate_spike_language_plasticity_replay(...)`,
`run_spike_language_plasticity_replay_experiment(...)`, and
`build_spike_language_plasticity_shadow_delta(...)` with
`SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT=32`; shadow delta additionally caps
each sparse side at `16` indices before pair scoring. The benchmark
`reports/bounded_replay_window_20260619/language-plasticity-replay-window.json`
passed with replay evaluation `32/2048`, replay experiment `32/2048`, shadow
pair checks `8192/134217728`, `64x` record-work reduction, `16384x` pair-work
reduction, CPU archival/source/replay placement, `14.474813 MiB` traced Python
peak allocation, and `0.0 MiB` CUDA allocation/reservation. The matching
`524288`-token rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json`
stayed in the maintained band at `5999.398 tokens/sec`, bounded `12/65536`
route rows, `65526` cached transition rows, and zero graph/native sequence
failures, but `velocity_environment.v1` marked GPU contention at `22%`; treat it
as protection evidence, not clean top-speed evidence. The old full-payload
semantics behavior is retired rather than left as a side implementation.

The checkpointed language application boundary now extends that same local
selection rule through the final mutation gate. Replay and shadow-delta paths
may prepare structural evidence, but the executor must still reject caller-sized
mutation payloads before topology validation or checkpoint writes.
`SNNLanguagePlasticityApplicationExecutor.apply_live_application(...)` and
`regenerate_transition_memory(...)` both use
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32`, expose
`bounded_snn_language_plasticity_live_application_synapse_window.v1` and
`bounded_snn_transition_memory_regeneration_candidate_synapse_window.v1`, and
require the source payload to be untruncated before mutation. The benchmark
`reports/bounded_replay_window_20260619/language-application-synapse-window.json`
blocked oversized live/regeneration payloads at `32/2048` with zero checkpoint
calls and zero state mutation, while exact-window payloads still applied or
regenerated `32` synapses through checkpointed paths. The report recorded CPU
archival/source/application placement, `1.982166 MiB` traced Python peak,
`0.0 MiB` CUDA allocation/reservation, no global candidate/score scan, no raw
text payload, and `64x` projected source-work reduction. The clean
`524288`-token hot-path run stayed in band at `6039.734 tokens/sec`, bounded
`12/65536` route rows, no observed contention, and zero graph/native sequence
failures. This retires the downstream caller-sized application side path rather
than keeping it as a second implementation.

The rollout-regeneration facade now applies the same rule before replay
permits and application preflight can authorize the checkpointed executor.
`RuntimeFacade.snn_language_readout_rollout_regeneration_permit_request(...)`,
`snn_language_readout_rollout_regeneration_application_preflight(...)`, and
`snn_language_readout_rollout_regeneration_application(...)` use the shared
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32` source-window operator,
forward only the bounded regeneration design, and require untruncated candidate
payloads before replay-controller or executor calls. The benchmark
`reports/bounded_replay_window_20260619/rollout-regeneration-facade-candidate-window.json`
blocked oversized permit, preflight, and application payloads at `32/2048`,
with zero replay-controller calls for oversized permits and zero
executor/checkpoint calls for oversized applications; exact-window flow still
reached the single executor path. It recorded CPU archival, source-window,
facade-gate, and active-application placement, `1.852119 MiB` traced Python peak, `0.0 MiB` CUDA
allocation/reservation, no global candidate/score scan, no raw text payload, no
hidden language reasoning, and `64x` projected source-work reduction. The
accepted `524288`-token rerun stayed in band at `6121.143 tokens/sec` with
bounded `12/65536` route rows, flat `2031 MiB` GPU memory, zero graph/native
failures, and sampled GPU contention. The old facade full-list route is now a
retired projection, not an implementation kept beside the selected path.

The readout-ledger rollout consolidation and regeneration-review chain now
shares that single bounded candidate path before the facade sees a permit
preview. Modern Hopfield-style association remains local to a selected window,
and CLS/continual replay/STC/sparse replay evidence rejects materializing every
candidate synapse during slow-path review. The ledger now applies
`bounded_application_synapse_window(...)` to sparse transition candidates,
design candidate synapses, developmental growth candidates, regeneration-design
candidate synapses, and Replay Controller regeneration-design normalization.
`reports/bounded_replay_window_20260619/readout-ledger-rollout-candidate-window.json`
passed with exact `32/32` rollout evidence reaching permit preview while
oversized design, shadow, developmental, adapter, replay-artifact review, and
direct controller payloads blocked at `32/2048`. The report records CPU
archival/source/gate placement, no global candidate/score scan, no raw text
payload, no hidden language reasoning, no live tick, no every-token cadence,
`64x` projected source-work reduction, `9.073439 MiB` traced Python peak, and
`0.0 MiB` CUDA allocation/reservation. The clean hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json`
processed `524288` tokens at `6075.293 tokens/sec` with bounded `12/65536`
route rows, `65526` cached rows, no observed contention, GPU memory
`2031->2043 MiB`, and zero graph/native sequence failures. This retires the
readout-ledger and controller full-list materialization path rather than
keeping a second review implementation.

The dense-readout training executor now applies the same local-window rule to
checkpointed training transitions. The old path copied every caller-supplied
`training_transitions` record and each caller-sized `pre_indices`/`post_indices`
list before slicing, while service schema/read-model budgets still advertised a
larger side entrance. `apply_dense_readout_training_loop(...)` now shares
`SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT=32` and
`SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT=32` with the schema,
design, and preflight surfaces, emits
`bounded_snn_dense_readout_training_transition_source_window.v1` and
`bounded_snn_dense_readout_training_transition_index_window.v1`, and requires
untruncated transition and sparse-index windows before checkpoint mutation. The
benchmark
`reports/bounded_replay_window_20260619/dense-readout-training-transition-window.json`
blocked oversized transition and index payloads at `32/2048` with zero
checkpoint calls and zero state mutation, while exact-window training still
committed `32` dense/sparse transition updates through checkpoints. It recorded
CPU archival/source/training placement, `5.696876 MiB` traced Python peak,
`0.0 MiB` CUDA allocation/reservation, no global candidate/score scan, no raw
text payload, no hidden language reasoning, and `64x` projected transition-work
reduction. The clean `524288`-token hot-path run stayed in band at
`6028.820 tokens/sec`, bounded `12/65536` route rows, no observed contention,
and zero graph/native sequence failures. This retires the caller-sized
checkpointed training side path instead of keeping it beside the selected
replay/application path.

Strong-capture slow-memory admission now follows the same selected replay rule.
Synaptic tagging/capture motivates retaining unusually strong traces for later
stabilization, but it does not justify an immediate archive write for every
tag. `bounded_strong_capture_admission_cadence.v1` keeps device strong-event
evidence, then admits at most one strong capture per
`slow_memory_archive_strong_capture_min_interval_tokens` window. The production
default is `16`, config validation rejects values `<=1`, and Runtime Truth
reports the min interval, strong archive count, refractory skip count, and last
archived strong token. The benchmark
`reports/bounded_replay_window_20260618/strong-capture-admission-cadence.json`
forced `256` strong candidates and archived `17` records instead of the retired
every-strong projection of `256`, a `15.058824x` write reduction, while keeping
max selected strong gap at `16` and final gap at `14`. The report states CPU
archival storage, active replay computation `none`, no CUDA allocation, no
global candidate/score scan, no raw text payload except archived entries, and
no hidden language reasoning. The `524288`-token hot-path check stayed in band
at `6100.415 tokens/sec` with `train_compute=0.133405 ms/token`, bounded
`12/65536` route rows, flat `2390 MiB` GPU memory, and zero runtime failures;
observed GPU contention keeps it as hot-path protection evidence rather than a
clean speed ceiling.

The maintained-only refresh
`..\..\MARULHO_reports\bounded_replay_window_20260624\strong-capture-admission-projection-removed.json`
removes the every-strong projection object from the benchmark report. It keeps
the same selected strong-capture path, records
`retired_every_strong_admission_absence.implementation_present=false`, archives
`17` bounded CPU records with `16` selected strong archives, skips `239`
refractory writes, projects `239` removed every-strong writes in the memory
budget, averages `1335.328410 ms`, and keeps CUDA allocation/reservation at
`0.0 MiB`.

Fixed-cadence slow-memory admission is now retired as an executable
per-token fallback. The maintained path archives only the first retained token
and selected strong captures; ordinary `slow_memory_archive_interval_tokens`
hits record `cadence_deferred` in the cognitive boundary controller and do not
call `DualMemoryStore.update(...)`. The retirement benchmark
`reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json`
kept the first-token record, produced `1` bounded archive versus `17` retired
fixed-cadence writes over `256` tokens (`17x` less archival write work), kept
the slow-memory fixed-cadence execution gate closed, reported CPU archival
placement, and performed no active replay computation, global scan, raw-text
reasoning, or hidden language reasoning. The refreshed strong-capture report
`reports/bounded_replay_window_20260620/strong-capture-admission-cadence-after-fixed-cadence-retirement.json`
still archived `17` selected strong captures versus `256` retired every-strong
writes (`15.058824x`), so useful STC-like selection remains available without
fixed-cadence writes. The first same-code `524288`-token run after retirement
is retained as below-band variance at `5758.051 tokens/sec`; the accepted rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json`
stayed in the maintained band at `6043.321 tokens/sec`,
`train_compute=0.134537 ms/token`, bounded `12/65536` route rows, `65526`
cached rows, `2048` deferred cadence hits, zero graph/native sequence failures,
and flat RTX 3060 memory at `1958 MiB`. Borderline GPU contention makes this
protection evidence, not a new speed ceiling.

The current maintained-only fixed-cadence report
`..\..\MARULHO_reports\bounded_replay_window_20260624\slow-memory-fixed-cadence-projection-removed.json`
removes the fixed-cadence projection object from the benchmark report, records
`retired_fixed_cadence_admission_absence.implementation_present=false`, keeps
`1` first-token archive, removes `16` projected fixed-cadence writes, defers
`16` cadence hits, averages `1326.868180 ms`, and keeps CUDA
allocation/reservation at `0.0 MiB`. The paired current-tree protection run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-slow-memory-admission-projections-removed-default-nosample.json`
processed `524288` tokens at `5957.637 tokens/sec`, p95 `21.679 ms`,
`train_compute=0.135551 ms/token`, `prepare_training=0.006811 ms/token`,
`finalize_total=0.006772 ms/token`, bounded `12/65536` route rows, `65526`
cached rows, `2048` deferred cadence hits, native sequence-loop and burst-replay
failure counts `0`, no observed before/after contention (`cpu max=22%`,
`gpu max=12%`), and RTX 3060 memory `2047->2046 MiB`.

Source tick sleep replay is also deferred out of the live service fallback.
BrainRuntime still owns source selection and tick orchestration, but it no
longer lets a per-token fallback run deep, micro, or repair sleep replay just
because metrics or burst eligibility forced `train_step`. The fallback now
passes `allow_sleep_maintenance=False`, exposes deferred sleep through trainer
metrics, and leaves sleep replay to explicit slow-path calls. The benchmark
`reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json`
passed with service fallback sleep calls `0`, explicit slow-path projection
sleep calls `1`, `sleep_maintenance_deferred=1`, CPU archival placement, no
global scan, no live replay execution, and no hidden language reasoning. The
paired `524288`-token protection run stayed in the maintained band at
`5993.959 tokens/sec`, `train_compute=0.135624 ms/token`, bounded `12/65536`
route rows, `65526` cached rows, no observed contention, flat RTX 3060 memory
at `1959 MiB`, and zero graph/native sequence failures.

Live memory summaries follow the same slow-window rule. Trainer telemetry,
BrainRuntime summaries, living-loop status, and status Runtime Truth call
`DualMemoryStore.live_summary_stats()` instead of full `summary_stats()`.
The live projection emits `bounded_memory_summary_projection.v1`, reports
fill/counter/last-report fields, keeps `summary_full_memory_scan=false` and
`summary_scan_entry_count=0`, and does not advance STC tag/PRP decay. Full
`summary_stats()` remains available only for explicit offline consolidation and
quality runners. The repo-local benchmark now removes the executable full
summary comparator too: the current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260623\live-memory-summary-legacy-baseline-removed.json`
passed with scalar fill/report parity, `retired_live_full_summary_scan_absence.implementation_present=false`,
`65536` retired scan rows removed, `0.237312 ms` mean bounded projection
latency, `0.259369 MiB` Python peak, and `0.0 MiB` CUDA allocation. The clean
paired protection run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-live-summary-legacy-baseline-removed-default-nosample-rerun.json`
processed `524288` tokens at `6530.655 tokens/sec`,
`train_compute=0.123872 ms/token`, bounded `12/65536` route rows, `65526`
cached rows, no observed contention, CPU max `6%`, GPU max `10%`, flat RTX
3060 memory `1929->1929 MiB`, and zero graph/native sequence failures. Status
remains read-only projection, not replay selection or hidden consolidation.

## Status

bounded slow-path selection, stored-experience recall, reconstruction-gated
candidate repair, reconstruction-guarded HF replay acceptance, skipped repeated
rejected replay attempts, target-specific repair-strength budgets, tensor-only
sleep replay payloads, bounded repair-replay input preparation,
selected-window SFA correction, capped pre-score replay candidate windows,
capped replay query collection, bounded query-memory
readout, returned-only query text payloads, bounded concept-frontier metrics,
bounded semantic frontier-gap planning, bounded recent tag/anchor setup,
merged bounded source-bank semantic recall, bounded runtime concept memory lookup,
bounded context-comparison memory recall, reported SFA sampling, bounded
query-memory episode readout, bounded source-episode admission,
bounded replay-plan source windows, bounded replay-query anchor-source windows,
bounded replay-dataset preview/source-link windows,
bounded emission replay-context review windows,
bounded generic replay-context observed-slot windows,
bounded status replay-path projections,
bounded readout-ledger normalization/store-state source windows,
bounded dense-label calibration update source windows,
bounded autonomous confidence-use source windows,
bounded autonomous output-chain source windows,
bounded autonomous text-surface source windows,
bounded readout replay target/sequence windows,
bounded checkpointed application synapse windows,
bounded rollout-regeneration facade candidate windows,
bounded readout-ledger rollout candidate windows,
bounded dense-readout training transition windows,
strong-capture admission cadence, fixed-cadence slow-memory admission
retirement, source tick sleep replay deferral, bounded live memory-summary
projection, awake-ripple tagging, and retired unscoped
random replay defaults plus the full-buffer replay-score, score-tensor,
list-only replay/SFA, concept-frontier report-dropping, input-unbounded
replay-plan construction, linear replay-artifact provenance lookups, and
source-bank wrapper APIs, retained-ledger replay-path status scans,
readout-ledger full-materialization normalization, every-strong slow-memory
admission, caller-supplied full-payload readout replay materialization,
caller-supplied checkpointed application synapse materialization,
runtime-facade rollout-regeneration full-payload candidate materialization,
readout-ledger rollout full-payload candidate materialization,
broad-normalized dense-label calibration update lookup/write,
broad-normalized autonomous confidence-use lookup/write,
broad-normalized autonomous output-chain lookup/write,
broad-normalized autonomous text-surface lookup/write,
replay-controller regeneration-design full-payload normalization,
caller-supplied emission replay-context full-payload bridge,
caller-supplied generic replay-context full-payload observed-slot bridge,
caller-supplied checkpointed dense-readout training transition materialization,
plus the all-anchor HF replay-query source pass and full hot-bucket candidate
source materialization;
future larger replay windows still require repeated long-run hot-path and
grounding checks

Autonomous hash-only output evidence now uses the same record-family source
window before duplicate checks or event-review lookup. `execute_autonomous_language_output(...)`
and `autonomous_language_output_event_review(...)` read only
`autonomous_language_output_events`; `execute_autonomous_decoded_output(...)`
and `autonomous_decoded_output_event_review(...)` read only
`autonomous_decoded_output_events`. The broad `_normalized_state()` production
path is retired because it normalized unrelated replay/readout families before
one output event could be appended or reviewed. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-output-chain.json`
preserved hash, review-match, and total-count parity across binding,
observation, training, decoder probe, language output, and decoded output while
checking `1536` target-family rows instead of `35328` broad-normalized rows
(`23x`) and reducing mean chain latency from `6778.768800 ms` to
`321.988933 ms` (`21.052801x`). The paired `524288`-token hot-path run stayed
in the maintained band at `6048.638 tokens/sec`, with bounded `12/65536` route
rows, `65526` cached transition rows, zero graph/native sequence failures, and
GPU memory `2046->2047 MiB` under observed GPU contention. This keeps output
review as selected CPU-resident ledger evidence rather than hidden language
reasoning or every-token replay work.

The downstream text-surface side of that output chain follows the same
selected-source rule. `execute_autonomous_bounded_text_emission(...)` and
`autonomous_bounded_text_emission_event_review(...)` read only
`autonomous_bounded_text_emission_events`; `execute_autonomous_text_surface_commit(...)`
and `autonomous_text_surface_commit_event_review(...)` read only
`autonomous_text_surface_commit_events` and update/read the single
`current_text_surface_commit` pointer without broad ledger normalization. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-text-surface-chain.json`
preserved hash, review-match, total-count, and current-commit parity across the
eight-family autonomous text-surface chain while checking `2048` target-family
rows instead of `47104` broad-normalized rows (`23x`) and reducing mean chain
latency from `9289.008333 ms` to `429.436800 ms` (`21.630676x`). The paired
`524288`-token hot-path run stayed in band at `5980.715 tokens/sec`, with
bounded `12/65536` route rows, `65526` cached transition rows, zero graph/native
sequence failures, no observed contention, and GPU memory `2045->2047 MiB`.
This keeps text-surface commit as selected CPU-resident ledger evidence rather
than an always-on or hidden language replay path.

Autonomous text-surface materialization and bounded language-surface commit now
complete that same selected-source chain. `execute_autonomous_text_surface_materialization(...)`
and its review read only `autonomous_text_surface_materialization_events`; the
bounded language-surface commit executor/review read only
`autonomous_bounded_language_surface_commit_events` and preserve
`current_text_surface_materialization` plus
`current_bounded_language_surface_commit` as single current pointers. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json`
preserved hash, review-match, total-count, and current-pointer parity across
the full ten-family autonomous language-surface chain while checking `2560`
target-family rows instead of `58880` broad-normalized rows (`23x`) and
reducing mean chain latency from `11175.229267 ms` to `525.534133 ms`
(`21.264517x`). The paired `524288`-token hot-path run stayed in the maintained
band at `5994.060 tokens/sec`, with bounded `12/65536` route rows, `65526`
cached transition rows, zero graph/native sequence failures, CUDA runtime on
the RTX 3060, and GPU memory `2044->2059 MiB`. The replay/ledger benchmark
itself kept archival/source/review metadata on CPU, reported no live tick or
every-token work, and did no hidden language reasoning.

Bounded language-surface use and SNN language-generation now continue that
chain without adding a second runtime path. The use executor/review read only
`autonomous_bounded_language_surface_use_events`; the generation executor/review
read only `autonomous_snn_language_generation_events`. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json`
preserved hash, review-match, total-count, and current-pointer parity across
the expanded autonomous language-generation chain while checking `3072`
target-family rows instead of `70656` broad-normalized rows (`23x`) and
reducing mean chain latency from `13505.919533 ms` to `631.221 ms`
(`21.396499x`). The paired `524288`-token hot-path run stayed in band at
`6074.417 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, zero graph/native sequence failures, no observed contention,
CUDA runtime on RTX 3060, and GPU memory `2044->2047 MiB`. The replay/ledger
benchmark kept archival/source/review metadata on CPU, reported no live tick or
every-token work, and did no hidden language reasoning.

SNN language decoding through readout-structural-plasticity now follows the same
single-family ledger boundary. Decoding, readout-surface, readout-memory,
readout-consolidation, and readout-structural-plasticity executor/review pairs
read only their target event family, return
`bounded_snn_readout_ledger_record_family_source_window.v1`, and do not normalize
all retained readout/replay ledger families. The benchmark
`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-memory-canonical.json`
passed after the readout-memory production rename with bounded mean
`380.146067 ms` versus legacy diagnostic `5619.687233 ms` (`16x` source-work
reduction), while the autonomous-chain bounded mean stayed `944.358700 ms`
versus `19539.615767 ms`. The accepted `524288`-token hot-path rerun
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-memory-canonical-rerun.json`
stayed in band at `5987.142 tokens/sec`, p95 `21.546300 ms`,
`train_compute=0.134307 ms/token`, bounded `12/65536` route rows, no observed
contention, GPU memory `1827->1825 MiB`, and zero graph/native sequence
failures. The older full-chain benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-thought-structural-chain.json`
preserved hash, review-match, total-count, and current-pointer parity across the
expanded seventeen-component autonomous readout/language/thought chain while
checking `4352` target-family rows instead of `100096` broad-normalized rows
(`23x`) and reducing mean chain latency from `19704.406867 ms` to
`1046.241300 ms` (`18.833520x`). The clean `524288`-token rerun
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-thought-structural-chain-rerun.json`
stayed in band at `6005.229 tokens/sec`, with bounded `12/65536` route rows,
`10` output candidates, `65526` cached transition rows, zero graph/native
sequence failures, no observed contention, CUDA runtime on RTX 3060, and GPU
memory `1856->1857 MiB`. A first same-shape run succeeded at
`5921.867 tokens/sec` but is not primary evidence because the sampler observed GPU
contention. The replay/ledger benchmark kept archival/source/review metadata on
CPU, reported no live tick or every-token work, and did no hidden language
reasoning.

The 2026-06-22 readout-structural-plasticity cleanup removes the last active
thought-structural production vocabulary from the readout-ledger path. Production
uses `snn_language_readout_structural_plasticity_*` route/facade/ledger names,
and checkpoint load/save keeps only canonical readout-ledger fields while
dropping noncanonical state. The focused benchmark
`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-structural-canonical.json`
passed with bounded mean `568.337767 ms` versus legacy diagnostic
`7518.428000 ms` (`16x` source-work reduction), and the downstream
autonomous-chain bounded mean stayed `967.423500 ms` versus
`21131.022800 ms`. The paired hot-path run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json`
processed `524288` measured tokens at `5885.572 tokens/sec`, p95
`22.879800 ms`, `train_compute=0.136747 ms/token`, bounded `12/65536` route
rows, no observed contention, CPU max `88%`, GPU max `13%`, RTX memory
`1741->1970 MiB`, and zero graph/native sequence failures. Prewarm was explicit
slow-path setup at `418.781 s` and reached `full_warm_ready=true` before
measurement.

The 2026-06-22 readout-surface naming cleanup applies the same literature
boundary to active vocabulary: internal representational surfaces are useful
only as bounded readout evidence, not hidden replay text or a second thought
path. Production now uses `snn_language_readout_surface_*` route/facade/ledger
names. Checkpoint load/save keeps only canonical readout-ledger fields and
drops noncanonical readout-ledger state instead of maintaining old field
aliases. The focused benchmark
`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-surface-canonical.json`
passed with bounded mean `408.799567 ms` versus legacy diagnostic
`6288.893500 ms` (`16x` work reduction). The paired hot-path run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-surface-canonical.json`
processed `524288` tokens at `6012.300 tokens/sec`, p95 `21.322 ms`,
`train_compute=0.134205 ms/token`, bounded `12/65536` route rows, no observed
contention, CPU max `59%`, GPU max `13%`, RTX memory `1771->1772 MiB`, and zero
graph/native sequence failures.

Synapse provenance audit now uses a bounded requested-hash event map rather than
normalizing every readout/replay ledger family before checking runtime synapse
lineage. `synapse_provenance_audit(...)` gathers only hashes from
`synapse_provenance_by_key`, looks up `events` through
`bounded_snn_readout_evidence_event_map_source_window.v1`, and keeps archival
metadata and lookup on CPU. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-synapse-provenance-map.json`
preserved requested event-map hash parity while checking `128` rows instead of
`2944` broad-normalized rows (`23x`) and reducing mean event-map latency from
`319.823233 ms` to `13.972533 ms` (`22.889424x`). The clean `524288`-token
hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-synapse-provenance-map.json`
stayed in band at `5994.111 tokens/sec`, with bounded `12/65536` route rows,
`10` output candidates, `65526` cached transition rows, zero graph/native
sequence failures, no observed contention, CUDA runtime on RTX 3060, and GPU
memory `1980->1976 MiB`. The old broad-normalized audit event-map shape is
retired; benchmark-local broad comparison remains only evidence.

Emission review history now avoids broad ledger recall while still exposing the
bounded reviewed text needed for operator display. `emission_review_history(...)`
uses `bounded_snn_emission_review_history_source_window.v1` over
`emission_review_events` only, keeps archival/lookup metadata on CPU, and
reports no replay, no hidden language reasoning, no live-tick work, and no
every-token cadence. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-emission-history.json`
preserved review-hash and text-hash parity while checking `128` rows instead of
`2944` broad-normalized rows (`23x`) and reducing mean display-history latency
from `345.815600 ms` to `25.503433 ms` (`13.559570x`). The clean `524288`-token
hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-history.json`
stayed in band at `6051.817 tokens/sec`, with bounded `12/65536` route rows,
`10` output candidates, `65526` cached transition rows, zero graph/native
sequence failures, no observed contention, CUDA runtime on RTX 3060, and flat
GPU memory at `1972 MiB`. The old broad-normalized display-history shape is
retired; benchmark-local broad comparison remains only evidence.

Replay-dataset preview/export now follows the same selected slow-window rule.
Modern Hopfield association is treated as a local operator after source
selection; complementary learning systems, continual-learning replay, and
synaptic tagging/capture do not justify traversing every retained trace or
replay sample whenever an operator asks for a dataset preview. The maintained
path emits `bounded_replay_dataset_preview_source_window.v1` over at most
`50/64` retained runtime episode traces and
`bounded_replay_dataset_sample_link_source_window.v1` over at most `64/256`
retained replay sample records, with `16` stored sanitized candidates per
sample. Archival/source/link work stays on CPU, CUDA archival metadata is
absent, and the reports state no live tick, every-token work, replay text,
language reasoning, mutation, plasticity, or adapter training.

The same slice removes the old count/payload mismatch where the preview could
claim `count=50` while the generic sanitizer returned only `16` items. The
export sanitizer now returns the declared bounded item window while replay
sample normalization keeps the existing `16`-candidate storage budget. The
15-run report
`reports/bounded_replay_window_20260620/replay-dataset-source-window.json`
preserved all `50` selected target IDs and replay-link coverage against a
diagnostic full-retained walk, reduced replay-sample and selected-candidate
source work by `4x`, and recorded `2006.587280 ms` mean preview latency,
`2.842 MiB` traced Python peak, and CUDA available but unused. The clean
`524288`-token protection run reached `5923.269 tokens/sec`,
`tick_duration_ms.p95=22.446`, `train_compute=0.136941 ms/token`, bounded
`12/65536` route rows, no observed contention, GPU memory `3554->3541 MiB`,
and zero graph/native sequence failures. The near-two-second preview cost is
explicit evidence that this path remains operator/export slow-path only.

The replay-dataset candidate endpoint is now retired instead of adding another
candidate report shape. `/terminus/replay-plan` remains the public candidate
source of truth, and dataset preview/bundle carry candidate context only through
the bounded preview source window. `bounded_replay_dataset_preview_source_window.v1`
now names the retired `/terminus/replay-dataset/candidates` endpoint and the
replacement `/terminus/replay-plan`. The refreshed report
`reports/bounded_replay_window_20260622/replay-dataset-candidates-retired-source-window.json`
passed with `50/50` selected target/link parity, CPU archival placement, no
live-tick/every-token work, no replay text or hidden language reasoning, and
CUDA available but unused. The paired `524288`-token protection run stayed in
band at `6131.415 tokens/sec` with no observed contention and zero graph/native
sequence failures.

## Readout Capacity Naming Retirement

The same research boundary now applies to readout-capacity mutation naming.
Expandable-SNN and neurogenesis literature support separating sparse topology
evidence from checkpoint-backed tensor capacity relayout, but they do not
justify keeping a second thought/generic production surface for the same resize
transaction. MARULHO therefore retires `thought_capacity` state keys, generic
`snn-language-capacity-mutation-*` routes, and
`snn_language_capacity_mutation_*` request fields. The maintained path is
`snn_language_readout_capacity_mutation_*` through API schema, facade, ledger,
checkpoint-backed executor, runtime snapshot, and event review.

Focused verification kept the path checkpoint-backed and single-route:
`tests/test_snn_language_plasticity_executor.py -k readout_capacity_mutation`
passed `2` tests, `tests/test_service_manager.py -k readout_capacity_growth`
passed `1` checkpoint-backed manager test, the API canonical-route slice
passed `2` tests, and the long ledger chain
`test_readout_ledger_autonomous_confidence_use_preflight_audits_candidates_without_execution`
passed after structural-to-capacity-to-newborn handoff used the canonical
readout-capacity event review. A stale-name scan found old capacity spellings
only in retired-path documentation.

The accepted clean `524288`-token hot-path protection rerun
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-capacity-canonical-noprofile-rerun.json`
processed `5826.031 tokens/sec`, p95 tick `22.668 ms`,
`train_compute=0.137453 ms/token`, `prepare_training=0.007546 ms/token`, and
`finalize_total=0.007064 ms/token`. Runtime Truth kept route scoring bounded at
`12/65536` input rows and `10` output candidates, cached `65526` transition
rows, kept `state_transition_runs_all_columns=false`, selected CUDA on the RTX
3060, observed no contention, moved RTX memory `1798->1796 MiB`, and recorded
zero graph/native/sequence failures. Prewarm took `344.672 s`. The earlier
trainer-stage-profiler run completed at `5555.868 tokens/sec` and is kept only
as diagnostic evidence, not the accepted throughput gate.

## Readout Newborn Developmental Naming Retirement

The same local-plasticity and adult-neurogenesis evidence applies to the
downstream newborn developmental chain: newborn capacity is not useful until
bounded live activity, checkpoint-backed integration, critical-period learning,
maturation review, and pruning evidence agree. MARULHO therefore retires the
`thought_newborn` and `autonomous_snn_language_thought_newborn_*` production
names instead of keeping a hidden-thought bridge beside the readout path.

The maintained path is now `snn_language_readout_newborn_*` through API schema,
facade, ledger, executor, developmental autonomy, runtime snapshot fields, and
checkpoint persistence. The API route family is
`snn-language-readout-newborn-*`, and the API mapper no longer translates
readout-newborn payloads back to thought-era internals.

Focused verification kept the path checkpoint-backed and bounded:
`tests/test_snn_language_plasticity_executor.py -k "readout_capacity_mutation or readout_newborn"`
passed `8` tests, `tests/test_snn_language_readout_ledger.py -k "newborn or readout_capacity"`
passed `4` tests, the API canonical/vocabulary slice passed `2` tests, the
manager checkpoint-backed readout-capacity-growth chain passed, and
developmental-autonomy plus runtime-persistence tests passed `17` tests. The
standing replay gates also passed: bounded replay selection `5` tests and
checkpoint/replay reload `4` tests.

The replay quality report
`reports/bounded_replay_window_20260622/synthetic-readout-newborn-canonical.json`
kept selection on `bucket_indexed_candidate_window`, ran `0` global fallback
cycles, blocked zero-pressure/no-anchor controls, and kept positive-pressure
sleep recall bounded to `4` queries with mean best input-pattern distance near
zero. The paired hot-path report
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-newborn-canonical.json`
processed `524288` tokens at `5783.832 tokens/sec`, p95 tick `23.205 ms`,
bounded `12/65536` route rows, cached `65526` transition rows, selected CUDA on
the RTX 3060, observed no contention, moved RTX memory `1915->1913 MiB`, and
recorded zero graph/native/sequence failures.

The replay-dataset history wrapper is also retired. It returned
`/terminus/replay-sample/history` records under dataset naming, so it preserved
a duplicate replay-history surface without adding a selected dataset window.
Service benchmarks now time only replay plan, replay-sample history, trace
export, dataset preview, and dataset bundle as replay/export slow paths. The
report
`reports/bounded_replay_window_20260622/service-benchmark-replay-dataset-history-retired.json`
shows `replay_dataset_history` and `replay_dataset_history_summary` absent,
slow-path endpoint count `5`, hot-path budget passing, and preview/bundle
summaries intact. Replay-plan, runtime-trace export, and replay-dataset preview
response models now expose their computed `source_window` reports, making the
bounded source evidence public instead of dropping it at the schema boundary.

The same selected-window rule now removes the non-reversible anchor-source
fallback. `replay_anchor_window.py` no longer materializes `list(anchors)` when
an anchor source lacks reverse-recency iteration; it fails closed with
`fallback_reason=non_reversible_anchor_bucket_source`,
`anchor_bucket_source_read_count=0`, and
`anchor_bucket_source_materialized_count=0`. The benchmark
`reports/bounded_replay_window_20260622/sleep-replay-anchor-nonreversible-fallback-retired.json`
passed over `8192` anchors with `16` bounded reads, `0` materialized entries,
`1.0` newest-anchor hit rate, CPU archival/source placement, and `0.0 MiB`
CUDA delta. The paired long run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-dataset-history-anchor-retired.json`
stayed in band at `6151.826 tokens/sec`, bounded `12/65536` route rows,
`65526` cached transition rows, and zero graph/native sequence failures.

The runtime trace export and replay-sample summary follow-up applies the same
rule to the remaining status/export read path. Trace export now reports
`bounded_runtime_trace_export_source_window.v1` before endpoint filtering or
trace-state lookup, replay-sample summary reports
`bounded_replay_sample_summary_source_window.v1`, living status reads bounded
recent traces, feedback summary reads bounded trace/action windows, and the
generic sanitizer uses bounded iteration instead of materializing then
trimming. The report
`reports/bounded_replay_window_20260620/replay-dataset-runtime-trace-export-summary-source-window.json`
passed with `50/64` trace-export records, `64/256` replay-sample summary
records, `64/256` replay-sample link records, and `1024/4096`
selected-candidate link records. Selected target IDs and trace-export IDs both
matched the diagnostic bounded window (`50/50`). The reports keep archival and
summary work CPU-resident and state no live tick, no every-token cadence, no
hidden replay-text language reasoning, no mutation/plasticity/training, and no
GPU-resident archival metadata. The accepted `524288`-token protection rerun
kept the live tick in the same 6k-ish band at `6047.311 tokens/sec`, with
bounded `12/65536` route rows, `state_transition_runs_all_columns=false`, no
observed contention, flat RTX 3060 memory at `1911 MiB`, and zero graph/native
sequence failures.

SNN readout-ledger service snapshots now follow the same selected-source rule.
The old snapshot path called `_normalized_state()`, which normalized every
retained readout-ledger event family, before returning only the requested
display rows. That was control-plane work rather than live replay, but it left
a broad source path beside the bounded replay/readout operators. The active
`snapshot(...)` path emits
`bounded_snn_readout_ledger_snapshot_source_window.v1`, reads only the snapshot
event families it returns, caps each family at the requested snapshot limit and
retention limit, keeps archival/source/snapshot metadata on CPU, and reports no
global candidate/score scan, no live tick, no every-token cadence, no hidden
language reasoning, and no CUDA archive. The old all-family normalization model
is now removed from the snapshot benchmark too; all-family comparison evidence
must stay external or in explicitly normalizer-scoped benchmark evidence.

The focused report
`reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window.json`
used `2048` rows per retained ledger family, `ledger_limit=128`, and
`snapshot_limit=20`. The bounded snapshot read `260` rows instead of `2944`,
preserved newest-first returned rows and retained-count parity, and reduced mean
latency from `393.040600 ms` to `67.334088 ms` (`5.837171x`) with
`0.575356 MiB` traced Python peak and `0.0 MiB` CUDA allocation/reservation.
The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\snn-readout-ledger-snapshot-comparator-removed.json`
deletes the executable comparator, verifies returned-field-only source reads
directly, reads `260` bounded CPU rows for `13` returned snapshot fields,
projects `2684` removed all-family rows from `23` retained ledger fields,
averages `77.561856 ms`, traces `0.581556 MiB` Python peak, and keeps CUDA
allocation/reservation at `0.0 MiB`.
The current matching long protection run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-readout-ledger-snapshot-comparator-removed-default-nosample.json`
processed `524288` tokens at `6069.794524 tokens/sec`,
`train_compute=0.132884 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, and zero graph/native sequence failures. Boundary
environment samples marked `contention_observed` (`cpu max=64%`,
`gpu max=20%`, GPU memory utilization max `16%`), so this is same-band
protection evidence rather than a speed ceiling. RTX 3060 memory stayed flat at
`2191 MiB`.

Applied-synapse provenance status now follows the same selected-source rule.
Modern Hopfield-style recall is useful only after the memory/replay window is
selected; complementary learning systems, continual replay, and synaptic
tagging/capture do not justify checking all applied synapse provenance whenever
operator status is projected. The maintained status evidence emits
`bounded_snn_status_applied_synapse_provenance_source_window.v1`, reads at most
`32` sparse-weight keys and `32` provenance rows, keeps archival metadata and
lookup on CPU, and reports no live tick, every-token work, replay execution,
raw text, or hidden language reasoning. If retained rows exceed the source
window, exact audit readiness is blocked with
`integrity_count_scope=bounded_source_window`.

The report
`reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json`
read `64` rows instead of `4096` in the benchmark-local retired broad scan
model, reduced mean latency from `66.313336 ms` to `3.242332 ms`
(`20.452358x`), preserved bounded-window replay-regeneration provenance health,
and used `0.0 MiB` CUDA allocation/reservation. The first long protection run
succeeded at `5875.245 tokens/sec` but is kept as secondary variance; the
accepted rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-status-applied-synapse-provenance-source-window-rerun.json`
stayed in band at `6350.288 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, zero graph/native sequence failures, no
observed contention, and flat RTX 3060 memory at `1936 MiB`. The old broad
status key scan is retired; benchmark-local broad comparison remains only
evidence.

Applied-synapse provenance audit now uses a bounded local source window before
the associative ledger lookup as well. This matches the research boundary:
modern Hopfield-style recall is a local associative operator after selection,
complementary learning systems separate online state from slow consolidation,
continual-learning replay and synaptic tagging/capture prioritize selected
salient traces, and sparse replay keeps the candidate set small. The maintained
`synapse_provenance_audit(...)` path emits
`bounded_snn_readout_synapse_provenance_audit_source_window.v1`, reads at most
`64` applied sparse-weight/provenance rows from CPU archival state, asks the
ledger only for hashes in that source window, and blocks exact audit review
when the source window is truncated. It reports no global candidate or score
scan, no raw replay text, no hidden language reasoning, no live tick, no
every-token work, no mutation/plasticity, and no GPU-resident archival
metadata.

The report
`reports/bounded_replay_window_20260620/synapse-provenance-audit-source-window.json`
matched the diagnostic first source window, requested only `64` ledger hashes,
read `64` bounded source rows instead of `4096` diagnostic records and `2048`
materialized rows (`32x` less source work by report metric), and reduced mean
audit latency from `259.221928 ms` to `75.262088 ms` with `1.909667 MiB`
traced Python peak allocation and no GPU production-audit use. The accepted
hot-path rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-source-window-rerun.json`
stayed in band at `6441.166 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, zero graph/native sequence failures, no
observed contention, and flat RTX 3060 memory at `1866 MiB`. The old full
applied-synapse audit scan is retired from production.

The maintained-only refresh
`..\..\MARULHO_reports\bounded_replay_window_20260624\synapse-provenance-audit-comparator-removed.json`
removes the executable full-scan comparator from the benchmark too. Quality is
now `seeded_bounded_applied_synapse_audit_source_window_reconstruction`: the
bounded audit reconstructs the expected seeded source keys directly, reads
`128` bounded CPU source rows over `2048` retained rows, projects `3968`
removed full-scan rows, records `full_scan_comparator_removed=true`, averages
`73.058288 ms`, and keeps CUDA allocation/reservation at `0.0 MiB`. The paired
`524288`-token protection run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-synapse-provenance-audit-comparator-removed-default-nosample.json`
stayed in the 6k-ish band at `6101.308 tokens/sec`, with `last_tick_duration_ms=17.730`,
bounded `12/65536` route rows, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, native sequence-loop and burst-replay
failure counts `0`, no observed before/after contention (`cpu max=42%`,
`gpu max=10%`), and RTX 3060 memory `2049->2050 MiB`. Broad comparison no
longer exists as repo-local executable benchmark code.

Transition-memory status projection now applies the same selected-source rule
to the broader status family. Modern Hopfield-style associative recall belongs
inside a selected local memory or replay window; CLS, continual replay,
synaptic tagging/capture, and sparse replay support salient bounded evidence,
not routine archive-wide integrity checks in operator status. MARULHO therefore
routes capacity pressure, dense readout tensor integrity, applied-synapse
provenance, and rollout/server binding through one bounded newest-first
transition-memory helper. The status-local insertion-order helper is removed,
so status and plasticity runtime-state projections share one source-window
rule. Each status projection reads at most `32` sparse-transition rows and
`32` provenance rows on CPU and blocks exact readiness when truncated.

The report
`reports/bounded_replay_window_20260620/status-transition-memory-source-window.json`
used `2048` retained transition/provenance rows. The maintained path read
`256` bounded rows across four projections instead of `10240` retired repeated
broad rows (`40x` less source work), reduced mean latency from `89.558896 ms`
to `11.162376 ms` (`8.023282x`), preserved retained counts, reported no global
candidate/score scan, no live tick, no every-token cadence, no replay, no raw
text or hidden language reasoning, and used `0.0 MiB` CUDA
allocation/reservation. The old transition-memory status broad projection
family is retired from production; any exact integrity pass must be an explicit
slow audit/replay window with its own quality and throughput evidence. The
accepted `524288`-token rerun processed `6371.238 tokens/sec` with
`train_compute=0.128035 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, and zero graph/native sequence failures; velocity
still reported borderline GPU contention (`23%`), so the run is throughput
protection rather than a clean speed ceiling.

The 2026-06-23 refresh
`..\..\MARULHO_reports\bounded_replay_window_20260623\status-transition-memory-source-window-recent-helper.json`
proves the shared helper is recency-correct: `32` stale invalid rows inserted
first lose to `32` valid recent rows inserted last, with
`invalid_synapse_key_count=0` and `32 + 32` selected source rows. The maintained
path still reads `256` rows versus `10240` retired broad rows (`40x` less
source work), reduces mean latency from `87.635592 ms` to `12.310596 ms`
(`7.118712x`), keeps CPU archival placement with bounded Python peak
`0.047835 MiB` mean, and uses `0.0 MiB` CUDA allocation/reservation. The paired
external `524288`-token protection run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-status-transition-recent-helper-default-nosample.json`
stayed in the maintained 6k-ish band at `6054.480 tokens/sec`, p95
`21.719 ms`, `train_compute=0.133249 ms/token`, bounded `12/65536` route rows,
`65526` cached rows, no observed contention (`cpu max=45%`, `gpu max=10%`),
RTX 3060 memory `1817->1816 MiB`, and zero graph/native sequence failures.

## Links

- [Research notes](../../research-living-brain.md)
- [Column Runtime](../concepts/column-runtime.md)
- [Replay Cost](../benchmarks/replay-cost.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)

## Generic Replay Entry Retired

`DualMemoryStore.replay_entry(...)` is removed. Production replay/recall row
access is now named by purpose: `sleep_repair_replay_row(...)` for mutating
sleep repair, `replay_recall_row(...)` for read-only sleep recall, and
`query_match_row(...)` for query/source-bank/context recall. This keeps
modern-Hopfield-style recall as a local operator over selected traces, not as a
generic hidden replay-text or mutating row side path.

Fresh cleanup reports under
`..\..\MARULHO_reports\bounded_replay_window_20260622\` passed: query payload
parity loaded `5` bounded text payloads instead of `192`; context comparison
kept selected-index parity while loading `8` payloads instead of `16` and
reusing `8` query-row cache hits. The current repair report
`..\..\MARULHO_reports\bounded_replay_window_20260624\sleep-repair-replay-dense-prepare-comparator-removed.json`
improves mean anchor distance by `0.076463`, defers `8` missing-key rows, makes
`0` dense input-assembly calls, removes the executable dense-prepare comparator,
keeps archive metadata on CPU, uses CUDA only for active repair compute, and
reports no global scan, live tick, every-token work, raw replay text, or language
reasoning.

## Replay Sample Single Path

Replay/consolidation literature supports separating nomination, review, and
execution authority. MARULHO therefore removes the execution-shaped audit alias
instead of preserving it as a second replay path. The maintained operator review
surface is `POST /terminus/replay-sample` plus
`GET /terminus/replay-sample/history`; `/terminus/replay-execute`,
`mode="execute"`, `execution_id`, and `replay_executor_summary` are retired.

`reports/bounded_replay_window_20260620/replay-sample-single-path-service-benchmark.json`
passed with no `replay_executor_summary`, replay-sample history latency
`4.798 ms`, CPU summary placement, no raw replay text, no hidden language
reasoning, no live tick, no every-token cadence, and no
training/plasticity/action side effects.

`reports/bounded_replay_window_20260620/replay-dataset-source-window-replay-sample-single-path.json`
passed with canonical `sample` records, `50/50` target/link parity, `64/256`
bounded replay-sample summary rows, CPU archival/source placement, and no
GPU-resident archival metadata.

`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-sample-single-path.json`
processed `524288` tokens at `5951.781 tokens/sec`, p95 `21.962 ms`,
`train_compute=0.136320 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, and zero graph/native sequence
failures.

## Replay Restore Source Window

Checkpoint/reload replay-controller state now follows the same selected-source
rule. `ReplayController` emits `bounded_replay_restore_source_window.v1` and
loads only the newest controller-retained source window for replay sample
history, regeneration permits, replay-evaluation contexts, review tickets,
scheduler installations, and transition-memory replay artifacts before
normalization or index rebuild. `MarulhoServiceManager` and
`RuntimePersistence` no longer add full `list(...)` copies around those replay
fields during restore.

The focused report
`reports/bounded_replay_window_20260620/replay-restore-source-window.json`
used `65536` records in each replay restore field. It matched the
benchmark-local retired full-materialized restore model for the latest window,
restored `64` valid evaluated artifacts, inspected `656` records instead of
`524288` (`799.219512x` less source work), and reduced mean restore latency
from `6605.339529 ms` to `15.600729 ms` (`423.399426x`). Placement stayed on
CPU; CUDA was available but unused with `0.0 MiB` allocation/reservation, Python
traced peak was `0.581783 MiB`, and the Runtime Truth report states no live
tick, no every-token work, no raw replay text, no hidden language reasoning, no
mutation/plasticity, and no GPU-resident archival metadata.

The accepted protection rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json`
processed `524288` tokens at `5945.577 tokens/sec`, p95 `22.062 ms`,
`train_compute=0.136201 ms/token`, `prepare_training=0.007152 ms/token`, and
`finalize_total=0.006825 ms/token`. Runtime Truth kept route scoring at
`12/65536` input rows and `10` output candidates, cached `65526` transition
rows, kept `state_transition_runs_all_columns=false`, selected CUDA on RTX
3060, and recorded zero graph/native sequence failures. Velocity reported no
observed contention, CPU max `30%`, GPU max `13%`, and RTX memory
`2061->2062 MiB`; the first same-shape run is secondary because GPU contention
was observed at `22%`.

## Applied Replay Lineage Checkpoint Summary

Replay-backed structural growth now keeps checkpoint lineage validation on a
mutation-maintained CPU summary instead of deriving it by scanning all
`synapse_provenance_by_key` rows during checkpoint save or restore. Replay
regeneration records one applied-lineage row hash per replay-regenerated
synapse; non-replay overwrites and pruning clear the row for that synapse.
`RuntimePersistence` reads
`snn_applied_replay_lineage_incremental_summary.v1` to publish
`snn_applied_replay_lineage_checkpoint_summary.v1`, and restore validation
compares saved/restored counts and digests. If the incremental summary is
missing, exact validation is blocked instead of falling back to a full
provenance rebuild or preserving a legacy-source compatibility field.

This follows the replay/consolidation research boundary: modern-Hopfield-like
association and continual replay are useful only inside selected local windows;
CLS and synaptic tagging/capture argue for durable tags that survive slow-path
consolidation, but not for replay-lineage scans inside checkpoint or live tick
work. Archival lineage metadata stays CPU-resident and carries hashes only, not
raw replay text, operator identity, or hidden language reasoning.

The focused benchmark
`reports/bounded_replay_window_20260620/applied-replay-lineage-checkpoint-summary.json`
passed on `65536` replay-lineage rows. The active checkpoint summary read `0`
provenance source records, matched the benchmark-local retired full-scan
diagnostic counts and digest, averaged `0.065529 ms`, and used `0.001343 MiB`
Python traced peak. The retired diagnostic read `196608` source records,
averaged `6766.639043 ms`, and used `24.036118 MiB` traced peak. CUDA was
available but unused with `0.0 MiB` allocated/reserved.

The 2026-06-23 maintained-only cleanup
`..\..\MARULHO_reports\bounded_replay_window_20260623\applied-replay-lineage-checkpoint-legacy-baseline-removed.json`
removes the executable benchmark-local full-provenance scan instead of keeping
it beside the maintained path. It passes by matching the seeded
mutation-maintained incremental summary for `65536` replay-lineage rows, reads
`0` provenance source records, averages `0.082714 ms`, uses `0.001343 MiB`
Python peak, keeps CPU archival placement with `0.0 MiB` CUDA allocation, and
reports `retired_full_scan_absence.implementation_present=false`.

The paired hot-path run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-applied-replay-lineage-checkpoint-summary.json`
processed `524288` tokens in `87.483241 s` at `5993.011 tokens/sec`, p95
`21.608 ms`, `train_compute=0.135253 ms/token`,
`prepare_training=0.007078 ms/token`, and
`finalize_total=0.006754 ms/token`. Runtime Truth kept route scoring bounded
at `12/65536` input rows and `10` output candidates, cached `65526`
transition rows, kept `state_transition_runs_all_columns=false`, selected CUDA
on the RTX 3060, and recorded zero graph/native sequence failures. Prewarm took
`335.271 s`; velocity reported no observed contention, CPU max `15%`, GPU max
`13%`, GPU memory utilization max `18%`, and RTX memory `2082->2084 MiB`.

The cleanup hot-path runs
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-applied-lineage-legacy-baseline-removed-default-nosample.json`
and
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-applied-lineage-legacy-baseline-removed-default-nosample-rerun.json`
succeeded without observed contention at `5744.182` and
`5790.952 tokens/sec`, with bounded `12/65536` route rows, `65526` cached rows,
no all-column transition, RTX memory `1816->1814` and `1813->1814 MiB`, and
zero graph/native sequence failures. This slice changes only evaluation/tests
and docs, so the lower readings are retained as variance evidence while the
production checkpoint/restore lineage path remains the mutation-maintained CPU
summary.

## Plasticity Runtime-State Source Window

SNN language plasticity runtime-state now uses the same selected-source rule
before exposing retained transition memory. Modern Hopfield-style association
is useful here only after a local memory has been selected; complementary
learning systems, continual replay, synaptic tagging/capture, and sparse replay
support slow selected consolidation, not full retained transition-memory export
from a status endpoint.

`SNNLanguagePlasticityApplicationExecutor.snapshot()` emits
`bounded_snn_language_plasticity_runtime_transition_memory_source_window.v1`,
returns at most `64` sparse-transition rows and `64` synapse-provenance rows,
keeps retained counts in the source-window metadata, and marks exact integrity
incomplete when truncated. Status and readout-ledger consumers read those
retained counts instead of treating bounded maps as complete. The old full
runtime-state deep copy is retired from production and the executable
benchmark-local comparator is now removed from
`plasticity_runtime_state_source_window_benchmark.py`.

The focused benchmark
`reports/bounded_replay_window_20260621/plasticity-runtime-state-source-window.json`
passed over `65536` sparse weights and `65536` provenance rows. The active path
read `256` source records versus `262144` in the retired diagnostic (`1024x`
less source work), averaged `7.770271 ms` versus `752.314014 ms`
(`96.819528x`), used `0.110454 MiB` traced Python peak versus `12.287186 MiB`,
and kept CUDA allocation/reservation deltas at `0`. The maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\plasticity-runtime-state-full-snapshot-comparator-removed.json`
now validates the same maintained path without executing the retired
comparator: it verifies recent sparse/provenance source-window selection
directly, reads `256` bounded CPU source rows for `65536` retained transition
rows, projects `261888` removed full-snapshot rows, averages `9.137240 ms`
with p95 `13.097 ms`, traces `0.109653 MiB` Python peak, and keeps CUDA
archive allocation/reservation at `0`. Runtime Truth reports CPU
archival/source/lookup placement, no global scan, no replay, no raw text, no
hidden language reasoning, no live tick, no every-token cadence, no
mutation/plasticity, no GPU-resident archival metadata, and no repo-local
executable side comparator.

The current paired hot-path gate
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-plasticity-runtime-state-full-snapshot-comparator-removed-default-nosample.json`
processed `524288` tokens at `6123.799 tokens/sec`, p95 `20.956 ms`,
`train_compute=0.131898 ms/token`, `prepare_training=0.006535 ms/token`,
`finalize_total=0.006678 ms/token`, and prewarm `247.843 s`. Runtime Truth kept
route scoring at `12/65536`, cached `65526` transition rows, kept
`state_transition_runs_all_columns=false`, observed no contention (`cpu max=19%`,
`gpu max=10%`), held RTX 3060 memory flat at `2190 MiB`, and recorded zero
native sequence-loop or burst-replay failures.

The paired `524288`-token protection runs succeeded with no observed
contention, bounded `12/65536` route scoring, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures. They measured `5642.888` and `5736.332 tokens/sec`, so they protect
the live tick from obvious runtime-state source-window tax but do not close the
durable completion condition or establish a new speed ceiling.

## Sleep Replay Associative Recall

Modern Hopfield networks support associative recall as a local memory operator,
not as a license for global memory scans. Complementary learning systems,
continual-learning replay, synaptic tagging/capture, latent replay, and sparse
replay all point toward selected slow-window rehearsal with explicit quality
and cost gates. MARULHO applies that boundary inside trainer-owned deep sleep:
after bounded anchor/replay-window selection, `bounded_sleep_replay_associative_recall.v1`
uses at most `4` selected replay entries as queries, reads replay tensors
through `bounded_replay_recall_row.v1`, and recalls only within the selected
bucket-indexed candidate window with `select_replay_window(...,
advance_stc_state=false)`. Runtime Truth reports CPU archival/source/score
placement, no live tick, no every-token cadence, no raw replay text, no hidden
language reasoning, no mutation authority, no STC state advance, and no
plasticity authority.

The focused benchmark
`reports/bounded_replay_window_20260622/sleep-replay-associative-recall-window.json`
passed the new sleep-recall gate for the positive-pressure arm with `4` bounded
queries and mean best input-pattern distance `5.96046447753906e-08`.
Zero-pressure and no-anchor controls ran `0` queries and made no quality claim.
The same report does not claim prototype-repair improvement; prototype repair
remains guarded separately by reconstruction evidence.

The same slice closed a reopened source-tick replay hole in the delegated
training sequence path. `BrainRuntime` passes `allow_sleep_maintenance=false`
to `train_text_sequence(...)`, and sequence fallback forwards that gate into
per-token `train_step(...)`. The deferral benchmark
`reports/bounded_replay_window_20260622/source-tick-sequence-sleep-deferral.json`
passed with service fallback sleep calls `0`, sequence fallback sleep calls `0`,
explicit slow-path sleep calls `1`, and visible sequence fallback deferred
counts. The paired `524288`-token protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-sleep-replay-associative-recall-source-sequence-deferral.json`
processed `6487.329 tokens/sec`, `train_compute=0.125633 ms/token`,
`prepare_training=0.005922 ms/token`, and
`finalize_total=0.005852 ms/token`, with bounded `12/65536` route scoring,
`65526` cached transition rows, zero graph/native sequence failures, no
observed contention, CPU max `12%`, GPU max `19%`, and RTX 3060 memory
`1709->1707 MiB`. Prewarm took `304.955 s`, so this is hot-tick protection and
bounded recall quality evidence, not a startup-speed claim.

The read-only row follow-up retires `DualMemoryStore.replay_entry(...)` as the
query reader for associative recall, and the generic API is now removed. Focused
tests prove the recall path does not call a mutating repair row and does not
advance `_state_token`, capture tags, local PRP, global PRP, or bucket PRP. The
external local replay report
`..\..\MARULHO_reports\bounded_replay_window_20260622\read-only-recall-row-telemetry-retired.json`
passed the positive-pressure arm with `1` query, mean best input-pattern
distance `5.96046447753906e-08`, `query_row_state_advance_count=0`,
`recall_selection_state_advance_count=0`, `read_only_replay_row=true`,
`recall_selection_read_only=true`,
`query_row_reader=DualMemoryStore.replay_recall_row`, and
`mutates_runtime_state=false`. HF replay-query collection now uses that same
store-owned row reader after bounded anchor-bucket selection and reports
`direct_slow_memory_input_pattern_reads_retired=true`; the smoke report
`..\..\MARULHO_reports\bounded_replay_window_20260622\hf-query-row-reader-retired\summary.json`
kept the memory-consolidation gate passing while leaving bounded recall quality
unpromoted. The paired no-profile `524288`-token protection runs stayed
same-band at `5872.559` and `5943.110 tokens/sec`, and the telemetry-retirement
rerun stayed in the current noisy band at `5819.770 tokens/sec` with
`train_compute=0.137724 ms/token`, bounded `12/65536` route scoring,
`524288` skipped graph consolidation lookups, CUDA selected on the RTX 3060,
zero native sequence-loop fallback, and RTX memory flat at `1934 MiB`. This
changes recall query/selection reads only; mutating replay/consolidation remains
explicit.

## Readout Consolidation Naming Retirement

Modern Hopfield-style recall and replay research supports local associative
operators only after a bounded source family or replay window has been selected.
Complementary learning systems, continual replay, synaptic tagging/capture, and
sparse replay do not justify keeping hidden-thought vocabulary as a second
production path. MARULHO therefore retires the active
`autonomous_snn_language_thought_consolidation_*` API/facade/ledger chain and
keeps readout consolidation on the canonical
`snn_language_readout_consolidation_*` path. Checkpoint load/save keeps
canonical readout-ledger fields and drops noncanonical readout-ledger state
instead of maintaining old field aliases.

The focused benchmark
`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-consolidation-canonical.json`

### Readout-Ledger Benchmark Comparator Removal

Modern Hopfield-style recall remains a local associative operator only after a
bounded source family or replay window has been selected. Complementary learning
systems, continual-learning replay, synaptic tagging/capture, and sparse replay
also argue against keeping executable archive-wide comparators around as
side-path evidence once the maintained path has its own quality gates.

The current readout-ledger normalization benchmark therefore removes the old
full-materialized normalization/store comparators and the broad-normalized
per-boundary comparators. The maintained report
`..\..\MARULHO_reports\bounded_replay_window_20260624\snn-readout-ledger-normalization-comparators-removed.json`
uses seeded newest-first reconstruction as the quality target, records absence
for both deleted comparator families, and keeps CPU archival/normalization/store
/lookup/evaluation placement with no live tick, no every-token cadence, no
global candidate/score scan, no hidden language reasoning, and `0.0 MiB` CUDA
allocation/reservation. It passed with `2944` bounded rows out of `47104`
source rows and `44160` full-materialized rows removed. The paired long run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-readout-ledger-normalization-comparators-removed-default-nosample.json`
processed `524288` tokens at `6507.349 tokens/sec`, p95 `19.722 ms`,
`train_compute=0.124647 ms/token`, route scoring `12/65536`, cached `65526`
transition rows, no observed contention, flat RTX memory, and zero graph/native
sequence failures.

Implication: rejected recall/replay comparators may be described in retired
docs, but repo-local executable code should expose only the maintained bounded
path unless a new architecture decision reopens the trade-off with stronger
quality and throughput evidence.
passed with bounded mean `371.891600 ms` versus the benchmark-local retired
diagnostic `5302.309467 ms` (`16x` source-work reduction). The downstream
autonomous-chain comparison stayed bounded at `921.487733 ms` versus
`19456.105067 ms`. Archival/source/review placement stays CPU-resident, CUDA
was available but unused for ledger archival work, and the path reports no
live tick, every-token work, replay execution, raw replay text, hidden language
reasoning, or GPU-resident archival metadata.

The `524288`-token hot-path protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-consolidation-canonical.json`
stayed same-band at `5938.794 tokens/sec`, p95 tick `22.043 ms`,
`train_compute=0.135649 ms/token`, `prepare_training=0.007072 ms/token`, and
`finalize_total=0.006942 ms/token`. Runtime Truth kept route scoring bounded at
`12/65536` input rows and `10` output candidates, cached `65526` transition
rows, kept `state_transition_runs_all_columns=false`, selected CUDA on the RTX
3060, observed no contention, moved GPU memory `1829->1828 MiB`, and recorded
zero graph/native/sequence failures. Prewarm took `316.785 s`, so this is
live-tick protection evidence, not startup-speed evidence.

## Readout Structural Plasticity Naming Retirement

The same research boundary now applies to readout-structural plasticity naming:
local growth/prune evidence belongs behind bounded source windows and explicit
review gates, not hidden-thought production vocabulary. MARULHO therefore
retires the active `autonomous_snn_language_thought_structural_plasticity_*`
API/facade/ledger chain and keeps structural evidence on
`snn_language_readout_structural_plasticity_*`. Checkpoint load/save keeps only
canonical readout-ledger fields and drops noncanonical readout-ledger state.

The focused benchmark
`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-structural-canonical.json`
passed with bounded mean `568.337767 ms` versus the benchmark-local retired
diagnostic `7518.428000 ms` (`16x` source-work reduction). The downstream
autonomous-chain comparison stayed bounded at `967.423500 ms` versus
`21131.022800 ms`. Archival/source/review placement stays CPU-resident, CUDA
was available but unused for ledger archival work, and the path reports no live
tick, every-token work, replay execution, raw replay text, hidden language
reasoning, or GPU-resident archival metadata.

The `524288`-token hot-path protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json`
stayed same-band at `5885.572 tokens/sec`, p95 tick `22.879800 ms`,
`train_compute=0.136747 ms/token`, `prepare_training=0.007215 ms/token`, and
`finalize_total=0.006836 ms/token`. Runtime Truth kept route scoring bounded at
`12/65536` input rows and `10` output candidates, cached `65526` transition
rows, kept `state_transition_runs_all_columns=false`, selected CUDA on the RTX
3060, observed no contention, moved GPU memory `1741->1970 MiB`, and recorded
zero graph/native/sequence failures. Prewarm took `418.781 s`, so this is
live-tick protection evidence, not startup-speed evidence.

## Query Fallback Retirement And Bundle Source Windows

Modern Hopfield recall supports attention-like associative lookup only after a
local memory window has been selected. Complementary learning systems,
continual-learning replay, synaptic tagging/capture, latent replay, and sparse
replay point in the same engineering direction: select a bounded evidence set,
then run local recall or package selected replay records with visible source
authority. They do not support a query readout widening its routed bucket window
with an archive-recent text-support sweep.

MARULHO therefore retires the query recent-entry text-support fallback from
production. `query_runner.memory_matches_with_report(...)` keeps the
routing-owned `bucket_indexed_candidate_window`, omits `recent_fallback_*`
fields, and does not call `collect_recent_entry_indices(...)` to find query
terms outside the routed buckets. The focused report
`reports/bounded_replay_window_20260622/query-recent-fallback-retired-bucket-only.json`
passed with capacity `65536`, candidate indices `[0]`, returned indices `[0]`,
no recent collector calls, raw text loaded only for the candidate, no global
candidate/score scan, no live tick, no hidden language reasoning, CPU archival
placement, and no CUDA allocation/reservation.

The same boundary now survives replay-dataset packaging. The bundle response
carries `bounded_replay_dataset_bundle_source_window.v1` with the nested preview
window, source and excluded counts, CPU archival/source placement, no GPU
archival metadata, no live/every-token work, and no training/plasticity
authority. The report
`reports/bounded_replay_window_20260622/replay-dataset-bundle-source-window-query-fallback-retired.json`
kept bundle source counts aligned with the preview (`50` source, `0` packaged,
`50` excluded), retained `4.0x` replay-sample and selected-candidate work
reductions, and measured bundle mean latency at `2183.314971 ms`. This is an
operator/export slow-path cost, not live tick work.

The paired `524288`-token protection run
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-fallback-retired-bundle-source-window.json`
stayed in the maintained band at `6117.434 tokens/sec`, p95 tick
`20.621 ms`, `train_compute=0.132414 ms/token`,
`prepare_training=0.006705 ms/token`, `finalize_total=0.006400 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, no observed contention, CPU max
`38%`, GPU max `13%`, RTX memory `1775->1775 MiB`, and zero graph/native
sequence failures.

## Query Row-Access Retirement

The same research boundary now applies inside explicit query recall after the
candidate window is chosen. Modern Hopfield recall remains a bounded local
operator; CLS, continual replay, synaptic tagging/capture, latent replay, and
sparse replay do not justify a query runner reading archival arrays directly
beside the memory-store API.

MARULHO therefore keeps candidate selection on
`bounded_query_memory_match_candidates.v1`, then reads scoring rows through
`bounded_query_memory_match_row.v1`. Raw text is opt-in and only fetched for
returned similarity matches unless text ranking explicitly requires candidate
payloads. Query-episode neighbor stitching uses
`bounded_query_neighbor_source_row.v1`. `query_runner.py` has no production
references to `slow_buffer`, `slow_raw_windows`, routing keys, input patterns,
bucket ids, timestamps, importance, or replay counts.

The focused report
`reports/bounded_replay_window_20260622/query-memory-store-owned-row-access.json`
preserved selected-index parity with the diagnostic eager payload path
(`[0, 16, 32, 48, 64]`), loaded `5` raw text payloads instead of `192`,
and reduced mean latency from `42.525 ms` to `33.718 ms` (`1.261x`). The query
report read `197` bounded rows (`192` scoring rows plus `5` text payload rows),
kept archival/score placement on CPU, and reported no global scan, no live
tick, no every-token work, no mutation/plasticity, and no hidden language
reasoning.

The paired replay quality report
`reports/bounded_replay_window_20260622/synthetic-query-row-access.json` kept
sleep associative recall bounded to `4` selected-window queries with mean best
input-pattern distance `5.960464477539063e-08`. The no-profile hot-path rerun
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-row-access-noprofile-rerun.json`
processed `524288` tokens at `5935.802 tokens/sec`, p95 tick `21.734 ms`,
`train_compute=0.135751 ms/token`, `prepare_training=0.007080 ms/token`, and
`finalize_total=0.006814 ms/token`, with bounded `12/65536` route rows,
`65526` cached transition rows, CUDA selected on the RTX 3060, RTX memory
`1952->1954 MiB`, and zero graph/native/sequence failures. Velocity observed
GPU-side contention during the run, so this is same-band protection and
retirement evidence, not a clean speed ceiling.

## Semantic Frontier Store-Owned Row Boundary

The same research boundary now applies to source-bank semantic recall,
frontier-gap planning, and concept-frontier metrics. Modern Hopfield-style
recall can score a bounded local window, while complementary learning systems,
continual replay, synaptic tagging/capture, latent replay, and sparse replay
argue for selected replay and consolidation windows. They do not justify
semantic planners reading archive arrays directly or using mutating replay
entry readers to fetch text after selection.

MARULHO keeps semantic frontier row access on
`DualMemoryStore.query_match_row(...)` under
`bounded_query_memory_match_row.v1`. `bank_memory_matches_with_report(...)`,
`frontier_gap_plan(...)`, and `concept_frontier_metrics_with_report(...)` now
read scoring/capture/consolidation/text rows through that store-owned surface.
Production `_effective_capture_strength(...)` is removed, and semantic
frontier code has no production `replay_entry(...)` reader or direct `slow_*`
archive row reads.

The external source-bank report
`..\..\MARULHO_reports\bounded_replay_window_20260622\source-bank-store-owned-row-reader.json`
passed on a `65536`-entry store with selected-index parity `1.0`, `196`
store-owned row reads (`192` scoring rows plus `4` text rows), raw text loaded
only for returned rows through `query_match_row(...)`,
`stc_state_advance=false`, CPU archival/score placement, `0.0 MiB` CUDA
allocation/reservation, and mean latency `160.781 ms` versus `958.681 ms` for
the diagnostic path.

The frontier-gap report
`..\..\MARULHO_reports\bounded_replay_window_20260622\frontier-gap-store-owned-row-reader.json`
kept term recall `1.0`, read `192/65536` rows, used no direct slow-memory row
reads, used no capture helper, and reduced mean latency from `229.118 ms` to
`8.897 ms` (`25.752x`). The concept-frontier report
`..\..\MARULHO_reports\bounded_replay_window_20260622\concept-frontier-store-owned-row-reader.json`
passed quality, bounded-scan, latency, and live-tick gates with `64` row reads
at `8192` capacity, `top1_match=true`, no direct slow-memory row reads, no
capture helper, no live tick, and CPU archival placement. The replay quality
check
`..\..\MARULHO_reports\bounded_replay_window_20260622\semantic-row-reader-replay-quality.json`
kept sleep recall passing with `1` query and best input-pattern distance
`5.96046447753906e-08`.

The follow-up maintained-only reports remove the benchmark-local legacy
comparators that still executed old paths after the bounded operators were
accepted. Concept signature lookup no longer carries an archive-materializing
signature helper, concept-frontier scope no longer carries a full slow-memory
metric helper, and frontier-gap planning no longer carries a full raw-window
term scanner. The reports
`..\..\MARULHO_reports\bounded_replay_window_20260623\concept-signature-legacy-baseline-removed.json`,
`concept-frontier-legacy-baseline-removed.json`, and
`frontier-gap-legacy-baseline-removed.json` passed seeded signature cosine min
`0.9999998212`, concept-frontier target hit rate `1.0`, and frontier-gap
expected term recall `1.0` with CPU archival placement, Python trace peaks
below `0.09 MiB`, and `0.0 MiB` CUDA allocation. The paired long run
`..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-semantic-frontier-legacy-baselines-removed-default-nosample.json`
kept throughput in band at `6496.154 tokens/sec` with bounded `12/65536` route
rows, cached `65526` transition rows, no observed contention, flat RTX memory
`1866 MiB`, and zero graph/native sequence failures. This keeps replay and
semantic recall evidence on the selected bounded path instead of preserving old
archive comparators as side implementations.

## Recent Anchor-Capture Row Boundary

Selected replay anchors now follow the same boundary after recent-window
selection. Synaptic tagging/capture and sparse replay support tagging useful
recent traces and replaying selected windows; they do not require trainer code
to read archive bucket arrays directly.

`capture_recent_memory_anchors(...)` now asks
`DualMemoryStore.recent_anchor_capture_row(...)` for selected bucket rows under
`bounded_recent_anchor_capture_row.v1`. The trainer reports
`anchor_row_reader_owned_by_store=true`,
`direct_slow_memory_bucket_reads_retired=true`, no raw replay text, no hidden
language reasoning, no live tick, and no every-token work.

The external report
`..\..\MARULHO_reports\bounded_replay_window_20260622\recent-anchor-capture-store-owned-row.json`
passed with `64` captured rows, `anchor_row_read_count=64`, zero invalid rows,
CPU archival placement, `0.0 MiB` CUDA allocation delta, and mean capture
latency `1.743 ms` with p95 `2.071 ms`. The paired hot-path report processed
`524288` tokens at `5916.223 tokens/sec`, p95 tick `21.992 ms`, bounded
`12/65536` route rows, no observed contention, and zero graph/native/sequence
failures.
## Source-Window Comparator Removal

Complementary learning systems, continual replay, synaptic tagging/capture,
latent replay, sparse replay, and modern Hopfield-style associative recall all
point to selected local windows rather than preserving full-source replay
comparators beside the active path. MARULHO now applies that rule to the
benchmark layer for hot-bucket source construction, selected-window SFA
sampling, and awake-ripple tagging.

The maintained-only reports
`..\..\MARULHO_reports\bounded_replay_window_20260624\bucket-candidate-source-window-comparator-removed.json`,
`sfa-sample-comparator-removed.json`, and
`awake-ripple-comparator-removed.json` pass newest-candidate hit rate `1.0`,
selected-window SFA purity `1.0`, and awake-ripple `0` scalar/vector scans over
`256` wake-bucket scans. The paired long run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-source-window-comparators-removed-default-nosample.json`
kept throughput in band at `6580.539 tokens/sec` with bounded `12/65536` route
rows, cached `65526` transition rows, no observed contention, flat RTX memory
`1875 MiB`, and zero graph/native sequence failures. This keeps replay
metabolism, SFA correction, and ripple tagging as selected source-window
operators instead of retaining full-memory benchmark side paths.
## Query Readout Comparator Removal

The same bounded-recall rule applies to the query/readout benchmark layer:
modern Hopfield-style association is useful only inside selected local evidence
windows, and CLS/continual replay/STC/sparse replay do not justify preserving
full or report-dropping readout comparators beside maintained reports. MARULHO
therefore removes benchmark-local report-dropping context readout, eager
candidate text payload loading, direct runtime concept archive lookup, and
fragment-only episode readout comparators.

The maintained-only reports
`..\..\MARULHO_reports\bounded_replay_window_20260624\context-memory-match-comparator-removed.json`,
`query-memory-payload-comparator-removed.json`,
`runtime-concept-memory-lookup-comparator-removed.json`, and
`query-episode-readout-comparator-removed.json` pass bounded context selection
consistency `1.0`, returned-only query payloads, explicit memory-index evidence
recall `1.0`, and target phrase recovery with no global scan, no live tick, and
no hidden language reasoning. The paired long run
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-query-readout-comparators-removed-default-nosample.json`
kept throughput in band at `6586.097 tokens/sec` with bounded `12/65536` route
rows, cached `65526` transition rows, no observed contention, flat RTX memory
`1886 MiB`, and zero graph/native sequence failures.
