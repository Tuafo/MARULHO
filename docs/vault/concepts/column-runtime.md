---
type: concept
status: draft
related_code:
  - ../../../src/marulho/core/column_runtime.py
  - ../../../src/marulho/core/hypercube.py
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/retrieval/routing_index.py
  - ../../../src/marulho/evaluation/bounded_replay_window_benchmark.py
  - ../../../src/marulho/evaluation/source_bank_memory_match_benchmark.py
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
  - ../../../src/marulho/evaluation/status_transition_memory_source_window_benchmark.py
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/status_read_model.py
  - ../../../src/marulho/service/brain_runtime.py
  - ../../../src/marulho/training/model.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/source_tick_sleep_deferral_benchmark.py
  - ../../../src/marulho/evaluation/live_memory_summary_projection_benchmark.py
  - ../../../src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py
  - ../../../src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py
related_docs:
  - ../modules/core.md
  - ../concepts/runtime-truth.md
related_papers:
  - ../papers/predictive-coding.md
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
  - ../benchmarks/replay-cost.md
  - reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery-sharded.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json
  - reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json
---

# Column Runtime

Column Runtime is the MARULHO control-plane direction where many small predictive columns can exist, but only a bounded subset should wake on a tick.

The registry, scheduler, voting, growth, and pruning model remains bounded and evidence-driven. `core.column_runtime` reads existing predictive-column and competitive-column tensors, then reports:

- total columns and awake budget
- active, idle, candidate, cached-vote, sleeping, deep-sleeping, and retired counts
- a sampled Column Registry with role, state, local prediction/surprise/usefulness/cost/memory-pressure state, last-run token estimate, and optional memory budget
- bounded column votes with confidence, prediction error, usefulness, cost, memory pressure, disagreement, wake reason, and cached-vote use
- growth-gate evidence from repeated prediction failure streaks, not one-shot surprise
- pruning/homeostasis evidence from weak, idle, or redundant columns
- the availability of a bounded single-column associative recall helper

Structural mutation remains non-executable from status. Several narrower execution slices are promoted: retrieved candidates bound competition and homeostasis after the early-learning window; retained CPU and CUDA state transition lazily advance stale counters, recent-spike rows, and assembly active-winner state only for candidates/winners; retained CPU routing can filter deep-sleep candidates from a bounded backfill pool without an all-column scan; retained routing can also filter high cached memory-pressure candidates from that bounded pool; CPU predictive updates wake on the same candidate mask after `candidate_predictive_update_start_tokens`; retained predictive voting recomputes reference-frame agreement only for the routed awake mask while non-awake columns keep cached consensus gain; trainer-owned awake-ripple replay tagging uses the wake-plan bucket IDs instead of scanning all recent memory entries; and an eligible CUDA checkpoint can use the trainer-owned in-place steady-state transition for the fused predictive/plasticity cluster. The retained route now creates a training-owned `column_wake_plan` so predictive vote, competitive scoring, predictive update/location update, homeostasis, retained CPU/CUDA state transition, column metabolism updates, awake-ripple tagging, and structural-review ticket capture consume one bounded awake mask with explicit wake/sleep/fallback reasons instead of treating a raw candidate tensor as the scheduler contract. Cached columns carry checkpointed step stamps plus checkpointed cost and memory-pressure state. When a cached candidate wakes, core/training materializes only that bounded candidate set through missed stale-counter age, missed idle homeostasis, missed fallback threshold-relaxation events, and ordered missed non-winner predictive decay before vote/scoring/update. Dense all-column startup/fallback remains truthful where it actually runs. Runtime Truth reports the wake plan, observed scored count, homeostasis scope, state-transition cached/materialized count, homeostasis/predictive materialization age, candidate deep-sleep and memory-pressure filter counts, column metabolism updated/cached counts, predictive location/update cached counts, predictive-vote cached count, awake-ripple bucket-index scan/candidate counts, structural-review queue counts, transition executor, warmup, execution/failure counts, fallback reason, and aggregate `runs_all_columns` truth rather than inferring execution from configured budgets.

State transition scope is now its own truth surface. On retained CPU and promoted CUDA graph/text paths, candidate-scoped ticks use `candidate_subset_lazy_state_transition` or `candidate_subset_sparse_cuda_*`: awake candidates and winners are physically materialized and updated, while non-awake columns keep cached stale-age stamps. Logical `steps_since_win` remains exact through snapshot/materialization helpers, checkpoint save/load, route-vote deep-sleep masking, spike-health reporting, and explicit maintenance. The CUDA transition carries device step counters, per-row stale-age stamps, an all-materialized scalar, recent-spike active IDs, and the assembly active winner so candidate-subset runs do not clear or increment every column just to keep report truth fresh. Runtime Truth exposes `state_transition_mode`, `state_transition_column_count`, `state_transition_cached_count`, `state_transition_cached_fraction`, `state_transition_materialize_*`, `state_transition_runs_all_columns`, and fallback truth beside route/vote/sleep evidence. Dense all-column startup/fallback still reports dense truth; the service read model only projects these fields and does not decide sleep, wake, route candidates, or transition scope.

Column reporting is control-plane work. When source state is on CUDA, Runtime Truth uses one latency-first column-state snapshot for scheduler evidence. A bounded device-export attempt reduced bytes but was slower at current 1024/8192-column sizes, so the active policy favors latency over smaller status payloads. CPU reports still materialize only bounded vote/registry samples. Runtime Truth exposes source device, report compute device, source tensor count, materialized column-state count, snapshot bytes, transfer count, report latency, and the hot-path effect boundary so the optimization cannot be mistaken for a full sleep scheduler or CUDA speedup of cognition.

`ColumnMetabolismState` owns per-column estimated cost, cached memory pressure, and last update stamps on the model device. The trainer updates it only for the live wake-plan mask and checkpoints it with the model. If a memory-store bucket-consolidation tensor is already cached on the compute device, the awake candidates can derive pressure from that evidence; otherwise the source remains `no_memory_store_bucket_evidence` and the filter does not fabricate pressure. The retained CPU route can request a bounded pressure backfill pool and filter only the retrieved candidate IDs against the cached pressure vector. There is no all-column pressure scan to decide sleep. CUDA route-vote now owns pressure masking only when cached pressure evidence exists: the fused route owner reads `ColumnMetabolismState.memory_pressure`, masks high-pressure route rows before top-k/winner selection, and reports pressure filter counts in the same scheduler-filter packet as deep sleep. When pressure evidence is absent, the pressure gate remains disabled rather than adding a decorative route-row pressure read.

Winner/bucket consolidation metrics use the same cache boundary. Retained and
graph metric code may read `DualMemoryStore.bucket_consolidation_level(...)`,
but that scalar API now uses the maintained CPU bucket consolidation cache and
reports `bucket_consolidation_level_cache_lookup.v1` with
`full_memory_scan=false`; a missing cache returns a no-scan miss instead of
rebuilding by iterating slow memory. Explicit tensor rebuilds remain checkpoint
load, graph capture/prewarm, offline diagnostic, or selected-replay recovery
work. The 65536-entry
benchmark reduced scalar lookup from `12.999192 ms` to `0.016260 ms` while
matching the retired full scan, and the paired 524288-token run stayed
same-band at `5967.267 tokens/sec` with bounded `12/65536` route rows and zero
graph/native sequence failures.

`ColumnStructuralReviewQueue` is the durable continuation path for column growth/prune review. It is training-owned, checkpointed with the model, and fed only by bounded awake/candidate IDs plus already-owned predictive/metabolism tensors: prediction error, confidence, prediction failure streak, estimated cost, usefulness, memory pressure, wake reason, and sleep reason. A growth ticket points to `explicit_binding_growth_trial_design`; a prune/sleep ticket points to `isolated_column_prune_or_sleep_review`. Both require operator review and a checkpoint transaction, and both report `mutates_runtime_state=false`. Retained CPU ticks can update the queue from the wake plan immediately. CUDA graph bursts update it only on a slow structural-review cadence; skipped bursts record deferred capture such as `structural_review_cuda_cadence_not_due` so the queue does not add a host sync to every graph host-truth boundary. Runtime Truth projects pending/growth/prune counts, last evaluated/cached column counts, update/deferred counts, baseline queue hash, candidate evidence hashes, next gate, and `runs_all_columns=false`; service does not create or rank tickets. `column_scheduler_benchmark` exposes those queue fields, including an opt-in forced-evidence mode that queues bounded growth and prune/sleep operator-review tickets from the wake plan after measured timing; the retained CPU forced audit is slower, so this is a scheduler truth/continuation surface, not a throughput promotion.

The isolated structural-plasticity evaluator now has a checkpointed-candidate gate. A ticket can feed reviewed transaction design only when the evaluator binds candidate reason, exact baseline hash, cost/usefulness metrics, latency/RAM/VRAM impact, Runtime Truth summary, rollback artifact, and no-mutation proof. Structural Mutation Preflight carries those candidate hashes into the final preflight hash, and the executor rejects tampered candidate provenance before mutation. This keeps repeated prediction failure and prune/sleep pressure on the slow evidence path and prevents service status from becoming a mutation selector.

Awake-ripple replay tagging now consumes the same scheduler boundary. `DualMemoryStore` keeps archival replay storage on CPU, but it maintains a bucket-to-entry index as entries are admitted, restored, or reservoir-replaced. When retained `train_step` or CUDA text burst flushing has awake route candidates, it calls `ripple_tag_awake(..., awake_bucket_ids=..., max_candidate_entries=...)`; the store then collects a recent round-robin candidate window from those awake buckets, reports `bounded_awake_ripple_tag.v1`, and touches only selected entries. If a production caller omits scheduler context, the method returns an empty report with `fallback_reason=awake_bucket_scope_required_for_ripple_tagging` and `last_ripple_scan_mode=awake_bucket_scope_required` instead of scanning all recent memory. The old scalar/vector global scan no longer exists as a runtime hook; benchmark-local retired baselines provide cost comparison only. The direct report `reports/bounded_replay_window_20260617/awake-ripple-bounded-scope-8192-i256.json` compared an 8192-entry diagnostic global scan with a 10-bucket scoped path over 256 iterations: the diagnostic path used `256` vector scans at `1.433332 ms` mean, while the scoped path used `0` scalar/vector scans, `256` bucket-index scans, `last_ripple_awake_candidate_count=10`, and `1.091997 ms` mean (`1.312579x`). The 2026-06-18 hook-retirement report `reports/bounded_replay_window_20260618/awake-ripple-runtime-global-hooks-retired.json` measured the benchmark-local retired scan at `1.285064 ms` versus `1.082768 ms` for the scoped path (`1.186832x`). The matching 65536-column 524288-token hot-path run at `reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-runtime-global-scan-hooks-retired.json` stayed above the maintained band at `6342.218 tokens/sec`, `train_compute=0.128534 ms/token`, `route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`, GPU memory `1801->1802 MiB`, zero graph/native/sequence failures, and brief GPU-side contention at `23%` max utilization.

Slow-memory admission now follows the same selected-window boundary. Fixed cadence is no longer a write path from retained `train_step`: first-token retained/fallback records and selected strong-capture events are the only live archive admissions. Ordinary `slow_memory_archive_interval_tokens` hits record `cadence_deferred` in the cognitive boundary controller and increment skip pressure instead of calling `DualMemoryStore.update(...)` or awake-ripple tagging. The fixed-cadence retirement benchmark `reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json` kept `1` first-token archive and retired `17` projected fixed-cadence writes over `256` tokens, while the refreshed strong-capture benchmark kept `17` selected strong archives. The accepted 65536-column `524288`-token rerun stayed in band at `6043.321 tokens/sec`, bounded route scoring at `12/65536`, cached `65526` transition rows, deferred `2048` cadence hits, and kept graph/native sequence failures at `0`. This makes fixed cadence a Runtime Truth/maintenance signal, not a second replay-admission implementation beside the wake plan.

BrainRuntime source ticks now defer sleep replay through the same boundary. If a background source tick falls back to per-token `train_step` for metrics or unsupported burst execution, it passes `allow_sleep_maintenance=False`; due sleep is counted as deferred maintenance and explicit trainer sleep windows remain the only replay execution path. `reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json` proved service fallback sleep calls stay at `0` while an explicit allowed projection still calls deep sleep once. The paired 65536-column `524288`-token run stayed in band at `5993.959 tokens/sec`, bounded route scoring at `12/65536`, cached `65526` transition rows, no observed contention, flat RTX 3060 memory at `1959 MiB`, and zero graph/native sequence failures. This keeps service as source orchestration and Runtime Truth projection, not a sleep scheduler.

Live memory summary projection now follows the same boundary. Trainer telemetry, BrainRuntime summaries, living-loop status, and status Runtime Truth call `DualMemoryStore.live_summary_stats()` instead of full `summary_stats()`. The live projection reports `bounded_memory_summary_projection.v1`, `summary_full_memory_scan=false`, `summary_scan_entry_count=0`, and `summary_projection_read_only=true`, while still exposing fill/counter aliases and last replay reports. It does not advance STC decay or build tensors over all retained entries; full summary remains an explicit offline consolidation/quality path. `reports/bounded_replay_window_20260620/live-memory-summary-projection.json` measured `0.149500 ms` mean bounded projection latency versus `658.789240 ms` for the retired 65536-entry full summary scan, and the paired `524288`-token run stayed in band at `6024.783 tokens/sec`, bounded route scoring at `12/65536`, cached `65526` transition rows, flat RTX 3060 memory `1959->1958 MiB`, and zero graph/native sequence failures. This keeps service/status as Runtime Truth projection, not memory maintenance or replay selection.

Selected sleep replay routing-index refresh is now the only normal
post-replay routing maintenance path. Deep/repair replay already selected a
bounded replay window and returned the prototype IDs it updated; trainer then
updates only those existing tensor-cache rows through
`routing_index_existing_row_refresh.v1`. `HierarchicalAssemblyIndex` and the
sharded wrapper keep CPU ID-to-row maps for routing-cache metadata while active
routing vectors/ids remain on the index device. Full routing-index rebuild is
reserved for checkpoint restore, bootstrap, explicit offline repair, or
benchmark-local diagnostics. Missing IDs, dirty caches, or missing row-update
APIs now report deferred recovery and do not call `add()+rebuild()` inside
selected replay. The refreshed `65536`-row benchmark updated `16` selected rows,
deferred `1` missing row, and kept exact top-1 recall in `4.171690 ms` mean
versus `118.414640 ms` for the retired rebuild baseline; the sharded variant
passed at `13.348040 ms` versus `140.566380 ms`. The accepted `524288`-token
rerun stayed same-band at `5943.512 tokens/sec` with bounded `12/65536` route
rows, CPU max `28%`, GPU max `19%`, flat `1878 MiB` RTX 3060 memory, and zero
graph/native sequence failures. Service surfaces may report the refresh mode,
row lookup mode, deferred recovery flag, and skipped-update count, but they do
not select replay rows or decide routing maintenance.

Sleep replay now has the same bounded-window accounting. `DualMemoryStore.select_replay_window(...)` reports `bounded_replay_window_selection.v1` before any replay mutation. Deep sleep passes checkpointed `column_anchors` as candidate bucket ids, so a positive-pressure replay window scores only entries reachable through the memory store's bucket-to-entry index. That bucket-indexed path now caps candidates before scoring by recent round-robin across anchor buckets and records `candidate_window_policy=recent_bucket_round_robin_candidate_pool`, `candidate_window_limit`, `candidate_index_available_count`, and scored `candidate_index_count`; a hot bucket can make more entries available, but it cannot make the selector score every stored entry. `DualMemoryStore.recall_replay_window(...)` reports `bounded_replay_window_recall.v1` as a non-mutating slow-path operator over the selected entries: routing-key and input-pattern recall stay CPU-resident, `runs_live_tick=false`, and no plasticity is applied. The report records candidate bucket ids/count, candidate entries scored, selected count, CPU archival/score placement, and whether the lower-level selector used `bucket_indexed_candidate_window` or blocked unscoped selection with `bucket_index_scope_required`. There is no runtime diagnostic global score/candidate branch. Production deep replay no longer mutates from the unscoped global scorer: no-anchor and zero-pressure bucket cases record `unscoped_global_fallback_retired=true` and apply `0` replay updates. The former list-only replay/SFA helpers are now removed: callers use `select_replay_window(...)` and `sample_for_sfa_with_report(...)` so bounded reports are retained; unscoped SFA now reports `selected_replay_window_required`. The capped-window long run processed `262144` tokens at `6148.125 tokens/sec`, kept route scoring bounded at `12/65536`, cached `65526` transition rows, reported no observed contention, held GPU memory flat at `1848 MiB`, and had zero graph/native/sequence failures.

SNN readout-ledger normalization and store-state persistence are also bounded
control-plane work. The Readout Evidence Ledger no longer materializes every
retained event family before capping a snapshot, replay/review helper, or
checkpoint-style persistence copy. Both paths now use the same bounded
newest-first event-field helper: normalization reports
`bounded_snn_readout_ledger_normalization_source_window.v1`, while the store
boundary is measured as `bounded_snn_readout_ledger_store_state_source_window.v1`.
They read at most `128` records per retained event family, keep
archival/normalization/store placement on CPU, and state no live tick, no
every-token cadence, no global score scan, and no hidden language reasoning. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-source-window.json`
used `23` event families with `2048` rows each, read `2944` rows instead of
`47104`, preserved newest-first normalization retention, preserved store-window
parity with the retired list-slice shape, reduced mean normalization latency
from `2415.385992 ms` to `159.388156 ms`, and measured store latency
`159.156636 ms` versus `169.042904 ms`. The 65536-column `524288`-token
no-profile protection rerun stayed in band at `6044.412 tokens/sec`, with
bounded `12/65536` route rows, no observed contention, GPU memory
`2029->2032 MiB`, and zero graph/native sequence failures. This is not a column
wake decision or replay execution path; it keeps replay/readout evidence
summaries and persistence copies from becoming archive scans.

The 2026-06-20 cleanup removes the production all-family normalizer callable
entirely. `SNNLanguageReadoutEvidenceLedger` no longer exposes
`_normalized_state()`; all-family normalization remains only in benchmark-local
retired comparisons that report `production_callable=false` and
`benchmark_local_only=true`. The replacement is one path per source window:
snapshot display, record-family append/review, known-hash lookup, dense-label
calibration/evaluation, emission history, and checkpoint-style store copies all
keep explicit source budgets. The benchmark
`reports/bounded_replay_window_20260620/snn-readout-ledger-normalization-production-normalizer-retired.json`
passed with `2944` bounded all-family rows versus `47104` full-materialized
legacy rows (`16x`), CPU archival/normalization placement, and `0.0 MiB` CUDA
allocation/reservation. The paired `524288`-token run stayed in band at
`6224.717 tokens/sec`, bounded route scoring at `12/65536`, cached `65526`
transition rows, and recorded zero graph/native sequence failures with
borderline `21%` GPU contention.

Readout-ledger snapshots now have their own bounded display source window
instead of reusing all-family normalization. `snapshot(...)` reads only the
event families it returns through
`bounded_snn_readout_ledger_snapshot_source_window.v1`, caps each family at the
requested snapshot limit and ledger retention limit, and keeps the full retained
summary counts as counters rather than by widening the returned history. The
snapshot report keeps archival/source/snapshot placement on CPU, reports no
live tick, no every-token cadence, no global candidate/score scan, no hidden
language reasoning, and no CUDA archive. The benchmark
`reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window.json`
read `260` rows instead of `2944`, preserved newest-first display quality and
retained-count parity, and reduced mean snapshot latency from `393.040600 ms`
to `67.334088 ms` (`5.837171x`) with `0.0 MiB` CUDA allocation/reservation.
The matching `524288`-token protection run stayed in the maintained band at
`6443.960 tokens/sec`, with `train_compute=0.127084 ms/token`, bounded
`12/65536` route rows, no observed contention, and flat RTX 3060 memory at
`1899 MiB`. The broad snapshot-through-normalizer shape is retired.

Known readout-evidence hashes now use the same one-path replay/readout ledger
boundary. Replay design, dry-run, preflight, and bridge provenance checks call
the bounded `events` source-window helper directly instead of normalizing every
retained ledger family first. The helper reports
`bounded_snn_readout_known_evidence_hash_source_window.v1`, CPU
archival/lookup placement, no raw text, no hidden language reasoning, no live
tick, no every-token cadence, no mutation/plasticity, and no CUDA archive. The
combined benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-source-window.json`
preserved known-hash set parity while checking `128` `events` rows instead of
`2944` normalized rows and reducing mean lookup latency from `156.192384 ms` to
`6.792628 ms`. The paired `524288`-token protection rerun stayed same-band at
`5938.461 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, GPU memory `2032->2031 MiB`, and zero
graph/native sequence failures. The broad normalized lookup is benchmark-local
diagnostic evidence only, not a production fallback.

The follow-up report-binding cleanup removes the set-only known-readout helper
as a production path. Column/replay status and artifact recording now use
`known_readout_evidence_hashes_with_report()` so every known readout-evidence
membership check carries `bounded_snn_readout_known_evidence_hash_source_window.v1`
beside the hashes. `ReplayController` requires that report for evaluated
transition-memory replay artifacts, validates CPU archival placement plus no
global scan, raw text, language reasoning, live tick, every-token cadence,
mutation/plasticity, or CUDA archive, and persists
`readout_evidence_source_window_hash` for permit/checkpoint verification. The
focused report
`reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json`
passed with source window `1/8`, `0.014095 MiB` traced Python peak, and
`0.0 MiB` CUDA allocation/reservation; indexed provenance verification reduced
worst-case retained checks from `256` to `4`. The paired hot-path rerun stayed
same-band at `6007.228 tokens/sec` with bounded `12/65536` route rows, `65526`
cached transition rows, zero graph/native sequence failures, and flat RTX 3060
memory under observed GPU contention. This keeps readout replay verification in
the slow/control-plane window and prevents a hash-only bypass from becoming a
second runtime path.

Replay-priority selection now has the same binding. The readout-ledger
transition-memory replay artifact proposal carries
`bounded_snn_readout_replay_priority_source_window.v1` and a matching hash from
`replay_priority(...)`; `ReplayController` rejects evaluated replay artifacts
without that bounded source-window report and includes the hash in artifact and
rollout-review recomputation. The focused report
`reports/bounded_replay_window_20260620/snn-replay-artifact-readout-priority-source-window.json`
passed with source window `1/32`, CPU archival/scoring placement, no global
scan, no raw text, no language reasoning, no live tick or every-token work,
CUDA available but unused, `0.014385 MiB` traced Python peak, and `0.421992 ms`
mean permit-verification latency. The accepted no-contention `524288`-token
rerun stayed same-band at `5937.908 tokens/sec`, with bounded `12/65536` route
rows, `65526` cached transition rows, flat RTX 3060 memory, and zero
graph/native sequence failures; a first contended run at `4662.031 tokens/sec`
is rejected as primary evidence. This retires the priority report-dropping
artifact shape instead of leaving a second replay-selection contract.

Raw caller-window transition-memory replay artifact recording is retired from
production. The Replay Controller keeps
`record_evaluated_snn_transition_memory_replay_artifact(...)` as the only
artifact-recording entrypoint; raw caller-window artifacts are dropped during
controller load, and permit verification recomputes persisted known-readout,
replay-priority, and provenance source-window hashes before accepting an
artifact. The updated report
`reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json`
passed with `public_raw_recorder_callable=false`, `raw_loaded_artifact_count=0`,
`raw_artifact_index_hit=false`, `4` bounded provenance lookups instead of `256`
retained-record checks, mean verification latency `0.538460 ms`, traced Python
peak `0.017773 MiB`, and no CUDA allocation/reservation. The matching hot-path
run processed `524288` tokens at `6004.719 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native sequence
failures, and RTX 3060 memory `1863->1865 MiB`; velocity observed GPU
contention, so the run is same-band throughput protection rather than a clean
speed ceiling.

Dense-label candidate history and calibration policy also use a single bounded
source window. `dense_label_candidate_history(...)` and
`dense_label_candidate_calibration_policy(...)` read only
`dense_label_candidate_events` through
`bounded_snn_dense_label_candidate_calibration_source_window.v1` instead of
normalizing every retained ledger family first. The report records CPU
archival/lookup placement, no raw text, no hidden language reasoning, no live
tick, no every-token cadence, no mutation/plasticity, and no CUDA archive. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json`
preserved history and policy parity while checking `128` dense-label rows
instead of `2944` normalized rows and reducing mean history+policy latency from
`222.453668 ms` to `49.093244 ms`. The paired `524288`-token protection run
stayed in band at `6018.915 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`2030->2029 MiB`, and zero graph/native sequence failures. The broad normalized
dense-label path is retired from production.

Dense-label calibration evaluation continues that single path after preflight.
`dense_label_candidate_calibration_evaluation(...)` now resolves only
preflight-selected candidate hashes through
`bounded_snn_dense_label_candidate_calibration_evaluation_source_window.v1`,
reading `dense_label_candidate_events` with CPU archival/lookup/evaluation
placement and no global scan, raw text payload, hidden language reasoning, live
tick, every-token cadence, mutation/plasticity, or CUDA archive. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json`
preserved sample and metric parity while checking `128` dense-label rows
instead of `2944` normalized rows and reducing mean evaluation latency from
`225.545020 ms` to `12.673884 ms`. The accepted `524288`-token rerun stayed in
band at `6116.710 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`2030->2030 MiB`, and zero graph/native sequence failures. The old
broad-normalized evaluation lookup is retired from production.

Dense-label calibration update application and application-review now use the
same update-family source window. Operator and autonomous calibration update
executors, plus their read-only reviews, read only
`dense_label_calibration_update_events` through
`bounded_snn_dense_label_calibration_update_source_window.v1`; the write side
persists only the update events, current update, update count, and last-applied
timestamp instead of rewriting every ledger family through `_store_state(...)`.
The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json`
preserved event/current hash parity while checking `128` update rows instead of
`2944` normalized rows and reducing mean lookup latency from `245.671760 ms` to
`11.647260 ms`. The paired `524288`-token protection run stayed same-band at
`6009.497 tokens/sec`, with bounded `12/65536` route rows, GPU memory
`2045->2046 MiB`, and zero graph/native sequence failures; its velocity sample
reported GPU-side contention at `21%`, so it is not a new top-speed claim. The
old broad-normalized update/current lookup and all-family update write are
retired from production.

Autonomous calibrated confidence-use now keeps the same ledger boundary. The
hash-only executor and event review read only `autonomous_confidence_use_events`
through `bounded_snn_autonomous_confidence_use_source_window.v1`; the write side
persists only that event family plus use count and last-used timestamp. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json`
preserved confidence-use event-hash parity while checking `128` rows instead of
`2944` normalized ledger rows and reducing mean lookup latency from
`350.647280 ms` to `13.261960 ms`. The paired `524288`-token protection run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json`
stayed same-band at `5965.377 tokens/sec`, with bounded `12/65536` route rows,
no observed contention, GPU memory `2045->2047 MiB`, and zero graph/native
sequence failures. The broad-normalized confidence-use duplicate/review lookup
is retired from production.

Readout-ledger recorders now share that single-family write path. Draft,
rollout-replay, emission-review, and dense-label candidate recorders append
through `bounded_snn_readout_ledger_record_family_source_window.v1`: duplicate
checks read only the target event family, and accepted writes persist only that
event family plus its count and timestamp fields. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json`
preserved latest-hash and total-count parity while checking `128` rows instead
of `2944` normalized ledger rows and reducing mean append latency from
`883.251340 ms` to `57.255420 ms`. The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-record-family-append.json`
stayed same-band at `5966.765 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`2046->2043 MiB`, and zero graph/native sequence failures. The old
broad-normalized single-family record append is retired from production.

The autonomous hash-readout chain now keeps the same boundary. Binding
execution/review read only `autonomous_hash_readout_binding_events`, and
bound-observation execution/review read only
`autonomous_bound_readout_observation_events`; no stage normalizes all retained
ledger families to append or review one hash-only event. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json`
preserved hash, review-match, and count parity while checking `512` target-family
rows instead of `11776` normalized rows and reducing mean chain latency from
`2371.472400 ms` to `110.685950 ms`. The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-autonomous-chain.json`
stayed same-band at `6272.156 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`2044->2045 MiB`, and zero graph/native sequence failures. The old
broad-normalized autonomous binding/observation append and review path is
retired from production.

The downstream autonomous training-window and decoder-probe event families now
share that boundary too. Training execution/review read only
`autonomous_readout_training_window_events`; decoder-probe execution/review read
only `autonomous_decoder_probe_events`; no stage normalizes all retained ledger
families or builds a `snapshot(limit=0)` summary to append/review one event. The
expanded benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-training-probe-chain.json`
preserved hash, review-match, and count parity across binding, observation,
training, and decoder events while checking `1024` target-family rows instead
of `23552` normalized rows and reducing mean chain latency from `4927.213200 ms`
to `197.573467 ms`. The paired `524288`-token hot-path run
`reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json`
stayed same-band at `6057.953 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, GPU memory `2046->2064 MiB`, and zero
graph/native sequence failures. GPU-side contention reached `24%`, so this is
live-tick protection evidence under contention, not a clean speed ceiling. The
old broad-normalized autonomous training/probe append and review path is retired
from production.

Replay query collection now uses the same bounded window. `DualMemoryStore.collect_replay_query_indices(...)` reports `bounded_replay_query_collection.v1` for HF replay recall and returns recent bucket-indexed query indices up to `max_queries` instead of walking `slow_bucket_ids` until enough anchor hits are found. The report records candidate buckets, available versus collected index counts, query indices, skipped missing input-pattern payloads, `score_count=0`, no global score/candidate scan, CPU archival placement, and `runs_live_tick=false`. The HF query-collection report at `reports/bounded_replay_window_20260617/hf-recall-capped-query-collection/summary.json` kept recall and consolidation gates passing, collected `3` Task-A anchor queries through a `candidate_window_limit=16`, accepted `6` guarded repairs, and kept after-consolidation input-pattern recall exact. The matching long hot-path run processed `262144` tokens at `6221.949 tokens/sec`, kept route scoring bounded at `12/65536`, cached `65526` transition rows, reported no observed contention, held GPU memory flat at `1848 MiB`, and had zero graph/native/sequence failures.

Explicit query readout is bounded the same way. `query_runner.memory_matches_with_report(...)` reports `bounded_query_memory_match.v1` and uses routing-owned candidate bucket ids to collect a capped bucket-indexed memory window before computing similarity, semantic term support, or replay-priority scores. The query report at `reports/bounded_replay_window_20260617/query-memory-match-bounded-window.json` used `candidate_window_limit=192`, scored `1` candidate, returned `1` memory match, reported no global score/candidate scan, kept archival placement on CPU, and marked `runs_live_tick=false` and `mutates_runtime_state=false`. The matching long hot-path run processed `262144` tokens at `6137.185 tokens/sec`, kept route scoring bounded at `12/65536`, cached `65526` transition rows, reported no observed contention, held GPU memory flat at `1848 MiB`, and had zero graph/native/sequence failures.
The returned-only payload follow-up keeps similarity-only query readout
tensor-first until after ranking: `query-memory-payload-returned-only.json`
loads raw text for `5` returned matches instead of all `192` candidates,
preserves selected indices, and the 524288-token hot-path check stays in band
at `6152.079 tokens/sec` with zero graph/native/sequence failures.

Autonomy source-acquisition frontier metrics now use the same memory boundary. `concept_frontier_metrics_with_report(...)` derives candidate buckets from the probe-bank routing signature, calls the memory store's capped bucket-indexed collector, and emits `bounded_concept_frontier_memory_metrics.v1` while scoring only selected routing keys for novelty, uncertainty, and support. The synthetic scope benchmark at `reports/bounded_replay_window_20260617/concept-frontier-bounded-scope.json` scored `64/8192` entries at `5.040 ms` mean versus `658.116 ms` for the diagnostic full scan, preserved the full-scan top-1, and reported no global score/candidate scan. The matching 262144-token hot-path check stayed in band at `6148.846 tokens/sec`, with bounded `12/65536` route rows, `65526` cached transition rows, flat `1805 MiB` GPU memory, no observed contention, and zero graph/native/sequence failures.

The source-bank signature that seeds that frontier metric is now bounded too. `concept_frontier_metrics_with_report(...)` and `candidate_semantic_signature(...)` sample an evenly spaced `16`-probe window from the source bank before computing the routing signature, and the Runtime Truth report carries `source_probe_count`, `source_probe_window_limit`, `source_probe_indices`, and source-probe selection-budget fields. The direct report `reports/bounded_replay_window_20260618/concept-frontier-source-probe-window-bounded.json` sampled `16/64` source probes, scored `64/16384` memory entries, preserved the diagnostic top-1, and reduced mean latency from `1556.602 ms` to `7.637 ms`. The paired 524288-token hot-path check kept the current tree in the same band as the committed baseline (`6303.548` versus `6307.437 tokens/sec`), with bounded `12/65536` route rows, `65526` cached transition rows, no observed contention, flat `1789 MiB` GPU memory, and zero graph/native/sequence failures.

Source-bank semantic recall also stays in the slow-path acquisition boundary.
`bank_memory_matches_with_report(...)` samples bounded source-bank probes,
delegates each probe to `bounded_query_memory_match.v1`, and emits
`bounded_source_bank_memory_match.v1` with per-probe windows, candidate totals,
unique candidate count, payload cache hits, CPU archival/score placement, and
no global scans. The 65536-entry benchmark
`reports/bounded_replay_window_20260618/source-bank-memory-match-bounded.json`
kept selected indices identical to the diagnostic legacy path while reducing
raw text payload loads from `32` to `4`; the 524288-token protection rerun
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json`
kept route scoring bounded at `12/65536`, cached `65526` state-transition rows,
and reached `6524.395 tokens/sec` with no observed contention. This is
source-acquisition evidence and not a live scheduler, topology, or mutation
authority.

ConceptStore signature lookup also stays bounded to already-selected evidence.
`ConceptStore.observe(...)` now reports
`bounded_concept_memory_signature_lookup.v1`, caps each evidence source at `8`
unique memory indices, direct-indexes CPU archival arrays, and records
`archive_list_materialization_count=0` with no global candidate or score scan.
The benchmark at
`reports/bounded_replay_window_20260617/concept-signature-lookup-bounded.json`
used `65536` entries and preserved the diagnostic legacy signature quality
(`min cosine=0.9999998212`) while reducing mean lookup latency from
`12.490 ms` to `1.454 ms`. The clean 262144-token hot-path check stayed in band
at `6143.768 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, flat `1746 MiB` GPU memory, no observed contention, and zero
graph/native/sequence failures. Longer 524288-token same-code checks were fast
but secondary because their condition report saw pre-measurement GPU
contention.

Semantic frontier-gap planning is also bounded to a selected slow-path window.
`frontier_gap_plan(...)` now calls
`DualMemoryStore.collect_frontier_gap_indices(...)` and scores only a capped
CPU recency or bucket candidate window before exposing
`bounded_frontier_gap_selection.v1`. The report records candidate budget,
selected indices, raw text payload count for selected candidates only, no global
candidate/score scan, CPU archival placement, `runs_live_tick=false`, and
`language_reasoning=false`. The benchmark at
`reports/bounded_replay_window_20260617/frontier-gap-bounded.json` preserved
expected and diagnostic legacy terms with `quality.min=1.0` while scoring
`192/65536` entries and reducing mean latency from `217.530 ms` to `9.073 ms`.
The refreshed gate also proves a store missing
`collect_frontier_gap_indices(...)` returns an empty reported fallback with
zero candidates, zero text payloads, and no global scans instead of reading
archive text through a compatibility path.
The longer 524288-token hot-path check
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json`
stayed in band at `6233.085 tokens/sec`, with bounded `12/65536` route rows,
`65526` cached transition rows, GPU memory `1844->1840 MiB`, no observed
contention, and zero graph/native/sequence failures.

Positive-pressure deep replay now commits through `bounded_reconstruction_gated_candidate_repair`, not the old stored-bucket `CompetitiveColumnLayer.process(...)` mutation. The trainer de-duplicates selected replay traces, builds a bounded candidate-column set from route candidates plus explicit stored-bucket fallback candidates, temporarily tests prototype repair, and commits only candidates that improve `mean_one_minus_best_similarity_over_selected_replay_routing_keys` inside the selected replay-window candidate columns. Runtime Truth records unique traces, duplicate skips, rejected commits, candidate-column union/trial counts, updated-column count, quality before/after, score device, archival storage device, and `runs_live_tick=false`. Emergency repair mode is also anchor-bucket scoped now: `bounded_repair_reanchor` can run only from a bucket-indexed anchor window, while no-anchor repair records `no_anchor_bucket_scope_for_repair_replay` and applies `0` updates. Micro maintenance is scoped the same way: anchored micro refresh reports `bounded_micro_maintenance_refresh`, selects through `bucket_indexed_candidate_window`, updates CPU memory metadata only, and bypasses the old zero-LR `CompetitiveColumnLayer.process(...)` call; no-anchor micro refresh records `no_anchor_bucket_scope_for_micro_replay` and applies `0` updates. The 2026-06-17 synthetic report `reports/bounded_replay_window_20260617/synthetic-selection-candidate-repair-bounded-micro.json` passes bounded stored input-pattern recall under positive pressure (`5.960464477539063e-08` mean input distance, threshold `0.01`) and passes the prototype reconstruction gate after `6` committed repairs across `4` cycles: Task-A reconstruction moved from `0.0052170157` after Task B to `0.0034434795` after consolidation, relative degradation was `0.0467838377` under the `0.05` threshold, overlap was `0.8981397152`, and zero-pressure/no-anchor arms still applied `0` updates. The HF-backed guarded report `reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json` passes bounded stored-experience recall after consolidation over `3` Task-A anchor-window queries with `mean_input_pattern_distance=0.0` and adds `reconstruction_guarded_replay_consolidation.v1`: replay cycles are selected from the bounded anchor window, then accepted only if Task-A `mean_reconstruction_error` does not regress. In that run the guard attempted `9` candidate repair updates across `3` cycles, rejected all `9`, restored model/memory state, kept effective updates at `0`, and left the memory-consolidation gate passing. The cadenced follow-up `reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json` adds `cadence_strategy=skip_repeated_rejected_selection`, so the first rejected cycle attempted `3` repairs and the next `2` identical rejected cycles were skipped without re-entering replay. The longer 65536-column hot-path check at `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-candidate-repair.json` stayed in the same band at `6306.507 tokens/sec`, `train_compute=0.129511 ms/token`, `tick_duration_ms.p95=20.176`, `route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`, zero graph/native/sequence failures, and no observed contention. The current-tree repair-scope rerun at `reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json` also stayed in band at `6252.073 tokens/sec` with `train_compute=0.129794 ms/token`; the micro-scope rerun processed `262144` tokens at `6332.439 tokens/sec` with `train_compute=0.129001 ms/token`. The guarded-consolidation current-tree rerun processed `262144` tokens at `6606.251 tokens/sec`, `train_compute=0.123393 ms/token`, `tick_duration_ms.p95=18.562`, `route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`, zero graph/native/sequence failures, flat `1539 MiB` GPU memory, and no observed contention. The cadenced rerun stayed in band at `6199.988 tokens/sec`, `train_compute=0.130574 ms/token`, `tick_duration_ms.p95=20.215`, the same `12/65536` route rows, `65526` cached transition rows, zero graph/native/sequence failures, and no observed contention after an earlier same-code run was rejected as low-throughput evidence.

Target-aware replay-strength search now sits inside `reconstruction_guarded_replay_consolidation.v1` rather than beside it. Runtime Truth records `repair_strength_strategy`, the exact schedule, `repair_strength_trial_budget`, budget policy, per-strength trial reports, selected strength, attempted versus effective applied counts, rejected trial attempts, and repeated-rejection skips. HF text consolidation now uses the single-strength `[0.1]` budget: `reports/bounded_replay_window_20260617/hf-recall-target-strength-budget-single-010-promoted/summary.json` accepted `6` post-Task-B repairs with `0` rejected trial attempts, improved Task-A reconstruction from `0.0170305534` to `0.0149637708`, cut post-B guard latency from the prior four-low-strength `3477.025 ms` to `1040.506 ms`, and preserved exact bounded recall (`mean_input_pattern_distance=0.0`) plus the memory-consolidation gate. The synthetic prototype stress default now uses compact escalation `[0.1, 0.5, 1.0]`: `reports/bounded_replay_window_20260617/synthetic-target-strength-budget-compact-default.json` keeps recall and prototype gates passing with `repair_strength_trial_budget=3` and `2585.941 ms` guard latency, while the single-strength synthetic control is rejected because it failed the prototype gate. The matching 65536-column hot-path check stayed in band at `6232.282 tokens/sec`, with `route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`, no all-column transition, zero graph/native/sequence failures, flat `1715 MiB` GPU memory, and no observed contention. Replay selection and archival metadata remain CPU-resident; only active guard scoring and repair trials touch the model device inside explicit slow windows.

Sleep replay now also has explicit replay-text and SFA boundaries. Query and
display paths may still request stored `raw_window`/`text` payloads, but sleep
replay calls `DualMemoryStore.replay_entry(..., include_text_payload=False)`.
Selection and recall reports record `raw_text_payload_loaded=false` and
`language_reasoning=false`; sleep replay records
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`,
`sleep_replay_text_payload_policy=sleep_replay_uses_tensor_payloads_only`, and
`sleep_replay_local_trace_source=stored_input_pattern_or_routing_key`. The old
raw-window eligibility-trace branch is retired for sleep replay. Deep replay
with abstraction now samples SFA correction from the selected
`processed_indices` rather than the whole slow buffer and exposes
`sleep_replay_sfa_correction_scope`, `sleep_replay_sfa_candidate_index_count`,
`sleep_replay_sfa_sample_count`, and
`sleep_replay_sfa_full_memory_sample_retired=true`. The final boundary report
`reports/bounded_replay_window_20260617/synthetic-replay-tensor-payload-boundary.json`
keeps bounded recall/prototype gates passing, and the matching 65536-column
262144-token check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json`
stayed in band at `6237.420 tokens/sec` with `12/65536` route rows, `65526`
cached transition rows, zero graph/native/sequence failures, flat `1719 MiB`
GPU memory, and no observed contention.

Anchored repair replay no longer builds a dense input assembly after selected
entries already provide stored routing keys. `bounded_repair_reanchor` prepares
input state with `prepare_input_for_candidate_routing(...)`, clears stale dense
caches when an input trace is absent, and reports
`sleep_replay_unconditional_dense_input_assembly_retired=true`,
`sleep_replay_dense_input_assembly_fallback_count`, and
`sleep_replay_bounded_input_prepare_count`. The 65536-column benchmark
`reports/bounded_replay_window_20260618/sleep-repair-replay-bounded-input-prepare.json`
selected `32` anchored entries, repaired all `32`, improved mean anchor
distance by `0.148684`, reduced selected input-prep latency from `61.351 ms` to
`32.613 ms`, made `0` dense assembly calls during repair, and kept archival
payloads on CPU. The paired long hot-path run stayed in band at
`6302.207 tokens/sec` with bounded `12/65536` route rows and no observed
contention.
The remaining missing-key repair fallback is also retired. Repair replay and
deep candidate repair now require stored routing keys; entries without that key
are deferred, counted under `sleep_replay_missing_routing_key_deferred_count`,
and no longer project selected stored assemblies. The mixed-key benchmark
`reports/bounded_replay_window_20260620/sleep-repair-replay-missing-routing-key-deferred.json`
updated `16` stored-key entries, deferred `16` missing-key entries, made `0`
dense input-assembly calls, improved stored-key repair quality by `0.149600`,
and the 524288-token hot-path check stayed in band at `5988.223 tokens/sec`
with bounded `12/65536` route rows, `65526` cached transition rows, GPU memory
`1877->1878 MiB`, and zero graph/native/sequence failures.

`PredictiveColumnState` owns `prediction_failure_streak` beside prediction error and confidence. Repeated raw failures increment the streak on the predictive tensor device; successful prediction resets it. The streak is saved in trainer checkpoints and restored with the model, so growth evidence survives rollback and does not live in `service`.

When that gate is ready, the explicit binding-growth trial endpoint can ask core for a deterministic candidate-scoped hypercube outreach plan. The plan is bounded by an edge budget, hashes the exact baseline adjacency, and remains read-only. This narrows the path from surprise to structural experimentation without making the scheduler, Runtime Truth, or status polling a mutation authority.

`PredictiveColumnState` also records the last predictive location/update scope, lazy predictive materialization scope, and the last predictive-vote execution scope. `candidate_predictive_update_start_tokens` now separates predictive-state wake from structural dead-column retirement: after that gate, location/velocity decay, prediction error, confidence, failure streaks, and high-prediction non-winner decay update only for the routed awake mask while non-candidate state remains cached on the retained CPU route and promoted fused CUDA route. If a non-awake column later appears in the routed candidate set, `PredictiveColumnState` advances only that candidate through missed non-winner predictive updates before vote/scoring uses its state. The retained CPU route now applies prediction error, location/velocity, prediction-weight decay, and cached-state materialization through one candidate predictive transition rather than re-canonicalizing the same awake set across three split calls. Vote and update can share one candidate materialization in a tick, and a repeated materialization request for the same completed candidate set reports `candidate_subset_completed_noop`. Checkpoint restore recomputes the cached-column flag from predictive step stamps so a restored runtime still wakes stale predictive columns correctly. Predictive wake currently keeps ordered replay for cached candidates because vectorized closed-form replacements improved cost but failed repeated long winner-parity checks near fallback-threshold boundaries. `CompetitiveColumnLayer` does the same for missed zero-activity homeostasis and fallback threshold-relaxation events, preserving dense update order without a hot-path all-column tax. The retained trainer route may first request a bounded backfill pool, sort it by retrieval distance, and remove candidate columns already at the deep-sleep threshold; before `dead_column_steps` is reachable, it skips backfill and reports `candidate_deep_sleep_filter_no_column_can_be_deep_sleep_yet` because no candidate can truthfully be deep-sleep eligible yet. If every retrieved candidate is deep-sleeping, it falls back to the bounded retrieved set and reports the reason. For retained predictive voting, the trainer obtains routing candidates before consensus voting, then recomputes agreement only for the awake mask and reuses cached gains for the other columns. If fused CUDA vote/competition must fall back because context gain is present, that retained selection fallback still passes the routed candidate IDs into `PredictiveColumnState.vote()` and hands `CompetitiveColumnLayer` a candidate-local gain tensor, so unsupported fused selection does not silently become an all-column predictive vote. On CUDA predictive updates, eager candidate indexing remains rejected: the 2026-06-15 isolated writeback experiment measured `7.0080195312499995 ms` mean for eager candidate indexing versus `3.080762890625 ms` dense writeback. The promoted fused in-place/graph route updates only the wake-plan candidates inside the existing transition launch, stamps those rows on device, and reports `candidate_predictive_transition_mode=fused_inplace`, active/fallback truth, execution count, last-transition cached-row count, and cumulative cached row-skips. When candidate homeostasis starts earlier than predictive wake, the persistent CUDA runtime captures a `candidate_subset_dense_predictive` graph for the interim dense predictive update and a separate candidate predictive graph for the later sparse wake, so the two gates can differ without disabling the fused transition. The standalone candidate predictive writeback helper, isolated predictive transition benchmark, configurable `compiled` dense predictive mode, configurable `legacy` dense-transition bypass, selectable `fused_eager` production mode, and hidden `apply_dense_transition(..., transition_mode=...)` selector are removed so the vault has one promoted CUDA mutation boundary. Unsupported in-place prerequisites fall back internally to `dense_eager_fallback` tensor semantics with a concrete Runtime Truth fallback reason before claiming sparse predictive updates.

The CUDA route-vote owner now applies deep-sleep and evidence-backed memory-pressure/usefulness gates before candidate vote/winner selection by masking route-score rows inside `core.fused_route_vote_cuda`. This is not a post-selection status projection: `ColumnTransitionRuntime` stages a six-value device control tensor, the graph/fused route writes a sixteen-value scheduler-filter packet, and `MarulhoTrainer` builds the `ColumnWakePlan` from that training-owned route evidence. The state packet is mirrored on a cadence, not every token, so Runtime Truth can expose current filtered counts, observed cumulative filtered totals, and fallback reasons without adding a hot-path all-column scan. If fewer than `k` eligible route rows exist, the implementation keeps the unfiltered fixed-k route result and reports an explicit fallback reason such as `insufficient_awake_route_scores_after_deep_sleep_filter`, `insufficient_awake_route_scores_after_memory_pressure_filter`, or `insufficient_awake_route_scores_after_usefulness_filter`. Route-cost truth is separate from awake-mask truth: `route_vote_scoring` reports `route_input_rows_scored`, `route_output_candidate_count`, `route_rows_run_all_columns`, `bounded_route_scoring`, and `route_scoring_unbounded_reason`. The promoted CUDA route now seeds a fixed `k_routing` `route_candidate_bank` once from the complete routing tensor cache, then uses `indexed_route_bank_vote_device_refresh` to score the bank plus a fixed two-row probe lane and write the next route-bank/probe positions on device during steady graph ticks. `route_candidate_bank_size` is retired as a production config selector; old checkpoint keys migrate away, and wider banks remain evaluation-only. `predictive_route_vote_mode` is fixed to `cuda_graph_text`; retired `tensor` and `fused_triton_text` values migrate out of checkpoints, and old modes can be forced only by explicit evaluation overrides for parity/benchmark controls. Runtime Truth keeps the seed visible with `candidate_boundary=exact_full_cache_score_seed_route_bank` and `route_candidate_bank_not_ready_exact_seed`, then reports `bounded_route_bank_probe_score_then_filter_select` or `bounded_route_bank_probe_burst_score_then_filter_select` with `route_input_rows_scored=k_routing+2`, `route_output_candidate_count=k_routing`, `bounded_route_scoring=true`, `refresh_owner=fused_route_vote_device`, device/host refresh counts, and `route_vote_requested_mode_source` showing promoted config, restore defer, or evaluation override.

The route bank now has a bounded discovery lane instead of pure k-only self-refresh. Steady CUDA graph/burst ticks score `k_routing` bank rows plus `2` probe rows (`score_rows=12` for the 8192-column checkpoint) while still waking and mutating only `k_routing` columns. The earlier graph-boundary probe refresh cadence remains a quality/evaluation shape, but the promoted device-refresh path reports effective `refresh_interval_tokens=1` when the fused route/vote owner advances the bank and probe cursor on device. Checkpoints save ready route-bank IDs plus probe cursor state and restore them before CUDA graph capture, so restored runs can avoid the first-tick exact-cache seed when the saved bank still maps into the routing tensor cache. Runtime Truth exposes `probe_rows`, `score_rows`, `probe_cursor`, `scored_since_refresh`, `probe_refresh_count`, `device_refresh_count`, `probe_device_refresh_count`, `checkpoint_restore_count`, and `restore_reason` beside `route_input_rows_scored`. The longer 8192-column real-path run `reports/column_scheduler_20260616/route-bank-probe-lane-8192-131072-i32.json` stayed in the promoted band at `6141.234 tokens/sec`, `train_compute=0.130664 ms/token`, `tick_duration_ms.p95=21.976`, `route_input_rows_scored=12/8192`, `route_output_candidate_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The checkpoint-backed restore gate at `reports/column_scheduler_20260616/route-bank-checkpoint-restore-8192-131072-i32.json` reported `seed_count=0`, `fallback_count=0`, `graph_bypass_count=0`, `checkpoint_restore_count=1`, `route_input_rows_scored=12/8192`, `state_transition_cached_count=8182`, and `6129.693 tokens/sec` with `0.130218 ms/token` train compute. The reproducible promoted-checkpoint builder then extended the same complete-runtime gate to `16384` and `32768` columns: the `16384` gate reached `6154.503 tokens/sec`, `train_compute=0.130874 ms/token`, `tick_duration_ms.p95=21.419`, `route_input_rows_scored=12/16384`, and `state_transition_cached_count=16374`; the `32768` gate reached `6298.380 tokens/sec`, `train_compute=0.130618 ms/token`, `tick_duration_ms.p95=20.991`, `route_input_rows_scored=12/32768`, and `state_transition_cached_count=32758`. Both scale gates kept `route_output_candidate_count=10`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The follow-up quality gate rejected fixed sequential probes as a relevance solution: probe2 q16 matched exact top-1 only `0.017578125` and winner only `0.001953125`; probe256 q16 still matched top-1 only `0.34765625` and winner only `0.044921875`. The 32768 same-checkpoint diagnostic kept the rejection after device refresh: probe16 per-token scored `26/32768` rows and barely moved top-1 (`0.279296875`) while winner stayed `0.046875`; probe64 raised top-1 to `0.591796875` and exact-winner-in-bank to `0.486328125`, but winner match stayed `0.046875`, and even the offline oracle-previous diagnostic reached only `0.455078125`. This is bounded outside-bank discovery and route-cost truth, not growth/pruning autonomy or a full relevance-quality solution; the next router needs structured GPU-owned discovery plus winner-state continuity.

The 2026-06-17 `65536`-column gate kept that same promoted path instead of adding an opt-in branch. The builder report `reports/column_scheduler_20260617/promoted-scheduler-65536-checkpoint.json` paid the exact full-cache seed before checkpoint save and restored into bounded steady scoring (`12/65536` route rows, `10` candidates, `65526` cached transition rows, `state_transition_runs_all_columns=false`). The longer run `reports/column_scheduler_20260617/promoted-scheduler-65536-131072-i32.json` processed `131072` tokens at `6154.501 tokens/sec`, `train_compute=0.130339 ms/token`, `prepare_training=0.006100 ms/token`, `finalize_total=0.006153 ms/token`, and `tick_duration_ms.p95=20.092`, with `route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`, `refresh_owner=fused_route_vote_device`, `device_refresh_count=131072`, `host_refresh_count=1`, `route_rows_run_all_columns=false`, `bounded_route_scoring=true`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The active-pressure fixture report `reports/column_scheduler_20260617/active-pressure-scheduler-65536-checkpoint.json` then marked exactly two cached route-bank rows as high pressure, matching the two probe rows so restored route-vote could mask them without fallback. The long run `reports/column_scheduler_20260617/active-pressure-scheduler-65536-131072-i32.json` reached `6297.455 tokens/sec`, `train_compute=0.130524 ms/token`, `prepare_training=0.006053 ms/token`, `finalize_total=0.006145 ms/token`, `tick_duration_ms.p95=20.115`, and no observed contention while reporting `route_input_rows_scored=12/65536`, `route_output_candidate_count=10`, `state_transition_cached_count=65526`, `observed_filtered_memory_pressure_total=2`, `observed_filtered_deep_sleep_total=254`, `observed_fallback_count=0`, zero graph/native/sequence failures, and `state_transition_runs_all_columns=false`. The service status surface now projects this as `route_vote_scheduler_filter` from training/core Runtime Truth, keeping active filter evidence visible without letting `service` decide wake or sleep. Total columns doubled again, but the scored route rows and awake mutation rows stayed fixed, and the pressure/sleep mask has active execution evidence rather than only a zero-count status field.

The wider-bank quality diagnostic narrows the next scheduler requirement without changing the promoted runtime. `route_candidate_bank_quality_gate` now separates retained bank capacity from awake candidates: `bank1024+probe256` on the 8192 checkpoint scored `1280/8192` rows and passed offline top-1/winner quality (`1.0` / `0.97265625`), while still waking only `10` candidates. A live CUDA attempt then added GPU-owned top-retained refresh for `1024+256` route rows, but the 65536-column 131072-token gate regressed to `4656.790 tokens/sec` and `train_compute=0.186216 ms/token` against the retained k+2 path at `6156.500` and `0.130623`; an unsorted top-k short probe was still only `4596.633` and `0.176464`. A same-row-budget graph-neighbor probe also failed promotion: `neighbor_count=2`, `capacity_rows=12`, and no sequential probe improved exact top-1 to `0.078125` versus the probe2 rerun at `0.013671875`, but exact winner match stayed `0.001953125`. The failed runtime branch was removed, so the promoted runtime remains `indexed_route_bank_vote_device_refresh` with `route_input_rows_scored=k_routing+2`. Smaller wider banks also failed winner parity (`bank256+probe64` winner `0.630859375`; `bank512+probe128` winner `0.826171875`), so simple widening or same-row neighbor replacement is not enough.

The promoted CUDA/text scaling path now starts from the candidate scheduler boundary instead of carrying unused dense startup work when the checkpoint is already past the candidate gate. `ColumnTransitionRuntime` precompiles only the routed candidate shape, the persistent text graph captures only `candidate_subset`, and Runtime Truth exposes `capture_graph_policy`. The 2026-06-15 8192-column real-path run proved the previous fallback is fixed: `reports/real_path_column_scaling_20260615/runtime-8192-promoted-131072-i32.json` stayed active on `cuda:0`, reported `precompiled_candidate_counts=[10]`, graph names `["candidate_subset"]`, route-vote output `10`, no graph/native fallback, and `3564.222 tokens/sec` with `0.251487 ms/token` train compute. The matching 1024-column control at `reports/real_path_column_scaling_20260615/runtime-1024-promoted-131072-i32.json` reached `6108.728 tokens/sec` with `0.133438 ms/token`. This proved bounded awake execution, not route-row cost invariance: route-vote input rows still grew from `1024` to `8192`, and the `route_vote_scoring` surface made that explicit. The 2026-06-16 sparse CUDA state-transition slice then closed the dense state-transition gap for candidate-subset graph/burst ticks: `reports/column_scheduler_20260616/sparse-cuda-state-transition-candidate-homeostasis-131072-i32.json` reports `state_transition_column_count=10`, `state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`, and `6007.582 tokens/sec` with `0.132374 ms/token` train compute. The route-candidate-bank slice moved the next boundary into route scoring itself: the 1024-column warm-seed run at `reports/column_scheduler_20260616/route-candidate-bank-warmseed-131072-i32.json` reports `route_input_rows_scored=10`, `route_rows_run_all_columns=false`, `bounded_route_scoring=true`, `state_transition_column_count=10`, `state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`, and `6109.301 tokens/sec` with `0.130536 ms/token` train compute. The matching 8192-column gate at `reports/column_scheduler_20260616/route-candidate-bank-8192-warmseed-131072-i32.json` reports `route_input_rows_scored=10`, `route_input_fraction=0.001220703125`, `route_rows_run_all_columns=false`, `bounded_route_scoring=true`, `state_transition_column_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, and `6110.715 tokens/sec` with `0.135007 ms/token` train compute. The follow-up route-candidate-bank quality gate keeps this throughput claim separate from relevance quality: the 8192-column default-text oracle at `reports/column_scheduler_20260616/route-candidate-bank-quality-8192-default-text-s512.json` kept steady bank scoring at `10` rows, but exact top-1 was in the bank only `0.009765625` of ticks, the simulated winner matched only `0.001953125`, mean top-k overlap was `0.02578125`, and the worst exact top-1 miss streak was `134`. A 32-tick evaluation-only exact reseed still failed winner parity, while a random/stable control passed. The corrected graph-neighbor quality probe at `reports/column_scheduler_20260616/route-candidate-graph-quality-8192-default-text-s512-neighbor208-cap1536.json` used `neighbor_count=208` and `capacity_rows=1536`, scored `1481.068` rows on average, matched exact top-1 `0.994140625`, matched the simulated exact winner `0.98828125`, and limited the worst miss streak to `1`. That proves bounded discovery is plausible, but the matching live runtime attempt at `reports/column_scheduler_20260616/route-candidate-graph-neighbor208-cap1536-8192-131072-i32.json` reached only `4682.167 tokens/sec` and `0.183927 ms/token` train compute while scoring about `1509/8192` route rows, below the promoted `6110.715`/`0.135007` route-bank baseline. After the runtime graph-neighbor path was removed, `reports/column_scheduler_20260616/route-candidate-bank-8192-warmseed-131072-i32-after-graph-reject-cleanup.json` reached `6150.296 tokens/sec`, `0.130390 ms/token` train compute, `route_input_rows_scored=10`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The live graph-neighbor path is therefore rejected and removed; only the offline quality gate remains. The promoted steady path is real bounded execution, but k-only self-refresh is not a complete relevance scheduler for changing text. The fixed probe lane is the promoted bounded-cost path, not a solved relevance router: it samples outside the current bank without a hidden hot-path all-column tax and preserves the long 6k-ish path, but later quality evidence keeps discovery-quality promotion open.

The bounded graph-walk probe narrowed the graph-neighbor question without promoting a second runtime. `route_candidate_bank_quality_gate` can now simulate a CAGRA-style fixed-degree walk: start from the route bank plus two probe rows, score a bounded frontier, keep a fixed beam, and expand neighbor frontiers for fixed rounds. The 8192-column default-text evidence stayed below the promotion gate. A 512-row walk reached exact top-1 `0.896484375` but winner match only `0.263671875`; the best 1024-row deep walk reached exact top-1 `1.0` but winner match only `0.73046875`; a 1536-row deep walk still matched the simulated winner only `0.767578125`. This rejects top-1-only graph walking as a scheduler boundary and keeps graph-walk code evaluation-only until a fused/GPU-owned variant can pass winner quality and the 131072-token 6k-ish gate. The same-session 131072-token recheck kept the promoted route-bank/probe-lane runtime unchanged at `6141.295 tokens/sec`, `train_compute=0.132462 ms/token`, `route_input_rows_scored=12/8192`, and `state_transition_runs_all_columns=false`; GPU contention was observed, so it is in-band evidence, not a new top-speed claim.

The column-ID hypercube-neighbor probe rejected another tempting topology shortcut before it reached runtime. The evaluation-only gate can score bounded bit-flip neighbors of the runtime previous winner, and a toy local-bitflip fixture passes, but the real 32768-column default-text checkpoint kept exact top-1 at `0.2734375`, exact-winner-in-bank at `0.09375`, winner match at `0.046875`, and oracle-previous diagnostic at `0.08984375` while scoring `25.215/32768` rows on average. The matching long real-path check stayed in band at `6149.283 tokens/sec`, `train_compute=0.131785 ms/token`, `route_input_rows_scored=12/32768`, and `state_transition_cached_count=32758` because runtime did not use the rejected neighbor lane. Current column IDs are not routing locality; this remains evaluation-only until topology or training makes that claim real.

The retained fallback vote-scope cleanup closes a smaller fake-sleep gap: when fused CUDA vote/competition falls back for a context-gain tick, `_retained_consensus_gain()` now receives the selected candidates and calls predictive vote with that awake mask. `CompetitiveColumnLayer.compete()` also accepts a candidate-local context-gain tensor, so fallback selection can preserve the bounded candidate contract instead of recomputing all predictive votes. Focused CUDA tests verify fallback vote reports `updated_column_count=4`, cached votes for the other `12` columns, and `runs_all_columns=false` on a 16-column shape. The matching 8192-column real-path run at `reports/column_scheduler_20260616/fallback-candidate-vote-scope-8192-131072-i32.json` processed `131072` tokens at `6016.247 tokens/sec`, `train_compute=0.1330716 ms/token`, `route_input_rows_scored=12/8192`, `route_output_candidate_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. This keeps the long path in the 6k-ish band; it is a fallback truth cleanup, not a new speed ceiling.

The cheap bounded-discovery probe rejected two tempting shortcuts before they reached runtime. Fixed farthest-landmark buckets and random-projection buckets were measured as evaluation-only candidate routers with explicit selector-row and steady-row accounting, then removed after rejection so they do not become a second scheduler path. Landmark256/top8/bucket128 at q16 scored `926.544` route rows on average but reached only `0.77734375` exact top-1 and `0.4296875` winner match. Per-token landmark256/top8 improved to `0.884765625` top-1 and `0.525390625` winner match while paying `256` selector rows each tick. Landmark512/top16/bucket128 reached exact top-1 `1.0`, but winner match remained `0.91015625` while scoring `1642.932` route rows plus `512` selector rows per tick. Random-projection512/top32/bucket64 q16 scored `1790.795` rows and reached only `0.255859375` exact top-1. The cleanup validation kept the promoted 32768-column path bounded and in-band at `6128.457 tokens/sec`, `route_input_rows_scored=12/32768`, and `state_transition_cached_count=32758`. The next discovery scheduler therefore cannot be a cheap landmark/projection bucket bolted beside the route bank; it must be a fused/GPU-owned graph or ANN boundary with winner-quality and 131072-token evidence.

The route-bank device-refresh slice removed the remaining host-side steady refresh from the promoted scheduler path without widening route rows. `core.fused_route_vote_cuda` now writes selected route-cache positions into the next bank and advances the two-row probe lane through a device cursor inside the select kernel. The 32768-column long run at `reports/column_scheduler_20260616/device-route-bank-refresh-32768-131072-i32.json` reported `route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`, `refresh_owner=fused_route_vote_device`, `device_refresh_count=131072`, `host_refresh_count=1`, `route_input_rows_scored=12/32768`, `state_transition_cached_count=32758`, zero graph/native/sequence failures, and `6008.953 tokens/sec` with `0.133379 ms/token` train compute under observed GPU contention. That is neutral-or-better against the immediately previous usefulness-filter run (`5886.235 tokens/sec`, `0.134212 ms/token`) but below the cleanest earlier 32768 route-bank ceiling, so it is an ownership/speed cleanup rather than a new maximum. Same-checkpoint quality gates stayed quality-incomplete: q16 probe2 top-1/winner were `0.271484375`/`0.046875`, while per-token device refresh was `0.2734375`/`0.046875`. New diagnostic gate fields now also report exact-winner-in-bank and oracle-previous winner match so top-1 recovery cannot be mistaken for a scheduler-quality pass.

Runtime Truth projection for this scheduler contract now has one service helper. Both `StatusReadModel` and `RuntimeStatusCore` call `service.column_runtime_projection.build_column_runtime_evidence`, so active status surfaces expose the same route-bank probe lane, wake-plan, cached transition, predictive cached-vote, fallback, and `runs_all_columns` fields. This is projection only: service still does not decide route candidates, wake/sleep, memory-pressure filtering, cached-vote use, or transition scope.

The text executor cleanup follows the same single-path rule. Conditional-WHILE q16 remains the promoted CUDA text sequence owner with `cuda_graph_sequence_loop_tokens` fixed at `16`, and the repeated-child native parent graph is a fixed exact-eight-token fallback. Native16/native32 repeated-child capacities, conditional-WHILE q8/q32 capacities, and partial native tail replay are no longer live config, env, or stress-benchmark options; checkpoint load coerces old `cuda_graph_native_burst_tokens` values back to `8` and old sequence-loop values back to `16`. Non-eight-token tails report explicit Python-loop fallback truth instead of compiling new parent graph shapes during measured cognition. The 8192-column 131072-token native-burst cleanup gate at `reports/column_scheduler_20260616/native-burst-capacity-cleanup-8192-131072-i32.json` stayed in the 6k-ish band at `6276.616 tokens/sec`, `train_compute=0.129187 ms/token`, with `10/8192` route rows scored, `8182` cached transition rows, no all-column state transition, `native_partial_burst_replay_enabled=false`, and zero graph/native/sequence failures. The follow-up fixed-q16 cleanup rerun at `reports/column_scheduler_20260616/sequence-loop-capacity-cleanup-8192-131072-i32-rerun.json` reached `6145.401 tokens/sec`, `train_compute=0.129706 ms/token`, with `persistent_executor_sequence_loop_capacity_fixed=true`, the same `10/8192` route rows, `8182` cached transition rows, zero graph/native/sequence failures, and no observed contention.

TurboQuant was revisited as a route-boundary candidate after the Google Research paper and community implementations were reviewed. The paper supports compressed approximate inner-product search; it does not by itself provide the Expert-Choice-style capacity contract MARULHO needs. The local `turboquant_plus` backend was a legacy approximation rather than the paper's practical Lloyd-Max/Hadamard implementation: it kept FP32 prototypes, rebuilt a compressed copy, scored every row, and lacked `routing_tensor_cache()` for the promoted CUDA route/vote graph. The CUDA audit in `reports/routing_backend_audit_20260615/` rejected it as the scheduler boundary at both 1024 and 8192 columns, so the backend, config option, and tests were removed from the live code. The follow-up cleanup also removed selectable `auto`, `faiss_hnsw`, and `exact_cosine` branches because they cannot feed the promoted graph cache and could silently create a second route path. The 131072-token cleanup gate at `reports/column_scheduler_20260616/routing-backend-consolidation-131072-i32.json` stayed in the 6k-ish band at `6152.191 tokens/sec`, `train_compute=0.132471 ms/token`, with zero graph/sequence/native failures and `state_transition_runs_all_columns=false`. A follow-up vocabulary cleanup moved live code from historical `hnsw_index`/`_hnsw_*` names to `routing_index`/`_routing_index_*`; `reports/column_scheduler_20260616/routing-index-vocabulary-cleanup-131072-i32.json` stayed in band at `6006.928 tokens/sec`, `train_compute=0.133305 ms/token`, with the same `10` active and `1014` cached transition columns. Exact `search_tensors()` pre-narrowing was also rejected as a scheduler shortcut because it still scores the full routing tensor before fused route/vote would score again. Sharded routing now always exposes the merged torch cache required by the promoted graph; the old `merge_torch_routing_shards=False` config and benchmark switches were removed, and checkpoint load drops that retired key with migration evidence. The 131072-token cleanup gate at `reports/column_scheduler_20260616/sharded-merge-cache-cleanup-131072-i32.json` reached `6169.616 tokens/sec`, `train_compute=0.131525 ms/token`, kept zero graph/sequence/native failures, and preserved `10` active plus `1014` cached transition rows. The list-returning routing API was then removed: query/detail code projects tensor results at the display edge, hot-window benchmarks no longer switch to list routing, and compiled route-candidate probes use `candidate_source=routing_index` for the tensor-native route. The matching 131072-token gate at `reports/column_scheduler_20260616/routing-list-surface-cleanup-131072-i32.json` reached `6142.710 tokens/sec`, `train_compute=0.132356 ms/token`, with zero graph/sequence/native failures. The follow-up backend-selector cleanup removed retrieval constructor selector arguments and private cache-compatibility branches, then removed `routing_index_mode` from live config while checkpoint load drops old keys with migration evidence; `reports/column_scheduler_20260616/routing-backend-selector-cleanup-8192-131072-i32.json` stayed in band at `6134.242 tokens/sec`, `train_compute=0.131117 ms/token`, `route_input_rows_scored=10/8192`, and `state_transition_cached_count=8182`. The config-field removal gate at `reports/column_scheduler_20260616/routing-index-mode-removal-8192-131072-i32.json` stayed in band at `6126.128 tokens/sec`, `train_compute=0.129371 ms/token`, zero graph/sequence/native failures, and the same `10/8192` route-row truth. The promoted route still uses the exact torch cache for seed/oracle ownership, but steady route/vote scoring is now bounded by the training-owned route candidate bank plus fixed probe lane.

`bounded_column_associative_recall` is a core helper for one column's local memory. It uses modern-Hopfield/attention-like top-k retrieval over a capped memory matrix, returns weights and a recalled vector on the caller's device, and never mutates runtime state. It is not a whole-mind memory, language model, or always-on runtime path.

## Latest Local Evidence

On 2026-06-10, live Runtime Truth after local backend restart reported `total_columns=1024`, `awake_budget=10`, `awake_count=10`, `runs_all_columns=false`, `vote_count=10`, growth gate `ready=false`, pruning/homeostasis `ready=false`, and claim boundary `column_scheduler_evidence_only_not_sparse_execution_promotion`.

The explicit service benchmark at ignored `reports/service_benchmark_column_runtime/service-benchmark.json` also captured the compact Runtime Truth evidence. That run succeeded, but CUDA/status-sidecar latency remained weak: hot-path p95 `2367.951 ms`, hot-path total `4017.128 ms`, and `/status` latency `30161.935 ms`. This means the column evidence is visible, but execution scheduling and status-scope cost still need benchmarked improvement before promotion.

On 2026-06-11, synchronized RTX 3060 microbenchmarks measured the column report before and after the bounded snapshot change:

- 4 columns: median `10.256 ms` to `1.915 ms`
- 1024 columns: median `14.246 ms` to `2.459 ms`
- 8192 columns: median `12.619 ms` to `3.495 ms`

The 1024-column snapshot is `16,384` bytes and records one CUDA-to-CPU transfer. The post-change service benchmark at `reports/service_benchmark_column_snapshot/service-benchmark.json` succeeded with `/status=126.554 ms`, `/terminus=25.08 ms`, and status-sidecar p95 `108.951 ms`. Its hot path still failed budget at p95 `2070.643 ms` and total `3169.906 ms`, so this is a localized reporting acceleration rather than proof that CUDA cognition is fast.

On 2026-06-11, synchronized RTX 3060 routing-plus-competition microbenchmarks compared dense pre-routing assembly with candidate-first execution at `k=10`:

- 256 columns: median `4.164 ms` to `2.787 ms` (`33.1%`)
- 1024 columns: median `4.038 ms` to `3.457 ms` (`14.4%`)
- 8192 columns: median `3.364 ms` to `2.994 ms` (`11.0%`)

Focused parity tests prove the routing key and winners match the prior dense behavior for learned chunking. A configured-source 1024-column service run at `reports/service_benchmark_sparse_candidate_1024/service-benchmark.json` reported Runtime Truth `alive`, observed CUDA execution, `10/1024` columns scored (`0.009766`), hot-path total `1919.574 ms`, and hot-path p95 `1157.332 ms`. Total budget passed; p95 remained above the `1000 ms` target.

On 2026-06-11, the Column Runtime report gained sampled registry evidence, stricter repeated-surprise growth gating, and bounded single-column associative recall. Focused tests passed for CPU report shape, fail-closed one-shot surprise, repeated-streak growth readiness, CUDA snapshot accounting, local top-k recall, and CUDA recall tensor placement. Current report growth remains non-mutating and requires operator review plus checkpoint evidence.

On 2026-06-11, live predictive-column state began owning and checkpointing `prediction_failure_streak`. Focused tests passed for streak increment/reset behavior, device-report exposure without scalar CUDA synchronization, checkpoint roundtrip, and Runtime Truth's five-vector column snapshot. A synthetic 12-column CPU/CUDA evidence check produced repeated failure streaks on the runtime device, `snapshot_tensor_count=5`, `snapshot_bytes=240`, and non-mutating growth readiness when `k_routing=3`.

On 2026-06-14, a bounded CUDA column-runtime export was tested and rejected as the default for current runtime sizes. It reduced a 1024-column five-tensor export from `20480` bytes to `400`, but warmed CUDA report latency stayed around `15-20 ms`; the retained latency-first full report measured median/p95 `6.3695/8.6799 ms`. The retained policy keeps CUDA status computation on CPU after one snapshot while CPU reports can still use bounded sample materialization. Before the 2026-06-15 scheduler slice, the scheduler reported `promoted_to_execution=true` only for `candidate_scoring_and_candidate_homeostasis_only`; cached votes, sleep/deep-sleep state, growth, pruning, and associative recall remained non-mutating Runtime Truth evidence.

On 2026-06-15, retained predictive voting became the next real scheduler execution effect. `MarulhoTrainer.train_step()` now retrieves the bounded candidate set before retained consensus voting, and `PredictiveColumnState.vote()` updates only those awake candidates while preserving cached consensus gain for non-awake columns. Runtime Truth projects `predictive_vote_execution` with updated-column count, cached-vote use, fallback reason, tensor device, and `runs_all_columns`. Focused tests passed for cached-vote correctness, empty/full awake-mask fallback truth, trainer-scoped vote execution, and service read-model projection without recomputation. The CPU A/B benchmark `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 2048 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-2048.json` preserved the exact winner sequence, updated `10/2048` predictive votes, cached `2038`, kept `runs_all_columns=false`, and improved complete `train_step` mean latency from `4.67866375` to `4.34479625 ms` (`7.14%`). This is a retained CPU/tensor scheduler win, not a CUDA speedup, sleep/deep-sleep mutation, or growth/pruning claim.

Later on 2026-06-15, predictive-update wake was promoted from dead-column-coupled to scheduler-owned CPU candidate scope. `candidate_predictive_update_start_tokens` defaults to `512`, so predictive state can stop touching every CPU column before structural dead-column retirement. Runtime Truth now projects `predictive_update_execution` beside `predictive_vote_execution`, and the aggregate `runs_all_columns` flag stays true whenever competitive scoring/homeostasis, predictive update, or predictive vote still runs all columns. The updated CPU benchmark `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 2048 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-2048-predictive-update-vote.json` preserved exact winner sequence, changed predictive update and vote from `2048/2048` to `10/2048`, cached `2038` predictive states/votes, reported scoped `runs_all_columns=false`, and improved mean complete `train_step` latency from `4.0616075` to `3.80747875 ms` (`6.26%`). This is CPU retained-path scheduler evidence; CUDA dense predictive update remains explicit fallback until a scoped CUDA implementation beats dense complete-runtime evidence.

The same day, candidate deep-sleep filtering and predictive-location caching closed another scheduler truth gap. `candidate_deep_sleep_filter_start_tokens` defaults to `512`; once due, the retained CPU route asks for a bounded backfill pool (`candidate_deep_sleep_backfill_factor=4`), sorts it by returned distance, filters only candidates at `dead_column_steps`, and passes the resulting awake mask into predictive location, predictive update, predictive vote, competition, and homeostasis. Runtime Truth now projects `candidate_sleep_filter_execution` plus predictive `location_update_count`/`location_cached_count`, so a hidden all-column location decay keeps `runs_all_columns=true` if it returns. The final 8192-column CPU A/B `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 8192 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-8192-deep-sleep-filter-location-update-vote.json` preserved exact winner sequence, changed predictive update, predictive location, vote, and sleep-filter output from all-column or unfiltered scope to `10/8192`, cached `8182`, reported scoped `runs_all_columns=false`, and improved mean complete `train_step` latency from `6.06891375` to `5.084465 ms` (`16.22%`). The scaling sweep `python -m marulho.evaluation.column_scheduler_benchmark --sweep-columns 2048 8192 16384 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-scaling-large-deep-sleep-filter-location-update-vote-final.json` kept predictive update, predictive location, vote, and sleep-filter output bounded at `10` for all sizes and stayed neutral-or-better, but `winner_sequence_equal=false` at `16384`; that remains a correctness gate before claiming durable total-column scaling completion.

On 2026-06-16, retained CPU state transition moved from dense hidden work to a lazy candidate-scoped execution boundary. `CompetitiveColumnLayer.process()` now materializes stale-age counters only for the routed candidates and winners, records sparse recent-spike rows, checkpoints logical `steps_since_win`, and reports cached/materialized state-transition counts. The final 8192-column CPU A/B at `reports/column_scheduler_20260616/cpu-8192-lazy-state-transition-final.json` preserved exact winners, kept predictive vote/update/location, candidate sleep filter, wake-plan count, competitive scoring, and state transition bounded at `10/8192`, reported `scoped_runs_all_columns=false`, and improved complete `train_step` from `7.8869` to `5.7537 ms` median and `9.15943125` to `8.3204845 ms` mean. The longer scaling sweep at `reports/column_scheduler_20260616/cpu-scaling-lazy-state-transition-long.json` kept awake work bounded at `10` and never ran all columns, but it did not pass neutral cost for every smaller size and the 2048 arm diverged after the deep-sleep filter changed the awake mask; treat it as boundedness evidence, not full scaling completion. The CUDA 131072-token rerun at `reports/column_scheduler_20260616/lazy-state-transition-current-131072-i32-after-marker.json` reached `6068.986 tokens/sec` (`0.164772 ms/token`) with RTX 3060 selected, conditional-WHILE q16 coverage, zero sequence/native fallbacks or failures, and `route_vote_deep_sleep_filter` active on `cuda:0`; CUDA state transition still truthfully reports dense all-column mutation, with scalar metadata only removing an extra bookkeeping fill.

A longer 8192-column CPU A/B with `400` measured samples at
`reports/column_scheduler_20260615/cpu-8192-deep-sleep-filter-location-update-vote-long.json`
kept all scoped specialist work bounded at `10/8192` and improved mean complete
`train_step` latency from `6.3080865` to `5.26286775 ms` (`16.57%`), but
`winner_sequence_equal=false` with the first divergence at measured sample
`361`. Treat the retained CPU scheduler promotion as proven for bounded awake
work and short exact-parity evidence, not yet as a long-run winner-parity
promotion. The matching 131072-token CUDA stress check preserved the promoted
conditional16 executor counters but measured `5653.175 tokens/sec` versus a
same-host clean `HEAD` control at `5807.210`, so the scheduler slice is not a
CUDA long-run throughput promotion.

The follow-up long-run parity fix made cached state wake real instead of
status-only. `CompetitiveColumnLayer` now checkpoints homeostasis step stamps,
missed fallback threshold-relaxation history, and per-column relaxation stamps;
`PredictiveColumnState` checkpoints predictive step stamps and lazily
materializes cached predictive state before retained vote/update. The corrected
8192-column CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-lazy-predictive-threshold-homeostasis-deep-sleep-filter-location-update-vote-long.json`
preserved exact winner sequence across `400` measured samples while keeping
predictive update, predictive location, vote, and sleep-filter output bounded
at `10/8192` with `runs_all_columns=false`. CPU median complete `train_step`
improved from `8.24905` to `6.85695 ms`; mean was `3.20%` slower from rare
lazy materialization bursts, so CPU mean is not a promotion claim. The
documented 131072-token CUDA stress rerun at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-lazy-scheduler.json`
reached `5867.701 tokens/sec`, `train_compute=0.141240 ms/token`, zero
sequence/native failures or fallbacks, `8190` conditional launches covering
`131040` tokens, host-truth cadence `4097/126975`, and
`velocity_environment.v1` contention `not_observed`. This is `+1.04%` versus
the same-host `5807.210` control, `-1.47%` versus the `5955.123`
completion-audit baseline, and `-4.07%` versus the `6116.646` post-promotion
top run, so the long-run path remains in the same broad 6k-ish throughput band
with scheduler correctness proven.

The explicit longer-run audit rerun at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-lazy-scheduler-long-rerun.json`
kept the same 131072-token shape and reached `5886.247 tokens/sec` with
`train_compute=0.141223 ms/token`, zero sequence/native fallbacks or failures,
`8190` conditional loop successes over `131040` tokens, host-truth cadence
`4097/126975`, and no observed contention. That is `-1.16%` versus the
`5955.123` completion-audit baseline and `-3.77%` versus the `6116.646`
post-promotion top, so the evidence still says stable 6k-ish sustained
throughput rather than exact top-run parity.

The next bounded-scaling rerun kept exact predictive wake replay but added an
exact closed-form homeostasis wake path for no-relaxation/no-clamp windows. The
CPU sweep at
`reports/column_scheduler_20260615/cpu-scaling-large-lazy-fast-homeostasis-final.json`
preserved winner parity at `2048`, `8192`, and `16384` columns while keeping
predictive update, predictive location, predictive vote, and sleep-filter
output at `10` candidates with `runs_all_columns=false`. The cost gate is still
open: scoped means were `11.3276475`, `11.382465`, and `11.70441 ms`, and
`neutral_or_better_all_sizes=false`. The longer `8192` CPU run at
`reports/column_scheduler_20260615/cpu-8192-lazy-fast-homeostasis-long.json`
also preserved winner parity and improved median complete `train_step` latency
from `7.257` to `6.4019 ms`, but mean latency was `6.27%` slower. This proves
bounded awake work and restores the old `16384` correctness gate, but it is not
yet durable total-column scaling completion.

The follow-up predictive wake cost attempt replaced per-step replay with a
closed-form vectorized materializer and initially looked promising on one
8192-column 400-sample seed, but repeated long evidence rejected it. The longer
sweep at
`reports/column_scheduler_20260615/cpu-scaling-large-lazy-exact-predictive-fast-long-sweep.json`
kept awake work bounded at `10`, but `winner_sequence_equal=false` for the
8192 arm. Repeated seed-20260616 reruns such as
`reports/column_scheduler_20260615/cpu-8192-lazy-exact-predictive-fast-seed20260616-long-rerun2.json`
also showed one-tick winner shifts near fallback-threshold boundaries. The
fresh CUDA stress run for that work tree still reached `5907.750 tokens/sec`,
but because the CPU scheduler parity gate failed, vectorized predictive wake is
retired and ordered candidate-bounded predictive replay remains the retained
correctness path.

The next scheduler-ownership slice promoted a training-owned wake-plan boundary
without changing the candidate math. `ColumnWakePlan` stores the bounded awake
IDs, wake reason, sleep reason, fallback reason, tensor device, and consumers
for retained predictive vote, competition, predictive update/location update,
and homeostasis. Runtime Truth now projects `column_wake_plan` while `service`
still only projects training/core evidence. Focused tests passed for the
trainer wake-plan mask, cached vote/update/homeostasis consumers, benchmark
boundedness, and service projection without recomputing scheduler decisions.
The 8192-column CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-wake-plan-scheduler-slots.json`
preserved exact winners and bounded predictive vote, predictive update,
predictive location, candidate sleep filtering, and wake-plan awake count at
`10/8192` with `runs_all_columns=false`, but scoped mean latency was
`12.24272625 ms` versus `7.4809125 ms`; this is an ownership/truth promotion,
not a CPU speed claim. The implementation removed eager per-tick legacy report
dict materialization and used a slotted wake-plan object before the final
evidence run.

The follow-up Runtime Truth projection audit tightened that ownership boundary:
live `column_runtime` now projects the training-owned `ColumnWakePlan` awake
IDs, wake/sleep/fallback reasons, and consumer list instead of recomputing a
second report-local top-k mask. The standalone report helper may still use the
top-k scheduler score only when no execution plan is supplied. Per-column report
rows now expose role, state, prediction, surprise, usefulness, estimated cost,
memory-pressure truth, cached-vote state, and wake/sleep reason from
training-owned metabolism state. Cached usefulness is now a checkpointed
`ColumnMetabolismState` signal, updated only for awake candidates from
predictive confidence/error, win-rate, and estimated cost; report-derived
usefulness is a Runtime Truth fallback only. Focused tests prove service remains
read-only projection and does not own scheduler decisions. The current benchmark
evidence keeps awake work bounded but does not prove complete-runtime neutrality,
so this is a truth-boundary correction rather than a scheduler speed promotion.

The column-metabolism cleanup promoted memory pressure from report placeholder
to checkpointed training-owned state. `ColumnMetabolismState` tracks estimated
cost and memory pressure for all columns but updates only the wake-plan IDs.
The retained CPU candidate route can drop high-pressure retrieved candidates
using cached candidate-local pressure values and a bounded backfill pool. Tests
prove high-pressure candidates are skipped, service projection remains
read-only, checkpoint restore preserves metabolism state, and no all-column
pressure scan is used to decide sleep. CPU benchmarks at
`reports/column_scheduler_20260616/cpu-8192-column-metabolism.json` and
`reports/column_scheduler_20260616/cpu-scaling-column-metabolism.json` kept
awake/metabolism work at `10` columns for `1024`, `8192`, and `32768` total
columns, but did not pass neutral CPU cost. The long CUDA stress gate at
`reports/column_scheduler_20260616/column-metabolism-current-131072-i32.json`
processed `131072` tokens at `5960.035 tokens/sec` with
`train_compute=0.135423 ms/token`, `131072` route-vote executions, the
route-vote deep-sleep filter active on `cuda:0`, `8190` q16 sequence-loop
successes over `131040` tokens, zero graph/sequence/native failures, and
observed GPU contention. Treat this as stable 6k-ish hot-path evidence, not a
new CUDA speed ceiling.

The follow-up CUDA pressure slice moved the memory-pressure gate into the same
route-vote owner instead of filtering after selected candidates. When
`ColumnMetabolismState.last_memory_pressure_source` proves cached pressure
evidence exists, `core.fused_route_vote_cuda` masks high-pressure rows before
top-k and winner selection, and the `ColumnWakePlan` reports
`candidate_memory_pressure_filter_route_vote` or the combined
`candidate_deep_sleep_memory_pressure_filter_route_vote` mode. When pressure
evidence is absent, the pressure gate stays disabled to avoid a no-op all-route
pressure read. The current long rerun at
`reports/column_scheduler_20260616/route-vote-pressure-filter-current-131072-i32-rerun.json`
processed `131072` tokens at `5947.863 tokens/sec` with
`train_compute=0.136041 ms/token`, zero graph/sequence/native failures, no
observed contention, and truthful pressure fallback:
`memory_pressure_applied=false` because only `6` pressure-eligible rows remained
for `k=10`.

The follow-up usefulness slice moved cached column usefulness into the same
route-vote owner. `ColumnMetabolismState.usefulness` is checkpoint-backed
training/core state, updated only for the awake mask, and the CUDA control packet
enables the usefulness gate only after `last_usefulness_source` proves cached
evidence exists. `core.fused_route_vote_cuda` now masks low-usefulness route rows
before top-k candidate and winner selection, and Runtime Truth reports
`filtered_low_usefulness_count`, `low_usefulness_count`,
`usefulness_eligible_route_count`, `usefulness_threshold`,
`usefulness_source`, and fallback reasons without service ownership. Focused
CUDA tests prove low-usefulness top route candidates are skipped before vote,
the trainer wake plan reports `candidate_usefulness_filter_route_vote`, cached
state survives checkpoint restore, and service projection only maps the
training/core packet. The 32768-column longer gate at
`reports/column_scheduler_20260616/usefulness-scheduler-32768-131072-i32.json`
processed `131072` tokens at `5886.235 tokens/sec` with
`train_compute=0.134212 ms/token`, `route_input_rows_scored=12/32768`,
`route_output_candidate_count=10`, `state_transition_cached_count=32758`,
`candidate_predictive_transition_cached_count=32758`, zero graph/sequence/native
failures, no observed contention, and route filter truth
`filtered_deep_sleep_count=2`, `filtered_memory_pressure_count=0`,
`filtered_low_usefulness_count=0`, `usefulness_applied=true`. Compared with the
same-shape previous fixed-count gate
`promoted-scheduler-32768-131072-i32-truth-counts-fixed.json`
(`6163.265 tokens/sec`, `0.132789 ms/token`, GPU contention observed), this is
still in the sustained 6k-ish band but not a new speed ceiling.

The sparse CUDA state-transition slice then removed another fake-sleep risk from
the real graph/burst path. The transition kernel now updates physical
`steps_since_win`, per-row state stamps, recent-spike active IDs, assembly active
winner, thresholds, and win-rate EMA only for the routed candidate set and
winner when the graph is running `candidate_subset`; non-awake columns keep
cached logical age through the step-counter/all-materialized-step contract. The
131072-token longer run at
`reports/column_scheduler_20260616/sparse-cuda-state-transition-candidate-homeostasis-131072-i32.json`
processed `131072` tokens at `6007.582 tokens/sec` with
`train_compute=0.132374 ms/token`, no observed contention, zero graph/sequence
native failures, and Runtime Truth `state_transition_column_count=10`,
`state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`.
Compared with the pressure-filter baseline at
`reports/column_scheduler_20260616/route-vote-pressure-filter-current-131072-i32-rerun.json`
(`5947.863 tokens/sec`, `0.136041 ms/token` train compute), this is neutral or
slightly faster while making the cached-sleep state truth real.

The fused CUDA follow-up keeps that same wake-plan truth but moves one more
piece of predictive metabolism below the graph-owned transition boundary. The
promoted fused in-place path updates prediction error, failure streak,
confidence, location/velocity, prediction weights, and
`predictive_last_update_step` only for the awake candidate set inside the
in-place/graph transition. Non-candidate predictive rows stay cached and stale
by construction, and Runtime Truth reports `candidate_predictive_transition`
mode, active/fallback truth, execution count, last-transition cached-row count,
and cumulative cached row-skips. The retired
`cuda_candidate_predictive_transition_mode` config switch was removed rather
than kept as a dense-retained compatibility path. Focused CUDA tests prove the
direct kernel branch, non-graph runtime path, and CUDA graph path preserve
candidate-row state and predictive step stamps against the retained semantics.
The promoted 131072-token conditional-WHILE stress gate at
`reports/column_scheduler_20260615/promoted-fused-candidate-predictive-131072-i32.json`
returned to the documented sustained band at `6141.078 tokens/sec` with
`0.126682 ms/token` train compute, `0.006629 ms/token` prepare,
`0.004710 ms/token` finalize, no observed contention, zero sequence/native
failures or fallbacks, `130816` fused candidate predictive updates, and
`132647424` cumulative cached predictive row-skips.

The next scheduler slice fused deep-sleep route filtering into the route-vote
owner instead of leaving CUDA sleep as a fallback label. The final longer stress
gate at
`reports/column_scheduler_20260615/route-vote-sleep-filter-131072-i32-sync-cadence.json`
processed `131072` tokens at `6135.026 tokens/sec` with
`train_compute=0.133995 ms/token`, no observed CPU/GPU contention, `8190`
conditional sequence-loop launches covering `131040` tokens, and zero
sequence/native fallbacks or failures. Runtime Truth reported
`route_vote_deep_sleep_filter.v1` enabled on `cuda:0`, filtering `1014` of
`1024` route rows down to `10` eligible route candidates, no fallback reason,
one control update, and `129` state syncs. This is effectively the same
6k-ish throughput band as the `6141.078 tokens/sec` fused predictive baseline,
but train-compute cost remains higher and should stay on the next speed queue.

The corresponding longer CUDA stress check at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-wake-plan-scheduler-slots.json`
reached `5822.624 tokens/sec`, with `train_compute=0.142080 ms/token`,
`prepare_training=0.006576 ms/token`, and `finalize_total=0.005888 ms/token`.
It preserved RTX 3060 execution, no observed contention, `8190` conditional
sequence-loop successes over `131040` tokens, zero sequence/native fallbacks or
failures, and host-truth cadence `4097/126975`. The preceding eager-report
wake-plan run reached `5808.990 tokens/sec` with
`train_compute=0.142889 ms/token`; the lazy/slotted version is slightly faster,
but the throughput answer remains broad 6k-ish sustained runtime rather than
exact parity with the historical `6116.646` top run.

The follow-up candidate-transition cleanup fused the retained CPU candidate
predictive split path inside `PredictiveColumnState` while leaving CUDA dense
predictive execution unchanged. Focused tests proved state parity with the old
split sequence and that cached predictive materialization evidence remains
visible after wake. The 8192-column CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-fused-candidate-transition.json`
preserved exact winners and bounded predictive vote/update/location,
candidate-sleep filtering, and wake-plan awake count at `10/8192`. Scoped mean
latency improved versus the previous scoped wake-plan baseline
(`12.24272625` to `10.637575 ms`), but it still did not beat the same-run
all-column mean (`9.8858125 ms`), so this is a local retained CPU cleanup, not
a complete scheduler speed promotion. The matching 131072-token CUDA stress at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-fused-candidate-transition.json`
reached `5889.241 tokens/sec`, `train_compute=0.140840 ms/token`,
`prepare_training=0.006423 ms/token`, and
`finalize_total=0.005995 ms/token`, with RTX 3060 CUDA selected, no observed
contention, `8190` conditional sequence-loop successes over `131040` tokens,
zero sequence/native fallbacks or failures, and host-truth cadence `4097/126975`.
This keeps the long path in the same 6k-ish band while making the before/after
ms tradeoff explicit.

The next retained scheduler cleanup removed two bounded but avoidable CPU costs
without changing the CUDA dense predictive boundary. Before any column can reach
`dead_column_steps`, the candidate deep-sleep filter now requests only the target
awake budget instead of a backfill pool and reports
`candidate_deep_sleep_filter_no_column_can_be_deep_sleep_yet`. Candidate
predictive vote/update share one materialization per tick, and checkpoint restore
recomputes predictive cached-state truth from step stamps. Focused tests passed
for the age gate, materialization reuse, completed-candidate no-op evidence,
checkpoint restore wake replay, benchmark fallback truth, and service ownership
guards. The CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-age-gate-single-materialization-completed-cache.json`
kept exact winners and bounded scoped work at `10/8192` with
`runs_all_columns=false`, but scoped mean complete `train_step` was
`11.2669375 ms` versus `6.57934625 ms` for the all-column arm. The CPU scaling
sweep at
`reports/column_scheduler_20260615/cpu-scaling-age-gate-single-materialization-completed-cache.json`
kept awake work bounded at `10` for `512`, `2048`, and `8192` columns, while
`neutral_or_better_all_sizes=false`. The CUDA scheduler A/B at
`reports/column_scheduler_20260615/cuda-8192-age-gate-single-materialization-completed-cache.json`
truthfully reported `runs_all_columns=true` because predictive update stayed
dense with fallback reason `cuda_sparse_prediction_update_launch_bound_dense_retained`;
scoped mean was `17.98001875 ms` versus `14.55886625 ms`. The longer CUDA stress
gate at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-age-gate-materialization-cache.json`
remained in the same sustained band at `5909.600 tokens/sec`, with
`train_compute=0.140558 ms/token`, `prepare_training=0.006582 ms/token`,
`finalize_total=0.005748 ms/token`, zero sequence/native failures, and no
observed contention. This is a bounded-truth and CUDA-throughput-stability slice,
not a complete scheduler speed promotion.

Also on 2026-06-15, the CUDA text path promoted the conditional-WHILE q16 sequence executor below the trainer-owned burst boundary. It does not change column scheduling policy, but it matters for Column Runtime evidence because the same sequential SNN column state is now advanced by a larger native CUDA Graph parent while Runtime Truth reports executor identity, token coverage, fallback/failure counts, host-truth cadence, startup compile/capture cost, and separate repeated-child versus sequence-loop capacities. The retained repeated-child native replay remains the exact native8 internal fallback path when conditional construction cannot run before mutation.

On 2026-06-11, candidate-scoped competitive homeostasis became the first sleep-aware execution effect. The trainer kept early learning all-column, then after `dead_column_steps` passed the retrieved candidate set into `CompetitiveColumnLayer.process()` so win-rate/threshold homeostasis updated only active candidates. Focused tests passed for frozen non-candidate homeostasis, delayed trainer promotion, learned-chunk diversity, and Runtime Truth execution fields. An 8192-column CPU/CUDA microbenchmark with 10 candidates measured all-column versus scoped process medians of `1.28555` to `1.08545 ms` on CPU and `5.06735` to `4.93965 ms` on CUDA; p95 changed from `4.3084` to `2.3779 ms` on CPU and `10.6108` to `9.195 ms` on CUDA.

On 2026-06-12, candidate-scoped homeostasis split from the structural dead-column threshold. `candidate_homeostasis_start_tokens` defaults to `512`, so threshold and win-rate maintenance can wake only retrieved candidates before columns are eligible for dead-column maintenance. Stale counters and spike windows still update every tick, and `force_revive_dead_columns()` remains an explicit maintenance path. A fused RTX 3060 hot-window A/B measured forced all-column homeostasis at `47.4682 ticks/sec` with `1024/1024` updates versus default candidate homeostasis at `64.8535 ticks/sec` with `10/1024` updates.

On 2026-06-11, candidate-scoped predictive updates became a CPU-only promoted execution effect. `PredictiveColumnState` preserves full-vector state but updates prediction error, confidence, failure streak, and high-prediction non-winner decay only for active candidates after `dead_column_steps`; non-candidate predictive state remains cached. Focused tests passed for frozen non-candidate state, delayed trainer promotion, CUDA fallback reporting, and device-report telemetry. Benchmarks with 8192 columns and 10 candidates measured predictive update median/p95 from `2.91545/55.6886 ms` all-column to `1.042/1.878 ms` scoped on CPU. CUDA tensors stayed on `cuda:0`, but scoped median was slower (`4.86425` to `6.9281 ms` at 8192 columns), so CUDA retains dense predictive updates and reports the fallback reason.

On 2026-06-12, lightweight input-synapse plasticity became conditional on a nonzero live contribution. The revision-960 checkpoint uses `input_weight_blend=0.0`, so winner-row rewrites could not affect routing or assembly output. MARULHO now skips only that lightweight update while preserving prototype learning, spike evidence, stale counters, and homeostasis; local STDP remains active regardless of blend because it owns wider neural state. Runtime Truth reports the input-plasticity mode and update/skip counters. Two uncontended 256-token RTX 3060 confirmations measured `42.8296` and `41.9188 tokens/sec`, median `21.9380` and `22.52325 ms`, p95 `36.1204` and `38.4941 ms`, with `21.1924/54.0 MB` allocated/reserved VRAM. A direct pre-change/skip probe showed roughly `9%` throughput improvement in the controlled pair, while run variance prevents treating that percentage as a universal ceiling.

On 2026-06-11, adaptive context plasticity moved to a four-token cadence while context state remains continuous on every token. This is metabolism rather than column scheduling: the dense context projection is still present, but three of four routine dense Hebbian weight updates are skipped. Context device reports expose state-update count, plasticity-update count, and whether the latest observation changed weights.

On 2026-06-11, hypercube hub topology refresh was removed from the always-on binding path. `bind()` now accumulates hub activation evidence only; an explicit reason-bearing maintenance helper owns adjacency refresh. Focused tests prove repeated binding leaves neighbor IDs, degrees, and structural mutation ledgers unchanged. A 1024-column synchronized CUDA microbenchmark improved bind median/p95 from `11.65755/15.2508 ms` with legacy-equivalent per-bind refresh to `6.6278/9.0511 ms` without it. CPU timing was noisy and did not show a speedup, so the performance claim is CUDA-local.

The explicit helper is now promoted through the existing structural mutation transaction. The design and preflight bind the binding-hub target, core method, operator reason, edge budget, revision, rollback checkpoint, and checkpointed candidate provenance. The executor verifies the serialized binding state, applies one refresh, reports exact growth/prune and edge deltas, and rolls back on no-op, over-budget, tampered, or unverified commits. Blocked/rejected/retired outcomes return candidate provenance, rollback artifact, and a Structural Candidate Tombstone instead of forgetting the candidate. The previous executor path that consumed binding evidence but mutated `ConceptStore` capacity was removed.

The preceding trial-design stage derives candidate source columns from live repeated prediction failures rather than caller-authored IDs. It proposes exact sparse edges but does not apply them. The explicit checkpoint-clone binding-growth evaluator can test those edges against prediction, spike health, Runtime Truth, and metabolism without touching the always-on runtime; only successful evidence should advance toward the operator transaction.

On 2026-06-11, Hypercube Binding became the first larger Subcortex specialist with an explicit event-driven wake policy. While learned binding usage is absent, the trainer runs a probe every four tokens and preserves cached state on the other ticks; active or checkpoint-restored binding runs every tick. Runtime Truth exposes `runtime_active`, bind/probe count, idle-skip count, last execution mode, interval, and CUDA tensor placement.

The repeatable runner at `marulho.evaluation.binding_wake_benchmark` compared interval 1 with interval 4 on the same 1024-column checkpoint and synchronized RTX 3060 inputs. Across 120 samples per arm, median latency improved from `32.2069` to `29.5535 ms` (`8.24%`), p95 from `46.3573` to `42.7967 ms` (`7.68%`), and mean by `5.07%`; allocated/reserved VRAM stayed `20.4585/48.0 MB`. An isolated profiler trace measured 70 CUDA kernels for an idle probe and zero for a cached skip. A post-restart live tick reported Runtime Truth `alive`, binding tensors on `cuda:0`, 3 probes, 11 cached skips, and a 14-token tick in `2521.655 ms`.

On 2026-06-11, Cross-Modal Grounding adopted the same specialist wake rule for text-only ticks. When no visual or audio spikes are accepted, the trainer updates text grounding every four tokens and records cached-idle skips on the others; accepted sensory evidence wakes text updates every tick. Runtime Truth exposes text update count, idle-skip count, execution mode, interval, and tensor placement. A live `/feed` text-only check advanced the CUDA runtime to revision 960 and reported 2 cross-modal text updates, 8 cached skips, and cross-modal tensors on `cuda:0`.

The repeatable runner at `marulho.evaluation.cross_modal_wake_benchmark` compared interval 1 with interval 4 on the revision-209 live checkpoint. Across 120 text-only samples per arm, median latency improved from `57.59725` to `51.5753 ms` (`10.46%`) and mean from `59.4546` to `54.8716 ms` (`7.71%`). P95 regressed from `82.0887` to `83.8702 ms`, so this is not a tail-latency claim. Isolated profiling measured 49 CUDA kernels and 108 ATen ops for one text update versus 2 CUDA kernels and 3 ATen ops for one cached skip.

On 2026-06-17, recent replay tag and anchor setup moved to a bounded CPU
recency index. `tag_recent_entries(...)` no longer walks all archival
timestamps; it emits `bounded_recent_memory_tag.v1` over a capped
`bounded_recent_memory_window.v1`. `capture_recent_memory_anchors(...)` uses
the same index with `require_bucket=true`, emits
`bounded_recent_anchor_capture.v1`, and only creates column anchors from
bucketed entries. The synthetic replay report
`reports/bounded_replay_window_20260617/synthetic-recent-anchor-window.json`
kept recall/prototype gates passing while tag and anchor setup used
`candidate_window_limit=256`, no global scans, CPU archival storage, and
`runs_live_tick=false`. The 65536-column hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json`
processed `262144` tokens at `6228.243 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native/sequence
failures, flat `1846 MiB` GPU memory, and no observed contention.

The same replay cleanup removed the old public full-buffer replay-priority
helper. Query/readout and replay tests now call `replay_scores_for_indices(...)`
with selected candidate indices, so replay priority remains available only
after a bounded window exists. The synthetic helper-retirement report
`reports/bounded_replay_window_20260617/synthetic-replay-score-helper-retired.json`
kept recall/prototype gates passing, and the 65536-column hot-path check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json`
processed `262144` tokens at `6211.859 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native/sequence
failures, flat `1852 MiB` GPU memory, and no observed contention.

The follow-up cleanup removes the remaining public full-buffer score tensor
helpers (`maintenance_scores(...)`, `consolidation_scores(...)`,
`repair_scores(...)`, `fragility_scores(...)`, plus unused capture/tag/PRP tensor
builders). Production replay selection now scores only bounded candidate
indices; the later runtime hook cleanup removes the private global scoring
branch too, leaving retired full-scan comparisons in benchmark-local baselines
only. The synthetic report
`reports/bounded_replay_window_20260617/synthetic-score-tensor-helpers-retired.json`
kept recall/prototype gates passing, and the accepted hot-path report
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json`
processed `262144` tokens at `6151.952 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native/sequence
failures, flat `1805 MiB` GPU memory, and no observed contention.

On 2026-06-18, runtime concept observation moved its memory lookup boundary
out of `OperatorInteractionRuntime` and into `DualMemoryStore`. The service no
longer direct-reads `slow_routing_keys`, `slow_texts`, `slow_raw_windows`,
`slow_importance`, `slow_capture_tag`, or `slow_consolidation_level` to build
concept matches. `DualMemoryStore.resolve_runtime_concept_memory_matches(...)`
now accepts only trainer-emitted `memory_index` evidence, caps each observation
batch, caches duplicate payload reads, and records
`bounded_runtime_concept_memory_lookup.v1` in device reports, summaries, and
checkpoints. This is cadenced live runtime observation, not sleep replay: the
report truthfully uses `runs_live_tick=true` while keeping
`runs_every_token=false`, `global_candidate_scan=false`,
`global_score_scan=false`, CPU archival/score placement, and
`language_reasoning=false`.

The bounded lookup benchmark
`reports/bounded_replay_window_20260618/runtime-concept-memory-lookup-bounded.json`
preserved selected-index parity across `512` observations on a `65536`-entry
archive, reduced raw payload reads from `512` to `64` with `448` cache hits,
and improved mean lookup latency from `47.156 ms` to `6.380 ms`. The matching
`524288`-token active-pressure protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-runtime-concept-memory-lookup.json`
processed `6237.075 tokens/sec`, with
`concept_observation=0.000474 ms/token`, bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`1809->1861 MiB`, and zero graph/native/sequence failures.

On 2026-06-18, context comparison stopped using the report-dropping query
memory wrapper. The old `query_runner.memory_matches(...)` compatibility
surface is deleted; `build_context_comparison(...)` now calls
`memory_matches_with_report(...)` once per compared context, shares one
returned replay-entry payload cache, exposes each per-context report, and
returns `bounded_context_comparison_memory_match.v1` at the comparison root.
This is explicit slow readout, not column-runtime mutation: the report carries
`runs_live_tick=false`, `runs_every_token=false`,
`global_candidate_scan=false`, `global_score_scan=false`, CPU archival/score
placement, and `language_reasoning=false`.

The bounded comparison benchmark
`reports/bounded_replay_window_20260618/context-memory-match-bounded.json`
preserved selected-index parity for both contexts (`quality.min=1.0`) over a
`65536`-entry synthetic archive, reduced raw payload reads from `16` to `8`
with `8` cache hits, and reduced mean latency from `71.927 ms` to
`70.550 ms`. The paired `524288`-token active-pressure protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-context-memory-match.json`
processed `6065.987 tokens/sec`, with `train_compute=0.135179 ms/token`,
`concept_observation=0.000474 ms/token`, bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, GPU memory
`1839->1845 MiB`, and zero graph/native/sequence failures.

On 2026-06-18, SFA correction sampling became a reported selected-window
operator. Deep sleep with abstraction no longer calls the deleted
`sample_for_sfa(...)` helper; it calls
`DualMemoryStore.sample_for_sfa_with_report(...)`, embeds
`bounded_sfa_sample.v1` in the sleep replay report, and persists the latest
sampler report through the memory-store summary/checkpoint path. The report
records selected replay-window indices, sampled indices, sample count, CPU
archival/sample placement, no global candidate scan, `runs_live_tick=false`,
`runs_every_token=false`, and `language_reasoning=false`. The list-only
`sample_replay_indices(...)` helper and source-bank `bank_memory_matches(...)`
wrapper are removed so bounded replay/source-bank callers cannot silently drop
their reports. The SFA benchmark
`reports/bounded_replay_window_20260618/sfa-sample-bounded-window.json` used a
`65536`-entry archive, `192` selected replay-window candidates, and `64` SFA
samples; selected-window sample purity improved from `0.00439453125` to `1.0`
and mean latency improved from `1.451 ms` to `0.656 ms` (`2.210x`) against the
retired full-buffer sampler. The accepted `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-reported-sfa-sampler-noprofile-rerun2.json`
stayed in the sustained band at `6127.490 tokens/sec`, with
last-tick `train_compute=17.738 ms` (`0.138579 ms/token` over the 128-token
tick), bounded `12/65536` route rows, `65526` cached transition rows, no
observed contention, GPU memory `1840->1861 MiB`, and zero
graph/native/sequence failures.

On 2026-06-18, explicit query memory episodes became a reported slow-path
readout. `build_memory_episodes(...)` is removed; query result construction now
uses `build_memory_episodes_with_report(...)` and returns
`bounded_query_memory_episode_readout.v1`. The report keeps selected-neighbor
text stitching bounded to already returned memory matches, records the neighbor
radius and payload budget, uses CPU archival/readout placement, and states no
global scans, no live tick, no every-token work, and no language reasoning. The
episode benchmark
`reports/bounded_replay_window_20260618/query-episode-readout-bounded.json`
recovered the target top episode from four selected fragments over a
`65536`-entry archive, at `0.936 ms` mean readout latency versus `0.490 ms` for
fragment-only readout. The paired `524288`-token protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-query-episode-readout.json`
stayed in band at `6219.926 tokens/sec`, with `train_compute=0.130647 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, no observed
contention, GPU memory `1810->1811 MiB`, and zero graph/native/sequence
failures.

On 2026-06-18, explicit feed gained bounded source-episode admission. This
does not run inside the column live tick. `feed_text(...)` runs the normal
training stream, then a slow-path source admission step records
`bounded_feed_source_episode_admission.v1` with a `32`-episode and `240`-char
payload budget, CPU archival storage, no global memory scan, no every-token
slow-memory admission, and no language reasoning. Query readout now respects
the source-admission boundary: complete admitted source episodes are returned
as whole evidence, zero-support episodes are filtered when supported evidence
exists, and cadence-fragment neighbor stitching requires direct character
overlap and cannot cross into source-episode entries. The quality benchmark
`reports/bounded_replay_window_20260618/source-episode-admission-bounded.json`
raised simple-animals grounded query pass rate from `0.25` to `1.0` while
keeping all archival tensors on CPU. The paired hot-path run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-episode-admission.json`
processed `524288` tokens at `6702.362 tokens/sec`, with
`train_compute=0.121727 ms/token`, bounded route scoring at `12/65536`,
`65526` cached transition rows, no observed contention, flat `1808 MiB` GPU
memory, and zero graph/native/sequence failures.

The v2 source-admission follow-up retires the dense post-selection assembly
call from that slow path. `bounded_feed_source_episode_admission.v1` now reports
`assembly_policy=bounded_offline_competition_winner_assembly` and
`dense_source_admission_assembly_retired=true`: one bounded offline competition
returns the source winner, assembly, and routing key, then archival payloads are
stored back on CPU. The v2 quality report
`reports/bounded_replay_window_20260618/source-episode-admission-bounded-v2.json`
kept simple-animals pass rate at `0.25 -> 1.0`, admitted `5/5` selected
episodes, measured `2725.253 ms` admission latency, and kept global
candidate/score scans false. The paired protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-episode-admission-v2.json`
processed `524288` tokens at `6412.209 tokens/sec`, with
`train_compute=0.126270 ms/token`, bounded route scoring at `12/65536`,
`65526` cached transition rows, zero runtime failures, and GPU memory
`1812->1866 MiB` under observed GPU-side contention.

On 2026-06-18, service replay planning became input-bounded instead of only
output-bounded. `build_replay_plan(...)` now emits
`bounded_replay_plan_source_window.v1` with source limits, source counts,
window counts, truncation flags, feedback-index counts, candidate counts,
`runs_live_tick=false`, and CPU-only device placement. The planner takes a
timestamp-oriented `64`-row window from each runtime-history stream, indexes up
to `128` recent feedback rows, and creates at most `32` feedback-target stubs
so a high-signal contradicted old target can still be selected without scanning
all runtime history. `RuntimeEvidenceReporter._replay_plan_summary(...)` preserves
that source-window report for status/export surfaces. The benchmark
`reports/bounded_replay_window_20260618/replay-plan-source-window-bounded.json`
kept the old contradicted target `ep-42` as the top candidate over synthetic
`20000`-row episode/action/prediction histories, reduced mean planner latency
from `6860.919 ms` to `14.684 ms`, used `0.519 MiB` traced peak Python
allocation, and used no CUDA/VRAM. The paired long protection run
`reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-replay-plan-source-window.json`
processed `524288` tokens at `6344.404 tokens/sec`, kept
`train_compute=0.128679 ms/token`, bounded route scoring at `12/65536`, cached
`65526` transition rows, reported no observed contention, kept GPU memory flat
at `1799 MiB`, and had zero graph/native/sequence failures.

## Next Gate

The in-place CUDA/Triton transition is now the promoted production executor owned by `MarulhoTrainer`. Startup compiles the all-column and routed-candidate shapes without launching the mutating kernel. Unsupported configurations fall back before mutation; failures after launch fail closed. Runtime Truth exposes requested/resolved mode, warmup, candidate shapes, device, execution/failure counts, fallback, and policy.

Production-backed complete hot-window runs reached `80.59` and `110.36 ticks/sec` versus retained observations of `70.77` and `51.56 ticks/sec`. A synthetic visual/audio gate passed at `42.81` versus `38.78 ticks/sec` with exact winners and bit-exact cross-modal state. Empty-cache compile-only startup took `80.75 s`, while a populated cache reduced startup to `0.35 s`.

Stage profiling then separated the live loop. Concept observation consumed `5490.65 ms` of a `7292.26 ms` 12-token tick; normalized-centroid caching and bounded source-window sampling improved same-process live throughput to `7.84-8.44 tokens/sec`. Scheduling the existing remote refill worker after chunk consumption produced a warm 64-token tick with `0.04 ms` source collection and `7.97 tokens/sec`, proving source waiting can overlap CUDA work.

The next velocity gate is now inside the remaining `MarulhoTrainer.train_step` stages, which consumed `6601.45 ms` of that 64-token tick. The design target remains persistent device scheduling, event-driven specialist wake, bounded batching, and fusion where synchronized profiling proves launch or host/device pressure. Real camera/microphone grounding and thousands of full cognitive ticks per second remain unproven.

An explicit profiler slow path found about 112 CUDA launches, 40 async copies, and 16 stream synchronizations per encoded tick. With the in-place transition active, the transition itself was about `2.2 ms/tick`; dense predictive voting plus candidate competition, routing-key projection, and tensor routing-index retrieval formed the larger remaining routing cluster at roughly `11 ms/tick`.

Candidate-scoped voting and a separately compiled dense vote both failed complete-hot-window A/B despite reducing isolated arithmetic. MARULHO therefore retains eager dense vote semantics, removes the unused persistent `hypothesis` tensor, and targets a broader fused/device-resident routing boundary. The retirement removed observed launches/copies without changing the vote result because no algorithm consumed the tensor.

That broader boundary is now implemented as Fused Text Route Vote and consumed by the default `cuda_graph_text` route. On the 1024-column RTX 3060 checkpoint, the two-launch exact-cache cluster matched all candidates, winners, and positive/silent decisions across 128 recurrent keys and improved isolated routing/vote from `415.16` to `1716.56 ticks/sec`. Production-owned reversed complete text/idle runs averaged `92.00` versus `66.80 ticks/sec` (`1.377x`). A serialized checkpoint then executed `144/144` fused text ticks with zero failures and nine cache refreshes at `82.40 ticks/sec`.

The specialization is deliberately modality-aware. A global sensory experiment preserved exact grounding but initially regressed; the promoted lifecycle executed zero fused routes on sensory ticks, reported `272/272` fallbacks in the reversed grounded gate, preserved every winner and cross-modal tensor, and passed the declared 0.90 no-regression floor. The default is now `cuda_graph_text` for eligible CUDA text ticks, with visual/audio ticks falling back before graph pre-routing.

Skipping repeated routing normalization was also rejected because small floating-point changes diverged sequential winners and predictive locations. A narrower exact cleanup was promoted instead: the transition's already-materialized CPU winner ID now feeds routing-index buffering, avoiding one duplicate CUDA-to-CPU transfer and synchronization per tick without changing routing or learning state.

The promoted default route is now `cuda_graph_text`. It captures production input normalization/projection, exact reconstruction distance, fused route/vote, and the in-place transition with fixed tensor addresses. A controlled 128-tick comparison was bit-exact across the sequential competitive, predictive, spike, input, and projection state. Three fresh-process hot-window arms averaged `264.46 ticks/sec` versus `176.24` for the fused path (`1.501x`), with graph median latency between `2.806` and `3.105 ms`. Runtime Truth on a real source tick recorded 24 graph replays and zero failures, but the 24-token tick still took `1240.473 ms`; remaining trainer/source orchestration is therefore the next velocity gate.

The replay-priority review surface now preserves the same live-tick boundary as the column scheduler. `snn_replay_consolidation_priority_queue.v1` verifies a bounded source window (`16` recent Replay Controller contexts plus up to `16` readout-target context IDs) through a controller-owned ID index and emits `bounded_snn_replay_priority_source_window.v1`. Due-cycle Runtime Truth carries `queue_source_window_policy`, source/verified context counts, and `queue_global_candidate_scan=false`; the queue itself reports CPU archival/score placement, no raw replay text, no hidden language reasoning, `runs_live_tick=false`, and `gpu_used=false`. The benchmark `reports/bounded_replay_window_20260618/snn-replay-priority-source-window.json` selected an old readout-targeted context outside the recent window while verifying `17/64` retained contexts, and the paired 65536-column `524288`-token run stayed in band at `6298.310 tokens/sec` with `train_compute=0.129349 ms/token`, bounded `12/65536` route rows, `65526` cached transition rows, and zero graph/native/sequence failures.

The upstream readout-priority ledger now preserves that boundary before the
Replay Controller queue is even called. `snn_language_readout_replay_priority.v1`
scores only a `32`-event recent CPU source window and emits
`bounded_snn_readout_replay_priority_source_window.v1` with source/returned
candidate counts, CPU archival/score placement, no raw text payload, no hidden
language reasoning, `runs_live_tick=false`, `runs_every_token=false`, and
`gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-readout-replay-priority-source-window.json`
matched the diagnostic full-retained scorer's top high-signal readout while
scoring `32/2048` retained events, reducing mean priority latency from
`51.002932 ms` to `1.424948 ms`. The paired 65536-column `524288`-token run
stayed in band at `6284.379 tokens/sec`, with `train_compute=0.129905
ms/token`, bounded `12/65536` route rows, `65526` cached transition rows, GPU
memory `1852->1858 MiB`, no observed contention, and zero graph/native/sequence
failures.

Emission-review replay policy now keeps reviewed display text out of replay
selection and applies the same bounded source rule before replay-context design.
`snn_language_readout_emission_replay_evaluation_policy.v1` and its design
verifier use `bounded_snn_emission_review_replay_policy_source_window.v1`, capped
to `16` recent reviewed emissions and `16` recent internal readout events. The
policy emits only hash-bound candidates, verifies selected seeds against the same
bounded readout window, and reports CPU archival/score placement, no raw reviewed
text payload, no hidden language reasoning, `runs_live_tick=false`,
`runs_every_token=false`, and `gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-emission-review-replay-policy-source-window.json`
matched the diagnostic full-retained policy/design top candidate while checking
`32` source events instead of `4096` retained review/readout records, reducing
mean policy+design latency from `166.924984 ms` to `2.476164 ms`. After rejecting
one contended hot-path profile, the clean 65536-column `524288`-token profiled
rerun stayed in band at `6376.714 tokens/sec`, with
`train_compute=0.128297 ms/token`, bounded `12/65536` route rows, `65526` cached
transition rows, GPU memory `2122->2123 MiB`, no observed contention, and zero
graph/native/sequence failures.

The emission replay-context review bridge now uses the same readout replay
source-window budget before it can record a Replay Controller context. The
facade bounds `selected_replay_context_seeds` and `observed_readout_slots` at
`32`, requires untruncated and well-formed source windows, and blocks oversized
payloads before mismatch, pressure, or context recording. The benchmark
`reports/bounded_replay_window_20260619/emission-replay-context-review-window.json`
recorded one exact `32/32` context and blocked oversized seed/slot payloads at
`32/2048` with zero blocked mismatch, pressure, or Replay Controller calls,
CPU source/gate placement, no global candidate/score scan, no hidden language
reasoning, `runs_live_tick=false`, and `runs_every_token=false`. This keeps
emission replay-context recording a selected slow/control-plane operation, not
a caller-sized replay batch or a second context route. The first clean hot-path
run was below band at `5877.891 tokens/sec`; the rerun stayed in the maintained
band at `5990.908 tokens/sec`, with bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, GPU memory `2032->2031 MiB`,
and zero graph/native sequence failures.

The generic SNN replay evaluation-context facade now uses that same source
window for observed sparse slots. `snn_replay_evaluation_context(...)` bounds
`observed_readout_slots` at `32`, requires the window to be untruncated and
well formed before mismatch, pressure, or context recording, and binds the
source-window report into the recorded context metadata. The benchmark
`reports/bounded_replay_window_20260619/snn-replay-evaluation-context-window.json`
recorded one exact `32/32` context and blocked oversized observed-slot payloads
at `32/2048` with zero blocked mismatch, pressure, or Replay Controller calls,
CPU source/gate placement, no global candidate/score scan, no hidden language
reasoning, `runs_live_tick=false`, and `runs_every_token=false`. The paired
hot-path run stayed in the maintained band at `6009.932 tokens/sec`, with
bounded `12/65536` route rows, `65526` cached transition rows, GPU memory
`2031->2045 MiB`, and zero graph/native sequence failures; sampled GPU
contention reached `22%`, so the evidence is throughput protection rather than
contention-free hardware. This retires the old caller-sized generic context
bridge as an active implementation shape.

Rollout rehearsal promotion now keeps the same slow/control-plane boundary for
trajectory evidence. `snn_language_readout_rollout_rehearsal_promotion_policy.v1`
scores only a `16`-event CPU source window with up to `32` replay targets per
event, and emits `bounded_snn_readout_rollout_rehearsal_source_window.v1` with
source/returned candidate counts, CPU archival/score placement, no raw text
payload, no hidden language reasoning, `runs_live_tick=false`,
`runs_every_token=false`, and `gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-rollout-rehearsal-source-window.json`
matched the diagnostic full-retained top rollout while scoring `16/2048`
retained events, reducing mean priority latency from `309.922768 ms` to
`2.090592 ms`. The paired 65536-column `524288`-token run stayed in band at
`6339.682 tokens/sec`, with `train_compute=0.129022 ms/token`, bounded
`12/65536` route rows, `65526` cached transition rows, GPU memory
`1867->1865 MiB`, and zero graph/native/sequence failures; the environment did
observe `22%` max GPU utilization, so this is throughput-protection evidence
rather than contention-free evidence.

Replay-path status projection now follows the same bounded ownership rule. The
service read model does not select replay or consolidation work, but it also
must not scan all retained replay ledgers just to summarize readiness.
`StatusReadModel` now projects emission review-history from a `16`
reviewed-emission source window, emission replay-design readiness from a `16`
reviewed-emission plus `16` internal-readout source window, and rollout
consolidation readiness from a `16` rollout-event plus `16` internal-readout
source window. The surfaces are
`bounded_snn_status_emission_review_history_source_window.v1`,
`bounded_snn_status_emission_replay_design_path_source_window.v1`, and
`bounded_snn_status_rollout_consolidation_path_source_window.v1`; all three expose
retained counts, window counts, truncation, CPU archival/score placement, no
global candidate/score scan, no raw text payload, no hidden language reasoning,
no live tick, no every-token cadence, and no CUDA archival metadata. The
benchmark `reports/bounded_replay_window_20260618/status-replay-path-source-window.json`
preserved latest history/emission/rollout evidence while checking `80/10240`
retained rows and reducing combined projection latency from `102.831789 ms` to
`1.309999 ms`; the no-profile paired long run stayed in band at `6408.252
tokens/sec` with bounded `12/65536` route rows and zero runtime failures, while
the profiled no-contention run reached `6081.034 tokens/sec`. This keeps
operator-facing Runtime Truth scalable without making service status a replay
scheduler.

The follow-on replay-artifact provenance path is indexed as well. Replay
artifact review tickets, evaluated transition-memory replay artifacts,
regeneration permits, sleep-plasticity review tickets, and scheduler-design
review tickets now use controller-owned ID indexes for verification instead of
linear retained-deque scans. Evaluated artifacts and permits bind
`bounded_snn_replay_artifact_provenance_source_window.v1`, capped to context,
ticket, artifact, and permit IDs, and report CPU archival placement, no global
candidate/score scan, no raw replay text, no language reasoning,
`runs_live_tick=false`, and `gpu_used=false`. The benchmark
`reports/bounded_replay_window_20260618/snn-replay-artifact-provenance-source-window.json`
kept the oldest retained chain verifiable at the retention tail with `4`
indexed lookups instead of `256` worst-case retained-record checks, averaging
`0.348376 ms` with no CUDA allocation. The accepted 65536-column `524288`-token
hot-path rerun stayed in band at `6286.248 tokens/sec` with
`train_compute=0.129585 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, no observed contention, and zero graph/native/sequence
failures after one slower same-code run was rejected. This keeps replay-backed
structural consent a bounded slow/control-plane path rather than a
retained-history scan.

Bucket-indexed memory windows now bound the source construction inside the
shared store helper as well as the returned candidate count. The previous
helper could return only `32` candidates while still materializing an entire
hot bucket through `list(reversed(...))`. `_candidate_indices_for_bucket_ids(...)`
now uses tail-indexed round-robin cursors and every bucket-scoped caller can
report `candidate_source_window_policy=tail_indexed_bucket_round_robin_no_full_bucket_materialization`,
source-read count, materialization count, CPU source placement, and no
full-bucket scan. `reports/bounded_replay_window_20260618/bucket-candidate-source-window-bounded.json`
used a `65536`-entry hot bucket, preserved newest-candidate parity, read `32`
source indices within a `32`-entry source-read budget, materialized `0`,
allocated `0.0 MiB` CUDA, and reduced mean source latency from `0.416944 ms`
to `0.060931 ms` (`6.843x`). The maintained
contract is that replay/query/frontier/ripple windows are bounded before any
modern-Hopfield-style local recall or replay scoring runs.

HF replay query collection now bounds retained column-anchor source selection
before the store collector runs. `capture_recent_memory_anchors(...)` records
recency metadata and refreshes anchor dict recency on recapture; checkpoints
preserve those fields for restored trainers. `_collect_anchor_replay_queries(...)`
then emits `bounded_replay_query_anchor_bucket_source_window.v1`, takes at most
`16` reverse-recency anchor buckets, and passes only that bucket window into
`collect_replay_query_indices(...)` and the follow-up HF recall evaluator. The
surface is slow-path replay/query work: it reports CPU archival placement, no
live tick, no every-token work, no global score/candidate scan, no raw replay
text, no hidden language reasoning, and `anchor_source_full_scan=false`. The
8192-anchor benchmark
`reports/bounded_replay_window_20260618/replay-query-anchor-source-window-bounded.json`
reduced source-selection latency from `16.414 ms` to `0.346 ms`, selected
newest-anchor queries with hit rate `1.0`, kept exact input recall, and used no
CUDA allocation. The paired `524288`-token protection run stayed in the same
sustained band at `6376.873 tokens/sec` with bounded `12/65536` route rows,
`65526` cached transition rows, flat `1787 MiB` GPU memory, and zero runtime
failures; the environment sampler did mark borderline GPU contention, so the
claim is live-tick protection rather than a clean speed ceiling.

Strong-capture slow-memory admission is now cadenced instead of every-strong.
The column/runtime path still keeps device strong-event evidence for threshold
crossings, but archival writes are selected by
`slow_memory_archive_strong_capture_min_interval_tokens` before
`DualMemoryStore.update(...)` runs. Runtime Truth exposes the configured
interval, archived strong count, refractory skip count, and last archived
strong token; ordinary live ticks in the 65536-column protection run reported
zero strong archives and zero refractory skips, so the mechanism does not add a
background archive workload. The focused report
`bounded_strong_capture_admission_cadence.v1` archived `17/256` forced-strong
tokens with a max selected gap of `16`, while the retired every-strong shape is
only a projection in the report, not a second executable trainer path.

Readout replay dry-run and plasticity bridge payloads are also bounded before
they can become tensor or bridge work. The column scheduler still does not run
these surfaces in the live tick: `SNNLanguageReadoutEvidenceLedger` caps
caller-supplied dry-run targets, dry-run trace records, and bridge candidate
sequences to `32` records with
`bounded_snn_readout_replay_dry_run_target_window.v1`,
`bounded_snn_readout_plasticity_preflight_trace_window.v1`, and
`bounded_snn_readout_plasticity_bridge_sequence_window.v1`. Each report states
CPU archival placement, no global candidate/score scan, no raw replay text, no
hidden language reasoning, `runs_live_tick=false`, and `runs_every_token=false`.
`reports/bounded_replay_window_20260619/readout-replay-target-window.json`
reduced dry-run and bridge materialization from `2048` caller records to `32`
records (`64x` less source work), while the paired `524288`-token protection
run stayed in band at `6109.000 tokens/sec` with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, and zero graph/native
sequence failures. The old full-payload shape is a retired projection in the
benchmark, not a side implementation path.

The exported language-plasticity replay semantics now share that single replay
budget instead of depending on callers to stay inside the API schema. Replay
evaluation and replay experiment inspect at most `32` caller records, and the
shadow-delta builder also caps each sparse side to `16` indices before local
pair scoring. Their bounded reports state CPU archival/source placement, CPU
active replay computation for the benchmark, no global candidate/score scan, no
raw replay text, no hidden language reasoning, `runs_live_tick=false`,
`runs_every_token=false`, and no runtime mutation. The benchmark
`reports/bounded_replay_window_20260619/language-plasticity-replay-window.json`
reduced replay records from `2048` to `32` and shadow pair checks from
`134217728` projected pairs to `8192`, while the longer `524288`-token protection
rerun stayed in the maintained band at `5999.398 tokens/sec` with bounded
`12/65536` route rows, `65526` cached transition rows, and zero graph/native
sequence failures. Because the sampler observed GPU-side contention, this is
protection evidence for the slow path, not a clean speed-ceiling claim.

The checkpointed language application executor now enforces the same selected
window at the mutation boundary. Live application consumes
`shadow_delta.bounded_synapses` through
`bounded_snn_language_plasticity_live_application_synapse_window.v1`, and
transition-memory regeneration consumes
`regeneration_design.candidate_synapses` through
`bounded_snn_transition_memory_regeneration_candidate_synapse_window.v1`; both
windows cap at `32` records and require the source payload to be untruncated
before checkpoint writes or runtime mutation. The benchmark
`reports/bounded_replay_window_20260619/language-application-synapse-window.json`
blocked oversized `2048`-record payloads after reading a bounded `33`-item
sentinel window, made zero checkpoint calls, and left runtime state unchanged;
exact-window payloads still applied or regenerated `32` synapses through
checkpointed paths. The clean `524288`-token protection run stayed in band at
`6039.734 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, and zero graph/native failures. This
keeps structural writes slow-path, operator/checkpoint gated, and bounded
without putting replay/application work into the live tick.

The rollout-regeneration facade now protects that same slow-path boundary
before permit issuance or application preflight. Permit, preflight, and
application use the shared `SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32`
operator, emit `bounded_snn_rollout_regeneration_permit_candidate_synapse_window.v1`,
`bounded_snn_rollout_regeneration_application_preflight_candidate_synapse_window.v1`,
and `bounded_snn_rollout_regeneration_application_candidate_synapse_window.v1`,
and require untruncated candidate payloads before replay-controller or executor
calls. Runtime Truth capacity inventory records the three facade
candidate-source-window boundaries as CPU source windows that do not run in the
live tick or every-token cadence. The facade benchmark
`reports/bounded_replay_window_20260619/rollout-regeneration-facade-candidate-window.json`
blocked oversized permit/preflight/application payloads at `32/2048` while the
exact `32`-candidate flow still reached the single executor path. The accepted
long rerun stayed in band at `6121.143 tokens/sec`, with bounded `12/65536`
route rows, `65526` cached transition rows, zero graph/native failures, and
flat `2031 MiB` GPU memory under sampled GPU contention. The old full-list
facade route is retired rather than preserved as side code.

The upstream readout-ledger rollout consolidation/regeneration reviews now use
the same source-window boundary before a permit preview can exist. Consolidation
design, shadow delta, developmental plasticity review, regeneration proposal
adapter, regeneration replay-artifact review, and Replay Controller
regeneration-design normalization all cap structural candidates at `32` through
the shared application-synapse window helper and require untruncated payloads.
The benchmark
`reports/bounded_replay_window_20260619/readout-ledger-rollout-candidate-window.json`
blocked oversized review/controller payloads at `32/2048`, kept exact `32`
candidate evidence on the single permit-preview path, and reported CPU
archival/source/gate placement, no global candidate/score scan, no hidden
language reasoning, `runs_live_tick=false`, and `runs_every_token=false`.
The clean `524288`-token hot-path run stayed in band at
`6075.293 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, GPU memory `2031->2043 MiB`, and zero
graph/native sequence failures. This keeps rollout consolidation a selected
slow-path review, not an always-on column tick or a second mutation route.

Dense-readout training now shares that single-window rule instead of preserving
an old caller-sized training side path. The design, schema, preflight, and
executor all use `SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT=32`
and `SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT=32`. The executor
emits `bounded_snn_dense_readout_training_transition_source_window.v1` and
`bounded_snn_dense_readout_training_transition_index_window.v1`, requires both
windows to be untruncated before checkpoint writes, and blocks oversized
payloads before runtime mutation. The benchmark
`reports/bounded_replay_window_20260619/dense-readout-training-transition-window.json`
blocked transition and index payloads at `32/2048` with zero checkpoint calls,
while an exact `32`-transition window still produced `32` dense/sparse updates.
The paired `524288`-token protection run stayed in band at
`6028.820 tokens/sec`, with bounded `12/65536` route rows, `65526` cached rows,
no observed contention, GPU memory `2029->2028 MiB`, and zero graph/native
failures.

Autonomous language-output and decoded-output event recording/review now share
the same record-family source-window helper as the preceding autonomous
binding, observation, training-window, and decoder-probe stages. Execution and
review read only `autonomous_language_output_events` or
`autonomous_decoded_output_events` before duplicate or ledger-presence checks,
return the `bounded_snn_readout_ledger_record_family_source_window.v1` report,
and write only the target event family plus count/timestamp fields. The
expanded benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-output-chain.json`
kept parity across the six-family autonomous output chain while reducing
checked rows from `35328` to `1536` and mean latency from `6778.768800 ms` to
`321.988933 ms`; CUDA was available but unused for ledger metadata, with
`0.0 MiB` allocation/reservation. The paired long run stayed in band at
`6048.638 tokens/sec`, `tick_duration_ms.p95=21.307`,
`train_compute=0.134492 ms/token`, `prepare_training=0.006912 ms/token`,
`finalize_total=0.006334 ms/token`, `route_input_rows_scored=12/65536`,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures. The old broad-normalized production path is removed rather than kept
as a side implementation.

Bounded text-emission and text-surface commit now use that same single-family
ledger route. `execute_autonomous_bounded_text_emission(...)` and its review
read only `autonomous_bounded_text_emission_events`; text-surface commit
execution/review read only `autonomous_text_surface_commit_events` and preserve
`current_text_surface_commit` as a single current pointer. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-text-surface-chain.json`
kept hash, review, total-count, and current-commit parity while reducing
checked rows from `47104` to `2048` and mean chain latency from
`9289.008333 ms` to `429.436800 ms`; CUDA was available but unused for ledger
metadata, with `0.0 MiB` allocation/reservation. The paired long run stayed in
band at `5980.715 tokens/sec`, `tick_duration_ms.p95=22.136`,
`train_compute=0.135992 ms/token`, `prepare_training=0.007115 ms/token`,
`finalize_total=0.006345 ms/token`, `route_input_rows_scored=12/65536`,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures. No observed contention was reported, with RTX 3060 memory
`2045->2047 MiB`.

Text-surface materialization and bounded language-surface commit now complete
the same ledger path. Execution/review reads only
`autonomous_text_surface_materialization_events` or
`autonomous_bounded_language_surface_commit_events`, reports the bounded
record-family source window, and keeps the single current materialization/commit
pointers without broad `_normalized_state()` duplicate or review scans. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json`
kept hash, review, total-count, and current-pointer parity across the ten-family
autonomous language-surface chain while reducing checked rows from `58880` to
`2560` and mean chain latency from `11175.229267 ms` to `525.534133 ms`; CUDA
was available but unused for ledger metadata, with archival/source/review
placement on CPU. The paired long run stayed in band at `5994.060 tokens/sec`,
`tick_duration_ms.p95=21.991`, `train_compute=0.135570 ms/token`,
`prepare_training=0.007057 ms/token`, `finalize_total=0.006414 ms/token`,
`route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`,
and zero graph/native sequence failures. RTX 3060 runtime memory moved
`2044->2059 MiB`.

Bounded language-surface use and SNN language-generation now extend that same
ledger path. Execution/review reads only
`autonomous_bounded_language_surface_use_events` or
`autonomous_snn_language_generation_events`, reports the bounded record-family
source window, and updates only the target family count/timestamp. The
benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json`
kept hash, review, total-count, and current-pointer parity across the expanded
autonomous language-generation chain while reducing checked rows from `70656`
to `3072` and mean chain latency from `13505.919533 ms` to `631.221 ms`; CUDA
was available but unused for ledger metadata, with archival/source/review
placement on CPU. The paired long run stayed in band at `6074.417 tokens/sec`,
`tick_duration_ms.p95=21.376`, `train_compute=0.133727 ms/token`,
`prepare_training=0.007038 ms/token`, `finalize_total=0.006252 ms/token`,
`route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`,
and zero graph/native sequence failures with no observed contention. RTX 3060
runtime memory moved `2044->2047 MiB`.

SNN language decoding, thought-surface, thought-memory,
thought-consolidation, and thought-structural-plasticity now stay on the same
one-path ledger boundary. Execution/review reads only the target downstream
language/thought event family, reports
`bounded_snn_readout_ledger_record_family_source_window.v1`, and updates only
that family count/timestamp. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-thought-structural-chain.json`
kept hash, review-match, total-count, and current-pointer parity across the
seventeen-component readout/language/thought chain while reducing checked rows
from `100096` to `4352` and mean chain latency from `19704.406867 ms` to
`1046.241300 ms`; CUDA was available but unused for ledger metadata, with
archival/source/review placement on CPU. The clean paired rerun stayed in band
at `6005.229 tokens/sec`, `tick_duration_ms.p95=22.012`,
`train_compute=0.135094 ms/token`, `prepare_training=0.007082 ms/token`,
`finalize_total=0.006415 ms/token`, `route_input_rows_scored=12/65536`,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures with no observed contention. RTX 3060 runtime memory moved
`1856->1857 MiB`; the first same-shape run is not primary evidence because it
observed GPU contention.

Synapse provenance audit now keeps the same one-path rule for readout evidence
hash validation. `synapse_provenance_audit(...)` collects only hashes referenced
by `synapse_provenance_by_key`, reads `events` through
`bounded_snn_readout_evidence_event_map_source_window.v1`, and exposes that
source window in the promotion gate. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-synapse-provenance-map.json`
kept requested event-map hash parity while reducing checked rows from `2944` to
`128` (`23x`) and mean event-map latency from `319.823233 ms` to
`13.972533 ms`; CUDA was
available but unused for ledger metadata, with archival/lookup placement on
CPU. The paired long run stayed in band at `5994.111 tokens/sec`,
`tick_duration_ms.p95=21.885`, `train_compute=0.135406 ms/token`,
`prepare_training=0.007135 ms/token`, `finalize_total=0.006412 ms/token`,
`route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`,
and zero graph/native sequence failures with no observed contention. RTX 3060
runtime memory moved `1980->1976 MiB`.

Emission review history now follows the same bounded display rule instead of
normalizing every readout-ledger event family before returning reviewed output.
`emission_review_history(...)` reads only `emission_review_events` through
`bounded_snn_emission_review_history_source_window.v1`, exposes reviewed
bounded text for operator display, and reports no replay, no hidden language
reasoning, no live-tick work, and no every-token cadence. The benchmark
`reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-emission-history.json`
kept review-hash and text-hash parity while reducing checked rows from `2944`
to `128` (`23x`) and mean display-history latency from `345.815600 ms` to
`25.503433 ms`. The paired long run stayed in band at
`6051.817 tokens/sec`, `tick_duration_ms.p95=21.323`,
`train_compute=0.134300 ms/token`, `prepare_training=0.006972 ms/token`,
`finalize_total=0.006364 ms/token`, `route_input_rows_scored=12/65536`,
`state_transition_runs_all_columns=false`, and zero graph/native sequence
failures with no observed contention. RTX 3060 runtime memory stayed flat at
`1972 MiB`.

Replay-dataset preview is now explicitly outside the Column Runtime tick. The
service reads at most `50` retained runtime episode traces, at most `64`
retained replay samples, and at most `16` stored sanitized candidate links per
sample through `bounded_replay_dataset_preview_source_window.v1` and
`bounded_replay_dataset_sample_link_source_window.v1`. Both reports state
`runs_live_tick=false`, `runs_every_token=false`, CPU archival/source
placement, no GPU-resident archival metadata, no global scan, no replay text
reasoning, and no mutation/plasticity/training authority. The old export shape
that walked every retained trace/link and could report `50` items while
returning only `16` is retired.

The focused report preserved `50/50` selected target IDs and replay links
against a diagnostic full-retained walk while reducing replay-sample and
candidate source work by `4x`. Its `2006.587280 ms` mean cost is accepted only
as explicit operator/export latency. The paired clean `524288`-token run stayed
in the maintained noisy complete-runtime band at `5923.269 tokens/sec`, with
`tick_duration_ms.p95=22.446`, `train_compute=0.136941 ms/token`,
`prepare_training=0.007302 ms/token`, `finalize_total=0.006512 ms/token`,
bounded `12/65536` route rows, `65526` cached transition rows, no observed
contention, and zero graph/native sequence failures. Runtime CUDA memory moved
`3554->3541 MiB`; replay-dataset archival/source work used no GPU.

Runtime trace export and replay-sample summary now use the same control-plane
source-window boundary. `export_runtime_trace_examples(...)` selects at most
`50` retained runtime traces through
`bounded_runtime_trace_export_source_window.v1`; replay-sample summary selects
at most `64` retained sample records through
`bounded_replay_sample_summary_source_window.v1`; living status and feedback
summary read bounded recent trace/action windows. The follow-up report
`reports/bounded_replay_window_20260620/replay-dataset-runtime-trace-export-summary-source-window.json`
preserved `50/50` trace/export target parity while reading `50/64` traces,
`64/256` replay-sample summary rows, and `1024/4096` candidate-link rows. This
is still operator/export work with CPU archival placement, no hidden replay-text
reasoning, and no live-tick or every-token authority. The accepted protection
rerun processed `524288` tokens at `6047.311 tokens/sec` with bounded
`12/65536` route rows, `state_transition_runs_all_columns=false`, no observed
contention, flat RTX 3060 memory at `1911 MiB`, and zero graph/native sequence
failures.

Applied-synapse provenance status is also read-only projection work. The
service no longer scans all retained applied sparse weights and provenance rows
to decide audit readiness. It reads a `32 + 32` CPU source window through
`bounded_snn_status_applied_synapse_provenance_source_window.v1`, reports
retained/source/truncated counts, and marks exact audit readiness unavailable
when the window is truncated. The benchmark
`reports/bounded_replay_window_20260620/status-applied-synapse-provenance-source-window.json`
reduced status source reads from `4096` to `64` rows and mean latency from
`66.313336 ms` to `3.242332 ms` while keeping CUDA allocation/reservation at
`0.0 MiB`. The accepted `524288`-token rerun stayed in band at
`6350.288 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, no observed contention, zero graph/native sequence failures,
and flat RTX 3060 memory at `1936 MiB`. Exact applied-synapse integrity belongs
in an explicit audit/slow window; status only reports whether its bounded
source window is complete enough to allow that review.

The exact applied-synapse provenance audit now uses that same one-path
boundary instead of keeping a full audit scan beside the status window.
`synapse_provenance_audit(...)` emits
`bounded_snn_readout_synapse_provenance_audit_source_window.v1`, reads at most
`64` applied sparse-weight/provenance rows from CPU archival state, requests
ledger rows only for the selected hashes, and blocks exact review when retained
applied synapse provenance exceeds the source window. The benchmark
`reports/bounded_replay_window_20260620/synapse-provenance-audit-source-window.json`
matched the diagnostic first source window, read `64` bounded rows instead of
`4096` diagnostic records and `2048` materialized rows (`32x` less source work
by report metric), and reduced mean audit latency from `259.221928 ms` to
`75.262088 ms` with no GPU audit allocation. The accepted `524288`-token rerun
stayed in band at `6441.166 tokens/sec`, with `tick_duration_ms.p95=19.527`,
`train_compute=0.127184 ms/token`, bounded `12/65536` route rows, no observed
contention, flat RTX 3060 memory at `1866 MiB`, and zero graph/native sequence
failures. Full applied-synapse audit scans are now benchmark-local diagnostics
only.

Transition-memory status projections now share that same source-window rule
instead of each projection materializing all retained sparse-transition and
provenance rows. Capacity pressure, dense readout tensor integrity, applied
synapse provenance, and rollout/server binding all read through a single
bounded helper: at most `32` sparse-transition weights and `32`
`synapse_provenance_by_key` rows per projection, CPU archival/lookup placement,
no replay, no mutation/plasticity, no hidden language reasoning, and no live
tick or every-token cadence. When the window is truncated, exact resize,
dense-integrity, audit, and rollout-review readiness are blocked rather than
computed from partial evidence.

The benchmark
`reports/bounded_replay_window_20260620/status-transition-memory-source-window.json`
used `2048` retained sparse weights and provenance rows. The maintained path
read `256` bounded rows across four projections instead of `10240` rows in the
benchmark-local retired repeated broad projection (`40x` less source work),
reduced mean status latency from `89.558896 ms` to `11.162376 ms`
(`8.023282x`), kept Python peak allocation at `0.065983 MiB` versus
`1.372842 MiB`, and used `0.0 MiB` CUDA allocation/reservation. This retires
the broad transition-memory status projection family; exact transition-memory
integrity belongs in selected slow audit/replay windows, not in routine
status. The accepted `524288`-token rerun processed `6371.238 tokens/sec`
with `train_compute=0.128035 ms/token`, bounded `12/65536` route rows,
`65526` cached transition rows, and zero graph/native sequence failures.
Velocity still reported borderline GPU contention (`23%`), so this is
same-band throughput protection rather than a clean speed ceiling.

## Links

- [Runtime Truth](runtime-truth.md)
- [Metabolism](metabolism.md)
- [Hot Path](hot-path.md)
- [Dynamic Growth](dynamic-growth.md)
- [Pruning](pruning.md)
- [Core module](../modules/core.md)

## Explicit Replay Text Payload Opt-In

Replay-entry access is tensor-first by default. `DualMemoryStore.replay_entry(...)`
returns assembly/input/routing/STC/consolidation metadata without raw text unless
a caller passes `include_text_payload=True`. Sleep replay continues to use tensor
payloads only, while query/source-bank/context readout must opt in to text only
inside an already bounded candidate or returned-match window.

The focused report
`reports/bounded_replay_window_20260620/replay-entry-text-payload-opt-in.json`
used a `65536`-entry store and passed `explicit_replay_entry_text_payload_opt_in.v1`:
default replay-entry reads loaded `0/192` raw text payloads, explicit opt-in
loaded `192/192`, and bounded query readout loaded only `5` returned-match
payloads with CPU archival placement, no global scans, no live tick, and
`language_reasoning=false`.

The hot-path protection report
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-entry-text-payload-opt-in.json`
kept the live tick in the maintained band at `5993.863 tokens/sec`,
`tick_duration_ms.p95=21.555`, `train_compute=0.135543 ms/token`, bounded
`12/65536` route rows, cached `65526` transition rows, RTX 3060 memory
`1878->1879 MiB`, no observed contention, and zero graph/native sequence
failures.

## Replay Sample Single Path

Operator-gated replay review now has one service path:
`POST /terminus/replay-sample` and `GET /terminus/replay-sample/history`.
The old `/terminus/replay-execute` alias, `mode="execute"`, `execution_id`, and
`replay_executor_summary` projection are retired. This keeps audit sampling
from being mistaken for a live replay/consolidation executor and preserves the
column-runtime rule that replay execution must be an explicit slow-path window,
not a duplicate control-plane name.

The bounded summary remains `bounded_replay_sample_summary_source_window.v1`
over at most `64` CPU replay-sample records. The service benchmark
`reports/bounded_replay_window_20260620/replay-sample-single-path-service-benchmark.json`
proved the duplicate executor summary is absent; the replay-dataset source
window
`reports/bounded_replay_window_20260620/replay-dataset-source-window-replay-sample-single-path.json`
kept canonical `sample` records with `50/50` target/link parity; and the
hot-path run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-sample-single-path.json`
stayed in band at `5951.781 tokens/sec` with bounded `12/65536` route rows,
`65526` cached rows, and zero graph/native sequence failures.
