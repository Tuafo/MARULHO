---
type: concept
status: draft
related_code:
  - ../../../src/marulho/core/column_runtime.py
  - ../../../src/marulho/core/hypercube.py
  - ../../../src/marulho/training/model.py
  - ../../../src/marulho/training/trainer.py
related_docs:
  - ../modules/core.md
  - ../concepts/runtime-truth.md
related_papers:
  - ../papers/predictive-coding.md
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
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

`ColumnStructuralReviewQueue` is the durable continuation path for column growth/prune review. It is training-owned, checkpointed with the model, and fed only by bounded awake/candidate IDs plus already-owned predictive/metabolism tensors: prediction error, confidence, prediction failure streak, estimated cost, memory pressure, wake reason, and sleep reason. A growth ticket points to `explicit_binding_growth_trial_design`; a prune/sleep ticket points to `isolated_column_prune_or_sleep_review`. Both require operator review and a checkpoint transaction, and both report `mutates_runtime_state=false`. Retained CPU ticks can update the queue from the wake plan immediately. CUDA graph bursts update it only on a slow structural-review cadence; skipped bursts record deferred capture such as `structural_review_cuda_cadence_not_due` so the queue does not add a host sync to every graph host-truth boundary. Runtime Truth projects pending/growth/prune counts, last evaluated/cached column counts, update/deferred counts, next gate, and `runs_all_columns=false`; service does not create or rank tickets.

Awake-ripple replay tagging now consumes the same scheduler boundary. `DualMemoryStore` keeps archival replay storage on CPU, but it maintains a bucket-to-entry index as entries are admitted, restored, or reservoir-replaced. When retained `train_step` or CUDA text burst flushing has awake route candidates, it calls `ripple_tag_awake(..., awake_bucket_ids=...)`; the store then touches only entries attached to those awake buckets and reports `last_ripple_scan_mode=awake_bucket_index`. The legacy scalar/vector recent-memory scan remains only when a caller omits scheduler context. The direct report `reports/column_scheduler_20260616/awake-ripple-scope-8192-i256.json` compared an 8192-entry unscoped scan with a 10-bucket scoped path over 256 iterations: scoped tagging used `0` scalar/vector scans, `256` bucket-index scans, `last_ripple_awake_candidate_count=10`, and improved mean tag time from `1.1831957031063212 ms` to `0.9895980468854759 ms` (`1.1956326175361276x`). The matching long real-path run at `reports/column_scheduler_20260616/awake-ripple-scope-8192-131072-i32.json` stayed in the promoted band at `6286.386 tokens/sec`, `train_compute=0.130081 ms/token`, `route_input_rows_scored=10`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention.

`PredictiveColumnState` owns `prediction_failure_streak` beside prediction error and confidence. Repeated raw failures increment the streak on the predictive tensor device; successful prediction resets it. The streak is saved in trainer checkpoints and restored with the model, so growth evidence survives rollback and does not live in `service`.

When that gate is ready, the explicit binding-growth trial endpoint can ask core for a deterministic candidate-scoped hypercube outreach plan. The plan is bounded by an edge budget, hashes the exact baseline adjacency, and remains read-only. This narrows the path from surprise to structural experimentation without making the scheduler, Runtime Truth, or status polling a mutation authority.

`PredictiveColumnState` also records the last predictive location/update scope, lazy predictive materialization scope, and the last predictive-vote execution scope. `candidate_predictive_update_start_tokens` now separates predictive-state wake from structural dead-column retirement: after that gate, location/velocity decay, prediction error, confidence, failure streaks, and high-prediction non-winner decay update only for the routed awake mask while non-candidate state remains cached on the retained CPU route and promoted fused CUDA route. If a non-awake column later appears in the routed candidate set, `PredictiveColumnState` advances only that candidate through missed non-winner predictive updates before vote/scoring uses its state. The retained CPU route now applies prediction error, location/velocity, prediction-weight decay, and cached-state materialization through one candidate predictive transition rather than re-canonicalizing the same awake set across three split calls. Vote and update can share one candidate materialization in a tick, and a repeated materialization request for the same completed candidate set reports `candidate_subset_completed_noop`. Checkpoint restore recomputes the cached-column flag from predictive step stamps so a restored runtime still wakes stale predictive columns correctly. Predictive wake currently keeps ordered replay for cached candidates because vectorized closed-form replacements improved cost but failed repeated long winner-parity checks near fallback-threshold boundaries. `CompetitiveColumnLayer` does the same for missed zero-activity homeostasis and fallback threshold-relaxation events, preserving dense update order without a hot-path all-column tax. The retained trainer route may first request a bounded backfill pool, sort it by retrieval distance, and remove candidate columns already at the deep-sleep threshold; before `dead_column_steps` is reachable, it skips backfill and reports `candidate_deep_sleep_filter_no_column_can_be_deep_sleep_yet` because no candidate can truthfully be deep-sleep eligible yet. If every retrieved candidate is deep-sleeping, it falls back to the bounded retrieved set and reports the reason. For retained predictive voting, the trainer obtains routing candidates before consensus voting, then recomputes agreement only for the awake mask and reuses cached gains for the other columns. If fused CUDA vote/competition must fall back because context gain is present, that retained selection fallback still passes the routed candidate IDs into `PredictiveColumnState.vote()` and hands `CompetitiveColumnLayer` a candidate-local gain tensor, so unsupported fused selection does not silently become an all-column predictive vote. On CUDA predictive updates, eager candidate indexing remains rejected: the 2026-06-15 isolated writeback experiment measured `7.0080195312499995 ms` mean for eager candidate indexing versus `3.080762890625 ms` dense writeback. The promoted fused in-place/graph route updates only the wake-plan candidates inside the existing transition launch, stamps those rows on device, and reports `candidate_predictive_transition_mode=fused_inplace`, active/fallback truth, execution count, and cached-row count. When candidate homeostasis starts earlier than predictive wake, the persistent CUDA runtime captures a `candidate_subset_dense_predictive` graph for the interim dense predictive update and a separate candidate predictive graph for the later sparse wake, so the two gates can differ without disabling the fused transition. The standalone candidate predictive writeback helper, isolated predictive transition benchmark, configurable `compiled` dense predictive mode, configurable `legacy` dense-transition bypass, and selectable `fused_eager` production mode are removed so the vault has one promoted CUDA mutation boundary. Unsupported in-place prerequisites fall back internally to dense eager tensor semantics with a concrete Runtime Truth fallback reason before claiming sparse predictive updates.

The CUDA route-vote owner now applies deep-sleep and evidence-backed memory-pressure gates before candidate vote/winner selection by masking route-score rows inside `core.fused_route_vote_cuda`. This is not a post-selection status projection: `ColumnTransitionRuntime` stages a four-value device control tensor, the graph/fused route writes a twelve-value scheduler-filter packet, and `MarulhoTrainer` builds the `ColumnWakePlan` from that training-owned route evidence. The state packet is mirrored on a cadence, not every token, so Runtime Truth can expose filtered counts and fallback reasons without adding a hot-path all-column scan. If fewer than `k` eligible route rows exist, the implementation keeps the unfiltered fixed-k route result and reports an explicit fallback reason such as `insufficient_awake_route_scores_after_deep_sleep_filter` or `insufficient_awake_route_scores_after_memory_pressure_filter`. Route-cost truth is separate from awake-mask truth: `route_vote_scoring` reports `route_input_rows_scored`, `route_output_candidate_count`, `route_rows_run_all_columns`, `bounded_route_scoring`, and `route_scoring_unbounded_reason`. The promoted CUDA route now seeds a fixed `k_routing` `route_candidate_bank` once from the complete routing tensor cache, then uses `indexed_route_bank_vote` to score the bank plus a fixed two-row probe lane on steady graph ticks. `route_candidate_bank_size` is retired as a production config selector; old checkpoint keys migrate away, and wider banks remain evaluation-only. `predictive_route_vote_mode` is fixed to `cuda_graph_text`; retired `tensor` and `fused_triton_text` values migrate out of checkpoints, and old modes can be forced only by explicit evaluation overrides for parity/benchmark controls. Runtime Truth keeps the seed visible with `candidate_boundary=exact_full_cache_score_seed_route_bank` and `route_candidate_bank_not_ready_exact_seed`, then reports `bounded_route_bank_probe_score_then_filter_select` or `bounded_route_bank_probe_burst_score_then_filter_select` with `route_input_rows_scored=k_routing+2`, `route_output_candidate_count=k_routing`, `bounded_route_scoring=true`, and `route_vote_requested_mode_source` showing promoted config, restore defer, or evaluation override.

The route bank now has a bounded discovery lane instead of pure k-only self-refresh. Steady CUDA graph/burst ticks score `k_routing` bank rows plus `2` probe rows (`score_rows=12` for the 8192-column checkpoint) while still waking and mutating only `k_routing` columns. Probe refresh is quantum-cadenced (`refresh_interval_tokens=16`) to preserve fused, graph, and burst parity. Checkpoints save ready route-bank IDs plus probe cursor state and restore them before CUDA graph capture, so restored runs can avoid the first-tick exact-cache seed when the saved bank still maps into the routing tensor cache. Runtime Truth exposes `probe_rows`, `score_rows`, `probe_cursor`, `scored_since_refresh`, `probe_refresh_count`, `checkpoint_restore_count`, and `restore_reason` beside `route_input_rows_scored`. The longer 8192-column real-path run `reports/column_scheduler_20260616/route-bank-probe-lane-8192-131072-i32.json` stayed in the promoted band at `6141.234 tokens/sec`, `train_compute=0.130664 ms/token`, `tick_duration_ms.p95=21.976`, `route_input_rows_scored=12/8192`, `route_output_candidate_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The checkpoint-backed restore gate at `reports/column_scheduler_20260616/route-bank-checkpoint-restore-8192-131072-i32.json` reported `seed_count=0`, `fallback_count=0`, `graph_bypass_count=0`, `checkpoint_restore_count=1`, `route_input_rows_scored=12/8192`, `state_transition_cached_count=8182`, and `6129.693 tokens/sec` with `0.130218 ms/token` train compute. The reproducible promoted-checkpoint builder then extended the same complete-runtime gate to 16384 columns at `reports/column_scheduler_20260616/promoted-scheduler-16384-131072-i32.json`: `6154.503 tokens/sec`, `train_compute=0.130874 ms/token`, `tick_duration_ms.p95=21.419`, `route_input_rows_scored=12/16384`, `route_output_candidate_count=10`, `state_transition_cached_count=16374`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The follow-up quality gate rejected fixed sequential probes as a relevance solution: probe2 q16 matched exact top-1 only `0.017578125` and winner only `0.001953125`; probe256 q16 still matched top-1 only `0.34765625` and winner only `0.044921875`. This is bounded outside-bank discovery and route-cost truth, not growth/pruning autonomy or a full relevance-quality solution.

The promoted CUDA/text scaling path now starts from the candidate scheduler boundary instead of carrying unused dense startup work when the checkpoint is already past the candidate gate. `ColumnTransitionRuntime` precompiles only the routed candidate shape, the persistent text graph captures only `candidate_subset`, and Runtime Truth exposes `capture_graph_policy`. The 2026-06-15 8192-column real-path run proved the previous fallback is fixed: `reports/real_path_column_scaling_20260615/runtime-8192-promoted-131072-i32.json` stayed active on `cuda:0`, reported `precompiled_candidate_counts=[10]`, graph names `["candidate_subset"]`, route-vote output `10`, no graph/native fallback, and `3564.222 tokens/sec` with `0.251487 ms/token` train compute. The matching 1024-column control at `reports/real_path_column_scaling_20260615/runtime-1024-promoted-131072-i32.json` reached `6108.728 tokens/sec` with `0.133438 ms/token`. This proved bounded awake execution, not route-row cost invariance: route-vote input rows still grew from `1024` to `8192`, and the `route_vote_scoring` surface made that explicit. The 2026-06-16 sparse CUDA state-transition slice then closed the dense state-transition gap for candidate-subset graph/burst ticks: `reports/column_scheduler_20260616/sparse-cuda-state-transition-candidate-homeostasis-131072-i32.json` reports `state_transition_column_count=10`, `state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`, and `6007.582 tokens/sec` with `0.132374 ms/token` train compute. The route-candidate-bank slice moved the next boundary into route scoring itself: the 1024-column warm-seed run at `reports/column_scheduler_20260616/route-candidate-bank-warmseed-131072-i32.json` reports `route_input_rows_scored=10`, `route_rows_run_all_columns=false`, `bounded_route_scoring=true`, `state_transition_column_count=10`, `state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`, and `6109.301 tokens/sec` with `0.130536 ms/token` train compute. The matching 8192-column gate at `reports/column_scheduler_20260616/route-candidate-bank-8192-warmseed-131072-i32.json` reports `route_input_rows_scored=10`, `route_input_fraction=0.001220703125`, `route_rows_run_all_columns=false`, `bounded_route_scoring=true`, `state_transition_column_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, and `6110.715 tokens/sec` with `0.135007 ms/token` train compute. The follow-up route-candidate-bank quality gate keeps this throughput claim separate from relevance quality: the 8192-column default-text oracle at `reports/column_scheduler_20260616/route-candidate-bank-quality-8192-default-text-s512.json` kept steady bank scoring at `10` rows, but exact top-1 was in the bank only `0.009765625` of ticks, the simulated winner matched only `0.001953125`, mean top-k overlap was `0.02578125`, and the worst exact top-1 miss streak was `134`. A 32-tick evaluation-only exact reseed still failed winner parity, while a random/stable control passed. The corrected graph-neighbor quality probe at `reports/column_scheduler_20260616/route-candidate-graph-quality-8192-default-text-s512-neighbor208-cap1536.json` used `neighbor_count=208` and `capacity_rows=1536`, scored `1481.068` rows on average, matched exact top-1 `0.994140625`, matched the simulated exact winner `0.98828125`, and limited the worst miss streak to `1`. That proves bounded discovery is plausible, but the matching live runtime attempt at `reports/column_scheduler_20260616/route-candidate-graph-neighbor208-cap1536-8192-131072-i32.json` reached only `4682.167 tokens/sec` and `0.183927 ms/token` train compute while scoring about `1509/8192` route rows, below the promoted `6110.715`/`0.135007` route-bank baseline. After the runtime graph-neighbor path was removed, `reports/column_scheduler_20260616/route-candidate-bank-8192-warmseed-131072-i32-after-graph-reject-cleanup.json` reached `6150.296 tokens/sec`, `0.130390 ms/token` train compute, `route_input_rows_scored=10`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. The live graph-neighbor path is therefore rejected and removed; only the offline quality gate remains. The promoted steady path is real bounded execution, but k-only self-refresh is not a complete relevance scheduler for changing text. The fixed probe lane is the promoted bounded-cost path, not a solved relevance router: it samples outside the current bank without a hidden hot-path all-column tax and preserves the long 6k-ish path, but later quality evidence keeps discovery-quality promotion open.

The bounded graph-walk probe narrowed the graph-neighbor question without promoting a second runtime. `route_candidate_bank_quality_gate` can now simulate a CAGRA-style fixed-degree walk: start from the route bank plus two probe rows, score a bounded frontier, keep a fixed beam, and expand neighbor frontiers for fixed rounds. The 8192-column default-text evidence stayed below the promotion gate. A 512-row walk reached exact top-1 `0.896484375` but winner match only `0.263671875`; the best 1024-row deep walk reached exact top-1 `1.0` but winner match only `0.73046875`; a 1536-row deep walk still matched the simulated winner only `0.767578125`. This rejects top-1-only graph walking as a scheduler boundary and keeps graph-walk code evaluation-only until a fused/GPU-owned variant can pass winner quality and the 131072-token 6k-ish gate. The same-session 131072-token recheck kept the promoted route-bank/probe-lane runtime unchanged at `6141.295 tokens/sec`, `train_compute=0.132462 ms/token`, `route_input_rows_scored=12/8192`, and `state_transition_runs_all_columns=false`; GPU contention was observed, so it is in-band evidence, not a new top-speed claim.

The retained fallback vote-scope cleanup closes a smaller fake-sleep gap: when fused CUDA vote/competition falls back for a context-gain tick, `_retained_consensus_gain()` now receives the selected candidates and calls predictive vote with that awake mask. `CompetitiveColumnLayer.compete()` also accepts a candidate-local context-gain tensor, so fallback selection can preserve the bounded candidate contract instead of recomputing all predictive votes. Focused CUDA tests verify fallback vote reports `updated_column_count=4`, cached votes for the other `12` columns, and `runs_all_columns=false` on a 16-column shape. The matching 8192-column real-path run at `reports/column_scheduler_20260616/fallback-candidate-vote-scope-8192-131072-i32.json` processed `131072` tokens at `6016.247 tokens/sec`, `train_compute=0.1330716 ms/token`, `route_input_rows_scored=12/8192`, `route_output_candidate_count=10`, `state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`, zero graph/native/sequence failures, and no observed contention. This keeps the long path in the 6k-ish band; it is a fallback truth cleanup, not a new speed ceiling.

The cheap bounded-discovery probe rejected two tempting shortcuts before they reached runtime. `route_candidate_discovery_probe` models fixed farthest-landmark buckets and random-projection buckets as evaluation-only candidate routers with explicit selector-row and steady-row accounting. Landmark256/top8/bucket128 at q16 scored `926.544` route rows on average but reached only `0.77734375` exact top-1 and `0.4296875` winner match. Per-token landmark256/top8 improved to `0.884765625` top-1 and `0.525390625` winner match while paying `256` selector rows each tick. Landmark512/top16/bucket128 reached exact top-1 `1.0`, but winner match remained `0.91015625` while scoring `1642.932` route rows plus `512` selector rows per tick. Random-projection512/top32/bucket64 q16 scored `1790.795` rows and reached only `0.255859375` exact top-1. The next discovery scheduler therefore cannot be a cheap landmark/projection bucket bolted beside the route bank; it must be a fused/GPU-owned graph or ANN boundary with winner-quality and 131072-token evidence.

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
training-owned metabolism state. Focused tests prove service remains read-only
projection and does not own scheduler decisions. The current benchmark evidence
keeps awake work bounded but does not prove complete-runtime neutrality, so this
is a truth-boundary correction rather than a scheduler speed promotion.

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
mode, active/fallback truth, execution count, and cached-row count. The retired
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
`132647424` cached predictive rows.

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

The explicit helper is now promoted through the existing structural mutation transaction. The design and preflight bind the binding-hub target, core method, operator reason, edge budget, revision, and rollback checkpoint. The executor verifies the serialized binding state, applies one refresh, reports exact growth/prune and edge deltas, and rolls back on no-op, over-budget, tampered, or unverified commits. The previous executor path that consumed binding evidence but mutated `ConceptStore` capacity was removed.

The preceding trial-design stage derives candidate source columns from live repeated prediction failures rather than caller-authored IDs. It proposes exact sparse edges but does not apply them. The explicit checkpoint-clone binding-growth evaluator can test those edges against prediction, spike health, Runtime Truth, and metabolism without touching the always-on runtime; only successful evidence should advance toward the operator transaction.

On 2026-06-11, Hypercube Binding became the first larger Subcortex specialist with an explicit event-driven wake policy. While learned binding usage is absent, the trainer runs a probe every four tokens and preserves cached state on the other ticks; active or checkpoint-restored binding runs every tick. Runtime Truth exposes `runtime_active`, bind/probe count, idle-skip count, last execution mode, interval, and CUDA tensor placement.

The repeatable runner at `marulho.evaluation.binding_wake_benchmark` compared interval 1 with interval 4 on the same 1024-column checkpoint and synchronized RTX 3060 inputs. Across 120 samples per arm, median latency improved from `32.2069` to `29.5535 ms` (`8.24%`), p95 from `46.3573` to `42.7967 ms` (`7.68%`), and mean by `5.07%`; allocated/reserved VRAM stayed `20.4585/48.0 MB`. An isolated profiler trace measured 70 CUDA kernels for an idle probe and zero for a cached skip. A post-restart live tick reported Runtime Truth `alive`, binding tensors on `cuda:0`, 3 probes, 11 cached skips, and a 14-token tick in `2521.655 ms`.

On 2026-06-11, Cross-Modal Grounding adopted the same specialist wake rule for text-only ticks. When no visual or audio spikes are accepted, the trainer updates text grounding every four tokens and records cached-idle skips on the others; accepted sensory evidence wakes text updates every tick. Runtime Truth exposes text update count, idle-skip count, execution mode, interval, and tensor placement. A live `/feed` text-only check advanced the CUDA runtime to revision 960 and reported 2 cross-modal text updates, 8 cached skips, and cross-modal tensors on `cuda:0`.

The repeatable runner at `marulho.evaluation.cross_modal_wake_benchmark` compared interval 1 with interval 4 on the revision-209 live checkpoint. Across 120 text-only samples per arm, median latency improved from `57.59725` to `51.5753 ms` (`10.46%`) and mean from `59.4546` to `54.8716 ms` (`7.71%`). P95 regressed from `82.0887` to `83.8702 ms`, so this is not a tail-latency claim. Isolated profiling measured 49 CUDA kernels and 108 ATen ops for one text update versus 2 CUDA kernels and 3 ATen ops for one cached skip.

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

## Links

- [Runtime Truth](runtime-truth.md)
- [Metabolism](metabolism.md)
- [Hot Path](hot-path.md)
- [Dynamic Growth](dynamic-growth.md)
- [Pruning](pruning.md)
- [Core module](../modules/core.md)
