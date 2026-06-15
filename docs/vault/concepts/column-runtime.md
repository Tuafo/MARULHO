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
- a sampled Column Registry with role, state, local prediction/surprise/usefulness/cost state, last-run token estimate, and optional memory budget
- bounded column votes with confidence, prediction error, usefulness, cost, disagreement, wake reason, and cached-vote use
- growth-gate evidence from repeated prediction failure streaks, not one-shot surprise
- pruning/homeostasis evidence from weak, idle, or redundant columns
- the availability of a bounded single-column associative recall helper

Growth and pruning decisions remain report-only. Several narrower execution slices are promoted: retrieved candidates bound competition and homeostasis after the early-learning window; CPU predictive updates can remain candidate-scoped; retained predictive voting recomputes reference-frame agreement only for the routed awake mask while non-awake columns keep cached consensus gain; and an eligible CUDA checkpoint can use the trainer-owned in-place steady-state transition for the dense predictive/plasticity cluster. Dense assembly remains active where the representation requires it. Runtime Truth reports observed scored count, update scope, predictive-vote cached count, transition executor, warmup, execution/failure counts, fallback reason, and `runs_all_columns` truth rather than inferring execution from configured budgets.

Column reporting is control-plane work. When source state is on CUDA, Runtime Truth uses one latency-first column-state snapshot for scheduler evidence. A bounded device-export attempt reduced bytes but was slower at current 1024/8192-column sizes, so the active policy favors latency over smaller status payloads. CPU reports still materialize only bounded vote/registry samples. Runtime Truth exposes source device, report compute device, source tensor count, materialized column-state count, snapshot bytes, transfer count, report latency, and the hot-path effect boundary so the optimization cannot be mistaken for a full sleep scheduler or CUDA speedup of cognition.

`PredictiveColumnState` owns `prediction_failure_streak` beside prediction error and confidence. Repeated raw failures increment the streak on the predictive tensor device; successful prediction resets it. The streak is saved in trainer checkpoints and restored with the model, so growth evidence survives rollback and does not live in `service`.

When that gate is ready, the explicit binding-growth trial endpoint can ask core for a deterministic candidate-scoped hypercube outreach plan. The plan is bounded by an edge budget, hashes the exact baseline adjacency, and remains read-only. This narrows the path from surprise to structural experimentation without making the scheduler, Runtime Truth, or status polling a mutation authority.

`PredictiveColumnState` also records the last predictive update scope and the last predictive-vote execution scope. After the stale/deep-sleep horizon, the trainer candidate-scopes predictive updates on CPU so sleeping columns keep cached prediction state. For retained predictive voting, the trainer obtains routing candidates before consensus voting, then recomputes agreement only for the awake mask and reuses cached gains for the other columns. On CUDA predictive updates, synchronized microbenchmarks showed candidate indexing was launch-bound, so the trainer keeps dense predictive updates and reports `cuda_sparse_prediction_update_launch_bound_dense_retained` instead of making a decorative sparse-CUDA claim.

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

Also on 2026-06-15, the CUDA text path gained an opt-in conditional-WHILE sequence executor prototype below the trainer-owned burst boundary. It does not change column scheduling policy, but it matters for Column Runtime evidence because the same sequential SNN column state can now be advanced by a larger native CUDA Graph parent while Runtime Truth reports executor identity, token coverage, fallback/failure counts, host-truth cadence, and startup compile/capture cost. The retained repeated-child native replay remains the default until the conditional path passes repeated long-run promotion gates.

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

The in-place CUDA/Triton transition is now a checkpoint-opt-in production executor owned by `MarulhoTrainer`. Startup compiles the all-column and routed-candidate shapes without launching the mutating kernel. Unsupported configurations fall back before mutation; failures after launch fail closed. Runtime Truth exposes requested/resolved mode, warmup, candidate shapes, device, execution/failure counts, fallback, and policy.

Production-backed complete hot-window runs reached `80.59` and `110.36 ticks/sec` versus retained observations of `70.77` and `51.56 ticks/sec`. A synthetic visual/audio gate passed at `42.81` versus `38.78 ticks/sec` with exact winners and bit-exact cross-modal state. Empty-cache compile-only startup took `80.75 s`, while a populated cache reduced startup to `0.35 s`.

Stage profiling then separated the live loop. Concept observation consumed `5490.65 ms` of a `7292.26 ms` 12-token tick; normalized-centroid caching and bounded source-window sampling improved same-process live throughput to `7.84-8.44 tokens/sec`. Scheduling the existing remote refill worker after chunk consumption produced a warm 64-token tick with `0.04 ms` source collection and `7.97 tokens/sec`, proving source waiting can overlap CUDA work.

The next velocity gate is now inside the remaining `MarulhoTrainer.train_step` stages, which consumed `6601.45 ms` of that 64-token tick. The design target remains persistent device scheduling, event-driven specialist wake, bounded batching, and fusion where synchronized profiling proves launch or host/device pressure. Real camera/microphone grounding and thousands of full cognitive ticks per second remain unproven.

An explicit profiler slow path found about 112 CUDA launches, 40 async copies, and 16 stream synchronizations per encoded tick. With the in-place transition active, the transition itself was about `2.2 ms/tick`; dense predictive voting plus candidate competition, routing-key projection, and HNSW retrieval formed the larger remaining routing cluster at roughly `11 ms/tick`.

Candidate-scoped voting and a separately compiled dense vote both failed complete-hot-window A/B despite reducing isolated arithmetic. MARULHO therefore retains eager dense vote semantics, removes the unused persistent `hypothesis` tensor, and targets a broader fused/device-resident routing boundary. The retirement removed observed launches/copies without changing the vote result because no algorithm consumed the tensor.

That broader boundary is now implemented as checkpoint-opt-in Fused Text Route Vote. On the 1024-column RTX 3060 checkpoint, the two-launch exact-cache cluster matched all candidates, winners, and positive/silent decisions across 128 recurrent keys and improved isolated routing/vote from `415.16` to `1716.56 ticks/sec`. Production-owned reversed complete text/idle runs averaged `92.00` versus `66.80 ticks/sec` (`1.377x`). A serialized checkpoint then executed `144/144` fused text ticks with zero failures and nine cache refreshes at `82.40 ticks/sec`.

The specialization is deliberately modality-aware. A global sensory experiment preserved exact grounding but initially regressed; the promoted lifecycle executed zero fused routes on sensory ticks, reported `272/272` fallbacks in the reversed grounded gate, preserved every winner and cross-modal tensor, and passed the declared 0.90 no-regression floor. The default remains `tensor`; operators opt in through checkpoint configuration.

Skipping repeated routing normalization was also rejected because small floating-point changes diverged sequential winners and predictive locations. A narrower exact cleanup was promoted instead: the transition's already-materialized CPU winner ID now feeds HNSW buffering, avoiding one duplicate CUDA-to-CPU transfer and synchronization per tick without changing routing or learning state.

The next checkpoint-opt-in slice is `cuda_graph_text`. It captures production input normalization/projection, exact reconstruction distance, fused route/vote, and the in-place transition with fixed tensor addresses. A controlled 128-tick comparison was bit-exact across the sequential competitive, predictive, spike, input, and projection state. Three fresh-process hot-window arms averaged `264.46 ticks/sec` versus `176.24` for the fused path (`1.501x`), with graph median latency between `2.806` and `3.105 ms`. Runtime Truth on a real source tick recorded 24 graph replays and zero failures, but the 24-token tick still took `1240.473 ms`; remaining trainer/source orchestration is therefore the next velocity gate.

## Links

- [Runtime Truth](runtime-truth.md)
- [Metabolism](metabolism.md)
- [Hot Path](hot-path.md)
- [Dynamic Growth](dynamic-growth.md)
- [Pruning](pruning.md)
- [Core module](../modules/core.md)
