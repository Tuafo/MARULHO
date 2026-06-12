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

The current implementation is report-only. `core.column_runtime` reads existing predictive-column and competitive-column tensors, then reports:

- total columns and awake budget
- awake, cached-vote, sleeping, and deep-sleeping counts
- a sampled Column Registry with role, state, local prediction/surprise/usefulness/cost state, last-run token estimate, and optional memory budget
- bounded column votes with confidence, prediction error, usefulness, cost, disagreement, and wake reason
- growth-gate evidence from repeated prediction failure streaks, not one-shot surprise
- pruning/homeostasis evidence from weak, idle, or redundant columns
- the availability of a bounded single-column associative recall helper

The scheduler, cached-vote, growth, and pruning decisions remain report-only. Three narrower execution slices are promoted. First, when learned chunking makes projected input the routing key, retrieval runs before competitive scoring and only retrieved candidates are scored. Second, after stale/deep-sleep counters can exist, competitive win-rate/threshold homeostasis updates only the retrieved candidate set. Third, on CPU only, predictive error/confidence/failure-streak and high-prediction decay updates use the same delayed candidate set. Dense assembly remains active for representations whose routing key depends on it, and early learning keeps all-column homeostasis and predictive updates to preserve diversity. Runtime Truth reports observed scored count, candidate count, homeostasis update count/fraction, predictive update count/fraction, and CUDA fallback reason rather than inferring execution from configured budgets.

Column reporting is control-plane work. When source state is on CUDA, the report takes one bounded four-vector snapshot to CPU, plus the predictive-column prediction-failure streak vector when available, and computes scheduling/voting evidence there. This avoids repeated tiny CUDA kernels and scalar synchronization while preserving CUDA as the source device for live column state. Runtime Truth exposes source device, report compute device, snapshot bytes, transfer count, report latency, and the hot-path effect boundary so the optimization cannot be mistaken for CPU column execution or CUDA speedup of cognition.

`PredictiveColumnState` owns `prediction_failure_streak` beside prediction error and confidence. Repeated raw failures increment the streak on the predictive tensor device; successful prediction resets it. The streak is saved in trainer checkpoints and restored with the model, so growth evidence survives rollback and does not live in `service`.

When that gate is ready, the explicit binding-growth trial endpoint can ask core for a deterministic candidate-scoped hypercube outreach plan. The plan is bounded by an edge budget, hashes the exact baseline adjacency, and remains read-only. This narrows the path from surprise to structural experimentation without making the scheduler, Runtime Truth, or status polling a mutation authority.

`PredictiveColumnState` also records the last predictive update scope. After the stale/deep-sleep horizon, the trainer candidate-scopes predictive updates on CPU so sleeping columns keep cached prediction state. On CUDA, synchronized microbenchmarks showed candidate indexing was launch-bound, so the trainer keeps dense predictive updates and reports `cuda_sparse_prediction_update_launch_bound_dense_retained` instead of making a decorative sparse-CUDA claim.

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

On 2026-06-11, candidate-scoped competitive homeostasis became the first sleep-aware execution effect. The trainer keeps early learning all-column, then after `dead_column_steps` passes the retrieved candidate set into `CompetitiveColumnLayer.process()` so win-rate/threshold homeostasis updates only active candidates. Focused tests passed for frozen non-candidate homeostasis, delayed trainer promotion, learned-chunk diversity, and Runtime Truth execution fields. An 8192-column CPU/CUDA microbenchmark with 10 candidates measured all-column versus scoped process medians of `1.28555` to `1.08545 ms` on CPU and `5.06735` to `4.93965 ms` on CUDA; p95 changed from `4.3084` to `2.3779 ms` on CPU and `10.6108` to `9.195 ms` on CUDA.

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

The fixed-shape hot-window benchmark shows the eager Python core loop remains tens of tokens per second on the 1024-column RTX 3060 checkpoint before endpoint overhead. Tensor routing is promoted into live trainer routing; its merged exact CUDA cache reduces warm routing preparation from `37863.10` to `127066.06 tokens/sec`. Dense CUDA predictive state is one compiled functional transition with stable-buffer writeback and repeated-step parity; that narrower unit produced consistent full-loop gains despite first-use compilation cost.

The attempted wider functional transition spanning competition, prediction, winner prototype plasticity, and homeostasis did not pass promotion. Although its isolated compiled arm reached `271.3224 transitions/sec` versus `64.9176` eager, two controlled full hot-window pairs averaged about `34.75` compiled versus `36.65 tokens/sec` for the retained runtime. Eleven replacement state tensors plus stable-buffer copies erased the isolated gain. It remains an evaluation-only parity oracle.

The first in-place CUDA/Triton transition now exists as evaluation-only code. It mutates predictive state, winner prototype/velocity plasticity, candidate-scoped homeostasis, stale counters, spike history, and assembly in one launch while preserving all state tensor addresses. On the 1024-column RTX 3060 checkpoint it reached `314.5419 transitions/sec` versus `59.8240/sec` for functional eager stable-writeback, a `5.26x` transition-cluster gain with `28 MB` reserved VRAM and `10.38 s` first-process warmup. CUDA parity tests cover the complete mutated state.

The live transition boundary is now explicit in `MarulhoTrainer`, allowing evaluation to replace exactly the awake-column mutation cluster without adding a production configuration path. Two reversed-order complete hot-window pairs favored the in-place executor, reaching `55.28` and `105.40 ticks/sec` versus `38.02` and `37.95` for the retained runtime, with better median and p95 latency in both pairs.

Promotion is still blocked, but the quality uncertainty is narrower. A same-checkpoint 128-tick A/B measured `68.0983` in-place versus `46.5678` retained ticks/sec (`1.4623x`) with better median/p95 latency. Winner agreement was only `31.25%`, yet the faster trajectory reduced reconstruction and prediction error, preserved confidence, increased winner diversity, remained finite and sparse-responsive, and passed all declared spike-health and memory-quality gates. Exact specialist identity is therefore diagnostic, not a required invariant when aggregate quality holds.

A synthetic grounded visual/audio gate now passes on a cross-modal-enabled checkpoint. The stabilized in-place arm reached `27.4966` versus `22.9012 ticks/sec` (`1.2007x`), improved median/p95 from `38.3108/49.6470 ms` to `30.8081/38.4805 ms`, preserved winners exactly, and left all cross-modal weights/confidences bit-exact. Triton instrumentation found that transient winner-pointer alignment caused a second 111-second specialization during a measured tick; suppressing alignment specialization for only that argument reduced execution to one warmup compile and zero measured compiles.

The next velocity gate is a production cold-start/cache lifecycle for the Triton kernel plus Runtime Truth mode/fallback evidence. Until that passes, the executor stays evaluation-only. Real camera/microphone grounding also remains unproven; the current grounded evidence uses synthetic correlated sensory spikes. After promotion, remaining Python/CPU scalar boundaries should move into persistent device buffers or slower telemetry windows. `1000 ticks/sec` remains a low reference floor, not the target.

## Links

- [Runtime Truth](runtime-truth.md)
- [Metabolism](metabolism.md)
- [Hot Path](hot-path.md)
- [Dynamic Growth](dynamic-growth.md)
- [Pruning](pruning.md)
- [Core module](../modules/core.md)
