# ADR 0006: Persistent Text Tick Executor

## Status

Accepted

## Context

The promoted `cuda_graph_text` path still used two graph replays separated by
a reconstruction scalar readback and Python neuromodulator update. Profiling
at 1024 columns attributed material time to graph replay, scalar readback,
surprise updates, and routing preparation. The split prevented CUDA from seeing
the fixed-shape text transition as one workflow.

The runtime must preserve sequential cognition, checkpoint restoration,
pre-mutation fallback, sensory grounding, and operator-visible truth. Archival
memory, source handling, replay, checkpointing, and cross-modal sensory work
must remain outside this fixed-shape graph.

## Decision

For checkpoint-opted-in eligible text ticks, `ColumnTransitionRuntime` owns one
persistent CUDA Graph replay spanning input normalization and projection,
reconstruction error, reconstruction-driven neuromodulator update on
persistent CUDA state, sparse route/vote, and the in-place competitive and
predictive transition. Reconstruction error is derived from the fused exact
route-score maximum inside the captured route/vote phase, removing the earlier
separate dense prototype scan for this graph-owned path.

The executor also owns a bounded fixed-address CUDA input ring. Brain Runtime
offers already encoded tensors in sequential execution quanta; training stages
each quantum with one or two contiguous device operations, verifies tensor
pointer order as tokens are consumed, and lets the captured graph advance the
input slot and recent-spike-row cursor. A mismatch or sensory boundary discards
unconsumed staged entries and falls back before mutation. Checkpoint restore
seeds the device cursor from the restored competitive state.

The same transition owns exact device routing-cache coherence for eligible
graph ticks. It writes the normalized next winner prototype into the captured
retrieval cache through a prevalidated column-id-to-cache-row tensor. The
trainer skips the duplicate per-token routing-index enqueue and treats the CPU index
store as a slow mirror. Before a retained slow-path index mutation, training
synchronizes that host mirror from live prototypes without invalidating the
captured device cache.

After graph eligibility validates the captured consolidation tensor generation,
post-transition bookkeeping does not reacquire that tensor from the memory
store. It also reuses one persistent empty revival-id tensor instead of
allocating a new CUDA tensor on every token. Retained and sensory paths keep
their normal lookup behavior, and explicit structural maintenance may still
replace `last_revived_indices` with real revival evidence.

The executor copies one bounded nine-scalar result packet to the host on the
configured truth cadence
to mirror reconstruction, neuromodulator, winner, effective-plasticity, and
post-transition competitive-surprise truth. The CPU surprise history records
that last scalar from the existing readback rather than launching a separate
CUDA norm and scalar synchronization. Sensory ticks and graph-ineligible states
use the retained path. Pointer changes and graph failures fail before further
mutation.

Training may use a Boundary-Aware Text Burst for exactly eight eligible
text-only ticks. The burst replays the existing one-tick graph eight times in
causal order, while collapsing repeated Python bookkeeping. It cannot cross
sleep/replay, slow-memory admission, cross-modal wake, routing-index mutation,
host-truth, or routing-mode boundaries. Trainer telemetry is observation-only.
Drift refresh and drift-floor closure are CPU archival maintenance performed
after a bounded event drain without replaying the quantum token-by-token.
Strong capture is data-dependent inside the burst, so a separate one-tick
burst graph snapshots the result packet into a bounded device ring every tick
and copies assembly/routing evidence only for threshold crossings. Training
drains the ring once at the host-truth or maintenance boundary and stores all
archival payloads on CPU. Brain Runtime may request the burst, but training
owns eligibility, event admission, maintenance, and neural/bookkeeping
semantics.

When available, the burst executor may instantiate a parent CUDA graph whose
children are the already-captured one-tick burst graph repeated eight times.
This is not the rejected eight-tick PyTorch graph body: it does not recapture
the transition cluster, duplicate evidence copies, or widen the cognitive
boundary. It only lowers the host launch boundary from eight graph launches to
one parent-graph launch for an otherwise identical eligible burst. If native
graph construction or launch is unavailable before mutation, the executor uses
the retained Python replay loop; if native launch fails after selection, the
runtime fails closed rather than falling back after possible mutation.

Brain Runtime submits one complete prepared text tick through the
Training-Owned Text Sequence API. Training retains the ordered eight-token
quantum boundary, checks cancellation between quanta, chooses burst versus
retained per-token execution, and drains bounded device evidence before
returning. Service supplies source metadata and requested metric positions,
then performs concept observation and Runtime Truth projection outside neural
execution. Runtime Sources separately treats a fully prepared cache generation
as immutable under consumption, so ordinary ticks do not rebuild or hash the
remaining source queue.

Device float neuromodulation and Triton reductions may differ from Python
double scalar arithmetic by floating-point noise. Promotion requires exact
winners, bounded reconstruction tolerance, bounded sequential tensor tolerance,
cognitive-quality evidence, and grounded fallback gates.

## Consequences

- The mid-tick reconstruction synchronization and second replay are removed
  from eligible text ticks.
- Runtime Truth reports persistent replay, host truth synchronization,
  neuromodulator and competitive-surprise update, capture, failure, and
  fallback evidence.
- Runtime Truth reports the reconstruction source as `fused_route_score_max`,
  whether fused reconstruction is active, and the update count; these are
  scalar ownership fields, not additional per-token readbacks.
- Runtime Truth also reports device-owned routing-cache updates, skipped
  per-token index buffering, host-mirror synchronization, and mirror freshness.
- Runtime Truth reports quantum-input stage/reuse/fallback/mismatch/discard
  counters and device-owned recent-spike-row updates.
- Runtime Truth reports graph consolidation-lookup skips and persistent empty
  revival-tensor reuses, proving the allocation-free bookkeeping path executed.
- Runtime Truth reports device strong-event ring ownership, strong-event count,
  and a bounded burst fallback-reason histogram without adding per-token host
  synchronization.
- Current evidence shows a fresh-process `1.506x` complete hot-window gain and
  a text quality-gate `1.202x` gain with exact winners.
- Fresh-process graph memory increased by about `8.13 MB` allocated and `24 MB`
  reserved on the tested RTX 3060.
- End-to-end service velocity remains dominated by stages outside this graph.
- A reversed 256-tick same-process A/B measured `86.11` versus `70.21
  ticks/sec` (`1.226x`) after folding competitive surprise into the result
  packet. Sixteen-tick parity preserved winners, surprise history, and
  precision within `1e-7`.
- Two reversed 256-tick continuous CUDA A/B runs measured quantum-ring means
  of `1026.38` and `877.53 ticks/sec` versus per-token-copy means of `758.57`
  and `746.19`, for `1.353x` and `1.176x` gains. Both runs reused every staged
  token with zero fallback copies, mismatches, or graph failures.
- A 1024-sample current-over-clean-HEAD comparison with `PYTHONPATH` pinned to
  the baseline worktree measured fused reconstruction plus quantum staging at
  `796.22 tokens/sec` versus `630.17 tokens/sec` (`1.264x`) for the quantum
  arms, with `1088` graph replays, `1088` fused reconstruction updates, zero
  graph failures, and zero staged-input mismatches.
- Removing the duplicate post-replay consolidation lookup and empty CUDA
  allocation reduced profiled transition bookkeeping from `0.054251` to
  `0.014375 ms/tick`. Two clean 32768-token continuous-runtime runs reached
  `2169.815` and `2191.057 tokens/sec`, versus the retained `1779.859`
  reference, with zero graph failures and exact skip/reuse counts for every
  token.
- A larger graph containing eight copied tick bodies was rejected after the
  32768-tick device-only run measured `0.951x`; the larger graph added device
  scheduling and evidence-copy work to an already efficient one-tick replay.
- The accepted host-burst path kept all 32768 CUDA transitions and measured
  `2387.898` and `2607.316 tokens/sec` in repeated complete-runtime runs.
  Runtime Truth counted `18032` and `18016` burst-owned tokens with zero burst
  replay failures.
- After preserving data-dependent strong events, final repeated 32768-token
  runs measured `2648.747`, `2533.719`, and `2599.013 tokens/sec`. The final
  run executed all `32768` transitions on `cuda:0`, used `9760` burst tokens,
  reported zero graph/burst failures, and exposed fallback counts:
  host truth `1033`, exploration `581`, drift refresh `441`, telemetry `369`,
  and drift floor `1`. Graph capture startup was `480.524 ms`.
- The Device-Owned Cognitive Boundary Controller later removed exploration,
  telemetry, drift-refresh, and drift-floor closure as execution gates.
  Inspection proved the former exploration noise scalar was never consumed by
  routing, plasticity, curiosity, or action, so it and its checkpoint field
  were deleted while device-owned norepinephrine/surprise dynamics remain.
  A clean 32768-token run reached `2745.790 tokens/sec` with `29136`
  burst-owned tokens and no old boundary fallbacks. A second sample at
  `2027.181` was rejected as promotion evidence because measurement began with
  source prewarm still running. The longer clean 131072-token gate reached
  `2126.013 tokens/sec` over `61.652 s`, executed all `131072` transitions on
  the RTX 3060, used `116696` burst-owned tokens, reported zero graph/burst
  failures, and retained only two real `sleep_boundary` fallbacks.
- The Prepared Source Generation and Training-Owned Text Sequence follow-up
  removed the largest remaining host preparation tax. The clean 131072-token
  CUDA run reached `3359.378 tokens/sec` over `39.017 s`, up `58.0%` from the
  prior `2126.013` long baseline. Mean tick latency fell from `57.714` to
  `35.667 ms`, p95 from `87.300` to `48.458 ms`, and preparation fell from
  `0.156315` to `0.008415 ms/token`. Runtime Truth reported `1024` complete
  training-owned sequences, `16384` quanta, `1024` stable cache-generation
  skips, all `131072` transitions on the RTX 3060, `116680` burst-owned
  tokens, and zero graph/burst failures.
- Device-Burst Lightweight Metrics keep ordinary prepared source ticks on the
  burst path instead of requesting full per-token `train_step` metrics for
  source concept samples. Full metrics remain available for explicit evaluator
  evidence positions, but normal service ticks use the final CUDA result packet
  as bounded Runtime Truth. The 131072-token CUDA service run improved to
  `3565.968 tokens/sec`, reduced `train_compute` to `0.241568 ms/token`, and
  raised burst-owned tokens to `126952` with zero graph/burst failures.
- Fixed slow-memory cadence is no longer a burst execution fallback. The
  hot path still admits first-token retained records and strong/surprise burst
  events, but ordinary cadence ticks are reported as deferred maintenance rather
  than forcing CPU replay-memory admission. The 131072-token CUDA service run
  reached `3901.906 tokens/sec`, reduced `train_compute` to
  `0.219858 ms/token`, executed `131056` burst-owned tokens, reported `512`
  deferred slow-memory cadence events, and retained zero graph/burst failures.
- Drift refresh and drift-floor closure no longer force burst event drains.
  Burst ticks refresh CPU drift evidence without synchronizing for a fresh
  winner mirror; when the mirror is stale, refresh uses global drift and reports
  that in Runtime Truth. The 131072-token CUDA service run reached
  `4045.419 tokens/sec`, reduced `train_compute` to `0.211096 ms/token`,
  reduced host-truth syncs to the configured cadence (`8193`), and recorded
  zero forced burst-event drains with zero graph/burst failures.
- The maintained service execution quantum is now `16` tokens, but the
  persistent CUDA burst executor remains an exact eight-token device boundary
  inside `MarulhoTrainer.train_text_sequence`. This retires the earlier
  q16 footgun where a wider service quantum bypassed the burst path and fell
  through to per-token `train_step`. The 131072-token CUDA service run reached
  `4247.306 tokens/sec`, `train_compute=0.200979 ms/token`, `8192`
  training-owned quanta, `16382` eight-token burst replays, all `131072`
  transitions on CUDA, and zero graph/burst failures.
- The native repeated-child parent graph is promoted for the current CUDA path.
  A 131072-token long run reached `4671.202 tokens/sec` with
  `train_compute=0.177193 ms/token`, all transitions on the RTX 3060,
  `16382` native parent-graph launches covering `131056` burst-owned tokens,
  zero native fallbacks/failures, zero graph/burst failures, and no observed
  CPU/GPU contention. The same long command with native replay disabled reached
  `4340.160 tokens/sec` and `train_compute=0.192680 ms/token`. The retained
  best prior long run was `4577.595 tokens/sec`, so this is a small but real
  new sustained-throughput ceiling.
- The refreshed base comparison on 2026-06-15 keeps that decision current:
  the native path reached `4992.049 tokens/sec` with
  `train_compute=0.166575 ms/token`, while the same shape with native replay
  disabled reached `4530.883 tokens/sec` and `0.185263 ms/token`. Both runs
  processed `131072` tokens on RTX 3060 with no observed contention, zero
  graph/burst failures, and `131056` burst-owned tokens.
- A startup-warmed exact sixteen-token native parent graph remains rejected as
  the default. The opt-in run covered `131040` tokens with `8190` native
  parent-graph launches, exposed parent graph token-count coverage `[16]`,
  preserved the host-truth cadence at `32`, and reported zero native,
  graph, or burst failures. It still reached only `4887.767 tokens/sec` with
  `train_compute=0.168278 ms/token` on the clean 131072-token gate, below the
  refreshed eight-token ceiling at `4992.049` and `0.166575`. Startup cost
  stayed visible outside measured warm throughput:
  `capture_latency_ms=6112.8292` and
  `native_burst_replay_compile_latency_ms=5609.5473`. Contended 8192-token
  profile pairs showed that doubling the parent capacity mainly halves parent
  launches (`1023` to `511`) while moving
  `text_burst_runtime_replay_loop` only from `0.159096` to
  `0.149719 ms/token`, so the next executor must move below repeated
  child-graph wrapping into C++/CUDA/Triton persistent or hybrid sequence
  ownership.
- A native32 probe under the maintained q16 execution quantum is invalid as a
  native coverage benchmark. The run warmed `[32]` parent graphs and exposed
  `persistent_executor_burst_tokens=32`, but q16 service/training quanta only
  offered sixteen-token bursts. Runtime Truth correctly reported
  `native_burst_replay_success_count=0`,
  `native_burst_replay_fallback_count=8190`,
  `native_burst_replay_python_loop_token_count=131040`, and backend
  `python_loop_partial_disabled`. The stress benchmark now rejects native
  burst capacities that exceed or fail to divide `quantum_tokens`; proving
  native32 would require changing the execution quantum boundary, which is a
  separate retired/default-sensitive trade-off rather than the exact fast
  q16 shape.
- A CUDA conditional-WHILE parent graph was added as a lower-level sequence
  executor candidate. It keeps the proven one-tick graph body but
  moves burst-loop control into a CUDA conditional node plus a device counter
  kernel. The clean 131072-token q16/native16 probe at
  `reports/conditional_sequence_20260615/conditional-while16-131072-i32.json`
  measured `5559.473 tokens/sec`, `train_compute=0.146978 ms/token`,
  `8190` conditional parent launches, `131040` conditional-owned tokens, zero
  sequence/native fallbacks, zero sequence/native failures, host-truth cadence
  `4097/126975`, and `velocity_environment.v1` contention `not_observed`.
  This beat the retained same-session native8 rerun at `5035.537 tokens/sec`
  and triggered the repeated paired promotion gate.
- The conditional-WHILE executor is promoted for eligible q16 text sequences
  after repeated clean paired gates and a post-promotion default run. The clean
  pairs measured conditional q16 at `5883.805` versus native8 `5485.105`
  tokens/sec, then conditional q16 at `6027.856` versus native8 `5816.477`
  tokens/sec in reversed order. The default post-promotion run reached
  `6116.646 tokens/sec` with `train_compute=0.134167 ms/token`,
  `8190` conditional launches covering `131040` tokens, zero sequence/native
  fallbacks or failures, host-truth cadence `4097/126975`, startup capture
  `5482.6059 ms`, conditional compile `4970.7865 ms`, and
  `velocity_environment.v1` contention `not_observed`. The former native8
  comparison run stayed clean at `5329.542 tokens/sec`.
- The promotion separates capacities: `cuda_graph_sequence_loop_tokens=16`
  owns the conditional sequence loop, while `cuda_graph_native_burst_tokens=8`
  remains the maintained repeated-child parent capacity for internal fallback.
  This avoids promoting the rejected native16 repeated-child path by changing
  the wrong default.
- ADR 0007 captures the follow-on boundary: the next promotable text executor
  must move below local CUDA Graph wrappers into a lower-level sequence owner.
- Wider event/truth cadences remain rejected. A sixty-four-token event window
  cut long-run drains from `4096` to `2049`, but measured only
  `4402.958 tokens/sec` in the clean `131072`-token run versus `4771.221`
  for the retained thirty-two-token repeat and `4992.049` for the refreshed
  native base comparison. Fewer host truth packets alone is not a promotion.
- The C++ loop over `cudaGraphLaunch(graph_exec)` is rejected as a promotion
  path. It moved the loop below Python, but still launched once per token and
  lost the 131072-token comparison (`4159.316` native-loop versus `4347.554`
  disabled Python replay under the recorded runs).
- Startup/capture cost increased when native parent graphs are built. The
  131072-token promoted run reported `capture_latency_ms=6790.4858` and
  `native_burst_replay_compile_latency_ms=6202.4909`. This is a startup
  slow-path cost, not measured token throughput, and must remain visible in
  Runtime Truth.
- Partial native parent graphs for non-eight-token burst tails remain opt-in.
  A 16640-token stress shape with `tick_tokens=130` proved the opt-in path can
  cover two-token tails with native parent graphs (`parent_graph_token_counts`
  `[2, 8]`, `lazy_compile_count=2`, zero native fallbacks), but complete
  runtime regressed to `2786.829 tokens/sec` versus `2907.600` with the default
  partial Python replay fallback. The same run exposed the bigger hazard:
  unaligned tick width forced `384` host-truth burst fallbacks. The maintained
  fast path therefore keeps exact eight-token native replay plus aligned
  128-token source ticks. The later cleanup removed partial native replay as a
  live diagnostic switch; non-eight-token tails report explicit Python-loop
  fallback truth until a future lower-level multi-tick executor exists.

## Reversal

Set `predictive_route_vote_mode` to `fused_triton_text` or `tensor`, or let
eligibility fail closed. Set `cuda_graph_quantum_input_staging=false` to retain
the persistent graph while restoring exact per-token input copies. Set
`cuda_graph_native_burst_replay=false` or
`MARULHO_CUDA_GRAPH_NATIVE_BURST_REPLAY=0` to keep the current persistent graph
while restoring the retained Python `CUDAGraph.replay()` loop. Checkpoints
preserve the retained execution paths. Set
The promoted conditional-WHILE sequence executor is fixed for eligible CUDA text
sequences; repeated-child native8 replay is retained only as internal
pre-mutation fallback when conditional construction is unavailable.
`cuda_graph_native_burst_tokens` is fixed at `8` for the maintained
repeated-child capacity; old checkpoint values are migrated back to `8`.
`cuda_graph_sequence_loop_tokens` is fixed at `16` for the promoted conditional
sequence capacity. Repeated-child `16`/`32`, conditional-WHILE q8/q32, and the
old sequence-executor selector are retired historical benchmarks; a separate
clean long-run gate must introduce a new reviewed executor path rather than
restoring old capacity or selector knobs.
