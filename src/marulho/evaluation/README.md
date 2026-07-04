# Evaluation

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`evaluation` owns promotion gates, benchmarks, readiness checks, and validation
harnesses.

## Owns

- Evidence standards for speed, readiness, CUDA placement, liveness, and
  promotion.
- Same-checkpoint comparison runners and sustained stress checks.
- Slow-path environment snapshots that separate architecture changes from
  CPU/GPU contention.
- The maintained service endpoint benchmark over `/health` and `/brain/*`.

## Must Not Own

- Runtime mutation.
- Production status verdicts not backed by current reports.

## Runtime Rules

- Benchmark complete runtime behavior when making throughput claims. Separate
  setup/compile overhead from steady-state token throughput.
- `service_benchmark.py` measures the thin brain adapter only; it must not
  turn non-`/brain/*` HTTP surfaces into performance gates.
- Hot-window reports may profile trainer stages, but profiling is evidence and
  must not become ordinary runtime work.
- CUDA claims need observed backend/device evidence, fallback counts, and
  failure counts. Treat `torch_sequence_graph_*` evidence as distinct from
  native conditional-WHILE evidence; do not merge the counters.
- Regressions should preserve exact failing repros and rejected reports rather
  than only reporting the winning run.

## Current Validation Snapshot

- `service_benchmark.py`, `continuous_runtime_quantum_benchmark.py`, and
  `continuous_runtime_stress_benchmark.py` now target `MarulhoBrain` or
  `/brain/*`.
- `source_tick_sleep_deferral_benchmark.py` builds `MarulhoBrain` directly. The
  retained 2026-06-30 local evidence passed with brain tick sleep calls `0`,
  sequence sleep calls `0`, explicit slow-path sleep calls `1`, and
  `runtime_owner=MarulhoBrain`.
- The retained 2026-06-30 long sequence-input staging gate reached `6601.19`
  sequence tokens/sec versus `6507.41` per-quantum tokens/sec. It used
  `cuda_graph_route_transition_burst`, backend `cuda_graph_conditional_while`,
  device `cuda:0`, and zero graph/native/burst failures.
  On PyTorch builds without `torch.cuda.CUDAGraph.raw_cuda_graph()`, current
  runs must report `torch_sequence_graph_*` separately instead of claiming this
  native backend.
- Continuous stress reports at `256`, `1024`, and `4096` tokens are smoke/debug
  history only. They passed through the same conditional-WHILE CUDA backend
  with zero graph/native/burst failures, but they are not promotion evidence.
- The sustained runtime evidence ladder is `8192` tokens for diagnostic
  evidence, `131072` tokens for the normal long-run promotion gate, and
  `524288` tokens for the preferred house-scale target when hardware/runtime
  budget allows. Promotion is not allowed from `256`, `1024`, or `4096`.
- `continuous_runtime_stress_benchmark.py` must write a final or partial JSON
  report for success, timeout, exception, interrupt, and manual stop. A report
  must include target tokens, token delta, elapsed time, tokens/sec when
  measurable, checkpoint, runtime owner, tick/quantum tokens, final/last
  `BrainTrace`, device report, CUDA/backend/executor evidence, graph/native/
  burst/sequence failure and fallback counters, event summary, and environment
  contention summary.
- `language_sustained_runtime_evidence.py` applies the same final/partial JSON
  discipline to the checkpointed `marulho_lm_head` component. It streams the LM
  recurrent cache, writes JSON plus README mirrors for final, timeout,
  manual-stop partial, interrupt, and exception outcomes, and reports
  checkpoint metadata, active routed columns, spike health, device/backend,
  fallback counts, environment contention, and promotion gates. CUDA runs now
  attempt ordered `torch_cuda_graph_burst` replay at the configured quantum,
  record graph replay/token counts and setup time, and fail back to eager
  streaming before hiding graph failures. It is component evidence only; the
  LM path remains `promotes_hot_path=false` until generation quality and
  Triton/CUDA kernel parity evidence exist.
- `language_training_experiment.py` is the fast mutable LM experiment runner.
  It trains a configurable MARULHO-owned routed selective-spiking LM on local
  text using packed device-resident windows, records training throughput,
  heldout loss/perplexity before and after training, owned generation samples,
  source-continuation quality probes, a checkpoint, and paired sustained
  inference reports. Its CUDA update loop keeps per-batch loss and gradient
  norm metrics as device scalars, performs one CUDA synchronization before
  stopping the measured training timer, and reads aggregate scalars back after
  the hot update window. It supports padded model vocabularies, sampled vocab
  loss, sparse vocab-row optimization, and tokenizer-row generation decode
  limits. It is meant to accelerate model experiments, not create a new
  promotion gate.
- `language_sampled_vocab_training_impact.py` measures complete sampled/
  adaptive vocabulary training-step impact for padded large-vocab LM configs.
  It compares dense full-vocab loss/optimizer work against sampled loss with
  sparse token-embedding and LM-head row gradients, records throughput, CUDA
  memory, and sampled-vocab CE backend counters, and keeps runtime/generation-
  quality promotion blocked until checkpoint, decode, long-run, and review
  evidence are separately proven.
- `language_sustained_runtime_evidence.py` records padded-vocab decode policy
  for checkpointed LM runs. Padded-vocab checkpoints must carry an explicit
  `generation_vocab_size`; sustained generation uses decode-limited logits so
  extra model rows cannot be emitted as tokenizer bytes.
- `language_generation_coherence.py` is the grounded prompt-suite review for
  checkpointed MARULHO-owned generation. It records raw continuations,
  source-prefix match, next-character source match, printability, token-run and
  bigram-diversity checks, active language path, and external-LLM absence. It
  can satisfy the benchmark suite's generation-coherence category, but it is
  not a human review, a broad generation-quality claim, or a runtime-promotion
  claim.
- `language_scale_ladder.py` defines the MARULHO LM target scale classes and
  writes JSON plus README evidence inventories. It estimates total parameters,
  active parameters per token, routed-column budgets, dense vocab-head cost, and
  memory footprint for the small fixture, 140M-class, 500M-class, 0.9B-class,
  and 2B+ research ladders. Large ladder entries are not instantiated or
  promoted by the report; their gate stays `configuration_defined_not_trained`
  until training, long-run, forgetting, kernel, restore, and generation-review
  evidence exists.
- `language_runtime_benchmark_suite.py` aggregates MARULHO LM-head benchmark
  evidence into one JSON plus README report. It covers next-token loss, heldout
  perplexity, generation smoke, grounding support, continual learning,
  forgetting, replay recovery, growth/prune safety, long-run throughput, active
  compute, GPU kernel correctness, checkpoint restore, rollback, service
  contract, and scale-ladder inventory. The suite writes a grounding-support
  source-term coverage subreport and can ingest existing final
  `marulho_language_sustained_runtime_evidence.v1` reports for the 8192/131072
  LM long-run gates, existing `marulho_language_triton_kernel_report.v1`
  reports for kernel correctness, and existing
  `marulho_language_generation_coherence_report.v1` reports for grounded
  prompt-suite coherence. Human review and broad generation-quality/runtime
  promotion remain false unless separately proven.
- Current 2026-07-03 fixed evidence:
  `reports/runtime_evidence_20260703/diagnostic-8192-after-feed-readout-fix.json`
  reached `8192/8192` tokens at `3120.356 tokens/sec`, mean tick
  `21.123 ms`, and p95 `19.287 ms`. GPU contention was observed in this
  diagnostic run. The normal long gate
  `reports/runtime_evidence_20260703/long-gate-131072-after-feed-readout-fix.json`
  reached `131072/131072` tokens at `5608.147 tokens/sec`, mean tick
  `17.800 ms`, p95 `20.073 ms`, CUDA RTX 3060, `conditional_while`, zero CUDA
  graph/native/sequence failures or fallbacks, bounded `12/65536` route rows,
  no all-column state transition, `brain_feed_streaming_refill` with `16` feed
  calls and zero source drops, and contention `not_observed`. The house-scale
  gate
  `reports/runtime_evidence_20260703/house-scale-524288-after-feed-readout-fix.json`
  reached `524288/524288` tokens at `5877.601 tokens/sec`, mean tick
  `17.445 ms`, p95 `19.358 ms`, with the same CUDA backend, zero CUDA graph/
  native/sequence failures or fallbacks, bounded `12/65536` route rows, no
  all-column state transition, `brain_feed_streaming_refill` with `64` feed
  calls and zero source drops, and contention `not_observed`.
- Current 2026-07-03 LM-head component evidence from
  `reports/language_training_experiments/cuda-exp-8192-checkpoint.pt` uses
  `torch_cuda_graph_burst` on `cuda:0` with `16`-token ordered bursts and no
  graph failures: `cuda-exp-8192-graph-cuda-sustained.json` reached
  `8192/8192` at `4853.244 tokens/sec`, `cuda-exp-131072-graph-cuda-sustained.json`
  reached `131072/131072` at `6898.430 tokens/sec`, and
  `cuda-exp-524288-graph-cuda-sustained.json` reached `524288/524288` at
  `6978.602 tokens/sec`. `language-suite-graph.json` accepts the long-run
  throughput category, including the house-scale report, but keeps promotion
  blocked on `generation_coherence` review and `gpu_kernel_correctness`
  Triton/parity evidence.
- Current 2026-07-03 batched LM training evidence in
  `reports/language_training_experiments/cuda-batched-quality-8192.json`
  trained `63744` tokens at `1240.010 train tokens/sec` with `batch_size=16`
  and `1024` tokens per optimizer step, versus the earlier single-window CUDA
  experiment at `74.551 train tokens/sec`. Heldout loss moved from `5.7651` to
  `0.2381`, heldout perplexity from `318.9603` to `1.2689`, printable
  generated continuation fraction reached `1.0`, and supported prompts moved
  from `0.0` to `1.0` next-character source-continuation match rate. The same
  trained checkpoint sustained `131072` tokens at `6938.880 tokens/sec` and
  `524288` tokens at `6996.703 tokens/sec` with `torch_cuda_graph_burst`.
  Generated continuations still show fractured local-corpus memorization, so
  this is training/quality-probe evidence, not a generation-coherence
  promotion.
- Current 2026-07-03 RMSNorm Triton evidence in
  `reports/language_kernel_evidence/rmsnorm-triton-20260703.json` passed six
  CUDA shape/dtype parity cases for `language_rmsnorm_forward` (`float32` and
  `float16`) with geometric microbenchmark speedup `1.440x` over the PyTorch
  RMSNorm expression. A forced one-token streaming diagnostic
  `cuda-batched-quality-rmsnorm-triton-8192-sustained.json` proved graph
  capture and zero Triton failures but regressed throughput to
  `4528.675 tokens/sec`, so the maintained model policy uses Triton only for
  batched rows and keeps one-token streaming on CUDA graph/PyTorch fallback.
  The short fallback-policy diagnostic
  `cuda-batched-quality-rmsnorm-policy-8192-sustained.json` reached
  `8192/8192` at `4346.188 tokens/sec` and is diagnostic only. The accepted
  house-scale policy run
  `cuda-batched-quality-rmsnorm-policy-524288-sustained.json` reached
  `524288/524288` tokens at `7003.004 tokens/sec`, CUDA graph burst, zero graph
  failures, zero Triton failures, and no observed environment contention.
  `language-suite-rmsnorm-kernel.json` now records `long_run_throughput=pass`
  and `rmsnorm_triton_parity=true` while keeping generation coherence and the
  then-remaining PLIF/scan/expert/vocab kernel parity blockers open.
- Current 2026-07-03 PLIF forward Triton evidence in
  `reports/language_kernel_evidence/plif-forward-triton-20260703.json` passed
  six CUDA shape/dtype parity cases for `language_plif_forward` (`float32` and
  `float16`) on the RTX 3060 with geometric microbenchmark speedup `3.145x`
  over the PyTorch PLIF forward reference. The kernel covers no-grad membrane,
  spike, selective-state, eligibility-trace, and mixed-state updates.
  `language-suite-plif-forward-kernel.json` records both
  `rmsnorm_triton_parity=true` and `plif_triton_forward_parity=true`, keeps
  long-run throughput available through the current `524288` LM sustained
  report, and left PLIF backward surrogate, selective-scan, expert-dispatch,
  sampled-vocab cross-entropy, and generation coherence as blockers.
- Current 2026-07-03 PLIF surrogate-backward Triton evidence in
  `reports/language_kernel_evidence/plif-surrogate-triton-20260703.json` passed
  three CUDA `float32` shape sweeps for `language_plif_surrogate_backward` with
  geometric forward+backward microbenchmark speedup `1.662x`. `float16`
  backward is explicitly unsupported until gradient parity is proven. The
  same state-block training scale in
  `reports/language_training_experiments/cuda-plif-surrogate-8192.json`
  trained `63744` tokens at `2596.380 train tokens/sec`, used `3840` Triton
  PLIF forward calls and `3840` Triton backward calls with zero failures,
  improved heldout loss from `5.8299` to `0.1907`, and sustained `8192` tokens
  at `6158.059 tokens/sec`. The paired house-scale report
  `cuda-plif-surrogate-524288-sustained.json` reached `524288/524288` at
  `7578.052 tokens/sec`, CUDA graph burst, zero graph failures, and no
  external LLM. `language-suite-plif-surrogate-impact.json` records RMSNorm,
  PLIF forward, and PLIF surrogate-backward parity while keeping promotion
  blocked on generation coherence plus selective-scan, expert-dispatch, and
  sampled-vocab cross-entropy evidence.
- Current 2026-07-04 deferred-metric CUDA training evidence in
  `reports/language_training_experiments/cuda-deferred-metrics-8192.json`
  trained the same PLIF-surrogate `63744` token shape at
  `2720.929 train tokens/sec`, versus the earlier `2596.380 train tokens/sec`
  per-batch-readback report. It records
  `metric_readback_mode=deferred_gpu_scalar_aggregation`,
  `per_batch_metric_cpu_sync=false`,
  `cuda_synchronized_before_timing_start=true`,
  `cuda_synchronized_before_timing_stop=true`,
  and `3840` Triton PLIF backward calls. The paired house-scale sustained
  report `cuda-deferred-metrics-524288-sustained.json` reached `524288/524288`
  at `7502.156 tokens/sec` on `torch_cuda_graph_burst`, `cuda:0`, with no
  external LLM and zero graph failures. This is training-loop host-sync
  reduction evidence; it does not promote inference speed or runtime quality.
- Current 2026-07-04 selective-scan Triton evidence in
  `reports/language_kernel_evidence/selective-scan-triton-20260704.json`
  passed six CUDA shape/dtype parity cases for
  `language_selective_state_scan` (`float32` and `float16`) at 64 recurrent
  steps on the RTX 3060 with geometric microbenchmark speedup `114.077x` over
  the PyTorch recurrent-loop reference. The kernel covers the standalone
  selective recurrence `state[t] = decay[t] * state[t-1] + input[t] * spike[t]`
  over `[batch,time,state_dim]` tensors with PyTorch fallback and runtime-use
  counters. `language-suite-selective-scan-kernel.json` records RMSNorm, PLIF
  forward, PLIF surrogate-backward, and selective-scan parity while keeping
  promotion blocked on generation coherence plus expert-dispatch and
  sampled-vocab cross-entropy evidence.
- Current 2026-07-04 expert-dispatch Triton evidence in
  `reports/language_kernel_evidence/expert-dispatch-triton-20260704.json`
  passed three CUDA `float32` shape sweeps for
  `language_block_sparse_expert_dispatch` with geometric microbenchmark
  speedup `4.389x` over the PyTorch selected-expert dispatch/combine reference.
  `float16` dispatch is explicitly unsupported until numerical parity is
  proven. The kernel covers selected two-layer expert MLP dispatch for
  `[token,state_dim]` hidden rows, selected expert IDs, combine weights, and
  stacked expert parameters. `language-suite-expert-dispatch-kernel.json`
  records RMSNorm, PLIF forward, PLIF surrogate-backward, selective-scan, and
  expert-dispatch parity while keeping promotion blocked on generation
  coherence plus sampled-vocab cross-entropy evidence.
- Current 2026-07-04 sampled-vocab CE Triton evidence in
  `reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json`
  passed three CUDA `float32` shape sweeps for
  `language_sampled_vocab_cross_entropy` with total vocab `8192`, sampled vocab
  `1024`, and geometric microbenchmark speedup `1.047x` over the PyTorch
  selected-vocab CE reference. `float16` sampled-vocab CE is explicitly
  unsupported until numerical parity is proven. The kernel covers forward loss
  for `[token,state_dim]` hidden rows, selected vocabulary IDs, LM-head
  weight/bias rows, and target IDs that must be present in the sample.
  The forceable Triton-forward/custom-autograd training path has sparse
  LM-head row-gradient parity, but b16/r8 sampled-only complete-runtime
  evidence kept the default on selected-row PyTorch autograd because it was
  faster (`2675.442` versus `2622.292` train tokens/sec).
  `language-suite-sampled-vocab-kernel.json` records GPU kernel correctness as
  `pass` across RMSNorm, PLIF forward, PLIF surrogate-backward, selective-scan,
  expert-dispatch, and sampled-vocab CE while keeping promotion blocked on
  generation coherence review.
- Current 2026-07-04 sampled-vocab training-impact evidence in
  `reports/language_training_experiments/sampled-vocab-training-impact-default-policy-524288-b16-r8-sampled-only.json`
  measures full MARULHO LM training steps, not a kernel microbenchmark. It uses
  a `524288` row model vocabulary, `1024` sampled vocabulary rows, `batch=16`,
  `seq=64`, warmup `2`, repeats `8`, backward, sparse-aware gradient clipping,
  and optimizer steps on `cuda:0`. The sampled arm avoids full vocab logits,
  uses sparse token-embedding and LM-head weight gradients with
  `AdamW_dense_core_plus_SparseAdam_vocab_rows`, stays on selected-row PyTorch
  autograd, reaches `2675.442` train tokens/sec, peaks at `2368.205 MiB` CUDA
  allocation, and records `8` measured selected-row fallback calls with zero
  Triton CE training calls. The report keeps `promotes_runtime_claim=false`.
- Current 2026-07-04 sampled-vocab precompute evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288-63744.json`
  moves sampled row ID and target-position construction out of the timed
  training update window for fixed packed batches. It keeps the same `524288`
  model-vocab, `1024` sampled-row, horizon-8, TF32, clip-8 shape, records
  `sampled_vocab_precompute.enabled=true`, trains `63744` tokens at `3041.246`
  train tokens/sec, improves forward/loss from `0.125210` to
  `0.121173 ms/token`, and lowers batch total from `0.333647` to
  `0.328424 ms/token` versus the retained all-awake route fastpath report. The
  paired `524288` sustained run reached `7203.369` tokens/sec; a same-checkpoint
  current-code sustained rerun of the retained all-awake checkpoint reached
  `7206.201` tokens/sec, so the evidence supports the training speed slice and
  not a new inference-speed claim.
- Current 2026-07-04 integrated sampled/padded training experiment evidence in
  `reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744.json`
  uses the normal LM experiment runner with `524288` model vocab rows, `1024`
  sampled rows, `262` tokenizer/generation rows, `524026` padded rows masked,
  `batch=16`, `seq=64`, `stride=32`, and `4` train epochs. It trains `63744`
  tokens at `2361.928` train tokens/sec with
  `AdamW_dense_core_plus_SparseAdam_vocab_rows`, selected-row PyTorch autograd,
  avoids full vocab logits, improves heldout loss from `7.1612` to `0.1997`,
  records source-continuation probes, saves a checkpoint, and sustains
  `524288/524288` tokens at `7273.947` tokens/sec on `torch_cuda_graph_burst`.
  It keeps
  `promotes_runtime_claim=false` and `promotes_generation_quality_claim=false`.
- Current 2026-07-04 LM training stage profile evidence in
  `reports/language_training_experiments/cuda-sampled-padded-stage-profile-524288-63744.json`
  uses the same `524288` model-vocab, `1024` sampled-row, `batch=16` training
  shape with `--profile-training-stages`. It trains `63744` tokens at
  `2342.586` train tokens/sec, improves heldout loss from `7.0637` to
  `0.1827`, sustains `524288/524288` generated tokens at `7244.434`
  tokens/sec, and records CUDA-event timings without synchronizing each stage
  inside the hot loop. The measured cost order is backward
  (`0.245897 ms/token`), forward/loss (`0.150728 ms/token`), sparse-aware
  gradient clipping (`0.020030 ms/token`), then optimizer step
  (`0.007629 ms/token`), so the next speed work should target state-block/PLIF
  backward and forward/loss before optimizer-step fusion.
- Current 2026-07-04 recurrent-gradient horizon evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-profile-524288-63744.json`
  uses horizon `8` on the same `524288` model-vocab, `1024` sampled-row,
  `batch=16` shape. It keeps recurrent forward state causal on CUDA but
  detaches recurrent state every `8` tokens during gradient training, records
  `7` detach boundaries per 64-token batch, trains `63744` tokens at
  `2435.816` train tokens/sec, improves heldout loss from `7.1213` to
  `0.2303`, and sustains `524288/524288` generated tokens at `7225.847`
  tokens/sec. Compared with the full-sequence profiled run, training throughput
  improves `3.980%`, batch-total cost falls by `0.016323 ms/token`,
  forward/loss falls by `0.009386 ms/token`, and backward falls by
  `0.006034 ms/token`; backward remains the largest remaining stage.
- Current 2026-07-04 CUDA math-policy evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-profile-524288-63744.json`
  runs the horizon-8 shape with explicit experiment-scoped TF32. The report
  records that the process started with `matmul_allow_tf32=false`, applied
  `matmul_allow_tf32=true`, `cudnn_allow_tf32=true`, and
  `float32_matmul_precision=high`, then restored the previous policy after the
  run. It trains `63744` tokens at `2487.980` train tokens/sec, improves
  heldout loss from `7.0918` to `0.2066`, and sustains `524288/524288`
  generated tokens at `7239.247` tokens/sec. Compared with horizon 8 without
  TF32, training throughput improves `2.142%`; compared with the full-sequence
  profiled baseline, it improves `6.207%`. This is a CUDA math speed policy,
  not bitwise-float32 equivalence or generation-quality promotion.
- Current 2026-07-04 gradient-clip cadence evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-profile-524288-63744.json`
  runs the horizon-8 TF32 shape with `--gradient-clip-interval 8`. It clips
  `8/64` optimizer updates, skips `56`, trains `63744` tokens at `2538.756`
  train tokens/sec, improves heldout loss from `7.0764` to `0.2083`, and
  sustains `524288/524288` generated tokens at `7223.490` tokens/sec. Compared
  with the horizon-8 TF32 every-step clip run, training throughput improves
  `2.041%`; compared with the full-sequence profiled baseline, it improves
  `8.374%`. The gradient-clip stage falls to `0.002918 ms/token`, while
  backward remains the largest stage.
- Current 2026-07-04 batched state-output projection evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-batched-state-output-524288-63744.json`
  keeps the causal membrane/spike/selective-state recurrence ordered, but
  applies `state_output_proj` once over the stacked `[batch,time,state]`
  mixed-state tensor. The report records
  `state_block_projection_mode=batched_token_and_state_output_projection_recurrent_loop`,
  trains the same `524288` model-vocab, `1024` sampled-row, horizon-8, TF32,
  clip-8 shape for `63744` tokens at `2954.763` train tokens/sec, improves
  heldout loss from `7.1370` to `0.2009`, and sustains `524288/524288`
  generated tokens at `7217.290` tokens/sec. Versus the prior clip-8 baseline,
  training throughput improves `16.386%`, forward/loss falls to
  `0.128683 ms/token`, backward falls to `0.196265 ms/token`, and batch total
  falls to `0.338111 ms/token`.
- Current 2026-07-04 selected-expert matmul evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-expert-matmul-524288-63744.json`
  replaces gradient-training selected-expert einsums with batched matmul while
  keeping the same bounded route candidates and active expert selection. The
  report records
  `expert_dispatch_backend=torch_selected_expert_batched_matmul_dispatch`,
  trains the same `524288` model-vocab, `1024` sampled-row, horizon-8, TF32,
  clip-8 shape for `63744` tokens at `3531.685` train tokens/sec, improves
  heldout loss from `7.1551` to `0.2409`, and sustains `524288/524288`
  generated tokens at `7166.620` tokens/sec. Versus the previous
  batched-state-output baseline, training throughput improves `19.525%`,
  forward/loss falls to `0.105695 ms/token`, backward falls to
  `0.165801 ms/token`, and batch total falls to `0.282857 ms/token`.
- Current 2026-07-04 all-awake route-candidate fastpath evidence in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-all-awake-route-fastpath-524288-63744.json`
  keeps the same bounded route-candidate rows and active expert selection, but
  maps candidate positions directly by modulo when no experts are sleeping. The
  report records `candidate_id_source=all_awake_direct_expert_ids` and
  `all_awake_candidate_fastpath=true`, trains the same `524288` model-vocab,
  `1024` sampled-row, horizon-8, TF32, clip-8 shape for `63744` tokens at
  `2994.386` train tokens/sec, improves heldout loss from `7.0738` to
  `0.2031`, and sustains `524288/524288` generated tokens at `7257.759`
  tokens/sec. The same-session control
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-current-control-rerun-524288-63744.json`
  reached `2880.361` train tokens/sec and `7050.359` sustained tokens/sec, so
  the retained fastpath is `+3.959%` training and `+2.942%` sustained generation
  for that paired run.
- Current 2026-07-04 sampled/padded continual-learning evidence in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-524288.json`
  runs `run_language_continual_learning_window` on CUDA with a `524288`
  model vocab, `1024` sampled rows, sparse token-embedding and LM-head row
  gradients, horizon `8`, TF32, and gradient clip interval `8`. It updates
  `65536` new+replay tokens at `2619.310` train tokens/sec using
  `AdamW_dense_core_plus_SparseAdam_vocab_rows`, avoids full vocab logits,
  clips `4/32` optimizer updates, improves new-domain heldout loss from
  `7.1125` to `0.5576`, improves old-domain heldout loss from `7.0595` to
  `1.7194`, improves replay loss from `7.0595` to `1.7182`, and accepts the
  update without rollback. This is online-learning/replay evidence; it is not
  a runtime-promotion or generation-quality claim.
- The follow-up precompute evidence in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288.json`
  keeps the same `524288` model vocab, `1024` sampled rows, horizon `8`, TF32,
  clip interval `8`, `65536` updated tokens, `32` optimizer steps, and `8`
  replay batches, but records sampled-vocab precompute for both online new and
  replay batches. It reaches `3023.964` train tokens/sec versus the retained
  `2619.310` baseline (`+15.449%`), uses
  `precomputed_batch_sampled_vocab_ids` in both the new and replay loss paths,
  avoids full vocab logits, and accepts the update. The source corpus differs
  from the older baseline, so this is same-shape throughput evidence and not a
  language-quality comparison.
- Current 2026-07-04 padded-vocab generation-policy evidence in
  `reports/language_training_experiments/padded-vocab-generation-policy-524288-sustained.json`
  loaded a `524288` row checkpoint with `generation_vocab_size=262`, masked
  `524026` padded rows from generation, kept the generated tail inside tokenizer
  range (`max_tail=261`), and reached `524288/524288` tokens at `7248.118`
  tokens/sec on `cuda:0` with `torch_cuda_graph_burst`, `16` token bursts,
  `32768` graph replays, and zero CUDA graph failures. The report keeps
  `promotes_runtime_claim=false` and `promotes_hot_path=false`; it proves
  checkpoint restore plus decode masking, not broad generation quality.
- Current 2026-07-04 generation coherence evidence in
  `reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json`
  passed `4/4` grounded prompts from the PLIF-surrogate checkpoint with mean
  prefix match `46` characters, mean prefix fraction `0.71875`, printable
  fraction `1.0`, and next-character match rate `1.0`. It records
  `external_llm_used=false`, `human_review_available=false`,
  `promotes_generation_quality_claim=false`, and `promotes_runtime_claim=false`.
  `language-suite-generation-coherence.json` ingests that prompt-suite report,
  the PLIF-surrogate 8192/524288 sustained reports, and all six current LM-head
  kernel reports. It records `generation_coherence=pass`,
  `gpu_kernel_correctness=pass`, `long_run_throughput=pass`,
  `missing_category_count=0`, and `ready_for_review`, while still keeping
  `promotes_runtime_claim=false`.
- Current 2026-07-03 vectorized state-block evidence precomputes the
  token-independent LM state-block projections across `[batch,time]` while
  preserving the causal recurrent membrane/spike/selective-state loop. The
  CUDA profile for `batch=16`, `seq=64`, `state_dim=128`, `16` experts, and
  `4` active experts moved state-block forward from `266.571 ms` to
  `149.390 ms`, full forward loss from `293.187 ms` to `170.986 ms`, and full
  train step from `823.405 ms` to `443.763 ms`. End-to-end
  `reports/language_training_experiments/cuda-vectorized-state-8192.json`
  trained `63744` tokens at `2293.991 train tokens/sec`, improved heldout loss
  from `5.6832` to `0.1788`, and wrote an 8192-token sustained smoke at
  `5638.687 tokens/sec`. The same checkpoint's house-scale report
  `cuda-vectorized-state-524288-sustained.json` reached `524288/524288` at
  `7264.683 tokens/sec`, CUDA graph burst, zero graph failures, and no
  observed contention. `language-suite-vectorized-state.json` keeps promotion
  blocked on generation coherence and the then-remaining PLIF/scan/expert/
  vocab kernel parity evidence; the later PLIF-forward report covers only the
  forward slice, the later PLIF surrogate-backward report closes the float32
  backward blocker, and the later selective-scan report closes standalone scan
  parity without yet fusing the full state-block training loop. The later
  expert-dispatch report closes `float32` selected-expert dispatch parity, and
  the sampled-vocab CE report closes forward vocab-loss parity while leaving
  generation coherence open.
- Rejected regression evidence: same-day unqualified `diagnostic-8192.json`,
  `long-gate-131072.json`, and `house-scale-524288.json` captured a wrapper
  regression where `MarulhoBrain.feed(..., learn=False)` still learned chunks
  and tick readout keys recomputed offline winners per token. Keep those files
  only as regression evidence, not current runtime speed evidence.
- Preserved failure evidence: `reports/runtime_evidence_20260703/long-gate-131072-source-exhausted-before-refill.json`
  shows the old one-shot feed path exhausted the bounded `8192`-token source
  buffer after `8192` tokens. Keep long targets on bounded streaming refill.

## Sustained Runtime Commands

```bash
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/diagnostic-8192.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600 --sample-interval-seconds 0.001
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/long-gate-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 7200 --sample-interval-seconds 0.001
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/house-scale-524288.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 21600 --sample-interval-seconds 0.001
```

LM-head component evidence:

```bash
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint checkpoints/marulho/language.pt --output reports/language_runtime_evidence/diagnostic-8192.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600
```

LM training experiment:

```bash
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/local-run.json --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 8192
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-vectorized-state-8192.json --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 8192 --sustained-timeout-seconds 600 --device cuda
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/cuda-vectorized-state-8192-checkpoint.pt --output reports/language_training_experiments/cuda-vectorized-state-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 3600 --map-location cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-deferred-metrics-8192.json --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 8192 --sustained-timeout-seconds 600 --device cuda
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/cuda-deferred-metrics-8192-checkpoint.pt --output reports/language_training_experiments/cuda-deferred-metrics-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 3600 --map-location cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-stage-profile-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --profile-training-stages --device cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-horizon8-profile-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --profile-training-stages --device cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-profile-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --profile-training-stages --device cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-profile-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --learning-rate 0.002 --max-grad-norm 1.0 --gradient-clip-interval 8 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --profile-training-stages --device cuda
```

LM scale ladder inventory:

```bash
python -m marulho.evaluation.language_scale_ladder --output reports/language_scale_ladder/scale-ladder.json --include-smoke-fixture
```

LM benchmark suite:

```bash
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite.json --sustained-target-tokens 8
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite.json --sustained-target-tokens 8 --sustained-evidence reports/language_runtime_evidence/diagnostic-8192.json --sustained-evidence reports/language_runtime_evidence/long-gate-131072.json
python -m marulho.evaluation.language_triton_kernel_report --output reports/language_kernel_evidence/rmsnorm-triton-20260703.json --shape 1024x64 --shape 2048x128 --shape 1024x256 --dtype float32 --dtype float16 --warmup 20 --repeats 100
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-rmsnorm-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-batched-quality-rmsnorm-policy-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-batched-quality-rmsnorm-policy-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-vectorized-state.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-vectorized-state-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-vectorized-state-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json
python -m marulho.evaluation.language_triton_kernel_report --kernel selective-scan --output reports/language_kernel_evidence/selective-scan-triton-20260704.json --shape 16x128 --shape 32x128 --shape 16x256 --dtype float32 --dtype float16 --scan-time-steps 64 --warmup 20 --repeats 100
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-selective-scan-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json
python -m marulho.evaluation.language_triton_kernel_report --kernel expert-dispatch --output reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --shape 256x64 --shape 512x64 --shape 256x128 --dtype float32 --dtype float16 --expert-count 64 --active-experts 4 --expert-hidden-dim 128 --warmup 20 --repeats 100
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-expert-dispatch-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json
python -m marulho.evaluation.language_triton_kernel_report --kernel sampled-vocab-ce --output reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json --shape 512x128 --shape 1024x128 --shape 512x256 --dtype float32 --dtype float16 --vocab-size 8192 --sampled-vocab-size 1024 --warmup 20 --repeats 100
python -m marulho.evaluation.language_sampled_vocab_training_impact --output reports/language_training_experiments/sampled-vocab-training-impact-default-policy-524288-b16-r8-sampled-only.json --vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 2 --repeats 8 --skip-dense-baseline --device cuda
MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING=1 python -m marulho.evaluation.language_sampled_vocab_training_impact --output reports/language_training_experiments/sampled-vocab-training-impact-forced-triton-autograd-524288-b16-r8-sampled-only.json --vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 2 --repeats 8 --skip-dense-baseline --device cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --device cuda
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/padded-vocab-generation-policy-524288-checkpoint.pt --output reports/language_training_experiments/padded-vocab-generation-policy-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 1200 --map-location cuda --no-environment-snapshot
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-sampled-vocab-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
python -m marulho.evaluation.language_generation_coherence --checkpoint reports/language_training_experiments/cuda-plif-surrogate-8192-checkpoint.pt --output reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json --map-location cuda --min-case-pass-rate 1.0
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-generation-coherence.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --generation-coherence-evidence reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
```
