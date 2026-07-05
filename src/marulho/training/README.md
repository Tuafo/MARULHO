# Training

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`training` owns trainer execution, checkpointing, CUDA/native graph lifecycle,
developmental and consolidation runners, query runners, and long-run evidence.

## Owns

- `MarulhoTrainer.train_text_sequence` and ordered SNN token mutation.
- Persistent text-tick executor and CUDA graph route/transition lifecycle.
- Conditional-WHILE q16 native sequence execution, separately reported
  `torch_sequence_graph_*` q16 execution when native parent graph handles are
  unavailable, and exact fallback before mutation.
- Trainer-owned checkpoint state, route caches, strong-event rings, and burst
  evidence.
- The MARULHO-owned next-token language-model foundation: token embedding,
  selective spiking recurrent state, LM head, next-token loss, train/eval split
  reports, heldout loss/perplexity, and language component checkpoints.

## Must Not Own

- HTTP lifecycle or UI contracts.
- Service-owned source selection and operator locks.
- Structural mutation authority outside explicit checkpoint/review gates.

## Runtime Rules

- The persistent text tick executor is not the whole brain loop. Source I/O,
  archival memory, replay review, service orchestration, and UI remain outside
  graph capture.
- The maintained service execution quantum is `16`. The promoted
  conditional-WHILE executor consumes q16 as one ordered native sequence loop
  when PyTorch exposes a raw child CUDA graph handle. If that handle is missing,
  `torch_sequence_graph_*` may consume full q16 chunks as one ordered Torch CUDA
  graph replay, but it must not increment native conditional-WHILE success
  counters. Repeated-child native parent graphs are internal fallback only.
- Boundary-aware text bursts must preserve exact token order and fail closed
  before mutation on drift, telemetry, sleep, slow-memory, metrics, sensory, or
  fallback boundaries.
- Trainer-stage profiling is evaluation evidence only. Do not add profiler
  synchronization to ordinary ticks.
- Slow replay-memory admission and ConceptStore observation are cadenced
  boundaries; they do not justify moving algorithms into service.
- `long_test_runner.py` is now a `MarulhoBrain` health runner. It feeds local
  preset text, starts/stops the brain loop, samples compact brain status, and
  checks feed/readout/tick progress through the active brain runtime.
- `language_model.py` is the Iteration 2 foundation for the active
  `marulho_lm_head` path when `MarulhoBrain` has checkpointed language
  components installed. It is training-owned, checkpoint-backed, and reports
  `external_llm_used=false`, but it is not promoted as live cognition until
  online learning, rollback, throughput, and long-run Runtime Evidence gates
  pass. `build_language_model_splits` supports packed fixed-length language
  windows so experiments can run multi-window optimizer steps on CPU or CUDA
  without changing the model contract.
- The current `MarulhoSelectiveSpikingStateBlock` is the Iteration 3 PyTorch
  foundation: RMSNorm stabilization, input-dependent leak/threshold, trainable
  current terms, selective recurrent state, eligibility trace cache, adaptive
  timestep budget, and spike/dead/over-firing telemetry. CUDA/Triton parity and
  complete-runtime impact evidence are still required before promotion. The
  batched training `forward` path precomputes token-independent RMSNorm,
  select, leak-input, threshold-input, current, input-drive, and residual
  projections across `[batch,time]`, keeps only the causal membrane/spike/
  selective-state recurrence inside the per-token loop, and applies
  `state_output_proj` once across the stacked mixed-state sequence. The local
  2026-07-04 CUDA report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-batched-state-output-524288-63744.json`
  records this as `state_block_projection_mode=batched_token_and_state_output_projection_recurrent_loop`,
  trains `63744` tokens at `2954.763` train tokens/sec on the `524288`
  model-vocab sampled/padded shape, and sustains `524288/524288` generated
  tokens at `7217.290` tokens/sec. The no-grad mixed-state preallocation probe
  in
  `reports/language_training_experiments/state-block-prealloc-runtime-impact-524288-b16-s64.json`
  is rejected as a default: the stacked sequence path reached `12321.430`
  tokens/sec while preallocation reached `12068.368` tokens/sec (`0.979x`) on
  the same full-forward `524288` batch-16/seq-64 shape. The no-grad deferred
  eligibility-trace probe in
  `reports/language_training_experiments/eligibility-trace-runtime-impact-524288-b16-s64.json`
  is also rejected as a default: inline PLIF eligibility reached `12760.575`
  tokens/sec while deferred sequence-scan eligibility reached `12148.414`
  tokens/sec (`0.952x`) with logit parity and real Triton final-scan use.
  `step` remains the streaming path for one-token CUDA graph generation.
- The state block can use `language_plif_triton.py` for no-grad PLIF forward
  updates when CUDA row-count policy allows it. Gradient-enabled `float32`
  training can use the same module's Triton surrogate backward path, which
  preserves the hard-spike forward value and sigmoid surrogate derivative used
  by the PyTorch update. Half-precision backward stays on PyTorch until
  separate gradient parity evidence exists.
- `language_selective_scan_triton.py` now covers the standalone selective
  recurrent state scan primitive with CUDA/Triton parity for
  `[batch,time,state_dim]` recurrence tensors. This closes standalone scan
  kernel evidence, but the training-owned state block still needs a separate
  full-loop integration and complete-runtime impact report before scan fusion
  can be promoted.
- `language_eligibility_trace_triton.py` covers the standalone local
  eligibility-trace final update primitive. State-block telemetry reports
  `eligibility_trace_update_mode`, `eligibility_trace_sequence_buffer_mode`,
  and `eligibility_trace_scan_backend`; the deferred no-grad path remains an
  evidence-only option until a complete-runtime report beats inline PLIF.
- `RMSNorm` now routes CUDA tensors through the language RMSNorm Triton
  primitive only for batched row counts where the kernel is measured useful.
  Streaming one-token LM generation keeps the faster CUDA graph/PyTorch
  fallback and exposes the fallback count in sustained reports.
- `language_continual_learning.py` is the first Iteration 6 foundation for the
  LM head. It applies bounded online updates, mixes replay batches, measures
  old/new heldout loss and replay retention, records spike-rate and throughput
  deltas, and keeps rollback snapshot hashes before accepting the update as
  review evidence. It now supports sampled/padded vocab models with sparse
  token-embedding and LM-head row gradients, the same dense-core plus sparse
  vocab-row optimizer policy used by fast LM experiments, sparse-aware
  gradient-clip cadence, telemetry-light training updates, sampled-vocab
  batch precompute for online new, replay, and heldout eval batches, deferred
  GPU-scalar metric aggregation for update/replay loss plus max-gradient-norm
  evidence, and full-window phase timings. The local
  2026-07-04 CUDA report
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-524288.json`
  updated `65536` new+replay tokens on the `524288` model-vocab, `1024`
  sampled-row, horizon-8, TF32, clip-interval-8 shape at `2619.310` train
  tokens/sec, improved new-domain heldout loss from `7.1125` to `0.5576`,
  improved old-domain loss from `7.0595` to `1.7194`, improved replay loss
  from `7.0595` to `1.7182`, and accepted the update without rollback.
  The follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288.json`
  records `sampled_vocab_precompute` for both new and replay batches, uses
  `precomputed_batch_sampled_vocab_ids` in both loss paths, updates the same
  `65536` token shape at `3023.964` train tokens/sec, and is `+15.449%`
  throughput versus the retained baseline. The corpus text differs from the
  older report, so this is same-shape online-throughput evidence rather than a
  language-quality comparison.
  The deferred-metric follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-metrics-precomputed-sampled-vocab-524288.json`
  records `metric_readback_mode=deferred_gpu_scalar_aggregation`,
  `per_step_metric_cpu_sync=false`, explicit CUDA timing-window syncs, and
  `3089.664` train tokens/sec on the same current corpus and shape. That is
  `+2.173%` over the precompute-only report and `+17.957%` versus the retained
  baseline.
  The eval-precompute follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-eval-precompute-deferred-metrics-524288.json`
  records precomputed sampled vocab for old/new heldout eval batches and
  `precomputed_batch_sampled_vocab_ids` in heldout loss evidence. It reaches
  `3057.041` update tokens/sec and `1690.405` total-window tokens/sec, with
  phase timings showing `21.438s` in update, `5.829s` in pre-update eval,
  `6.245s` in post-update eval, `2.283s` in optimizer setup, and `0.355s` in
  sampled-vocab precompute. This is full-window visibility and eval-precompute
  evidence; update throughput is in the same band as the previous deferred
  report, not a new update-loop promotion.
  The deferred-eval-metric follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-eval-metrics-524288.json`
  uses the repeatable `language_continual_learning_experiment.py` runner,
  keeps the same `22` old and `27` new heldout eval batch counts as the
  eval-precompute report, and records heldout
  `metric_readback_mode=deferred_gpu_scalar_aggregation`,
  `per_batch_metric_cpu_sync=false`, and CUDA start/stop syncs. It reaches
  `2990.395` update tokens/sec and `1712.349` total-window tokens/sec, with
  phase timings of `0.434s` sampled-vocab precompute, `5.682s` pre-update eval,
  `2.172s` optimizer setup, `21.915s` update, and `5.311s` post-update eval.
  That is `+1.298%` total-window throughput versus eval-precompute with the
  same eval batch counts, while update throughput is `-2.180%`; read it as
  heldout-eval sync reduction plus repeatable evidence plumbing, not a new
  update-loop promotion.
  The generation-quality follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-quality-524288.json`
  keeps the same `524288`/`1024` sampled continual shape and adds MARULHO-owned
  old/new prompt continuations before and after online learning. It records
  `records_generation_quality_probe=true`, `records_generation_quality_delta=true`,
  and `promotes_generation_quality_claim=false`; next-character match improved
  from `0.0` to `1.0`, mean source-prefix match improved only from `0.0` to
  `1.0` character, printable fraction improved from `0.956` to `1.0`, and
  distinct-bigram fraction fell from `0.362` to `0.191`. The raw old-domain
  continuation improved from broken bytes to repetitive `replay` text, so this
  proves generation is being measured around continual learning, not that broad
  language quality is solved.
  The decode-control follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-decode-controls-524288.json`
  keeps the same continual shape and enables transparent greedy decode controls
  with `repetition_penalty=1.15` and `no_repeat_ngram_size=2`. It records
  `decode_controls_backend=torch_device_tensor`,
  `decode_controls_cpu_token_copy=false`, `1249` repetition-penalty token
  adjustments, `59` no-repeat banned-token events, zero decode-control
  fallbacks, `3687.327` update tokens/sec, and `2051.118` total-window
  tokens/sec. Compared with the prior generation-quality report, update
  throughput is `+20.267%`, total-window throughput is `+21.491%`, and
  after-learning distinct-bigram fraction improves from `0.191` to `1.000`;
  source-prefix match remains only `1.0` character, so this is
  repetition-control evidence rather than a language-quality promotion.
  The memory-slot online-learning follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-524288.json`
  keeps the same continual-learning shape with `1024` memory slots, `8`
  bounded candidates, and `2` active slots per token. It updates `65536`
  new+replay tokens at `2951.259` update tokens/sec and `1131.696`
  total-window tokens/sec, accepts the checkpoint update, improves new-domain
  heldout loss by `5.8397`, improves old-domain loss by `5.2612` rather than
  forgetting, and improves replay loss by `5.2574`. The report records
  `marulho_language_continual_memory_slots.v1`, `524288` memory candidates
  scored across the online update, `runs_all_slots=false`,
  `bounded_memory_slot_path=true`, `memory_gate_readback=false`, and
  `records_memory_slot_online_update_path=true`. Compared with the no-memory
  decode-control report, it is deliberately marked as a different memory-slot
  shape and pays `-19.962%` update throughput and `-44.825%` total-window
  throughput; compared with the older retained baseline it is `+12.673%`
  update throughput. This is bounded online memory/replay/forgetting evidence,
  not a runtime or language-quality promotion.
  The memory-candidate precompute follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-precomputed-candidates-524288.json`
  reuses the same shape but precomputes GPU-resident memory candidate IDs for
  online new/replay batches and old/new heldout eval batches. It records
  `candidate_id_source=precomputed_batch_memory_candidate_ids`,
  `precomputed_candidate_ids_used=true`,
  `records_memory_slot_candidate_precompute=true`, `runs_all_slots=false`, and
  the same `524288` scored memory candidates. The run reaches `3074.256`
  update tokens/sec and `1147.115` total-window tokens/sec, accepts the update,
  improves new-domain heldout loss by `5.8398`, improves old-domain loss by
  `5.2632`, and improves replay loss by `5.2592`. That is `+4.168%` update
  throughput and `+1.363%` total-window throughput versus the prior memory-slot
  report, but it still pays `-16.626%` update throughput and `-44.074%`
  total-window throughput versus the no-memory decode-control report.
  The eval-matched comparison report
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-precomputed-candidates-evalmatched-524288.json`
  uses `match_comparison_eval_batches` to trim heldout eval to the no-memory
  decode-control report's `22` old and `27` new eval batches. It records
  `records_comparison_eval_batch_match=true`, accepts the update, improves
  new-domain heldout loss by `5.8439`, improves old-domain loss by `5.2648`,
  improves replay loss by `5.2623`, reaches `2969.385` update tokens/sec and
  `1723.213` total-window tokens/sec, and keeps bounded precomputed memory
  candidates with `runs_all_slots=false`. With eval counts matched, memory
  slots pay `-19.471%` update throughput and `-15.987%` total-window
  throughput versus the no-memory decode-control report; the memory shape is
  still intentionally different, so this is a fair eval-count comparison rather
  than a same-architecture speed promotion.
  The no-grad memory-slot Triton follow-up
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-triton-nograd-evalmatched-524288.json`
  keeps the same eval-matched online shape and routes heldout old/new/replay
  evaluation memory retrieval through `triton_no_grad_bounded_memory_slots`.
  Gradient-enabled online updates in that historical artifact remain on
  `torch_autograd_bounded_memory_slots`, so memory-slot gate and slot gradients
  stay intact. The run accepts the update, improves new-domain heldout loss by
  `5.8408`, improves old-domain loss by `5.2584`, improves replay loss by
  `5.2560`, reaches `2945.988` update tokens/sec and `1756.779`
  total-window tokens/sec, and records Triton no-grad eval memory retrieval in
  all six old/new/replay before/after eval sections. Against the no-memory
  decode-control report, the matched total-window tax improves to `-14.350%`
  while the update-loop tax remains `-20.105%`; treat this as heldout-eval
  backend acceleration.
  The matching kernel report
  `reports/language_kernel_evidence/memory-slots-triton-20260705.json` passes
  three CUDA `float32` shape sweeps for `language_memory_slot_retrieval` with
  geometric microbenchmark speedup `4.950x`; `float16` remains explicitly
  unsupported.
  The newer `524288` update-token architecture-cost pair
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-no-memory-evalmatched-update524288-rerun.json`
  and
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-default-evalmatched-update524288-rerun.json`
  reruns the full continual window with matched eval counts. The no-memory arm
  reaches `3765.911` update tokens/sec and `3451.048` total-window tokens/sec;
  the default bounded memory-slot arm reaches `3753.246` and `3436.735` while
  scoring `4194304` precomputed memory candidates, avoiding all-slot scans,
  staying on `torch_autograd_bounded_memory_slots`, and accepting the update.
  The explicit
  `marulho_language_continual_memory_slot_architecture_cost.v1` section records
  a small current cost (`-0.336%` update, `-0.415%` total-window), a nearly
  unchanged new-domain delta, slightly better old-domain/replay retention, and
  the same generation-probe prefix score. This is the current full-window cost
  baseline for memory slots, not a broad language-quality promotion.
- `RoutedLanguageExpertLayer` is the first Iteration 4 foundation for the LM
  head. It narrows token-hidden states through a bounded candidate plan, wakes
  only top-k experts, reports total/active columns, candidate rows scored,
  active parameters per token, route device, route latency, candidate-ID source,
  route-selection backend, all-awake candidate fastpath use, and explicit
  all-column fallback truth. The
  all-awake candidate path maps token-hash candidate positions directly by
  modulo instead of materializing an awake-expert `arange`, while sleeping-mask
  runs keep the explicit awake-index select path. Its no-telemetry inference
  path avoids host sleeping-expert materialization so CUDA graph capture can
  replay fixed-shape LM bursts. `language_route_topk_triton.py` now covers
  no-grad CUDA route/vote top-k selection for large enough token batches with
  `float32` parity, and
  `reports/language_training_experiments/route-topk-runtime-impact-524288-b16-s64.json`
  records a full no-grad LM forward comparison at the current `524288`
  model-vocab batch-16/seq-64 shape: `12972.201` tokens/sec with Triton
  route-top-k versus `12418.282` tokens/sec with PyTorch route-top-k fallback,
  `1.045x` throughput ratio, exact route-kernel use/fallback counters, and
  output parity within absolute tolerance. `language_expert_dispatch_triton.py`
  covers no-grad CUDA selected-expert dispatch/combine for large enough token
  batches with `float32` parity, and
  `reports/language_training_experiments/expert-dispatch-runtime-impact-524288-b16-s64.json`
  records a full no-grad LM forward comparison at the same `524288`
  model-vocab batch-16/seq-64 shape with route-top-k held constant:
  `12371.776` tokens/sec with Triton dispatch versus `11699.767` tokens/sec
  with PyTorch dispatch fallback, `1.057x` throughput ratio, exact
  dispatch-kernel use/fallback counters, and output parity within absolute
  tolerance; gradient training uses
  `torch_selected_expert_batched_matmul_dispatch` for selected expert MLPs and
  keeps route scoring/top-k on PyTorch so route keys retain gradients, while
  half precision keeps the PyTorch fallback until separate parity and
  complete-runtime impact evidence exists. The local 2026-07-04 CUDA report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-expert-matmul-524288-63744.json`
  is the historical fast run for this `524288` model-vocab sampled/padded shape
  at `3531.685` train
  tokens/sec and sustains `524288/524288` generated tokens at `7166.620`
  tokens/sec. A paired same-session rerun keeps the direct all-awake route
  candidate fastpath: `cuda-sampled-padded-horizon8-tf32-clip8-all-awake-route-fastpath-524288-63744.json`
  reached `2994.386` train tokens/sec and `7257.759` sustained tokens/sec,
  versus `2880.361` and `7050.359` for
  `cuda-sampled-padded-horizon8-tf32-clip8-current-control-rerun-524288-63744.json`.
- `language_sampled_vocab_ce_triton.py` now covers forward sampled-vocabulary
  cross-entropy parity for CUDA `float32` hidden rows and selected vocabulary
  IDs that include every target token. It also has a forceable Triton-forward/
  custom-autograd training probe, but the maintained training path keeps that
  probe off by default because complete b16/r8 CUDA evidence was slower than
  selected-row PyTorch autograd (`2622.292` versus `2675.442` train tokens/sec).
  `MarulhoLanguageModel.next_token_loss` uses sampled/adaptive vocabulary
  training without materializing full logits when `sampled_vocab_size` is
  configured, uses selected-row PyTorch autograd by default, and can opt into
  sparse token-embedding and LM-head weight gradients for row-sparse optimizers.
  It also accepts precomputed sampled row IDs and target positions for fixed
  training batches so experiment runners can keep sampled-vocab construction
  out of the measured hot update window. Memory-slot batches can also carry
  precomputed bounded memory candidate IDs, keeping token-hash candidate-plan
  construction out of measured online/eval loss calls without changing the
  active slot count or permitting all-slot scans.
- `language_sampled_vocab_training_impact.py` is the complete training-step
  impact report for sampled/adaptive vocabulary loss. The local 2026-07-04
  CUDA report
  `reports/language_training_experiments/sampled-vocab-training-impact-default-policy-524288-b16-r8-sampled-only.json`
  used a `524288` row model vocabulary with `1024` sampled rows, `batch=16`,
  `seq=64`, warmup `2`, repeats `8`, backward, sparse-aware gradient clipping,
  and optimizer steps. The sampled arm used
  `AdamW_dense_core_plus_SparseAdam_vocab_rows`, avoided full vocab logits,
  stayed on selected-row PyTorch autograd, reached `2675.442` train tokens/sec,
  peaked at `2368.205 MiB`, and recorded zero Triton CE training calls. This is
  large-vocab training impact evidence, not generation-quality or
  runtime-promotion evidence.
- Padded-vocab checkpoints now require an explicit generation decode policy.
  `generation_vocab_size` limits generation and sustained runs to tokenizer
  rows while keeping the larger model vocabulary available for sampled training.
  The local 2026-07-04 checkpoint-loaded report
  `reports/language_training_experiments/padded-vocab-generation-policy-524288-sustained.json`
  used `524288` model vocab rows, `262` tokenizer/generation rows, masked
  `524026` padded rows from generation, kept generated tail IDs inside the
  tokenizer range, and reached `524288/524288` tokens at `7248.118`
  tokens/sec on `torch_cuda_graph_burst`. This is decode/checkpoint/long-run
  policy evidence, not broad generation quality.
- `language_training_experiment.py` can now train, checkpoint, generate, and
  sustain padded-vocab sampled models directly. The local 2026-07-04 integrated
  report
  `reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744.json`
  trained `63744` tokens with `524288` model vocab rows, `1024` sampled rows,
  `batch=16`, sparse vocab-row optimization, selected-row PyTorch autograd, and
  no full vocab logits at `2361.928` train tokens/sec. It improved heldout loss
  from `7.1612` to `0.1997`, saved a checkpoint with generation limited to
  `262` tokenizer rows, and sustained `524288/524288` tokens at `7273.947`
  tokens/sec on CUDA graph burst. This is the normal experiment-runner path for
  large-vocab science loops; broad language quality and runtime promotion remain
  separate gates.
- The experiment runner can opt into `--profile-training-stages` for CUDA-event
  hot-window timings without per-stage synchronization. The local 2026-07-04
  profile
  `reports/language_training_experiments/cuda-sampled-padded-stage-profile-524288-63744.json`
  trained the same `524288` model-vocab, `1024` sampled-row, `batch=16` shape
  for `63744` tokens at `2342.586` train tokens/sec and sustained
  `524288/524288` generated tokens at `7244.434` tokens/sec. Backward
  dominates at `0.245897 ms/token`, followed by forward/loss at
  `0.150728 ms/token`; optimizer step is only `0.007629 ms/token`. Treat this
  as evidence to aim the next speed slice at state-block/PLIF backward and
  forward/loss before optimizer fusion.
- `LanguageModelConfig.recurrent_gradient_horizon` is an opt-in bounded BPTT
  policy for fast LM training experiments. When enabled, multi-token gradient
  training preserves causal forward recurrence on device but detaches membrane,
  spike, selective-state, and eligibility tensors at fixed token boundaries;
  one-token streaming generation is unchanged. The local 2026-07-04 horizon-8
  report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-profile-524288-63744.json`
  recorded `7` detach boundaries per 64-token batch, trained the `524288`
  model-vocab, `1024` sampled-row, batch-16 shape at `2435.816` train
  tokens/sec, improved heldout loss from `7.1213` to `0.2303`, and sustained
  `524288/524288` generated tokens at `7225.847` tokens/sec. This is a
  throughput/credit-horizon tradeoff, not full long-context BPTT or a
  generation-quality promotion.
- `language_training_experiment.py` applies an explicit experiment-scoped CUDA
  math policy. On CUDA it enables TF32 matmul by default and sets
  `float32_matmul_precision=high`, records the before/active policy in the
  experiment and training reports, and restores the previous process policy
  afterward. The local 2026-07-04 horizon-8 TF32 report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-profile-524288-63744.json`
  trained the `524288` model-vocab shape at `2487.980` train tokens/sec and
  sustained `524288/524288` generated tokens at `7239.247` tokens/sec. This is
  `+2.142%` training throughput over the horizon-8 non-TF32 run and `+6.207%`
  over the full-sequence profiled baseline; it is a precision/speed tradeoff
  and not a generation-quality claim.
- `language_training_experiment.py` also supports `gradient_clip_interval` for
  fast LM experiments. The default clips every optimizer update; interval `8`
  runs the sparse-aware GPU norm/clip pass every eighth update and records both
  applied and skipped counts. The local 2026-07-04 report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-profile-524288-63744.json`
  clipped `8/64` updates, trained the `524288` model-vocab shape at
  `2538.756` train tokens/sec, improved heldout loss from `7.0764` to
  `0.2083`, and sustained `524288/524288` generated tokens at `7223.490`
  tokens/sec. This is a fast-experiment stability/speed tradeoff and must not
  hide skipped gradient-norm passes.
- The current batched state-output projection path keeps the same horizon-8,
  TF32, clip-8, sampled/padded shape but moves `state_output_proj` out of the
  per-token recurrent loop. The local 2026-07-04 report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-batched-state-output-524288-63744.json`
  trains `63744` tokens at `2954.763` train tokens/sec, records
  `state_output_projection_batched=true`, improves heldout loss from `7.1370`
  to `0.2009`, and sustains `524288/524288` generated tokens at `7217.290`
  tokens/sec. This is the previous large-vocab fast-experiment baseline before
  selected-expert matmul dispatch.
- The selected-expert training dispatch keeps the same large-vocab
  baseline shape but replaces selected-expert einsums with batched matmul. The
  local 2026-07-04 report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-expert-matmul-524288-63744.json`
  trains `63744` tokens at `3531.685` train tokens/sec, records
  `expert_dispatch_backend=torch_selected_expert_batched_matmul_dispatch`,
  improves heldout loss from `7.1551` to `0.2409`, and sustains
  `524288/524288` generated tokens at `7166.620` tokens/sec. This remains the
  historical absolute fast run for the large-vocab shape.
- The retained all-awake route-candidate fastpath avoids the awake-index tensor
  when no experts are sleeping and records
  `candidate_id_source=all_awake_direct_expert_ids`. The local paired report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-all-awake-route-fastpath-524288-63744.json`
  trains the same shape at `2994.386` train tokens/sec and sustains
  `524288/524288` generated tokens at `7257.759` tokens/sec, compared with
  `2880.361` train tokens/sec and `7050.359` sustained tokens/sec in
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-current-control-rerun-524288-63744.json`.
  The paired win is `+3.959%` training and `+2.942%` sustained generation, with
  batch total down from `0.346854` to `0.333647 ms/token`.
- The sampled-vocab batch-precompute training path keeps the same all-awake
  routed shape but precomputes sampled row IDs and target positions outside the
  measured update window. The local 2026-07-04 report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288-63744.json`
  trains `63744` tokens at `3041.246` train tokens/sec, records
  `sampled_vocab_precompute.enabled=true`, improves forward/loss from
  `0.125210` to `0.121173 ms/token`, and lowers batch total from `0.333647` to
  `0.328424 ms/token` versus the retained all-awake fastpath report. The paired
  sustained run reached `7203.369` tokens/sec, and a same-checkpoint current-code
  sustained rerun of the retained all-awake checkpoint reached `7206.201`
  tokens/sec, so this is a training hot-window speed slice rather than an
  inference promotion.
- The same precompute helper is training-owned and reused by
  `language_continual_learning.py` so online new/replay update windows can keep
  sampled row ID, target-position, and bounded memory-candidate construction
  out of the hot loop instead of limiting the speedup to fixed experiment
  batches.
- Continual-learning update metrics now follow the fast experiment runner's
  deferred-readback pattern: detached device scalars aggregate update loss,
  replay loss, and max observed grad norm in the measured loop, then read back
  after a single CUDA stop synchronization.
- `evaluate_language_model` accepts precomputed sampled row IDs and target
  positions plus bounded memory candidate IDs from `LanguageBatch`, so
  continual before/after heldout and replay evaluations can reuse the same
  sampled-vocab and memory-candidate contracts as the update loop. It also
  aggregates heldout loss as a detached device scalar, reads it back once after
  the evaluation timing stop, records evaluation tokens/sec and sync evidence,
  and restores the caller's original train/eval mode.
- `evaluation/language_continual_learning_experiment.py` is the repeatable
  sampled/padded continual-learning evidence runner for old/new/replay windows.
  It writes JSON plus README reports, applies the CUDA math policy, caps heldout
  eval batch counts when comparing same-shape runs, records throughput deltas,
  can optionally match old/new eval batch counts to a comparison report for
  fair total-window comparisons, and captures MARULHO-owned before/after
  generation-quality probes. Optional repetition-penalty and no-repeat-ngram
  decode controls are recorded as decode policy and counters, not hidden
  generation authority or quality promotion.
- `language_structural_plasticity.py` is the Iteration 7 transaction path for
  LM expert growth, column split, synapse-bundle hidden-capacity growth,
  memory-slot expansion, explicit expert prune, explicit expert retire,
  explicit expert merge, explicit expert deep sleep, and bounded route-bank
  expansion. It builds non-mutating expert-spawn proposals from route/learning
  pressure, column-split proposals from overload or high-surprise pressure,
  synapse-bundle proposals from high-surprise/replay-conflict/uncertainty
  pressure, memory-slot proposals from novel-concept/replay-conflict/surprise
  pressure, expert-prune proposals from explicit inactive or low-utility expert
  evidence, expert-retire proposals from terminal stale/dead/harmful evidence,
  expert-merge proposals from duplicate or high-similarity expert-pair
  evidence, expert-deep-sleep proposals from stale, low-activation, low-utility,
  high-cost, or dead-spike expert evidence, and route-bank proposals from
  bounded candidate saturation.
  Application requires operator approval, writes a baseline checkpoint snapshot,
  applies the candidate topology/config change or checkpointed sleep mask under
  heldout non-regression, and records rollback hashes before accepting the
  candidate. Route-bank expansion is capped so all-awake runs do not silently
  become dense/all-column route scans; memory-slot expansion uses bounded
  token-hash candidate retrieval so added slots do not silently become an
  all-slot retrieval path; synapse-bundle growth preserves old expert weights
  and initializes the added hidden rows/columns neutrally.
- `evaluation/language_structural_plasticity_experiment.py` writes portable
  JSON plus README evidence for one or more checkpoint-backed structural
  transactions instead of leaving growth/prune proof embedded only in a suite
  fixture. The current CUDA report
  `reports/language_structural_plasticity/structural-memory-route-524288.json`
  uses a `524288` model vocab, `1024` sampled eval rows, and `cuda` device
  placement to apply two reviewed transactions: memory-slot expansion from `0`
  to `1024` slots with `8` bounded candidates and `2` active slots, and
  route-bank expansion from `8` to `12` bounded candidates on `16` experts.
  Both reports record non-mutating proposals, operator approval, baseline
  checkpoints, checkpoint restore verification, rollback verification, and
  heldout non-regression. This is structural transaction evidence, not runtime
  or generation-quality promotion.
- `evaluation/language_memory_slot_runtime_impact.py` is the complete-forward
  evidence report for the LM memory-slot path. The local 2026-07-05 CUDA
  report
  `reports/language_training_experiments/memory-slot-runtime-impact-524288-b16-s64.json`
  uses the `524288` model-vocab, batch-16/seq-64 shape with `1024` memory
  slots, `8` bounded candidates, and `2` active slots. It measured
  `12857.975` tokens/sec with memory disabled, `10784.141` tokens/sec with
  bounded trainable-neutral memory slots (`0.839x`), and `11652.834`
  tokens/sec for the all-slot contrast (`1.081x` versus bounded on this
  shape). Bounded retrieval scores `8192` memory candidates per forward, avoids
  all-slot scan, preserves exact neutral logit parity while the memory gate is
  zero, proves `131072` nonzero slot values with zero gate, keeps
  `memory_gate_readback=false`, and peaks at `412.950 MiB`; the all-slot
  contrast scores `1048576` candidates, reports
  `memory_slot_candidate_plan_unbounded`, and peaks at `1440.856 MiB`. This is
  memory-capacity runtime-impact evidence, not a hot-path promotion or
  generation-quality claim; training and sustained-generation impact are
  separate gates.
  The no-grad Triton retrieval follow-up
  `reports/language_training_experiments/memory-slot-runtime-impact-triton-nograd-524288-b16-s64.json`
  repeats that complete-forward shape after adding
  `language_memory_slots_triton.py`. Bounded retrieval records
  `memory_slot_retrieval_backend=triton_no_grad_bounded_memory_slots`,
  `bounded_memory_slot_triton_kernel_used=true`, still scores only `8192`
  candidates per forward, and reaches `12087.778` tokens/sec versus
  `12468.722` disabled-memory control (`0.969x`). The all-slot contrast stays
  on the torch fallback because `1024` candidates exceed the Triton candidate
  policy, scores `1048576` candidates, reaches `10880.043` tokens/sec, and
  peaks at `1440.856 MiB` versus `411.333 MiB` for bounded retrieval. This is
  bounded no-grad retrieval acceleration evidence, not autograd training
  promotion.
  The matching kernel report
  `reports/language_kernel_evidence/memory-slots-triton-20260705.json` records
  parity for three CUDA `float32` shapes and `4.950x` geometric microbenchmark
  speedup over the PyTorch selected-slot reference.
- `evaluation/language_memory_slot_training_impact.py` is the complete
  optimizer-step evidence report for the LM memory-slot path. The local
  2026-07-05 CUDA report
  `reports/language_training_experiments/memory-slot-training-impact-model524288-b16-s64-t524288.json`
  measures `524288` optimizer tokens per arm on the `524288` model-vocab,
  `1024` sampled-row, batch-16/seq-64, horizon-8, TF32, clip-interval-8 shape.
  The disabled-memory control reached `3148.695` train tokens/sec; bounded
  memory slots reached `3056.539` train tokens/sec (`0.971x`) while scoring
  `8192` memory candidates per optimizer step, avoiding all-slot scan, keeping
  `memory_gate_readback=false`, avoiding full vocab logits, using
  `AdamW_dense_core_plus_SparseAdam_vocab_rows`, clipping gradients on
  `64/512` measured steps, and peaking at `2378.977 MiB` versus `2368.389 MiB`
  control (`1.004x`). The memory gate moved from `0.0` to `-0.04059`, the gate
  gradient was nonzero, and memory-slot gradients became nonzero after the gate
  update. This is trainability and long training-window impact evidence, not a
  hot-path promotion; sustained generation and longer online-learning windows
  still need separate measurement before memory-slot growth can promote.
  The 2026-07-05 follow-up
  `reports/language_training_experiments/memory-slot-training-impact-triton-autograd-compare-524288-b16-s64-t524288-long.json`
  keeps the same `524288` optimizer-token shape, feeds precomputed batch memory
  candidate IDs into the measured update window, and compares forced-off torch
  autograd against Triton-forward/custom-autograd memory retrieval. Disabled
  memory reached `3171.113` train tokens/sec, bounded torch autograd reached
  `3076.582` (`0.970x` control), and bounded Triton-forward/custom-autograd
  reached `3110.440` (`0.981x` control, `1.011x` versus bounded torch). The
  Triton training arm records `512` Triton autograd forwards, `512` custom
  backward calls, zero fallback memory-slot calls, precomputed memory
  candidates, and nonzero gate/slot gradients. The later full-window
  continual-learning comparison rejects that isolated win as the maintained
  default: forced-off/default torch autograd reached `3134.337` update
  tokens/sec and `2849.240` total-window tokens/sec for `524288` update tokens,
  while opt-in Triton training reached `3074.512` and `2823.885` on the same
  shape (`-1.909%` update throughput, `-0.890%` total-window throughput) with
  `512` Triton autograd forwards, `512` custom backward calls, zero fallback,
  and the same bounded precomputed candidates. Keep training Triton opt-in
  through `MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING=1` until complete
  continual-window evidence wins.
- `MarulhoLanguageModel.forward_step` applies the same bounded memory-slot
  retrieval used by batched training before routed experts, so checkpointed
  streaming generation can execute memory slots rather than bypassing them. The
  local integrated CUDA report
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-524288.json`
  trains the `524288` model-vocab, `1024` sampled-row, horizon-8, TF32,
  clip-interval-8 memory-slot shape for `31936` tokens at `3112.320` train
  tokens/sec, improves heldout loss by `5.3550`, saves a checkpoint, and
  sustains `524288/524288` generated tokens at `5780.913` tokens/sec on
  `torch_cuda_graph_burst`. The paired sustained report records `32768` graph
  replays, zero graph failures, `generation_vocab_size=262`, no full model-vocab
  logits during generation, `1024` memory slots, `8` bounded candidates,
  `2` active slots, `64` scored memory candidates per streaming step,
  `runs_all_slots=false`, `memory_gate_readback=false`, and
  `records_bounded_memory_slot_path=true`. This is checkpointed memory-slot
  sustained evidence, not a runtime-promotion or language-quality claim.
- `language_checkpoint_evolution.py` is the first Iteration 8 evaluation path
  for controlled LM checkpoint evolution. It writes a parent checkpoint, forks
  an isolated child checkpoint, runs child-only learning/replay/optional growth,
  compares parent and child heldout evidence, verifies rollback to the parent
  hash, and emits lineage metadata for later operator promotion review.
- LM-head sustained report writing lives in
  `marulho.evaluation.language_sustained_runtime_evidence`, not in status or
  service code. It can stream this package's checkpointed LM state for
  final/partial evidence. CUDA runs may use `torch_cuda_graph_burst` replay over
  ordered `quantum_tokens` steps, while stop/EOS-sensitive runs and graph
  failures fall back to eager streaming with the fallback reason recorded. The
  path stays unpromoted until generation quality, CUDA/Triton parity, and
  complete-runtime impact gates exist.
