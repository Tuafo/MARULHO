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
  tokens at `7217.290` tokens/sec. `step` remains the streaming path for
  one-token CUDA graph generation.
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
- `RoutedLanguageExpertLayer` is the first Iteration 4 foundation for the LM
  head. It narrows token-hidden states through a bounded candidate plan, wakes
  only top-k experts, reports total/active columns, candidate rows scored,
  active parameters per token, route device, route latency, candidate-ID source,
  all-awake candidate fastpath use, and explicit all-column fallback truth. The
  all-awake candidate path maps token-hash candidate positions directly by
  modulo instead of materializing an awake-expert `arange`, while sleeping-mask
  runs keep the explicit awake-index select path. Its no-telemetry inference
  path avoids host sleeping-expert materialization so CUDA graph capture can
  replay fixed-shape LM bursts. `language_expert_dispatch_triton.py` now covers
  no-grad CUDA selected-expert dispatch/combine for large enough token batches
  with `float32` parity; gradient training uses
  `torch_selected_expert_batched_matmul_dispatch` for selected expert MLPs,
  while half precision keeps the PyTorch fallback until separate parity and
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
  out of the measured hot update window.
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
  sampled row ID and target-position construction out of the hot loop instead
  of limiting the speedup to fixed experiment batches.
- Continual-learning update metrics now follow the fast experiment runner's
  deferred-readback pattern: detached device scalars aggregate update loss,
  replay loss, and max observed grad norm in the measured loop, then read back
  after a single CUDA stop synchronization.
- `evaluate_language_model` accepts precomputed sampled row IDs and target
  positions from `LanguageBatch`, so continual before/after heldout and replay
  evaluations can reuse the same sampled-vocab contract as the update loop. It
  also aggregates heldout loss as a detached device scalar, reads it back once
  after the evaluation timing stop, records evaluation tokens/sec and sync
  evidence, and restores the caller's original train/eval mode.
- `evaluation/language_continual_learning_experiment.py` is the repeatable
  sampled/padded continual-learning evidence runner for old/new/replay windows.
  It writes JSON plus README reports, applies the CUDA math policy, caps heldout
  eval batch counts when comparing same-shape runs, records throughput deltas,
  and captures MARULHO-owned before/after generation-quality probes without
  turning them into generation-quality claims.
- `language_structural_plasticity.py` is the Iteration 7 transaction path for
  LM expert growth, explicit expert prune, explicit expert merge, and explicit
  expert deep sleep. It builds non-mutating expert-spawn proposals from
  route/learning pressure, expert-prune proposals from explicit inactive or
  low-utility expert evidence, expert-merge proposals from duplicate or
  high-similarity expert-pair evidence, and expert-deep-sleep proposals from
  stale, low-activation, low-utility, high-cost, or dead-spike expert evidence.
  Application requires operator approval, writes a baseline checkpoint snapshot,
  applies the candidate topology change or checkpointed sleep mask under
  heldout non-regression, and records rollback hashes before accepting the
  candidate.
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
