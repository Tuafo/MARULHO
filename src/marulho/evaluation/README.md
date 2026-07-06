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
  loss, sparse vocab-row optimization, tokenizer-row generation decode limits,
  sampled-vocab batch precompute, bounded memory-candidate precompute, and
  bounded route-candidate precompute. The measured update loop receives
  `precomputed_batch_memory_candidate_ids` and
  `precomputed_batch_route_candidate_ids` for fixed batches so candidate-plan
  construction does not run inside the hot loss call. It also records
  `hot_update_evidence_mode=post_window_telemetry_probe` and
  `per_step_evidence_dict_build=false`: loss/routing/memory evidence is
  gathered after the measured window rather than rebuilt every optimizer step.
  It is meant to accelerate model experiments, not create a new promotion gate.
- `language_state_block_runtime_impact.py` measures complete no-grad LM forward
  impact for state-block sequence-buffer experiments. The current
  `524288` model-vocab batch-16/seq-64 report rejects no-grad mixed-state
  preallocation as a default because the existing stacked path is faster.
- `language_memory_slot_runtime_impact.py` measures complete no-grad LM forward
  impact for bounded memory-slot retrieval. The current no-grad Triton follow-up
  `reports/language_training_experiments/memory-slot-runtime-impact-triton-nograd-524288-b16-s64.json`
  records `triton_no_grad_bounded_memory_slots` for the bounded arm, keeps the
  bounded path at `8192` scored candidates per forward, improves bounded
  forward throughput to `12087.778` tokens/sec (`0.969x` of disabled-memory
  control), and keeps the all-slot contrast slower with `1048576` candidates
  and much higher memory. This is complete-forward evidence only; gradient
  training and online continual update impact remain separate gates.
- `language_triton_kernel_report.py --kernel memory-slots` writes bounded
  memory-slot retrieval parity evidence. The current
  `reports/language_kernel_evidence/memory-slots-triton-20260705.json` report
  passes three CUDA `float32` shapes for `language_memory_slot_retrieval` with
  `4.950x` geometric microbenchmark speedup and marks `float16` unsupported.
- `language_memory_slot_training_impact.py` compares forced-off torch
  memory-slot autograd with supported CUDA Triton-forward/custom-autograd on the
  same complete optimizer-step window. The current
  `reports/language_training_experiments/memory-slot-training-impact-triton-autograd-compare-524288-b16-s64-t524288-long.json`
  report measures `524288` optimizer tokens per arm, records `512` Triton
  autograd forwards plus `512` custom backward calls, and improves bounded
  memory-slot training from `3076.582` to `3110.440` train tokens/sec versus
  forced-off torch autograd. It is training-backend evidence, but the matching
  `524288` full continual-learning update window rejected Triton training as
  the maintained default: torch autograd reached `3134.337` update tokens/sec
  and `2849.240` total-window tokens/sec, while opt-in Triton training reached
  `3074.512` and `2823.885` on the same shape with `512` Triton autograd
  forwards, `512` custom backward calls, and zero fallback. Keep training Triton
  opt-in until a complete continual-window report wins.
  The current-code 2026-07-06 rerun
  `reports/language_training_experiments/memory-slot-training-impact-current-hot-evidence-524288-b16-s64-t524288-20260706.json`
  keeps the same `524288` optimizer-token shape, writes partial JSON after each
  completed arm, and records `report_status=final` after all three arms. Its
  measured update steps use `post_window_telemetry_probe` with
  `per_step_evidence_dict_build=false` and
  `per_step_memory_slot_stats_delta=false`. It measured disabled memory at
  `2531.717` train tokens/sec, bounded torch memory at `2523.593` (`0.9968x`
  control), and opt-in Triton-forward/custom-autograd at `2533.075` (`1.0038x`
  bounded, `1.0005x` control), while preserving nonzero memory-gate/slot
  gradients and zero memory-slot fallback calls.
- `language_continual_learning_experiment.py` now writes
  `marulho_language_continual_memory_slot_architecture_cost.v1` when a
  memory-slot run compares against a no-memory baseline with the same model
  vocab, sampled vocab, update-token count, and matched heldout eval counts. The
  retained `524288` update-token pair is
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-no-memory-evalmatched-update524288-rerun.json`
  versus
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-default-evalmatched-update524288-rerun.json`:
  no-memory reached `3765.911` update tokens/sec and `3451.048` total-window
  tokens/sec; bounded memory slots reached `3753.246` and `3436.735`, scored
  `4194304` precomputed candidates without all-slot scans, stayed on the
  default torch-autograd training backend, and accepted the update. The measured
  memory-slot architecture cost is `-0.336%` update throughput and `-0.415%`
  total-window throughput, while old-domain and replay losses improve slightly
  more than the no-memory baseline. The current min1-policy
  training-accounting matched pair,
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-no-memory-triton-min1-training-accounting-evalmatched-update524288-20260705.json`
  versus
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-triton-min1-training-accounting-evalmatched-update524288-20260705.json`,
  accepts both arms with rollback verification at `524288` update tokens:
  no-memory reaches `3171.732` update tokens/sec and `2910.873` total-window
  tokens/sec, while bounded memory reaches `3144.572` and `2880.835`, records
  `-0.856%` update and `-1.032%` total-window throughput versus no-memory, and
  keeps old-domain/replay retention slightly better. Treat the newer pair as
  current same-session memory-cost evidence, not a broad speed promotion over
  the older absolute-throughput pair or a broad quality claim.
- `language_continual_learning.py` reports
  `marulho_language_continual_training_window_triton_accounting.v1` inside
  `learning_evidence.training_window_triton_accounting`. The block is scoped to
  the measured update window and tracks RMSNorm, PLIF, route top-k, expert
  dispatch, memory slots, and sampled-vocab CE Triton/fallback/failure counts,
  so update-throughput regressions can be attributed without relying on a
  single aggregate tokens/sec number. The current accounting pair shows RMSNorm
  and PLIF as Triton-active in training, sampled-vocab CE as the maintained
  torch-autograd selected-row path, and memory-slot training as bounded
  torch-autograd rather than Triton autograd. Installed-brain continual
  learning and generation-repair summaries preserve the same measured fallback
  count under `tracked_torch_fallback_calls` with the legacy singular alias
  kept only for compatibility, so repair/learning evidence cannot silently
  report zero fallback work when the update window recorded PyTorch fallbacks.
  They also preserve `batch_device_staging` and
  `measured_update_loop_caller_device_transfer_calls=0` so installed evidence
  shows update batches were staged on the model device before timing.
- Continual-learning reports record `paired_update_replay_fusion` when compatible
  update/replay batches share one hidden forward in the measured loop while
  preserving separate update and weighted replay loss values. The report keeps
  `measured_update_loop_model_loss_calls` and avoided replay forward-call counts
  visible so speed claims can be checked against real skipped work.
  The accepted 2026-07-06 paired-fusion default report at `524288` update tokens
  records `4938.007` update tokens/sec, `4302.285` total-window tokens/sec,
  `256`
  fused steps, `256` avoided replay forward loss calls, `256` measured
  model-loss calls, `768` tracked torch fallback calls, and zero tracked Triton
  failures. The opt-in `--paired-sampled-vocab-loss` report records `256`
  fused sampled-vocab CE steps and reduces tracked torch fallback calls to
  `512`, but is slower at `4595.157` update tokens/sec and `4037.457`
  total-window tokens/sec, so it stays off by default as rejection evidence.
- Continual-learning experiment reports now require
  `experiment_review.records_active_compute=true` when
  `learning_evidence.active_compute.surface` is
  `marulho_language_continual_active_compute.v1`. That block records the
  current update shape's active columns, route candidates, memory-slot
  candidates, sampled-vocab loss rows, full-vocab materialization truth, total
  parameters, active-parameter estimates, and all-column/all-slot fallback truth
  without turning active-compute reporting into a runtime-promotion shortcut.
  The current `524288` active-compute rerun reaches `4635.942` update tokens/sec
  and `4054.424` total-window tokens/sec, `-6.117%` and `-5.761%` versus the
  retained paired default, while preserving `768` tracked torch fallback calls
  and zero tracked Triton failures.
- `language_continual_learning_experiment.py` exposes
  `--profile-update-stages` and records
  `experiment_review.records_update_stage_profile=true` when
  `learning_evidence.update_stage_profile.surface` is
  `marulho_language_continual_update_stage_profile.v1`. The profiler uses CUDA
  events for CUDA runs without per-stage synchronizations. The current profiled
  `524288` horizon-8 report identifies backward (`0.133416` ms/token) and
  paired forward/loss (`0.068868` ms/token) as the bottlenecks. The runner now
  accepts repeated `--sweep-recurrent-gradient-horizon` values to write
  `marulho_language_continual_speed_sweep.v1` plus one complete child report per
  candidate, refreshing the aggregate after each completed child. The current
  CUDA sweep accepts horizons `4`, `2`, and `1`; horizon `2` is selected at
  `4946.932` update tokens/sec and `4383.376` total-window tokens/sec, while
  horizon `1` accepts but is slower at `4910.548`/`4322.016`. All three children
  preserve `768` tracked torch fallback calls and zero Triton failures.
- Heldout and replay evaluation keeps the deferred scalar readback boundary but
  no longer builds loss/memory evidence for every eval batch. Evaluation
  reports now record `evidence_collection_mode=last_batch_only`,
  `per_batch_evidence_dict_build=false`, `evidence_probe_batch_tokens`, and
  `caller_device_transfer_calls`; total heldout loss still aggregates all eval
  batches on device before one readback.
- `language_continual_learning_experiment.py` exposes full-window backend
  toggles through `--sampled-vocab-ce-triton-training` and
  `--memory-slots-triton-training`, plus
  `--dense-adamw-backend {default,foreach,fused}` for dense parameters while
  sparse sampled-vocab rows remain on `SparseAdam`. It records
  `marulho_language_continual_training_backend_policy.v1`, and writes
  `training_sampled_vocab_ce_backend_summary` plus
  `training_memory_slot_backend_summary`. The `524288` no-memory sampled-vocab
  CE Triton report
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-no-memory-sampled-ce-triton-train-evalmatched-update524288-20260705.json`
  proves the opt-in path uses `512` Triton/autograd calls with zero fallback,
  but it is slower than the default selected-row torch path (`3083.988` versus
  `3171.732` update tokens/sec, `-2.766%`). The `524288` memory-slot
  Triton-training report
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-triton-train-evalmatched-update524288-20260705.json`
  proves `512` memory-slot Triton/autograd calls with zero memory fallback, but
  still trails bounded torch autograd (`3128.685` versus `3144.572` update
  tokens/sec, `-0.505%`). Both reports accept the online update with rollback
  verified; both are backend rejection evidence, not quality or runtime
  promotion. The dense AdamW backend sweep is also rejection evidence for now:
  the current-code default rerun reaches `3174.883` update tokens/sec and
  `2902.962` total-window tokens/sec, `foreach` reaches `3164.245` and
  `2900.942`, and `fused` reaches `3147.494` and `2879.977`, with rollback
  verified in all three runs.
- `language_eligibility_trace_runtime_impact.py` measures complete no-grad LM
  forward impact for deferred eligibility-trace updates. The current `524288`
  model-vocab batch-16/seq-64 report rejects deferred final-scan eligibility as
  a default because inline PLIF eligibility is faster despite kernel parity.
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
  extra model rows cannot be emitted as tokenizer bytes. It also accepts
  optional sustained decode controls and records their device-tensor state,
  counters, graph compatibility, graph-disable reason, and timeout/final status;
  graph-compatible controlled decode can stay on CUDA graph burst, while
  incompatible controls still fall back with an explicit reason.
- `language_generation_coherence.py` is the grounded prompt-suite review for
  checkpointed MARULHO-owned generation. It records raw continuations,
  source-prefix match, next-character source match, printability, token-run and
  bigram-diversity checks, active language path, and external-LLM absence. It
  can satisfy the benchmark suite's generation-coherence category, but it is
  not a human review, a broad generation-quality claim, or a runtime-promotion
  claim.
- `language_brain_generation_evidence.py` is the installed-brain generation
  evidence runner. It restores a `MarulhoBrain` checkpoint, verifies the
  installed LM tokenizer and non-mutating status reads, then runs the grounded
  prompt suite through public `MarulhoBrain.generate()`. The current
  direct-reviewed CUDA report
  `reports/language_brain_generation/direct-reviewed-horizon2-fresh-post-structure-generation-20260706.json`
  restores the fresh post-structure brain checkpoint, records brain-owned
  generation with no external/service-owned cognition, and passes `0/4`
  grounded cases. Treat it as path-ownership evidence and as the baseline that
  requires the installed repair lane, not as generation-quality evidence.
- `language_brain_generation_repair_evidence.py` is the installed-brain
  generation repair runner. It restores a `MarulhoBrain` checkpoint, builds
  hard-prompt replay windows, learns through
  `MarulhoBrain.learn_language_window()`, saves/restores a repaired brain
  checkpoint, then re-scores public `MarulhoBrain.generate()` and can attach a
  post-repair sustained run. It supports multiple installed-brain repair
  passes and candidate sweeps with per-candidate generation deltas, aggregate
  update throughput, and sustained generation only for the selected child. The
  current fresh CUDA report
  `reports/language_brain_generation_repair/direct-reviewed-horizon2-fresh-repair2-sweep-sustained524288-20260706.json`
  selects `candidate-01`, reaches `4/4` grounded prompt cases with zero prompt
  regressions, records `393216` update tokens at `5943.300` update tokens/sec
  and `5365.354` total-window tokens/sec, improves mean prefix match by `4.5`
  chars, and reaches `524288/524288` selected post-repair controlled sustained
  tokens at `8070.687` tokens/sec with zero tracked Triton failures. It is
  still evidence, not a broad generation-quality claim.
- `language_hf_curriculum_materializer.py` is the bounded Hugging Face
  curriculum materializer for moving generation repair beyond local prompt-only
  corpora. It fetches rows through Dataset Viewer, flattens structured rows
  through the `data` loader, writes a local corpus plus JSON report, records
  source/config/split/license/field provenance plus offsets and page counts,
  paginates larger row requests beyond one Dataset Viewer page, and keeps
  `external_llm_used=false`, `service_owned_cognition=false`, and
  `promotes_runtime_claim=false`. The `nvidia-open-repair-v1` preset covers
  open CC-BY NVIDIA/Nemotron SFT, math, preference, code, and persona-diversity
  sources; gated Nemotron pretraining corpora remain optional until access and
  terms are explicit.
- `language_training_experiment.py` now passes `max_train_batches` and
  `max_eval_batches` into `build_language_model_splits()` before fixed windows
  are packed as CPU/CUDA tensors. This keeps larger materialized corpora moving
  through bounded train/eval evidence windows instead of spending the run on
  full-corpus eval packing before the measured update begins. Reports carry
  pre-limit and post-limit split counts so bounded eval is visible.
- `language_generation_coherence.py` accepts `--auto-source-prompt-cases` to
  build grounded prompt cases from the supplied source corpus. Prompt-suite
  metadata records prompt text, thresholds, source hash, source length, and
  `raw_source_text_retained=false` instead of repeating large source bodies in
  every prompt case.
- `language_quality_replay_experiment.py` is the checkpoint-backed hard-prompt
  replay runner for fast quality iteration. It loads a parent LM checkpoint,
  builds replay pressure from grounded prompt continuations, can run one or
  more isolated child candidate arms with different learning/replay settings,
  ranks them by trained-prompt repair, heldout non-regression, replay/old-domain
  evidence, accepted online-update status, and update throughput, writes child
  checkpoints, then records selected-child generation coherence, heldout
  source-prompt coherence not used for replay training, one or more sustained
  runtime targets, and optional benchmark-suite aggregation. Its sustained
  summary distinguishes same-child controlled-decode evidence from ordinary
  sustained runs, including controlled house-scale availability when a selected
  child reaches `524288` tokens with decode controls. When
  `benchmark_suite_output_path` is set, it writes its selected-child report
  first, runs the benchmark suite with that report as quality-replay evidence,
  forwards optional memory-slot runtime impact, memory-slot architecture-cost,
  structural-plasticity, and GPU-kernel evidence paths, then rewrites the final
  report with the suite result. It is meant to move quality and speed evidence
  together without turning the benchmark suite into a gate-only workflow or
  hiding prompt regressions. It passes its max train/eval batch limits into the
  split builder before CPU/CUDA tensor packing, and its CLI can also use
  `--auto-source-prompt-cases` for source-anchored hard-prompt cases. It can
  import all or failed cases from saved generation-coherence reports with
  `--coherence-prompt-evidence` and `--failed-coherence-prompt-evidence`, and
  can import saved coherence reports as fixed heldout banks with
  `--heldout-coherence-prompt-evidence` or
  `--failed-heldout-coherence-prompt-evidence`. Reports expose any overlap
  between hard-prompt replay cases and heldout cases so validation cannot
  silently train on its own prompt bank.
- The current NVIDIA/Nemotron open-curriculum replay report
  `reports/language_quality_replay/nvidia-open-repair-preview-128x-auto-quality-replay-20260706.json`
  starts from the 128x materialized-corpus checkpoint, accepts `1048576`
  update tokens at `5930.903` update tokens/sec, repairs trained prompt
  coherence from `0/4` to `2/4`, repairs auto heldout prompt coherence from
  `0/4` to `4/4` with zero regressions, and reaches same-child controlled
  sustained decode at `131072/131072` and `524288/524288` tokens. The
  house-scale child report reaches `8526.804` tokens/sec through
  `torch_cuda_graph_burst_decode_controls` with zero tracked Triton fallback or
  failure. The paired suite
  `reports/language_benchmark_suite/language-suite-nvidia-open-repair-preview-128x-quality-replay-20260706.json`
  remains `blocked_missing_required_evidence` because trained prompt coherence
  is still partial. The prompt-bank repair follow-up
  `reports/language_quality_replay/nvidia-open-repair-preview-128x-prompt-bank-repair4-20260706.json`
  imports accumulated coherence cases, accepts `1048576` update tokens at
  `5861.141` update tokens/sec, reaches trained prompt coherence `9/9`, keeps
  heldout at `5/6`, and reaches `524288/524288` controlled sustained decode at
  `8605.296` tokens/sec. The fixed-heldout sweep
  `reports/language_quality_replay/nvidia-open-repair-preview-128x-fixed-heldout-repair5-sweep-20260706.json`
  trains three child candidates, selects `candidate-01` because it preserves a
  non-overlapping fixed heldout bank at `5/5`, accepts `1048576` update tokens
  at `5784.471` update tokens/sec, and reaches `524288/524288` at `8567.510`
  tokens/sec, but the selected child still regresses one trained prompt
  (`I'm ready to`, trained `9/10`). The paired suites
  `reports/language_benchmark_suite/language-suite-nvidia-open-repair-preview-128x-prompt-bank-repair4-20260706.json`
  and
  `reports/language_benchmark_suite/language-suite-nvidia-open-repair-preview-128x-fixed-heldout-repair5-sweep-20260706.json`
  remain `blocked_missing_required_evidence` on required generation coherence
  because no selected child currently has both full trained-prompt coherence and
  full heldout coherence. Treat this as data-backed repair progress plus speed
  evidence, not generation-quality promotion.
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
  compute, memory-slot runtime and architecture cost, GPU kernel correctness,
  checkpoint restore, rollback, service contract, and scale-ladder inventory.
  The suite writes a grounding-support
  source-term coverage subreport and can ingest existing final
  `marulho_language_sustained_runtime_evidence.v1` reports for the 8192/131072
  LM long-run gates, existing `marulho_language_triton_kernel_report.v1`
  reports for kernel correctness, including bounded memory-slot retrieval
  parity, and existing
  `marulho_language_generation_coherence_report.v1` reports for grounded
  prompt-suite coherence. It also ingests accepted
  `marulho_language_brain_installed_generation_evidence.v1` reports for
  installed-brain generation path evidence and can use a valid all-case
  installed-brain generation report as grounded generation coherence when the
  same brain checkpoint also has aligned long-run evidence. Accepted
  `marulho_language_brain_installed_generation_repair_evidence.v1` reports for
  installed-brain hard-prompt repair evidence. It also ingests accepted
  `marulho_language_continual_learning_experiment.v1` reports with
  `marulho_language_continual_memory_slot_architecture_cost.v1` sections, so
  bounded memory slots can be judged against no-memory baselines on update
  throughput, total-window throughput, forgetting, replay retention, and
  generation-probe deltas. Saved
  `marulho_language_structural_plasticity_experiment.v1` reports can enrich the
  growth/prune safety category when their transactions are MARULHO-owned,
  operator-approved, checkpoint-backed, rollback-verified, heldout-non-
  regressing, and bounded for route-bank or memory-slot expansion. Saved
  `marulho_language_quality_replay_experiment.v1` reports can enrich the
  generation-coherence category with selected-child replay evidence when the
  replay experiment is MARULHO-owned, checkpoint-lineaged, parent-preserving,
  heldout-protective, rollback-backed, and externally paired with same-child
  long-run evidence. Human review and broad generation-quality/runtime
  promotion remain false unless separately proven.
- `language_checkpoint_evolution_experiment.py` writes reusable
  `marulho_language_checkpoint_evolution_experiment.v1` reports around the
  controlled child-fork evaluator. These reports can start from a fresh model or
  a parent LM checkpoint, record backend policy, CUDA math policy, child update
  throughput, parent/child checkpoint hashes, parent-preserving rollback, and
  structural review evidence, then feed `language_runtime_benchmark_suite.py`
  through `--checkpoint-evolution-evidence`. They do not promote the child into
  the parent runtime; parent promotion still requires separate 131072/524288
  sustained evidence. Checkpoint artifacts are written under a short hashed
  `evo-*` directory derived from the output path, so long descriptive report
  names remain usable on Windows without weakening checkpoint lineage.
- The 2026-07-05 controlled child evolution report
  `reports/language_checkpoint_evolution/memory-slot-longtrain-triton-min1-child-evolution-update524288-20260705.json`
  starts from the memory-slot longtrain Triton-min1 parent checkpoint at the
  `524288` model-vocab, `1024` sampled-row, `16` expert, `1024` memory-slot
  shape. It runs isolated child learning plus replay for `1048576` accounted
  update tokens, reaches `3112.667` child update tokens/sec and `2762.033`
  total-window tokens/sec on `cuda:0`, keeps per-step metric CPU sync disabled,
  accepts the online update, applies a reviewed `expert_spawn` transaction from
  `16` to `18` experts, verifies rollback to the parent, and leaves the parent
  runtime unchanged. The child is eligible for operator promotion review but
  does not promote the parent or runtime claim by itself.
- Same-child controlled sustained reports for that evolved checkpoint reach
  `8192`, `131072`, and `524288` tokens at `4710.929`, `7794.403`, and
  `8066.423` tokens/sec through `torch_cuda_graph_burst_decode_controls` on
  `cuda:0`. The `524288` report uses `32768` graph replays, materializes no
  full `524288`-row generation logits, keeps decode controls on device, scores
  bounded memory slots without all-slot scans, and records all five tracked
  Triton kernels active with zero tracked Triton fallback or failure calls.
  The same-child grounded prompt suite still fails `0/4` cases with mean prefix
  `0.75` chars, so
  `language-suite-checkpoint-evolution-child-20260705.json` remains blocked on
  `generation_coherence` while passing the speed, rollback, checkpoint,
  structural, active-compute, and GPU-kernel evidence categories.
- The follow-up quality-replay repair
  `reports/language_quality_replay/evo-child-quality-repair-20260705.json`
  starts from that evolved child checkpoint and trains five candidate children
  under hard-prompt replay pressure while keeping heldout prompt review separate
  from replay training. The selected `candidate-03`
  (`learning_rate=0.0005`, `replay_loss_weight=1.5`, `max_steps=6`) accepts
  the online update at `3042.957` update tokens/sec and `2735.450`
  total-window tokens/sec. It repairs trained prompt coherence from `0/4` to
  `4/4`, moves trained mean prefix from `1.25` to `35.75` chars, repairs
  heldout prompt coherence from `0/4` to `4/4`, moves heldout mean prefix from
  `0.5` to `27.5` chars, and records zero heldout prompt regressions. Same
  selected-child controlled sustained reports reach `8192`, `131072`, and
  `524288` tokens at `5343.657`, `8037.887`, and `8136.788` tokens/sec through
  `torch_cuda_graph_burst_decode_controls`, with no full-vocab generation
  logits, no decode-control CPU token copy, all tracked Triton kernels active,
  and zero tracked Triton fallback calls. The combined suite
  `language-suite-evo-child-quality-repair-with-evolution-20260705.json`
  includes checkpoint-evolution, quality-replay, generation-coherence,
  house-scale controlled sustained, memory-slot cost/runtime, structural, and
  GPU-kernel evidence; it is `ready_for_review` with no failed or missing
  required categories and keeps `promotes_runtime_claim=false`.
- `language_checkpoint_promotion_review.py` turns that report set into an
  operator-review packet without installing a checkpoint. The current review
  `reports/language_checkpoint_promotion/evo-child-quality-repair-parent-promotion-review-20260705.json`
  verifies the selected `candidate-03` checkpoint hash, verifies that the
  quality-replay parent hash matches the evolved child checkpoint hash, keeps
  rollback bound to the evolved child parent, requires the combined suite's
  `ready_for_review` gate, and records
  `ready_for_operator_parent_promotion_review`. It explicitly keeps
  `eligible_for_live_parent_replacement=false`, `writes_live_checkpoint=false`,
  `mutates_runtime_state=false`, and `promotes_runtime_claim=false`; the next
  step is an operator-reviewed installation gate, not a status/read mutation.
- `MarulhoBrain.install_language_checkpoint_from_promotion_review()` is now that
  operator-reviewed installation gate. It consumes the promotion-review packet,
  requires explicit operator approval, re-hashes the selected child checkpoint,
  verifies lineage/rollback fields, loads the MARULHO LM checkpoint through
  `load_language_model_checkpoint`, installs it into brain-owned
  `marulho_lm_head` runtime state, and records a checkpointed
  `marulho_brain_language_checkpoint_installation.v1` report. Failed approval
  or hash evidence blocks without mutating the active language path. The install
  report remains a parent-installation fact, not a broad runtime-promotion
  claim.
- `language_brain_checkpoint_runtime_evidence.py` verifies the installed parent
  path from the brain side. The current report
  `reports/language_brain_runtime/evo-child-quality-repair-installed-brain-runtime-fast-surface-8192-20260705.json`
  installs the reviewed `candidate-03` child into `MarulhoBrain`, saves
  `reports/language_brain_runtime/evo-child-quality-repair-installed-brain-fast-surface-20260705.pt`,
  restores that brain checkpoint, and proves restored
  `active_language_path=marulho_lm_head` with no status-read mutation. The
  restored comparison `MarulhoBrain.generate()` loop reaches `8192/8192` at
  `112.177` tokens/sec, while the public brain-owned
  `MarulhoBrain.generate_sustained_language()` surface reaches `524288/524288`
  at `8064.765` tokens/sec through `torch_cuda_graph_burst_decode_controls` in
  `reports/language_brain_runtime/evo-child-quality-repair-installed-brain-public-sustained-524288-20260705.json`.
  This closes the private-evaluation reach-in for the fast installed-parent
  path and keeps `promotes_runtime_claim=false`.
- `language_brain_continual_learning_evidence.py` verifies installed-parent
  online learning from the brain side. It installs the reviewed parent, saves
  and restores the brain before learning, uses the verified checkpoint tokenizer
  only to build old/new/replay `LanguageBatch` inputs, calls
  `MarulhoBrain.learn_language_window()`, saves and restores the learned brain,
  and can run a post-learning sustained generation check. The current CUDA
  report
  `reports/language_brain_continual_learning/evo-child-quality-repair-installed-parent-continual-update524288-20260705.json`
  accepts `524288` update tokens at `3079.877` update tokens/sec and
  `2810.819` total-window tokens/sec, records `4194304` bounded memory-slot
  candidates without all-slot scans, improves new-domain loss by `4.7118`,
  improves old-domain loss instead of forgetting (`-4.0211`), improves replay
  loss (`-4.0014`), verifies the learned brain checkpoint restore, and reaches
  `524288/524288` post-learning sustained tokens at `8132.276` tokens/sec with
  all tracked generation Triton kernels active and zero tracked failures. This
  is public brain-surface continual-learning evidence, not a runtime promotion.
  The runner also accepts a recurrent-gradient horizon override, applies it
  through `MarulhoBrain.set_language_recurrent_gradient_horizon()` before the
  pre-learning checkpoint save, and gates that both the config horizon and the
  live state-block horizon survive pre-learning and learned-brain restore.
  It can also install a direct-reviewed local checkpoint through
  `MarulhoBrain.install_language_checkpoint_from_direct_review()` when stale
  promotion-review checkpoint payloads have been cleaned up; that path requires
  operator approval and an exact checkpoint SHA. The fresh CUDA report
  `reports/language_brain_continual_learning/direct-reviewed-horizon2-fresh-installed-continual-update524288-20260706.json`
  installs a `524288` model-vocab, horizon-2 direct checkpoint, accepts
  `524288` update tokens at `5938.519` update tokens/sec and `4913.377`
  total-window tokens/sec, improves new-domain loss by `7.0373`, improves
  old-domain loss instead of forgetting (`-6.9046`), improves replay retention
  (`-6.9228`), restores the learned brain checkpoint, and reaches
  `524288/524288` post-learning sustained tokens at `8341.802` tokens/sec with
  zero tracked Triton failures. This is direct installed-brain speed/learning
  evidence, not a runtime or generation-quality promotion.
- `language_brain_structural_plasticity_evidence.py` verifies installed-brain
  structural mutation from the brain side. It loads the learned brain checkpoint,
  saves and restores the pre-structure brain, builds eval batches with the
  checkpoint tokenizer, proposes and applies structure through
  `MarulhoBrain.propose_language_structure()` and
  `MarulhoBrain.apply_language_structure()`, saves and restores the
  post-structure brain, and can run post-mutation sustained generation. The
  current direct-reviewed CUDA report
  `reports/language_brain_structural_plasticity/direct-reviewed-horizon2-fresh-learned-route-bank-structure-524288-20260706.json`
  starts from the fresh horizon-2 learned brain checkpoint, applies route-bank
  expansion from `4` to `7` bounded candidates on the learned `524288`
  model-vocab brain, verifies baseline checkpoint restore and rollback
  evidence, records no status-read mutation, restores the post-structure brain
  checkpoint, and reaches `524288/524288` post-structure sustained tokens at
  `8157.211` tokens/sec through `torch_cuda_graph_burst_decode_controls` with
  zero tracked Triton failures.
  This is public brain-surface structural evidence, not a runtime or
  generation-quality promotion.
- `language_runtime_benchmark_suite.py` now accepts
  `--brain-installed-continual-learning-evidence` so that public brain-owned
  learning reports can strengthen the central continual-learning, forgetting,
  replay-recovery, and checkpoint-restore categories instead of living only as
  standalone JSON. The current aggregate
  `reports/language_benchmark_suite/language-suite-evo-child-installed-parent-learning-20260705.json`
  ingests the installed-parent learning report above, keeps continual learning,
  forgetting, replay recovery, and checkpoint restore at `pass`, records the
  `524288` brain update-token count plus `3079.877` update tokens/sec, and
  reaches `ready_for_review` while still keeping `promotes_runtime_claim=false`.
  It also accepts `--brain-installed-structural-plasticity-evidence` so
  installed-brain structural reports can strengthen the growth/prune category
  separately from standalone structural transactions.
- The current installed generation aggregate
  `reports/language_benchmark_suite/language-suite-evo-child-installed-parent-learning-structure-generation-20260706.json`
  accepts `--brain-installed-generation-evidence` beside the installed learning
  and structural reports. It reaches `ready_for_review` with long-run,
  checkpoint-level generation coherence, quality replay, Triton parity, and
  installed-brain generation evidence available. The installed-brain generation
  lane records `0/4` prompt cases passed and keeps
  `promotes_generation_quality_claim=false`.
- The current installed repair aggregate
  `reports/language_benchmark_suite/language-suite-direct-reviewed-horizon2-fresh-repair2-20260706.json`
  accepts `--brain-installed-generation-repair-evidence` and reaches
  `ready_for_review` with no failed or missing categories. Its generation gate
  accepts the repair lane only because the repaired child reaches `4/4`
  grounded prompt cases with zero regressions and aligns that same repaired
  checkpoint with `524288/524288` controlled sustained generation at
  `8070.687` tokens/sec, while keeping
  `promotes_generation_quality_claim=false`.
- The suite summarizes controlled sustained decode evidence inside the
  long-run throughput category when saved sustained reports include
  `generation_decode` or execution-level decode-control telemetry. Controlled
  8192/131072/524288 gates remain additive evidence; they do not replace
  generation-coherence or Triton/parity requirements.
- Generation coherence and long-run throughput must now pair on the same
  checkpoint before the generation category can pass. Installed generation-repair
  evidence can satisfy this category only when post-repair coherence is complete,
  prompt regressions are zero, and same-repaired-checkpoint controlled
  house-scale throughput is present. The controlled padded-vocab suite is blocked
  until that fast checkpoint has same-checkpoint prompt-suite
  evidence; the local same-checkpoint controlled report failed `0/4` cases with
  mean prefix match `0.0` and mean printable fraction `0.842`, so its
  `5537.062` token/sec controlled house-scale run remains speed evidence plus a
  quality blocker. The controlled coherence command below currently writes the
  blocked report and exits nonzero by design.
- The 2026-07-05 `diagnostic-post-window-evidence` memory-slot checkpoint also
  demonstrated why short diagnostic training is not enough quality evidence:
  the parent prompt suite failed `0/4`, and a four-candidate hard-prompt replay
  sweep selected `candidate-01` only as the least-bad child; it still failed
  `0/4`, with trained mean prefix moving only from `1.0` to `1.25` characters.
  Its same-child sustained runs did complete at `8192`, `131072`, and `524288`
  tokens, peaking at `4528.098` tokens/sec, but suite aggregation stayed
  blocked on `generation_coherence`. Preserve that report as rejection
  evidence, not a quality repair.
- The current same-architecture repair is longer training, not a decode trick.
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-524288-20260705.json`
  trains the `524288` vocab, `1024` sampled-row, routed-expert, bounded
  memory-slot shape for `128` optimizer records at `3009.616` train tokens/sec,
  lowers heldout loss from `7.0983` to `0.0890`, and records
  `per_step_evidence_dict_build=false`. The paired grounded suite
  `reports/language_generation_coherence/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-grounded-prompt-suite-20260705.json`
  passes `4/4` cases with mean prefix `32.75` chars, printable fraction `1.0`,
  and next-character match rate `1.0`. Same-checkpoint controlled sustained
  reports reach `8192/8192` at `3496.802`, `131072/131072` at `4400.930`, and
  `524288/524288` at `4524.673` tokens/sec through
  `torch_cuda_graph_burst_decode_controls`, with zero external/eager/decode
  control fallbacks and bounded `1024` memory slots (`8` candidates, `2`
  active, `runs_all_slots=false`). The suite
  `language-suite-memory-slot-longtrain-quality-speed-20260705.json` is
  `ready_for_review` with `17/17` pass/smoke categories and no missing required
  category, while keeping `promotes_runtime_claim=false` and
  `promotes_hot_path=false` because one-token sustained decode still reports
  Triton kernel fallback.
- The 2026-07-05 one-token Triton policy follow-up fixes that sustained
  fallback. `language_sustained_runtime_evidence.py` now records RMSNorm, PLIF,
  route-topk, expert-dispatch, and memory-slot Triton deltas separately, with
  explicit full-run versus streaming-core scope. The core defaults now use
  minimum row/token policy `1` for these inference kernels, and memory-slot
  retrieval treats `torch.no_grad()` as no-grad even when memory parameters are
  trainable. Same-checkpoint maintained reports
  `cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-default-triton-min1-8192-20260705.json`,
  `cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-default-triton-min1-131072-20260705.json`,
  and
  `cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-default-triton-min1-524288-20260705.json`
  reach `4460.070`, `7778.335`, and `8044.912` tokens/sec. All three use
  `torch_cuda_graph_burst_decode_controls`, all five tracked Triton kernels,
  `memory_slot_retrieval_backend=triton_no_grad_bounded_memory_slots`, and zero
  tracked Triton fallback calls. The refreshed suite
  `language-suite-memory-slot-longtrain-triton-min1-quality-speed-20260705.json`
  is `ready_for_review` with `17/17` pass/smoke categories and still keeps
  `promotes_runtime_claim=false` pending review/promotion.
- The same training shape rerun after the one-token policy change writes
  `reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-triton-min1-524288-20260705.json`
  and is the current same-checkpoint quality/speed evidence for this memory-slot
  LM shape. It reaches `3019.697` train tokens/sec, heldout loss `0.0862`, and
  source-continuation mean prefix `92.0` chars. The paired grounded prompt suite
  `reports/language_generation_coherence/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-triton-min1-grounded-prompt-suite-20260705.json`
  passes `4/4` cases with mean prefix `29.5`, printable fraction `1.0`, and
  next-character match rate `1.0`. Same-checkpoint controlled sustained reports
  at `8192`, `131072`, and `524288` tokens reach `4784.503`, `7740.123`, and
  `8013.881` tokens/sec with all five tracked Triton kernels active, zero
  tracked Triton fallback calls, and
  `memory_slot_retrieval_backend=triton_no_grad_bounded_memory_slots`. The
  refreshed suite
  `language-suite-memory-slot-longtrain-triton-min1-newcheckpoint-quality-speed-20260705.json`
  is `ready_for_review` with `17/17` pass/smoke categories and still keeps
  `promotes_runtime_claim=false` pending review/promotion.
- The current aligned same-scale controlled evidence uses
  `cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt`. Its anchored
  controlled prompt suite passed `4/4` cases with mean prefix match `16.75`,
  printable fraction `1.0`, and distinct-bigram fraction `0.865`; paired
  controlled sustained runs reached `8192/8192` at `1903.567` tokens/sec and
  `524288/524288` at `5444.635` tokens/sec with CUDA graph decode controls,
  zero CPU token copy, and zero decode-control fallbacks.
- The current quality-replay child evidence uses the protective candidate sweep
  `reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective.json`
  from the same parent checkpoint. Four child candidates were trained from the
  immutable parent. Faster candidates at `3149.910`, `3183.910`, and
  `3112.300` update tokens/sec repaired `Structural pressure` but each
  regressed one heldout source prompt. The selected replay-heavy child
  `candidate-03` (`learning_rate=0.0001`, `replay_loss_weight=2.5`,
  `max_steps=1`) accepted the update at `2796.957` update tokens/sec, moved
  trained prompt coherence from `3/4` to `4/4`, improved trained mean prefix by
  `18.0` chars, kept heldout coherence at `4/4`, recorded zero heldout
  regressions, and improved heldout mean prefix by `6.75` chars. The selected
  checkpoint sustained `8192/8192` at `3868.306` tokens/sec and
  `524288/524288` at `5341.039` tokens/sec through
  `torch_cuda_graph_burst_decode_controls`, with graph-compatible decode
  controls, zero CPU token copy, and zero decode-control fallbacks. Suite
  aggregation
  `language-suite-quality-replay-protective-selected-child-quality-ingested.json`
  now ingests the quality-replay report itself through
  `--quality-replay-evidence`, pairs selected child checkpoint lineage with the
  same-child `8192` and `524288` sustained reports, and reports
  `ready_for_review`, generation `pass`, long-run throughput `pass`, GPU kernel
  correctness `pass`, memory-slot cost `pass`, structural plasticity `pass`,
  all `17/17` categories pass/smoke, and `promotes_runtime_claim=false`. This
  is checkpoint-selected quality/speed tradeoff evidence, not a broad
  generation-quality or runtime promotion. Suite alignment now also blocks
  mixed-checkpoint controlled decode: when controlled-decode house-scale
  evidence exists, the generation-coherence checkpoint and the selected
  quality-replay child must be the same checkpoint that produced that
  controlled `524288` report.
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
- Current 2026-07-04 state-block mixed-state preallocation impact evidence in
  `reports/language_training_experiments/state-block-prealloc-runtime-impact-524288-b16-s64.json`
  uses `language_state_block_runtime_impact.py` to compare full no-grad LM
  forward passes while holding route-top-k and expert-dispatch policy constant.
  At the `524288` model-vocab, decode-limited, batch-16/seq-64, `16` expert,
  `8` route-candidate, `4` active-expert shape, the existing stacked
  mixed-state path reached `12321.430` tokens/sec and the preallocated
  no-grad mixed-state sequence reached `12068.368` tokens/sec, a `0.979x`
  ratio. Logits matched exactly within tolerance, PLIF/route/dispatch Triton
  kernels were active in the measured path, and the report keeps the
  preallocation unpromoted.
- Current 2026-07-05 local eligibility-trace Triton evidence in
  `reports/language_kernel_evidence/eligibility-trace-triton-20260705.json`
  passed six CUDA shape/dtype cases for final trace update over 64 recurrent
  steps, with geometric microbenchmark speedup `77.352x` over the PyTorch
  reference. The complete no-grad LM forward impact report
  `reports/language_training_experiments/eligibility-trace-runtime-impact-524288-b16-s64.json`
  rejects deferred eligibility as a default: inline PLIF eligibility reached
  `12760.575` tokens/sec while deferred final-scan eligibility reached
  `12148.414` tokens/sec (`0.952x`), with logit parity, `3200`
  no-eligibility PLIF Triton calls, and `50` eligibility final-scan Triton
  calls.
- Current 2026-07-04 route/vote top-k Triton evidence in
  `reports/language_kernel_evidence/route-topk-triton-20260704.json` passed
  three CUDA `float32` shape sweeps for `language_route_vote_topk`
  (`1024/2048/4096 x 128`, `64` experts, `8` candidates, `4` active experts)
  with exact selected-ID parity and geometric microbenchmark speedup `2.555x`
  over the PyTorch route-selection reference. The no-grad runtime path can use
  this primitive when the row-count policy allows it; gradient training remains
  on PyTorch route scoring/top-k so route keys keep gradients.
- Current 2026-07-04 route/vote top-k complete-runtime impact evidence in
  `reports/language_training_experiments/route-topk-runtime-impact-524288-b16-s64.json`
  uses `language_route_topk_runtime_impact.py` to compare full no-grad LM
  forward passes, not a kernel microbenchmark. At the `524288` model-vocab,
  decode-limited, batch-16/seq-64, `16` expert, `8` route-candidate, `4`
  active-expert shape, the PyTorch route-top-k fallback arm reached
  `12418.282` tokens/sec with 50 fallback route-top-k calls and the Triton arm
  reached `12972.201` tokens/sec with 50 Triton route-top-k calls, giving a
  `1.045x` throughput ratio. Logits matched within absolute tolerance and the
  report keeps `promotes_runtime_claim=false`.
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
- Current 2026-07-04 expert-dispatch complete-runtime impact evidence in
  `reports/language_training_experiments/expert-dispatch-runtime-impact-524288-b16-s64.json`
  uses `language_expert_dispatch_runtime_impact.py` to compare full no-grad LM
  forward passes while holding route-top-k policy constant. At the `524288`
  model-vocab, decode-limited, batch-16/seq-64, `16` expert, `8`
  route-candidate, `4` active-expert shape, the PyTorch expert-dispatch
  fallback arm reached `11699.767` tokens/sec with 50 fallback dispatch calls
  and the Triton arm reached `12371.776` tokens/sec with 50 Triton dispatch
  calls, giving a `1.057x` throughput ratio. The measured Triton arm also
  recorded 50 route-top-k Triton calls, `8192` candidate rows scored,
  `197888` active expert parameters per token, logit parity within absolute
  tolerance, and `promotes_runtime_claim=false`.
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
  `language-suite-sampled-vocab-kernel.json` closed the prior six-kernel GPU
  correctness set. The current eight-kernel suite snapshot is
  `language-suite-quality-replay-protective-selected-child-eligibility-controlled-aligned.json`,
  which includes route-top-k and local eligibility-trace parity,
  same-checkpoint generation coherence, and 524288-token sustained evidence
  while still keeping broad runtime promotion false.
- Current 2026-07-05 bounded memory-slot retrieval Triton evidence in
  `reports/language_kernel_evidence/memory-slots-triton-20260705.json` passed
  three CUDA `float32` shape sweeps for `language_memory_slot_retrieval` with
  `1024` memory slots, `8` bounded candidates, `2` active slots, and geometric
  microbenchmark speedup `4.950x` over the PyTorch selected-slot reference.
  `float16` memory-slot retrieval is explicitly unsupported until parity is
  proven. The runtime benchmark suite now includes
  `bounded_memory_slot_retrieval_parity` in GPU kernel correctness.
- Current 2026-07-05 memory-slot training-backend evidence in
  `reports/language_training_experiments/memory-slot-training-impact-triton-autograd-compare-524288-b16-s64-t524288-long.json`
  proves supported CUDA `float32` memory-slot training forward can use
  Triton-forward/custom-autograd when forced on. The forced-off torch arm
  reached `3076.582` train tokens/sec, the Triton training arm reached
  `3110.440`, and both arms measured `524288` optimizer tokens with precomputed
  memory candidate IDs and nonzero gate/slot gradients. Current full-window
  continual evidence keeps the default on torch autograd because the same
  `524288` update-token online shape reached `3134.337` update tokens/sec on
  torch versus `3074.512` with opt-in Triton.
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
- The deferred-metric follow-up in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-metrics-precomputed-sampled-vocab-524288.json`
  keeps the same current synthetic corpus and online shape, records
  `metric_readback_mode=deferred_gpu_scalar_aggregation`,
  `per_step_metric_cpu_sync=false`, and explicit CUDA synchronization before
  timing start/stop. It reaches `3089.664` train tokens/sec, `+2.173%` over the
  precompute-only report and `+17.957%` over the retained baseline, while still
  accepting the online update and avoiding full vocab logits.
- The eval-precompute/phase-timing follow-up in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-eval-precompute-deferred-metrics-524288.json`
  keeps the same current synthetic corpus and online shape, records old/new
  heldout eval sampled-vocab precompute, and shows heldout evaluation loss
  using `precomputed_batch_sampled_vocab_ids`. It reaches `3057.041` update
  tokens/sec and introduces `1690.405` total-window tokens/sec with phase
  timings: `0.839s` snapshot, `0.355s` sampled-vocab precompute, `5.829s`
  pre-update eval, `2.283s` optimizer setup, `21.438s` update, and `6.245s`
  post-update eval. The update-loop result is `+16.712%` versus the retained
  baseline but `-1.056%` versus the previous deferred-metric report, so this
  should be read as full-window visibility and eval-precompute evidence rather
  than a fresh update-loop speed promotion.
- The deferred-eval-metric follow-up in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-eval-metrics-524288.json`
  uses the repeatable `language_continual_learning_experiment.py` runner, keeps
  the same `22` old and `27` new heldout eval batch counts as the
  eval-precompute report, and records heldout
  `metric_readback_mode=deferred_gpu_scalar_aggregation`,
  `per_batch_metric_cpu_sync=false`, plus CUDA start/stop sync evidence. It
  reaches `2990.395` update tokens/sec and `1712.349` total-window tokens/sec,
  with phase timings of `0.840s` snapshot, `0.434s` sampled-vocab precompute,
  `5.682s` pre-update eval, `2.172s` optimizer setup, `21.915s` update, and
  `5.311s` post-update eval. Compared with the matched eval-precompute report,
  pre-update eval is `0.147s` faster, post-update eval is `0.934s` faster, and
  total-window throughput is `+1.298%`; update throughput is `-2.180%`, so this
  is heldout-eval sync reduction and repeatable report plumbing, not an
  update-loop speed promotion or language-quality claim.
- The generation-quality continual follow-up in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-quality-524288.json`
  keeps the same `524288` model vocab, `1024` sampled rows, `65536` updated
  tokens, `22` old heldout eval batches, and `27` new heldout eval batches, and
  adds MARULHO-owned old/new prompt continuations before and after learning. It
  records `records_generation_quality_probe=true`,
  `records_generation_quality_delta=true`, `external_llm_used=false`, and
  `promotes_generation_quality_claim=false`. The report improved
  next-character match from `0.0` to `1.0`, mean source-prefix match from `0.0`
  to only `1.0` character, and printable fraction from `0.956` to `1.0`, while
  distinct-bigram fraction fell from `0.362` to `0.191`. It reached
  `3065.952` update tokens/sec and `1688.289` total-window tokens/sec. This is
  useful generation-quality instrumentation around continual learning, but the
  raw old-domain continuation still repeats `replay`, so it is not a broad
  generation-quality promotion.
- The decode-control continual follow-up in
  `reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-decode-controls-524288.json`
  keeps the same `524288` model vocab, `1024` sampled rows, `65536` updated
  tokens, `22` old heldout eval batches, and `27` new heldout eval batches, but
  enables transparent greedy decode controls with `repetition_penalty=1.15` and
  `no_repeat_ngram_size=2`. It records
  `decode_controls_backend=torch_device_tensor`,
  `decode_controls_cpu_token_copy=false`, `1249` repetition-penalty token
  adjustments, `59` no-repeat banned-token events, zero decode-control
  fallbacks, `external_llm_used=false`, and
  `promotes_generation_quality_claim=false`. The after-learning
  distinct-bigram fraction improves from the previous `0.191` report to
  `1.000`, while update throughput is `3687.327` tokens/sec (`+20.267%` versus
  the previous generation-quality report) and total-window throughput is
  `2051.118` tokens/sec (`+21.491%`). Source-prefix match remains only `1.0`
  character and the raw text remains incoherent, so this is repetition-control
  evidence, not broad generation-quality promotion.
- Current 2026-07-04 padded-vocab generation-policy evidence in
  `reports/language_training_experiments/padded-vocab-generation-policy-524288-sustained.json`
  loaded a `524288` row checkpoint with `generation_vocab_size=262`, masked
  `524026` padded rows from generation, kept the generated tail inside tokenizer
  range (`max_tail=261`), and reached `524288/524288` tokens at `7248.118`
  tokens/sec on `cuda:0` with `torch_cuda_graph_burst`, `16` token bursts,
  `32768` graph replays, and zero CUDA graph failures. The report keeps
  `promotes_runtime_claim=false` and `promotes_hot_path=false`; it proves
  checkpoint restore plus decode masking, not broad generation quality.
- Current 2026-07-04 sustained decode-control evidence keeps the same
  checkpoint and enables `generation_repetition_penalty=1.15` plus
  `generation_no_repeat_ngram_size=3`. The superseded diagnostic report
  `reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-8192-sustained.json`
  reached `8192/8192` tokens at `104.126` tokens/sec on
  `torch_eager_cuda_decode_controls`, with CUDA graph burst disabled by
  `decode_controls_require_eager_history`, dense device prefix-table capacity
  `17984728`, zero decode-control fallbacks, and
  `decode_controls_cpu_token_copy=false`; its house-scale target report
  `reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-524288-timeout-sustained.json`
  wrote a timeout artifact after `68050/524288` tokens in `600.005s`
  (`113.416` tokens/sec), also with zero decode-control fallbacks. The
  graph-compatible follow-up
  `reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-8192-sustained.json`
  reached `8192/8192` at `3883.537` tokens/sec, and the house-scale graph
  report
  `reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-524288-sustained.json`
  reached `524288/524288` at `5537.062` tokens/sec with
  `torch_cuda_graph_burst_decode_controls`, `32768` graph replays,
  `decode_controls_graph_compatible=true`, `cuda_graph_decode_controls_used=true`,
  dense device prefix-table capacity `17984728`, zero decode-control fallbacks,
  and `decode_controls_cpu_token_copy=false`. This is `48.821x` faster than the
  previous eager controlled timeout path, while still `-23.607%` versus the
  plain graph-burst decode-limited report, so it is controlled-decode speed
  evidence rather than a broad runtime or language-quality promotion.
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
python -m marulho.evaluation.language_continual_learning_experiment --output reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-eval-metrics-524288.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-old-eval-batches 22 --max-new-eval-batches 27 --max-new-batches 8 --max-replay-batches 8 --max-steps 4 --learning-rate 0.002 --max-grad-norm 1.0 --gradient-clip-interval 8 --device cuda --comparison-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-eval-precompute-deferred-metrics-524288.json --original-baseline-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-524288.json --precompute-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288.json --deferred-metric-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-metrics-precomputed-sampled-vocab-524288.json
python -m marulho.evaluation.language_continual_learning_experiment --output reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-quality-524288.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-old-eval-batches 22 --max-new-eval-batches 27 --max-new-batches 8 --max-replay-batches 8 --generation-tokens 48 --max-steps 4 --learning-rate 0.002 --max-grad-norm 1.0 --gradient-clip-interval 8 --device cuda --comparison-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-eval-metrics-524288.json --original-baseline-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-524288.json --precompute-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288.json --deferred-metric-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-metrics-precomputed-sampled-vocab-524288.json
python -m marulho.evaluation.language_continual_learning_experiment --output reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-decode-controls-524288.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-old-eval-batches 22 --max-new-eval-batches 27 --max-new-batches 8 --max-replay-batches 8 --generation-tokens 48 --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 2 --max-steps 4 --learning-rate 0.002 --max-grad-norm 1.0 --gradient-clip-interval 8 --device cuda --comparison-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-generation-quality-524288.json --original-baseline-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-524288.json --precompute-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-precomputed-sampled-vocab-524288.json --deferred-metric-report reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-deferred-metrics-precomputed-sampled-vocab-524288.json
```

LM scale ladder inventory:

```bash
python -m marulho.evaluation.language_scale_ladder --output reports/language_scale_ladder/scale-ladder.json --include-smoke-fixture
```

LM benchmark suite:

```bash
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite.json --sustained-target-tokens 8
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite.json --sustained-target-tokens 8 --sustained-evidence reports/language_runtime_evidence/diagnostic-8192.json --sustained-evidence reports/language_runtime_evidence/long-gate-131072.json
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-memory-slot-architecture-cost.json --sustained-target-tokens 8 --memory-slot-architecture-cost-evidence reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-default-evalmatched-update524288-rerun.json
python -m marulho.evaluation.language_structural_plasticity_experiment --output reports/language_structural_plasticity/structural-memory-route-524288.json --proposal-kind memory_slot_expansion --proposal-kind route_bank_expansion --model-vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --memory-slot-growth 1024 --memory-slot-candidate-count 8 --active-memory-slot-count 2 --route-candidate-growth 4 --sequence-length 64 --stride 64 --batch-size 16 --max-eval-batches 2 --device cuda
python -m marulho.evaluation.language_checkpoint_evolution_experiment --output reports/language_checkpoint_evolution/default-policy-child-evolution-524288.json --parent-checkpoint reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt --model-vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --recurrent-gradient-horizon 8 --sequence-length 64 --stride 32 --batch-size 16 --max-parent-eval-batches 22 --max-child-eval-batches 27 --max-child-train-batches 8 --max-replay-batches 8 --max-steps 4 --learning-rate 0.002 --max-grad-norm 1.0 --gradient-clip-interval 8 --device cuda
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-structural-plasticity-524288.json --sustained-target-tokens 8 --structural-plasticity-evidence reports/language_structural_plasticity/structural-memory-route-524288.json --memory-slot-architecture-cost-evidence reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-default-evalmatched-update524288-rerun.json
python -m marulho.evaluation.language_triton_kernel_report --output reports/language_kernel_evidence/rmsnorm-triton-20260703.json --shape 1024x64 --shape 2048x128 --shape 1024x256 --dtype float32 --dtype float16 --warmup 20 --repeats 100
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-rmsnorm-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-batched-quality-rmsnorm-policy-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-batched-quality-rmsnorm-policy-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-vectorized-state.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-vectorized-state-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-vectorized-state-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json
python -m marulho.evaluation.language_triton_kernel_report --kernel selective-scan --output reports/language_kernel_evidence/selective-scan-triton-20260704.json --shape 16x128 --shape 32x128 --shape 16x256 --dtype float32 --dtype float16 --scan-time-steps 64 --warmup 20 --repeats 100
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-selective-scan-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json
python -m marulho.evaluation.language_state_block_runtime_impact --output reports/language_training_experiments/state-block-prealloc-runtime-impact-524288-b16-s64.json --vocab-size 524288 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 5 --repeats 50 --route-topk-min-rows 1 --expert-dispatch-min-tokens 1 --device cuda
python -m marulho.evaluation.language_triton_kernel_report --kernel eligibility-trace --output reports/language_kernel_evidence/eligibility-trace-triton-20260705.json --shape 16x128 --shape 32x128 --shape 16x256 --dtype float32 --dtype float16 --scan-time-steps 64 --warmup 20 --repeats 100
python -m marulho.evaluation.language_eligibility_trace_runtime_impact --output reports/language_training_experiments/eligibility-trace-runtime-impact-524288-b16-s64.json --vocab-size 524288 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 5 --repeats 50 --route-topk-min-rows 1 --expert-dispatch-min-tokens 1 --eligibility-trace-min-elements 1 --device cuda
python -m marulho.evaluation.language_triton_kernel_report --kernel route-topk --output reports/language_kernel_evidence/route-topk-triton-20260704.json --shape 1024x128 --shape 2048x128 --shape 4096x128 --dtype float32 --expert-count 64 --route-candidate-count 8 --active-experts 4 --warmup 10 --repeats 50
python -m marulho.evaluation.language_route_topk_runtime_impact --output reports/language_training_experiments/route-topk-runtime-impact-524288-b16-s64.json --vocab-size 524288 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 5 --repeats 50 --device cuda
python -m marulho.evaluation.language_triton_kernel_report --kernel expert-dispatch --output reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --shape 256x64 --shape 512x64 --shape 256x128 --dtype float32 --dtype float16 --expert-count 64 --active-experts 4 --expert-hidden-dim 128 --warmup 20 --repeats 100
python -m marulho.evaluation.language_expert_dispatch_runtime_impact --output reports/language_training_experiments/expert-dispatch-runtime-impact-524288-b16-s64.json --vocab-size 524288 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 5 --repeats 50 --route-topk-min-rows 1 --device cuda
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-expert-dispatch-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json
python -m marulho.evaluation.language_triton_kernel_report --kernel sampled-vocab-ce --output reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json --shape 512x128 --shape 1024x128 --shape 512x256 --dtype float32 --dtype float16 --vocab-size 8192 --sampled-vocab-size 1024 --warmup 20 --repeats 100
python -m marulho.evaluation.language_triton_kernel_report --kernel memory-slots --output reports/language_kernel_evidence/memory-slots-triton-20260705.json --shape 1024x128 --shape 2048x128 --shape 1024x256 --dtype float32 --dtype float16 --memory-slot-count 1024 --memory-slot-candidate-count 8 --active-memory-slots 2 --warmup 20 --repeats 100
python -m marulho.evaluation.language_memory_slot_training_impact --output reports/language_training_experiments/memory-slot-training-impact-triton-autograd-compare-524288-b16-s64-t524288-long.json --device cuda --vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --memory-slot-count 1024 --bounded-memory-slot-candidate-count 8 --active-memory-slot-count 2 --sequence-length 64 --batch-size 16 --warmup-steps 8 --repeats 512 --recurrent-gradient-horizon 8 --gradient-clip-interval 8
python -m marulho.evaluation.language_sampled_vocab_training_impact --output reports/language_training_experiments/sampled-vocab-training-impact-default-policy-524288-b16-r8-sampled-only.json --vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 2 --repeats 8 --skip-dense-baseline --device cuda
MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING=1 python -m marulho.evaluation.language_sampled_vocab_training_impact --output reports/language_training_experiments/sampled-vocab-training-impact-forced-triton-autograd-524288-b16-r8-sampled-only.json --vocab-size 524288 --sampled-vocab-size 1024 --embedding-dim 64 --state-dim 128 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --batch-size 16 --warmup-steps 2 --repeats 8 --skip-dense-baseline --device cuda
python -m marulho.evaluation.language_training_experiment --output reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744.json --model-vocab-size 524288 --sampled-vocab-size 1024 --state-dim 128 --embedding-dim 64 --expert-count 16 --active-expert-count 4 --route-candidate-count 8 --expert-hidden-dim 192 --sequence-length 64 --stride 32 --batch-size 16 --max-train-batches 256 --train-epochs 4 --generation-tokens 96 --sustained-target-tokens 524288 --sustained-timeout-seconds 1800 --device cuda
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/padded-vocab-generation-policy-524288-checkpoint.pt --output reports/language_training_experiments/padded-vocab-generation-policy-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 1200 --map-location cuda --no-environment-snapshot
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/padded-vocab-generation-policy-524288-checkpoint.pt --output reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-8192-sustained.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/padded-vocab-generation-policy-524288-checkpoint.pt --output reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 1200 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_generation_coherence --checkpoint reports/language_training_experiments/padded-vocab-generation-policy-524288-checkpoint.pt --output reports/language_generation_coherence/padded-vocab-decode-controls-grounded-prompt-suite-20260704.json --map-location cuda --min-case-pass-rate 1.0 --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-controlled-decode.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-8192-sustained.json --sustained-evidence reports/language_training_experiments/padded-vocab-generation-policy-decode-controls-graph-524288-sustained.json --generation-coherence-evidence reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json --generation-coherence-evidence reports/language_generation_coherence/padded-vocab-decode-controls-grounded-prompt-suite-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
python -m marulho.evaluation.language_generation_coherence --checkpoint reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt --output reports/language_generation_coherence/cuda-sampled-padded-default-policy-anchored-decode-controls-grounded-prompt-suite-20260704.json --map-location cuda --min-case-pass-rate 1.0 --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --prompt-case "MARULHO|64|8|0.10" --prompt-case "Replay protects|64|8|0.10" --prompt-case "Structural pressure can|64|8|0.10" --prompt-case "Long sustained runs|64|8|0.10"
python -m marulho.evaluation.language_quality_replay_experiment --checkpoint reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt --output reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective.json --device cuda --sequence-length 64 --stride 32 --batch-size 16 --max-new-batches 8 --max-replay-batches 8 --max-old-eval-batches 8 --max-new-eval-batches 8 --max-steps 2 --learning-rate 0.0008 --replay-loss-weight 0.35 --candidate-learning-rate 0.0008 --candidate-replay-loss-weight 0.35 --candidate-max-steps 2 --candidate-learning-rate 0.0004 --candidate-replay-loss-weight 0.75 --candidate-max-steps 2 --candidate-learning-rate 0.0002 --candidate-replay-loss-weight 1.5 --candidate-max-steps 1 --candidate-learning-rate 0.0001 --candidate-replay-loss-weight 2.5 --candidate-max-steps 1 --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --heldout-prompt-case-count 4 --prompt-case "MARULHO|64|8|0.10" --prompt-case "Replay protects|64|8|0.10" --prompt-case "Structural pressure|64|8|0.10" --prompt-case "Long sustained runs|64|8|0.10"
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-candidate-03-child-checkpoint.pt --output reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-8192.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-candidate-03-child-checkpoint.pt --output reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-524288.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 1200 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-quality-replay-protective-selected-child-eligibility-controlled-aligned.json --sustained-target-tokens 8 --sustained-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-8192.json --sustained-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-524288.json --generation-coherence-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-candidate-03-child-coherence.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/eligibility-trace-triton-20260705.json
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt --output reports/language_training_experiments/cuda-sampled-padded-default-policy-decode-controls-graph-8192-sustained.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_sustained_runtime_evidence --checkpoint reports/language_training_experiments/cuda-sampled-padded-default-policy-524288-63744-checkpoint.pt --output reports/language_training_experiments/cuda-sampled-padded-default-policy-decode-controls-graph-524288-sustained.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 1200 --map-location cuda --generation-repetition-penalty 1.15 --generation-no-repeat-ngram-size 3 --no-environment-snapshot
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-default-policy-controlled-aligned.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-sampled-padded-default-policy-decode-controls-graph-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-sampled-padded-default-policy-decode-controls-graph-524288-sustained.json --generation-coherence-evidence reports/language_generation_coherence/cuda-sampled-padded-default-policy-anchored-decode-controls-grounded-prompt-suite-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-sampled-vocab-kernel.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-quality-replay-protective-selected-child-quality-ingested.json --sustained-target-tokens 8 --sustained-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-8192.json --sustained-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-selected-child-sustained-524288.json --generation-coherence-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective-candidate-03-child-coherence.json --quality-replay-evidence reports/language_quality_replay/cuda-sampled-padded-default-policy-candidate-sweep-heldout-protective.json --memory-slot-runtime-impact-evidence reports/language_training_experiments/memory-slot-runtime-impact-triton-nograd-524288-b16-s64.json --memory-slot-architecture-cost-evidence reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-default-evalmatched-update524288-rerun.json --structural-plasticity-evidence reports/language_structural_plasticity/structural-memory-route-524288.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/eligibility-trace-triton-20260705.json --gpu-kernel-evidence reports/language_kernel_evidence/memory-slots-triton-20260705.json
python -m marulho.evaluation.language_generation_coherence --checkpoint reports/language_training_experiments/cuda-plif-surrogate-8192-checkpoint.pt --output reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json --map-location cuda --min-case-pass-rate 1.0
python -m marulho.evaluation.language_runtime_benchmark_suite --output reports/language_benchmark_suite/language-suite-generation-coherence.json --sustained-target-tokens 8 --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-8192-sustained.json --sustained-evidence reports/language_training_experiments/cuda-plif-surrogate-524288-sustained.json --generation-coherence-evidence reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/rmsnorm-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-forward-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/plif-surrogate-triton-20260703.json --gpu-kernel-evidence reports/language_kernel_evidence/selective-scan-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/route-topk-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/expert-dispatch-triton-20260704.json --gpu-kernel-evidence reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json
```
