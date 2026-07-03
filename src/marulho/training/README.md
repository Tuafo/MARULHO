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
  complete-runtime impact evidence are still required before promotion.
- `RMSNorm` now routes CUDA tensors through the language RMSNorm Triton
  primitive only for batched row counts where the kernel is measured useful.
  Streaming one-token LM generation keeps the faster CUDA graph/PyTorch
  fallback and exposes the fallback count in sustained reports.
- `language_continual_learning.py` is the first Iteration 6 foundation for the
  LM head. It applies bounded online updates, mixes replay batches, measures
  old/new heldout loss and replay retention, records spike-rate and throughput
  deltas, and keeps rollback snapshot hashes before accepting the update as
  review evidence.
- `RoutedLanguageExpertLayer` is the first Iteration 4 foundation for the LM
  head. It narrows token-hidden states through a bounded candidate plan, wakes
  only top-k experts, reports total/active columns, candidate rows scored,
  active parameters per token, route device, route latency, and explicit
  all-column fallback truth. Its no-telemetry inference path avoids host
  sleeping-expert materialization so CUDA graph capture can replay fixed-shape
  LM bursts. It is PyTorch/CUDA graph execution evidence, not a promoted
  block-sparse Triton dispatch path.
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
