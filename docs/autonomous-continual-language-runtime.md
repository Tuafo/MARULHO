# MARULHO Autonomous Continual Language Runtime

Last reviewed: 2026-07-04

This document is the maintained architecture lock for building MARULHO from the
current bounded readout runtime toward a MARULHO-owned continual language model.
It is a design contract, not implementation evidence. A capability is promoted
only when current code, tests, checkpoints, and long-run reports prove it.

## Current Boundary

The active runtime is `MarulhoBrain` plus `MarulhoTrainer`. It can load or
restore a checkpoint, feed source windows, tick ordered SNN token updates, emit
`BrainTrace`, generate through the local transition readout, run explicit replay
windows, review growth/prune hooks, and save state.

The current generation path is MARULHO-owned and reports
`external_llm_used=false`, but it is still a local transition readout over sparse
state. It is not yet a next-token language model with a vocabulary LM head,
heldout perplexity, checkpointed token embeddings, and scale-ladder evidence.

Iteration 2 has a training-owned foundation in `marulho.training.language_model`
and `marulho.data.language_tokenizer`: a deterministic byte-level tokenizer,
token embeddings, a selective spiking recurrent state block, a vocabulary LM
head, next-token loss, train/eval split reports, heldout loss/perplexity,
component checkpoint save/restore, and generation evidence that reports
`external_llm_used=false`. `MarulhoBrain` can now install those checkpointed
language components through a brain-owned adapter and route `generate()` through
`active_language_path=marulho_lm_head`; the local transition readout remains
fallback evidence. This is implementation evidence for the LM-head path, not a
promotion of online learning, the full live brain loop, or long-run language
capability.

Iteration 3 has a PyTorch foundation in `MarulhoSelectiveSpikingStateBlock`.
The block now uses RMSNorm, input-dependent leak and threshold terms, trainable
current terms, selective recurrent state, an eligibility trace cache, adaptive
timestep budgeting, streaming state-cache reuse, and spike/dead/over-firing
telemetry. It now has partial CUDA/Triton evidence for RMSNorm forward, PLIF
forward, `float32` PLIF surrogate backward, and standalone selective recurrent
scan, plus `float32` selected expert dispatch/combine and sampled-vocab CE
forward loss. This is not full hot-path promotion; half-precision expert/vocab
coverage, fallback integration, generation review, and complete-runtime impact
reports remain required.

Iteration 6 has a first bounded online-learning window for the LM head. The
training-owned executor snapshots model weights, applies new-domain gradient
updates with replay loss, measures old-domain forgetting, new-domain loss
delta, replay retention, spike-rate delta, update throughput, and rollback
hashes, then records the report in `MarulhoBrain` when the LM runtime is
installed. This is review evidence for continual learning, not long-run
promotion.

Iteration 4 has a first LM routed-expert foundation in
`RoutedLanguageExpertLayer`. The PyTorch path builds bounded token-hash route
candidates, scores only candidate rows, wakes top-k experts, reports total and
active columns, output candidate count, active parameters per token, route
device, latency, and whether all columns were scored. This is routing evidence
for the LM path; block-sparse CUDA/Triton dispatch and complete-runtime impact
reports remain future gates.

Iteration 7 has checkpoint-backed LM structural transactions for expert growth,
explicit expert prune, explicit expert merge, and explicit expert deep sleep.
The training-owned path creates a non-mutating expert-spawn proposal from
routing and learning pressure, a non-mutating expert-prune proposal from
explicit inactive/low-utility expert evidence, a non-mutating expert-merge
proposal from duplicate/high-similarity expert-pair evidence, or a non-mutating
expert-deep-sleep proposal from stale, low-activation, low-utility, high-cost,
or dead-spike expert evidence. Application requires operator approval, writes a
baseline checkpoint snapshot, applies the candidate topology change or
checkpointed sleep mask in isolation, checks heldout non-regression, and records
rollback hashes before `MarulhoBrain` accepts the candidate. This covers
expert-column growth/prune/merge/deep-sleep; split, synapse bundle, memory
expansion, route-bank expansion, and retire remain future transaction types.

Iteration 8 has a first controlled checkpoint-evolution evaluator in
`marulho.training.language_checkpoint_evolution`. It writes an immutable parent
checkpoint, forks an isolated child checkpoint, runs child-only
learning/replay/optional structural growth, compares parent and child heldout
evidence, verifies rollback to the parent hash, and records lineage metadata in
`MarulhoBrain` without replacing the installed parent model. This is controlled
internal evolution evidence, not self-copying, deployment, or automatic
promotion.

Iteration 9 now has a first LM-head sustained evidence runner in
`marulho.evaluation.language_sustained_runtime_evidence`. It streams the
checkpointed MARULHO LM head with recurrent cache state and writes JSON plus
README evidence for final, timeout, manual-stop partial, interrupt, and
exception outcomes. Reports include checkpoint metadata, token delta,
tokens/sec, active language path, device/backend, active routed columns, spike
health, fallback counts, environment contention, and promotion gates. CUDA runs
attempt ordered `torch_cuda_graph_burst` replay over the configured
`quantum_tokens`, with graph setup time, graph replay count, graph token count,
eager tail tokens, and graph failure reason recorded. The runner is component
evidence for the LM head; it does not promote generation quality or replace the
full `MarulhoBrain` sustained runtime gate.

`marulho.evaluation.language_training_experiment` is now the fast mutable LM
science loop. It trains a configurable routed selective-spiking LM on local
text using packed device-resident windows, records training throughput plus
heldout loss/perplexity before and after the update, emits MARULHO-owned
generation samples with source-continuation probes, saves a checkpoint, and
runs paired sustained inference. Its CUDA update loop now defers per-batch
scalar metric readback, keeps loss and gradient-norm records as device scalars,
and synchronizes once before stopping the measured training timer. Its job is
to accelerate bigger experiment cycles; it records generated text and metrics
honestly rather than turning every run into a new gate. The current
deferred-metric CUDA report trained the same PLIF-surrogate `63744` token shape
at `2720.929 train tokens/sec`, versus the older `2596.380 train tokens/sec`
per-batch-readback report, with `3840` Triton PLIF backward calls,
`cuda_synchronized_before_timing_start=true`,
`cuda_synchronized_before_timing_stop=true`, and
`per_batch_metric_cpu_sync=false`. The paired `524288` sustained report reached
`7502.156 tokens/sec` on `torch_cuda_graph_burst`, so this is training-loop
host-boundary improvement with neutral house-scale inference evidence, not a
general language-coherence or runtime-promotion claim.

`marulho.evaluation.language_generation_coherence` is the grounded prompt-suite
review for checkpointed MARULHO-owned generation. It records raw continuations,
source-prefix match, next-character source match, printability, token-run and
bigram-diversity checks, active language path, and external-LLM absence for each
prompt. The current 2026-07-04 report
`reports/language_generation_coherence/plif-surrogate-grounded-prompt-suite-20260704.json`
uses the PLIF-surrogate checkpoint and passes `4/4` anchored prompts with mean
prefix match `46` characters, mean prefix fraction `0.71875`, printable
fraction `1.0`, and next-character match rate `1.0`. It is prompt-suite
coherence evidence, not a human review or broad generation-quality claim.
The 2026-07-05 bounded memory-slot longtrain checkpoint
`reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-524288-20260705-checkpoint.pt`
extends that evidence to the `524288` model-vocab, `1024` sampled-row,
`16`-expert, `1024` memory-slot architecture. Its grounded prompt suite passes
`4/4` cases with mean prefix match `32.75` characters, printable fraction
`1.0`, and next-character match rate `1.0`; same-checkpoint controlled
sustained reports reach `8192`, `131072`, and `524288` tokens at `3496.802`,
`4400.930`, and `4524.673` tokens/sec. This repairs the short diagnostic
memory-slot checkpoint that failed `0/4` coherence even after a shallow
quality-replay sweep. It remains review evidence only: broad generation quality,
runtime promotion, and one-token Triton hot-path promotion remain false.
The follow-up one-token Triton policy rerun keeps the same checkpoint and
decode controls, lowers default inference kernel row/token thresholds to `1`,
fixes no-grad memory-slot dispatch for trainable parameters, and reaches
`4460.070`, `7778.335`, and `8044.912` tokens/sec at `8192`, `131072`, and
`524288` tokens with RMSNorm, PLIF, route-topk, expert-dispatch, and
memory-slot Triton kernels active and zero tracked Triton fallback calls. The
refreshed suite
`reports/language_benchmark_suite/language-suite-memory-slot-longtrain-triton-min1-quality-speed-20260705.json`
is `ready_for_review` with `17/17` pass/smoke categories, while still keeping
runtime-promotion false pending review.
The same longtrain shape rerun after that policy change produces
`reports/language_training_experiments/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-longtrain-triton-min1-524288-20260705.json`:
`3019.697` train tokens/sec, heldout loss `0.0862`, source-continuation mean
prefix `92.0`, grounded prompt suite `4/4` with mean prefix `29.5`, and
controlled `8192`/`131072`/`524288` sustained decode at `4784.503`,
`7740.123`, and `8013.881` tokens/sec with all five tracked Triton kernels
active and zero tracked Triton fallback calls. Its refreshed suite
`reports/language_benchmark_suite/language-suite-memory-slot-longtrain-triton-min1-newcheckpoint-quality-speed-20260705.json`
is also `ready_for_review` with `17/17` pass/smoke categories; runtime promotion
remains false until review/promotion criteria are explicitly satisfied.
For online continual learning at the preferred `524288` update-token scale, the
current min1-policy training-accounting matched pair
`reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-no-memory-triton-min1-training-accounting-evalmatched-update524288-20260705.json`
versus
`reports/language_continual_learning/cuda-sampled-padded-horizon8-tf32-clip8-memory-slots-triton-min1-training-accounting-evalmatched-update524288-20260705.json`
accepts both arms with rollback verification, no forgetting regression, and
replay retention improvement. No-memory reaches `3171.732` update tokens/sec
and `2910.873` total-window tokens/sec; bounded memory reaches `3144.572` and
`2880.835`, giving memory slots a small `-0.856%` update and `-1.032%`
total-window cost in the same session while scoring `4194304` bounded memory
candidates without all-slot scans. This shows memory slots are not the primary
training bottleneck, but the older retained pair had higher absolute update
throughput, so broader training speed remains an active goal item. The reports
expose
`marulho_language_continual_training_window_triton_accounting.v1` for RMSNorm,
PLIF, route top-k, expert dispatch, memory slots, and sampled-vocab CE inside
the measured update window; current training accounting shows RMSNorm and PLIF
as Triton-active, sampled-vocab CE on the maintained torch-autograd selected-row
path, and memory-slot training on bounded torch autograd.

The target runtime must preserve these boundaries:

- no hidden external LLM, NIM, Cortex, or ThoughtLoop as the brain;
- no service-owned cognition or service-owned scheduler policy;
- no status/read endpoint mutation;
- no hot-path structural mutation;
- no promotion from toy-only or short-run evidence;
- GPU/Triton/native graph evidence for promoted hot paths;
- checkpoint-backed rollback for learning, replay, and topology mutation.

## Ownership Map

- `brain`: lifecycle owner, source queue, feed/tick/generate/replay/grow-prune
  orchestration, compact `BrainTrace`, checkpoint path continuity.
- `data`: tokenizer adapter, source windowing, source-to-token preparation,
  encoder tensors, and device evidence for input material.
- `training`: token sequence execution, recurrent state cache, language loss,
  online update loop, checkpoint serialization, CUDA/native graph execution, and
  sustained runners.
- `core`: PLIF/adaptive-LIF mechanics, sparse columns, routing, plasticity,
  Triton kernels, and tensor-level device evidence.
- `semantics`: bounded language/readout artifacts, grounding support,
  generation evidence packets, and language capability gates.
- `consolidation`: CPU archival memory, replay records, replay selection
  metadata, and explicit consolidation evidence.
- `retrieval`: routing caches, exact tensor candidate search, and future
  bounded GPU candidate routers.
- `evaluation`: promotion gates, long-run reports, loss/perplexity, forgetting,
  rollback, kernel parity, and service contract checks.
- `service`: thin `/brain/*` adapter only.

## Runtime Truth And Evidence

Runtime Truth is the operator-facing proof surface for this architecture. It is
not a broad status schema and it is not a read-side worker. For the autonomous
language runtime, Runtime Truth must be emitted from runtime-owned state and
validation reports, while `/brain/status`, status streams, report summaries, and
UI views stay read-only projections.

Required Runtime Truth fields for language work:

- `runtime_owner=MarulhoBrain`;
- `trainer_owner=MarulhoTrainer` when neural execution runs;
- `active_language_path`, with values such as
  `local_transition_readout` or `marulho_lm_head`;
- `external_llm_used=false`;
- `thought_loop_used=false`;
- `cortex_used=false`;
- checkpoint path and checkpoint lineage hash;
- tokenizer hash and vocabulary hash when the LM path is active;
- train/eval split hash for loss and perplexity claims;
- tick tokens and quantum tokens;
- device/backend and executor names;
- CUDA/Triton/native graph failure and fallback counters;
- active columns, total columns, and active parameters per token;
- route candidate rows scored and `runs_all_columns`;
- spike rate, dead-neuron rate, and over-firing rate;
- replay/consolidation events and bounded replay window ids;
- growth/prune proposal counts, applied mutation ids, and rollback ids;
- final or partial report status for every sustained run.

Runtime Truth must fail closed. If device evidence is missing, the claim is
CPU/unknown. If route evidence is missing, bounded routing is not promoted. If a
generation packet cannot prove the active language path and external-LLM
absence, it is not promoted as MARULHO language generation.

## Tokenizer And Vocabulary

The language runtime uses a MARULHO-owned tokenizer adapter. The adapter may load
a local tokenizer spec or train a local byte-level BPE or unigram vocabulary from
approved source windows, but it must not import a hidden language-model
checkpoint or hidden generation stack.

Required tokenizer state:

- vocabulary tokens and ids;
- byte fallback for unknown text;
- normalization policy;
- special token ids for BOS, EOS, PAD, UNK, checkpoint boundaries, and replay
  markers;
- tokenizer hash and vocabulary hash;
- source corpus manifest hash;
- deterministic encode/decode tests;
- checkpoint serialization and restore validation.

Small CI fixtures may use tiny vocabularies, but the same adapter contract must
support production vocabularies in the 32k to 256k range without changing the
runtime ownership model.

## Source Windows To Spike State

Source ingestion remains a runtime-source boundary, not a cognition claim. The
target path is:

1. `data` selects bounded source windows and encodes them to token ids.
2. `training` stages token ids into the ordered execution quantum.
3. A token embedding table maps ids to `d_model`.
4. Source-position, source-kind, and replay-kind features are added as bounded
   conditioning, not as hidden text reasoning.
5. `core` converts the embedded stream into sparse spike/state input through a
   learned projection and optional top-k sparsifier.
6. `training` executes recurrent selective spiking blocks in causal order.
7. `semantics` and `evaluation` receive evidence packets; they do not own the
   neural update.

The production path must keep sequential token semantics exact. Wider quanta or
native graph loops are execution boundaries, not parallel language cognition.

## Selective Spiking State Block

The language core is a scale-ready recurrent SNN/state-space block:

```text
x_t = token_embedding[token_id] + source_features
r_t = route_context(x_t, state_cache, active_columns)
i_t = W_in x_t + W_rec z_(t-1) + W_route r_t
beta_t = sigmoid(beta_0 + W_beta x_t + W_beta_state s_(t-1))
theta_t = softplus(theta_0 + W_theta x_t)
u_t = beta_t * u_(t-1) + i_t - z_(t-1) * theta_t
z_t = step(u_t - theta_t)
a_t, b_t, c_t = selective_state_parameters(x_t)
s_t = a_t * s_(t-1) + b_t * z_t
y_t = RMSNorm(x_t + W_out(c_t * s_t + z_t))
```

Where:

- `u_t` is membrane voltage;
- `z_t` is the sparse spike event vector;
- `s_t` is the selective recurrent state;
- `beta_t` is trainable leak;
- `theta_t` is trainable threshold;
- `r_t` is routed column/expert context;
- `y_t` is the residual stream passed to the next block or LM head.

The block may run as PyTorch CPU/CUDA for correctness tests. Promotion requires a
CUDA/Triton/native graph path with parity, fallback-before-mutation, and
complete-runtime impact evidence.

## Surrogate Gradient

The forward spike uses a hard threshold. The backward path uses a bounded
surrogate gradient, selected per block and recorded in checkpoint metadata.

Initial promoted candidates:

- fast sigmoid: `d step(v) / dv ~= 1 / (1 + alpha * abs(v))^2`;
- triangular window: nonzero only inside `abs(v) <= gamma`;
- straight-through estimator only as a test baseline.

Promotion requirements:

- deterministic seed coverage where supported;
- gradient finite checks;
- dead-neuron and over-firing telemetry;
- spike-rate bounds;
- heldout loss/perplexity comparison;
- throughput impact on the complete runtime, not only a kernel microbench.

## Selective State-Space Recurrence

MARULHO avoids quadratic attention as the default long-context mechanism. The
language path uses recurrent selective state inspired by state-space and RWKV
style execution while preserving SNN event sparsity.

State cache per layer:

- membrane `u`;
- spike trace `z`;
- selective recurrent state `s`;
- route/expert cache;
- short eligibility trace;
- RNG/determinism metadata where needed.

Streaming inference must reuse the cache without recomputing prior tokens.
Training may use truncated BPTT windows plus replay windows. Long-context
capability is measured through loss/perplexity and generation review at longer
source lengths, not claimed from recurrence alone.

## GPU/Triton-First Execution

The hot path is designed for GPU-first execution while preserving CPU/PyTorch
fallbacks for correctness tests and fail-closed recovery. Test fixtures may run
on CPU, but promoted runtime claims require observed device evidence and
complete-runtime impact reports.

GPU-first rules:

- keep token embeddings, recurrent state, route scoring, active column state,
  LM-head logits, and plastic module tensors on the selected model device;
- avoid per-token host synchronization except at explicit host-truth,
  telemetry, stop, checkpoint, replay, or fallback boundaries;
- use Triton kernels for narrow tensor hot spots only after PyTorch parity and
  complete-runtime profiling justify the kernel;
- use native CUDA graph sequence execution only when pointer, shape, device, and
  state-cache preflight passes before mutation;
- require `torch.cuda.CUDAGraph.raw_cuda_graph()` or an equivalent audited raw
  child `cudaGraph_t` bridge before promoting parent-graph executors; otherwise
  report `torch_cudagraph_raw_handle_unavailable` and fall back before mutation;
- allow a separately reported `torch_sequence_graph_*` executor for full
  sequence quanta when it preserves state parity and records native-parent
  graph counters as unavailable, not as conditional-WHILE success;
- stage CPU archival memory to the model device only for explicit replay
  computation, never as a status/read side effect;
- report setup/compile/capture overhead separately from steady-state tokens/sec;
- record fallback reason before any CPU/PyTorch path mutates model state.

The first LM implementation may start with PyTorch CPU/CUDA correctness. It is
not promoted until the sustained runner shows the active LM path, device/backend
truth, fallback counts, and neutral-or-better runtime impact against the same
checkpoint family.

## Routed Column And Expert Runtime

Columns become sparse routed specialists. The router is training-owned and uses
bounded candidate evidence before waking expensive work.

Runtime contract:

- route bank indexes column/expert keys and current utility;
- top-k active columns are selected per token or per short quantum;
- candidate scoring is bounded by retrieved route rows when available;
- dense/all-column fallback is allowed only with explicit fallback truth;
- block-sparse dispatch/combine handles active experts;
- active parameter count per token is reported.

Required route evidence:

- total columns;
- active columns;
- active parameters per token;
- candidate rows scored;
- output candidate count;
- `runs_all_columns`;
- fallback reason;
- route device;
- route latency;
- load/utility/cost telemetry.

## Memory, Transition, And Replay Stores

The goal term "memory cortex" maps to MARULHO-owned memory and consolidation
machinery. It is not the retired Cortex/ThoughtLoop path.

Stores:

- recent source window cache for current learning context;
- transition memory for sparse sequence associations;
- CPU archival replay records in `consolidation`;
- surprise replay queue;
- general replay protection set;
- old-domain heldout set;
- checkpoint lineage and mutation proposal ledgers.

Replay is selected, bounded, explicit, and evidence-backed. It must not become
every-token hidden work, status-read mutation, or service-owned cognition.

## Language Readout And LM Head

The promoted language path is a vocabulary LM head:

```text
logits_t = W_vocab y_t + b_vocab
loss_t = cross_entropy(logits_t, token_id_(t+1))
```

Required components:

- tokenizer adapter;
- token embedding table;
- selective spiking state stack;
- vocabulary LM head;
- sampled/adaptive vocab loss for large vocabularies;
- train/eval split loader;
- heldout loss and perplexity;
- checkpoint save/restore for tokenizer, embeddings, recurrent state, LM head,
  optimizer, replay state, and route bank;
- generation packet with `active_language_path=marulho_lm_head` and
  `external_llm_used=false`.

The existing local transition readout may remain as evidence and fallback while
the LM head is being built. It must not be renamed into a full LM claim.

## Online Continual Learning Loop

Online learning runs under `MarulhoBrain.tick` orchestration and
`MarulhoTrainer` execution ownership:

1. consume a bounded source token window;
2. run causal recurrent SNN forward;
3. compute next-token loss on train positions;
4. update fast local plasticity or eligibility traces;
5. run bounded gradient update for plastic language modules;
6. record replay candidates from surprise, loss spikes, and grounding gaps;
7. evaluate old/new heldout probes on cadence;
8. emit `BrainTrace` and report JSON evidence.

Learning improvement is not accepted without forgetting measurement. The online
loop must report new-domain improvement, old-domain forgetting, general replay
retention, perplexity delta, spike-rate delta, active-column delta, memory
growth, and throughput impact.

## Replay And Consolidation Loop

Replay and consolidation are slow-path windows:

- recent source replay repairs short-term drift;
- surprise replay revisits high-loss or high-uncertainty tokens;
- general replay protects old-domain capability;
- transition replay updates sparse associations;
- consolidation compacts replay records into checkpoint-backed state.

Replay promotion gate:

- selected replay window is bounded and hash-backed;
- replay has no hidden raw-text reasoning;
- old/new heldout metrics are both reported;
- forgetting is below the configured threshold;
- throughput impact is measured against the same checkpoint family;
- replay artifacts survive save/restore validation.

## Structural Growth And Prune Controller

Structural mutation is internal developmental mutation, not uncontrolled
replication. It is proposal-first and never hot-path.

Proposal kinds:

- new column;
- column split;
- new synapse bundle;
- expert spawn;
- memory slot expansion;
- route-bank expansion;
- deep sleep;
- prune;
- merge;
- retire.

Growth triggers:

- repeated prediction failure;
- high-surprise streak;
- novel concept cluster;
- replay conflict;
- low confidence with high uncertainty;
- route saturation;
- specialist overload.

Prune/sleep triggers:

- low utility;
- low activation;
- high cost with low contribution;
- duplicate function;
- harmful interference;
- dead spike rate;
- stale expert.

Every proposal records trigger evidence, bounded budget, expected active compute
impact, checkpoint parent, rollback plan, isolated evaluation plan, and operator
review status where required.

## Checkpoint Mutation Protocol

All durable topology, tokenizer, optimizer, replay, and LM-head mutations follow
the same transaction shape:

1. capture parent checkpoint metadata and hash;
2. create mutation proposal with budget and evidence;
3. run dry-run or isolated child evaluation when applicable;
4. snapshot parent checkpoint before apply;
5. apply mutation in a bounded transaction;
6. run post-mutation tests and heldout gates;
7. promote only if gates pass;
8. rollback to parent on failure;
9. record lineage metadata and rollback evidence.

Failure must restore the previous checkpoint or keep the child quarantined. No
service endpoint may mutate state through a read/status path.

## Triton Kernel Map

Kernel promotion starts with correctness and complete-runtime evidence.

| Priority | Kernel | Owner | Promotion evidence |
| --- | --- | --- | --- |
| 1 | PLIF/adaptive-LIF forward | `core`/`training` | forward parity exists; spike-rate telemetry and complete tick impact remain |
| 2 | PLIF/adaptive-LIF backward | `core`/`training` | `float32` surrogate parity and training impact exist; half precision remains open |
| 3 | selective recurrent state scan | `core`/`training` | standalone parity exists; training-loop fusion, cache restore, and long-context impact remain |
| 4 | route/vote top-k selection | `core`/`retrieval` | bounded rows, no all-column scan, route latency |
| 5 | block-sparse column transition | `core`/`training` | active params/token, dense fallback truth |
| 6 | expert dispatch/combine | `core`/`training` | `float32` sparse dispatch parity exists; half precision and complete-runtime impact remain |
| 7 | fused RMSNorm/residual/membrane centering | `core` | numerical parity, stability impact |
| 8 | eligibility trace update | `core`/`training` | local plasticity parity, no hidden replay |
| 9 | replay gather/scatter | `consolidation`/`training` | bounded replay placement, memory footprint |
| 10 | sampled/adaptive vocab cross entropy | `training` | loss parity, vocab sweep, large-vocab throughput |
| 11 | fused optimizer step | `training` | optimizer parity, checkpoint restore fidelity |

Each kernel must have PyTorch fallback, shape sweep, dtype coverage where
supported, deterministic mode where possible, failure-before-mutation behavior,
benchmark against baseline, and complete-runtime impact report.

The first LM-head kernel evidence slice covers the RMSNorm forward primitive.
`language_rmsnorm_triton.py` provides the Triton kernel, PyTorch fallback,
autograd-compatible backward, and runtime-use counters. The 2026-07-03 report
`reports/language_kernel_evidence/rmsnorm-triton-20260703.json` passed
`float32`/`float16` shape sweeps with geometric microbenchmark speedup `1.440x`.
Forced Triton on one-token streaming was measured and rejected for sustained
generation throughput, so the maintained policy uses Triton for batched rows
and reports PyTorch/CUDA fallback for streaming rows until a narrower streaming
kernel wins complete-runtime evidence.

The second LM-head kernel evidence slice covers PLIF/adaptive-LIF forward.
`language_plif_triton.py` provides the Triton kernel, PyTorch fallback,
runtime-use counters, and a hard-spike forward reference for membrane, spike,
selective-state, eligibility-trace, and mixed-state updates. The 2026-07-03
report `reports/language_kernel_evidence/plif-forward-triton-20260703.json`
passed `float32`/`float16` shape sweeps with geometric microbenchmark speedup
`3.145x`.

The third LM-head kernel evidence slice covers `float32` PLIF/adaptive-LIF
surrogate backward. `language_plif_triton.py` now has a custom autograd path
whose Triton backward matches the current hard-spike forward and sigmoid
surrogate derivative. The 2026-07-03 report
`reports/language_kernel_evidence/plif-surrogate-triton-20260703.json` passed
three CUDA `float32` shape sweeps with geometric forward+backward microbenchmark
speedup `1.662x` and marks `float16` backward unsupported. The complete
training-impact report `reports/language_training_experiments/cuda-plif-surrogate-8192.json`
trained `63744` tokens at `2596.380 train tokens/sec` with `3840` Triton
forward and `3840` Triton backward calls, and the paired house-scale sustained
report reached `524288` tokens at `7578.052 tokens/sec`.

The fourth LM-head kernel evidence slice covers the standalone selective
recurrent state scan. `language_selective_scan_triton.py` provides the Triton
kernel, PyTorch fallback, runtime-use counters, and forced parity/benchmark
execution for `state[t] = decay[t] * state[t-1] + input[t] * spike[t]` over
`[batch,time,state_dim]` tensors. The 2026-07-04 report
`reports/language_kernel_evidence/selective-scan-triton-20260704.json` passed
six CUDA `float32`/`float16` shape sweeps at 64 recurrent steps with geometric
microbenchmark speedup `114.077x`. The paired suite report
`reports/language_benchmark_suite/language-suite-selective-scan-kernel.json`
records RMSNorm, PLIF forward, PLIF surrogate backward, and selective-scan
parity while keeping promotion blocked on generation coherence plus
block-sparse expert dispatch and sampled-vocab cross-entropy evidence.

The fifth LM-head kernel evidence slice covers selected block-sparse expert
dispatch/combine for the routed expert layer. `language_expert_dispatch_triton.py`
provides the Triton kernels, PyTorch fallback, runtime-use counters, and
no-grad CUDA integration for large enough `[token,state_dim]` dispatch batches.
The 2026-07-04 report
`reports/language_kernel_evidence/expert-dispatch-triton-20260704.json` passed
three CUDA `float32` shape sweeps for `language_block_sparse_expert_dispatch`
with geometric microbenchmark speedup `4.389x`; `float16` dispatch is marked
unsupported until parity is proven. The paired suite report
`reports/language_benchmark_suite/language-suite-expert-dispatch-kernel.json`
records RMSNorm, PLIF forward, PLIF surrogate backward, selective-scan, and
expert-dispatch parity while keeping promotion blocked on generation coherence
plus sampled-vocab cross-entropy evidence.

The sixth LM-head kernel evidence slice covers sampled/adaptive vocabulary
cross entropy. `language_sampled_vocab_ce_triton.py` provides a Triton forward
loss kernel pair, PyTorch fallback, runtime-use counters, and forced
parity/benchmark execution for CUDA `float32` hidden rows against selected
vocabulary IDs that include all target tokens. The 2026-07-04 report
`reports/language_kernel_evidence/sampled-vocab-ce-triton-20260704.json`
passed three CUDA `float32` shape sweeps for
`language_sampled_vocab_cross_entropy` with total vocab `8192`, sampled vocab
`1024`, and geometric microbenchmark speedup `1.047x`; `float16` sampled-vocab
CE is marked unsupported until numerical parity is proven. The paired suite
report `reports/language_benchmark_suite/language-suite-sampled-vocab-kernel.json`
records GPU kernel correctness as `pass` across RMSNorm, PLIF forward,
PLIF surrogate backward, selective-scan, expert-dispatch, and sampled-vocab CE,
while keeping promotion blocked on generation coherence review. This is
forward-loss parity evidence; dense gradient training still uses the existing
full-vocab `F.cross_entropy` path until sampled/adaptive vocab backward or
training-impact evidence exists.

## Evaluation Gates

Language model gates:

- next-token train loss;
- heldout loss;
- heldout perplexity;
- generation packet with no external LLM;
- generation coherence review;
- grounding/support report where applicable;
- checkpoint save/restore fidelity;
- replay old/new retention;
- forgetting delta;
- online learning improvement;
- active columns and active parameters per token;
- spike-rate health;
- kernel parity;
- service contract read-only checks.

Runtime gates:

- 8192 tokens: diagnostic boundary;
- 131072 tokens: normal long-run promotion gate;
- 524288 tokens: house-scale target;
- larger runs when stable.

Every sustained run writes final or partial JSON for success, timeout,
exception, interrupt, or manual stop. Required fields include target tokens,
token delta, elapsed seconds, tokens/sec, checkpoint path, runtime owner, active
language path, tick tokens, quantum tokens, last trace, device/backend, CUDA
graph/native/burst/sequence failures, fallback counts, active columns, total
columns, active parameters per token, spike rate, replay/consolidation events,
growth/prune proposals, environment contention, and report status.

`language_sustained_runtime_evidence.py` is the first component-level LM-head
runner for this contract. It must keep short runs marked as smoke/debug only and
must keep `promotes_hot_path=false` until a Triton/CUDA language hot path has
parity, fallback, and complete-runtime impact evidence. The current CUDA graph
burst path is valid sustained execution evidence and removes the per-token
Python launch bottleneck for fixed-shape checkpoint streaming, but it is not a
Triton block-sparse expert kernel and does not prove generated language quality
by itself.

The batched training `forward` path now precomputes token-independent
state-block projections across `[batch,time]` before the causal recurrent loop.
This preserves streaming `step` parity while cutting the measured CUDA training
shape (`batch=16`, `seq=64`, `state_dim=128`) from `823.405 ms` to
`443.763 ms` per full optimizer step in the local stage profile. The
`cuda-vectorized-state-8192.json` training report reached `2293.991 train
tokens/sec` for `63744` train tokens and the paired `524288` sustained report
reached `7264.683 tokens/sec`. This is a PyTorch/CUDA projection-vectorization
speed slice; PLIF forward, `float32` PLIF backward, and standalone selective
scan are now covered by separate parity evidence. Later expert-dispatch
evidence closes `float32` selected-expert dispatch parity, while full
state-block scan fusion, half-precision expert/vocab coverage, and
complete-runtime training impact remain promotion blockers.

The training experiment runner now also defers scalar loss and gradient-norm
metric readback out of the per-batch optimizer hot loop. It synchronizes CUDA
before starting and stopping the training timer, so elapsed time includes the
real GPU work without per-batch scalar stalls. The 2026-07-04
`cuda-deferred-metrics-8192.json` report improved the same PLIF-surrogate
training shape from `2596.380` to `2720.929 train tokens/sec` (`1.048x`) while
preserving Triton PLIF backward use. The paired
`cuda-deferred-metrics-524288-sustained.json` report reached `524288` tokens at
`7502.156 tokens/sec`, within the current PLIF-surrogate house-scale band but
not an inference-speed promotion.

## Scale Ladder

The implementation must support tiny test fixtures without becoming a toy-only
architecture. CI may run small shapes, but every contract above must map to the
scale ladder.

| Ladder | Purpose | Required evidence |
| --- | --- | --- |
| small fixture | CI correctness | deterministic encode, loss, checkpoint, rollback |
| 140M-class | first meaningful sparse LM comparison | loss/perplexity, active compute, replay retention |
| 500M-class | growth/routing stability | active columns, route saturation, memory footprint |
| 0.9B-class | NeuronSpark-scale comparison class | long-run stability, kernel coverage, restore fidelity |
| 2B+ research | larger recurrent sparse research | memory budget, throughput, generation review |

`language_scale_ladder.py` now defines these target classes for the LM head and
writes an evidence inventory with analytic total-parameter, active-parameter,
routed-column, dense vocab-head, and memory estimates. The small fixture can run
heldout loss and owned generation checks in CI. The 140M/500M/0.9B/2B+ entries
remain `configuration_defined_not_trained` until train-token, long-run,
forgetting, restore, kernel, and generation-review evidence exists.

`language_runtime_benchmark_suite.py` aggregates the LM-head evidence categories
into one JSON plus README report. The suite can run tiny fixtures for next-token
loss, heldout perplexity, generation smoke, grounding-support source-term
coverage, continual learning, forgetting, replay recovery, structural
transaction safety, sustained-runtime smoke, active compute, checkpoint restore,
rollback, service-read contract, and scale-ladder inventory. It must keep
grounded generation review, Triton/CUDA kernel parity, and true
8192/131072-token long-run gates visible as explicit evidence rather than
promoting the smoke report. Existing `marulho_language_sustained_runtime_evidence.v1`
JSON reports can be passed into the suite to satisfy the long-run throughput
category only when they are final MARULHO-owned LM reports that reach the
diagnostic and long-gate token counts. Existing
`marulho_language_generation_coherence_report.v1` reports can satisfy
generation coherence only when a grounded prompt suite passes and still leaves
broad quality/runtime promotion false. Existing
`marulho_language_quality_replay_experiment.v1` reports can enrich generation
coherence when selected-child replay is MARULHO-owned, parent-preserving,
heldout-protective, rollback-backed, and paired with same-child sustained
evidence; `language_quality_replay_experiment.py` can now self-ingest its own
final report into a requested benchmark suite and forward optional memory-slot,
structural-plasticity, and GPU-kernel evidence paths. The structural safety
category now exercises expert-spawn growth, explicit expert-prune, explicit
expert-merge, and explicit expert-deep-sleep checkpoint transactions.

`language-suite-rmsnorm-kernel.json` ingests both the RMSNorm kernel report and
the updated sustained LM reports. It records `long_run_throughput=pass`,
`rmsnorm_triton_parity=true`, and keeps promotion blocked on generation
coherence plus the then-remaining PLIF, selective-scan, block-sparse expert, and
sampled-vocab kernel parity evidence.

`language-suite-vectorized-state.json` records the same blocker posture after
the vectorized state-block training slice: long-run throughput remains passing,
RMSNorm parity remains covered, and generation coherence plus the then-open
PLIF/selective-scan/expert/vocab kernel parity remain open.

`language-suite-plif-forward-kernel.json` ingests the RMSNorm and PLIF-forward
kernel reports with the current sustained LM reports. It records
`long_run_throughput=pass`, `rmsnorm_triton_parity=true`, and
`plif_triton_forward_parity=true`, while keeping promotion blocked on
generation coherence plus PLIF backward surrogate, selective-scan,
block-sparse expert, and sampled-vocab kernel evidence.

`language-suite-plif-surrogate-impact.json` ingests RMSNorm, PLIF-forward, and
PLIF-surrogate-backward kernel reports with the new PLIF-surrogate sustained
reports. It records `long_run_throughput=pass`,
`plif_triton_backward_surrogate_parity=true`, and the 524288-token house-scale
LM report at `7578.052 tokens/sec`, while keeping promotion blocked on
generation coherence plus selective-scan, block-sparse expert, and sampled-vocab
kernel evidence.

`language-suite-selective-scan-kernel.json` ingests RMSNorm, PLIF-forward,
PLIF-surrogate-backward, and selective-scan kernel reports with the current
PLIF-surrogate sustained reports. It records `long_run_throughput=pass`,
`selective_scan_triton_parity=true`, and the 524288-token house-scale LM report
at `7578.052 tokens/sec`, while keeping promotion blocked on generation
coherence plus block-sparse expert dispatch and sampled-vocab kernel evidence.

`language-suite-expert-dispatch-kernel.json` ingests RMSNorm, PLIF-forward,
PLIF-surrogate-backward, selective-scan, and expert-dispatch kernel reports with
the PLIF-surrogate sustained reports. It records
`long_run_throughput=pass`, `block_sparse_expert_dispatch_parity=true`, and the
524288-token house-scale LM report at `7578.052 tokens/sec`; this older suite
snapshot kept promotion blocked on generation coherence plus then-missing
sampled-vocab kernel evidence.

`language-suite-sampled-vocab-kernel.json` ingests RMSNorm, PLIF-forward,
PLIF-surrogate-backward, selective-scan, expert-dispatch, and sampled-vocab CE
kernel reports with the current PLIF-surrogate sustained reports. It records
`long_run_throughput=pass`, `gpu_kernel_correctness=pass`,
`sampled_vocab_cross_entropy_parity=true`, and the 524288-token house-scale LM
report at `7578.052 tokens/sec`, while keeping promotion blocked on generation
coherence review.

`language-suite-generation-coherence.json` ingests the current PLIF-surrogate
long-run reports, all six LM-head kernel reports, and
`plif-surrogate-grounded-prompt-suite-20260704.json`. It records
`generation_coherence=pass`, `gpu_kernel_correctness=pass`,
`long_run_throughput=pass`, `missing_category_count=0`, and suite status
`ready_for_review`, while keeping `promotes_runtime_claim=false`.

`language-suite-memory-slot-longtrain-quality-speed-20260705.json` is the
current same-checkpoint suite for the bounded memory-slot LM shape. It ingests
the longtrain memory-slot checkpoint's grounded prompt-suite report, controlled
`8192`/`131072`/`524288` sustained reports, memory-slot runtime and
architecture-cost reports, structural-plasticity evidence, and all current
LM-head kernel reports. It records `generation_coherence=pass`,
`long_run_throughput=pass`, `gpu_kernel_correctness=pass`,
`memory_slot_architecture_cost=pass`, `missing_category_count=0`, and suite
status `ready_for_review`, while keeping `promotes_runtime_claim=false` because
human review and sustained one-token Triton promotion are still open.

Current 2026-07-03 LM component reports from
`reports/language_training_experiments/cuda-exp-8192-checkpoint.pt` reached the
diagnostic, long, and house-scale sustained targets on `cuda:0` with
`torch_cuda_graph_burst`: `8192` tokens at `4853.244 tokens/sec`, `131072`
tokens at `6898.430 tokens/sec`, and `524288` tokens at `6978.602 tokens/sec`.
The suite accepts long-run throughput from these reports in older snapshots;
the latest suite also ingests grounded prompt-suite and Triton/kernel evidence
before reaching `ready_for_review`.

Do not claim frontier competitiveness from parameter count alone. Report active
compute/token, throughput, memory footprint, heldout loss/perplexity, forgetting,
restore fidelity, and generation quality.

## Iteration Lock

Iteration 1 is complete when this document exists, is linked from maintained
docs, and validation proves the Markdown/doc navigation is clean.

Iteration 2 starts the implementation of the MARULHO-owned next-token language
core. The first implementation slice should create the tokenizer adapter,
checkpointed token embeddings, LM head, train/eval split loader, next-token
loss, generation evidence packet, and focused tests without weakening the
existing sustained runtime evidence gates.
