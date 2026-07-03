# MARULHO Autonomous Continual Language Runtime

Last reviewed: 2026-07-03

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
telemetry. This is not CUDA/Triton promotion; kernel parity, fallback, and
complete-runtime impact reports remain required.

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
health, fallback counts, environment contention, and promotion gates. The runner
is component evidence for the LM head; it does not promote the PyTorch path as a
Triton/CUDA hot path or replace the full `MarulhoBrain` sustained runtime gate.

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
| 1 | PLIF/adaptive-LIF forward | `core`/`training` | parity, spike-rate telemetry, complete tick impact |
| 2 | PLIF/adaptive-LIF backward | `core`/`training` | surrogate parity, finite gradients, loss impact |
| 3 | selective recurrent state scan | `training` | cache correctness, streaming restore, long-context impact |
| 4 | route/vote top-k selection | `core`/`retrieval` | bounded rows, no all-column scan, route latency |
| 5 | block-sparse column transition | `core`/`training` | active params/token, dense fallback truth |
| 6 | expert dispatch/combine | `training` | load balance, sparse dispatch parity |
| 7 | fused RMSNorm/residual/membrane centering | `core` | numerical parity, stability impact |
| 8 | eligibility trace update | `core`/`training` | local plasticity parity, no hidden replay |
| 9 | replay gather/scatter | `consolidation`/`training` | bounded replay placement, memory footprint |
| 10 | sampled/adaptive vocab cross entropy | `training` | loss parity, vocab sweep, large-vocab throughput |
| 11 | fused optimizer step | `training` | optimizer parity, checkpoint restore fidelity |

Each kernel must have PyTorch fallback, shape sweep, dtype coverage where
supported, deterministic mode where possible, failure-before-mutation behavior,
benchmark against baseline, and complete-runtime impact report.

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
parity, fallback, and complete-runtime impact evidence.

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
loss, heldout perplexity, generation smoke, continual learning, forgetting,
replay recovery, structural transaction safety, sustained-runtime smoke, active
compute, checkpoint restore, rollback, service-read contract, and scale-ladder
inventory. It must keep missing grounding support, human/grounded generation
review, Triton/CUDA kernel parity, and true 8192/131072-token long-run gates
visible as blockers rather than promoting the smoke report. The structural
safety category now exercises expert-spawn growth, explicit expert-prune,
explicit expert-merge, and explicit expert-deep-sleep checkpoint transactions.

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
