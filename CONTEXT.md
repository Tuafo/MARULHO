# MARULHO Domain Language

This file is the current vocabulary and decision source of truth. Package-local
`README.md` files describe the machinery that owns each concept. Historical
reports are evidence, not current architecture.

## Project Claim

MARULHO is a local continual-language-system research project. It investigates
whether a MARULHO-owned language cortex, adaptive memory, grounded experience,
and checkpointed online learning can produce a system that is more useful per
local compute budget than a conventional static model.

Current evidence supports a small causal Transformer as the language base.
Current evidence supports coherent but imperfect general-language generation
and one narrow continual-learning result with bounded old-language loss. It does
not support frontier capability, general continual learning, an admitted durable
memory interface, or a superior scaling law.

The active research bet no longer treats the Transformer as an architecture to
replace on principle. It is MARULHO's qualified causal-compute substrate until a
matched candidate beats it. Near-term novelty moves to a self-improving system
around that substrate: curriculum choice, verified synthetic experience,
temporary adaptation, durable continual learning, memory, and eventually
self-proposed architecture changes. None of those mechanisms is admitted before
the base model passes unseen-language quality, and every self-change remains
checkpointed, reversible, and benchmarked against the unchanged substrate.

## Runtime Owners

**MarulhoBrain** — owns the installed language model, tokenizer, generation,
source/tick lifecycle, replay/growth hooks for the separate grounded runtime,
compact traces, and durable checkpoint state.

_Avoid_: generation or durable neural mutation owned by FastAPI, status
projections, the UI, an external LLM, ThoughtLoop, or Cortex.

**Brain Language Runtime** — the adapter that installs one matching Transformer
and tokenizer inside `MarulhoBrain`. Its active path is
`marulho_transformer`. It can generate, serialize, restore, and run bounded
sustained generation.

_Avoid_: compatibility loading for retired recurrent checkpoints, mismatched
tokenizer/model vocabularies, or pretending planned continual memory exists.
MARULHO is pre-user research: no external compatibility path is retained merely
in case somebody might depend on it. Retired machinery is deleted after its
evidence and decision are durable.

**Brain Service Adapter** — the `/brain/*` HTTP and UI adapter. It calls
`MarulhoBrain` and exposes read-only evidence projections. It does not train,
route, replay, select memory, or mutate model state during status reads.

**BrainTrace** — compact runtime telemetry. A trace can show what executed; it
does not prove intelligence or quality.

## Active Language Architecture

**MARULHO Transformer** — the only maintained language state core. It is a
decoder-only causal Transformer implemented in
`src/marulho/training/language_transformer.py`, with:

- RMS normalization;
- rotary positional encoding;
- causal scaled-dot-product attention;
- SwiGLU feed-forward blocks;
- bounded per-layer streaming KV state;
- full-vocabulary logits;
- no external model weights.

The model wrapper and checkpoint contract live in
`src/marulho/training/language_model.py`.

**Checkpoint-Owned Tokenizer** — either the byte tokenizer for small tests or a
BPE tokenizer trained on the selected corpus. The complete vocabulary state and
hash are stored with the checkpoint. Production experiments use BPE.

**Transformer Language Checkpoint v2** — an atomic payload containing the exact
model configuration, model tensors, tokenizer state, tokenizer hash, and
metadata. Legacy recurrent/SNN language checkpoints are intentionally rejected.
Scaling checkpoints additionally persist optimizer/scaler state, cumulative
update counts, RNG state, and batch position in metadata so the maintained
runner can continue one MARULHO-owned arm without rebuilding its tokenizer.
Older v2 checkpoints that predate this state may continue with a fresh
optimizer only when the evidence report says so explicitly.

**Full-Vocabulary Next-Token Learning** — standard causal cross-entropy over the
checkpoint vocabulary. Sampled or padded vocabulary shortcuts are retired until
a matched quality experiment justifies a replacement.

## Quality and Scale

**Base-Language Qualification** — the first promotion boundary. A checkpoint
must show:

- improving heldout loss on a genuinely heldout split;
- coherent multi-sentence continuations on unseen prompts;
- no hidden external model;
- checkpoint save/restore fidelity;
- reproducible configuration and corpus provenance.

Throughput and isolated prompt matches are diagnostic evidence, not substitutes
for this boundary.

**Matched Architecture Experiment** — candidates use the same corpus,
tokenizer, split, model-shape intent, optimizer, token budgets, prompts, and
seed. Parameter counts and observed throughput are reported, but the branch is
selected primarily by heldout quality and unseen generation.

**Local Scaling Law** — a fitted relationship, not a slogan. The initial model
is:

`L(N,D) = E + A/N^alpha + B/D^beta`

where `L` is heldout next-token loss, `N` is non-embedding model parameters,
and `D` is unique or explicitly repeated training tokens. Estimation requires
multiple model sizes, token budgets, and seeds. A two-point loss curve at one
model size is not a scaling law.

The first intended grid is roughly 5M, 20M, and the largest 60-100M-class model
that fits the RTX 3060, each measured at several data/compute budgets. The
objective is to find the local compute-optimal region and a falsifiable
projection before renting larger hardware.

**Frontier Comparison** — a resource-normalized comparison against a strong
conventional baseline at the same task. MARULHO does not claim frontier quality
because a tiny model has high tokens per second. A meaningful win would be, for
example, better retained adaptation or long-context recall than a larger static
baseline under the same VRAM, wall-clock, and data budget.

## Adaptive Architecture Status

These concepts are hypotheses and must not appear as implemented capabilities.

**Muon Training Geometry v29 (active falsifier, uninstalled)** — V29 asks
whether MARULHO's blocker is partly the way weights learn rather than the shape
of the model. All four arms use the exact 20,976,128-parameter Transformer,
initial weights, tokenizer, batches, and evaluation. AdamW and Muon each run at
the historical 3e-4 peak learning rate and the official-reference 1e-3 rate.
Muon applies momentum plus five-step Newton-Schulz orthogonalization to the
16,777,216 hidden-matrix parameters and AdamW to the tied embedding and norms.
The grouped CUDA implementation passes all-gradient and compiled-model parity
truth. At 16,777,728 identical tokens, best Muon/AdamW is the 1e-3 pair:
heldout loss/perplexity 4.0961/60.10 versus 4.2606/70.85 and exact free relation
17.58% versus 5.47%. The gains are 0.1645 loss and 12.11 percentage points,
above the joint 0.01/2-point gate. Muon uses 96.0 versus 160.0 MiB optimizer
state and trains at 55.8k versus 96.3k tokens/s. The 3e-4 pair is disjoint:
Muon improves loss by 0.0198 but loses 11.33 generation points. Decision:
`advance_v29_muon_to_unseen_generation`. No checkpoint, optimizer installation,
or broad capability claim follows from that gate alone. The independent
reproduction reaches loss 4.0955 and 26.95% exact free generation, then saves a
strict 100.9 MB checkpoint whose tensors, tied weights, tokenizer, config, and
sample logits reload bit-exactly. Unseen FineWeb-Edu/Cosmopedia source suites
remain 0/4 and 0/4; controlled Cosmopedia raises distinct-bigram fraction to
0.976 but direct text review still finds generic templates, repetition, factual
confusion, and semantic drift. V29 is retained as promising training geometry,
not a qualified base model or installed optimizer. V30 must test general-first
training and longer context before another relation-heavy or memory phase.

**General-First Context v30 (selected baseline, uninstalled)** — V30 holds the
20,976,128-parameter Transformer, Muon 1e-3 recipe, tokenizer, initial tensors,
general-source byte ranges, 16,777,728 processed tokens, and 7,282 optimizer
steps fixed. It compares context 72/batch 32 with context 256/batch 9; both
process exactly 2,304 tokens per step and receive zero synthetic relation
updates. The strict V29 checkpoint supplies the common context-72 general
holdout baseline. Relation cases are metrics-only diagnostics and cannot block
base-language selection. The CUDA preflight passes exact initialization,
complete gradients, and compiled/eager parity at 0.000055/0.000195 while the
long arm peaks near 0.51 GB. A candidate needs at least 0.05 common-heldout loss
gain; context 256 must beat a qualifying context-72 arm by 0.02 to justify its
extra attention cost. General72 wins: common heldout loss/perplexity is
4.0093/55.11 versus V29's 4.0955/60.07 and general256's 4.0258/56.03. Both
relation free scores are 0%, confirming that the synthetic skill is not learned
without its data. The selected checkpoint reloads bit-exactly. FineWeb-Edu/
Cosmopedia source loss improves from 4.5952/3.8875 to 4.4801/3.8488, but all
eight source cases still fail and direct text remains generic, repetitive, and
factually unstable. Decision:
`retain_v30_general72_scale_unique_general_data_before_redesign`. V31 scales a
fresh model to about 67M unique scheduled tokens using a 256 MiB, 16-range
sample from each replay shard. Both byte ranges and selected token windows must
span each full source, and no prepared batch may repeat. No base quality,
runtime, memory, or continual-learning claim exists.

**General Scaling v31 (strongest base checkpoint, uninstalled)** — V31 keeps
V30's 20,976,128 parameters, context 72, Muon 1e-3 recipe, tokenizer, initial
tensors, and holdout while increasing fresh general-only training to 67,110,912
tokens. All 29,128 batches are unique within the run and span both sampled
sources. Heldout loss/perplexity improves from V30's 4.0093/55.11 to
3.6291/37.68, a 0.3802 loss gain that exceeds the preregistered 0.15 gate.
Training sustains 56.1k tokens/s, peaks at 593.6 MiB CUDA allocation, and gives
every parameter a final gradient. The 100,933,202-byte checkpoint reloads every
tensor and sample logit bit-exactly. Unseen FineWeb-Edu/Cosmopedia source loss
also improves to 4.2053/3.4896, but all eight anchored cases still fail. Direct
text is more locally grammatical yet remains generic, repetitive without decode
controls, and factually or semantically unstable. Controlled decoding restores
0.960 distinct-bigram fraction on Cosmopedia without grounding the answers.
Decision: `retain_v31_scaling_curve_expand_unique_data_not_base_quality`. V31 is
a credible scaling point and current research checkpoint, not an installed
runtime, qualified continual model, or commitment to the Transformer.

**General Scaling v32 (closed; redesign decision)** — V32 keeps the exact
V31 model shape, tokenizer, initial tensors, context 72, Muon 1e-3 recipe, and
common holdout while training a fresh state on 201,323,520 tokens. Five disjoint
FineWeb-Edu/Cosmopedia shards contribute exactly 17,476 unique batches each;
raw byte selections and token windows span every source. The V31 checkpoint is
evaluation-only and does not initialize V32. V32 reaches heldout
loss/perplexity 3.4983/33.06 versus V31's 3.6291/37.68: a real 0.1308 gain, but
below the frozen 0.20 requirement. All parameters receive gradients, parity
passes at 0.000103, and training sustains 56.2k tokens/s, so the miss is not a
broken run. Decision: `stop_v32_general_scaling_no_durable_loss_gain`. No
checkpoint or unseen review is admitted. Fixed 21M data scaling stops here; the
next base experiment changes the computational architecture.

**Retired Editable-State Local-Attention v33** — V33 alternated two bounded
local-attention layers with two continuous editable matrix-state layers. The
state uses separate key-channel decay and value-channel write gates, an exact
parallel diagonal-affine scan for training, and the equivalent recurrent update
for decoding. Its production shape contains exactly 20,976,128 useful
parameters, equal to the Transformer without padding. Focused tests pass
causality, full/recurrent agreement, bounded state, and nonzero gradients for
every tensor; BF16 compiled/eager loss parity also passes. A local cached-graph
Muon preflight admitted the matched screen. At 16,777,728 identical update
tokens, Transformer/V33 loss is 4.0082/4.0056: the 0.0025 candidate gain misses
the frozen 0.02 requirement. V33 trains at 41.1k versus 54.3k tokens/s and uses
1.030 versus 0.733 GB peak CUDA allocation. Parameter equality, bit-exact shared
initialization, complete gradients, compiled/eager parity, source balance, and
zero relation training all pass, so this is a valid negative. Decision:
`retire_v33_editable_state_no_heldout_language_win`. No checkpoint or event
controller survives; the model, runner, tests, and experimental checkpoint
surface are deleted. The compact report retains the evidence.

**Capacity Scaling v34 (likelihood-qualified, generation-blocked)** — repeated 21M mixer
swaps now have enough negative evidence that another tiny exotic core is a poor
use of the 3060. V34 instead asks whether the existing local pipeline can train a
materially stronger continuous semantic cortex. The preregistered model has
100,679,424 parameters (width 768, ten Transformer layers, twelve heads) and uses
the same 8,192-token BPE, context 72, Muon 1e-3 recipe, source-balanced 67.11M
unique-token schedule, and common holdout as V31. V31 is evaluation-only and
does not initialize V34. Advancement requires at least 0.20 lower heldout loss,
complete gradients, compiled/eager parity, source/uniqueness audits, and exact
checkpoint reload before unseen generation. The valid run processes exactly
67,110,912 unique tokens and reaches loss/perplexity 3.3902/29.67 versus V31's
3.6291/37.68, a 0.2389 gain. It sustains 11.14k tokens/s, peaks at 3.319 GB CUDA,
gives every parameter a final gradient, and strict-reloads its 428.1 MB checkpoint
bit-exactly. Decision: `save_v34_capacity_scaling_100m_for_unseen_generation`.

Fresh FineWeb/Cosmopedia review improves source loss to 4.0012/3.2831 and produces
readable multi-sentence English, but remains 0/8 anchored. Controlled decoding
raises Cosmopedia distinct-bigram fraction 0.817 to 0.952 without grounding.
Scientific branch: `retain_v34_capacity_checkpoint_continue_unique_data_not_base_quality`.
This is substrate progress, not a claim that scaling Transformers is MARULHO's
final architecture. The earlier
disposable full-shape CUDA preflight compiles in 52.5 seconds, passes BF16 parity
at 0.00063, trains at 11.3k tokens/s, and peaks at 3.32 GB; its two-step quality
values and report are deleted.

**Capacity Continuation v35 (invalid evidence, no checkpoint)** — V35 strict-loads
V34 and adds 134,219,520 tokens from exactly three hash-pinned shards absent from V34:
FineWeb-Edu train shard 0 and Cosmopedia train shards 1 and 3. Cumulative updates
become 201,330,432 tokens. Muon state is fresh, peak learning rate is reduced to 3e-4,
and the parent model state must match bit-exactly before training. Advancement
requires another 0.15 heldout-loss gain plus the existing parity, gradient,
unique-source, checkpoint, and unseen-generation contracts. The run reaches
loss 3.1654, a diagnostic +0.2248 gain, but preparation contains 19,419 batches
per source while the manifest schedules 19,419/19,418/19,418. Unique indices,
hashes, parity, gradients, and exact tokens pass; full prepared-batch coverage
does not. Decision: `invalid_v35_capacity_continuation_evidence`. No checkpoint
or unseen review is admitted. Report SHA-256 is
`de18b99e21d89fd9741d6c27a4d3c89612b72c00075b82dae6419c9a7b53657f`.

**Capacity Continuation v35r (base-language-qualified, grounding-blocked)** — V35R is a
new immutable manifest rather than a post-hoc relabel. It restarts from the exact
hash-pinned V34 report/checkpoint and consumes all 19,419 batches from each of
the same three sources: 58,257 updates, 134,224,128 new tokens, and 201,335,040
cumulative tokens. Model, initialization, data, Muon 3e-4 recipe, and +0.15 loss
gate are unchanged. The runner rejects CLI changes to the corrected manifest.
The valid rerun reaches loss/perplexity 3.1649/23.69, a 0.2253 gain over the
bit-exact V34 baseline, at 10.65k tokens/s and 3.330 GB peak CUDA allocation.
Every prepared batch is consumed exactly once, every parameter receives a final
gradient, BF16 compiled/eager parity passes, and the 428.1 MB checkpoint reloads
bit-exactly. FineWeb/Cosmopedia unseen source loss improves from V34's
4.0012/3.2831 to 3.8020/2.9282. Repetition-controlled Cosmopedia generation has
0.968 distinct bigrams and produces coherent multi-sentence paragraphs. Exact
source anchoring remains 0/8, so V35R qualifies the continuous base needed to
reopen continual-learning and conditional-compute experiments but does not
qualify memory grounding or runtime installation. Decision:
`save_v35r_capacity_continuation_201m_for_unseen_generation`.

**Consumer-GPU Throughput v36 (batch-256 recipe advances)** — All six arms
restart from the exact V35R tensors and consume the same 2,359,296 ordered
tokens. Batch 32 with whole-QKV Muon reaches heldout loss 3.2455 at 11.08k
tokens/s. Batch 256 with the unchanged 3e-4 learning rate reaches 3.1423 at
25.07k tokens/s: 2.262 times throughput with 0.1033 better loss, 7.72 GiB peak
CUDA allocation, and compiled/eager parity. The 8.5e-4 and 1.2e-3 large-batch
rates are worse; 1.2e-3 fails the loss gate. Per-head Muon passes its separate
batch-32 gate at +7.27% throughput and +0.00238 loss, but adds only 1.76% at
batch 256 while slightly worsening same-rate loss. The frozen fastest-arm
selector names batch-256/per-head/8.5e-4, but the standing quality-first rule
makes batch-256/whole-QKV/3e-4 the recipe for the next durable stage. The raw
artifact remains unchanged and no checkpoint is saved. Report SHA-256 is
`e57ec348e588c073712c6c1a03613a6fc7b3400c205a7fae8e28fcc42f346719`.

**Depth Assembly v37 (retired on bounded runtime)** — The identity-initialized
45-route candidate passes exact batch/cached parity and gradient tests, but its
frozen 16,773,120-token matched run cannot finish within 3,600 seconds. Observed
device allocation reaches 11,744 of 12,288 MiB while the process remains active;
the unchanged V36 recipe previously runs at 25.07k tokens/s and 7.72 GiB. No
terminal arm report, heldout score, or checkpoint is emitted, so V37 establishes
a systems failure but makes no language-quality claim. The full-width history
module, runner, and tests are deleted. Timeout artifact SHA-256 is
`cf7465cdeec25be68bbb75af07096e2ae420d233260aaa1a985496c2cee86442`.

**Continual Replay v38 (retained near-positive, no checkpoint)** — All arms
complete 16,773,120 tokens at about 25.0k tokens/s. The 50/50 replay arm reaches
100% label-safe candidate accuracy and 46.88% strict free-answer accuracy while
improving old-language loss from 3.1649 to 3.1124. It misses the frozen 50% free
gate by 3.12 points, so no checkpoint is saved. Relation-only training reaches
only 37.50% free accuracy and catastrophically regresses general loss to
15.8373; 20/80 replay reaches 37.11% free and loss 3.0769. The 50/50 free result
is 85.94% event order, 76.56% property, 20.31% container, and 4.69% ownership.
Recognition is solved on this holdout, replay bounds forgetting, and the live
blocker is exact answer formation. Decision:
`redesign_v38_relation_objective_no_free_learning`. Report SHA-256 is
`e356bf9a44ccb7fd1986be256c41128c2bb79a086c903d1be7bd110a841cc1d2`.

**Answer Objective v39 (continual-qualified, uneven binding)** — V39 preserves
V38's 50/50 schedule and changes only normalized causal credit on answer spans.
The 4x arm reaches exactly 128/256 strict free answers (50.00%), 98.44% candidate
accuracy, and general loss 3.11336 versus V35R's 3.16492 at 24.86k tokens/s.
Its 58/64 property and 51/64 event-order answers are strong, but container is
15/64 and ownership only 4/64. Open prose remains coherent but small-model
repetition and occasional odd semantics remain. The 100,679,424-parameter state
has 218,108,160 cumulative tokens and reloads exactly from a 428,148,518-byte
checkpoint with tokenizer identity. Decision:
`advance_v39_answer_objective_continual_checkpoint`. Checkpoint/report SHA-256
are `6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d`
and `3b64d702ed2db458587c78316d34fe826138bef8d4d72b8093dc861d11289127`.
The 50.00% value is immutable historical evidence under the old prompt-inclusive
no-repeat policy; V44 below is the authoritative current decode result.

**Sustained Runtime v40 (qualified dense runtime)** — The exact V39 checkpoint
generates 256 independently prompted streams of 2,048 consecutive tokens: all
524,288 aggregate tokens complete on CUDA in 74.84 seconds at 7,005 tokens/s
and 3,165,870,592 bytes peak allocation. Pre/post tensor hashes are identical,
KV and decode-control history remain bounded to 72 tokens, all raw logits are
finite, all output IDs are in vocabulary, and 248/256 full continuations have
distinct hashes. Observed hooks cover all 100,679,424 parameters, ten attention
blocks, and ten MLP blocks. V39 is therefore measured as 100% dense with zero
structural sparsity. Some 2,048-token previews cycle or drift, so V40 qualifies
runtime stability, not long-generation quality or long-context memory. Decision:
`qualify_v40_same_checkpoint_sustained_runtime`. Report SHA-256 is
`4757c0a0f0972fabe1de3e0b742f91a049f166994a9421d141c117a7ddcf2331`.

**Hidden-State Episodic Memory v41 (retired)** — V41 froze every V39 weight and
built 65,536 answer-token hidden keys from 8,192 training-only relation
documents. Disjoint calibration selected top-1, similarity 0.85, and 0.8
interpolation. On the frozen 256 cases, base/true/shuffled strict free accuracy
is 50.00%/51.56%/1.17%. True memory perfects property at 100% but leaves
ownership at 6.25% and lowers container to 20.31%; candidate accuracy remains
98.44%. General gate-off logits are bit-exact and model tensors are unchanged.
The shuffled collapse proves causal logit intervention, but the full datastore
repeats the 4,096-entry smoke result and does not create relational binding.
Full search touched 740,950,016 keys while only one value per active query was
read, so it also earns no sparse-compute claim. Decision:
`retire_v41_hidden_state_memory_no_joint_free_binding_win`. No checkpoint or
implementation survives. Report SHA-256 is
`96a34833e573638b4bcbe06c2fba47b99b709c671a16c47b93089eb9302c0e2a`.

**Retired Role-Contrastive Continual Objective v42** — V42 attempted to keep
V39's 50/50 replay and 4x answer loss while suppressing tokenizer-trie branches
toward wrong entity/container/color/event fillers. The mechanism passed unit
parity and a full-batch gradient preflight, but the exact 32x8 eager pilot ran
for 16,507.6 seconds at sustained full GPU use without persisting one arm
result. It was stopped on execution feasibility; no quality conclusion and no
checkpoint exist. The objective, runner, and tests are deleted. Before another
architecture falsifier, the shared experiment loop now gates projected wall
time from complete warmup optimizer steps and atomically persists exact-contract
arm results with optional model state. The exact V39 100.68M model then validates
the integrated BF16/Muon gate on RTX 3060: effective batch 224 reaches 19.22k
training tokens/s at 10,490,205,696 bytes peak allocation, restores every weight
exactly, and executes zero counted steps after forced rejection. Batch 256 slows
to 3.83k tokens/s at 11,847,494,144 bytes and is rejected. This is runtime
qualification, not batch-size quality parity. Decision:
`stop_v42_execution_infeasible_no_quality_conclusion`. Report SHA-256 is
`9ecd6e1e4ba8e603624eb15797f9fe4f5a534388e2221401f9537c98286f7808`;
the matched-runtime preflight report SHA-256 is
`284b35710e6b59572459a35ff9d79dd9f3a8b02921fbc7a6e3f4bb3d43884c15`.

**Grounded Prompt-Copy Readout v43 (preimplementation stop)** — V43 asked
whether V39's 98.44% ranked/50.00% free gap could be repaired by pointing from
the output state to token identities already present in the causal prompt. Its
frozen prerequisite fails: only 66.53% of correct-answer BPE tokens occur
anywhere in the prompt against an 85% requirement, event-order reaches only
57.84%, and no complete answer span is present. The answer requires synthesis,
not only copying. Decision:
`stop_v43_prompt_copy_insufficient_answer_token_coverage`. No implementation,
training, or checkpoint exists. Report SHA-256 is
`6b9580d3097d34fbd28b3edc49965ec0851026743ab98fba77fabc95fe9afc70`.

**Generated-Only Decode Controls v44 (promoted)** — the old decoder applied
repetition and no-repeat-3 controls to prompt plus continuation. Relation answers
legitimately reuse source triples, so the evaluator was forbidding correct
tokens. A same-checkpoint causal sweep isolates no-repeat prompt history as the
failure: old default/no-controls/repetition-only/no-repeat-only strict free
accuracy is 50.00%/88.67%/87.11%/51.56%. Decode policy v4 applies controls only
to generated continuation history. The maintained evaluator then reaches 227/256
strict free answers (88.67%) at unchanged 98.44% ranking; container/ownership/
property/event-order is 60.94%/100%/100%/93.75%, and pre/post model hashes are
exact. This corrects evaluation and runtime behavior, not learned weights, and
does not prove general grounding. Decision:
`promote_v44_generated_only_decode_controls_requalify_v39`. Report SHA-256 is
`e413abd919fb25ea546046b76652c7e011666fa0b7c8ecda8e7a454bdb0b2315`.
The old V40 long-generation qualification is no longer current for decode-policy
behavior and must be repeated from the same checkpoint.

**Generated-Only Sustained Runtime v45 (qualified dense runtime)** — the final
generic v4 runtime contract reloads the exact V39 checkpoint and uses V44's
generated-continuation-only decode controls. All 256 streams complete 2,048
tokens each: 524,288/524,288 tokens in 73.1564 seconds at 7,166.67 tokens/s and
3,165,493,760 bytes peak allocation. Checkpoint and pre/post model hashes match;
logits and token IDs are valid; KV/control history stays bounded; 247/256 stream
hashes are unique; observed hooks cover all 100,679,424 parameters. The path is
still 100% dense with no structural sparsity. Preview text remains locally
coherent but drifts and invents facts, so this is not long-generation quality.
Decision: `qualify_same_checkpoint_sustained_runtime`. Report SHA-256 is
`51eefbbd66c8869217c4ca5a53fa1e5006f44887de028c654a1a3995d0572175`.

**Unseen Exact-Continuation Reaudit v46 (retained negative, benchmark
redesign)** — continuation loss no longer inserts a second BOS token or encodes
the entire 37–51 MB remaining source to keep at most one context window. The
evaluator expands a bounded source prefix until its requested BPE prefix is
stable. On the same four FineWeb-Edu and four Cosmopedia prefixes, V35R/V39 loss
is 3.60076/3.64029 and 2.71844/2.64983 respectively: mixed, bounded change rather
than catastrophic forgetting. V39 passes 0/4 FineWeb, 0/4 Cosmopedia greedy,
and 0/4 Cosmopedia controlled cases. Generated-only controls do not repair it.
The suite exposes only a three-word heldout prefix and hides the source document,
so it is exact continuation prediction—not evidence-conditioned grounding.
Decision: `redesign_unseen_grounding_benchmark_keep_exact_continuation_diagnostic`.
Composite report SHA-256 is
`9df4477f806f99f46892ca828e3e1b058588f2a8e6501e5d94ae15d6f43914e2`.
The next benchmark gives MARULHO a visible unseen source passage and compares
intact, question-only, and corrupted-source conditions before any new training.

**Source-Visible Grounding v47 (valid baseline, training admitted)** — a
tokenizer-bound immutable manifest selects 64 `rajpurkar/squad` validation rows
through the Hugging Face Dataset Viewer. Every source/question prompt fits 64
V39 BPE tokens; accepted answers occur in the intact source but not the question
or mismatched source. The frozen V39 checkpoint answers 3/64 intact cases (4.69%)
and 0/64 question-only and mismatched-source controls. Model hashes are exact and
all validity checks pass. The causal source gain is real but misses the frozen
5-point weak-use threshold by one case and the 25%/+10-point capability gate.
Decision: `v39_no_visible_source_use_train_grounding_with_replay`. Manifest/
report SHA-256 are
`9b3392f137a2ca467bc329815810581a98169da170f74f50e8ccb41cb06e12d6`
and `5a4d36afec1f20f8bf777e7f5eaef35e171e07c2e238bbd7001e028113477b71`.
The official SQuAD training split is now admitted only under matched replay and
heldout validation; no external model participates.

**Continual Source Grounding v48 (objective-only repair retired)** — V48 uses a
hash-pinned 512-case SQuAD training corpus and the immutable V47 validation
manifest. Two exact-reset arms process 4,193,280 tokens each with identical 50%
SQuAD, 16.67% relation replay, and 33.33% general replay schedules. Ordinary
causal loss reaches 9/64 (14.06%) intact answers; normalized 4x answer loss
reaches 14/64 (21.88%). Their stronger-control gains are 12.50 and 20.31 points,
and answer weighting beats ordinary by 7.81 points. The useful learning signal
does not survive the joint gate: V39's stratified relation panel falls from
89.06% to 40.62%/43.75%, while matched general loss rises from 3.13964 to
3.24415/3.24195. The 4x arm also misses the 25% grounding floor by two cases.
Decision: `retire_v48_objective_only_grounding_repair`. No candidate checkpoint
or temporary arm state survives. A failed physical batch-224 preflight executed
zero counted steps; true batch-8 gradient accumulation completes at 5.43k/5.36k
tokens/s without memory paging. The next falsifier freezes V39 and tests a small
conditionally activated residual plasticity module, with bit-exact base behavior
when inactive. Report SHA-256 is
`834e1bce825675f0c18cac77c39e30b8403fcb5368e3937b9c91a46b5b9fb968`.

**Conditional Residual Sidecar v49 (retired)** — V49 freezes every V39 tensor
and adds one explicit-condition causal Transformer block after the final base
normalization. The 4,130,304-parameter sidecar is 4.10% of the parent and sees
exactly 2,096,640 SQuAD tokens, matching V48's new-domain exposure without
replay. All sidecar tensors receive gradients; loss falls through the run; the
130 optimizer steps finish in 37.81 seconds at 55.45k tokens/s and 2.20 GiB
measured peak allocation. Structural isolation works: inactive parent hashes and
sample logits are bit-exact, general loss remains exactly 3.149025917, relation
ranking/free recall remains 98.44%/89.06%, and adapter KV state is bounded.
Capability fails decisively: active intact/question-only/mismatched grounding is
1/64, 0/64, and 0/64, worse than V39 and V48. Decision:
`retire_v49_final_sidecar_insufficient_grounding`. A final-layer sidecar cannot
recover source information that the frozen cortex did not expose in a useful
form. No candidate checkpoint, model, loader, runner, test, or compatibility
surface survives. Report SHA-256 is
`204bbd170158834017fe5b52c0874491a02112c257ca912586fecc77d3aef7a1`.

**Hierarchical Conditional LoRA v50 (retired)** — V50 freezes V39 and installs
rank-16 conditional deltas on all ten layers' attention QKV/output and SwiGLU
gate-up/down projections. The 2,457,600 added parameters are 2.44% of V39; every
delta tensor receives a nonzero final gradient and answer-weighted loss falls
from about 3.55 to 2.76 across the same 2,096,640 SQuAD tokens as V49. Training
finishes in 88.76 seconds at 23.62k tokens/s and 8,967,276,544 bytes measured
peak allocation. Isolation again works exactly: inactive parent state, sample
logits, 3.149025917 general loss, 98.44% relation ranking, and 89.06% free recall
are unchanged. Active intact/question-only/mismatched grounding is 5/64, 0/64,
and 0/64. This improves on V49's final sidecar but loses decisively to V48's
14/64 shared-weight arm and misses the +10 source-control gate. Decision:
`retire_v50_hierarchical_lora_insufficient_grounding`. No candidate checkpoint,
model, runner, test, or compatibility surface survives. Report SHA-256 is
`c97ba0505aa06c3976802430851abc8a3f321f110960ac437320a26307d46541`.

**Full Specialist Fork v51 (retired)** — V51 copies all 100,679,424 V39
parameters to remove adapter capacity as an explanation. Every specialist
parameter receives a final gradient across exactly 2,096,640 SQuAD tokens. True
gradient accumulation finishes the 130 updates in 395.80 seconds at 5.30k
tokens/s and 3,422,686,208 bytes peak allocation; an earlier concatenated
batch-224 execution was terminated without an artifact after crossing the 12 GB
memory cliff. The completed specialist reaches 12/64 intact answers versus 1/64
question-only and 1/64 mismatched-source controls, a real 17.19-point source
gain, but loses to V48's 14/64 and misses the 18/64 gate. Its training loss falls
to 0.010 while general heldout loss worsens from 3.149025917 to 5.153229237.
Under the executed stream-packed curriculum, adding capacity was not the limiting
variable: the matched objective overfits a narrow corpus without learning robust
source-conditioned extraction. A post-run alignment audit narrows that inference.
Although every encoded training record is at most 73 tokens, global stride-72
packing keeps the complete context, question, and answer together for only
80/512 records; 432 cross a window boundary. V48--V51 remain valid negatives for
their executed pipeline, but do not establish a capacity-independent limit under
correctly aligned supervision. The immutable V39
checkpoint and state remain exact. Decision:
`retire_v51_full_specialist_insufficient_grounding`. No specialist checkpoint,
fork runner, test, or compatibility surface survives. Report SHA-256 is
`9b74355e9c287270e28a6fa5b9c54ad79bd25428983d3c5e61a6bd10ea033fad`.

**Document-Aligned Grounding v52 (capability pass, retention fail)** — V52 isolates the newly
measured packing confound before changing architecture. It creates one causal
window per SQuAD record, retains BOS, the full context/question/answer, and EOS,
right-pads only after EOS, and excludes pad targets from the existing 4x
answer-weighted next-token loss. The parent V39 checkpoint, 4,193,280 processed
tokens, 50% grounding schedule, replay sources, optimizer, seeds, heldout V47
controls, relation panel, and general holdout remain frozen against V48. The
source gate requires at least 18/64 intact answers, at least +10 points over the
stronger control, and at least +5 points over V48's 14/64. Retention still allows
at most five relation points and +0.05 general loss regression. Passing both
saves a candidate for confirmation; source success with retention failure keeps
only the aligned learning contract and advances to isolated copy/span machinery;
source failure deletes the V52 runner and alignment path.

The completed run proves alignment matters. Intact/question-only/mismatched
grounding is 19/64, 0/64, and 0/64: a 29.69-point causal source gain and a
7.81-point improvement over V48. All 512 records are aligned versus 80 under
global stride packing. The 4,193,280-token arm trains every parameter in 802.53
seconds at 5.23k tokens/s with 3.19 GiB peak allocation. Retention still fails:
free relation recall falls from 89.06% to 56.25%, and general loss regresses from
3.139640808 to 3.229799509 (+0.09016). Decision:
`advance_v52_aligned_signal_to_isolated_copy`. No candidate or temporary arm
checkpoint survives. The generic document-aligned batch/loss contract remains
active for V53; the one-off V52 runner and gate tests are deleted. Report SHA-256
is `23ef805fae825cd3bd46dd5a85c1deebc3eaabe38db59a9f3750657b6557e33d`.

**Frozen Source Pointer v53 (retired)** — V53 keeps the exact V39 language
model immutable and trains only a rank-64 source pointer on V52's aligned records.
The pointer attends from frozen final query states to token states located inside
the explicit `Context:` field, scatters that attention into tokenizer-vocabulary
copy probability, and learns a scalar mix with unchanged V39 vocabulary
probability. It receives exactly 2,096,640 aligned SQuAD positions with answer-
only loss and no replay. Added parameters must remain below 0.25%; inactive parent
checkpoint/state/logits/general/relation evidence must be exact. Promotion needs
at least 18/64 intact answers, at least +20 source-control points, no more than one
case below V52, full pointer gradients, bounded runtime, and strict reload. A miss
deletes the complete path and advances to a trainable source encoder or explicit
span supervision.

The terminal run reaches 17/64 intact answers while question-only and
mismatched-source controls remain 0/64, a genuine 26.56-point source gain. It
misses the 18/64 capability floor by one case and regresses two cases from V52,
so the preregistered gate fails. All 99,073 pointer parameters receive gradients;
they are 0.0984% of V39. Training finishes in 185.39 seconds at 11.31k tokens/s
with 605,372,416 bytes peak allocation. Parent checkpoint file, state, logits,
general loss, and relation evidence are exact. Decision:
`retire_v53_frozen_source_pointer_insufficient_grounding`. No candidate, model,
runner, test, loader, or compatibility path survives. Report SHA-256 is
`3af6ebad988b2844d83b91f73fe3f7c22443dab933e5f6fcbf9a1bbf48ae4620`.

**Trainable Source Encoder v54 (retired)** — V54 freezes the exact V39
checkpoint and trains a separate width-128, two-layer, four-head bidirectional
encoder with direct start/end span supervision. All 373,506 added parameters
receive nonzero gradients, loss falls from 3.4005 to 1.5942, and 2,096,640
padded positions train in 12.05 seconds at 173,965 positions/s with 498,480,128
bytes peak CUDA allocation. Parent checkpoint/state/logits/general/relation and
the compact parent-bound reload are exact. The terminal result is only 16/64
intact answers versus 0/64 for both controls, missing the 19/64 capability gate
by three and falling below V53 and V52. Ten failed continuations contain only a
fragment of a valid answer. Across the separate V52, V53, and V54 reports, the
oracle union contains 35 distinct successes while only three cases are shared by
all three; this is a diagnostic of complementary errors, not ensemble accuracy.
Decision: `retire_v54_span_encoder_insufficient_grounding`. No checkpoint,
encoder, runner, tests, loader, or compatibility path survives. Report SHA-256
is `32f2c700c8168c6fdccb4c681afda978f9113645b0686ca44253b81aed04d0e0`;
source-audit SHA-256 is
`3eefb53d6a448fb024a79b0f84469f4ee1f9d198d2cd9876decd19f4e2923f9c`.

**Multi-View Answer Transducer v55 (retired)** — V55 is a replacement,
not a larger V54 span head. The exact V39 checkpoint remains frozen. One view
projects its read-only causal hidden states; a second width-192, two-layer,
six-head bidirectional encoder reads frozen token embeddings plus source/question
types. A learned fusion feeds a two-layer autoregressive pointer decoder that
emits the complete source-token answer sequence followed by EOS. Direct
start/end loss remains only an auxiliary training signal. Deterministic view
dropout trains both single-view paths instead of allowing silent branch collapse.

The corrected V55b immutable training manifest contains 8,192 official SQuAD
train cases from 134 titles, excludes every V48 train and V47 validation case
ID, validates answer length inside its source-token context, and preserves the
fixed 64-case validation/control panel. The first manifest is deleted after its
preflight exposed two nominally eight-token answers occupying nine contextual
BPE tokens; no training used it. V55b contract SHA-256 is
`d58ac8d0b5b8337ab9cf5577991cb3f9ca015d86f44aecb6e9379e6db2fcf395` and
file SHA-256 is
`d51c56125e5e3a31e857b90b00de8def45fdf56b8afc1077a7a9c793dfc508fc`.
Fifteen full batch-64 epochs process exactly 8,847,360 padded source positions.
Added parameters must remain below 2.5% of V39 and total cache-plus-training time
below 1,200 seconds. Promotion requires 32/64 intact answers, at least +45
source-control points, both-view accuracy at least four cases above each trained
single-view inference ablation, complete nonzero gradients, exact parent file/
state/logits/general/relation evidence, and strict parent-bound compact reload.
A miss deletes the model, runner, tests, and checkpoint surface.

The terminal run processes all 8,847,360 positions. All 2,130,819 trainable
parameters receive nonzero gradients; they are 2.116% of V39. Total loss falls
from 4.1729 to 1.1699, ending at pointer loss 0.8435 and auxiliary span loss
1.3055. Frozen causal caching takes 12.15 seconds and 905,969,664 host bytes;
training takes 145.45 seconds at 60,827.8 positions/s, or 56,137.4 positions/s
after cache amortization, with 891,945,472 bytes peak CUDA allocation. Parent
checkpoint/state/logits/general/relation and strict compact reload are exact.

Heldout intact/question-only/mismatched accuracy is 20/64, 0/64, and 0/64.
Causal-only reaches 16/64 and bidirectional-only only 2/64, so the fused organ
passes the preregistered synergy bar by exactly four cases. It nevertheless
misses the 32/64 capability bar by twelve and the +45-point source-gain bar by
nine cases. Fourteen misses overlap answer words, but unconstrained position
decoding often joins noncontiguous BPE pieces into corrupt strings such as
`Carololina`, `William the Conquor`, and `historical divisionsisions`. V55 adds
five successes absent from V48/V52/V53/V54, but the 43/64 oracle union across
separate terminal systems is not a runnable model or learned selection result.
Decision: `retire_v55_multiview_transducer_capability_or_ablation_failure`.
No checkpoint, model, runner, tests, loader, cache, or compatibility path
survives. The next source branch must test longer-context retrieval with token-
safe segment realization rather than another pointer head. Report SHA-256 is
`d1d30b6aec1237277d57be534c7029c66364546b16439ce4ad9830d59bfc6911`;
both/bidirectional/causal source-audit SHA-256 values are
`2e10e523f77a8f446c2599c816d554c554259929938d616778761c9973dfd96d`,
`0922f5e9b82c0bf0e4baae370d255c822c1fc0e2733785493b16dd5a0b3f38c4`, and
`0d37b0493c16fdcb471fd407e4ecb21eb366622e6c8cadfcec9dc95ea16cfdfd`.

**Landmark Evidence Retrofit v56 (retired)** — V56 leaves the repeated
short-context answer-head family. The exact V39 checkpoint and tokenizer remain
frozen. Long sources are encoded as 48-token blocks by V39 itself. A learned
128-wide question/block projection scores frozen block landmarks using only the
question—never the answer—and selects two blocks. A 256-wide, two-layer causal
cross-attention residual then injects the selected frozen block-token states into
the frozen V39 question/answer hidden stream. V39's own tied full-vocabulary head
produces every output token. Source-absent prompts bypass retrieval and the
residual, preserving exact V39 generation.

This is a small MARULHO-owned retrofit informed by RETRO's retrieved-chunk cross-
attention and Landmark Attention's learned block selection. It is not external
RAG: no downloaded retriever, encoder, generator, vector database, or model
weight participates. Training uses the corrected V56c official-train manifest:
8,192 new cases, 170 titles, 96–228 source tokens, 2–5 blocks, and zero overlap
with V48/V55. Contract/file SHA-256 values are
`efd56051f98ea32fad2474e3f9504d33bec6aac4d7e69978378f3c9547d5552d` and
`ebc512f0a1d680ce3c9b0f11b52ed9a86395f035be125c5470d4b326e902a5e3`.
The new 128-case validation panel excludes V47 and all train IDs; its contract/
file SHA-256 values are
`feca4f4088d3452265f2fc35240f7aa45de68dfc856e0be80af7f45a9e470a84` and
`fa609b4c6c381d1d0c347fc3286dc2ed5e35daea4c57da8400b20056f0facbc6`.

Frozen V39 source/query states may be cached in host BF16 memory, but cache time,
bytes, and content hash are reported and no cache is durable. Fifteen exact
batch-32 epochs give 3,840 updates and 20,643,840 padded adapter positions
(72 causal-query plus 96 selected-evidence positions per case). AdamW uses 3e-4,
5% warmup, cosine decay to 0.1, BF16 CUDA, clip 1.0, data seed 56121, and model
seed 56131. Multi-label retrieval loss marks every block overlapping the answer;
gold top-two evidence therefore preserves boundary-spanning answers. Generator
loss covers answer tokens plus EOS, with no answer token entering retriever
inputs. Added parameters must remain below 3% of
V39 and cache plus training below 1,200 seconds.

Implementation preflight made the BPE boundary contract explicit before any
counted training. The retriever encodes only the bare question. The causal
question/answer prefix and the accepted answer are encoded separately, with a
virtual trailing answer-space, so runtime generation begins from exactly the
same token IDs as teacher forcing. This split encoding still fits every frozen
case: prefix plus answer plus EOS is at most 73 IDs, hence at most 72 model
inputs. Standalone accepted validation answers require at most ten generated
tokens, so the frozen evaluation requests twelve tokens rather than inheriting
V55's invalid eight-token pointer limit. This correction changes no manifest,
training position budget, capability threshold, or control.

Promotion requires predicted-retrieval intact accuracy at least 64/128, at least
+45 points over question-only and mismatched-source controls, predicted top-two
block-union answer coverage at least 80%, oracle-evidence accuracy at least
72/128, predicted evidence
within ten cases of oracle, and shuffled-evidence accuracy no more than 8/128.
The old short V47 panel is diagnostic, not a substitute. Parent checkpoint,
state, tokenizer, source-absent logits/general/relation behavior, every trainable
gradient, and strict parent-bound compact reload must remain exact. A miss
deletes the retrofit, runner, tests, cache, and checkpoint surface.

The terminal run is a valid negative. All 2,383,361 retrofit parameters receive
nonzero final gradients (2.367% of V39), and total loss falls from 6.4843 to
2.8132, with final generator/retrieval losses 2.4581/0.3551. Training processes
20,643,840 adapter positions in 268.02 seconds at 77.02k positions/s. Train plus
validation caching takes 64.91 seconds and 4,907,335,680 host bytes; cache plus
training is 332.93 seconds at an amortized 62.01k positions/s, with 1,236,118,528
bytes peak CUDA allocation. Predicted top-two contains a full answer for 91/128
cases (71.09%); top-one contains 55/128 (42.97%).

Capability fails before any durable-memory claim. Predicted top-two, predicted
top-one, oracle, and shuffled evidence produce 0/128 exact answers; question-only
produces 1/128 and mismatched source 0/128. Oracle failures emit fluent-looking
substitutions such as `January 1`, `Airflex`, and `Speedy Bowl 50`, showing that
the frozen residual interface cannot make V39 realize even known-correct
evidence. Parent checkpoint/state/tokenizer/logits/general loss/relation
behavior, every gradient, budget, and strict compact reload are exact. Decision:
`retire_v56_landmark_retrofit_capability_or_retrieval_failure`. No candidate,
model, runner, tests, cache, loader, or checkpoint surface survives. Report
SHA-256 is `7b53df754d275a211412df2c006956901cb7f7a910da26556c4a8a8abfef6e3d`.
The next source branch must test context expansion or recurrent segment memory,
not another frozen answer extractor.

**Retired Native Long-Context v57** —
V57 has a fresh official SQuAD boundary before model work. The train manifest has
8,192 cases from 171 titles and excludes every V48, V55, and V56 train ID. The
256-case validation manifest spans 22 titles and excludes every V47 and V56
validation ID. Full causal sequences use 151–315 tokens and fit a 320-token
window; answer-localized oracle prompts use 31–96 tokens. Both store an exact
trailing-space causal prefix so teacher forcing and generation share BPE IDs.
Train contract/file SHA-256 values are
`fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7` and
`aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133`;
validation values are
`9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c` and
`b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80`.
V57 reconstructs V39 with context 320 and strict-loads the exact same tensors;
rotary positions add no parameter. Before training, all common prefixes up to 72
tokens must be logit-exact. Two exact-reset arms then update all 100,679,424
parameters. `native_full` reads the full 128–278-token source. `oracle_short`
reads the answer-bearing localized source stored in the manifest. Both are padded
to 320 tokens and use identical questions, answers, batch 32, schedule, Muon
3e-4, BF16, answer weight 4, and eager execution. Four grounding epochs give
1,024 grounding updates. A 50/25/25 grounding/general/relation schedule gives
2,048 total optimizer steps and 20,971,520 padded positions per arm; the two-arm
comparison processes 41,943,040 positions.

The oracle arm is a localization control, not a candidate. Promotion requires
oracle-short accuracy at least 128/256, native-full accuracy at least 128/256,
native source gain at least 45 points over the stronger question-only or
mismatched-source control, native no more than 16 cases behind oracle, mismatch
at most 16/256, general heldout loss regression at most 0.10, relation exact-
generation regression at most five points, unchanged parameter count, complete
nonzero final gradients, at most 1,800 counted training seconds per arm, and a
strict exact checkpoint reload. Both arms are also evaluated cross-conditionally.
If oracle passes and native fails, localization/long-range integration is the
blocker and the branch moves to recurrent segment state. If oracle fails, the
base/task objective is inadequate and context length is not blamed. Only a
native joint pass can become the next continual checkpoint.

The real RTX 3060 preflight selects eager batch 32: a full BF16 forward/backward,
clipped Muon/AdamW update reaches 16,129.8 positions/s and 7,150,854,144 peak
CUDA bytes with every gradient nonzero. A larger aggregate probe enters memory
pressure and is terminated. Inductor takes 165.24 seconds to compile, misses the
0.001 loss-parity tolerance at 0.001868, and collapses to 328.5 steady
positions/s, so it is rejected. Preflight report SHA-256 is
`bf4f5a74b3710835085bc152a4c1d0eababdc339066c2372944cde8eef831a5e`.
The implemented runner prepares 256 grounding batches per arm, 2,773 relation
batches, 346/296 general batches, and the exact frozen 1,024/512/256/256 update
schedule with hash
`bc736bbb94434c79d2a1e59d667a751ca3dfd211cc38b0603b46b6bb79037d9d`.
The terminal run is mechanically valid and negative. Oracle-short reaches
122/256 exact answers and native-full reaches 43/256, versus required 128/256
for both. Native gains only 16.80 source-control points and trails oracle by 79
cases. Its oracle cross-view reaches 90/256, proving that localization remains a
real difficulty even after native training. Capability is not the only failure:
general loss regresses from 3.1490 to 3.3712/3.3553 and relation generation from
89.06% to 34.38%/75.00% for oracle/native. Both arms retain exact positions,
steps, parameters, final nonzero gradients, parent/tokenizer fidelity, and
strict tensor/logit reload. Training sustains 16,765.7/17,030.1 positions/s in
1,250.9/1,231.4 seconds. Decision:
`retire_v57_context_exonerated_base_or_objective_failure`. This rejects merely
expanding context and unrestrictedly fine-tuning every base tensor. No candidate,
runner, tests, or checkpoint surface survives; the report SHA-256 is
`fe93519ca693837796c76ba8e1161e68e7f4d210ad31a47341f854f90660cb99`.

V58 is preregistered as a protected capacity ceiling for exact evidence
localization. The immutable V39 causal cortex remains the only source-absent
language path. A separate evidence organ clones its embedding, ten full-width
blocks, and final norm, changes only the clone's attention visibility to
bidirectional over a bounded source/question record, and learns start/end
scorers for one contiguous Unicode-character source span. Token hidden states
plus bounded within-token features select exact character boundaries, removing
V55's malformed noncontiguous output surface without making partial Unicode
bytes legal. Preflight resolves 34 training and 2 validation annotations whose
stored bounded-source offsets do not exactly point to their immutable answer
text; all 256 validation answers then pass the mechanical copy oracle. The
primary arm trains all organ
parameters for eight epochs over the title-disjoint V57 8,192-case manifest:
2,048 batch-32 updates and 20,971,520 padded positions at context 320. It must
reach at least 192/256 exact heldout spans, gain 70 source-control points, keep
mismatched answers at or below 8/256, finish within 1,800 seconds, cover every
final gradient, preserve parent/tokenizer/logit identity, and strict-reload the
organ. Only a capability pass triggers the same-budget random-initialization
control. Failure closes the extractive evidence-organ family rather than
authorizing another compact pointer or span-head sweep.

V58 is terminally negative and deleted. Its 100,686,146 trainable parameters
all receive nonzero final gradients over exactly 2,048 updates and 20,971,520
padded positions. Training takes 871.75 seconds at 24.06k positions/s with
5.39 GB peak allocation and lowers sampled span loss to 1.9638. The 256/256
mechanical oracle and exact V39 checkpoint/tokenizer/state/logit fidelity pass.
Title-disjoint extraction reaches only 20/256 versus 0/256 mismatched, a 7.81-
point source gain against the required 70 points. Sixty-one predictions at
least overlap an answer, but recurrent date/number/`Stadium` shortcuts show that
full capacity did not yield transferable question-conditioned localization.
Decision: `retire_v58_extractive_evidence_organ_capacity_failure`. The random-
initialization control does not run, no checkpoint survives, and the report
SHA-256 is
`761f3f385d0524a880f568f956aaafbf0f520b4124c4a2a525302707229331e2`.
Do not reopen the V53--V58 SQuAD pointer/span family with another size or endpoint
sweep. The next memory hypothesis must learn from source-native write-time
signals while protecting the slow causal cortex.

V59 is preregistered as a source-native write-time learning ceiling. A temporary
full V39 copy resets per heldout document, receives four context-72 epochs of
ordinary next-token AdamW learning on the source text alone, answers the
question-only prompt, and is discarded. The parent is never in the optimizer.
The frozen 64-case panel round-robins across all 22 V57 validation titles; its
ordered case-ID SHA-256 is
`185a9963bd28d53f04d075cc54937e0d6ca75ffc7719ac5979359ca1ee84e94f`.
No-write, mismatched-write, true-write, and diagnostic oracle-short-write arms
use identical questions and V44 decoding. True write needs 16/64 exact and a
12-case margin over both controls; mismatch is capped at 8/64 and oracle-short
needs 24/64. Source loss must improve in at least 90% of true cases, all tensors
must receive gradients, resets and parent fidelity must be exact, and total wall
time is capped at 2,400 seconds. This tests raw gradient-written memory, not a
production TTT layer or a speed claim.

V59 is terminally negative and deleted. True-source writing lowers per-document
next-token loss in 64/64 cases; its 844 full-model updates process 52,012
positions in 91.23 seconds at 570.1 positions/s, all 62 tensors receive final
gradients, all 192 case resets are exact, and total wall time is 388.30 seconds
with 1.13 GB peak CUDA allocation. Parent fidelity is exact. Yet no-write,
mismatched, true, and oracle-short writes all score 0/64 strict answers. True
and oracle each contain an accepted answer inside 5/64 verbose or corrupted
continuations, versus zero for controls, which is source-dependent bias but not
a usable answer. Decision: `retire_v59_naive_source_only_gradient_memory`.
No transient state or checkpoint survives; report SHA-256 is
`388c43f79c10cc306fc12b1f1d7ad245ba42c317e40d18007e11d357d18247f0`.
Raw next-token adaptation is closed; a future write-time learner must be
meta-trained for later readout rather than tuned post hoc.

**Meta-gradient Episodic Matrix v60 (retired)** — Frozen V39 causal source states
and exact next-token embeddings produced eight per-document 16x96 matrices
through one differentiable linear reconstruction write. The 786,449-parameter
slow controller was 0.7811% of V39 and trained only through downstream answer
loss; questions, answers, spans, and labels never entered the write.

The corrected exact rerun completed all 2,048 batch-32 updates and 20,971,520
padded source positions in 372.55 seconds at 56,292 positions/s. Setup plus
training took 383.82 seconds, peak CUDA allocation was 1,272,701,952 bytes, all
six controller tensors received final nonzero gradients, and the tokenizer,
parent state, and common-prefix logits remained exact. Nevertheless, untrained
true, learned zero, shuffled, true, and oracle-short memory each scored 0/256
strict answers; all five also contained zero accepted answers. Capability,
source gain, and oracle gates fail. Decision:
`retire_v60_one_step_linear_meta_gradient_memory`. No checkpoint exists; the
runner, tests, and logs are deleted. Report SHA-256 is
`76becda7f4d4986eb0bfca1056d2dd14f074c4d348bf5cf0f735c6125e9718fb`.
This closes one-step linear fast memory, not meta-learning as a class. The frozen
branch advances to an iterative nonlinear MLP fast learner rather than another
span pointer, raw full-model update, or larger linear matrix.

**Iterative Nonlinear Fast Learner v61 (retired)** — Eight per-document
two-layer MLP heads took two exact source-reconstruction gradient steps while a
1,605,657-parameter slow controller meta-trained through answer loss. The slow
state is 1.5948% of frozen V39; all 12 tensors receive final nonzero gradients,
parent/tokenizer/context-prefix fidelity is exact, and no validation information
enters the write.

All 2,048 batch-32 updates and 20,971,520 padded source positions complete in
416.16 seconds at 50,392 positions/s. Setup plus training takes 428.90 seconds,
peak CUDA allocation is 1,410,242,048 bytes, and final answer loss is 3.3408.
The first final-batch inner step lowers reconstruction loss from 35.71 to 18.28,
but the second diverges to 5,640.99, failing the frozen mechanistic gate.
Untrained true, learned no-write, shuffled, true, and oracle-short memory all
score 0/256 exact. Accepted-answer containment is respectively 1, 1, 0, 0, and
1 of 256, so it is neither usable nor source-selective. Decision:
`retire_v61_final_residual_nonlinear_fast_learner`. No checkpoint exists; the
runner, tests, and logs are deleted. Report SHA-256 is
`12d3cc8b3a1aa14937e68f8323607c9fb1322645b24aec4a3710c8a680b9c358`.
V60 and V61 jointly close linear and nonlinear final-hidden residual fast memory.
The next controlled branch must keep V39 protected while letting memory influence
multiple cortex depths, rather than changing capacity, steps, or source search.

**Protected Three-Depth Shared Memory v62 (retired)** — V62 restored V60's
stable source-only eight-head matrix write and read the same per-document state
before frozen V39 blocks 2, 5, and 8 through site-specific queries and token-
dependent gates. Its 1,001,483 parameters are 0.9947% of V39; all 13 tensors
receive final nonzero gradients, and inactive hidden states, logits, and all 21
streaming-state tensors remain bit-exact before and after training.

All 2,048 updates and 20,971,520 source positions complete in 473.36 seconds at
44,303 positions/s. Setup plus training takes 482.80 seconds, peak CUDA allocation
is 1,460,438,528 bytes, and final answer loss is 3.1527. The three BF16 scalar
site gates remain at 0.1192, but learned write/query/read and token-gate tensors
all receive gradients. Behavior is decisive: inactive, shuffled, true, and
oracle-short memory score 0, 1, 1, and 1 of 256; untrained true is 0. Every view
contains an accepted answer in exactly one continuation, so the isolated exact
matches are not source-conditioned. Decision:
`retire_v62_compressed_three_depth_fast_memory`. No checkpoint exists; runner,
tests, and logs are deleted. Report SHA-256 is
`7742199d52ed13c11cf20816fc4e593500dec7ee99486fd41f44bf416cf5e5b1`.
V60--V62 jointly close compressed matrix memory at final and spaced internal
read sites. The next branch must retain exact source-token KV state or replace
the base computational substrate, not enlarge this interface.

V63 closes exact-token adaptive KV memory and protected V39 adaptation. A
983,040-parameter FP32 controller applied bounded head-specific 64x64 residual
maps only to exact source-token keys and values through all ten frozen V39
blocks. All 25,344 tokenizer-boundary views, zero hidden/logit/all-21-state
parity, immutable-parent checks, finite state, and 240/240 matrix gradients pass.
The title-disjoint 8,192/256 run completes 2,048 updates and 20,971,520 padded
context-320 positions in 751.07 seconds at 27,922 positions/s, with 4.32 GB peak
CUDA allocation. Loss falls to 3.2313, but raw true/oracle score 0/0 and terminal
question-only, shuffled, true, and oracle score 0/0/0/1 of 256. The failure is
behavioral, not mechanical. Decision: `retire_v39_protected_memory_adaptation`;
no checkpoint survives, and the failed runner/tests are deleted. Report SHA-256
is `08baf18c9b203c85fe6a2e8ef1913e31cbf025173be3789e64ef789033cd5e43`.

V60--V63 now close compressed, nonlinear, multi-depth, and exact-token protected
memory adaptation around V39. No extra rank, reader, gate, injection site, or
SQuAD replay sweep is admissible. The next experiment must change the base
language computation or learning objective.

**V64 delta-state cortex (preregistered base replacement)** — V64 starts both
arms from random weights and compares a 100M-class MARULHO delta-state/local-
attention cortex with a fresh 100,679,424-parameter Transformer. It does not
load V39 and does not restore the retired V33 serial delta implementation. The
candidate uses twelve width-640 blocks: nine bounded matrix-state blocks with
independent channel decay, key-side erase, and value-side write, plus three
window-64 local-attention blocks in a repeated 3:1 pattern. Ten heads keep a
64-dimensional per-head state; chunk size is 32 and streaming state is exact.
Its 2,624-wide SwiGLU is chosen to keep the final parameter count within one
percent of the control, not to add an unmatched capacity advantage.

Both arms see the same deterministic 8,192-step, batch-32, context-320 schedule:
6,144 unique general batches split equally across frozen FineWeb-Edu and
Cosmopedia sources, interleaved with 2,048 source-QA batches covering all 8,192
title-disjoint records for eight epochs. That is 83,886,080 padded positions,
75% general and 25% source-QA, with the existing four-times answer emphasis only
on QA batches. A fresh fused AdamW recipe, cosine schedule, tokenizer, exact
batch order, evaluation manifests, and generation policies are shared. The
architecture verdict requires owned sequential/chunk/recurrent agreement,
complete gradients, finite state, strict checkpoint reload, a heldout general
loss no more than 0.02 above the control, at least 64/256 exact true-source
answers, at least 20 cases above the Transformer, and a 51-case gain over the
stronger source-absent/shuffled control,
oracle at least 128/256, coherent unseen prose, at least 70% of control training
throughput, and no more than 11.5 GiB peak allocation. If neither arm reaches
the oracle floor, the training objective is invalid rather than evidence for a
candidate win. Quality outranks speed; no terminal checkpoint survives a failed
joint gate.

V64 terminally stops before language training. Its MARULHO-owned direct Triton
recurrence passes checkpoint/replay backward, stacked-model, BF16-loss, and all-
parameter parity. CUDA Graph then captures the complete two-microbatch fused-
AdamW step in 0.51 seconds and reproduces eager loss, gradient norm, and all
100,202,970 updated parameters exactly. The candidate nevertheless reaches only
9.24k positions/s versus 21.03k for the fresh matched Transformer: 43.93% misses
the frozen 50% preflight floor. Candidate peak allocation is 8.70 GB. Decision:
`stop_v64_for_kernel_redesign_no_quality_verdict`. No language training or
checkpoint started. Four failed Inductor attempts, their cache, and that backend
remain deleted.

**V65 parallel editable-state cortex (current bet)** — V64's terminal result
rejects token-sequential training, not separately controlled state editing.
Gated DeltaNet-2 independently reports that channel-wise decay plus separate
channel-wise erase and write can compete strongly at 1.3B parameters when the
same recurrence is trained through a fused chunkwise WY algorithm. V65 tests
that computational distinction directly. MARULHO will derive an owned
chunk-parallel implementation from the published equations; it will not import
external model weights, use an external cognition service, or copy NVIDIA's
non-commercial implementation. The sequential equation remains an FP64 oracle
and streaming-inference form only.

The first gate is operator truth and speed, not a new language run. An isolated
CUDA kernel must match sequential outputs, final state, and every input gradient
at chunk sizes 16/32/64, avoid Inductor, survive the process-tree watchdog, and
materially beat V64's 182.2k forward/backward positions/s without its 1.033 GB
incremental workspace. Only then may a fresh approximately 100M-parameter stack
alternate two parallel editable-state blocks with one bounded exact-attention
block and face the fresh Transformer at matched tokenizer, data, parameters,
optimizer, context, and effective batch. The 50% preflight floor and 70%
terminal throughput gate remain minimums, not success claims. Language quality,
source use, continual retention, checkpoint fidelity, and sustained runtime
still decide the architecture.

V65 stops at that first gate. The independently derived cumulative-decay change
of coordinates reduces each chunk to one causal triangular solve and dense
matrix products. FP64 recurrent/parallel output, final state, continuation, and
all seven gradients pass at chunk sizes 16/32/64; the owned Triton coordinate
kernel also passes FP32 parity. Chunk 64 is best. The transparent CUDA reference
reaches 240.4k positions/s at 460 MB incremental; one-warp coordinate fusion
reaches 249.7k at 518 MB. Fusing writes, using two/four warps, and CUDA Graph do
not improve it. The best result is 1.37x V64 and uses half its workspace, but is
only 83.24% of the frozen 300k operator floor. Decision:
`stop_v65_stage_a_parallel_kernel_misses_throughput`. No model or language run
exists; all live V65 code is deleted after the compact report is retained.
Report SHA-256 is
`dc141e7d6df1a25f1f238bfe68d37865556ef3c875e3523a84a67a77bddff755`.

**V66 causal micro-macro exchange (stopped)** — V66 proves that compressed
summary attention has a real long-context crossover but rejects its two-local-
pass implementation before model construction. Causality is exact, completed
blocks change later blocks, and every audited gradient is finite and nonzero.
At context 1,024 it reaches 1.678M positions/s versus full attention's 1.574M,
but uses 694 MB versus 505 MB peak allocation. At the active context 320 it
reaches only 1.516M versus 2.729M positions/s, or 55.55% of control, below the
frozen 70% floor. The implementation, runner, and tests are deleted; the compact
report owns the result.

**V67 queried-summary exchange (stopped)** — replacing V66's first full local
pass with four cross-attention queries fixes both speed gates. V67 reaches 73.91%
of full attention throughput at context 320 and 136.82% at context 1,024, with
exact causal isolation and complete gradients. It still peaks at 646 MB versus
505 MB for the long-context control, failing the frozen memory gate. No model is
built and all V67 implementation code is deleted.

**V68 block-native queried exchange (stopped)** — native block layout aliases
storage exactly, removes 84.5 MB from V67's context-1,024 peak, and increases
long-context throughput to 2.451M positions/s or 1.528x full attention. Explicit
copies were therefore real overhead, but not the whole problem: candidate peak
remains 561 MB versus 505 MB, and the frozen context-320 ratio is 0.691 versus
the 0.70 floor. No model is built and all V68 implementation code is deleted.

**V69 macro-conditioned local attention (systems stop, useful mechanism)** —
the completed macro state conditions the next block's query and attention output
without prefix tokens. With hash-identical Q/KV/output weights, exact causality,
and complete gradients, the full block reaches 91.53% of control throughput at
context 320 and 127.25% at context 1,024. It fails only the old requirement to
use less long-context peak memory: 703 MB versus 652 MB (1.078x). V69 is not
retroactively promoted; its transient code is deleted and it has no language-
quality verdict.

**V70 parameter-matched macro cortex (stopped)** — V66--V69 show that the
strict below-FlashAttention memory gate selects against carrying any additional
macro state even when compute is faster. V70 must be freshly preregistered at
the actual model boundary: approximately 100M parameters, at least 70% end-to-
end Transformer throughput, below the RTX 3060's 11.5 GiB ceiling, and superior
held-out/source behavior. This changes the scientific question transparently;
it does not rewrite V69's failed gate. The frozen first quality screen is 512
fresh context-320 batch-32 updates per arm (5,242,880 positions) and requires at
least a 0.02 heldout-loss win before any longer curriculum or checkpoint exists.
Phase 0 now passes: at 100,733,184 versus 100,679,424 parameters, candidate and
control share one exact common-state hash; complete eager-Muon steps reach
18.57k/19.41k positions/s (95.67% retention) and 5.53/5.48 GB peak allocation.
Every gradient is present, finite, and nonzero. The frozen 512-update screen is
nevertheless terminally negative: candidate/control start tied at 9.1875 loss,
then finish at 5.7188/5.5000. V70 is worse at every 128-step milestone while
retaining 94.61% throughput and only 69.4 MB extra peak allocation. The failure
is language learning, not execution. Its model, runners, tests, and partials are
deleted; no checkpoint or runtime surface exists.

**V71 periodic global-reset hierarchy (stopped)** — V70 compressed
every cross-block interaction through four summaries. A successor should retain
mostly fast macro-local layers but periodically run exact full-token attention,
so detailed token relations can be reconstructed instead of being permanently
compressed. V71 freezes `local,local,local,local,global` twice and compares the
macro channel against an otherwise matched local/global hierarchy with no macro
channel plus V70's immutable Transformer control. Macro must beat both by 0.02;
this is a new architecture, not a V70 depth or hyperparameter rescue.
Phase 0 passes: macro/local full steps reach 18.24k/19.32k positions/s, or
99.86%/105.76% of the immutable Transformer, at 5.52/5.48 GB peak. Both reproduce
the exact common hash and complete gradients. The quality result is 5.5000
Transformer, 5.5625 periodic-local, and 5.6875 periodic-macro. Exact resets
recover most of V70's deficit, but macro makes the matched topology 0.125 worse
and 6.17% slower. V66--V71's four-summary path is retired for base language;
periodic-local is a near-control, not a promoted model. All V71 code is deleted.

**V72 persistent cross-segment workspace (retired)** — compact state is causal
but does not improve the joint language system. Stage A1 reaches 100% delayed
recall versus 6.23%--7.15% controls in all three seeds. In the real 100M-parameter
long-document screen, swapping the state worsens later-segment loss by 0.03867,
proving document-specific contents matter; the candidate also retains 96.61% of
Transformer throughput at 5.89 GB peak allocation. Nevertheless, persistent/
Transformer later-segment loss is 5.95039/5.85117: memory regresses language by
0.09922 instead of improving it by 0.02. It is worse on both FineWeb-Edu and
Cosmopedia and even on the first segment. The terminal Transformer gate fires,
so reset/shuffled arms are not spent after they can no longer rescue promotion.
All V72 model, evaluator, and test code is deleted. The installed language path
remains the Transformer.

**Transformer-centered self-improvement (current direction)** — stop replacing
the only language cortex that survives matched scale. Preserve the full
Transformer and improve the system around its demonstrated weaknesses: training
efficiency, continual learning, durable exact context, adaptive compute, and
verification. V59--V63 already reject post-hoc source-time fine-tuning and small
memory retrofits onto a frozen parent; V72 rejects paying for compressed latent
state by narrowing the base model. The next experiment must therefore keep
Transformer capacity exact and make any adaptation conditional or temporary,
with a zero-action path that is bit-exact and equal to the base model.

**V73 exact-cortex adaptive sidecar (retired)** — preserving all Transformer
capacity removes V72's language regression but does not make locally trained
state useful. Disabled V73 is bit-exact to the Transformer; all 101.932M
parameters train, sustained throughput is 20.41k versus 22.20k positions/s, and
peak allocation is 6.068 GB. Persistent/Transformer later loss is
5.85078/5.85117, only 0.00039 better instead of 0.02. FineWeb-Edu and Cosmopedia
losses are exactly unchanged, and swapping document states changes loss by
exactly zero despite nonzero read gates. Reset/shuffled cannot rescue the failed
admission gates and are not run. All V73 machinery is deleted. This closes
locally reconstructed latent sidecars, not jointly meta-trained test-time
learning.

**V74 end-to-end test-time learning (retired with a positive mechanism
signal)** — a standard causal Transformer was meta-trained with a bounded
first-order temporary update to rank-8 weights in its final MLP. Ordinary
next-token gradients from earlier segments produced 68.695% delayed recall,
versus 6.293% when the same gradients were discarded and 6.342% when another
document's gradients were applied. Disabled output is bit-exact, future-token
perturbation leaves earlier losses and updates exact, all required gradients are
finite and nonzero, and the safe run peaked at 1.53 GB allocated. The mechanism
is therefore real and document-specific, but seed 7401 misses the frozen 80%
accuracy gate by 11.305 points. Because every seed had to pass, seeds 7402/7403
and the 100M language stage are not spent. The V74 model, evaluator, and tests
are deleted; only the compact report remains. This rejects the fixed always-
update formulation, not test-time learning. The next admissible experiment must
test whether a learned retention/overwrite rule can preserve useful updates
through interference without changing the Transformer runtime by default.

**V75 adaptive gradient retention (retired)** — causal gradient statistics do
not learn useful selectivity on the V74 mechanism. Adaptive updates reach
42.004% delayed recall, while a fixed acceptance matched to their mean reaches
42.261%; forcing the gate open reaches 36.188%, and discard/wrong-document
controls remain near chance at 6.927%/6.891%. The gate varies only narrowly
across the two pre-query updates (5th--95th percentile 0.240--0.285) and its mean
0.259 acceptance explains the benefit. Mechanics pass exactly, every gradient
is complete and finite, the run peaks at 1.526 GB, and the matched control rules
out hidden selectivity. Seeds 7402/7403 are not spent. All V75 machinery is
deleted. Together V74--V75 show that first-order TTT gradients carry facts, but
neither a fixed update nor a four-statistic retention gate meets the frozen
memory target. The next TTT experiment must test the missing exact meta-gradient,
not tune another gate or inner rate.

**V76 exact end-to-end TTT (retired after real-language falsification)** — exact
meta-gradients nearly solve the synthetic delayed-binding task at 49,147/49,152,
but that result does not transfer to the 100.679M V39 Transformer on disjoint
FineWeb-Edu/Cosmopedia documents. Same-data ordinary continuation improves later
loss from immutable 3.96320 to 2.90234. First-order TTT reaches 2.90913 and exact
TTT 2.90876; exact is 0.00643 worse than static and only 0.00037 better than
first-order. Discarding or shuffling exact updates worsens loss by only 0.00100/
0.00130, far below the causal 0.02 gate. Exact test-time throughput is 36.18% of
the immutable Transformer, also below its 50% Stage-A1 gate. Hashes, schedules,
parent tensors, gradients, BF16 state, and CUDA evidence all pass, so this is a
mechanism failure rather than a broken run. V76 is deleted with no checkpoint or
runtime surface. The result warns that structured synthetic gradient memory can
be a false positive for natural language. The strongest live direction is now
the ordinary Transformer continuation that achieved the best real heldout loss.

**V77 quantitative base (strict, uninstalled)** — the exact safe-batch ordinary
continuation reproduces V76: later-segment loss improves from 3.963203 to 2.902100
after 7,864,320 positions at 20.36k positions/s and 4.24 GB peak allocation. Its
100.679M tensors, tokenizer, configuration, tied weights, metadata, and sample
logits reload bit-exactly from the 402,982,569-byte checkpoint with SHA-256
`3755bfb683b77bbf74811d58b9d3db404cdca4143b82e1f6f427077ea4487074`.
The frozen V46 unseen panels remain 0/12: FineWeb continuation loss improves by
0.06360, Cosmopedia worsens by 0.01847, and visible text is grammatical but
generic, repetitive, and topic-unstable. Decision:
`continue_base_language_training_before_continual_learning`. V77 is the strongest
quantitative base, not an installed runtime or coherence claim. TTT remains
absent. Continual learning and structural plasticity remain closed until a base
checkpoint passes visible unseen-language review.

**V78 quantitative base (strict, uninstalled)** — four-times-larger continuation
on 32,768 new long documents improves V77 later loss 2.902100 to 2.798199 and
FineWeb-Edu/Cosmopedia to 3.157677/2.438721. The 100.679M checkpoint at 257,429,760
cumulative positions reloads bit-exactly; SHA-256 is
`b66753983316b5a0cf61b293d36e4fda9b15929168067a59ed95ef816da4313b`.
The frozen panels nevertheless remain 0/12 with unchanged prefix agreement and
visible template repetition/topic substitution. Decision:
`redesign_base_data_or_objective_after_v78_scale_only_failure`. V78 is retained
as the strongest quantitative base, not installed. More of the same two-source
objective is no longer the active answer; continual learning remains closed.
The original training report's peak-memory field was interval-censored by curve
evaluation resets. The correction retracts that exact value; the selected-path
optimizer-complete preflight proves 4.253 GB under the 8-GiB gate, and future runs
preserve interval maxima.

**V79 DCLM replacement (retired)** — exact 50% source replacement learns DCLM
but forgets the removed Cosmopedia distribution. Candidate versus control DCLM
loss improves by 0.156170 and FineWeb improves slightly, but Cosmopedia regresses
by 0.297005; old-source mean worsens 0.138523 and joint loss is 2.991823 versus
2.951530. All validity checks pass, throughput matches within 1.25%, and both
runs peak at 4.24 GB, so this is a scientific failure rather than an execution
failure. No candidate checkpoint exists and the failed runner is deleted. The
content-addressed DCLM text/tensors remain reusable. This rejects substitution,
not DCLM or a future curriculum that keeps all learned sources represented.

**V80 active scale run** — packed segments gave a real 1.0849x speedup
but missed the frozen 1.10x gate and its BF16 forward-loss delta missed parity;
the temporary runner is deleted and sequential segment training remains. The
capability run retains FineWeb-Edu, Cosmopedia, and DCLM together on one
materially larger continuous schedule with exact reload-boundary training state
and strict numerical post-reload fidelity. It is
not another source-replacement sweep. Full gradient auditing is required on the
first update, not redundantly on every later update.

V80's frozen capability target is now a 1,006,632,960-position continuation from
V78: exact 40% FineWeb-Edu, 30% Cosmopedia, and 30% expanded DCLM, with all
sources deduplicated against holdouts and the worst known templates/encoding
failures removed. The sequential batch-8 layout uses a 256-step warmup, stable
1.5e-4 phase, 20% cooldown, three-source curves every 2,048 updates, and
optimizer snapshots every 1,024 updates, retaining only the latest two.
Quantitative loss gates only admit direct generation review; coherent unseen
multi-sentence text remains mandatory
before continual learning reopens.

The V80 corpus is materialized and independently audited: 58,999 FineWeb-Edu,
150,910 DCLM, and 62,298 Cosmopedia long documents, 272,207 total. Per-source
and holdout tensors strict-reload; exact hash sets have zero within-source,
cross-source, or train/eval collisions. This is 261.3M unique predicted
positions, so the 1.007B schedule averages 3.85 balanced-within-source epochs.
The 4.24 GB recreatable raw parquets are deleted; content-addressed tensors and
selected DCLM text remain.

The exact million-document source/row schedule is also frozen at SHA-256
`a886ef76...fb449fa`. It contains the preregistered 419,430/314,573/314,573
FineWeb/Cosmopedia/DCLM slots and 1,006,632,960 predicted positions. Exposure
histograms are balanced within one for every source, and the 5.25 MB artifact
strict-reloads all IDs, bounds, counts, and first/last anchors.

The V80 resume-fidelity gate is frozen at report SHA-256
`a1ce9559...6c4dac`. Model, optimizer, RNG, counters, and schedule reload exactly;
the canonical next update is exact across 100.679M gradients, 106.971M model
values, and 106.987M optimizer values at 4.260 GB peak allocation. Repeated CUDA
execution also exposed one harmless `9.31e-10` gradient-order difference while
leaving the final model exact, so the maintained contract reports post-compute
hashes but gates them with strict elementwise tolerances. A real stop/resume smoke
then continued updates 2--4 at about 17.0k positions/s and reproduced the frozen
initial three-source loss within `3e-6`; its 428 MB smoke snapshots were deleted.
The production run's first snapshot at update 1,024 is 428,212,885 bytes with
SHA-256 `6f6d11b8...976017`; immediate reload verifies exact model, optimizer,
RNG, counters, schedule offset, and ownership before training continues. This
confirms real recovery behavior but says nothing yet about final language quality.

The first full production curve at update 2,048 is positive on every source.
Overall later loss improves 2.982864 to 2.904556; FineWeb-Edu, Cosmopedia, and
DCLM improve by 0.045616, 0.021927, and 0.167381. This reaches 31.32% of the
required overall gain and 83.69% of the DCLM gain after only 6.25% of the
schedule, without V79's old-source tradeoff. The update-2,048 optimizer snapshot
also strict-reloads (`6de0f377...cb156ad`). These are early curve and recovery
facts, not final checkpoint or coherent-generation claims.

Production snapshot rotation is now observed rather than unit-tested only. The
update-3,072 state strict-reloads at 428,213,141 bytes and SHA-256
`13049e42...aeb3022`; update 1,024 is deleted while updates 2,048 and 3,072 remain,
and training continues. This establishes the bounded two-rollback contract, not
additional model quality.

The update-4,096 curve remains positive after 125,829,120 new positions. Overall
later loss is 2.877130, a 0.105735 gain or 42.29% of the terminal 0.25 target.
FineWeb-Edu/Cosmopedia/DCLM are 3.083427/2.394356/3.153606, improving their own
initial values by 0.074173/0.044449/0.198582; DCLM is 99.29% of its required
0.20 gain without trading away either old source. The 428,213,333-byte snapshot
strict-reloads at SHA-256 `7106aa33...ec7c96`, rotates update 2,048 away, and
becomes a controlled resume boundary. Training advances past update 4,128 under
the terminal-retention code at about 19.74k positions/s. This strengthens the
curve and optimizer-continuity evidence, not the still-unmet terminal or visible
language claims.

The first rollback transaction after that process reconstruction also passes.
Update 5,120 writes and strict-reloads 428,213,397 bytes at SHA-256
`33bc5d4a...ab88e36`, deletes only update 3,072, retains updates 4,096/5,120, and
continues through 5,152 at about 19.74k positions/s. The patched process therefore
preserves the same bounded rolling contract; this adds recovery evidence, not a
new quality point.

The update-6,144 heldout curve continues improving every source after 188,743,680
new positions. Overall later loss is 2.861635, a 0.121230 gain or 48.49% of the
terminal target. FineWeb-Edu/Cosmopedia/DCLM reach
3.069839/2.380089/3.134975, gains of 0.087761/0.058716/0.217213 from the frozen
start. DCLM now clears its complete 0.20 requirement while both retained sources
continue improving. The 428,213,845-byte update-6,144 snapshot strict-reloads at
SHA-256 `21495160...c68eac7`, rotates update 4,096 away, and training advances
through 6,176 at about 19.80k positions/s. The overall gate and visible language
review remain unresolved.

Update 7,168 writes and strict-reloads another 428,213,845-byte optimizer state
at SHA-256 `d1d3cfd6...ef30a0d`, deletes only update 5,120, retains updates
6,144/7,168, and continues through 7,200 at about 19.84k positions/s. Rolling
recovery therefore remains healthy between authoritative curve points.

At update 8,192, after 251,658,240 new positions or 25% of the schedule, overall
later loss reaches 2.850349. The 0.132515 total gain is 53.01% of the terminal
target. FineWeb-Edu/Cosmopedia/DCLM reach 3.058319/2.371162/3.121567, improving
their initial values by 0.099281/0.067642/0.230621. All sources improve again and
DCLM reaches 115.31% of its terminal requirement. The 428,214,037-byte snapshot
strict-reloads at SHA-256 `72281077...c6ac6cb`, rotates update 6,144 away, and
training continues through 8,224 at about 19.88k positions/s. The latest
2,048-update interval adds 0.011285 overall gain; the declining but positive
slope still requires the full frozen schedule and cooldown.

Terminal ownership is split deliberately. If V80 passes its quantitative gate,
the FP32 checkpoint owns generation/runtime while exactly one verified BF16
training snapshot retains model, Muon/Adam state, RNG, tokenizer, and completed
schedule for rollback and genuine optimizer-continuous learning. Older snapshots
are deleted. That terminal state cannot start V81 unless the direct language
review also passes, and it is deletable if visible generation rejects V80.

V80's post-training language audit is also frozen before a candidate exists. It
will refuse an unqualified checkpoint, rerun all twelve V78 FineWeb/Cosmopedia
cases, and add twelve generations over four DCLM holdout documents under greedy,
repetition-controlled, and seeded nucleus decoding. The sampled panel freezes
temperature 0.8, top-p 0.9, repetition penalty 1.05, and seed 80080. Those raw
DCLM documents must encode exactly to their frozen evaluation rows. A readable
direct-review sheet and explicit observation
are mandatory; automatic prefix scores cannot claim coherence. Coherent text
opens continual and grounded self-challenge validation, incoherent text forces
an objective/tokenizer redesign, and invalid provenance rejects the evidence.

**Dynamic byte hierarchy (deferred scale-aware direction)** — MEGABYTE,
SpaceByte, BLT, and H-Net establish that multiscale byte processing can beat or
match token pipelines under controlled compute. H-Net is especially relevant to
the user's event idea because learned causal boundaries decide when its large
inner model runs. Its published controlled language models, however, begin near
680M parameters and require tens of billions of bytes before crossover. A 21M
BPE-level imitation would discard the mechanism's tokenizer-free advantage and
is not the immediate V34 experiment.

**Retired Particle-Field Recurrent Core v28** — a MARULHO-owned implementation
of positive particle dynamics inspired by BDH-GPU passed causal parallel versus
recurrent agreement, complete gradients, owned generation, and compiled-loss
parity. At 20,971,520 parameters it matched the 20,976,128 Transformer within
0.022%, and both arms saw the same 16,777,728 tokens. The particle field then
lost decisively: heldout loss/perplexity was 4.9132/136.08 versus
4.3193/75.14, and exact free relation generation was 11.33% versus 40.23%.
It trained at 11.1k versus 92.6k tokens/s and used 5.36 GB versus 0.60 GB peak
CUDA memory. Its 99.6%--100% candidate-ranking accuracy does not rescue weak
label-free generation. Decision:
`retire_v28_particle_field_no_joint_language_win`. No checkpoint or unseen
review is admitted; the model, falsifier, and tests are deleted. The result
rejects this implementation, not every possible population model, but MARULHO
does not require small units, biological metaphors, or any named architecture.

**Retired Editable Delta-Memory Candidate v1** — a tested fixed-state recurrent
fast-weight competitor with channel-wise decay and separate erase/write gates.
The 2-delta/2-attention hybrid beat the Transformer early, but the advantage
disappeared by 16.78M tokens. It then lost heldout loss and free relation recall,
failed unseen semantic generation, and trained about ten times slower. Its
implementation, runner, tests, rejected checkpoint, and schedule caches are
deleted; compact reports and git history retain the evidence.

**Retired Distributed Predictive Organism v1** — a tested parallel exact,
recurrent-unit, workspace, and episodic-memory base model. It beat the matched
Transformer at 4.20M and 16.79M tokens, but failed source-absent semantic
generation. At 67.11M tokens its advantage disappeared: organism/Transformer
loss was 3.8949/3.8924, strict free relation was 31.6%/32.0%, and steady
throughput was 33,963/110,345 tokens/s. The learned mixer still sent 60.6% of
traffic through the population path and 99.8% of units remained active. V1 did
not learn sparse specialization; it paid event-memory cost everywhere while
reducing exact-stream capacity. Its implementation, runner, audit, tests, and
rejected checkpoints are deleted. Compact reports and `RESEARCH.md` retain the
evidence.

**Retired Sparse Event-Memory v2** — this replacement preserved a
full-strength exact language stream and make event memory an optional residual,
not a competing half-model in every block. Event specialists must earn writes,
reads, and residual influence through counterfactual future utility plus an
explicit compute budget. Inactive specialists must consume no recurrent update
compute. The first decisive comparison must separate language-stream capacity
from memory benefit with exact-only, dense-sidecar, random-sparse, and
utility-sparse arms. The first causal PyTorch reference preserves all 20,976,128
baseline parameters and adds 133,124 sidecar parameters (0.635%). One-of-four
execution measures 25% specialist activity, scan/step equality passes, and warm
eager training retained 91.7% of Transformer throughput. No language-quality
result was claimed from machinery alone. The first 16.79M four-arm comparison
then found exact/dense/random/utility losses of 4.6140/4.6146/4.6128/4.6116 and
strict free relation scores of 14.5%/25.4%/27.0%/14.8%. Random and utility both
used 25% specialist compute. Utility did not beat random behavior, so the
chosen-expert-only credit interface is retired while the sidecar hypothesis
remained open for one comparative-credit test. Comparative all-expert probes
restored utility free relation to 25.8% but still lost to random's 27.0%, with
loss 4.6153 versus 4.6128. The selector/interface met its kill criterion. Its
implementation, runner, and tests are deleted; compact reports retain evidence.

**Modular Predictive Society v3 (retired)** — four independent two-layer causal
cells were matched within 0.12% of the 21M monolith and trained for 16.79M tokens
under one frozen schedule. Monolith/average/no-message/shuffled/real losses were
4.6140/5.0261/5.0460/5.0973/5.1073; strict free relation scores were
14.5%/5.1%/2.0%/0.4%/0%. Real messages lost every relevant control. Compiled
society arms were tightly compute-matched at 74.4--74.6k steady tokens/s, so the
negative is not a control-speed artifact. The model, runner, and tests are
deleted; the compact local report retains exact evidence.

**Depth-Preserving Modular Workspace v4 (no scale)** — the shared-interface
architecture repaired much of v3's capacity loss. At 16.79M tokens,
monolith/no-exchange/shuffled/real losses were 4.6147/4.8549/4.8518/4.8507 and
strict free relation scores were 32.0%/10.2%/11.7%/21.5%. Real exchange produced
a 9.8--11.3 point behavior gain over controls but no 0.005 loss win, and the
monolith remained stronger. The original mean-exchange mechanism is not scaled;
the report and separate review preserve the mechanical gate and interpretation.

**Content-Addressed Modular Workspace v5 (retired)** — at 16.79M matched tokens,
monolith/no-exchange/shuffled/real losses were 4.6142/4.8526/4.8479/4.8494 and
strict free relation scores were 17.2%/24.6%/22.7%/6.6%. Real associative memory
was worse than both controls. Write competition became more selective, and the
workspace controls ran within 0.13% steady throughput, so the failure is not an
unused-memory or compute mismatch explanation. The model, runner, and tests are
deleted; local v4/v5 reports retain evidence. These results retire the current
modular/Hopfield/column language line, not every future use in grounded tasks.

**Hyperspherical Transformer v6 (retired)** — the 20.988M normalized candidate
ran a recipe-separated 2x2 against the 20.976M frozen Transformer at context 72
and 16.79M tokens. Transformer-standard/Transformer-native/normalized-standard/
normalized-native losses were 4.6144/4.6448/6.2844/4.7092; strict free relation
scores were 14.8%/0%/0%/0%. The normalized native arm also lost the same-recipe
Transformer and its candidate-likelihood accuracy was 94.1% versus the frozen
baseline's 96.5%. All parameters received gradients, compiled/eager parity
passed, final matrix norm error was at most 1.79e-7, and all arms sustained
128.4k--130.1k tokens/s, so the negative is not an unused-parameter, projection,
or throughput explanation. No checkpoint was saved. The failed model, runner,
and tests are deleted; the local full report retains evidence.

**Gated Multiscale Dynamical Memory v7 (retired)** — the 20.977M candidate kept
all four attention layers, narrowed their feed-forward blocks, and inserted four
fixed-stable rotating memory banks between layers two and three. At 16.79M
matched tokens, Transformer/memory-off/single-scale/always-write/random-write/
learned-write losses were 4.6137/4.6092/4.6061/4.6076/4.6088/4.6066; strict free
relation scores were 21.5%/3.9%/10.5%/6.2%/3.5%/4.7%. Learned multiscale memory
therefore failed the Transformer quality guard and did not beat the simpler
single-scale control. Its candidate-likelihood relation score rose to 96.9%
versus the Transformer's 93.0% while free generation collapsed, another direct
warning that answer ranking is not generative competence. This is not explained
by a dead sidecar: its mean learned
write gate was 0.614 with entropy 0.599, all four bank norms were nonzero, every
parameter received gradients, parity passed, and memory-control throughput
varied by only 0.65%. Candidate training reached 112.7k tokens/s versus the
Transformer's 129.1k. Grouped causal convolution reduced recurrence compile cost
and the runner avoided four redundant graph compiles, but execution quality did
not become language quality. No checkpoint was saved. The model, runner, exports,
and tests are deleted; the compact local report retains the evidence.

**Depth-Allocated Transformer v8 (retired)** — exact-budget uniform,
early-heavy, and late-heavy profiles tested whether fixed nonlinear capacity was
better placed at different depths. Early-heavy produced a real but non-durable
short-budget result: at 16.79M tokens it beat uniform under two independent
model/schedule seeds, with losses 4.5843 versus 4.6067 and 4.5839 versus 4.6021,
and strict free relation 25.4% versus 7.0% and 30.9% versus 9.0%. Late-heavy lost
both screens. Successive halving then trained only uniform and early-heavy for
67.11M tokens under a third seed. Uniform won heldout loss 3.8861 versus 3.8957,
while free relation tied at 20.3%. Early-heavy candidate ranking reached 100%
versus uniform's 93.0% but did not improve free generation. Both arms contained
20,976,128 parameters, ran within 0.30% throughput, used about 2.61 GB including
the staged schedule, passed parity, and gave every parameter gradients. The
evidence therefore supports a budget/schedule-sensitive optimization effect, not
a superior static architecture or a known training-step crossover. No checkpoint
was saved. The model, runner, and tests are deleted; three local reports retain
the screen, replication, and durability evidence.

**Depth-Weighted Representation Reuse v9 (retired)** — two independent
16.79M-token comparisons tested identity, fixed-mean, fixed-random,
learned-unconstrained, and learned-simplex reuse against the Transformer. The
learned-unconstrained arm replicated a small heldout-loss improvement over the
Transformer (0.0092 and 0.0075), but strict free generation improved by 14.8
points in one seed and fell by 0.4 points in the other. It never beat identity
and every fixed control on both metrics. Fixed-mean's 0.0277 first-seed loss gain
shrunk to 0.0021 on replication; random mixing hurt loss; learned simplex stayed
near identity. The signed learned rows consistently attenuated the current
stream and subtracted small earlier-depth components, suggesting residual-scale
control rather than durable content reuse. Candidate controls shared one graph,
matched throughput within 0.14%, passed parity and gradient audits, and added
only 14 parameters, so the negative is credible. Both reports decide
`redesign_v9_disjoint_loss_and_behavior_signals`. No checkpoint was saved; the
model, runner, and tests are deleted.

**Product-Key Singleton Micro-Experts v10 (retired evidence)** — fixed token
hashing replicated a loss gain across two seeds, while learned product-key
routing collapsed to roughly 9% pool usage and did not improve loss. The query,
key, top-k, frozen-random, and learned-router machinery is deleted. Compact
reports retain the evidence; V11 directly owns the surviving shared path and
fixed-hash micro-capacity without a compatibility path.

**Hashed Singleton Micro-Experts v11 (active experiment, uninstalled)** — the
pruned durability candidate retains V10's 1024-wide shared path, 16,384
singleton functions, and eight deterministic token assignments, while deleting
query projection, product keys, top-k search, and learned/frozen router modes.
It has 36,180,480 parameters and 1,581,056 theoretical replacement-path
multiplies per token, 50.26% of the dense MLP's 3,145,728 before gather overhead.
After copying every surviving tensor, V11 token-hash logits exactly match V10's
winning arm. Causality, streaming equivalence, owned generation, hash uniqueness,
shared-only control, gradient coverage, and active-compute accounting pass. It
also passes a CUDA/Inductor smoke: candidate compile fell from V10's 39.4s to
22.8s, peak memory from 1.80 GB to 1.70 GB, and shared/hash steady rates reached
122.7k/124.2k tokens/s within 1.21%. The smoke report and two-step scores are
discarded. At the 67,112,064-token durability budget,
Transformer/shared-only/token-hash loss is 3.8951/3.9088/3.8747 and strict free
relation is 19.1%/25.8%/35.9%. Token-hash sustains 125.2k tokens/s versus the
Transformer's 130.4k, peaks at 2.70 GB including the 1.07 GB staged schedule,
passes parity and all gradients, and beats both controls on both required
margins. Decision: `promote_v11_hash_for_checkpoint_and_unseen_generation`.
An independent hash-only reproduction on the exact configuration, schedule,
tokenizer, token count, and initialization reaches loss 3.8738 and 30.9% strict
free relation. It therefore re-passes the original fixed joint gate against the
qualified Transformer and shared-only controls. The resulting 154.3 MiB strict
checkpoint is
`reports/language_scaling/hashed-micro-v11-qualified-seed2026-67m-20260711.pt`
(SHA-256 `6303ba4beabe49e163d4b8842ff798bc89215780c3ba269404895d1249f4b81b`).
A fresh strict reload restores all 36,180,480 parameters, token-hash mode, the
8,192-token vocabulary and tokenizer hash, tied embedding/head weights, and
MARULHO ownership. This promotes checkpoint fidelity and admits genuinely
unseen-generation qualification, not runtime installation.

The first unseen-generation qualification does not pass. Four FineWeb-Edu
holdout prompts produce mean source-continuation loss 4.3092 / perplexity 75.04;
four Cosmopedia prompts reach 3.6194 / 48.48. All eight fail the source-prefix
gate. The text is grammatical and multi-sentence but generic, repetitive, and
semantically unstable. Repetition penalty 1.1 plus a three-token no-repeat rule
raises Cosmopedia distinct-bigram fraction from 0.675 to 0.948 without changing
loss or prefix agreement, separating a real decode-loop defect from the deeper
model-quality blocker. Decision:
`continue_v11_general_language_pretraining_before_runtime_or_memory`. Continue
the same checkpoint toward roughly 251M cumulative general-language tokens,
then repeat heldout and unseen-generation comparison before runtime, continual
memory, or architectural promotion.

That continuation now completes at exactly 251,662,464 cumulative tokens.
Heldout loss improves 3.8709 to 3.4865 (perplexity 47.99 to 32.67) at 124.9k
tokens/s, and the strict 154.3 MiB candidate reloads with SHA-256
`fbf874923ebce6f4d36497f52a622dc8e222e01672b60876c910941af3fc1894`.
FineWeb-Edu/Cosmopedia prompt-local loss improves from 4.3092/3.6194 to
4.0272/3.3689. Against the local 251M Transformer, V11 ties FineWeb loss and has
higher diversity, but trails Cosmopedia loss 3.3689 versus 3.2047. Both models
still pass zero of eight source gates and direct review remains below coherent
base-language qualification. The general-only phase also catastrophically
forgets relations: candidate accuracy 95.7% to 32.8%, free generation 30.9% to
0%. Decision: `retain_v11_checkpoint_redesign_token_only_routing_before_more_scaling`.
Keep V11 as the strongest sparse base evidence, but test context-sensitive
routing and longer training context at matched compute before another large
scale run. Runtime and continual-memory work remain blocked.

**Counterfactual Route Regret (V12 admission)** — a frozen, read-only audit of
the 251M checkpoint changes only the final token's eight singleton IDs across
four equal-compute deterministic alternatives. Across 4,608 heldout contexts,
the metrics-only oracle lowers next-token loss by 0.1911 on average and finds
regret of at least 0.05 on 40.5% of tokens. FineWeb-Edu/Cosmopedia gains are
0.2020/0.1802; fragile-token gains are 0.3159/0.2963 versus 0.0882/0.0641 for
confident tokens. Every fixed alternative is globally 0.62–0.66 loss worse, so
there is no better static seed; useful choices are context-conditional and spread
across all alternatives. Forced-baseline logit delta is exactly zero, within-route
duplicates are zero, the route bank touches 95–97% of the pool, and parameter
hashes are unchanged. Decision: `train_v12_counterfactual_gate`. This admits only
training a small causal utility predictor on disjoint training contexts. Oracle
selection uses targets and is never an inference route or promotion claim.

The disjoint-data utility-gate test rejects the simple V12 realization. A
2,052-parameter linear gate underfits even its 18,432 training examples (realized
gain -0.0205) and worsens FineWeb-Edu/Cosmopedia heldout loss by
0.0381/0.0334. A 33,028-parameter MLP fits training counterfactuals (+0.1126) but
reverses to -0.0757 combined heldout gain. Harmful alternative selections exceed
helpful ones for both gates. Parent parameters remain frozen and hash-identical;
evaluation routes use no targets, and no gate artifact is saved. Decision:
`retire_v12_gate_cannot_predict_counterfactual_utility`. Oracle route regret is
real but not predictably accessible from the causal pre-expert hidden state with
this fixed route bank. The failed trainer/tests are deleted; the read-only audit
and compact negative report remain.

**Long-context V11 control** — the strict 251M checkpoint is rebuilt from a
72-token to a 256-token rotary context with no learned tensor or parameter-count
change. Before training, two heldout 72-token prefixes produce bit-exact logits
(maximum absolute delta 0). The matched 256 x 40 continuation adds 67,112,960
tokens and reaches 318,775,424 cumulative tokens. On the 256-token heldout set,
loss improves 4.2033 to 3.3243 / perplexity 66.91 to 27.78 at 123.6k tokens/s,
3.04 GB peak CUDA allocation, and 704.7 seconds total wall time. The strict
154.3 MiB checkpoint is
`reports/language_scaling/hashed-micro-v11-long-context-318m-candidate-20260711.pt`
(SHA-256 `cebe5ac7b5a84da1208d61c61715f58f61aa91c1ae2211208d005ac3f99506ae`).
FineWeb-Edu/Cosmopedia source-continuation loss changes only 4.0272 to 3.9951
and 3.3689 to 3.3586; all eight anchored cases still fail. Controlled decoding
improves Cosmopedia prefix overlap and diversity but still drifts into generic
topic boilerplate. Decision:
`retain_long_context_infrastructure_reject_context_only_quality_explanation`.
The checkpoint is a matched research control, not a promoted base or runtime
artifact. The next falsifier trains a multi-horizon future-prediction objective
from the same 251M parent and exact long-context schedule, so it must beat this
318M next-token-only control rather than benefiting from extra tokens.

That V13 falsifier is retired. Its 2/4/8-token heads do learn their auxiliary
tasks on a bounded heldout slice: losses fall 7.8794 to 6.0638, 8.1342 to 6.8102,
and 8.1896 to 6.9785. But after the exact same 67,112,960-token schedule, the
stripped inference model's full ordinary heldout loss is 4.9522 / perplexity
141.49 versus the next-token-only control's 3.3243 / 27.78. Training runs at
82.5k tokens/s and head removal is bit-exact, so this is objective interference,
not a checkpoint or inference-graph defect. Decision:
`retire_v13_future_prediction_no_control_gain`. No checkpoint exists; the
trainer, runner, and tests are deleted. Do not tune its weight on the same
holdout. The next scale test keeps the retained 318M next-token checkpoint and
first removes the schedule's linear GPU-memory growth so a materially larger
token/parameter comparison is possible.

The continuation path now has an exact indexed-host schedule mode. It preserves
the same source/index tuple and schedule hash but stores each sampled full batch
once on host and transfers only the active batch, instead of expanding every
scheduled input and target on CUDA. A 100-update, 1,024,000-token CUDA/Inductor
benchmark reaches 121.8k tokens/s and 1.97 GB peak allocation versus 123.6k
tokens/s and 3.04 GB for the retained expanded-device long-context run. The
roughly 1.4% throughput cost removes the old 16-bytes-per-requested-token CUDA
growth; the disposable benchmark report is deleted. This admits a materially
larger next-token scale test on the 3060 without changing training order or
quality semantics.

That scale test reaches exactly 1,000,001,664 cumulative tokens. The indexed-host
continuation adds 681,226,240 tokens and improves heldout loss 3.3243 to 3.0805 /
perplexity 27.78 to 21.77 at 121.9k tokens/s, 1.97 GB peak CUDA allocation, and
6,003.5 seconds total wall time. It stores 2.06 GB of unique full-shard batches
on host while avoiding a 10.90 GB expanded CUDA schedule. The strict 154.3 MiB
checkpoint is
`reports/language_scaling/hashed-micro-v11-indexed-continuation-1b-candidate-20260711.pt`
(SHA-256 `9e98a5f517f6f93f8d89544979990be8849ab4d03b2c206a98483ca3b3b68d64`).
FineWeb-Edu/Cosmopedia source-continuation loss reaches 3.9678/3.1405, but all
eight anchored cases still fail. Controlled Cosmopedia decoding raises bigram
diversity to 0.960 and produces readable paragraphs, yet remains generic and can
lose the prompt topic. FineWeb proposition loops worsen. Frozen relation ranking
rises 34.8% to 47.7% with no replay, while free relation generation stays 0%.
Decision: `retain_v11_1b_sparse_base_redesign_persistent_semantic_state`. Scaling
remains productive for likelihood, but the checkpoint is not runtime-qualified.
The next candidate must add a persistent semantic/topic state under ordinary
next-token loss, with V11 retained as the matched sparse token-capacity baseline.

V14 tests that persistent-state hypothesis and rejects its implemented form.
Four exact-reset arms each receive 67,112,960 tokens: V11/off, a
parameter-matched local residual, ungated segment delta state, and gated segment
delta state finish at heldout loss 3.0746086/3.0745938/3.0746429/3.0746036.
Relation ranking stays 48.4%/48.0%/48.4%/48.4% and free relation generation is
0% throughout. The gated memory is active, full rank, and fully reached by
gradient, but learns mean write 0.082 with no write above 0.5; its 0.0000050
loss gain over off is negligible and it trails local. Decision:
`retire_v14_no_segment_state_gain`. No V14 checkpoint exists. Its live model,
runner, tests, and checkpoint surface are deleted; only the decisive report
remains. The next architecture may not reuse mean segment summaries or simply
retune this gate.

V15 rejects ordered Haar detail while preserving a narrower multiscale clue.
Across three fresh seeds, mean 128/256/512/overwrite recall accuracy is
6.25/5.97/6.42/6.16% for a stronger same-state-byte flat GRU;
25.84/22.64/19.99/18.93% for seven raw-average dyadic banks;
25.54/21.90/19.11/16.32% for random balanced contrasts; and
25.44/21.70/18.84/17.12% for ordered Haar contrasts. Chance is 6.25% and the
metrics-only oracle is perfect. Haar does not beat equal-parameter controls and
is not admitted to language. Raw averages win every mean profile despite lower
512-token effective state rank than both contrast arms. Decision:
`redesign_v15_retain_multiscale_clocks_reject_haar_ordering`. The live preflight
and wavelet path are deleted; only its report remains. The next cheap test must
isolate whether raw's gain comes from several small banks, low-pass averaging,
or genuinely different clocks before any change to the 1B language model.

V16 isolates the V15 signal and finds that different clocks are not the source.
Across three fresh seeds, mean 128/256/512/overwrite recall accuracy is
6.34/6.48/6.25/6.05% for a larger same-state-byte flat GRU;
38.34/38.29/37.87/22.68% for seven independent token-rate banks;
25.35/24.92/21.47/20.10% for seven uniform low-pass banks;
20.01/20.21/20.21/14.91% for dyadic last-token banks; and
25.23/22.72/19.11/18.65% for dyadic low-pass banks. Token banks win every seed
and extrapolate from 128 to 512 tokens with almost no accuracy loss. They perform
seven times more recurrent updates and are 2.35x slower than flat in the current
implementation, so no sparse-compute claim is made. Decision:
`redesign_v16_retain_small_banks_reject_clock_claim`. The synthetic model and
runner are deleted.

V17 tests whether V16's small-bank organization transfers to language and finds
that it does not. Eight all-active 32-wide GRUs, the identical parameters used
token-locally, a dense 256-wide GRU, and V11/off each receive 33,556,480 exact-
schedule tokens from the one-billion-token parent. Off/local/grouped/dense loss
is 3.0788569/3.0790505/3.0789700/3.0786710. Grouped relation accuracy is 46.9%
versus 45.3/45.3/45.7%, but its loss is worse than off and dense and misses the
predeclared 0.02 joint margin. The grouped state is active, full-rank, receives
complete gradients, and runs at 97.8k tokens/s versus off's 121.9k, so the
negative is not explained by a dead organ or failed execution. Decision:
`retire_v17_grouped_recurrence_no_language_gain`. No checkpoint exists; the
model, partial-compile path, runner, and tests are deleted. Independent small
units remain a synthetic-memory observation, not an admitted language design.

The V18 inter-segment bridge line is retired. Its first run is retained at
`reports/language_scaling/v18-segment-memory-800step-20260711.json` (SHA-256
`6141cf272002764c0ae52e5c894937e92fddb9199ac4b3aee01464dcf5f44c89`).
Off/exact/local/recency/mean/learned candidate accuracy is
24.2/73.0/72.7/72.3/74.2/72.7%; greedy exact accuracy is
0.0/18.4/17.2/16.4/19.5/4.7%. Exact history fails its frozen-interface gate,
and query-local capacity nearly matches it, exposing answer-template shortcuts.
The learned writer receives complete gradients but collapses to effective rank
2.01 with state norm 548.3. Decision:
`retire_v18_frozen_segment_bridge_interface`. V18b does not rewrite that result:
it adds standard post-write normalization and a paired audit over 47 identical-
question groups and 558 differing-answer pairs. Within each pair, all
distractors and their order remain fixed; only the target source changes.

The final V18b report is
`reports/language_scaling/v18b-segment-memory-counterfactual-800step-20260711.json`
(SHA-256
`7c8d330a76f7b421d3a0281fc8eb7a54ab5488d7e6297201785a5f74efed6e6d`).
Exact history reaches 80.5% candidate and 25.4% greedy exact accuracy, versus
72.7%/17.2% for source-independent local capacity. On paired source swaps its
exact/local accuracy is 25.33/17.90%, a 7.42-point causal margin below the
predeclared 10 points. Normalized learned slots reach only 68.4% candidate,
4.7% greedy exact, and 3.93% paired accuracy. Their norm is repaired to 23.8,
but effective rank remains only 1.78 despite complete gradients. Decision:
`retire_v18b_exact_history_no_source_causal_gain`. No checkpoint exists; the
model, runner, tests, transient feature caches, and smoke reports are deleted.
The next persistent-state architecture must be jointly learned with the cortex
on contiguous streams rather than attached to frozen V11 representations.

V19 jointly trains that state with the cortex and retires its recurrent form.
The retained report is
`reports/language_scaling/v19-joint-memory-800step-20260711.json` (SHA-256
`ce73f309a84ab80a0a1faa1fb192bbdcc2b17abcba409a57eb7e44a44a56f7af`).
Off/exact/local/recency/mean/recurrent candidate accuracy is
67.6/87.1/66.4/76.6/82.0/84.0%; exact free accuracy is
15.6/49.6/15.2/25.0/28.5/30.1%; paired source-following is
16.59/47.60/16.59/24.45/29.69/30.13%. Exact history proves a 31-point causal
margin over local. Bounded summaries carry real source information, but
recurrence ties parallel mean instead of beating it and remains 17.47 points
behind exact. Every recurrent parameter receives nonzero gradient, matrix rank
is 502, both general holdouts remain within the 0.10 loss guard, and state uses
the intended 32 KiB. The more diagnostic failure is overwrite: recurrent output
changes on only 7.17% of answer-changing pairs versus 23.48% for mean and 60.04%
for exact, while recurrent effective rank is 24.16 versus mean's 29.30.
Decision: `retire_v19_joint_memory_tokens_insufficient_source_following`. No
checkpoint exists.

V19b closes the overwrite explanation without rescuing the interface. Its
retained report is
`reports/language_scaling/v19b-partitioned-memory-800step-20260711.json`
(SHA-256
`efb24df8e4c9fe1c1fe89a398ffcf753f2c03b730300a6181c2949303d417a73`).
Two fixed eight-slot segment banks concatenate into the same sixteen-token,
32-KiB state. Candidate/free/paired accuracy reaches
85.9/31.6/31.44%, only 1.9/1.5/1.31 points above recurrence and still 16.16
paired points below exact history. Output changes on 24.19% of source swaps,
matching mean's source sensitivity but not exact's behavior. All partition
parameters receive nonzero gradients, matrix rank is 447, effective rank is
27.51, and the two general holdouts remain inside the guard at +0.0740/+0.0848
loss. Decision: `retire_v19b_partitioned_memory_insufficient_source_following`.
No checkpoint exists. The model, runner, and tests are deleted; do not add a
gate, more slot layouts, or a longer schedule to this interface.

The next memory hypothesis preserves content instead of repeatedly compressing
it. A source-only write stores exact episode tokens plus a small retrieval key
and provenance. At query time, a bounded selector retrieves a fixed number of
raw episodes into the local exact-attention window. Archival storage may grow,
but active tokens and compute remain bounded. Before training, a read-only audit
must compare random, recency, lexical overlap, and frozen-cortex keys against an
all-history upper bound and a metrics-only oracle. Only a label-safe selector
with sufficient target-source recall earns a language screen.

V20 completes that read-only audit at
`reports/language_scaling/v20-exact-episodic-retrieval-audit-20260711.json`
(SHA-256
`8436c6fdbb1976d75b22b4974c6acb5c1aa884702f0a5a579b2125a9697fc57d`).
Random/recency/lexical/frozen-last/frozen-mean top-1 recall is
27/34/71/38/41%; frozen V11 state is not yet a reliable address. Lexical TF-IDF
misses the predeclared top-1 gate, including only 53% of both answer-changing
paired targets. Decision: `redesign_v20_no_fixed_key_retrieves_exact_episode`.
The same label-safe lexical ranking nevertheless reaches 98.8% target inclusion
at top two, using 96 raw source tokens instead of all-history's 192. V21 is
admitted as a separate language falsifier: lexical top-two retrieval must recover
paired free generation, stay close to all-history, beat random/recency/top-one
controls, and preserve general heldout loss. Top-two recall alone is not a
quality promotion.

V21 passes that full language gate. The retained report is
`reports/language_scaling/v21-exact-episodic-retrieval-800step-20260711.json`
(SHA-256
`b2a60cc1e3c0a45ea65811238210c344d8d6f124773556952bc0fe41e3a4def1`).
Off/all-four/random-two/recency-two/lexical-one/lexical-two candidate accuracy is
68.8/87.9/79.3/79.3/89.5/100.0%; free exact accuracy is
16.0/39.5/27.7/25.0/44.9/51.6%; paired source-following is
17.0/38.0/27.5/24.9/45.4/52.0%. Lexical-two includes the target in 98.83% of
cases, changes its output on 82.62% of answer-changing swaps, and gets both
paired answers correct 41.58% of the time. It beats all-history behavior while
using 96 rather than 192 active source tokens. The two general holdouts regress
only +0.0631/+0.0772 loss, all model parameter tensors receive nonzero gradient,
and peak allocation is 0.90 GiB versus all-history's 1.03 GiB. Wall time is
effectively tied, so no speed claim is made. Decision:
`advance_v21_exact_episodic_retrieval_to_contiguous_streams`.

This is the first admitted memory architecture in the current iteration, not
Base-Language Qualification. It uses a relation-specific lexical key, retains a
growing exact-token archive, has no saved checkpoint/index contract, and has not
improved real document continuation. The next screen must build a causal archive
from prior general-document spans, retrieve under a fixed active-token budget,
compare lexical retrieval with recency/random/local controls on disjoint
documents, and save the selected cortex plus archive/index state only if heldout
loss and source-anchored generation improve together.

V22 rejects the always-read general-document interface without rejecting the
archive itself. The retained report is
`reports/language_scaling/v22-causal-document-retrieval-audit-20260711.json`
(SHA-256
`af1898ece04196ebad35adcdf5c89c56d13cc2a0419f5abe7f9fc4ee18c6ea10`).
On 128 disjoint FineWeb-Edu plus 128 Cosmopedia documents, local-only loss is
3.0048. Non-promotable oracle-one reaches 2.9707, a paired +0.0341 gain with
95% bootstrap interval +0.0116 to +0.0584, so the older exact episode contains
usable predictive information. Lexical-one and frozen-mean-one retrieve that
episode in 75.0% and 74.2% of cases, but reach only 3.0031 and 3.0021 loss; their
confidence intervals cross zero. Top-two retrieval is worse despite higher
inclusion, and all-four regresses to 3.0355.

The failure is asymmetric. When lexical-one selects the true document its mean
paired gain is +0.0372; when it selects a distractor the mean is -0.1050.
Lexical score margin identifies a plausible abstention region: the highest-margin
50% has 95.3% retrieval precision, while always reading is only 75.0% precise.
These curves use target identity for metrics and cannot promote a policy.
Decision: `redesign_v22_addressing_does_not_recover_useful_episode`. The next
falsifier must calibrate a retrieve-or-abstain rule on separate training
documents, freeze it, and evaluate it on these disjoint documents before any
joint training, archive checkpoint, or runtime integration.

V22b freezes that abstention policy and rejects it. The retained report is
`reports/language_scaling/v22b-confidence-gated-document-retrieval-20260711.json`
(SHA-256
`189966d147b10a6ff1a5b003e86ced389f8b562e23635213b73101d510476aa8`).
Replay calibration selects margin 0.055112 at 51.76% coverage and 95.09%
precision. On the untouched 256-document evaluation it transfers to 54.30%
coverage and 97.84% precision. Gated lexical retrieval improves paired loss by
+0.0356 with 95% interval +0.0180 to +0.0552; equal-mask random and recency
controls regress by 0.0375 and 0.0307. However, always-on lexical-one improves
by +0.0388 on the same cases, so the gate is 0.0032 worse instead of at least
0.0025 better. Decision:
`retire_v22b_fixed_confidence_gate_insufficient_language_gain`.

Lexical margin predicts same-document retrieval correctness, but that label is
not the same as marginal predictive utility. This repeats V12's warning against
training another detached utility predictor on a frozen choice bank. The
confidence-gate runner and tests are deleted. The next admissible exact-memory
test must co-train the cortex on causal selected/off/random document contexts so
it can learn how to use or ignore retrieved evidence; it still needs a disjoint
likelihood win and anchored generation before any checkpoint or runtime claim.

V23 tests that co-adaptation and fails promotion while preserving a real source-
use signal. The retained report is
`reports/language_scaling/v23-joint-document-retrieval-800step-20260711.json`
(SHA-256
`5b0010dbb3361362ec174b067efaf93e783c7860cb54db1e5dae23532a45cb6e`).
Off/random-one/lexical-one/oracle-one disjoint document loss is
3.2274/3.2454/3.2083/3.1857. Oracle gains +0.0417 over off with 95% interval
+0.0112 to +0.0743, so the joint interface is learnable. Lexical gains +0.0192,
but its interval is -0.0049 to +0.0481. Within the lexical-trained model, true
history beats a guaranteed distractor by +0.0833 with interval +0.0580 to
+0.1108; random training is significantly worse than off. The cortex therefore
learns to use source identity, but the selector includes the target in only
69.92% of eval cases and the aggregate win remains heterogeneous.

Retention independently blocks the branch. Lexical FineWeb-Edu/Cosmopedia
general loss regresses +0.1200/+0.1346, both above the +0.10 limit. Eight free
continuations reach only 3.13% expected token-position accuracy and 19.17%
unique-target-token recall. Decision:
`retire_v23_lexical_retrieval_breaks_general_language`; no checkpoint exists.
The next bounded redesign increases ordinary replay to 50% and trains lexical
top-two beside top-one and equal-token random-two. Top-two raises target
inclusion and must learn to ignore its extra distractor; it survives only with a
significant two-corpus likelihood win, source use, and restored retention.

V24 restores retention but rejects top-two. The retained report is
`reports/language_scaling/v24-balanced-top2-document-retrieval-800step-20260711.json`
(SHA-256
`340e397a4b90d035c26ab30ce849e42d670def6386dd12d5fcfe1be5692e700d`).
Off/random-two/lexical-one/lexical-two/oracle-two loss is
3.1261/3.1281/3.1006/3.1070/3.0946. Lexical-two includes the target in 82.42%
of cases, below its 85% gate, and gains +0.0191 with interval -0.0046 to
+0.0443. It is 0.0064 worse than lexical-one. True-plus-distractor still beats
two wrong spans by +0.0802 with a positive interval, and general regression is
only +0.0677, so failure is added distraction rather than lost source use or
forgetting. Decision: `retire_v24_balanced_top_two_no_joint_language_win`.

The lexical-one control is a new positive that cannot be discarded: it gains
+0.0255 over off with interval +0.0036 to +0.0513, true history beats wrong
history by +0.0682, and general regression is +0.0701. V24 did not include an
equal-token balanced random-one arm and did not preregister lexical-one for
promotion. The next test is therefore one fresh-seed replication of balanced
top-one against off, random-one, and oracle-one. Top-two is retired. A top-one
replication must win both corpora and source-use controls before any checkpoint.

V25 replicates the top-one likelihood result and then fails anchored generation.
The retained screen is
`reports/language_scaling/v25-balanced-top1-replication-800step-20260711.json`
(SHA-256
`dafae2ddabbebb62200e2b8758120e30e38c2c0c4e8ca7705e80f36dd114af76`).
Off/random-one/lexical-one/oracle-one loss is
3.0877/3.0959/3.0447/3.0112. Lexical gains +0.0430 over off with 95% interval
+0.0204 to +0.0668, beats matched random by 0.0512, and includes the target in
71.88% of eval cases. Cosmopedia and FineWeb-Edu improve separately by 0.0535
and 0.0326. True history beats a guaranteed wrong episode by +0.1127 with
interval +0.0885 to +0.1389. General regression remains bounded at
+0.0657/+0.0823. Decision:
`advance_v25_replicated_top_one_to_anchored_review`.

The required manual review is retained at
`reports/language_scaling/v25-anchored-generation-review-20260711.json`
(SHA-256
`5e1eb137c2949b60579eab50f5ee91db2183349665a5921497a2e3965afe5d7e`).
All eight continuations fail. They switch from GPRS to GPL/airplanes, contradict
the multiple-income premise, restart earlier sections, corrupt technical terms,
or drift to unrelated sun/Celtic-pyramid text. Expected token-position accuracy
is 7.81% and unique target-token recall is 21.53%. Final decision:
`retain_v25_likelihood_signal_redesign_separate_evidence_reader_before_checkpoint`.
No checkpoint is saved or installed.

Raw episode concatenation is now closed and its live runner/tests are deleted.
The surviving evidence is architectural: exact top-one memory improves causal
likelihood and source sensitivity, but ordinary self-attention does not turn it
into reliable free generation. The next candidate must encode the episode on a
separate evidence path and inject it through bounded gated cross-attention while
preserving the local cortex sequence and the same off/random/true controls.

V26 rejects a separate reader applied only after the final cortex layer. The
retained report is
`reports/language_scaling/v26-separate-evidence-reader-800step-20260711.json`
(SHA-256
`bc8b3f9ec03fcbf6f241ba0c73320c1f2986e15fdc6bf6ae832098b447fe7a7f`).
The reader adds 1,049,601 parameters; every one of its five tensors and all 28
cortex tensors receive nonzero gradients. Gate-zero/shuffled/raw/lexical/oracle
loss is 3.09210/3.09178/3.08659/3.09205/3.09199. Oracle-reader gains only
+0.00010 with an interval crossing zero, true-vs-wrong evidence is +0.00002,
and its gate moves only from 0.11920 to 0.11949. Lexical-reader source-swap
output change is 12.5%. Decision:
`retire_v26_reader_task_not_learnable_with_oracle_evidence`.

The failure localizes the interface: raw context can influence every V11 layer,
whereas V26 adds evidence after local computation is already finished. More
selector tuning or a larger final residual is not justified.

V27 rejects the bounded interleaved replacement. The retained report is
`reports/language_scaling/v27-interleaved-evidence-reader-800step-20260711.json`
(SHA-256
`c5db3e4d84cddb4c5707861fa610513e6c0812a7fd66264ff6786bd04bfa4751`).
One shared eight-head reader was injected after V11 blocks zero and two, with
an independent scalar gate at each injection. Gate-zero/shuffled/raw/lexical/
oracle loss is 3.12936/3.16483/3.08680/3.16858/3.16858. Raw context therefore
replicates a +0.04256 gain with interval +0.01674 to +0.07216, while oracle
evidence loses 0.03922 with its entire interval below zero. Oracle true-vs-wrong
gain is only +0.00617 with an interval crossing zero. Both gates move slightly
down from 0.11920 to about 0.1186; all reader and cortex tensors receive
nonzero gradients, and general retention passes. Decision:
`retire_v27_interleaved_task_not_learnable_with_oracle_evidence`.

V25's exact-history likelihood signal remains real, but neither final-layer nor
interleaved cross-attention converts it into a useful evidence interface. The
current cross-attention document-memory line is closed. Its model, runner, and
tests are deleted; no checkpoint is saved or installed. Work returns to the
base-language architecture rather than widening reader gates, layers, or
selector sweeps.

**Execution-Coupled Structured Memory** — a possible later reasoning organ,
inspired by LCWM's retained markerless role/path evidence and its V10 diagnosis.
Candidate memories or latent programs should earn selection because executing
them improves downstream prediction, not merely because an input was locally
surprising. It is excluded from the base token mixer until the replacement base
model demonstrates coherent unseen language.

**Continual Language Learning** — sequential domain updates from a
quality-qualified base checkpoint, with old-domain, new-domain, and replay
losses measured before and after. The required result is new learning with
bounded forgetting and restored checkpoint fidelity.

**Structural Plasticity** — changing model or memory structure under a
checkpointed transaction. It is paused until a replacement base model beats its
matched fixed-capacity control. No old routed-expert expansion path is
maintained.

## Grounded Subsystem

**Grounded Sparse/Column Runtime** — existing SNN, column, binding, surprise,
and sensorimotor machinery under `src/marulho/core`. It is a separate
experimental substrate, not the language generator and not automatically part
of the future architecture.

**Columns** — candidate units for grounded object/action reference frames or
sparse competition. They are retained only where a grounded experiment can
compare them with a simpler baseline.

**LCO Transfer Experiments** — future tests of persistent identity,
object-action binding, movement/intervention separation, and causal transfer.
Results from another repository are inspiration, not MARULHO evidence until
reproduced here.

**Thousand Brains Theory** — optional scientific inspiration, not a design
constraint. Reference-frame or column mechanisms must win a bounded grounded
task to enter the architecture.

## Evidence Language

**Evidence Artifact** — a JSON report produced by an explicit experiment. It
records inputs, configuration, metrics, ownership flags, and branch decision.
Its existence does not make the decision positive.

**Accepted Run** — the command completed and its invariants held.

**Quality Promotion** — a checkpoint crossed the current quality boundary.
MARULHO has no quality-promoted language checkpoint yet.

**Branch Decision** — one of:

- `scale`: the mechanism wins and merits a larger experiment;
- `redesign`: the hypothesis remains plausible but the implementation or
  experiment is inadequate;
- `retire`: a matched baseline falsifies the maintained path.

**Runtime Truth** — observed execution and state, not configured intent. CUDA,
checkpoint restore, active compute, memory use, and mutation claims require
direct measurements.

## Current Evidence State

The equal-time experiment first selected the 20,976,128-parameter model over
the 62,924,544-parameter model: heldout loss 4.0942 versus 4.6129 after 565.9
versus 560.8 synchronized training seconds. The smaller model processes about
2.43 times more tokens per second on the RTX 3060.

The subsequent unique-data curve trained a fresh 20,976,128-parameter model
with a MARULHO-owned 8,192-token BPE over three provenance-recorded FineWeb-Edu
shards and the same disjoint later-offset holdout:

| Update tokens | Unique | Repeated | Heldout loss | Perplexity | Train time | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16,777,216 | 16,777,216 | 0 | 4.5754 | 97.07 | 232.6 s | 72,125 |
| 33,554,432 | 33,554,432 | 0 | 4.1328 | 62.35 | 462.8 s | 72,498 |
| 50,331,648 | 50,331,648 | 0 | 3.9889 | 54.00 | 693.9 s | 72,531 |

The selected stream contains 57,963,348 BPE tokens, so the final point is 0.87
unique corpus epochs with no repeated update tokens. The loss curve improves,
but the interval gain contracts from 0.4426 to 0.1439. Unseen continuations are
more prompt-related yet still repetitive, sometimes malformed, and causally
incoherent.

The subsequent explicit-record TinyStories diagnostic used the same 21M
architecture, 250,000 official training rows, all 21,990 validation rows, and
50,334,464 unique update tokens:

| Update tokens | Heldout loss | Perplexity | Train time | Tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 16,777,216 | 2.1879 | 8.92 | 251.5 s | 66,698 |
| 33,554,432 | 1.9520 | 7.04 | 500.7 s | 67,013 |
| 50,334,464 | 1.8573 | 6.41 | 748.9 s | 67,210 |

All four unseen story prompts produced grammatical, prompt-conditioned,
multi-sentence continuations. Three emitted EOS before the 192-token cap; one
continued to the cap. The remaining failures are entity drift, object-property
contradiction, role confusion, and incomplete causal closure.

Decision: `keep_transformer_scale_curated_general_curriculum`.

Interpretation: 21M parameters remains the local compute optimum in the tested
range, and the base recipe can learn coherent English. Raw general-web scale and
curriculum quality—not basic Transformer incapacity—are the active blockers.
TinyStories is a restricted diagnostic and does not qualify general language.
The next run must mix structured synthetic textbooks/stories with explicit-
record educational web data, then evaluate the original general prompts.

The first structured-general ablation used 100,000 explicit Cosmopedia v2
records (84,770,600 BPE tokens). At 16,777,216 / 33,554,432 / 50,331,648 update
tokens, heldout loss was 3.7038 / 3.2881 / 3.1318. The final point was 0.82 of
the selected training stream with no repeated selected updates. Grammatical
form improved, but all six unseen continuations hit the 192-token cap and lost
entity, property, or causal state. `cache` became unrelated games or fabric;
the coin/cup relation disappeared. The checkpoint therefore does not promote
general-language quality.

Decision: `continue_21m_on_new_disjoint_structured_documents`.

Interpretation: the improving curve at less than one selected epoch does not
justify retiring the Transformer or promoting the checkpoint. Continue the
same weights on new Cosmopedia records, evaluate against a separate shard that
never trained the checkpoint tokenizer, and preserve fresh-versus-repeated
token accounting. If the longer curve flattens without state consistency,
scale/redesign from that evidence rather than from prose fluency alone.

The document-disjoint continuation restored the 50.33M-token weights/tokenizer,
trained on 150,000 new shard-1 records, and held out 10,000 shard-2 records that
never trained the tokenizer. On this strict holdout, loss moved from 3.1289
before the phase to 2.9863 at 100,663,296 cumulative tokens and 2.7681 at
150,994,944. Both phase points used fresh selected windows. Rain/soil and
server/data adherence improved, but every continuation still hit 192 tokens;
key/box and coin/cup state mutated or vanished, and causal explanations still
degenerated. This remains below Base-Language Qualification.

Decision: `mix_fresh_structured_and_educational_web_data_at_21m`.

The active checkpoint is the 150,994,944-token continuation with exact
optimizer/scaler/RNG/batch-position metadata. Its SHA-256 is
`7fcaa42ed2a32c2c4f2bbba60d632b9a4b78385852a6613141c77372a59998fd`.
Continue these weights on an explicit Cosmopedia/FineWeb-Edu ablation. Do not
add episodic memory merely to compensate for an undertrained base; do not keep
scaling synthetic textbook style if strict loss improves without entity/causal
binding.

The mixed continuation then restored exact AdamW state and trained on 75,000
fresh FineWeb-Edu plus 75,000 fresh Cosmopedia records, with 10,000 disjoint
holdout records from each source. Combined holdout loss moved from 3.6216 before
the phase to 3.4429 at 201,326,592 cumulative tokens and 3.2534 at 251,658,240.
Neither phase point repeated a selected update. Yet entity and causal binding
did not improve: notebook ownership vanished, valve/pump ordering became word
association, and coin/cup state still drifted.

A same-checkpoint decode comparison falsified the simplest alternative
explanation. Seeded temperature-0.8/top-p-0.9 nucleus sampling increased lexical
variety but did not restore prompt relations and often worsened factual drift.
The blocker is not greedy argmax alone.

Decision: `falsify_relation_binding_before_more_generic_pretraining`.

The active checkpoint owns 251,658,240 cumulative update tokens, 61,440
optimizer steps, and exact optimizer/scaler/RNG/batch state. Its SHA-256 is
`25e16893fd6bec4c8f7c858f7fc7bdd969e13fbe733104f4467d7f2f784a7fd3`.
Build a procedural entity/event curriculum with compositionally disjoint
holdouts, continue a branch from this checkpoint, and measure both relation
accuracy and general heldout-loss retention. If the Transformer learns the
relations, redesign curriculum; if it cannot, test PMRM-like episodic binding or
larger capacity against the same benchmark. Do not continue generic token scale
without resolving this branch.

The controlled relation-binding falsification used 200,000 procedural training
documents and 256 compositionally held-out cases. Candidate answers were scored
before the correct index was used for metrics. After 16.78M relation-phase
tokens, total accuracy improved from 47.7% to 87.9%. Container, ownership, and
property accuracy reached 100%; event-order accuracy improved only from 29.7%
to 51.6%.

This is evidence that the 21M Transformer can represent static bindings under a
focused objective. It is not a promotable checkpoint: unchanged mixed-language
loss regressed from 3.2534 to 8.7139, and free generation remained unreliable.
The candidate is rejected for catastrophic forgetting; the 251,658,240-token
mixed checkpoint remains active.

Decision: `relation_learned_but_catastrophic_forgetting_test_replay`.

Next compare a budgeted relation-plus-general replay mixture from the active
base. Require both held-out relation gain and bounded mixed-language loss. If
replay succeeds, continue toward consolidation/replay policies; if it fails,
compare parameter isolation and PMRM-style surprise-selected episodic binding
under equal memory/compute budgets. Event order remains a separate causal
blocker and must not be hidden by perfect static-binding subtasks.

The relation-plus-general replay run used an approximately 20/80 input mixture.
It preserved mixed-language loss (3.2534 to 3.2485) and raised candidate-ranking
accuracy to 98.0%, including 92.2% event order. A subsequent label-safe greedy
audit exposed the remaining gap: exact free-answer accuracy rose from 0% to
44.9%, but ownership reached only 10.9% and container persistence remained 0%.
Property reached 93.8% and event order 75%. Open general prompts also inherited
procedural Q/A template fragments.

Decision: `replay_improves_candidate_ranking_not_free_binding`.

The replay candidate is rejected; the 251,658,240-token mixed checkpoint stays
active. This result motivated the subsequently falsified output-adapter test and
the move to PMRM-style surprise-selected episodic binding under explicit
memory/latency budgets. Do not call multiple-choice accuracy alone relation
competence.

Frozen-base residual output adapters were tested at rank 32 and rank 128, then
removed. Rank 32 used 32,768 trainable parameters and reached 83.2% candidate /
3.1% exact free accuracy. Rank 128 used 131,072 trainable parameters and reached
84.4% / 2.3%. Rank 128 sustained ~151k training tokens/s at ~681 MiB peak
allocated VRAM with only +0.0227 mixed-loss regression, but increasing rank did
not improve free binding and remained far below full replay's 44.9%.

Decision: `retire_output_adapter_test_selective_episodic_binding`.

The adapter architecture, CLI, tests, and local candidates are retired rather
than carried as compatibility debris. The active base remains the
251,658,240-token mixed checkpoint. This decision led to the subsequent prompt-
memory comparison of surprise, random, and recency policies under equal slot,
byte, read, and write budgets.

The first prompt-level PMRM-inspired memory interface was falsified and removed.
With eight distractors and two stored/read episodes, exact free accuracy was
18.4% for no memory, 12.1% random, 5.9% recency, 8.6% surprise, 11.7% full-store
retrieval, and 3.9% for the non-promotable oracle. Surprise required 109.7 s
(2.33 cases/s) versus 72.8 s (3.52 cases/s) for no memory. Even full/oracle
retrieval hurt, so prepending selected text is the rejected interface; this does
not falsify learned hidden-state episodic memory in general.

Decision: `retire_prompt_memory_build_answer_masked_post_training`.

The replay model's 98% candidate ranking versus 44.9% free generation indicates
an objective/interface gap. The answer-masked test restored exact AdamW state
from the active checkpoint and alternated 2,048 relation-answer updates with
2,048 ordinary general-language updates. It processed 10,621,968 tokens, of
which 8,688,968 bore loss, in 312.4 seconds including milestone evaluation at
1,258,403,840 peak allocated CUDA bytes.

Mixed heldout loss moved from 3.2534 to 3.2684, candidate accuracy from 47.7% to
87.1%, and strict free accuracy from 0% to 19.5%. Free container, ownership,
property, and event-order accuracy reached 1.6%, 26.6%, 7.8%, and 42.2%. This
misses the preregistered 60% free-answer threshold and underperforms full
replay's 44.9%, despite bounded scalar loss. The rejected checkpoint and
experiment implementation are deleted; the compact local report is
`reports/language_scaling/answer-masked-post-training-21m-4096-20260710.json`.

Decision: `retire_integrated_pmrm_build_editable_delta_memory_competitor`.

The replacement screen compared pure editable delta memory and local-attention
hybrids with the same 20.98M Transformer. Pure recurrence reached 8.0018 loss at
269,568 tokens; one attention layer improved it to 7.6833; two attention and two
delta layers reached 7.5461 versus Transformer 7.4972. The 2/2 hybrid then
crossed the baseline: 6.9042 versus 7.0625 at 1,057,536 tokens, and 5.6966
versus 5.9962 at 4,199,040 tokens. Its candidate relation accuracy at the last
point was 90.6% versus 73.8%, while strict free generation remained only 0.8%
versus 0%.

The 2/2 hybrid had 20,977,152 parameters and complete gradient coverage. Its
unfused PyTorch reference was about ten times slower than the Transformer. This
early result justified durable scaling and unseen-generation testing, not
runtime installation or a replacement claim. The compact finalist report is
`reports/language_scaling/delta-editable-half-finalist-4m-20260710.json`.

At 16,785,792 tokens, the early advantage reversed. The Transformer reached
heldout loss 4.5657, 98.4% candidate relation accuracy, and 17.2% exact free
accuracy. The hybrid reached 4.5858, 87.9%, and 7.8%, while sustaining 7,984
training tokens/s versus 83,505. Its four source-absent unseen prompts produced
English-shaped text but failed semantic continuation and conflict binding; the
silver-key prompt ended with an unrelated water/shelf answer. Surface metrics do
not override the human semantic failure. Compact reports are
`reports/language_scaling/delta-editable-half-durable-16m-20260710.json` and
`reports/language_scaling/delta-editable-half-unseen-generation-16m-20260710.json`.

Decision: `retire_delta_v1_design_distributed_predictive_organism`.

The first distributed-organism finalist used the same tokenizer, selected
windows, schedule, optimizer, model seed, parameter budget, and evaluation as a
fresh Transformer. Its compiled reproduction at 4,199,040 update tokens reached
5.5257 heldout loss versus 6.0113 and 98.4% candidate relation ranking versus
72.7%. Strict free
relation generation was 0% for both. Every candidate parameter received a
gradient. Its learned parallel mix averaged 37.7% exact attention and 62.3%
predictive population; all units remained active, so sparse-compute benefit has
not been demonstrated. The compiled runner executed 354 ordinary full-graph
steps and 51 explicit eager probes. The organism sustained 50,264 steady and
45,758 compile-amortized tokens/s, versus 124,073 and 105,193 for the
Transformer. The temporary 70.4 MB bounded corpus cache was deleted after the
report. Compact eager and compiled results are
`reports/language_scaling/distributed-organism-finalist-4m-20260710.json` and
`reports/language_scaling/distributed-organism-compiled-finalist-4m-20260710.json`.

Decision: `continue_organism_to_durable_budget_and_unseen_generation`.

At 16,785,792 fresh matched update tokens, the organism reached 4.5101 heldout
loss, 96.9% candidate relation ranking, and 28.1% strict exact free relation
generation. The Transformer reached 4.6130, 91.8%, and 12.5%. The organism
sustained 51,994 steady and 50,797 compile-amortized tokens/s versus 123,815 and
119,269. Every candidate parameter received a gradient. Its 202 explicit probes
split into 154 unit and 48 episodic interventions; the mean counterfactual target
was positive but small at 0.00202. The learned mix was 40.4% exact and 59.6%
population, while every unit remained active, so sparse compute is still
unproven. The 272.2 MB schedule cache was deleted. The strict checkpoint is
268,848,073 bytes with SHA-256
`2e1406e4df0a1d04aa589777ef9a58b807337ed2a21f758a3f6c91900872c0fd`.
The compact report is
`reports/language_scaling/distributed-organism-compiled-durable-16m-20260710.json`.

Decision: `test_organism_unseen_generation_before_any_promotion`.

The source-absent audit verified all six prompts absent from the five declared
corpora, then rejected every greedy and seeded nucleus continuation on semantic
review. Failures included relation-template contamination, invented concepts,
irrelevant causal/procedural text, truncation, and conflict reversion from the
glass jar to the wooden drawer. The report and explicit review are
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710.json`
and
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710-review.md`.

The matched curve also does not yet justify blind scaling. From 4.20M to 16.79M
tokens, Transformer loss improved by 1.3983 while organism loss improved by
1.0155, shrinking the organism margin from 0.4857 to 0.1029. A crude two-point
log-linear fit crosses near 24.4M tokens. This extrapolation is a falsification
target, not a claimed scaling law, because the points are fresh runs with
different schedule realizations.

Decision: `no_promotion_scale_to_64m_and_retest_loss_slope`.

At 67,112,064 fresh matched update tokens, the predicted crossover occurred.
The Transformer reached 3.8924 heldout loss, 98.0% candidate relation ranking,
32.0% strict free relation generation, and 110,345 steady tokens/s. Organism v1
reached 3.8949, 89.8%, 31.6%, and 33,963 tokens/s. Its loss margin was +0.0025,
free margin -0.4 percentage points, and throughput only 30.8% of the baseline.
The population still received 60.6% of the learned mix, 99.8% of units were
active, and utility gates remained near 0.55-0.58 despite 809 explicit probes.
The 640.0 MB schedule cache and both 16M/64M rejected checkpoints were deleted;
compact reports retain the evidence. The 64M report is
`reports/language_scaling/distributed-organism-compiled-scaling-64m-20260710.json`.

Decision: `retire_organism_v1_design_sparse_event_memory_v2`.

The active checkpoint remains the 251,658,240-token mixed Transformer. The final
corrected integrated PMRM screen trained six fresh matched arms for 269,568
identical scheduled tokens. Every full-memory parameter received a gradient;
surprise, random, and recency each made 576 permanent writes and 13,824 reads.
The Transformer reached general loss 7.4972 at 71,961 training tokens/s and
2.41 GiB peak allocated VRAM. PMRM losses were 7.6701 surprise, 7.6600 random,
7.6571 recency, 7.6999 without memory, and 7.6557 temporal-only. Full PMRM used
about 9.45 GiB and 2,777-2,830 tokens/s. All arms produced 0% exact free
relation answers.

The experiment refutes surprise as the best selector and provides no useful
margin for the full PMRM stack over temporal-only. The Transformer remains both
better and about 26 times faster. No checkpoint was retained; the PMRM model,
runner, and tests are deleted. The compact local report is
`reports/language_scaling/pmrm-integrated-trainable-memory-screening-262k-20260710.json`.

## Retired Language Concepts

The following are not maintained language paths:

- selective-spiking or dense-spiking language recurrence;
- routed language columns/experts;
- dense GRU language state;
- sampled or padded vocabulary training;
- language memory slots from the recurrent checkpoint lineage;
- recurrent-gradient horizons;
- route-bank, column-split, expert, synapse-bundle, or memory-slot structural
  transactions;
- quality-repair sweeps that optimize old prompt gates without solving unseen
  continuation;
- old SNN language readout ledgers as a generation architecture;
- frozen residual output adapters for relation binding.
- prompt-text episodic retrieval by prepending selected episodes.
- answer-masked relation post-training.
- integrated PMRM fixed columns, dual state, episodic selector, and recurrent
  workspace as a base-language architecture.
- token surprise as an assumed memory-utility signal.
- editable delta-memory v1 as a base-language architecture.
- distributed predictive organism v1 as a base-language architecture.
- fixed-stable gated multiscale dynamical memory v7 as a language sidecar.
- static depth-allocated Transformer v8 as a durable base architecture.

Historical reports may mention these terms. New code, status, and documentation
must not present them as active capability.

## Decision Order

1. Clean and validate the Transformer-only runtime.
2. Select 21M as the current compute-optimal size from equal-time evidence.
3. Pass the 21M TinyStories coherence falsification.
4. Retire answer-masked post-training after it preserves scalar loss but fails
   strict free relation generation.
5. Retire integrated PMRM after the corrected equal-budget screen shows no
   base-language advantage and surprise loses to naive selectors.
6. Scale the 2-delta/2-attention hybrid after its early win, then retire it when
   the win reverses at 16.78M and unseen semantic generation fails.
7. Build and scale one parallel, multi-timescale distributed predictive
   candidate, then retire it when the loss/free advantage disappears at 64M and
   dense event computation remains about three times slower.
8. Preserve full exact-stream capacity in v2 and require event-memory residuals
   to earn sparse activation from counterfactual utility and a compute budget.
9. Use LCWM-style execution-coupled selection only after a base model survives;
   do not make typed synthetic machinery the token mixer.
10. Retire v3-v7 after matched controls show that duplicated language cells,
    associative workspaces, hyperspherical constraints, and fixed-stable memory
    sidecars do not beat the maintained Transformer.
11. Retire static depth allocation v8 after its two replicated 16.79M wins
    reverse at 67.11M; retain the budget-sensitive optimization insight without
    promoting the architecture.
12. Continue only non-dominated arms through successive halving, then fit the
    first defensible local scaling law only for architectures that
    survive the pilot, using repeated seeds near a branch boundary.
13. Rebuild continual learning, exact resume, and retention measurement.
14. Re-establish sustained 524,288-token generation from the same checkpoint.
15. Add grounded causal experiments, then scale, redesign, or retire.
