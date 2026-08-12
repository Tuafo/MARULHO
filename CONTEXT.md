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
