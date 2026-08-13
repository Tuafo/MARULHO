# Training

This package owns MARULHO model execution, optimization machinery, and
checkpoint serialization. The installed language runtime is Transformer-only;
replacement candidates must use the same causal-language protocol and earn
promotion through matched evidence.

## Active Language Modules

**`language_transformer.py`** — MARULHO's decoder-only causal Transformer
state block. It owns RMSNorm, rotary positions, causal attention, SwiGLU, and
bounded per-layer KV state for incremental decoding.

**`language_model.py`** — owns full-vocabulary logits, bounded decode controls,
preallocated batched generation, exact model-state hashing, and the atomic v2
checkpoint. Decode controls inspect no more history than the active Transformer
context and only generated continuation tokens are eligible for penalties;
source prompt tokens may be reused in grounded answers. Sustained output growth
does not cause repeated full-history tensor concatenation or scanning.

**`language_protocol.py`** — the shared causal-language model interface used by
matched evaluations. The active Transformer and every replacement candidate
meet this seam; service/runtime installation remains a separate promotion
decision.

V49's frozen-base final residual sidecar is retired. It preserved inactive V39
behavior bit-exactly and trained efficiently, but active grounding fell to 1/64.
The model and checkpoint surface are deleted; no conditional-adapter option or
compatibility import remains.

V50's hierarchical conditional low-rank path is retired. It preserved inactive
V39 behavior exactly and improved grounding over the final sidecar, but reached
only 5/64 versus V48's 14/64. The model and checkpoint surface are deleted; no
conditional LoRA option or compatibility import remains.

V51's full specialist fork is retired. A complete V39 copy reached 12/64 with a
real source-control gain, but still lost to V48 while collapsing training loss
and worsening its general holdout to 5.1532. A later audit found that only 80/512
stream-packed grounding records retained the full prompt and answer in one
window, so the result rejects full copying for that pipeline rather than adapter
capacity in general. No specialist model, checkpoint, runner, test, routing
option, or compatibility path remains.

Distributed predictive organism v1 is retired. It beat the matched Transformer
at 4.20M and 16.79M tokens, but failed source-absent semantic generation and lost
both loss and free-relation advantages at 67.11M. Its final throughput was 33,963
tokens/s versus 110,345, while 99.8% of units remained active. The model,
checkpoint surface, runner, audit, tests, and rejected checkpoints are deleted;
no compatibility import remains.

Sparse event-memory v2 is retired. At 16.79M matched tokens, random one-of-four
specialists reached 27.0% strict free relation versus 14.5% exact-only at tied
loss. Chosen-expert utility reached 14.8%; all-expert comparative utility restored
25.8% but still did not beat random and slightly worsened loss. The model,
runner, and tests are deleted.

Modular predictive society v3 is retired. Four independent two-layer language
cells consumed 21,000,608 parameters, but their real-message arm reached 5.1073
heldout loss and 0% strict free relation versus the monolith's 4.6140 and 14.5%.
It also lost to no-message and shuffled controls. The model, runner, and tests
are deleted. The next candidate must share the vocabulary interface, preserve
full-gradient depth, and test communication between internal latent cells rather
than duplicate full language models.

The modular workspace line is retired from the live tree. V4's transient mean
raised strict free relation behavior to 21.5% versus 11.7% shuffled and 10.2%
without exchange, but loss stayed tied near 4.85 and behind the monolith. V5's
selective content-addressed workspace then fell to 6.6% versus 22.7% shuffled
and 24.6% without exchange, again near 4.85 loss. The model, runner, exports, and
tests are deleted. No Hopfield or column language compatibility path remains.

The integrated PMRM reference, runner, and tests were deleted after the final
corrected screen. Full PMRM remained behind the matched Transformer and did not
meaningfully beat temporal-only despite higher state, compute, and memory cost.
Its surprise selector also lost to random and recency under identical write and
read budgets. No PMRM compatibility code remains.

The editable delta-memory v1 model and the later V64 delta-state cortex are both
deleted after durable falsification. Delta v1 lost its early small-scale
advantage by 16.78M tokens. V64 then tested a much stronger 100,202,970-parameter
matrix-state/local-attention design with an exact MARULHO-owned Triton backward
and complete CUDA Graph optimizer step. Its final graph matched eager loss,
clipped gradient norm, and every updated parameter exactly, but reached only
9.24k positions/s versus 21.03k for the fresh matched Transformer. The 43.93%
ratio failed the frozen 50% admission floor at 8.70 GB, so terminal language
training never started and no V64 checkpoint exists. Its model, kernels,
one-shot evaluators, and tests are deleted; `RESEARCH.md` and the retained JSON
reports own the evidence. A successor must change the parallel training
computation, not preserve a delta compatibility surface or add another launch
wrapper around token-sequential recurrence.

V65 tests that requirement directly and also stops before a model exists. Its
exact chunk factorization and owned Triton coordinate backward pass all parity
gates, but the best chunk-64 operator reaches 249.7k positions/s—83.24% of the
frozen 300k admission floor. CUDA Graph does not change speed. The V65 reference,
kernels, runner, and tests are deleted; only the compact report and research
conclusion remain. No delta/editable-state runtime is installed.

V66's causal micro-macro exchange is also stopped before model construction.
Its two local attention passes plus compressed global summary pass are causal
and fully differentiable. The core becomes 6.57% faster than full attention at
context 1,024, but reaches only 55.55% of control throughput at context 320 and
uses 37.55% more peak allocation at 1,024. The reference, evaluator, partial
artifacts, and tests are deleted. V67 may test queried summary extraction with
only one local token pass; no micro-macro model or checkpoint surface exists.

V67 queried-summary exchange is stopped as well. It clears both speed gates at
73.91% of control throughput at context 320 and 136.82% at context 1,024, but
uses 27.94% more peak allocation in the long-context arm. Its implementation,
evaluator, tests, and partial artifacts are deleted. V68 may isolate native
block-major layout; no queried-summary model or checkpoint surface exists.

V68 proves that native block-major layout removes real copies but stops before
model construction. It saves 84.5 MB versus V67 and reaches 152.81% of full
attention throughput at context 1,024, yet still uses 11.18% more peak allocation
and reaches only 69.13% of control at context 320. Its implementation, evaluator,
tests, and partial artifacts are deleted. No block-native language surface
exists.

V69 tests macro conditioning inside a complete projected attention block. It
passes causality, common-weight hash, gradient, and both speed gates: 91.53% of
control at context 320 and 127.25% at context 1,024. It stops only because 703 MB
peak allocation remains 7.81% above the long-context FlashAttention control.
Its implementation, evaluator, tests, and partial artifacts are deleted. The
result justifies a separately preregistered model-level quality falsifier, not a
V69 model or checkpoint surface.

V70's training-only macro cortex is retired. Its 100,733,184 parameters retain
94.61% of matched Transformer throughput and complete gradients, but after 512
unique-data updates its heldout loss is 5.7188 versus control's 5.5000. The model,
preflight, quality runner, tests, and partial arms are deleted. No incremental
generation, checkpoint, compatibility, or runtime surface exists.

V71's periodic hierarchy is retired. Periodic-local nearly recovers Transformer
quality at 5.5625 versus 5.5000 while running slightly faster, but does not win.
Adding the macro channel worsens loss to 5.6875. The model, runners, tests, and
partials are deleted; no generation, checkpoint, or runtime surface exists.

`language_persistent_workspace.py` is V72's temporary Stage-A1 mechanism
falsifier. A complete small causal token path reads and writes eight latent
tokens across three synthetic document segments. Persistent, reset,
wrong-document, and document-anonymous batch-mean arms execute identical
modules; only state identity and lifetime differ. The state is detached at
segment boundaries, and local reconstruction trains writes without leaking the
later query answer backward. This is training-only machinery, not a language
runtime or checkpoint surface. It clears all three frozen seeds at 100% versus
6.23%--7.15% controls with exact mechanical contracts, so it remains only while
V72 undergoes its real sequential-language falsifier. A failure there deletes
the candidate and this A1 machinery together.

**`language_model.py`** — the language model contract. It owns:

- `LanguageModelConfig`;
- token embeddings and tied full-vocabulary LM head;
- full-vocabulary next-token cross-entropy;
- greedy generation with repetition and no-repeat controls;
- tensor-indexed stratified fixed-window train/eval splits whose contract hashes
  are computed on CPU before one-way device transfer;
- exact pre-window text-token counts emitted by the split builder, avoiding a
  second full-corpus tokenizer pass in experiment reports;
- explicit evaluation-only splits that do not tokenize or pack a discarded
  training source, while preserving evaluation windows and split hashes;
- chunked host-to-device split transfer whose batch views share large tensor
  storage instead of creating thousands of tiny CUDA allocations;
- a versioned row-major selected-window hash that is independent of batch and
  transfer chunk boundaries;
- CPU-owned immutable split tensors with only the active batch transferred to
  the model device during training or evaluation;
- heldout loss and perplexity;
- atomic Transformer checkpoint save/load.

Generation supports greedy argmax and seeded temperature/top-p nucleus
sampling over the full checkpoint vocabulary. Both use the bounded per-layer
KV state and the same repetition controls; every result reports its exact
policy, temperature, top-p threshold, and seed.

Maintained training and scaling runners support opt-in full-graph Inductor on
CUDA. Compilation is admitted only after an eager/compiled loss check, restores
RNG state before real updates, and reports one-time compile cost separately from
steady training. On Windows, the backend explicitly records and applies the
Triton 3.7 cache-key compatibility alias when PyTorch still expects the old
module location. Eager remains the default for short experiments.

The retired v6 hyperspherical candidate never became an installed or checkpoint
format. Its best normalized arm reached loss 4.7092 / 0% strict free relation,
behind the frozen Transformer's 4.6144 / 14.8%. The failed model is deleted. Its
useful systems result remains maintained: compiled post-step projection removed
the eager slowdown, and the generic Windows Inductor compatibility and
compile-amortized reporting stay available for future candidates.

Gated dynamical memory v7 is retired. The 20.977M candidate kept all four
attention layers and compared memory-off, single-scale, always-write,
fixed-random-write, and learned multiscale modes from exact resets. At 16.79M
tokens, the Transformer reached loss 4.6137 / 21.5% strict free relation. The
learned memory reached 4.6066 / 4.7% and did not beat single-scale's 4.6061 /
10.5%. Its gate remained active, all memory parameters received gradients, and
control throughput was matched, so the result is not a dead-memory or compute
imbalance artifact. Candidate training was also 12.7% slower than the
Transformer. No checkpoint was saved; the failed model, runner, exports, and
tests are deleted. The grouped-convolution recurrence was an effective execution
technique, but it did not earn a maintained language architecture.

Static depth allocation v8 is retired. Uniform, early-heavy, and late-heavy
profiles held total MLP width and all 20,976,128 parameters fixed. Early-heavy
improved loss by 0.0224/0.0182 and strict free relation by 18.4/21.9 points in two
independent 16.79M-token screens, but the advantage reversed at 67.11M: uniform/
early-heavy loss was 3.8861/3.8957 and free relation tied at 20.3%. The durable
arms ran within 0.30% throughput and passed initialization, gradient, memory, and
parity audits. No checkpoint was saved; the failed core, runner, and tests are
deleted. Static layer width is not a maintained language option.

Depth-weighted representation reuse v9 is retired. Across two independent
16.79M-token comparisons, learned-unconstrained connections replicated a small
loss improvement but not a reliable free-generation improvement or a joint win
over identity and fixed controls. Fixed-mean did not replicate its first strong
loss gain, fixed-random hurt loss, and learned-simplex remained near identity.
The core, runner, and tests are deleted; no depth-connection option exists in the
maintained training surface.

The rejected V10 product-key router has no maintained module or checkpoint
surface. Its two compact reports retain the useful result: fixed token hashing
replicated a loss gain while learned routing collapsed its pool usage and did
not improve loss. V11 owns the surviving mechanism directly.

`language_hashed_micro_experts.py` is the active uninstalled v11 successor. It
removes V10's query projection, product keys, top-k search, and failed routing
modes while retaining the shared 1024-wide SwiGLU path and 16,384 singleton
functions. Four deterministic token-hash heads select two functions each. The
model stores 36,180,480 parameters; its 1,581,056 theoretical replacement-path
multiplies per token are 50.26% of the dense MLP before gather overhead. The
shared-only and token-hash modes reuse one graph. Exact tensor transfer proves
the hash path is functionally equivalent to V10's winning control. This remains
an uninstalled experimental path pending unseen-generation qualification. Its CUDA/Inductor smoke compiles
the pruned candidate in 22.8s, peaks at 1.70 GB, and measures 124.2k token-hash
tokens/s; the two-step quality values are discarded. The 67.11M-token run passes
both controls at loss 3.8747 / 35.9% strict free relation and advances the model
to checkpoint qualification. An independent exact-recipe checkpoint run reaches
loss 3.8738 / 30.9% and retains the same fixed joint margins.

The experimental checkpoint surface is
`marulho_hashed_micro_expert_language_checkpoint.v1`. It owns the exact V11
configuration, strict tensor state, tied embeddings, tokenizer state and hash,
ownership flags, and qualification metadata. Atomic save and strict load reject
wrong surfaces, tokenizer mismatches, shared-only mode, missing tensors, and
untied restoration. The qualified local artifact is
`reports/language_scaling/hashed-micro-v11-qualified-seed2026-67m-20260711.pt`,
154.3 MiB with SHA-256
`6303ba4beabe49e163d4b8842ff798bc89215780c3ba269404895d1249f4b81b`.
A fresh strict load restores 36,180,480 parameters, token-hash mode, tied
weights, the 8,192-token vocabulary and tokenizer hash, and ownership metadata.
The installed Transformer loader remains separate until V11 passes unseen
generation.

The V11 general-continuation runner creates a new strict model/tokenizer
checkpoint only after a predeclared heldout-loss gain. It starts a fresh AdamW
and cosine phase from the exact qualified model; this fact is recorded, and
optimizer state is not persisted or claimed to resume. The resulting artifact
is an unseen-generation candidate, not a quality-promoted or runtime checkpoint.
Large runs can retain the exact schedule order/hash in indexed-host mode: each
sampled full batch is stored once on host and transferred only when selected,
instead of materializing the expanded schedule on CUDA. Expanded-device mode
remains available for exact historical recipes.

The current research candidate contains exactly 1,000,001,664 cumulative update
tokens at context 256 and heldout loss 3.0805. It is
`reports/language_scaling/hashed-micro-v11-indexed-continuation-1b-candidate-20260711.pt`,
154.3 MiB with SHA-256
`9e98a5f517f6f93f8d89544979990be8849ab4d03b2c206a98483ca3b3b68d64`.
Strict reload restores all 36,180,480 parameters, tokenizer identity, tied
weights, token-hash mode, context, parent/schedule hashes, and ownership. The
artifact remains uninstalled: controlled generation is readable but generic,
and all eight anchored source cases still fail grounding.

The V13 future-prediction trainer is retired and deleted. Three temporary
2/4/8-token heads learned their auxiliary losses, but the stripped inference
model regressed to 4.9522 heldout loss versus the matched control's 3.3243.
Attachment/removal parity was exact and no checkpoint was saved, ruling out an
inference-surface explanation. No future-head training or compatibility path
remains.

The V14 segment-associative state is retired and deleted. Its exact-reset
67.11M-token arms finish at heldout loss 3.0746086/off, 3.0745938/local,
3.0746429/ungated delta, and 3.0746036/gated delta. The learned gate receives
complete parameter gradients and the memory reaches full matrix rank, but mean
write falls to 0.082, no write exceeds 0.5, and its advantage over off is only
0.0000050. No checkpoint exists and no V14 model, loader, compatibility surface,
or tests remain. The retained report identifies the rejected mechanism without
keeping dead training code.

The V17 grouped-recurrent state is retired and deleted. Eight independent
32-wide GRUs remain tied with exact V11/off, their equal-parameter token-local
control, and a larger dense 256-wide GRU after 33.56M tokens per arm. The state
is active, full-rank, label-free, and fully trained, but does not improve
heldout language loss and costs about 20% throughput. No grouped-recurrent model,
runner, checkpoint, loader, compatibility surface, or partial-compile exception
remains. Small recurrent banks are not a maintained language option.

`forward_with_forced_expert_ids(...)` is a read-only V11 audit surface. It
requires explicit `[batch,time,head,slot]` pool indices and is not used by normal
training, generation, checkpoint loading, or runtime. Forcing the installed hash
is exactly logit-identical; counterfactual reports must prove parameter hashes
unchanged and keep target labels out of route construction.

The counterfactual utility-gate candidate is retired after both linear and MLP
predictors worsen disjoint heldout loss. No gate checkpoint exists and no gate
loader or runtime path is maintained. The frozen route-regret audit remains a
diagnostic surface only.

The separate evidence-reader line is retired. V26's final-layer reader cannot
use oracle evidence, and V27's reader after V11 blocks zero and two makes both
lexical and oracle loss about 0.0392 worse than gate-zero while raw context gains
0.0426. Both V27 gates and every reader/cortex tensor receive gradients, so this
is not dead machinery. `language_evidence_reader.py`, its screen, and their
tests are deleted; no reader checkpoint or runtime surface exists. The retained
reports preserve the exact parity, ownership, anti-cheat, and failure evidence.

The V28 particle-field training path is deleted. Its 20.972M-parameter positive
recurrent field passed causal, recurrent, gradient, generation, and compile
truth but lost the matched 16.78M-token language comparison: loss 4.9132 versus
4.3193, exact free generation 11.33% versus 40.23%, 11.1k versus 92.6k training
tokens/s, and 5.36 GB versus 0.60 GB peak CUDA memory. No particle checkpoint,
loader, or runtime state is maintained; the retained report and git history own
the evidence.

**`language_muon.py`** — owns the active uninstalled V29 optimizer candidate.
It applies 0.95 Nesterov momentum and five bfloat16 Newton-Schulz iterations to
shape-grouped hidden-matrix gradients, scales each update to the published 0.2
RMS target, and uses an AdamW fallback for the tied embedding and one-dimensional
norm parameters. The 20.976M control assigns 16,777,216 parameters to Muon and
4,198,912 to AdamW. No external weights or optimizer package are loaded.
`language_matched_support.py` accepts an explicit optimizer builder and records
the optimizer recipe and tensor-state bytes; existing callers still use fused
AdamW. At the durable 16.78M-token budget, 1e-3 Muon beats same-rate AdamW on
loss 4.0961/4.2606 and exact free generation 17.58%/5.47%, while using 40% less
optimizer state and training 42% slower. This passes only into exact checkpoint
reproduction and unseen review; the installed trainer still owns AdamW. The
reproduction checkpoint intentionally stores model/tokenizer/qualification
truth but not Muon state, so it cannot claim resumable optimizer continuity.
That strict 100.9 MB artifact reloads every tensor and sample logit bit-exactly,
but its 0/8 unseen source result blocks base-quality and runtime promotion.

V30 reuses the same Muon implementation without installing it. Fresh
general-only arms train at context 72/batch 32 and context 256/batch 9, giving
both 2,304 tokens per optimizer step, 7,282 steps, identical parameters and
initial tensors, and zero relation updates. Context length is configuration, not
additional capacity. The strict V29 model is evaluation-only and does not seed
either candidate's weights. General72 wins common loss 4.0093 versus 4.0258 for
general256 and 4.0955 for V29. Its strict checkpoint remains 0/8 on unseen
sources, so the recipe advances to a unique-data 67M scale point rather than
runtime installation. V31 keeps the same model and optimizer while consuming
29,128 non-repeated batches selected across 256 MiB from each full replay-shard
span. It reaches heldout loss/perplexity 3.6291/37.68 versus V30's 4.0093/55.11
at 56.1k tokens/s, with complete gradients and exact checkpoint reload. Unseen
loss improves but anchored generation remains 0/8 and unstable. It is a
retained data/optimization scaling point, not a new architecture or installed
base model.

V32 uses the same model and Muon code for a fresh 201,323,520-token run. The
general data preparer now builds five source splits sequentially, limiting raw
host-memory residency, and schedules 17,476 non-repeated batches from each
source. This changes data scale only; V31 remains the evaluation baseline and
does not seed candidate weights. V32 reaches loss 3.4983 versus V31's 3.6291,
but the 0.1308 gain misses its frozen 0.20 gate. No V32 checkpoint exists; the
fixed 21M training path does not receive another data-only scale point.

V34 reuses the maintained Transformer/Muon machinery at a different capacity:
width 768, ten layers, twelve heads, and 100,679,424 parameters. The tokenizer,
context 72, 67.11M-token source-balanced schedule, optimizer recipe, and holdout
remain V31-compatible, but the larger model is freshly initialized and does not
pretend to share an initial-state hash with V31. This is an uninstalled semantic
substrate test; a checkpoint exists only after a 0.20 heldout-loss gain and exact
reload.
The full shape is locally feasible: a disposable Inductor preflight passes
compiled/eager BF16 parity, trains at 11.3k tokens/s, and peaks at 3.32 GB on the
RTX 3060. Those two steps carry no quality meaning and are not retained.

The durable V34 state reaches heldout loss 3.3902 after 67.11M unique updates,
passes its +0.20 gate over V31, and strict-reloads from a 428.1 MB checkpoint.
It produces substantially more coherent unseen prose but remains 0/8 grounded.
V35 continues this exact model on three non-overlapping shards with fresh Muon
state at 3e-4; optimizer state is deliberately not claimed to resume. Its first
run is invalid because the token budget leaves two of 58,257 prepared batches
unused, despite a diagnostic loss gain from 3.3902 to 3.1654. V35R restarts from
V34 under a hash-pinned, override-locked manifest that consumes all batches; the
quality gate and training recipe remain unchanged. The valid V35R run reaches
loss 3.1649 after 134.22M new tokens, passes its +0.15 gate by 0.0753, and
strict-reloads the resulting checkpoint. Its 10.65k tokens/s and 3.330 GB peak
allocation define the exact step to optimize; throughput changes require matched
numerical and short-run quality evidence before entering another durable stage.

The optimizer now exposes an opt-in `per_head_attention_qkv` research path.
It keeps the combined QKV parameter and full-size momentum buffer unchanged, but
views the momentum update as independent Q/K/V head matrices during
Newton--Schulz orthogonalization. This is MARULHO-owned code derived from the
mechanism described in the Kimi K3 report; no external optimizer or weights are
loaded. On the exact V35R model shape, a 20-step optimizer-only CUDA benchmark
reduces measured optimizer time from 117.02 to 104.91 ms/step. This is not yet a
training-quality or end-to-end throughput result, so the default remains the
unpartitioned V35R optimizer until the token-matched screen passes.

V36 tests that path together with larger physical batches without changing the
ordered data or total token budget. `run_matched_training_arm` can concatenate
consecutive immutable schedule entries into one optimizer step while retaining
source-microbatch accounting. The candidate recipe is research-only until its
heldout loss and measured end-to-end throughput clear the preregistered gates.

The screen advances grouped physical batch 256 with whole-QKV Muon and the
unchanged 3e-4 learning rate: 25.07k versus 11.08k tokens/s and heldout loss
3.1423 versus 3.2455. Per-head Muon gains 7.27% at batch 32 within its loss gate,
but at batch 256 its same-rate gain is only 1.76% with slightly worse loss. It
remains an opt-in measured path, not the next durable default.

The deleted V37 depth-assembly candidate was bit-exact to the ordinary residual
chain before training, but retaining and combining every full-width depth state
exceeded the bounded one-hour screen and reached 11.74/12.29 GiB observed device
allocation. No quality result or checkpoint was produced. Do not restore the
module; a future depth mechanism must use fused low-rank or few-channel compute.

V38 confirms that 50/50 replay preserves and slightly improves old-language
loss while learning the relation holdout, but strict free accuracy stops at
46.88% against the 50% gate. No checkpoint is serialized. Recognition is 100%,
so V39 targets answer-bearing token learning or structured lexical realization
instead of more replay, capacity, or candidate scoring.

**`language_answer_objective.py`** — V39's retained continual-learning objective
detects answer spans directly in tokenizer ID space and moderately reweights
them while preserving ordinary causal loss elsewhere. The selected 4x objective
crosses the joint free-generation/retention gate and saves a standard exact-
reload Transformer checkpoint. It is a domain-training tool, not the universal
base-language default. Document-aligned supervised batches may pass a pad ID;
post-EOS pad targets then receive zero weight while every real token retains the
same objective.

V53's frozen-cortex source pointer is retired and deleted. Its 99,073 parameters
preserved V39 exactly and reached 17/64 with a real source-control gain, but
missed the 18/64 gate and fell two cases below V52. No pointer model, checkpoint,
loader, runner, tests, routing option, or compatibility path remains.

V54's trainable source encoder is retired and deleted. Its 373,506 parameters
train at 173.97k padded positions/s with exact parent isolation and checkpoint
reload, but direct span copying reaches only 16/64 versus zero for both controls.
The generic tokenizer offset contract remains for future auxiliary labels; no
span encoder, loader, runner, test, checkpoint, or compatibility surface remains.

V55's multi-view autoregressive transducer is retired and deleted. Fusion raises
heldout grounding from the causal-only ablation's 16/64 to 20/64, but misses the
32/64 capability bar and often assembles noncontiguous BPE pieces into malformed
answers. No model, loader, checkpoint, cache, test, runner, or compatibility
surface remains.

V56's frozen-parent landmark retrofit is retired and deleted. Its retriever
reaches only 71.09% top-two answer coverage, while decisive oracle-evidence
generation remains 0/128 despite falling losses, complete gradients, exact
parent fidelity, and strict compact reload. The result rejects a small residual
cross-attention adapter as the language-realization interface. No model, loader,
checkpoint surface, cache, runner, test, or compatibility import remains.

V57's full-model native-context continuation is retired.
The exact V39 tensors are loaded into a 320-token rotary context with unchanged
parameter count, then all layers train. Full-source and oracle-localized arms use
the same 320-token padded shape and 20,971,520-position mixed grounding/general/
relation schedule. Eager batch 32 is selected by a real Muon/AdamW full-step
preflight. Inductor is rejected after parity failure and a collapse to 328.5
positions/s versus eager's 16.1k. The valid terminal arms sustain 16.77k/17.03k
positions/s, but oracle/native exact generation reaches only 122/256 and 43/256
while both regress general retention. Context expansion plus unrestricted base
fine-tuning is therefore rejected. No V57 optimizer, checkpoint, runner, test,
or loading surface remains.

V58 is preregistered outside the installed language model as a protected
evidence-organ falsifier. It leaves V39 entirely outside the optimizer, clones
the checkpoint-trained embedding and ten-block body into a bidirectional
source/question encoder, and trains only that independent organ plus two exact
Unicode-character boundary scorers. Eight epochs equal 2,048 batch-32 steps and
20,971,520 padded positions.
No optimizer, loader, or checkpoint surface becomes production machinery unless
the organ first passes the frozen 192/256 capability and control gates.

V58 fails that gate and is deleted. Training is mechanically healthy at 24.06k
positions/s for 871.75 seconds, with all 68 tensors receiving gradients and V39
remaining exact, but title-disjoint extraction reaches only 20/256. No organ
checkpoint, optimizer, model, test, runner, loader, or compatibility path
survives. The next training experiment must test protected write-time learning,
not another supervised span-head capacity sweep.

V59 preregisters transient full-model AdamW as a capacity ceiling for
source-native write-time learning. Each case resets an independent BF16 V39
copy, performs four ordered context-72 epochs on ordinary source next-token
loss at 1e-4 with no weight decay, generates one question-only answer, and
discards the weights. The durable parent never enters the optimizer. This is a
mechanism screen for a later compact/meta-learned TTT module, not an admitted
training or checkpoint surface.

V59 completes mechanically but fails behavior. All source losses improve, all
full-model tensors receive gradients, resets are exact, and V39 remains
immutable, yet true and oracle source writes both produce 0/64 strict answers.
The runner, tests, transient model, optimizer, and any loading path are deleted;
no checkpoint is saved. Report SHA-256 is
`388c43f79c10cc306fc12b1f1d7ad245ba42c317e40d18007e11d357d18247f0`.
Do not restore raw source-only AdamW with a different epoch or learning rate;
the next inner learner must be meta-trained for downstream readout.

V60's meta-gradient linear write is terminally negative and deleted. Its
786,449 trainable parameters are 0.7811% of V39, all receive gradients, and
20,971,520 padded source positions train at 56,292 positions/s with 1.27 GB peak
CUDA allocation. Exact parent fidelity also passes. However, untrained, zero,
shuffled, true, and oracle memory all produce 0/256 strict answers and contain no
accepted answer. No checkpoint or live loading surface survives. Report SHA-256
is `76becda7f4d4986eb0bfca1056d2dd14f074c4d348bf5cf0f735c6125e9718fb`.
Do not restore the one-step linear matrix at a different width; the surviving
branch is iterative nonlinear fast learning with the source-only write boundary.

V61's nonlinear fast learner is terminally negative and deleted. Its 1.606M slow
parameters all receive gradients and train at 50,392 source positions/s, but the
second inner step diverges from 18.28 to 5,640.99 reconstruction loss. Untrained,
no-write, shuffled, true, and oracle states all score 0/256 exact, so no durable
state or checkpoint survives. Report SHA-256 is
`12d3cc8b3a1aa14937e68f8323607c9fb1322645b24aec4a3710c8a680b9c358`.
Do not restore the MLP with a smaller inner rate or extra width; the next isolated
variable is protected memory participation across multiple frozen V39 depths.

V62's three-depth compressed memory is terminally negative and deleted. Its
1.001M parameters all receive gradients, inactive hidden/logit/KV output remains
exact, and it trains at 44,303 positions/s with 1.46 GB peak allocation. But
inactive, shuffled, true, and oracle-short score 0/1/1/1 of 256; correct source
does not beat wrong source. No checkpoint survives. Report SHA-256 is
`7742199d52ed13c11cf20816fc4e593500dec7ee99486fd41f44bf416cf5e5b1`.
Do not restore the matrix with FP32 gates, wider keys, or another read site. The
next admissible memory must retain exact source-token KV state.

V63 exact-token adaptive KV memory is retired. Its sub-1% FP32 controller passed
all tokenizer-boundary, zero-parity, parent-fidelity, finite-state, timing, and
240/240 matrix-gradient checks. It processed 20.97M context-320 positions at
27.92k positions/s and reduced answer loss to 3.2313, yet question-only,
shuffled, true, and oracle scored 0/0/0/1 of 256. No checkpoint survives. The
runner/tests are deleted, and protected V39 memory adaptation is closed rather
than widened or replay-tuned.

V42's tokenizer-trie role-contrastive objective is deleted. It passed
mechanical parity and full-batch gradient checks, but the exact 32x8 eager pilot
ran for 16,507.6 seconds without persisting an arm result. No quality conclusion
or checkpoint exists. The shared experiment runner now times complete warmup
optimizer steps, rejects projected over-budget arms before counted training,
and can atomically persist exact-contract arm results. Do not restore the loss
without a new preregistration. The real V39 BF16/Muon runtime preflight selects
effective batch 224 at 19.22k training tokens/s and 10.49 GB peak allocation;
batch 256 is rejected after memory pressure collapses throughput. This validates
execution feasibility and exact restoration, not quality parity across batch
sizes.

The V33 editable-state hybrid training path is deleted. Its exact parallel
matrix recurrence, local-attention blocks, strict experimental checkpoint, and
tests were mechanically valid, but the matched 16.78M-token result was dominated:
loss 4.0056 versus Transformer 4.0082, 41.1k versus 54.3k tokens/s, and 1.030
versus 0.733 GB peak CUDA allocation. The 0.0025 gain missed the frozen 0.02
requirement, so no checkpoint or event controller was retained. The report and
git history own the evidence; do not restore the path under a new name.

**`checkpointing.py`** — the broader `MarulhoTrainer` checkpoint lifecycle
used by `MarulhoBrain`.

The active installed language path is `marulho_transformer`; the only accepted
runtime `state_core` value is `transformer`. Delta runtime state is experimental
and cannot be loaded through the active checkpoint loader.

## Checkpoint Contract

`marulho_transformer_language_checkpoint.v2` contains:

- exact `LanguageModelConfig`;
- strict model tensor state;
- complete byte or BPE tokenizer state;
- tokenizer vocabulary hash;
- metadata and ownership flags.

The scaling experiment stores optional training-continuation metadata inside
this atomic payload: optimizer/scaler state, cumulative token/step counts, RNG
state, and batch position. Inference loading does not depend on those fields.

The tokenizer vocabulary must exactly match the model vocabulary. Legacy
recurrent, routed, spiking, sampled-vocabulary, and padded-vocabulary
checkpoints are rejected rather than upgraded through compatibility code.

Checkpoint writes use a temporary file, flush and fsync the payload, then
atomically replace the target.

Retired candidate checkpoint surfaces are rejected, so an experiment cannot
silently replace the active brain model.

The retired `marulho_delta_language_checkpoint.v1` surface is rejected by the
active Transformer loader. No compatibility loader remains in the live tree.

## Runtime Boundaries

- Training code owns model tensors and optimization.
- `MarulhoBrain` owns installation, runtime lifecycle, and durable brain state.
- Evaluation runners may train isolated candidates and write reports.
- Service and UI code do not implement training or mutate model state on reads.
- External pretrained model weights are not part of the language path.

## Retired Language Machinery

The matched BPE pilot selected the Transformer over the dense GRU and earlier
spiking/routed candidates. The following language implementations were deleted:

- selective-spiking and dense-spiking recurrent cores;
- routed experts and route-bank dispatch;
- GRU production state;
- sampled/padded vocabulary training;
- language eligibility traces and recurrent memory slots;
- recurrent continual-learning repair;
- recurrent structural-plasticity transactions;
- their Triton kernels, evaluation runners, and tests.

SNN and column code elsewhere in MARULHO belongs to separate grounded
experiments and must not be reported as the language generator.

## Validation

The minimum focused suite is:

```powershell
python -m pytest -q `
  tests/test_language_transformer.py `
  tests/test_language_tokenizer.py `
  tests/test_language_training_experiment.py `
  tests/test_language_sustained_runtime_evidence.py `
  tests/test_marulho_brain.py
```

Passing tests validate contracts, not language quality. Quality requires a
real-corpus experiment with heldout curves and unseen generation.
