# Evaluation

This package owns experiments and evidence reports. It does not own cognition
or mutate an installed `MarulhoBrain` during status reads.

## Maintained Language Experiments

**`language_training_experiment.py`** — the primary Transformer training
runner. It owns corpus loading, byte/BPE tokenizer creation, stratified fixed
windows, AdamW, warmup plus cosine decay, precision selection, heldout
evaluation, unseen prompt generation, optional sustained generation, and
checkpoint/report output. CUDA runs may explicitly request full-graph Inductor;
the runner restores RNG state, requires compiled/eager warm-up loss parity, and
reports both steady and compile-amortized throughput. Eager remains the default
because compilation can lose on short runs.

**`language_scaling_experiment.py`** — trains a matched model-size grid over
shared token budgets, records heldout curves, throughput, peak VRAM, unseen
generations, OOM failures, and a provisional fitted `L(N,D)` law. It retains
only the best checkpoint by default. Repeated `--eval-corpus` arguments keep
multiple provenance-recorded holdout sources entirely outside tokenizer
training and gradient updates. Its separate
`empirical_wall_clock` budget basis applies recorded per-arm throughput
multipliers, reports unique and repeated updates, and rejects the comparison
when actual training times differ by more than 15 percent. Unequal-token runs
cannot be mislabeled as scaling-law grids. Repeated `--corpus` arguments train
one tokenizer and split over multiple provenance-recorded corpus shards. A
single arm is a maintained data-scaling curve and emits
`continue_data_scaling_at_selected_model_size`, not a fictitious size winner.
`--resume-checkpoint` continues exactly one MARULHO-owned arm with the
checkpoint tokenizer. Scaling checkpoints persist AdamW/scaler state,
cumulative updates, RNG state, and batch position; a pre-state checkpoint may
continue with fresh optimizer state, which the report marks explicitly. Each
continuation is a new cosine schedule phase. Corpus and checkpoint SHA-256
hashes are streamed rather than loaded into one large byte allocation. Optional
Inductor execution uses the same parity contract as the primary runner; wall
clock comparisons include compile time rather than comparing only steady-state
steps.

**`language_matched_support.py`** — shared mechanics for replacement falsifiers:
deterministic corpus range selection, full-batch filtering, mixed-source schedule
construction, one-time device staging, optimizer/gradient accounting, heldout and
label-safe relation evaluation, and optional post-arm diagnostics. An explicit
optimizer warmup may compile optimizer kernels, then restores the exact initial
weights, discards the warmed optimizer, rebuilds fresh state, resets RNG, and
counts none of those tokens as training. The warmup now times each complete
forward, backward, clipping, and optimizer step, projects full-arm wall time
from the late-step median, and can reject an over-budget arm before counted
training begins. Completed arm rows and optional model state use a strict
contract hash and atomic replacement, so a runner can reuse only an exact
completed arm after interruption. Architecture decisions remain owned by the
specific runner; this support cannot promote a model.

For supervised records that individually fit the causal context, the support can
instead build one right-padded row per document. Padding occurs only after EOS
and the report includes the counterfactual global-stride boundary count. This
mode is explicit; continuous language corpora retain ordinary stream windows.

The deleted V33 editable-state falsifier compared an exact 20,976,128-parameter
Transformer with alternating local-attention/editable-matrix state on the same
16,777,728 general-only tokens. It passed parameter, shared-initialization,
gradient, parity, and leakage audits, but loss 4.0056 only tied the control's
4.0082 while retaining 75.5% throughput and using 1.41 times peak memory.
Decision: `retire_v33_editable_state_no_heldout_language_win`. No checkpoint or
event phase exists; the compact report and git history retain the evidence.

**`language_geometry.py`** — generic read-only depth instrumentation for the
Transformer and compatible candidates. It captures bounded hidden samples after
the input projection and each block, then reports participation ratio, effective
rank, RMS, mean-vector norm, and adjacent-depth cosine. It restores model mode,
does not mutate state, and is never a promotion metric.

**`language_hashed_micro_expert_durability.py`** — the active uninstalled v11
runner. It compares the Transformer, pruned shared-only, and pruned token-hash at
a default 67.11M-token budget and independent seed. Shared-only and token-hash
reload one exact 36,180,480-parameter state and share one compiled graph; the
20,976,128-parameter Transformer is hash-matched on every unchanged tensor. The
hash arm must beat both controls by 0.005 heldout loss and 0.02 strict free
generation while retaining at least half the local throughput. A pass advances
to checkpoint fidelity and unseen-generation work, not direct runtime install.
The mechanism smoke compiles the candidate in 22.8s, peaks at 1.70 GB, and keeps
shared/hash throughput within 1.21%; its report and quality values are discarded.
The durable report is local at
`reports/language_scaling/hashed-micro-v11-durability-seed2026-67m-20260711.json`.
At 67,112,064 tokens, token-hash reaches loss 3.8747 / 35.9% strict free relation
versus Transformer 3.8951 / 19.1% and shared-only 3.9088 / 25.8%, retaining 96.0%
of Transformer throughput. Decision:
`promote_v11_hash_for_checkpoint_and_unseen_generation`.
The runner can then execute only `token_hash` with `--qualification-report` and
`--checkpoint-output`. This independent run must preserve the exact token count,
common initialization, configuration, schedule, and tokenizer, then re-pass the
unchanged joint loss/behavior/throughput gate against the qualified controls;
long BF16/Inductor trajectories and discontinuous greedy exact-match scores are
not required to be bit-identical. The checkpoint reproduction report is
`reports/language_scaling/hashed-micro-v11-checkpoint-reproduction-seed2026-67m-20260711.json`.
It reaches loss 3.8738 / 30.9% strict free relation and saves the 154.3 MiB
strict artifact at
`reports/language_scaling/hashed-micro-v11-qualified-seed2026-67m-20260711.pt`
with SHA-256
`6303ba4beabe49e163d4b8842ff798bc89215780c3ba269404895d1249f4b81b`.
A fresh strict load restores model, tokenizer, tied weights, and qualification
metadata. This admits unseen generation, not runtime installation.

The first strict-checkpoint unseen reports are local at
`reports/language_scaling/hashed-micro-v11-unseen-fineweb-greedy-20260711.json`,
`reports/language_scaling/hashed-micro-v11-unseen-cosmopedia-greedy-20260711.json`,
and
`reports/language_scaling/hashed-micro-v11-unseen-cosmopedia-controlled-20260711.json`.
FineWeb-Edu/Cosmopedia mean source loss is 4.3092/3.6194, and zero of eight
source-anchored cases pass. Repetition penalty 1.1 plus no-repeat 3 raises
Cosmopedia distinct-bigram fraction from 0.675 to 0.948 without changing source
loss or prefix agreement. Direct review finds grammatical multi-sentence but
generic and semantically unstable text. Decision:
`continue_v11_general_language_pretraining_before_runtime_or_memory`.

**`language_hashed_micro_expert_continuation.py`** — continues only the strict
durability-qualified V11 token-hash checkpoint. The default phase adds
184,550,400 general-language tokens, for 251,662,464 cumulative tokens, using
equal FineWeb-Edu/Cosmopedia alternation and exactly zero relation updates. It
records the parent SHA-256, tokenizer identity, before/after heldout and frozen
relation metrics, schedule/source ranges, exact token count, fresh optimizer and
cosine-phase status, throughput, and memory. A candidate checkpoint is saved
only after at least 0.10 heldout-loss improvement. That artifact remains
unpromoted and must pass the same unseen-generation suite; optimizer state is
not claimed to persist. A two-step CUDA smoke passed the full path and its report
was deleted because it has no quality value.

The full continuation report is
`reports/language_scaling/hashed-micro-v11-general-continuation-251m-20260711.json`.
It adds exactly 184,550,400 tokens, reaches 251,662,464 cumulative tokens, and
improves heldout loss 3.8709 to 3.4865 / perplexity 47.99 to 32.67 at 124.9k
tokens/s. The strict candidate is
`reports/language_scaling/hashed-micro-v11-general-continuation-251m-candidate-20260711.pt`
with SHA-256
`fbf874923ebce6f4d36497f52a622dc8e222e01672b60876c910941af3fc1894`.
Frozen relation candidate/free behavior falls 95.7% to 32.8% / 30.9% to 0% with
no replay. The repeated unseen reports improve prompt-local loss but remain 0/8;
V11 ties the local 251M Transformer on FineWeb loss and trails on Cosmopedia.
Decision: `retain_v11_checkpoint_redesign_token_only_routing_before_more_scaling`.

**`language_hashed_micro_expert_counterfactual.py`** — freezes the strict 251M
V11 checkpoint and changes only the final token's eight singleton-expert IDs in
each heldout 72-token sequence. The installed token hash and four deterministic
alternatives have identical active compute. Exact next-token loss scores each
completed route; labels never choose a prediction route, oracle selection is
metrics-only, parameters are hashed before/after, and forcing installed IDs must
reproduce baseline logits exactly. A result can admit training a causal V12 gate
only when mean oracle improvement is at least 0.02 and at least 10% of tokens
have regret of 0.05 or more. It cannot promote a model. The one-batch CUDA smoke
passed and was deleted before the full audit.

The full report is
`reports/language_scaling/hashed-micro-v11-counterfactual-route-audit-251m-20260711.json`.
Across 4,608 contexts, mean oracle loss improvement is 0.1911 and 40.5% have
regret ≥0.05. FineWeb-Edu/Cosmopedia gains are 0.2020/0.1802; fragile halves gain
0.3159/0.2963 versus 0.0882/0.0641 on confident halves. Every fixed alternative
is globally 0.62–0.66 worse, so the opportunity is conditional rather than a
better static seed. Forced parity is zero, parameters are unchanged, duplicates
are zero, and pool coverage is 95–97%. Decision:
`train_v12_counterfactual_gate`. This is permission to test whether a label-safe
gate can predict the oracle opportunity, not permission to use oracle routes.

The deleted V12 utility-gate trainer is retained only through
`reports/language_scaling/v12-counterfactual-utility-gate-251m-20260711.json`.
The linear gate underfits and worsens FineWeb-Edu/Cosmopedia loss by
0.0381/0.0334; the 64-wide MLP fits training (+0.1126) but reverses to -0.0757
combined heldout gain. Parent hashes remain unchanged and evaluation routes are
label-safe. Decision: `retire_v12_gate_cannot_predict_counterfactual_utility`.
No gate artifact exists; the failed trainer and tests are deleted.

The general-continuation runner also accepts its own strict saved candidate as a
qualified parent and exposes sequence length plus batch size. The strict 251M
checkpoint has a 72-token rotary context. A longer request explicitly rebuilds
the same learned state at the requested context and must preserve parameter count
and produce bit-exact logits on heldout 72-token prefixes before training. This
supports the independent long-context ablation: 256-token updates at batch 40
preserve roughly the same tokens per optimizer step as 72 x 144, while changing
the temporal training horizon rather than the sparse mechanism.

That ablation is complete at
`reports/language_scaling/hashed-micro-v11-long-context-318m-20260711.json`.
The parity gate is exact, 67,112,960 additional tokens reach 318,775,424
cumulative tokens, and 256-token heldout loss improves 4.2033 to 3.3243
(perplexity 66.91 to 27.78) at 123.6k tokens/s and 3.04 GB peak CUDA allocation.
The strict candidate is
`reports/language_scaling/hashed-micro-v11-long-context-318m-candidate-20260711.pt`
with SHA-256
`cebe5ac7b5a84da1208d61c61715f58f61aa91c1ae2211208d005ac3f99506ae`.
Its repeated FineWeb-Edu/Cosmopedia source losses improve only to 3.9951/3.3586,
and all eight anchored generation cases still fail. Greedy FineWeb diversity
falls to 0.571; controlled Cosmopedia decoding reaches 0.944 bigram diversity and
6.75 mean prefix characters but remains generic and topic-unstable. Decision:
`retain_long_context_infrastructure_reject_context_only_quality_explanation`.
The checkpoint remains an unpromoted matched control.

The deleted V13 future-prediction experiment is retained only at
`reports/language_scaling/v13-future-prediction-318m-20260711.json`. Its strict
parent, schedule, tokens, sequence, batch, seed, optimizer, and backend match the
318M next-token control. The +2/+4/+8 auxiliary losses improve, but ordinary
heldout loss after exact head removal is 4.9522 versus the control's 3.3243.
Decision: `retire_v13_future_prediction_no_control_gain`. No candidate checkpoint
or future-head code remains.

The maintained continuation runner supports `indexed_host` schedule storage for
large token budgets. The schedule tuple and SHA-256 are unchanged; sampled full
batches remain host-resident and only the active batch moves to CUDA. Reports
separate resident, expanded-equivalent, host, and device bytes. A disposable
100-update benchmark at 256 x 40 measures 121.8k tokens/s, 1.97 GB peak CUDA
allocation, and zero CUDA-resident schedule bytes. The retained expanded-device
control measures 123.6k tokens/s and 3.04 GB, so bounded scheduling costs about
1.4% throughput while removing 16 bytes of CUDA growth per requested token. The
benchmark report is deleted because it has no quality value.

Matched-data preparation builds the heldout set through an explicit
evaluation-only split. It no longer re-tokenizes and packs both full training
shards merely to discard that split. Evaluation token streams, selected windows,
and split hashes are exact against the prior paired builder.

The full indexed continuation report is
`reports/language_scaling/hashed-micro-v11-indexed-continuation-1b-20260711.json`.
It adds 681,226,240 tokens and reaches exactly 1,000,001,664 cumulative tokens.
Heldout loss improves 3.3243 to 3.0805 / perplexity 27.78 to 21.77 at 121.9k
tokens/s and 1.97 GB peak CUDA allocation. The 2.06 GB unique host pool replaces
a 10.90 GB expanded CUDA schedule. The strict candidate is
`reports/language_scaling/hashed-micro-v11-indexed-continuation-1b-candidate-20260711.pt`
with SHA-256
`9e98a5f517f6f93f8d89544979990be8849ab4d03b2c206a98483ca3b3b68d64`.
FineWeb-Edu/Cosmopedia source loss is 3.9678/3.1405. All eight cases still fail
source grounding; controlled Cosmopedia reaches 0.960 bigram diversity and
readable paragraphs but remains generic and can lose topic. Decision:
`retain_v11_1b_sparse_base_redesign_persistent_semantic_state`. This checkpoint
is the retained sparse baseline for the next architecture, not a runtime model.

The deleted V14 segment-associative falsifier is retained only at
`reports/language_scaling/v14-segment-associative-67m-20260711.json` (SHA-256
`ee20dbb54769845ec60e9564ddc2525c00d432c32b2edec36ab4626204111190`).
Each exact-reset arm receives 67,112,960 tokens. Off/local/ungated/gated loss is
3.0746086/3.0745938/3.0746429/3.0746036, relation ranking remains
48.4%/48.0%/48.4%/48.4%, and free relation generation remains 0%. The gated
memory receives complete gradients and reaches full matrix rank but learns to
suppress writing. Decision: `retire_v14_no_segment_state_gain`. No checkpoint,
model, runner, loader, or compatibility path remains.

The deleted V15 dyadic-memory preflight is retained only at
`reports/language_scaling/v15-dyadic-memory-preflight-20260711.json` (SHA-256
`7386a4093eeb3b20265ba3a3d4c86f50ffc6d4c6d5b89350cad75d91b05dab9f`).
Three frozen seeds compare the same 112-float state and exact input/readout
initialization on multi-query recall. Mean 128/256/512/overwrite accuracy is
6.25/5.97/6.42/6.16% for the 52,624-parameter flat GRU; 25.84/22.64/19.99/18.93%
for raw dyadic averages; 25.54/21.90/19.11/16.32% for random balanced contrasts;
and 25.44/21.70/18.84/17.12% for ordered Haar contrasts. Chance is 6.25% and the
metrics-only oracle is perfect.

Haar misses its three-point admission margin against raw and shuffled controls,
so no dyadic language model is admitted. Raw averages win every mean profile,
showing a narrower multiscale-bank signal. At length 512 their effective state
rank is 16.6 versus 18.9/Haar and 20.8/shuffled, so occupying more dimensions
does not explain quality. Decision:
`redesign_v15_retain_multiscale_clocks_reject_haar_ordering`. The preflight
model, runner, tests, and wavelet-specific path are deleted. The next synthetic
test must separate small-bank modularity, averaging, and nonuniform clocks before
any language run.

The deleted V16 clock-isolation preflight is retained only at
`reports/language_scaling/v16-multiscale-clock-isolation-20260711.json`
(SHA-256
`6a33f3b18e39b1b4ba301aa1a67e974cdf0898162100e6367b18f23ae791019d`).
Three fresh seeds use identical candidate tensors, 31,120 parameters, 112 state
floats, task, schedule, optimizer, and readout. Mean 128/256/512/overwrite
accuracy is 6.34/6.48/6.25/6.05% for the larger flat GRU;
38.34/38.29/37.87/22.68% for seven token-rate banks;
25.35/24.92/21.47/20.10% for seven uniform low-pass banks;
20.01/20.21/20.21/14.91% for dyadic last-token banks; and
25.23/22.72/19.11/18.65% for dyadic low-pass banks.

Token banks win every seed and retain almost exact accuracy from 128 to 512
tokens. They use fewer parameters than flat but perform 896 versus 128 recurrent
updates and run at 574k versus 1.35M task tokens/s, so this is an organization
win, not an efficiency win. Their 512-token effective rank is 23.3 versus 23.1
for flat, ruling out global rank as an explanation. Decision:
`redesign_v16_retain_small_banks_reject_clock_claim`. Clock, averaging, and
synthetic-runner code are deleted.

The deleted V17 grouped-recurrent screen is retained only at
`reports/language_scaling/v17-grouped-recurrent-33m-20260711.json` (SHA-256
`fe164f90b342759eb281e6c98e7696155082baea906e37e9809c6f8d35133d91`).
Exact V11/off, equal-parameter token-local, eight independent 32-wide GRUs, and
one dense 256-wide GRU each receive 33,556,480 general-language tokens from the
strict one-billion-token V11 parent. Their loss is respectively
3.0788569/3.0790505/3.0789700/3.0786710; relation accuracy is
45.3/45.3/46.9/45.7%. Grouped misses its 0.02 loss margin against every control
and is slightly worse than off and dense. Attachment is exact, compiled/eager
loss parity passes, active recurrent parameters receive complete gradients, and
the grouped state is full-rank with nonzero residuals. Grouped throughput is
97.8k tokens/s versus off's 121.9k. Decision:
`retire_v17_grouped_recurrence_no_language_gain`. No checkpoint exists, and the
model, runner, tests, and candidate-only partial-compile path are deleted.

The deleted V18a segment-bridge preflight is retained at
`reports/language_scaling/v18-segment-memory-800step-20260711.json` (SHA-256
`6141cf272002764c0ae52e5c894937e92fddb9199ac4b3aee01464dcf5f44c89`).
Exact history reaches 73.0% candidate/18.4% greedy exact, but source-independent
local slots already reach 72.7%/17.2%; streaming mean reaches 74.2%/19.5%.
Learned slots fall to 72.7%/4.7% and collapse to effective rank 2.01 despite
complete gradients. Decision: `retire_v18_frozen_segment_bridge_interface`.
The v2 runner is one bounded redesign: post-write LayerNorm plus paired
counterfactual episodes that hold the question, seven distractors, positions,
and decoding policy fixed while changing only the target source and answer.
This contains 47 query groups and 558 different-answer pairs. The original
report and gate remain unchanged.

The final V18b report is
`reports/language_scaling/v18b-segment-memory-counterfactual-800step-20260711.json`
(SHA-256
`7c8d330a76f7b421d3a0281fc8eb7a54ab5488d7e6297201785a5f74efed6e6d`).
Exact/local paired source-following accuracy is 25.33/17.90%, missing the
predeclared 10-point causal margin at 7.42 points. Normalized learned slots reach
only 3.93%, despite complete gradients; their norm is 23.8 but effective rank
remains 1.78. Candidate/free exact accuracy is 68.4/4.7% for learned versus
80.5/25.4% for exact history. Decision:
`retire_v18b_exact_history_no_source_causal_gain`. No checkpoint or throughput
claim exists. The model, runner, tests, feature-cache path, and smoke artifacts
are deleted.

The V19 jointly trained recurrent-token report is
`reports/language_scaling/v19-joint-memory-800step-20260711.json` (SHA-256
`ce73f309a84ab80a0a1faa1fb192bbdcc2b17abcba409a57eb7e44a44a56f7af`).
Off/exact/local/recency/mean/recurrent candidate accuracy is
67.6/87.1/66.4/76.6/82.0/84.0%; free exact is
15.6/49.6/15.2/25.0/28.5/30.1%; paired source-following is
16.59/47.60/16.59/24.45/29.69/30.13%. Recurrent state is bounded to 32 KiB,
receives complete nonzero gradients, reaches matrix rank 502, and keeps the two
general holdouts at +0.0660/+0.0790 loss. It nevertheless ties mean, misses exact
by 17.47 paired points, and changes output on only 7.17% of source swaps versus
23.48% for mean. Decision:
`retire_v19_joint_memory_tokens_insufficient_source_following`. No checkpoint
exists.

The final V19b report is
`reports/language_scaling/v19b-partitioned-memory-800step-20260711.json`
(SHA-256
`efb24df8e4c9fe1c1fe89a398ffcf753f2c03b730300a6181c2949303d417a73`).
Two independent eight-token segment banks retain the same 32-KiB budget and
reach 85.9/31.6/31.44% candidate/free/paired accuracy. That improves recurrence
by only 1.9/1.5/1.31 points and remains 16.16 paired points behind exact history.
The partition is active, rank 447, fully reached by gradient, and keeps both
general losses within +0.085. Decision:
`retire_v19b_partitioned_memory_insufficient_source_following`. No checkpoint
exists. The runner, model interface, tests, and smoke report are deleted; the two
compact full reports retain the evidence.

The next preflight audits exact episodic retrieval before training another
memory architecture. Source-only writes retain raw token spans plus small keys;
question-only reads retrieve a fixed number of spans into local exact attention.
Random, recency, lexical overlap, and frozen-cortex keys must be compared with
all-history and a metrics-only oracle. Retrieval recall is only an admission
gate; a surviving policy must later improve paired free generation under exact
model resets and bounded active tokens.

The V20 audit is retained at
`reports/language_scaling/v20-exact-episodic-retrieval-audit-20260711.json`
(SHA-256
`8436c6fdbb1976d75b22b4974c6acb5c1aa884702f0a5a579b2125a9697fc57d`).
Random/recency/lexical/frozen-last/frozen-mean recall at one is
27/34/71/38/41%. No fixed key clears the top-one gate, so the report decides
`redesign_v20_no_fixed_key_retrieves_exact_episode`. Lexical recall at two is
98.8% with 96 active source tokens versus all-history's 192. That diagnostic
admits a separate matched language screen; it does not revise the top-one result
or promote retrieval quality by itself.

The V21 language screen is retained at
`reports/language_scaling/v21-exact-episodic-retrieval-800step-20260711.json`
(SHA-256
`b2a60cc1e3c0a45ea65811238210c344d8d6f124773556952bc0fe41e3a4def1`).
Off/all-four/random-two/recency-two/lexical-one/lexical-two candidate accuracy is
68.8/87.9/79.3/79.3/89.5/100.0%; free exact is
16.0/39.5/27.7/25.0/44.9/51.6%; paired source-following is
17.0/38.0/27.5/24.9/45.4/52.0%. Lexical-two retrieves 96 source tokens, includes
the target 98.83% of the time, changes output on 82.62% of source swaps, and
preserves the two general holdouts at +0.0631/+0.0772 loss. It beats
all-history while halving active source context. Peak allocation falls from
1.03 to 0.90 GiB, but elapsed training time is tied, so no throughput win is
claimed. Decision: `advance_v21_exact_episodic_retrieval_to_contiguous_streams`.
No checkpoint or runtime promotion exists yet.

The next evidence must use causal, document-disjoint general streams rather than
relation groups. Writes retain prior exact spans and provenance; reads use only
the visible current prefix. Local-only, random, recency, and lexical arms must
be active-token matched where possible. A survivor needs both heldout
continuation improvement and better anchored free generation, followed by a
strict checkpoint containing cortex plus archive/index state.

`language_causal_document_retrieval_audit.py` preregisters the frozen-cortex V22
interface test before joint training. It samples 128 unique long documents from
each of the disjoint FineWeb-Edu and Cosmopedia eval sources. Every case writes
the first 48 tokens, leaves a 48-192-token unseen gap, exposes a later 48-token
prefix to retrieval, and scores only the following 16 tokens. The true older
episode is mixed with three same-corpus distractors whose write order is random;
future tokens and document identity are unavailable to lexical and frozen-V11
keys. Off, all-four, matched random-one/two and recency-one/two,
lexical-one/two, frozen-last-one/two, frozen-mean-one/two, and non-promotable
oracle-one share the same cases. A diagnostic smoke showed that one exact span
can help while an extra distractor can erase the gain, so the final audit treats
one versus two retrieved episodes as an explicit variable rather than assuming
top-two is universally correct.

V22 advances only if oracle-one improves paired loss by at least 0.005 with a
positive 4,096-sample bootstrap lower bound. A label-safe key must retrieve the
target at least 50% at top-one or 70% at top-two and beat the better matched
random/recency inclusion by 20 points. It must also improve local loss by 0.005,
beat both equal-token language controls by 0.0025, stay within 0.01 loss of
all-history, and avoid regressing either source. A pass admits a jointly trained
document screen; it does not promote language quality, checkpointing, runtime
installation, or speed.

The final V22 report is
`reports/language_scaling/v22-causal-document-retrieval-audit-20260711.json`
(SHA-256
`af1898ece04196ebad35adcdf5c89c56d13cc2a0419f5abe7f9fc4ee18c6ea10`).
Local-only loss is 3.0048. Oracle-one reaches 2.9707, a paired +0.0341 gain with
95% bootstrap interval +0.0116 to +0.0584, proving the prior exact span is
predictively useful. Lexical-one/frozen-mean-one target inclusion is
75.0/74.2%, but loss is only 3.0031/3.0021 and both intervals cross zero.
Lexical-two/frozen-mean-two inclusion rises to 84.8/89.1% while loss worsens to
3.0423/3.0311; all-four reaches 3.0355. Decision:
`redesign_v22_addressing_does_not_recover_useful_episode`.

Conditioning explains the failure: lexical-one gains 0.0372 when the target is
included and loses 0.1050 when it is absent. The metrics-only lexical margin
curve reaches 100.0% precision at 25% coverage and 95.3% at 50%, versus 75.0%
when always active. This curve uses target identity and is not a deployable
threshold. The next audit must calibrate retrieval or abstention on separate
replay documents, freeze the threshold, and evaluate once against equal-write-
rate controls on the disjoint V22 documents.

The deleted V22b confidence-gate audit sampled 256 causal cases from each replay
corpus, froze one lexical top-one-minus-top-two margin threshold, and then loaded
128 cases from each disjoint eval corpus. Calibration used same-document
identity but never the future continuation or language loss. Lexical, random,
recency, and non-promotable oracle selection shared the exact gate mask.

The retained report is
`reports/language_scaling/v22b-confidence-gated-document-retrieval-20260711.json`
(SHA-256
`189966d147b10a6ff1a5b003e86ced389f8b562e23635213b73101d510476aa8`).
Calibration freezes margin 0.055112 at 51.76% coverage and 95.09% precision. It
transfers to 54.30% coverage and 97.84% precision. Gated lexical improves loss
by +0.0356 with 95% interval +0.0180 to +0.0552; matched random/recency gates
regress by 0.0375/0.0307. Always-on lexical nevertheless improves by +0.0388,
making the gate 0.0032 worse rather than the required 0.0025 better. Decision:
`retire_v22b_fixed_confidence_gate_insufficient_language_gain`.

The result separates address correctness from predictive utility. A detached
same-document gate removes useful low-margin episodes as well as harmful ones,
and the earlier V12 frozen-bank utility predictor already failed disjoint
transfer. The V22b runner and tests are deleted. The next experiment may reuse
the causal V22 document builder but must co-train fresh cortex arms on off,
random, and lexical contexts, then require disjoint likelihood and anchored
generation together before saving a checkpoint.

The deleted joint-document runner preregistered V23. It sampled
2,048 causal cases from each replay source and 128 cases from each disjoint eval
source. Off, random-one, lexical-one, and non-promotable oracle-one restore the
identical 1B-token V11 state, optimizer recipe, 800-step schedule, and 25%
ordinary general replay. The 75% document steps score only the hidden 16-token
continuation; each memory arm reads exactly one earlier 48-token episode through
ordinary cortex attention. Retrieval sees only the visible 48-token prefix.

Oracle-one must first beat the off arm by +0.02 paired loss with a positive
4,096-sample bootstrap lower bound. Lexical-one must retrieve the planted older
episode in at least 70% of eval cases, beat off by +0.01 with a positive lower
bound, beat equal-token random-one by +0.005, and assign at least +0.02 more
likelihood to the true episode than a guaranteed distractor with a positive
lower bound. Neither general holdout may regress by more than 0.10. A pass
advances only to anchored free-generation review; the runner saves no checkpoint
and makes no speed or base-quality claim.

The V23 report is
`reports/language_scaling/v23-joint-document-retrieval-800step-20260711.json`
(SHA-256
`5b0010dbb3361362ec174b067efaf93e783c7860cb54db1e5dae23532a45cb6e`).
Off/random-one/lexical-one/oracle-one disjoint loss is
3.2274/3.2454/3.2083/3.1857. Oracle's matched gain is +0.0417 with 95% interval
+0.0112 to +0.0743. Lexical gains +0.0192, but its interval is -0.0049 to
+0.0481 and target inclusion is 69.92%. The lexical-trained model nevertheless
uses evidence: true history beats a guaranteed distractor by +0.0833 with
interval +0.0580 to +0.1108. All 28 model parameter tensors receive nonzero
gradient.

Lexical general retention fails independently at +0.1200 FineWeb-Edu and
+0.1346 Cosmopedia. Eight free continuations reach 3.13% expected token-position
accuracy and 19.17% unique-target-token recall. Decision:
`retire_v23_lexical_retrieval_breaks_general_language`; no checkpoint is saved.
The next and final raw-context test uses 50% ordinary replay and compares
lexical top-two with top-one and equal-token random-two. A failure there moves
document memory behind a separate reader rather than another curriculum tweak.

The same temporary runner replaced V23 with V24 rather than maintaining both
paths. V24 restored the same 1B-token
V11 state for off, random-two, lexical-one, lexical-two, and non-promotable
oracle-two. Its 800 steps are split 50/50 between causal document targets and
ordinary general replay. The candidate reads two 48-token spans; oracle-two
contains the planted target plus one distractor, while wrong-two contains two
distractors, keeping the 96-token source budget matched.

Oracle-two must beat off by +0.02 with a positive paired bootstrap lower bound.
Lexical-two must include the target in at least 85% of eval cases, beat off by
+0.01 with a positive lower bound, and beat both random-two and separately
trained lexical-one by +0.005. True-plus-distractor context must beat two wrong
spans by +0.02 with a positive lower bound, and neither general source may
regress by more than 0.10. A pass advances to anchored review only; a failure
retires raw prompt-style document memory in favor of a separate reader.

The V24 report is
`reports/language_scaling/v24-balanced-top2-document-retrieval-800step-20260711.json`
(SHA-256
`340e397a4b90d035c26ab30ce849e42d670def6386dd12d5fcfe1be5692e700d`).
Off/random-two/lexical-one/lexical-two/oracle-two loss is
3.1261/3.1281/3.1006/3.1070/3.0946. Lexical-two target inclusion is 82.42%; its
+0.0191 gain has interval -0.0046 to +0.0443 and it is 0.0064 worse than
lexical-one. True-plus-distractor beats two wrong spans by +0.0802, while general
regression falls inside the gate at +0.0677. Decision:
`retire_v24_balanced_top_two_no_joint_language_win`.

Lexical-one is a significant control result: +0.0255 over off with interval
+0.0036 to +0.0513, +0.0682 true-vs-wrong source use, and +0.0701 general
regression. Because V24 lacks a balanced random-one arm, the live runner next
becomes a fresh-seed top-one replication with off/random-one/lexical-one/
oracle-one. Top-two is not maintained. A failed replication retires raw context
memory; a pass advances to anchored review before checkpointing.

Its final V25 configuration changed training/evaluation/model seeds to
12101/12201/12301 and restored off, random-one, lexical-one, and non-promotable
oracle-one from the identical parent under the 50/50 schedule. Lexical-one must
include the target in at least 68% of cases, beat off by +0.01 with a positive
paired lower bound, beat equal-token random-one by +0.005, and never lose to off
on either corpus. True history must beat a guaranteed wrong span by +0.02 with a
positive lower bound, oracle-one must clear +0.02, and each general regression
must remain at or below +0.10. This is the final raw-context replication; no
further top-k, seed, threshold, or replay sweep follows a failure.

V25 passes that likelihood gate. The retained report is
`reports/language_scaling/v25-balanced-top1-replication-800step-20260711.json`
(SHA-256
`dafae2ddabbebb62200e2b8758120e30e38c2c0c4e8ca7705e80f36dd114af76`).
Off/random-one/lexical-one/oracle-one loss is
3.0877/3.0959/3.0447/3.0112. Lexical's matched gain is +0.0430 with 95% interval
+0.0204 to +0.0668; it beats random by 0.0512, improves Cosmopedia/FineWeb-Edu
by 0.0535/0.0326, includes the target in 71.88% of cases, and gives true history
+0.1127 over a guaranteed wrong episode. General regression is
+0.0657/+0.0823. Decision: `advance_v25_replicated_top_one_to_anchored_review`.

The manual review is
`reports/language_scaling/v25-anchored-generation-review-20260711.json`
(SHA-256
`5e1eb137c2949b60579eab50f5ee91db2183349665a5921497a2e3965afe5d7e`).
All eight cases fail the requirement to remain topical, noncontradictory,
coherent, and free of broken entities/repetitive drift. Automated context is
7.81% expected token-position accuracy and 21.53% unique-target-token recall.
Decision:
`retain_v25_likelihood_signal_redesign_separate_evidence_reader_before_checkpoint`.

The raw-context runner and tests are deleted. Future evidence memory must use a
separate bounded reader/cross-attention interface and beat gate-zero, shuffled,
and raw-context controls on anchored generation as well as likelihood. No V25
checkpoint exists.

The deleted evidence-reader runner tested V26 around a separate final-layer
reader. Gate-zero, shuffled-reader, raw-context, lexical-reader, and
non-promotable oracle-reader restored identical V11 plus reader tensors and used
fresh seeds 13101/13201/13301 under the 50/50 800-step schedule. The reader
encoded one 48-token episode in its own V11 pass and injected a gated eight-head
cross-attention residual after the local cortex; local positions remained
unchanged. Gate-zero and raw-context passed exact parity tests against their V11
definitions.

Oracle-reader must beat gate-zero by +0.02 with a positive paired lower bound.
Lexical-reader must include the target in at least 68% of cases, beat gate-zero
by +0.01 with a positive lower bound, beat shuffled-reader and raw-context by
+0.005 each, improve both corpora, and give true evidence +0.02 over a wrong
episode with a positive lower bound. At least half of eight greedy outputs must
change under a true-to-wrong evidence intervention, and each general regression
must remain at or below +0.10. A statistical pass advances to the same manual
anchored review used by V25; only that review can admit checkpoint work.

The V26 report is
`reports/language_scaling/v26-separate-evidence-reader-800step-20260711.json`
(SHA-256
`bc8b3f9ec03fcbf6f241ba0c73320c1f2986e15fdc6bf6ae832098b447fe7a7f`).
Gate-zero/shuffled/raw/lexical/oracle loss is
3.09210/3.09178/3.08659/3.09205/3.09199. Oracle-reader gains only +0.00010
with an interval crossing zero, and true-vs-wrong evidence is +0.00002. All five
reader tensors and all 28 cortex tensors receive nonzero gradients in active
reader arms, but the oracle gate moves only 0.11920 to 0.11949 and lexical
source-swap output change is 12.5%. General retention passes. Decision:
`retire_v26_reader_task_not_learnable_with_oracle_evidence`.

Final-layer injection is retired. V27 reused the exact document and control
protocol with one shared eight-head cross-attention reader interleaved after V11
blocks zero and two. Each injection had its own scalar gate; the query and
48-token evidence streams otherwise shared the V11 cortex, and evidence never
occupied a query position. Gate-zero and raw-context parity were exact. Fresh
training/evaluation/model seeds were 14101/14201/14301 under the same 50/50
800-step schedule.

The retained report is
`reports/language_scaling/v27-interleaved-evidence-reader-800step-20260711.json`
(SHA-256
`c5db3e4d84cddb4c5707861fa610513e6c0812a7fd66264ff6786bd04bfa4751`).
Gate-zero/shuffled/raw/lexical/oracle loss is
3.12936/3.16483/3.08680/3.16858/3.16858. Raw context gains +0.04256 with
interval +0.01674 to +0.07216. Lexical and oracle instead lose 0.03923 and
0.03922 to gate-zero, each with a wholly negative interval. Oracle true-vs-wrong
gain is +0.00617 with interval -0.00754 to +0.02001. The two oracle gates move
from 0.11920 to 0.11869/0.11859; every reader and cortex tensor receives a
nonzero gradient, and both general-source regressions remain below +0.10.
Decision: `retire_v27_interleaved_task_not_learnable_with_oracle_evidence`.

No checkpoint or manual review is admitted. Cross-attention document memory is
closed rather than followed by a wider gate, layer, or selector sweep. The
failed model, runner, and tests are deleted; V25's raw-context likelihood result
remains evidence that exact history has value, not an active memory
architecture.

`language_muon_falsification.py` preregisters V29 as a learning-geometry test,
not a new model-capacity claim. Four arms reuse the exact
20,976,128-parameter Transformer and common initial state: AdamW and Muon at
peak rates 3e-4 and 1e-3. They receive one frozen context-72, batch-32 schedule
with 20% relation curriculum, balanced FineWeb-Edu/Cosmopedia general text,
16,777,216 intended tokens, zero dropout, seeds 16121/16131, and one shared
fullgraph CUDA/Inductor model compile. Labels remain metrics-only.

The MARULHO Muon implementation follows the published scalable recipe: hidden
matrices use 0.95 Nesterov momentum, five Newton-Schulz steps with coefficients
3.4445/-4.7750/2.0315, 0.2 update-RMS scaling, and weight decay; the tied
embedding and norms use AdamW. Orthogonalization batches the four repeated
Transformer matrices of each shape and compiles those shape graphs on CUDA.
The model itself, initialization, parameter count, batches, gradient clipping,
and cosine schedule are fixed, so the two learning-rate controls distinguish an
optimizer effect from merely raising the step size.

The discarded eight-step smoke passes model compile parity at 0.000462 and all
parameters receive gradients. The 1,050,624-token diagnostic then gives
Muon/AdamW at 1e-3 heldout loss/perplexity 5.75274/315.05 versus
6.10653/448.78. Muon sustains 49.1k versus 84.7k training tokens/s, uses 96.0
versus 160.0 MiB optimizer state, and peaks at 595.8 versus 570.1 MiB CUDA
allocation. Both score 100% on metrics-only candidate ranking and 0/32 on exact
free generation, so the early evidence is loss-only and non-promotable.

At the durable budget, V29 compares the best learning rate within each optimizer
family. Muon advances only if it beats best-AdamW heldout loss by at least 0.01
and exact free relation generation by at least 0.02 together. A disjoint gain
routes to redesign; no gain retires Muon. No checkpoint, runtime optimizer, or
continual-learning claim is admitted before a joint pass and genuinely unseen
generation.

The durable four-arm report is
`reports/language_scaling/v29-muon-falsification-16m-20260711.json`, SHA-256
`940a0649865480f6c7b7a57dfc9c96aadd847e4239738f528eb0ef397eb9d4d4`.
Every arm processes 16,777,728 identical tokens from common initial weights and
every parameter receives a gradient. At 1e-3, Muon/AdamW heldout
loss/perplexity is 4.09608/60.10 versus 4.26061/70.85; exact free relation is
17.58% versus 5.47%, a +0.16453 loss gain and +0.12109 generation gain. Muon
candidate ranking is 100% versus 81.64%. It sustains 55.8k versus 96.3k
tokens/s, uses 96.0 versus 160.0 MiB optimizer state, and peaks at 508.3 versus
571.3 MiB CUDA allocation. The 3e-4 Muon arm improves loss by 0.0198 but loses
11.33 generation points, exposing a real learning-rate/behavior interaction.
Decision: `advance_v29_muon_to_unseen_generation`. Container and ownership free
generation remain 0%, so the pass admits only an exact candidate reproduction,
strict checkpoint reload, and unseen prose review.

`language_muon_reproduction.py` owns that next boundary. It accepts only the
durable V29 report with the advancing decision, reconstructs and hash-checks the
same tokenizer and indexed schedule, and reruns only Muon 1e-3 from the original
seed. The candidate must independently re-pass the recorded best-AdamW loss and
free-generation margins before a checkpoint can be written. A fresh strict load
must restore every tensor bit-exactly, preserve tied embedding/head storage and
tokenizer/config identity, and reproduce sample logits exactly. The checkpoint
records that optimizer state is not saved and admits only unseen generation,
not continuation or runtime installation.

The independent reproduction report is
`reports/language_scaling/v29-muon-checkpoint-reproduction-16m-20260711.json`,
SHA-256
`7c98acb3ff77869943f72974c582b0f996cd0b9d6ba617689689339555492f88`.
It reaches heldout loss/perplexity 4.09555/60.07 and 26.95% exact free relation,
independently clearing the recorded AdamW margins. The strict checkpoint is
`reports/language_scaling/v29-muon-qualified-16m-20260711.pt`, 100,933,330
bytes, SHA-256
`e4ad48aea9d02cabca457255637c884131d55ac7f65998e3b9e025475e13415d`.
Every model tensor, tied embedding/head storage, tokenizer hash, config, and
sample logit reloads exactly. Optimizer state is explicitly absent.

The ensuing unseen reports are
`reports/language_scaling/v29-muon-unseen-fineweb-greedy-20260711.json`,
`reports/language_scaling/v29-muon-unseen-cosmopedia-greedy-20260711.json`, and
`reports/language_scaling/v29-muon-unseen-cosmopedia-controlled-20260711.json`.
FineWeb-Edu and Cosmopedia remain 0/4 source passes with mean source loss
4.5952/3.8875. Controlled decoding leaves source loss unchanged but raises
Cosmopedia distinct-bigram fraction from 0.794 to 0.976. Direct review finds
some grammatical, topical educational paragraphs, but FineWeb generations are
often repetitive or nonsensical and Cosmopedia drifts into generic templates,
factual confusion, and unstable entities. Decision:
`retain_v29_muon_redesign_base_curriculum_and_context`. Muon remains an
uninstalled training candidate; the checkpoint is a V30 baseline, not a
quality-qualified language model.

`language_general_context_falsification.py` preregisters V30 around the unseen
failure rather than the synthetic relation win. Two fresh Muon 1e-3 arms use
the same 20,976,128 parameters, exact initial tensors, tokenizer, sampled
FineWeb-Edu/Cosmopedia ranges, and 16,777,728 general-only update tokens.
Context 72/batch 32 and context 256/batch 9 both process 2,304 tokens per step
for 7,282 steps. Relation fraction is exactly zero; the label-safe 256-case
relation suite is diagnostic only.

The strict V29 checkpoint is never used to initialize the candidates. It owns
the common context-72 general holdout baseline. A candidate advances only with
at least 0.05 lower common loss. If both pass, context 256 must beat context 72
by at least 0.02 to justify its extra attention cost; otherwise the shorter arm
wins. Only the selected state may be checkpointed, and strict tensor/tokenizer/
config/tied-weight/logit fidelity must pass before unseen review. The discarded
eight-step preflight establishes equal 2,304-token steps, exact initial hashes,
complete gradients, common source ranges, no checkpoint write, and
compiled/eager loss deltas 0.000055/0.000195. These are mechanical facts, not
quality evidence.

The durable report is
`reports/language_scaling/v30-general-context-falsification-16m-20260711.json`,
SHA-256
`fc78d39f7ee27522298e50c041c864d5b6242952637d414f455eeefd669497d6`.
Both arms process 16,777,728 tokens with all gradients and identical initial
tensors. V29/general72/general256 common loss is 4.09555/4.00933/4.02583;
general72 therefore clears the 0.05 baseline gate and context256 misses its
extra 0.02 premium. General72/general256 train at 55.85k/55.57k tokens/s and
both produce 0% exact free relation after zero relation updates. The selected
strict checkpoint is
`reports/language_scaling/v30-general-context-qualified-16m-20260711.pt`,
SHA-256
`a0863eaa85f354f4eacb4c7c0ae422f516d93630c8d11d47de87eabe053440b2`;
tensor, tokenizer, config, tying, and sample-logit fidelity are exact.

FineWeb-Edu/Cosmopedia unseen source loss improves from V29's 4.5952/3.8875 to
4.4801/3.8488, but the three V30 unseen reports remain 0/4, 0/4, and 0/4.
Controlled Cosmopedia reaches 0.968 distinct-bigram fraction without correcting
generic templates, repetition, factual confusion, or semantic drift. Decision:
`retain_v30_general72_scale_unique_general_data_before_redesign`. The 16M
schedule uses 7,282 unique batches exactly once. The complete local FineWeb-Edu
and Cosmopedia replay shards total about 647 MB, enough for a fresh roughly
67M-token scale test without fabricated repeated-data progress.

`language_general_scaling.py` owns V31. It preserves V30's 20,976,128-parameter
context-72 model, exact initial tensors, Muon 1e-3 recipe, tokenizer, and common
holdout, but trains a fresh state for 29,128 steps. Each source contributes a
256 MiB sample built from 16 ranges spanning the full shard. The split builder
then selects windows across each sampled stream rather than taking a prefix.
The reloaded V30 holdout must reproduce its recorded loss within 1e-5; the
actual value, absolute delta, and tolerance are all persisted.
The report must prove that all 67,110,912 scheduled tokens are unique within
the run, every prepared general batch is consumed exactly once, byte and token
windows span both sources, all parameters receive gradients, and the strict
checkpoint reloads exactly. Advancement requires at least 0.15 heldout-loss
gain over V30 before the same unseen suite is allowed. Relation scores remain
metrics-only diagnostics and do not select the model.

The durable report is
`reports/language_scaling/v31-general-scaling-67m-20260711.json`, SHA-256
`f65fcef87445004c5fbbeacd9240e89e1b46be4b6d6770108e86ff20037f5798`.
It processes 67,110,912 unique scheduled tokens in 29,128 steps, with exact
14,564/14,564 source balance and full-span byte/window audits. V30/V31 common
loss is 4.00934/3.62911, a 0.38022 gain; V31 perplexity is 37.68. It sustains
56,126.7 training tokens/s, uses 96.04 MiB optimizer state, peaks at 593.58 MiB
CUDA allocation, gives all parameters gradients, and passes compiled/eager
parity at 0.0000496. The strict 100,933,202-byte checkpoint is
`reports/language_scaling/v31-general-scaling-qualified-67m-20260711.pt`,
SHA-256
`d4250e16cf69ea7e13d222826f41068be01b5c319475b71cd1685a149e4a73bc`.
Tensor, tokenizer, config, tied-weight, and sample-logit fidelity are exact.

The unseen FineWeb-Edu greedy, Cosmopedia greedy, and Cosmopedia controlled
reports have SHA-256
`0622e1bc80d51c5bd055251f2a138244dd309aa8dd5a006384e7f235c983faf8`,
`2f7901de2428520a6aa25b76cfc8cfa69b1c1cad05d8e127c329ffa0027a4dfc`,
and `db9216ea57d4d075abadd5ba612b8afc8286c3510e6d1d51af04b18555a2c550`.
FineWeb-Edu/Cosmopedia source loss is 4.2053/3.4896, improving on V30, but all
three suites remain 0/4. Controlled decoding corrects the worst repetition and
raises Cosmopedia distinct-bigram fraction 0.667→0.960 without grounding the
answer. Decision: `retain_v31_scaling_curve_expand_unique_data_not_base_quality`.
No runtime, memory, continual-learning, or qualified-coherence claim follows.

`language_general_scaling.py` is now a stage-driven v2 engine rather than a
copied V32 runner. The V31 stage remains reproducible from V30; V32 strictly
accepts only V31's advancing report and checkpoint as evaluation evidence. V34
also accepts V31 as evaluation evidence but explicitly does not require an
initial-state hash match because it is a preregistered capacity change rather
than a continuation of the same shape.
`language_matched_support.py` accepts two or more general training sources,
derives per-source preparation depth from the actual source count, and releases
each raw source string after its batches are built. Existing two-source
experiments are numerically unchanged.

V32 preregisters 201,323,520 tokens: 87,380 context-72/batch-32 steps across five
disjoint shards, exactly 17,476 batches and 40,264,704 tokens per source. The
default sample cap is 512 MiB per source; a smaller source may be selected whole
only when its single recorded range spans the entire file. Prepared token
windows remain stratified over each selected stream. V31's reloaded common loss
must reproduce within 1e-5. Advancement requires at least 0.20 lower loss,
complete gradients, unique scheduling, coverage audits, and exact checkpoint
fidelity. The discarded ten-step CUDA audit passes five-source balance,
uniqueness, full-span byte/window coverage, complete gradients, V31 loss exact
reproduction, and compiled/eager parity at 0.000367. It writes no checkpoint
and its temporary report is deleted.

The durable V32 report is
`reports/language_scaling/v32-general-scaling-201m-20260711.json`, SHA-256
`0c305ba84ae259bdaeb94e511a007aef4de50a0161b2d37ccf0a05f166742d64`.
All five sources contribute exactly 17,476 distinct batch indices and all
201,323,520 tokens are unique within the run. V31 loss reproduces exactly;
every parameter receives a final gradient; compiled/eager delta is 0.000103.
V31/V32 heldout loss is 3.62911/3.49830 and V32 perplexity is 33.06. The 0.13082
gain misses the frozen 0.20 requirement despite 56,177.4 training tokens/s,
96.04 MiB optimizer state, and 593.58 MiB peak CUDA allocation. Metrics-only
candidate accuracy is 48.83% while free exact relation generation is 0%.
Decision: `stop_v32_general_scaling_no_durable_loss_gain`. No checkpoint is
written and the unseen suite is not run. The result closes fixed-21M data-only
scaling and routes to architecture redesign.

V34 uses the same maintained runner for a fresh 100,679,424-parameter capacity
test: width 768, ten layers, twelve heads, context 72, batch 32, Muon 1e-3, and
V31's 67,110,912-token two-source schedule. V31's checkpoint reproduces the
common holdout but never initializes the larger candidate. Muon orthogonalizer
shapes and three optimizer steps warm outside measurement; exact initial weights
are restored, the optimizer is rebuilt, RNG is reset, and no warmup token is
counted. Advancement requires at least 0.20 lower loss plus the existing
gradient, parity, source, uniqueness, and strict-checkpoint audits. The stage is
a stronger-cortex qualification, not a Transformer-successor claim.
The disposable production-shape preflight compiles in 52.5 seconds, passes BF16
parity at 0.00063, trains at 11.3k tokens/s, peaks at 3.32 GB, and reproduces the
V31 holdout exactly. Its two-step loss and report are deleted as non-evidence.

The durable V34 run processes 67,110,912 unique tokens and reaches loss 3.3902
versus V31 3.6291, passing the 0.20 gate by 0.0389. It sustains 11.14k tokens/s,
peaks at 3.319 GB, and saves a 428.1 MB strict checkpoint with exact fidelity.
FineWeb/Cosmopedia source loss improves to 4.0012/3.2831 and prose is more
coherent, but anchored generation remains 0/8. The checkpoint is retained for
V35, not installed.

The V35 stage requires the V34 advancing report/checkpoint and initializes the
same 100,679,424-parameter shape from that strict state. It rejects any training
source list or content hashes except FineWeb shard 0 and Cosmopedia shards 1/3, which do not overlap
V34. It adds 134,219,520 tokens, records 201,330,432 cumulative updates, uses a
fresh Muon optimizer at 3e-4, and requires another 0.15 loss gain before saving.
The run's diagnostic loss improves 3.3902 to 3.1654, but the manifest consumes
58,255 of 58,257 prepared unique batches. Full coverage fails, the decision is
`invalid_v35_capacity_continuation_evidence`, and no checkpoint is saved.

V35R is the corrected immutable stage. It pins the exact V34 report/checkpoint
hashes, three corpus hashes, and 19,419 prepared batches per source. Exactly
58,257 updates consume 134,224,128 new tokens for 201,335,040 cumulative tokens.
Model shape, initialization, optimizer, learning rate, and +0.15 quality gate do
not change; CLI changes to this repair manifest are rejected. The valid rerun
reaches loss/perplexity 3.1649/23.69 from V34's reproduced 3.3902/29.67, with
complete coverage, gradients, parity, and strict checkpoint fidelity. Unseen
FineWeb/Cosmopedia continuation loss is 3.8020/2.9282. Controlled Cosmopedia
output is coherent and reaches 0.968 distinct-bigram fraction, but exact source
anchoring remains 0/8. V35R therefore admits continual-learning and conditional-
compute research from this checkpoint while memory grounding and runtime
installation remain unqualified.

V36 is the preregistered consumer-GPU throughput screen for that exact V35R
state. Six arms consume the same 2,359,296 ordered tokens from the same frozen
schedule: batch 32 versus physical batch 256, three batch-256 learning rates,
and whole-QKV versus per-head Q/K/V Muon. Compile and optimizer warmup time are
reported separately; every arm is reloaded from the same checkpoint tensors.
A large-batch recipe advances only with at most 0.01 heldout-loss regression and
at least 1.80 times measured training throughput. Per-head Muon has its own
at-most-0.005 loss and at-least-1.03 speed gates. Faster but worse learning does
not advance.

The completed screen advances physical batch 256 with whole-QKV Muon at 3e-4.
It reaches heldout loss 3.1423 at 25.07k tokens/s versus the batch-32 control's
3.2455 at 11.08k, a 2.262x speedup with better loss, and peaks at 7.72 GiB.
The frozen fastest-arm selector reports per-head/8.5e-4, but that arm has 0.0844
worse loss for only 1.68% more throughput. The artifact is retained unchanged;
the project's quality-first rule resolves the next recipe to whole-QKV/3e-4.
Per-head Muon separately passes at batch 32 but does not clear its speed gate at
batch 256. No V36 checkpoint is saved.

V37's exactly identity-initialized, 45-route depth assembly is mechanically
valid but fails the bounded consumer-GPU run. It remains active beyond the fixed
3,600-second parent timeout and reaches 11,744/12,288 MiB sampled device
allocation. No terminal report or heldout result exists, so the experiment makes
no quality claim. The candidate and one-shot evaluator are deleted; the timeout
artifact and git history retain the evidence. Future depth routing must use a
few fused dense or low-rank operations rather than full-width history retention.

V38's 50/50 replay arm reaches 100% candidate recognition and 46.88% strict free
answers while improving general loss from 3.1649 to 3.1124 at 25.02k tokens/s.
It misses the 50% free gate, so no checkpoint is saved. Relation-only learning
catastrophically regresses general loss to 15.8373 and produces fewer free exact
answers (37.50%); 20/80 replay also produces 37.11%. The retained evidence says
replay works and answer realization is now limiting. The one-shot V38 runner is
deleted; V39 must change the answer objective/interface without weakening exact
evaluation.

V39's selected 4x answer objective reaches exactly 50% strict free accuracy,
98.44% candidate accuracy, and general loss 3.11336 at 24.86k tokens/s. It saves
and exactly reloads the 428.1 MB standard Transformer checkpoint. Per-kind free
accuracy remains uneven: property/event-order/container/ownership is
90.62%/79.69%/23.44%/6.25%. The one-shot evaluator is deleted after retaining
the report; checkpoint fidelity and the maintained training objective survive.
That 50% is the immutable original result under prompt-inclusive no-repeat-3.
V44's same-checkpoint correction below is authoritative for current decoding.

The deleted V28 particle-field falsifier tested a wider base-architecture jump.
It compared the 20,976,128-parameter Transformer with a 20,971,520-parameter
positive particle-field core: width 256, 24,576 particles, four heads, eight
parameter-shared recurrences, strict causal Hebbian linear attention, and the
same tied 8,192-token BPE. The 4,608-parameter difference is 0.022%. No external
weights were loaded.

Both arms received the identical frozen context-72, batch-32 schedule: 20%
relation curriculum and balanced FineWeb-Edu/Cosmopedia general text, 16,777,216
tokens, AdamW intent, zero dropout, fresh data/model seeds 15121/15131, and
fullgraph CUDA/Inductor execution. The particle paper's native dropout was not
swept in the first comparison; zero dropout kept the arms deterministic and
isolated the architecture. Labels remained metrics-only.

The discarded 18,432-token preflight established mechanical viability. At the
durable budget, both arms processed 16,777,728 identical tokens and every
parameter received a final gradient. Transformer/particle heldout
loss/perplexity was 4.31934/75.14 versus 4.91323/136.08. Exact free relation
generation was 40.23% versus 11.33%, although metrics-only candidate ranking was
99.61% versus 100%. This distinction prevents target-choice likelihood from
being mistaken for autonomous generation.

The particle field also sustains only 11,104.9 training tokens/s versus
92,604.9 and peaks at 5,360,724,480 versus 597,788,160 CUDA bytes. Its mean zero
activation fraction rises to 66.05%, but observed zeros do not compensate for
the quality and cost loss. The durable report decides
`retire_v28_particle_field_no_joint_language_win`; SHA-256 is
`bd05b350f8af73f9c5d2b5229981b92bfc1da61b1aaa862f42c07ae9cbc545cb`.
It is retained locally at
`reports/language_scaling/v28-particle-field-falsification-16m-20260711.json`.
No checkpoint or unseen review is admitted. The failed model, runner, and tests
are deleted rather than retained as a dormant architecture path.

The deleted V10 product-key falsifier is retained only as two compact local
reports:
`reports/language_scaling/micro-experts-v10-falsification-16m-20260711.json` and
`reports/language_scaling/micro-experts-v10-replication-seed7331-16m-20260711.json`.
They show a replicated fixed-hash loss gain and learned-router pool collapse;
V11 owns the surviving hash mechanism without a compatibility path.

The retired depth-connection v9 evidence is local at
`reports/language_scaling/depth-connections-v9-falsification-16m-20260711.json`
and
`reports/language_scaling/depth-connections-v9-replication-seed7331-16m-20260711.json`.
Both 16.79M-token reports decide
`redesign_v9_disjoint_loss_and_behavior_signals`: signed learned reuse replicated
a small loss gain but not a reliable strict-generation gain or a joint win over
identity and fixed controls. All controls used exact resets, matched common
initialization, complete gradient and parity audits, and one shared candidate
graph with within-0.14% candidate throughput. No checkpoint was saved; the v9
model, runner, and tests are deleted.

**`language_generation_coherence.py`** — evaluates checkpoint generation on
explicit prompts or heldout exact-continuation prefixes. It records text evidence
and source-continuation loss. The source document is metrics-only, so this is not
evidence-conditioned grounding. The CLI strictly loads either the installed
Transformer surface or the experimental hashed-micro-expert surface, and records
checkpoint, tokenizer, source hashes, qualification metadata, and ownership.
Continuation loss encodes no second BOS token and scans only a progressively
expanded source prefix until the requested token prefix is stable; it never
tokenizes a tens-of-megabytes source tail to score at most one context window.
Automated passes are diagnostic and do not alone promote quality.

V46 reruns the repaired exact-continuation path on V35R and V39. FineWeb-Edu
loss changes 3.60076 to 3.64029 and Cosmopedia 2.71844 to 2.64983; V39 passes
0/12 across greedy and corrected-control reports. This is mixed bounded retention
and a negative exact-continuation result. The next evaluator must supply visible
heldout evidence with question-only and corrupted-source controls. Composite
report SHA-256 is
`9df4477f806f99f46892ca828e3e1b058588f2a8e6501e5d94ae15d6f43914e2`.

**`language_source_grounding.py`** — materializes a tokenizer-bound, hashable
subset of the public SQuAD validation split through the Hugging Face Dataset
Viewer and evaluates visible heldout evidence. Every case fits the active context,
keeps an extractive answer in its source passage, excludes answer leakage from
the question, and owns question-only plus answer-absent mismatched-source
controls. Intact source must reach 25% strict answer accuracy and beat the
stronger control by ten points. External text is data only; MARULHO owns all
inference and `external_llm_used=false`. The path is a grounding benchmark and
future continual-training target, not a memory claim.

V47's frozen V39 audit passes every validity check and keeps model hashes exact.
Intact/question-only/mismatched-source strict answers are 3/64, 0/64, and 0/64.
The 4.69-point source gain misses the 5-point weak-use threshold and 25%/+10-
point promotion gate, so the branch admits matched training with replay rather
than a base-grounding claim. Manifest/report SHA-256 are
`9b3392f137a2ca467bc329815810581a98169da170f74f50e8ccb41cb06e12d6`
and `5a4d36afec1f20f8bf777e7f5eaef35e171e07c2e238bbd7001e028113477b71`.

**`language_source_grounding_continual.py`** — owns V48's matched continual
grounding screen. It freezes the V39 checkpoint, V47 validation manifest, SQuAD
training corpus, replay sources, schedule, optimizer, seeds, and token budget;
ordinary and 4x answer-weighted loss are the only arm difference. Real batch-8
gradient accumulation keeps effective batch 224 without CUDA paging. At
4,193,280 tokens, ordinary/weighted intact accuracy is 14.06%/21.88%, but
relation recall falls from 89.06% to 40.62%/43.75% and general loss regresses
by 0.1045/0.1023. Decision: `retire_v48_objective_only_grounding_repair`.
Atomic arm states are deleted after the terminal report; no candidate checkpoint
survives. Report SHA-256 is
`834e1bce825675f0c18cac77c39e30b8403fcb5368e3937b9c91a46b5b9fb968`.

V49's frozen-base final-sidecar falsifier is retired and deleted. The 4.13M-
parameter module trains at 55.45k tokens/s and preserves inactive V39 logits,
general loss, and relations exactly, but active grounding reaches only 1/64.
No model, runner, checkpoint surface, tests, or compatibility import survives.
Report SHA-256 is
`204bbd170158834017fe5b52c0874491a02112c257ca912586fecc77d3aef7a1`.

V50's matched rank-16 all-layer delta falsifier is retired and deleted. It
preserves inactive V39 evidence exactly, trains all 2.46M deltas at 23.62k
tokens/s, and improves active grounding over V49 to 5/64, but remains far below
V48's 14/64. No model, runner, checkpoint surface, tests, or compatibility path
survives. Report SHA-256 is
`c97ba0505aa06c3976802430851abc8a3f321f110960ac437320a26307d46541`.

V51's full-capacity specialist falsifier is retired and deleted. The complete
100.68M-parameter fork trains every parameter at 5.30k tokens/s and preserves the
inactive parent exactly. Active grounding reaches 12/64 versus 1/64 for both
controls, but loses to V48's 14/64 while its general loss regresses to 5.1532.
No specialist checkpoint, runner, test, routing option, or compatibility path
survives. Report SHA-256 is
`9b74355e9c287270e28a6fa5b9c54ad79bd25428983d3c5e61a6bd10ea033fad`.

V52's document-alignment falsifier is terminal and its one-off runner is deleted.
Global stride-72 packing retained the complete prompt and answer for only 80/512
individually fitting SQuAD records; alignment raises heldout grounding from
14/64 to 19/64 with both controls at zero. Retention still fails, so no checkpoint
survives. The generic aligned-batch and post-EOS pad-mask machinery remains live
for the preregistered isolated copy/span successor. Report SHA-256 is
`23ef805fae825cd3bd46dd5a85c1deebc3eaabe38db59a9f3750657b6557e33d`.

V53's compact frozen-base source-pointer falsifier is retired and deleted. The
99,073-parameter path preserves the parent exactly and reaches 17/64 versus zero
for both controls, but misses the 18/64 gate and V52-retention threshold. No
model, runner, test, checkpoint surface, or compatibility path survives. Report
SHA-256 is
`3af6ebad988b2844d83b91f73fe3f7c22443dab933e5f6fcbf9a1bbf48ae4620`.

V54's trainable source-encoder falsifier is retired and deleted. It passes every
mechanical, isolation, runtime, and checkpoint check at 173.97k padded
positions/s, but reaches only 16/64 intact answers versus zero for both controls.
Ten failures contain truncated answer fragments, and its successes overlap weakly
with V52/V53. The compact checkpoint, encoder, runner, tests, loader, and
compatibility surface are absent. Report SHA-256 is
`32f2c700c8168c6fdccb4c681afda978f9113645b0686ca44253b81aed04d0e0`.

V55's materially larger multi-view falsifier is retired and deleted. The fused
model reaches 20/64 versus 16/64 causal-only and 2/64 bidirectional-only, but
misses the 32/64 capability and 45-point source-gain gates. Parent fidelity,
runtime, gradients, and compact reload all pass. Noncontiguous BPE pointer
assemblies identify a structural output failure; no runner, model, tests, cache,
or checkpoint survives. Report SHA-256 is
`d1d30b6aec1237277d57be534c7029c66364546b16439ce4ad9830d59bfc6911`.

V56's frozen data boundary is active before its model exists. Preflight rejected
one manifest for an empty-normalized Unicode answer and V56b because four query-
plus-answer sequences exceeded V39's 72-token decoder window; neither trained a
model. The corrected V56c manifest contains 8,192 new official-train cases from
170 titles, 96–228 source tokens, no empty normalized answers, query/answer/EOS
length at most 71, and 2–5 fixed 48-token blocks. Contract/file SHA-256 values
are `efd56051f98ea32fad2474e3f9504d33bec6aac4d7e69978378f3c9547d5552d`
and `ebc512f0a1d680ce3c9b0f11b52ed9a86395f035be125c5470d4b326e902a5e3`.
The 128-case official-validation panel excludes V47, spans 127–246 prompt tokens,
and has contract/file SHA-256 values
`feca4f4088d3452265f2fc35240f7aa45de68dfc856e0be80af7f45a9e470a84` and
`fa609b4c6c381d1d0c347fc3286dc2ed5e35daea4c57da8400b20056f0facbc6`.
Train, validation, and all prior case-ID intersections are zero. This proves the
long-context benchmark contract only; no V56 quality result exists yet.

V56's frozen-parent landmark-evidence retrofit is terminally negative. It
processes all 20,643,840 adapter positions, receives every gradient, preserves
the parent exactly, and reloads its compact state. Predicted top-two answer-block
coverage reaches 91/128, below the required 80%. Predicted top-two, top-one,
oracle, and shuffled evidence all generate 0/128 exact answers, so correct block
selection does not rescue the frozen residual interface. The failed runner,
model, tests, cache, and checkpoint are deleted. Report SHA-256 is
`7b53df754d275a211412df2c006956901cb7f7a910da26556c4a8a8abfef6e3d`.

V57 freezes a new native-context data boundary. Its 8,192-case official-train
manifest excludes all V48/V55/V56 train IDs; its 256-case official-validation
manifest excludes V47/V56 validation IDs. Every full causal prompt, standalone
answer, and EOS fits 320 tokens, with exact trailing-space BPE prefixes shared by
training and generation. Each case also stores an answer-containing oracle-short
prompt of at most 96 tokens for a matched localization control. This data was
frozen before the architecture and gate immediately below.

The frozen V57 comparison used full-source native context
against an oracle-localized control under exact 320-token padded compute. Both
update all V39 parameters for 20,971,520 positions with 50% grounding, 25%
general replay, and 25% relation replay. Native and oracle each need 128/256;
native must gain 45 source-control points, remain within 16 cases of oracle, and
retain general/relation behavior. Eager batch 32 is frozen after the real
preflight; compiled execution is invalid and slower. Preflight report SHA-256 is
`bf4f5a74b3710835085bc152a4c1d0eababdc339066c2372944cde8eef831a5e`.
The terminal runner prepared 256 grounding batches per arm, 2,773 relation
batches, 346/296 general batches, and schedule hash
`bc736bbb94434c79d2a1e59d667a751ca3dfd211cc38b0603b46b6bb79037d9d`.
The terminal result retires the path. Oracle-short reaches 122/256 and
native-full 43/256; native reaches 90/256 only when its evaluation evidence is
oracle-localized. General loss regresses 3.1490→3.3712/3.3553 and relation exact
generation 89.06%→34.38%/75.00%. All mechanical and fidelity checks pass at
16.77k/17.03k positions/s, so this is not a broken run. The runner and tests are
deleted; the retained report SHA-256 is
`fe93519ca693837796c76ba8e1161e68e7f4d210ad31a47341f854f90660cb99`.

V58 is preregistered as a protected full-capacity evidence-localization test,
not another compact source head. It clones V39's embedding, ten Transformer
blocks, and norm into an independently optimized bidirectional organ, predicts
one contiguous Unicode-character interval from token states plus within-token
boundary features, and copies only consecutive source characters. Preflight
corrects 34 training and 2 validation stored offsets against the immutable exact
answer strings and proves a 256/256 mechanical copy oracle.
The causal parent remains immutable and source-absent requests bypass the organ.
The primary arm processes the V57 8,192-case title-disjoint training manifest
for eight epochs: 2,048 batch-32 updates and 20,971,520 padded context-320
positions. At least 192/256 heldout exact spans, a 70-point source gain, at most
8 mismatched exact answers, complete gradients, bounded runtime, parent
identity, and strict organ reload are required. A passing primary arm triggers
the same-budget random-initialization control; a primary miss closes the
extractive evidence-organ line.

The terminal V58 result retires that line. All 100,686,146 organ parameters
receive nonzero final gradients across exactly 2,048 updates and 20,971,520
padded positions. Training takes 871.75 seconds at 24.06k positions/s with
5.39 GB peak allocation. The mechanical oracle is 256/256 and every parent
fidelity check passes, but intact/mismatched exact extraction is only 20/256 and
0/256. Capability and the required source margin fail, so the random-init
control is not run and no checkpoint exists. The runner and tests are deleted;
report SHA-256 is
`761f3f385d0524a880f568f956aaafbf0f520b4124c4a2a525302707229331e2`.
Further evidence work must leave the SQuAD span/pointer family.

V59 preregisters a different interface: per-document source-only test-time
learning in a reset full-capacity V39 copy. The frozen 64-case/all-22-title panel
compares no write, mismatched write, true write, and oracle-short write under the
same question-only generation policy. The candidate needs 16/64 exact and a
12-case control margin; oracle-short needs 24/64, mismatch is capped at 8/64,
and source loss must improve in 90% of true writes. No question, answer, span,
or label enters the write objective. Exact reset, complete gradients, parent
fidelity, and a 2,400-second total wall limit are required.

V59 is terminally negative and deleted. All 64 true writes lower source loss,
all 62 tensors receive gradients, all 192 resets are exact, and the parent stays
exact. The true arm processes 52,012 positions in 844 full-model updates at
570.1 positions/s; total wall time is 388.30 seconds and peak allocation is
1.13 GB. Nevertheless, all four arms score 0/64 strict answers. True and oracle
writes contain accepted answer text in 5/64 continuations versus zero for both
controls, but extra text invalidates every answer. No checkpoint survives;
report SHA-256 is
`388c43f79c10cc306fc12b1f1d7ad245ba42c317e40d18007e11d357d18247f0`.
Naive full-model test-time adaptation is closed; later work requires an
end-to-end meta-learned write/read contract.

V60's sub-1% meta-gradient episodic matrix is terminally negative and deleted.
All 2,048 updates finish, all six controller tensors receive gradients, frozen
V39 fidelity is exact, and 20,971,520 source positions train in 372.55 seconds
at 56,292 positions/s with 1.27 GB peak CUDA allocation. But untrained true,
learned zero, shuffled, true, and oracle-short views all score 0/256 exact and
contain zero accepted answers. The failure is capability, not execution.
Decision: `retire_v60_one_step_linear_meta_gradient_memory`; no checkpoint is
saved. Report SHA-256 is
`76becda7f4d4986eb0bfca1056d2dd14f074c4d348bf5cf0f735c6125e9718fb`.
The next admissible evaluator must test iterative nonlinear fast learning, not
another one-step linear matrix or extractive reader.

V61 preregisters eight temporary two-layer MLP heads adapted by two explicit
source-only reconstruction-gradient steps. A sub-2% slow controller meta-learns
the initialization, source key/target views, question query view, positive step
sizes, and bounded final residual while frozen V39 owns vocabulary generation.
The exact V57 8,192/256 title split and V60 compute schedule remain unchanged.
Terminal no-write, shuffled, true, and oracle-short views require 64 true, a
20-point source gain, at most 16 shuffled, at least 128 oracle, and a true/oracle
gap no larger than 64. Inner-loss decrease, complete gradients, finite fast
state, exact parent/context fidelity, timing, CUDA allocation, and strict reload
are required evidence. Oracle failure sends the next experiment to protected
all-depth reading; it does not authorize a wider final-residual MLP.

**`language_decode_comparison.py`** — compares greedy argmax with deterministic
temperature/top-p sampling from the exact same checkpoint and prompts. It
records checkpoint/tokenizer hashes and full decode-policy evidence. Decode
improvement is an ablation result, never a substitute for heldout quality or
human review.

**`language_relation_binding_experiment.py`** — continues one checkpoint on a
procedural entity/event curriculum and evaluates compositionally held-out
container, ownership, property, and event-order cases. Predictions rank every
candidate by continuation loss before the correct index is used for metrics.
The same mixed-language holdouts measure retention, so relation gains cannot
silently trade away general language. Repeated `--replay-corpus` sources test
whether a budgeted general-data mixture preserves both capabilities before
parameter isolation or episodic memory is introduced.

**`language_relation_binding_audit.py`** — reruns both candidate-likelihood and
strict greedy free-answer metrics for any checkpoint against a frozen relation
case artifact. This catches cases where multiple-choice ranking improves while
open generation still loses the relation.

The retired modular predictive society v3 report is
`reports/language_scaling/modular-society-v3-falsification-16m-20260710.json`.
At 16.79M matched tokens, monolith/average/no-message/shuffled/real losses were
4.6140/5.0261/5.0460/5.0973/5.1073 and strict free relation scores were
14.5%/5.1%/2.0%/0.4%/0%. Compiled society controls sustained 74.4--74.6k
tokens/s with compile/eager loss deltas at or below 0.000026. Real communication
lost both required controls, so the model, runner, and tests are deleted.

The v4 report is
`reports/language_scaling/modular-workspace-v4-falsification-16m-20260710.json`;
its scientific interpretation is in the adjacent `-review.md`. Real exchange
reached 4.8507 loss / 21.5% free relation versus 4.8518 / 11.7% shuffled and
4.8549 / 10.2% no exchange. It did not pass the loss or monolith guard and is not
scaled.

The v5 report is
`reports/language_scaling/content-addressed-workspace-v5-falsification-16m-20260710.json`.
Monolith/no-exchange/shuffled/real losses were
4.6142/4.8526/4.8479/4.8494 and strict free relation scores were
17.2%/24.6%/22.7%/6.6%. Real associative memory lost both controls while all
workspace parameters received gradients and control throughput stayed matched.
The model, matched runner, and tests are deleted.

The retired v6 report is
`reports/language_scaling/hyperspherical-transformer-v6-falsification-16m-20260710.json`.
Transformer-standard/Transformer-native/normalized-standard/normalized-native
losses were 4.6144/4.6448/6.2844/4.7092 and strict free relation scores were
14.8%/0%/0%/0%. All parameters received gradients, parity and projection audits
passed, and throughput remained matched. No checkpoint was made; the failed v6
model, runner, and tests are deleted.

The retired v7 report is
`reports/language_scaling/dynamical-memory-v7-falsification-16m-20260711.json`.
Transformer/memory-off/single-scale/always-write/random-write/learned-write
losses were 4.6137/4.6092/4.6061/4.6076/4.6088/4.6066; strict free relation
scores were 21.5%/3.9%/10.5%/6.2%/3.5%/4.7%. Learned memory failed both the
Transformer quality guard and the simpler single-scale control. Its write gate
was active, every parameter received a gradient, compiled/eager parity passed,
and memory-control throughput varied by only 0.65%. The candidate sustained
112.7k training tokens/s versus the Transformer's 129.1k. The run used two loss
graph compiles rather than six and retained no checkpoint. Decision:
`retire_v7_no_quality_or_control_gain`; the failed model, runner, and tests are
deleted. Candidate-likelihood relation accuracy was 96.9% for learned memory
versus 93.0% for the Transformer, so its opposite free-generation result is also
evidence that candidate ranking cannot stand in for language generation.

The retired v8 reports are
`reports/language_scaling/depth-allocation-v8-falsification-16m-20260711.json`,
`reports/language_scaling/depth-allocation-v8-replication-seed7331-16m-20260711.json`,
and
`reports/language_scaling/depth-allocation-v8-durability-seed2026-67m-20260711.json`.
Early-heavy beat uniform twice at 16.79M tokens, then lost the successive-halving
durability comparison: uniform/early-heavy loss was 3.8861/3.8957 and strict free
relation tied at 20.3%. Each durable arm processed 67,112,064 tokens. Parameters,
common initialization, gradients, parity, and theoretical compute matched;
observed throughput differed by 0.30%. The excluded source tails were one partial
batch from each general corpus and are recorded in the report. Decision:
`retire_v8_early_heavy_not_durable`. No checkpoint was made; the model, runner,
and tests are deleted.

The retired integrated-PMRM runner established the architecture-neutral matched
experiment contract now used for replacements: same checkpoint-owned tokenizer,
frozen source-balanced schedule, parameters, optimizer, relation/general
evaluation, gradient coverage, state bytes, wall time, throughput, and CUDA
memory. Candidate screens cannot install a runtime model.

The retired sparse event-memory v2 reports are
`reports/language_scaling/sparse-event-v2-falsification-16m-20260710.json`.
Random-sparse reached 4.6128 loss / 27.0% strict free relation; utility reached
4.6116 / 14.8%. Decision:
`retire_v2_utility_selector_not_better_than_random`.
Comparative all-expert credit is stored at
`reports/language_scaling/sparse-event-v2-comparative-utility-16m-20260710.json`;
it reached 4.6153 / 25.8% and still lost to random. The implementation, runner,
and tests are deleted.

The retired delta falsification and unseen-generation runners established the
same-tokenizer, frozen-schedule contract for recurrent replacement experiments.
Their code and rejected checkpoint are deleted after the 16.78M-token result;
compact local reports retain the measured loss, relation, generation, state,
throughput, and CUDA evidence.

### Retired distributed predictive-organism v1

The compact matched reports are local at:

- `reports/language_scaling/distributed-organism-finalist-4m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-finalist-4m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-durable-16m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-scaling-64m-20260710.json`.

At 16,785,792 fresh matched update tokens, the organism reached 4.5101 heldout
loss versus 4.6130, 96.9% candidate relation ranking versus 91.8%, and 28.1%
strict free relation generation versus 12.5%. It sustained 51,994 steady and
50,797 compile-amortized tokens/s versus 123,815 and 119,269 for the Transformer.
All candidate parameters received gradients. The run executed 1,417 compiled
ordinary steps and 202 eager probes, passed compiled/eager loss parity at a
0.00012 delta, and deleted its 272.2 MB schedule cache. Branch:
`test_organism_unseen_generation_before_any_promotion`.

The subsequent source-absent report is
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710.json`.
All six prompts were absent from all declared sources, but all twelve greedy and
sampled continuations failed semantic review. The explicit review is
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710-review.md`.
Branch: `no_promotion_scale_to_64m_and_retest_loss_slope`.

At 67,112,064 tokens the Transformer/organism losses were 3.8924/3.8949,
candidate relation was 98.0%/89.8%, strict free relation was 32.0%/31.6%, and
steady throughput was 110,345/33,963 tokens/s. V1's early advantage was gone,
99.8% of units remained active, and the population still received 60.6% of the
mix. The implementation, falsification runner, source-absence audit, tests,
schedule caches, and rejected checkpoints are deleted. Branch:
`retire_organism_v1_design_sparse_event_memory_v2`.

**`language_grounding_support.py`** — records whether prompt/source terms and
generation evidence exist for later grounded comparison. It does not prove
semantic grounding.

**`language_sustained_runtime_evidence.py`** — qualifies exact-checkpoint
Transformer generation with explicit aggregate tokens, consecutive tokens per
stream, elapsed time, throughput, bounded KV/decode history, pre/post model-state
hashes, finite full-vocabulary logits, bounded output hashes/previews, and
observed parameter-owning module execution. The qualification shape is 256
streams by 2,048 tokens under an expected checkpoint hash. The generic v4
surface reports the dense V39 path as zero structural sparsity; sustained speed
does not promote language quality.

The historical V40 run passes at 524,288/524,288 aggregate tokens: 256 streams each
continue for 2,048 tokens in 74.84 seconds at 7,005 tokens/s and 3.17 GB peak
allocation. Model hashes remain exact, KV stays at 72, invalid/non-finite counts
are zero, and hooks observe all 100,679,424 parameters. This qualifies dense
runtime stability while explicitly rejecting a sparsity or long-generation-
quality claim. Report SHA-256 is
`4757c0a0f0972fabe1de3e0b742f91a049f166994a9421d141c117a7ddcf2331`.

The deleted V41 hidden-state-memory falsifier built 65,536 training-only keys and
compared frozen V39, true values, and identical-key shuffled values. True memory
improves strict free accuracy only 50.00% to 51.56%, leaves ownership at 6.25%,
and lowers container to 20.31%; shuffled values collapse to 1.17%. General
gate-off logits and model tensors remain exact. The causal but unhelpful reader
performs 740.95M dense key comparisons and leaves no checkpoint or live code.
Decision: `retire_v41_hidden_state_memory_no_joint_free_binding_win`.

The V42 role-contrastive falsifier is deleted after a terminal execution stop.
Its vectorized objective and batch-256 gradient preflight were mechanically
valid, but the exact 32x8 eager pilot saturated the RTX 3060 for 16,507.6
seconds and persisted no arm result. Quality is unknown; no checkpoint exists.
Decision: `stop_v42_execution_infeasible_no_quality_conclusion`. Compact report
SHA-256 is
`9ecd6e1e4ba8e603624eb15797f9fe4f5a534388e2221401f9537c98286f7808`.
The shared matched runner now owns real-step wall-time projection and strict,
atomic per-arm artifacts. Unit checks cover projection, rejection, round-trip,
and contract-mismatch behavior. On the exact V39 100.68M checkpoint, three-step
BF16/Muon probes select effective batch 224 at 19.22k training tokens/s and
10,490,205,696 bytes peak allocation; batch 256 degrades to 3.83k tokens/s at
11,847,494,144 bytes. Every probe restores the exact model hash and the forced
budget rejection permits zero counted steps. This qualifies the infrastructure,
not cross-batch training quality. Compact report SHA-256 is
`284b35710e6b59572459a35ff9d79dd9f3a8b02921fbc7a6e3f4bb3d43884c15`.

V43's metrics-only prompt-copy audit stops before implementation. Under the
checkpoint-owned V39 BPE tokenizer, only 66.53% of correct-answer token IDs occur
anywhere in their prompt against the frozen 85% gate; event-order coverage is
57.84%, and complete-span coverage is zero. No runner, model, test, training, or
checkpoint exists. Decision:
`stop_v43_prompt_copy_insufficient_answer_token_coverage`. Compact report
SHA-256 is
`6b9580d3097d34fbd28b3edc49965ec0851026743ab98fba77fabc95fe9afc70`.

V44 finds and removes a decode-policy confound. Prompt-inclusive no-repeat-3
forbade factual answers from reusing source triples; the causal default/no-
controls/repetition-only/no-repeat-only sweep produces 50.00%/88.67%/87.11%/
51.56% strict free accuracy. Decode policy v4 sees generated continuation history
only. The unchanged V39 checkpoint now reaches 88.67% free and 98.44% ranked;
container/ownership/property/event-order is 60.94%/100%/100%/93.75%, and model
hashes remain exact. This requalifies relation decoding, not general grounding.
Decision: `promote_v44_generated_only_decode_controls_requalify_v39`. Compact
report SHA-256 is
`e413abd919fb25ea546046b76652c7e011666fa0b7c8ecda8e7a454bdb0b2315`.

V45 requalifies sustained runtime under that policy using the generic v4 report
contract. The exact V39 checkpoint completes 524,288/524,288 tokens across 256
streams in 73.1564 seconds at 7,166.67 tokens/s and 3,165,493,760 bytes peak
allocation. Model/checkpoint hashes remain exact, outputs are finite and in
vocabulary, state is bounded, and observed parameter coverage is 100%. There are
247 unique continuation hashes; previews still drift, so the evidence promotes
runtime stability but not long-generation quality or sparsity. Decision:
`qualify_same_checkpoint_sustained_runtime`. Compact report SHA-256 is
`51eefbbd66c8869217c4ca5a53fa1e5006f44887de028c654a1a3995d0572175`.

**`language_hf_curriculum_materializer.py`** — materializes bounded,
provenance-recorded Hugging Face dataset rows into local training corpora. A
materialized corpus is data evidence, not learned capability. For large public
shards it range-reads only the row groups needed by a bounded experiment and
records the source size/ETag plus the exact corpus hash. Full-shard requests
retain the stronger full-download SHA-256 path. Temporary parquet files are
always deleted after conversion, and each explicit document carries its actual
curriculum role rather than a diagnostic-only label.

## Current Branch Decision

The replay-controlled relation run and strict free-generation audits are stored
locally at:

- `reports/language_scaling/relation-replay-21m-16m-20260710.json`
- `reports/language_scaling/active-251m-free-binding-audit-20260710.json`
- `reports/language_scaling/relation-replay-21m-free-binding-audit-20260710.json`

| Metric | Active base | Replay candidate |
| --- | ---: | ---: |
| Candidate relation accuracy | 47.7% | 98.0% |
| Exact free relation accuracy | 0.0% | 44.9% |
| Free container persistence | 0.0% | 0.0% |
| Free ownership transfer | 0.0% | 10.9% |
| Free property persistence | 0.0% | 93.8% |
| Free event order | 0.0% | 75.0% |
| Mixed-language loss | 3.2534 | 3.2485 |

Replay prevents scalar-loss forgetting and improves free answers, but it does
not solve ownership/container binding and contaminates open generation with Q/A
templates. The candidate is rejected. The branch is
`replay_improves_candidate_ranking_not_free_binding`.

Frozen-base output adapters were then falsified and removed. Rank 32 reached
83.2% candidate / 3.1% exact free accuracy; rank 128 reached 84.4% / 2.3%.
Rank 128 trained only 131,072 parameters at ~151k tokens/s with a +0.0227 mixed
loss delta, but capability did not improve with rank and remained far below
full replay's 44.9% free accuracy. The branch is
`retire_output_adapter_test_selective_episodic_binding`.

The first prompt-level episodic memory interface was also falsified and removed.
With eight distractors and two slots, exact free accuracy was 18.4% without
memory, 12.1% random, 5.9% recency, 8.6% surprise, 11.7% full-store retrieval,
and 3.9% for the non-promotable oracle. Surprise also slowed throughput from
3.52 to 2.33 cases/s. Because even full/oracle retrieval hurt, prepending
retrieved text is the rejected interface—not evidence against all learned
episodic memory. The branch is
`retire_prompt_memory_build_answer_masked_post_training`.

Answer-masked relation post-training was then falsified and removed. The run
restored exact optimizer state from the active checkpoint and alternated 2,048
answer-only relation steps with 2,048 ordinary general replay steps:

| Metric | Active checkpoint | Answer-masked candidate |
| --- | ---: | ---: |
| Candidate relation accuracy | 47.7% | 87.1% |
| Exact free relation accuracy | 0.0% | 19.5% |
| Free container persistence | 0.0% | 1.6% |
| Free ownership transfer | 0.0% | 26.6% |
| Free property persistence | 0.0% | 7.8% |
| Free event order | 0.0% | 42.2% |
| Mixed-language loss | 3.2534 | 3.2684 |

The 4,096-step BF16 loop processed 10,621,968 tokens, applied loss to 8,688,968
tokens, took 312.4 seconds including milestone evaluation, and peaked at
1,258,403,840 allocated CUDA bytes. Scalar retention held, but exact free
generation missed the 60% criterion and underperformed full replay's 44.9%.
The candidate checkpoint and single-purpose runner are deleted; the local
evidence remains at
`reports/language_scaling/answer-masked-post-training-21m-4096-20260710.json`.
The branch is
`retire_answer_masked_post_training_build_integrated_pmrm_competitor`.

Those precursor runs did not test integrated PMRM. The complete architecture
was subsequently built, corrected for equal write budgets and episodic-key
gradient flow, and evaluated in the final screen below.

The final corrected 269,568-token screen is local at
`reports/language_scaling/pmrm-integrated-trainable-memory-screening-262k-20260710.json`:

| Arm | General loss | Candidate relation | Exact free | Train tokens/s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 7.4972 | 30.5% | 0.0% | 71,961 | 2.41 |
| PMRM surprise | 7.6701 | 24.2% | 0.0% | 2,777 | 9.45 |
| PMRM no memory | 7.6999 | 27.7% | 0.0% | 4,101 | 9.04 |
| PMRM random | 7.6600 | 21.9% | 0.0% | 2,771 | 9.45 |
| PMRM recency | 7.6571 | 23.8% | 0.0% | 2,830 | 9.45 |
| PMRM temporal only | 7.6557 | 18.4% | 0.0% | 3,838 | 5.95 |

The full PMRM used 20,977,792 parameters versus 20,976,128 for the Transformer;
all full-arm parameters received gradients. Surprise, random, and recency each
made 576 writes and 13,824 reads. Surprise lost to both naive selectors, and
recency did not meaningfully beat temporal-only. Every free score is zero, the
Transformer is materially better and about 26 times faster, and no screening
checkpoint was saved. The PMRM implementation, runner, and tests are retired.
Branch: `retire_integrated_pmrm_build_editable_delta_memory_competitor`.

### Editable delta-memory finalist

The matched delta reports are local at:

- `reports/language_scaling/delta-editable-screening-262k-20260710.json`;
- `reports/language_scaling/delta-editable-half-screening-262k-20260710.json`;
- `reports/language_scaling/delta-editable-half-screening-1m-20260710.json`;
- `reports/language_scaling/delta-editable-half-finalist-4m-20260710.json`;
- `reports/language_scaling/delta-editable-half-durable-16m-20260710.json`;
- `reports/language_scaling/delta-editable-half-unseen-generation-16m-20260710.json`.

Pure delta memory was dominated at 262k tokens (8.0018 loss), while adding one
local-attention layer reached 7.6833. The 2-delta/2-attention hybrid reached
7.5461 versus Transformer 7.4972 at 269,568 tokens, then 6.9042 versus 7.0625 at
1,057,536 and 5.6966 versus 5.9962 at 4,199,040. The last candidate relation
scores were 90.6% versus 73.8%; exact free scores were only 0.8% versus 0%.
Every hybrid parameter received a gradient, the parameter delta was 0.0049%,
and peak allocated VRAM was 3.90 GiB. The unfused reference sustained 8,021
tokens/s versus 82,960 for the Transformer.

At 16,785,792 tokens, the Transformer reached loss 4.5657, 98.4% candidate
relation, 17.2% exact free relation, and 83,505 tokens/s. The hybrid reached
4.5858, 87.9%, 7.8%, and 7,984 tokens/s. Its four verified source-absent prompt
continuations failed human semantic review, including an exact location-update
prompt. The early learning gain did not survive. Branch:
`retire_delta_v1_design_distributed_predictive_organism`.

### Narrow relation fine-tune

The 2026-07-10 controlled relation-binding falsification is stored locally at:

`reports/language_scaling/relation-binding-21m-16m-20260710.json`

| Metric | Before | After 16.78M relation tokens |
| --- | ---: | ---: |
| Overall relation accuracy | 47.7% | 87.9% |
| Container persistence | 20.3% | 100.0% |
| Ownership transfer | 92.2% | 100.0% |
| Property persistence | 48.4% | 100.0% |
| Event order | 29.7% | 51.6% |
| Mixed-language loss | 3.2534 | 8.7139 |

Candidate likelihood never reads the correct index; labels are metrics-only.
The result proves static relation learnability but also catastrophic forgetting.
The relation candidate is rejected and the 251,658,240-token mixed checkpoint
remains active. The branch is
`relation_learned_but_catastrophic_forgetting_test_replay`.

Next run a budgeted relation-plus-general replay mixture from the active base.
Require relation gain and bounded mixed-language loss together; keep event order
separate from the static-binding aggregate.

### Mixed-data continuation

The active 2026-07-10 mixed-data continuation is stored locally at:

`reports/language_scaling/mixed-cosmopedia-fineweb-21m-251m-continuation-20260710.json`

It restored exact optimizer state, trained on 75,000 fresh FineWeb-Edu and
75,000 fresh Cosmopedia records, and evaluated 10,000 disjoint records from
each source.

| Phase update tokens | Cumulative tokens | Mixed loss | Perplexity | Selected epochs | Repeated | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 150,994,944 | 3.6216 | 37.40 | 0.00 | 0 | — |
| 50,331,648 | 201,326,592 | 3.4429 | 31.28 | 0.41 | 0 | 64,878 |
| 100,663,296 | 251,658,240 | 3.2534 | 25.88 | 0.82 | 0 | 64,932 |

The general distribution improved, but relation binding did not. The decode
artifact at
`reports/language_scaling/mixed-cosmopedia-fineweb-21m-251m-decode-comparison-20260710.json`
shows that deterministic nucleus sampling changes style without recovering
entity/causal state. This is not Base-Language Qualification.

The branch is `falsify_relation_binding_before_more_generic_pretraining`.
The active 256.4 MiB checkpoint reloads with 251,658,240 cumulative tokens,
61,440 optimizer steps, and 26 AdamW state entries; SHA-256 is
`25e16893fd6bec4c8f7c858f7fc7bdd969e13fbe733104f4467d7f2f784a7fd3`.

### Structured-only continuation

The active 2026-07-10 document-disjoint continuation is stored locally at:

`reports/language_scaling/cosmopedia-v2-21m-150m-continuation-20260710.json`

It restored the prior 50,331,648-token weights/tokenizer, trained on 150,000
new shard-1 Cosmopedia records, and evaluated 10,000 shard-2 records that did
not train the checkpoint tokenizer.

| Phase update tokens | Cumulative tokens | Strict loss | Perplexity | Selected epochs | Repeated | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50,331,648 | 3.1289 | 22.85 | 0.00 | 0 | — |
| 50,331,648 | 100,663,296 | 2.9863 | 19.81 | 0.41 | 0 | 64,765 |
| 100,663,296 | 150,994,944 | 2.7681 | 15.93 | 0.82 | 0 | 64,785 |

All six unseen generations still reached 192 tokens. Some topical adherence
improved, but entity/property binding and causal closure still fail. This is a
better continual-pretraining base, not Base-Language Qualification. The branch
is `mix_fresh_structured_and_educational_web_data_at_21m`. The active 256.4 MiB
checkpoint reloads with 150,994,944 cumulative tokens, 36,864 optimizer steps,
26 AdamW state entries, and exact RNG/batch position; SHA-256 is
`7fcaa42ed2a32c2c4f2bbba60d632b9a4b78385852a6613141c77372a59998fd`.

### First structured-general ablation

The 2026-07-10 first Cosmopedia v2 structured-general ablation is stored
locally at:

`reports/language_scaling/cosmopedia-v2-21m-50m-20260710.json`

| Update tokens | Selected epochs | Heldout loss | Perplexity | Train tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 16,777,216 | 0.27 | 3.7038 | 40.60 | 61,823 |
| 33,554,432 | 0.55 | 3.2881 | 26.79 | 63,338 |
| 50,331,648 | 0.82 | 3.1318 | 22.91 | 63,906 |

The source owns 100,000 explicit general-structured-prose records and
84,770,600 BPE tokens. No selected update was repeated. All six unseen prompts
produced grammatical long-form text, but none emitted EOS before 192 tokens and
all lost entity, property, or causal state. This is not a general-language
quality promotion. The scientific branch is
`continue_21m_on_new_disjoint_structured_documents`; the generic single-arm
artifact string remains `continue_data_scaling_at_selected_model_size`.

The next continuation must train on new records and evaluate a separate shard
that did not participate in checkpoint-tokenizer training. The internal tail
holdout in this first ablation is adequate for the within-run curve, not for a
final cross-document qualification claim.

### Restricted coherence diagnostic

The 2026-07-10 explicit-record coherence diagnostic is stored locally at:

`reports/language_scaling/tinystories-21m-50m-diagnostic-20260710.json`

| Update tokens | Unique | Repeated | Heldout loss | Perplexity | Train tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16,777,216 | 16,777,216 | 0 | 2.1879 | 8.92 | 66,698 |
| 33,554,432 | 33,554,432 | 0 | 1.9520 | 7.04 | 67,013 |
| 50,334,464 | 50,334,464 | 0 | 1.8573 | 6.41 | 67,210 |

The split owns 250,001 training records and 21,991 evaluation records, including
one provenance header per file. All four unseen prompts produce grammatical,
prompt-conditioned multi-sentence continuations. The scientific branch is
`keep_transformer_scale_curated_general_curriculum`; the artifact's generic
single-arm branch string remains `continue_data_scaling_at_selected_model_size`.

This passes a restricted coherence diagnostic, not Base-Language Qualification.
Entity names, object properties, character roles, and causal closure still
fail, and general FineWeb-Edu prompts remain unqualified.

## Next Evidence Program

### 1. Relation-binding falsification

Continue a branch from the active checkpoint on a procedural entity/event
curriculum with compositionally disjoint names, colors, containers, transfers,
and event orders. Keep the mixed holdouts unchanged. Record:

- relation accuracy on unseen compositions;
- mixed heldout loss before/after the relation phase;
- training and evaluation token counts;
- unique versus repeated corpus tokens;
- unseen prompt continuations;
- exact checkpoint hash and restoration;
- observed CUDA throughput and peak memory.

Evaluate entity-name retention, object-property consistency, role consistency,
causal closure, and the original general prompts. Do not let a narrow story
pass substitute for general quality.

### 2. Local scaling grid

Fit, rather than assume:

`L(N,D) = E + A/N^alpha + B/D^beta`

Use approximately 5M, 20M, and the largest feasible 60-100M-class model on the
RTX 3060, several token budgets per size, and repeated seeds where the result
could change the branch. A two-point curve for one model is insufficient.

### 3. Distributed predictive competitor

Build one MARULHO-owned, vectorized population of small predictive units with
parallel bounded exact attention, recurrent multi-rate state, a small latent
episodic organ, and counterfactual future-utility credit. Match it against the
Transformer on tokenizer, corpora, tokens, parameters, optimizer, hardware, and
evaluation code. A PyTorch reference must prove causal forward/backward truth
before fused execution is claimed. Synthetic relation cases diagnose mechanisms;
real heldout language loss and open generation remain mandatory. See
`RESEARCH.md` for hypotheses and falsifiers.

### 4. Continual and sustained validation

From the same quality-qualified checkpoint:

- learn sequential domains;
- measure new-domain gain and old-domain forgetting;
- verify replay/retention;
- save and restore the learned checkpoint;
- measure active compute;
- complete 524,288 generated tokens.

## Evidence Rules

- A completed run is not automatically a positive result.
- Heldout loss outranks throughput for base architecture selection.
- Prompt pass counts do not hide fractured or copied continuations.
- Labels used for metrics cannot leak into prediction.
- CUDA and kernel claims require observed execution.
- Reports remain read-only artifacts.
- Historical recurrent/SNN language reports do not describe current runtime
  capability.
- Close each material experiment with `scale`, `redesign`, or `retire`.

## Commands

Inspect the maintained runner:

```powershell
python -m marulho.evaluation.language_training_experiment --help
```

Run the focused contract suite:

```powershell
python -m pytest -q `
  tests/test_language_transformer.py `
  tests/test_language_tokenizer.py `
  tests/test_language_training_experiment.py `
  tests/test_language_sustained_runtime_evidence.py `
  tests/test_language_generation_coherence.py `
  tests/test_language_grounding_support.py `
  tests/test_marulho_brain.py
```
