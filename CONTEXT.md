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
Current evidence does not support coherent general language, frontier
capability, continual learning, or a superior scaling law.

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
