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

## Planned Adaptive Architecture

These concepts are hypotheses and must not appear as implemented capabilities.

**Adaptive Episodic Memory** — the first post-quality PMRM-inspired experiment.
Store a bounded subset of surprising episodes and retrieve them for relevant
continuations. Compare against no memory, full KV history, random memory, and
simple recency under equal storage and compute.

**Fast Associative Weights** — a later delta-rule memory candidate. Test only
after episodic selection earns its complexity.

**Continual Language Learning** — sequential domain updates from a
quality-qualified base checkpoint, with old-domain, new-domain, and replay
losses measured before and after. The required result is new learning with
bounded forgetting and restored checkpoint fidelity.

**Structural Plasticity** — changing model or memory structure under a
checkpointed transaction. It is paused for language until a simpler fixed
Transformer plus memory demonstrates quality. No old routed-expert expansion
path is maintained.

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
- old SNN language readout ledgers as a generation architecture.

Historical reports may mention these terms. New code, status, and documentation
must not present them as active capability.

## Decision Order

1. Clean and validate the Transformer-only runtime.
2. Select 21M as the current compute-optimal size from equal-time evidence.
3. Pass the 21M TinyStories coherence falsification.
4. Continue the curated explicit-record general-language curriculum at 21M on
   new documents and test entity, property, role, and causal consistency on a
   tokenizer-disjoint holdout.
5. Fit the first defensible local size/data scaling law with at least three
   model sizes and repeated seeds near a branch boundary.
6. If quality qualifies, test surprise-selected episodic memory.
7. Rebuild continual learning and retention measurement.
8. Re-establish sustained 524,288-token generation from the same checkpoint.
9. Add grounded causal experiments.
10. Scale, redesign, or retire each mechanism from evidence.
