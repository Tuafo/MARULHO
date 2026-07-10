# Evaluation

This package owns experiments and evidence reports. It does not own cognition
or mutate an installed `MarulhoBrain` during status reads.

## Maintained Language Experiments

**`language_training_experiment.py`** — the primary Transformer training
runner. It owns corpus loading, byte/BPE tokenizer creation, stratified fixed
windows, AdamW, warmup plus cosine decay, precision selection, heldout
evaluation, unseen prompt generation, optional sustained generation, and
checkpoint/report output.

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
hashes are streamed rather than loaded into one large byte allocation.

**`language_generation_coherence.py`** — evaluates checkpoint generation on
explicit or source-anchored unseen prompt cases. It records text evidence and
source-continuation loss. Automated passes are diagnostic and do not alone
promote quality.

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
silently trade away general language.

**`language_grounding_support.py`** — records whether prompt/source terms and
generation evidence exist for later grounded comparison. It does not prove
semantic grounding.

**`language_sustained_runtime_evidence.py`** — measures exact-token
Transformer generation from a checkpoint, including elapsed time, throughput,
bounded KV configuration, and checkpoint hash. Sustained speed does not promote
language quality.

**`language_hf_curriculum_materializer.py`** — materializes bounded,
provenance-recorded Hugging Face dataset rows into local training corpora. A
materialized corpus is data evidence, not learned capability. For large public
shards it range-reads only the row groups needed by a bounded experiment and
records the source size/ETag plus the exact corpus hash. Full-shard requests
retain the stronger full-download SHA-256 path. Temporary parquet files are
always deleted after conversion, and each explicit document carries its actual
curriculum role rather than a diagnostic-only label.

## Current Branch Decision

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

### 3. Adaptive memory

Only after base-language qualification, compare surprise-selected episodic
memory with:

- no external memory;
- full recent KV history;
- random retained episodes;
- simple recency;
- equal storage and retrieval compute.

Fast associative weights are a later ablation, not bundled into the first
memory test.

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
