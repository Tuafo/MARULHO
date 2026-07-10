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
only the best checkpoint by default. An optional explicit evaluation corpus
keeps later-offset holdout documents entirely outside training. Its separate
`empirical_wall_clock` budget basis applies recorded per-arm throughput
multipliers, reports unique and repeated updates, and rejects the comparison
when actual training times differ by more than 15 percent. Unequal-token runs
cannot be mislabeled as scaling-law grids. Repeated `--corpus` arguments train
one tokenizer and split over multiple provenance-recorded corpus shards. A
single arm is a maintained data-scaling curve and emits
`continue_data_scaling_at_selected_model_size`, not a fictitious size winner.

**`language_generation_coherence.py`** — evaluates checkpoint generation on
explicit or source-anchored unseen prompt cases. It records text evidence and
source-continuation loss. Automated passes are diagnostic and do not alone
promote quality.

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

### 1. General quality scale run

Train the 21M model with a provenance-recorded mixture of structured synthetic
textbooks/stories and educational web text. Every source must use explicit
document records. Record:

- heldout loss at multiple token budgets;
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
