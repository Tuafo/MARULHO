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
only the best checkpoint by default.

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
materialized corpus is data evidence, not learned capability.

## Current Branch Decision

The 2026-07-10 matched BPE CUDA pilot is stored locally at:

`reports/language_architecture_bakeoff/nvidia-open-bpe-transformer-pilot-20260710.json`

| Model | Parameters | Update tokens | Heldout loss | Train tokens/s |
| --- | ---: | ---: | ---: | ---: |
| MARULHO Transformer | 5,249,280 | 4,194,304 | 6.0762 | 38,269.5 |
| Dense GRU | 3,812,352 | 4,194,304 | 6.6979 | 18,590.3 |

The Transformer also beat the final GRU loss after only 1,048,576 update
tokens. The explicit branch is
`retire_recurrent_language_base_scale_transformer`.

The result does not promote language quality. Diverse unseen generation remains
`0/4`; coherent unseen multi-sentence continuation is the active blocker.

## Next Evidence Program

### 1. Quality scale run

Train a materially larger Transformer with the maintained BPE tokenizer and
record:

- heldout loss at multiple token budgets;
- training and evaluation token counts;
- unique versus repeated corpus tokens;
- unseen prompt continuations;
- exact checkpoint hash and restoration;
- observed CUDA throughput and peak memory.

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
