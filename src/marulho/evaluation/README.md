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
keeps later-offset holdout documents entirely outside training.

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

The 2026-07-10 disjoint-holdout FineWeb-Edu scale run is stored locally at:

`reports/language_scaling/fineweb-edu-19m-vs-60m-16m-20260710.json`

| Model | Update tokens | Heldout loss | Train tokens/s | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| 20,976,128 | 4,194,304 | 5.6115 | 70,624 | 1.44 GiB |
| 20,976,128 | 16,777,216 | 4.7018 | 72,210 | 1.44 GiB |
| 62,924,544 | 4,194,304 | 5.5344 | 29,388 | 2.71 GiB |
| 62,924,544 | 16,777,216 | 4.6177 | 29,490 | 2.71 GiB |

The 62.9M arm wins at equal tokens by 0.0841 loss. The 21.0M arm is about 2.45
times faster, while scaling data across the measured range improves the 62.9M
arm by 0.9167 loss. The data gain is 10.89 times the equal-token size gain. The
artifact branch is `scale_data_at_selected_model_size`; the next experiment
must compare the sizes at equal wall-clock or approximate training compute
before declaring a local compute optimum.

The result does not promote language quality. General data removed the earlier
single-domain template collapse, but all four unseen continuations remain
repetitive or weakly related to the prompt. Coherent unseen multi-sentence
continuation is the active blocker.

## Next Evidence Program

### 1. Compute-normalized size decision

Compare the 21.0M and 62.9M models under the same RTX 3060 wall-clock or
approximate training FLOPs, using the same tokenizer, unique-token stream,
disjoint holdout, and prompt set. Equal-token comparisons alone over-reward the
larger, slower model for the local-compute objective.

### 2. Quality scale run

Train the compute-optimal Transformer with the maintained BPE tokenizer and
record:

- heldout loss at multiple token budgets;
- training and evaluation token counts;
- unique versus repeated corpus tokens;
- unseen prompt continuations;
- exact checkpoint hash and restoration;
- observed CUDA throughput and peak memory.

### 3. Local scaling grid

Fit, rather than assume:

`L(N,D) = E + A/N^alpha + B/D^beta`

Use approximately 5M, 20M, and the largest feasible 60-100M-class model on the
RTX 3060, several token budgets per size, and repeated seeds where the result
could change the branch. A two-point curve for one model is insufficient.

### 4. Adaptive memory

Only after base-language qualification, compare surprise-selected episodic
memory with:

- no external memory;
- full recent KV history;
- random retained episodes;
- simple recency;
- equal storage and retrieval compute.

Fast associative weights are a later ablation, not bundled into the first
memory test.

### 5. Continual and sustained validation

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
