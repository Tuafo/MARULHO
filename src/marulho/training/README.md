# Training

This package owns MARULHO model execution, optimization machinery, and
checkpoint serialization. The maintained language path is Transformer-only.

## Active Language Modules

**`language_transformer.py`** — MARULHO's decoder-only causal Transformer
state block. It owns RMSNorm, rotary positions, causal attention, SwiGLU, and
bounded per-layer KV state for incremental decoding.

**`language_model.py`** — the language model contract. It owns:

- `LanguageModelConfig`;
- token embeddings and tied full-vocabulary LM head;
- full-vocabulary next-token cross-entropy;
- greedy generation with repetition and no-repeat controls;
- tensor-indexed stratified fixed-window train/eval splits whose contract hashes
  are computed on CPU before one-way device transfer;
- exact pre-window text-token counts emitted by the split builder, avoiding a
  second full-corpus tokenizer pass in experiment reports;
- heldout loss and perplexity;
- atomic Transformer checkpoint save/load.

**`checkpointing.py`** — the broader `MarulhoTrainer` checkpoint lifecycle
used by `MarulhoBrain`.

The active language path is `marulho_transformer`; the only accepted
`state_core` value is `transformer`.

## Checkpoint Contract

`marulho_transformer_language_checkpoint.v2` contains:

- exact `LanguageModelConfig`;
- strict model tensor state;
- complete byte or BPE tokenizer state;
- tokenizer vocabulary hash;
- metadata and ownership flags.

The tokenizer vocabulary must exactly match the model vocabulary. Legacy
recurrent, routed, spiking, sampled-vocabulary, and padded-vocabulary
checkpoints are rejected rather than upgraded through compatibility code.

Checkpoint writes use a temporary file, flush and fsync the payload, then
atomically replace the target.

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
