# Training

This package owns MARULHO model execution, optimization machinery, and
checkpoint serialization. The installed language runtime is Transformer-only;
replacement candidates must use the same causal-language protocol and earn
promotion through matched evidence.

## Active Language Modules

**`language_transformer.py`** — MARULHO's decoder-only causal Transformer
state block. It owns RMSNorm, rotary positions, causal attention, SwiGLU, and
bounded per-layer KV state for incremental decoding.

**`language_protocol.py`** — the shared causal-language model interface used by
matched evaluations. The active Transformer and every replacement candidate
meet this seam; service/runtime installation remains a separate promotion
decision.

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

The integrated PMRM reference, runner, and tests were deleted after the final
corrected screen. Full PMRM remained behind the matched Transformer and did not
meaningfully beat temporal-only despite higher state, compute, and memory cost.
Its surprise selector also lost to random and recency under identical write and
read budgets. No PMRM compatibility code remains.

The editable delta-memory v1 model, falsification runner, generation audit, and
tests are also deleted after durable falsification. Its 2/2 hybrid beat the
Transformer at 1.06M and 4.20M tokens, then lost heldout loss, free relation
recall, throughput, and unseen semantic generation at 16.78M. New work starts
from the distributed multi-timescale hypothesis in `RESEARCH.md`, not from a
delta compatibility surface.

**`language_model.py`** — the language model contract. It owns:

- `LanguageModelConfig`;
- token embeddings and tied full-vocabulary LM head;
- full-vocabulary next-token cross-entropy;
- greedy generation with repetition and no-repeat controls;
- tensor-indexed stratified fixed-window train/eval splits whose contract hashes
  are computed on CPU before one-way device transfer;
- exact pre-window text-token counts emitted by the split builder, avoiding a
  second full-corpus tokenizer pass in experiment reports;
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
