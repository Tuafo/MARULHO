# Training

This package owns MARULHO model execution, optimization machinery, and
checkpoint serialization. The installed language runtime remains Transformer-
only; PMRM is an isolated architecture candidate until matched evidence earns
promotion.

## Active Language Modules

**`language_transformer.py`** — MARULHO's decoder-only causal Transformer
state block. It owns RMSNorm, rotary positions, causal attention, SwiGLU, and
bounded per-layer KV state for incremental decoding.

**`language_protocol.py`** — the shared causal-language model interface used by
matched evaluations. The Transformer and PMRM are the two adapters at this
seam; service/runtime installation remains a separate promotion decision.

**`language_pmrm.py`** — the integrated continuous-state PMRM reference model.
One deep module owns event encoding, honest dense-cost top-k routing into a
fixed column pool, coupled selective temporal and delta-rule associative state,
sparse relation messages, hidden episodic memory, a weight-shared recurrent
workspace, full-vocabulary language loss/generation, runtime-state
reset/serialization, and an atomic PMRM-only checkpoint. Configuration switches
produce temporal-only, associative-only, fusion, memory-policy, and workspace
ablations without parallel implementations. It does not import the grounded
SNN/column runtime and cannot be installed by `MarulhoBrain`.

The causal column/memory scan remains sequential. The per-event recurrent
workspace is scratch state, so training flattens all scanned event summaries
and executes workspace layers in large batched tensors; streaming generation
uses the same one-event implementation. Surprise writes use the prior column
prediction, keeping memory causal without serializing the expensive workspace
through the scan.

Budget-matched surprise, random, and recency policies commit exactly one hidden
event per completed `episodic_write_interval`. Surprise retains the maximum
prior-prediction error within the past block, random retains the maximum of a
deterministic random priority (a reservoir sample), and recency retains the last
event. The candidate buffer is tensor-only runtime state and is checkpointed;
it is not readable as episodic memory until the block closes.

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
runtime `state_core` value is `transformer`. The experimental PMRM path is
`marulho_pmrm_v0` and uses a distinct checkpoint surface.

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

`marulho_pmrm_language_checkpoint.v1` follows the same tokenizer/hash and atomic
write discipline while also allowing exact tensor-only recurrent runtime state.
The Transformer loader rejects it, so an experimental candidate cannot silently
replace the active brain model.

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
