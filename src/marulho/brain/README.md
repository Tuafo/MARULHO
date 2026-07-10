# Brain

`MarulhoBrain` is the runtime owner. It composes the language cortex, the
separate grounded trainer, checkpoint state, compact traces, and the local
lifecycle exposed to the service.

## Language Cortex

An installed `BrainLanguageModelRuntime` owns exactly one
`MarulhoLanguageModel` and its matching checkpoint-owned tokenizer.
`MarulhoBrain.generate()` routes through `active_language_path =
marulho_transformer` and returns:

- generated token IDs and decoded continuation;
- tokenizer and vocabulary identity;
- decode-control evidence;
- Transformer state-core identity;
- explicit MARULHO ownership and no-external-LLM flags.

If no Transformer checkpoint is installed, the old local transition readout is
available only as a small grounded-runtime fallback. It is not a language-model
quality path.

## Checkpoints

Brain save/restore includes the Transformer configuration, strict model tensors,
tokenizer state, evaluation metadata, and reviewed installation records.
Restoration rejects a legacy language-runtime surface.

`install_language_checkpoint_from_direct_review()` is the bounded mutation
surface for a local checkpoint. It requires operator approval, an expected
SHA-256 hash, a matching file hash, and the Transformer v2 checkpoint contract.

## Sustained Generation

`generate_sustained_language()` delegates token generation to the
training/evaluation-owned Transformer runner, then records brain-owned trace and
checkpoint context. It does not convert throughput into a quality claim.

## Ownership Rules

`MarulhoBrain` owns:

- model installation and active-path selection;
- source buffering and grounded trainer ticks;
- generation, replay/growth hooks, and the background loop;
- compact `BrainTrace` history;
- durable brain checkpoint state.

It does not own:

- optimizer or Transformer algorithms, which stay in `training`;
- experiments and report selection, which stay in `evaluation`;
- HTTP/UI shape, which stays in `service` and `MARULHO_UI`;
- hidden external model generation;
- status-triggered learning or mutation.

## Focused Validation

```powershell
python -m pytest -q tests/test_marulho_brain.py
```

These tests cover installation, generation, direct checkpoint review,
save/restore, sustained generation, and service restoration. They validate
ownership and fidelity, not language quality.
