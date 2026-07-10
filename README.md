# MARULHO

MARULHO is a local research project for building a continual language system
whose model, tokenizer, memory, learning rules, checkpoints, and evaluation are
owned by MARULHO.

MARULHO is not currently an AGI or a frontier language model. It is an
experimental system running on a single RTX 3060, with a deliberately aggressive
research policy: matched experiments decide which mechanisms survive.

## Architecture

The active language base is a decoder-only causal Transformer implemented in
this repository. It uses:

- a checkpoint-owned BPE tokenizer trained on the selected corpus;
- RMS normalization, rotary positions, causal attention, SwiGLU, and a bounded
  per-layer KV cache;
- full-vocabulary next-token cross-entropy;
- checkpointed model and tokenizer state with no downloaded model weights;
- brain-owned generation through `MarulhoBrain`;
- heldout loss, unseen continuation, checkpoint fidelity, and sustained
  generation as separate measurements.

MARULHO now has one active baseline and one replacement architecture under
construction:

```mermaid
flowchart LR
    Data["Local text and grounded experience"] --> Tok["Checkpoint-owned BPE"]
    Tok --> Baseline["Active 21M Transformer baseline"]
    Tok --> Event["PMRM event encoder - reference candidate"]
    Event --> Router["Sparse fixed-pool column router"]
    Router --> Dual["Dual temporal and associative column state"]
    Dual <--> Episode["Internal surprise-budgeted episodes"]
    Dual <--> Workspace["Weight-shared recurrent workspace"]
    Episode <--> Workspace
    Baseline --> Compare["Matched language, binding, memory, and speed evidence"]
    Workspace --> Compare
    Compare --> Selected["Evidence-selected language model"]
    Selected --> Output["Unseen language generation"]
    Brain["MarulhoBrain"] --> Selected
    Brain --> Checkpoint["Atomic checkpoint and rollback"]
    Service["Thin /brain API"] --> Brain
```

Only the Transformer path is installed in `MarulhoBrain`. It remains the quality
and systems baseline. MARULHO now has a runnable PMRM reference candidate with
continuous recurrent state, not spikes; fixed persistent columns; selective
temporal state; editable delta-rule associative state; internal episodic memory;
sparse relations; and recurrent latent workspace iterations. It has correctness
tests but no quality evidence yet. Component switches explain the eventual
result, but a detached memory heuristic does not count as a PMRM test. Grounded
identity and intervention tasks from LCO can later evaluate whether surviving
columns represent persistent objects rather than text correlations.

## Current Evidence

The 2026-07-10 equal-time run selected the 21M model over the 63M model on the
RTX 3060: loss 4.0942 versus 4.6129 after 565.9 versus 560.8 seconds. A fresh
21M run then measured a three-point unique-data curve over 57.96M available
FineWeb-Edu BPE tokens and a disjoint holdout:

| Update tokens | Repeated | Heldout loss | Perplexity | Train time |
| ---: | ---: | ---: | ---: | ---: |
| 16,777,216 | 0 | 4.5754 | 97.07 | 232.6 s |
| 33,554,432 | 0 | 4.1328 | 62.35 | 462.8 s |
| 50,331,648 | 0 | 3.9889 | 54.00 | 693.9 s |

The final point uses 0.87 unique corpus epochs with zero repeated updates. This
is still not a quality promotion: continuations are more prompt-related, but
remain repetitive, sometimes malformed, and incoherent. The marginal loss gain
also contracts sharply in the last interval.

The coherence diagnostic passed. With 250,000 official TinyStories training
records, the complete 21,990-record validation split, and 50.33M unique updates,
the same 21M model reached loss 1.8573 / perplexity 6.41. All four unseen prompts
produced grammatical, prompt-conditioned multi-sentence stories; three emitted
EOS before the 192-token cap.

This does not promote general-language quality. Names still drift, object
properties contradict, character roles blur, and one story does not close. But
it falsifies basic architecture incapacity: the 21M MARULHO Transformer can
learn coherent English. The active blocker is a general curriculum that teaches
structured knowledge and consistency. The [TinyStories paper](https://arxiv.org/abs/2305.07759)
motivates the diagnostic; the next mixture follows the data lesson from
[Hugging Face's SmolLM work](https://huggingface.co/blog/smollm): structured
synthetic textbooks/stories plus deduplicated educational web data. The current
artifact is local at
`reports/language_scaling/tinystories-21m-50m-diagnostic-20260710.json`.

The first structured-general ablation used 100,000 explicit records from the
official [SmolLM-Corpus Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)
and the same 21M model. Heldout loss improved from 3.7038 at 16.78M updates to
3.2881 at 33.55M and 3.1318 at 50.33M. The final point covered only 0.82 of the
selected training stream, so the model is not saturated. However, all six
unseen continuations reached the 192-token cap and lost prompt state: objects
vanished, `cache` changed meaning, and causal explanations drifted into generic
textbook prose. This checkpoint is not general-language qualified. The branch
is to continue the same weights on new document-disjoint structured data with
a tokenizer-disjoint holdout, then decide from the longer curve whether to
scale data/model size or add memory/grounding machinery. The local artifact is
`reports/language_scaling/cosmopedia-v2-21m-50m-20260710.json`.

The next continuation restored those weights/tokenizer, trained on 150,000 new
Cosmopedia records from shard 1, and evaluated 10,000 tokenizer-disjoint records
from shard 2. Strict holdout loss fell from 3.1289 before the phase to 2.9863 at
100.66M cumulative updates and 2.7681 at 150.99M; neither 50M phase repeated a
selected training window. Prompt adherence improved slightly, especially for
rain/soil and server/data, but all six outputs still hit 192 tokens and entity/
causal binding still failed. The checkpoint is a better continual-pretraining
base, not a quality promotion. It now owns exact AdamW/scaler/RNG/batch state:
`reports/language_scaling/cosmopedia-v2-21m-150m-continuation-20260710-21m-checkpoint.pt`
(SHA-256 `7fcaa42ed2a32c2c4f2bbba60d632b9a4b78385852a6613141c77372a59998fd`).
The next ablation mixes fresh structured prose with educational-web text rather
than amplifying Cosmopedia's synthetic style alone.

That mixed continuation is now complete. It restored exact AdamW state, trained
on 75,000 fresh FineWeb-Edu plus 75,000 fresh Cosmopedia documents, and held out
10,000 documents from each source. Combined holdout loss fell from 3.6216 to
3.4429 at 201.33M cumulative tokens and 3.2534 at 251.66M, again with no
repeated selected updates. However, entity/causal binding stayed flat: notebook
ownership vanished, valve ordering collapsed into word association, and the
coin/cup relation drifted. A same-checkpoint decode ablation showed that seeded
temperature-0.8/top-p-0.9 sampling increased variety but did not recover those
relations, so greedy decoding was not hiding a capable model.

The active checkpoint is
`reports/language_scaling/mixed-cosmopedia-fineweb-21m-251m-continuation-20260710-21m-checkpoint.pt`
(SHA-256 `25e16893fd6bec4c8f7c858f7fc7bdd969e13fbe733104f4467d7f2f784a7fd3`).
It is a stronger pretraining base, not Base-Language Qualification. The next
step is a controlled, compositionally held-out relation-binding falsification:
determine whether this Transformer can learn persistent entities/events before
spending more compute on generic text or adding episodic memory.

The controlled relation falsification answered that question. From the active
base, 16.78M narrow relation tokens moved candidate-likelihood accuracy from
47.7% to 87.9%: container, ownership, and property tasks reached 100%, while
event order reached only 51.6%. This proves the 21M Transformer has capacity for
static binding under a focused objective. But the branch is rejected as a
checkpoint: mixed-language loss catastrophically regressed from 3.2534 to
8.7139, and free generation remained unreliable. The active checkpoint stays
the 251.66M-token mixed model.

Decision: test a budgeted relation-plus-general replay mixture from the active
base. If replay preserves general loss while retaining relation gains, build
continual curricula around replay/consolidation. If it cannot, compare
parameter-isolated adapters and PMRM-style surprise-selected episodic binding
against the same label-safe benchmark.

Replay preserved the mixed holdout and learned the candidate task: after a
roughly 20% relation / 80% fresh-general mixture, candidate accuracy reached
98.0% and mixed loss slightly improved from 3.2534 to 3.2485. A stricter
256-case free-generation audit prevented a false promotion. Exact free answers
improved from 0% to 44.9%, with property at 93.8% and event order at 75%, but
ownership reached only 10.9% and container persistence remained 0%. Original
open prompts also picked up benchmark-template contamination.

Decision: keep the 251.66M-token mixed checkpoint active and reject the replay
candidate. This result motivated the subsequently falsified output-adapter
comparison and the move to PMRM-style selective episodic binding.

The frozen-base output-adapter line is now falsified and deleted. Rank 32
trained 32,768 parameters and reached 83.2% candidate / 3.1% exact free
accuracy. Rank 128 trained 131,072 parameters and reached 84.4% / 2.3% despite
~151k training tokens/s, ~681 MiB peak allocated VRAM, and only +0.0227 mixed
loss regression. More rank did not translate latent ranking into free binding.

Decision: `retire_output_adapter_test_selective_episodic_binding`. This led to
the subsequent prompt-memory comparison of surprise-selected, random, and
recency retrieval under equal budgets. The failed adapter code and checkpoints
are not maintained.

The first PMRM-inspired prompt-memory interface was also falsified and deleted.
Under eight distractors and a two-slot budget, exact free accuracy fell from
18.4% with no memory to 12.1% random, 5.9% recency, and 8.6% surprise. Full-store
retrieval reached 11.7% and the non-promotable oracle only 3.9%; surprise also
reduced throughput from 3.52 to 2.33 cases/s. The failure is therefore the
“retrieve text and prepend it” interface, not merely surprise selection.

Decision: `retire_prompt_memory_build_answer_masked_post_training`. This result
falsifies prompt-text retrieval only. It did not implement or test the integrated
PMRM architecture.

Answer-masked post-training has now also been run and retired. Starting from the
active checkpoint with exact AdamW state, 4,096 BF16 updates alternated 50%
answer-only relation batches with 50% ordinary general-language replay. Mixed
heldout loss changed only from 3.2534 to 3.2684, while candidate accuracy rose
from 47.7% to 87.1%. Strict free answers reached just 19.5%: container 1.6%,
ownership 26.6%, property 7.8%, and event order 42.2%. This is below the 60%
criterion and below full replay's 44.9%, so the objective does not close the
ranking/generation gap. The 268.8 MB rejected checkpoint and experiment code are
deleted; the compact report remains at
`reports/language_scaling/answer-masked-post-training-21m-4096-20260710.json`.

Decision: `retire_answer_masked_post_training_build_integrated_pmrm_competitor`.
The active checkpoint remains the 251.66M-token Transformer. The next model is a
coherent PMRM recurrent competitor, evaluated both end-to-end and through
matched temporal-only, associative-only, memory-policy, routing, and workspace
ablations. PMRM itself remains untested until that system exists.

The first integrated PMRM screen now exists. Eight fresh 20.98M-parameter arms
shared the checkpoint tokenizer, a hashed 20% relation / 40% FineWeb-Edu / 40%
Cosmopedia schedule, and 269,568 update tokens. The matched Transformer reached
loss 7.4972 at 76,519 tokens/s. Full PMRM-surprise reached 7.6460 at 2,997
tokens/s; temporal-only was the best PMRM loss at 7.6309, associative-only was
7.7484, and reducing the workspace from two iterations to one regressed loss to
7.6823. Every arm remained at 0% strict free relation generation, so this is
screening evidence only. PMRM used 9.45 GiB peak allocated VRAM versus 2.41 GiB
for the Transformer training arm.

The screen also found an invalid H2 comparison: surprise, random, and recency
had equal slots/reads but different permanent write counts. Their loss ordering
therefore cannot select a memory policy. The code now closes this confound with
one causal permanent write per 16-event block for every policy: surprise keeps
the highest-error past event, random a reservoir sample, and recency the last.
The compact exploratory report remains at
`reports/language_scaling/pmrm-integrated-screening-262k-20260710.json`; no
screening checkpoint was retained. Decision:
`repair_equal_write_budget_rerun_pmrm_policy_screen`.

## Research Objective

MARULHO aims to find a local architecture that is better than a conventional
larger model at a clearly measured task, rather than pretending to reproduce a
frontier model's parameter count on consumer hardware.

The priority order is:

1. Keep the 21M Transformer as the reproducible local baseline and active
   checkpoint, not the assumed final architecture.
2. Build one coherent PMRM v0 with fixed persistent columns, dual temporal and
   associative state, internal episodic memory, and a recurrent workspace.
3. Compare the complete model and its in-place ablations against the Transformer
   on the same tokenizer, data, tokens, parameters, hardware, and evaluations.
4. Require open relation generation, real-language loss, memory growth, latency,
   and throughput together; synthetic mechanism wins do not qualify the model.
5. Scale only a surviving architecture across model size, data, and context.
6. Demonstrate sequential-domain learning with bounded forgetting and exact
   checkpoint/resume state.
7. Re-establish measured active compute and a 524,288-token sustained run from
   the same quality-qualified checkpoint.
8. Test grounded object identity, action binding, and intervention transfer
   before structural growth or a Transformer-replacement claim.

The initial scaling model is the standard decomposition
`L(N,D) = E + A/N^alpha + B/D^beta`, measured locally over multiple parameter
and token budgets. Continual-memory experiments will extend it with memory
capacity and online-compute terms only when those variables exist in working
code.

## Ownership Boundaries

- `MarulhoBrain` owns language-model installation, generation, lifecycle, and
  durable checkpoint state.
- `src/marulho/training` owns model and optimization machinery.
- `src/marulho/evaluation` owns experiments and reports; reports do not mutate
  the runtime.
- `src/marulho/service` is a thin adapter and does not own cognition.
- No hidden external LLM, Cortex loop, or ThoughtLoop generates MARULHO output.
- External papers and datasets may inform training, but model weights remain
  MARULHO-owned.
- Throughput, report count, or prompt pass count alone never proves capability.

## Repository Map

- `CONTEXT.md`: current domain language and research decisions.
- `src/marulho/brain`: brain-owned runtime and Transformer installation.
- `src/marulho/training/language_model.py`: active language model contract.
- `src/marulho/training/language_transformer.py`: causal Transformer state
  block and streaming KV state.
- `src/marulho/data/language_tokenizer.py`: byte and BPE tokenizers.
- `src/marulho/evaluation/language_training_experiment.py`: maintained
  training/evaluation runner.
- `src/marulho/evaluation/language_scaling_experiment.py`: matched local
  model-size/token-budget curves and provisional scaling-law fit.
- `src/marulho/evaluation/language_generation_coherence.py`: unseen
  continuation evaluation.
- `src/marulho/evaluation/language_sustained_runtime_evidence.py`: bounded
  sustained generation.
- `src/marulho/core`: separate grounded sparse/column experiments.
- `MARULHO_UI`: local control-room UI.

## Setup

Requirements:

- Python 3.10+
- PyTorch-compatible CPU or CUDA environment
- Node.js only for the optional UI

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .[dev]
pytest
```

Inspect the maintained training runner:

```powershell
python -m marulho.evaluation.language_training_experiment --help
```

Example bounded local run:

```powershell
python -m marulho.evaluation.language_training_experiment `
  --corpus reports/language_curriculum/fineweb-edu-train-20k-20260710.txt `
  --eval-corpus reports/language_curriculum/fineweb-edu-eval-2k-offset20k-20260710.txt `
  --output reports/language_training/local-transformer.json `
  --tokenizer-kind bpe `
  --tokenizer-vocab-size 8192 `
  --device auto
```

Run the local API from a brain checkpoint:

```powershell
python -m marulho.service.server --checkpoint checkpoints/marulho/model.pt --port 8000
```

Generated reports and model checkpoints are ignored local artifacts unless a
specific result is promoted into the documentation.

## License

No open-source license has been selected. The public repository is available
for inspection but does not grant reuse rights beyond GitHub's default terms.
