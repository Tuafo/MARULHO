# Evaluation

This package owns experiments and evidence reports. It does not own cognition
or mutate an installed `MarulhoBrain` during status reads.

## Maintained Language Experiments

**`language_training_experiment.py`** — the primary Transformer training
runner. It owns corpus loading, byte/BPE tokenizer creation, stratified fixed
windows, AdamW, warmup plus cosine decay, precision selection, heldout
evaluation, unseen prompt generation, optional sustained generation, and
checkpoint/report output. CUDA runs may explicitly request full-graph Inductor;
the runner restores RNG state, requires compiled/eager warm-up loss parity, and
reports both steady and compile-amortized throughput. Eager remains the default
because compilation can lose on short runs.

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
hashes are streamed rather than loaded into one large byte allocation. Optional
Inductor execution uses the same parity contract as the primary runner; wall
clock comparisons include compile time rather than comparing only steady-state
steps.

**`language_matched_support.py`** — shared mechanics for replacement falsifiers:
deterministic corpus range selection, full-batch filtering, mixed-source schedule
construction, one-time device staging, optimizer/gradient accounting, heldout and
label-safe relation evaluation, and optional post-arm diagnostics. Architecture
decisions remain owned by the specific runner; this support cannot promote a
model.

The retired depth-connection v9 evidence is local at
`reports/language_scaling/depth-connections-v9-falsification-16m-20260711.json`
and
`reports/language_scaling/depth-connections-v9-replication-seed7331-16m-20260711.json`.
Both 16.79M-token reports decide
`redesign_v9_disjoint_loss_and_behavior_signals`: signed learned reuse replicated
a small loss gain but not a reliable strict-generation gain or a joint win over
identity and fixed controls. All controls used exact resets, matched common
initialization, complete gradient and parity audits, and one shared candidate
graph with within-0.14% candidate throughput. No checkpoint was saved; the v9
model, runner, and tests are deleted.

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
silently trade away general language. Repeated `--replay-corpus` sources test
whether a budgeted general-data mixture preserves both capabilities before
parameter isolation or episodic memory is introduced.

**`language_relation_binding_audit.py`** — reruns both candidate-likelihood and
strict greedy free-answer metrics for any checkpoint against a frozen relation
case artifact. This catches cases where multiple-choice ranking improves while
open generation still loses the relation.

The retired modular predictive society v3 report is
`reports/language_scaling/modular-society-v3-falsification-16m-20260710.json`.
At 16.79M matched tokens, monolith/average/no-message/shuffled/real losses were
4.6140/5.0261/5.0460/5.0973/5.1073 and strict free relation scores were
14.5%/5.1%/2.0%/0.4%/0%. Compiled society controls sustained 74.4--74.6k
tokens/s with compile/eager loss deltas at or below 0.000026. Real communication
lost both required controls, so the model, runner, and tests are deleted.

The v4 report is
`reports/language_scaling/modular-workspace-v4-falsification-16m-20260710.json`;
its scientific interpretation is in the adjacent `-review.md`. Real exchange
reached 4.8507 loss / 21.5% free relation versus 4.8518 / 11.7% shuffled and
4.8549 / 10.2% no exchange. It did not pass the loss or monolith guard and is not
scaled.

The v5 report is
`reports/language_scaling/content-addressed-workspace-v5-falsification-16m-20260710.json`.
Monolith/no-exchange/shuffled/real losses were
4.6142/4.8526/4.8479/4.8494 and strict free relation scores were
17.2%/24.6%/22.7%/6.6%. Real associative memory lost both controls while all
workspace parameters received gradients and control throughput stayed matched.
The model, matched runner, and tests are deleted.

The retired v6 report is
`reports/language_scaling/hyperspherical-transformer-v6-falsification-16m-20260710.json`.
Transformer-standard/Transformer-native/normalized-standard/normalized-native
losses were 4.6144/4.6448/6.2844/4.7092 and strict free relation scores were
14.8%/0%/0%/0%. All parameters received gradients, parity and projection audits
passed, and throughput remained matched. No checkpoint was made; the failed v6
model, runner, and tests are deleted.

The retired v7 report is
`reports/language_scaling/dynamical-memory-v7-falsification-16m-20260711.json`.
Transformer/memory-off/single-scale/always-write/random-write/learned-write
losses were 4.6137/4.6092/4.6061/4.6076/4.6088/4.6066; strict free relation
scores were 21.5%/3.9%/10.5%/6.2%/3.5%/4.7%. Learned memory failed both the
Transformer quality guard and the simpler single-scale control. Its write gate
was active, every parameter received a gradient, compiled/eager parity passed,
and memory-control throughput varied by only 0.65%. The candidate sustained
112.7k training tokens/s versus the Transformer's 129.1k. The run used two loss
graph compiles rather than six and retained no checkpoint. Decision:
`retire_v7_no_quality_or_control_gain`; the failed model, runner, and tests are
deleted. Candidate-likelihood relation accuracy was 96.9% for learned memory
versus 93.0% for the Transformer, so its opposite free-generation result is also
evidence that candidate ranking cannot stand in for language generation.

The retired v8 reports are
`reports/language_scaling/depth-allocation-v8-falsification-16m-20260711.json`,
`reports/language_scaling/depth-allocation-v8-replication-seed7331-16m-20260711.json`,
and
`reports/language_scaling/depth-allocation-v8-durability-seed2026-67m-20260711.json`.
Early-heavy beat uniform twice at 16.79M tokens, then lost the successive-halving
durability comparison: uniform/early-heavy loss was 3.8861/3.8957 and strict free
relation tied at 20.3%. Each durable arm processed 67,112,064 tokens. Parameters,
common initialization, gradients, parity, and theoretical compute matched;
observed throughput differed by 0.30%. The excluded source tails were one partial
batch from each general corpus and are recorded in the report. Decision:
`retire_v8_early_heavy_not_durable`. No checkpoint was made; the model, runner,
and tests are deleted.

The retired integrated-PMRM runner established the architecture-neutral matched
experiment contract now used for replacements: same checkpoint-owned tokenizer,
frozen source-balanced schedule, parameters, optimizer, relation/general
evaluation, gradient coverage, state bytes, wall time, throughput, and CUDA
memory. Candidate screens cannot install a runtime model.

The retired sparse event-memory v2 reports are
`reports/language_scaling/sparse-event-v2-falsification-16m-20260710.json`.
Random-sparse reached 4.6128 loss / 27.0% strict free relation; utility reached
4.6116 / 14.8%. Decision:
`retire_v2_utility_selector_not_better_than_random`.
Comparative all-expert credit is stored at
`reports/language_scaling/sparse-event-v2-comparative-utility-16m-20260710.json`;
it reached 4.6153 / 25.8% and still lost to random. The implementation, runner,
and tests are deleted.

The retired delta falsification and unseen-generation runners established the
same-tokenizer, frozen-schedule contract for recurrent replacement experiments.
Their code and rejected checkpoint are deleted after the 16.78M-token result;
compact local reports retain the measured loss, relation, generation, state,
throughput, and CUDA evidence.

### Retired distributed predictive-organism v1

The compact matched reports are local at:

- `reports/language_scaling/distributed-organism-finalist-4m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-finalist-4m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-durable-16m-20260710.json`;
- `reports/language_scaling/distributed-organism-compiled-scaling-64m-20260710.json`.

At 16,785,792 fresh matched update tokens, the organism reached 4.5101 heldout
loss versus 4.6130, 96.9% candidate relation ranking versus 91.8%, and 28.1%
strict free relation generation versus 12.5%. It sustained 51,994 steady and
50,797 compile-amortized tokens/s versus 123,815 and 119,269 for the Transformer.
All candidate parameters received gradients. The run executed 1,417 compiled
ordinary steps and 202 eager probes, passed compiled/eager loss parity at a
0.00012 delta, and deleted its 272.2 MB schedule cache. Branch:
`test_organism_unseen_generation_before_any_promotion`.

The subsequent source-absent report is
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710.json`.
All six prompts were absent from all declared sources, but all twelve greedy and
sampled continuations failed semantic review. The explicit review is
`reports/language_scaling/distributed-organism-unseen-generation-16m-20260710-review.md`.
Branch: `no_promotion_scale_to_64m_and_retest_loss_slope`.

At 67,112,064 tokens the Transformer/organism losses were 3.8924/3.8949,
candidate relation was 98.0%/89.8%, strict free relation was 32.0%/31.6%, and
steady throughput was 110,345/33,963 tokens/s. V1's early advantage was gone,
99.8% of units remained active, and the population still received 60.6% of the
mix. The implementation, falsification runner, source-absence audit, tests,
schedule caches, and rejected checkpoints are deleted. Branch:
`retire_organism_v1_design_sparse_event_memory_v2`.

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

The replay-controlled relation run and strict free-generation audits are stored
locally at:

- `reports/language_scaling/relation-replay-21m-16m-20260710.json`
- `reports/language_scaling/active-251m-free-binding-audit-20260710.json`
- `reports/language_scaling/relation-replay-21m-free-binding-audit-20260710.json`

| Metric | Active base | Replay candidate |
| --- | ---: | ---: |
| Candidate relation accuracy | 47.7% | 98.0% |
| Exact free relation accuracy | 0.0% | 44.9% |
| Free container persistence | 0.0% | 0.0% |
| Free ownership transfer | 0.0% | 10.9% |
| Free property persistence | 0.0% | 93.8% |
| Free event order | 0.0% | 75.0% |
| Mixed-language loss | 3.2534 | 3.2485 |

Replay prevents scalar-loss forgetting and improves free answers, but it does
not solve ownership/container binding and contaminates open generation with Q/A
templates. The candidate is rejected. The branch is
`replay_improves_candidate_ranking_not_free_binding`.

Frozen-base output adapters were then falsified and removed. Rank 32 reached
83.2% candidate / 3.1% exact free accuracy; rank 128 reached 84.4% / 2.3%.
Rank 128 trained only 131,072 parameters at ~151k tokens/s with a +0.0227 mixed
loss delta, but capability did not improve with rank and remained far below
full replay's 44.9% free accuracy. The branch is
`retire_output_adapter_test_selective_episodic_binding`.

The first prompt-level episodic memory interface was also falsified and removed.
With eight distractors and two slots, exact free accuracy was 18.4% without
memory, 12.1% random, 5.9% recency, 8.6% surprise, 11.7% full-store retrieval,
and 3.9% for the non-promotable oracle. Surprise also slowed throughput from
3.52 to 2.33 cases/s. Because even full/oracle retrieval hurt, prepending
retrieved text is the rejected interface—not evidence against all learned
episodic memory. The branch is
`retire_prompt_memory_build_answer_masked_post_training`.

Answer-masked relation post-training was then falsified and removed. The run
restored exact optimizer state from the active checkpoint and alternated 2,048
answer-only relation steps with 2,048 ordinary general replay steps:

| Metric | Active checkpoint | Answer-masked candidate |
| --- | ---: | ---: |
| Candidate relation accuracy | 47.7% | 87.1% |
| Exact free relation accuracy | 0.0% | 19.5% |
| Free container persistence | 0.0% | 1.6% |
| Free ownership transfer | 0.0% | 26.6% |
| Free property persistence | 0.0% | 7.8% |
| Free event order | 0.0% | 42.2% |
| Mixed-language loss | 3.2534 | 3.2684 |

The 4,096-step BF16 loop processed 10,621,968 tokens, applied loss to 8,688,968
tokens, took 312.4 seconds including milestone evaluation, and peaked at
1,258,403,840 allocated CUDA bytes. Scalar retention held, but exact free
generation missed the 60% criterion and underperformed full replay's 44.9%.
The candidate checkpoint and single-purpose runner are deleted; the local
evidence remains at
`reports/language_scaling/answer-masked-post-training-21m-4096-20260710.json`.
The branch is
`retire_answer_masked_post_training_build_integrated_pmrm_competitor`.

Those precursor runs did not test integrated PMRM. The complete architecture
was subsequently built, corrected for equal write budgets and episodic-key
gradient flow, and evaluated in the final screen below.

The final corrected 269,568-token screen is local at
`reports/language_scaling/pmrm-integrated-trainable-memory-screening-262k-20260710.json`:

| Arm | General loss | Candidate relation | Exact free | Train tokens/s | Peak GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 7.4972 | 30.5% | 0.0% | 71,961 | 2.41 |
| PMRM surprise | 7.6701 | 24.2% | 0.0% | 2,777 | 9.45 |
| PMRM no memory | 7.6999 | 27.7% | 0.0% | 4,101 | 9.04 |
| PMRM random | 7.6600 | 21.9% | 0.0% | 2,771 | 9.45 |
| PMRM recency | 7.6571 | 23.8% | 0.0% | 2,830 | 9.45 |
| PMRM temporal only | 7.6557 | 18.4% | 0.0% | 3,838 | 5.95 |

The full PMRM used 20,977,792 parameters versus 20,976,128 for the Transformer;
all full-arm parameters received gradients. Surprise, random, and recency each
made 576 writes and 13,824 reads. Surprise lost to both naive selectors, and
recency did not meaningfully beat temporal-only. Every free score is zero, the
Transformer is materially better and about 26 times faster, and no screening
checkpoint was saved. The PMRM implementation, runner, and tests are retired.
Branch: `retire_integrated_pmrm_build_editable_delta_memory_competitor`.

### Editable delta-memory finalist

The matched delta reports are local at:

- `reports/language_scaling/delta-editable-screening-262k-20260710.json`;
- `reports/language_scaling/delta-editable-half-screening-262k-20260710.json`;
- `reports/language_scaling/delta-editable-half-screening-1m-20260710.json`;
- `reports/language_scaling/delta-editable-half-finalist-4m-20260710.json`;
- `reports/language_scaling/delta-editable-half-durable-16m-20260710.json`;
- `reports/language_scaling/delta-editable-half-unseen-generation-16m-20260710.json`.

Pure delta memory was dominated at 262k tokens (8.0018 loss), while adding one
local-attention layer reached 7.6833. The 2-delta/2-attention hybrid reached
7.5461 versus Transformer 7.4972 at 269,568 tokens, then 6.9042 versus 7.0625 at
1,057,536 and 5.6966 versus 5.9962 at 4,199,040. The last candidate relation
scores were 90.6% versus 73.8%; exact free scores were only 0.8% versus 0%.
Every hybrid parameter received a gradient, the parameter delta was 0.0049%,
and peak allocated VRAM was 3.90 GiB. The unfused reference sustained 8,021
tokens/s versus 82,960 for the Transformer.

At 16,785,792 tokens, the Transformer reached loss 4.5657, 98.4% candidate
relation, 17.2% exact free relation, and 83,505 tokens/s. The hybrid reached
4.5858, 87.9%, 7.8%, and 7,984 tokens/s. Its four verified source-absent prompt
continuations failed human semantic review, including an exact location-update
prompt. The early learning gain did not survive. Branch:
`retire_delta_v1_design_distributed_predictive_organism`.

### Narrow relation fine-tune

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

### 3. Distributed predictive competitor

Build one MARULHO-owned, vectorized population of small predictive units with
parallel bounded exact attention, recurrent multi-rate state, a small latent
episodic organ, and counterfactual future-utility credit. Match it against the
Transformer on tokenizer, corpora, tokens, parameters, optimizer, hardware, and
evaluation code. A PyTorch reference must prove causal forward/backward truth
before fused execution is claimed. Synthetic relation cases diagnose mechanisms;
real heldout language loss and open generation remain mandatory. See
`RESEARCH.md` for hypotheses and falsifiers.

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
