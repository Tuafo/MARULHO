# MARULHO Research Notebook

This is the living notebook for architecture hypotheses. It is intentionally
allowed to change quickly. `CONTEXT.md` remains the source of truth for what is
implemented and supported by evidence; this file records what might be built
next and why.

## How to read this notebook

- **Observed** means MARULHO or a linked local project measured it.
- **Borrowed** means prior research reported it; MARULHO has not validated it.
- **Hypothesis** means a proposed connection or mechanism.
- **Retired** means local evidence was strong enough to stop that path.
- Every architecture hypothesis needs an experiment that could kill it.

Ideas are not commitments. A mechanism survives only if it improves behavior,
not because it is biologically attractive, mathematically elegant, or new.

## Inspiration is not architecture

SNNs, cortical columns, the Thousand Brains theory, Hopfield networks, LCO,
Transformers, state-space models, and symbolic systems are research lenses. None
is a requirement and none is accepted wholesale.

- SNNs ask whether activity and compute can be event-driven and sparse. MARULHO's
  former SNN language implementation failed; spike-inspired conditional activity
  can still be tested without restoring that implementation.
- Cortical-column and
  [Thousand Brains](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full)
  work asks whether many parallel reference-frame models can reach useful
  consensus. The biological theory is a hypothesis, not a software blueprint.
- Classical and
  [modern Hopfield networks](https://arxiv.org/abs/2008.02217) ask how partial or
  noisy cues can settle into stored patterns. Modern Hopfield updates are closely
  related to attention, so “using Hopfield memory” is not automatically a path
  beyond Transformers.
- LCO asks how many limited local units, communication, specialization, and
  lifecycle rules can organize. Its broad emergence thesis remains unproved even
  though some narrow mechanisms are useful.
- Transformers show the value of exact content-addressed lookup. Their success is
  evidence to explain, not a design prohibition.
- [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893) and the
  [shared global workspace](https://arxiv.org/abs/2103.01197) are borrowed
  evidence that modules can specialize and coordinate through limited channels
  on some tasks. They motivate controls; they are not language-model evidence
  for MARULHO.
- [Perceiver](https://arxiv.org/abs/2103.03206) shows that a learned latent
  bottleneck can compress large inputs. V4/V5 borrow the bottleneck question, not
  the Perceiver architecture or its claims.

The experimental method is to extract a computational question, implement the
smallest faithful mechanism inside a coherent system, and compare behavior. A
failed implementation retires that mechanism, not every future idea that shares
one word with it.

## Current observations

1. The active 20.98M Transformer is MARULHO's strongest language baseline, but
   it is not coherent enough to be called a capable general language model.
2. SNN, GRU, routed-column, PMRM, output-adapter, and narrow post-training paths
   did not beat the matched Transformer on the full quality problem.
3. Integrated PMRM did not make token surprise a useful write signal. Under
   equal write/read budgets, surprise lost to random and recency, and the full
   memory stack did not meaningfully beat its temporal-only control.
4. The 2-delta/2-attention model learned faster at 1.06M and 4.20M tokens, but
   the advantage disappeared by 16.78M. It then had worse heldout loss, worse
   free relation recall, failed unseen semantic generation, and trained about
   ten times slower than the Transformer reference.
5. Exact recent retrieval and compressed editable memory solve different
   problems. Removing too much exact access destroys facts that a recurrent
   state cannot reconstruct reliably.
6. Lower loss or higher multiple-choice ranking does not prove usable memory.
   MARULHO has repeatedly observed lower proxy loss without correct free-form
   recall.
7. The distributed organism produced a large matched loss advantage at 4.20M
   tokens in both eager and compiled runs. The strict compiled reproduction
   reached 5.5257 versus 6.0113, with 98.4% versus 72.7% candidate relation
   ranking. Both arms still scored 0% strict free relation generation. This is
   evidence to continue, not evidence of behavioral memory or durable
   superiority.
8. Unlike delta v1, the organism advantage survived 16.78M matched tokens. It
   reached 4.5101 heldout loss versus 4.6130, 96.9% versus 91.8% candidate
   relation ranking, and 28.1% versus 12.5% strict free relation generation.
   This is the first durable positive replacement result, but source-absent
   semantic generation still decides whether it represents usable language
   capability rather than a stronger corpus/task fit.
9. That source-absent audit failed all twelve continuations. The model can now
   learn the matched distribution better and freely answer more relation cases,
   but it still cannot reliably compose causal, narrative, abstract, physical,
   or procedural continuations outside the source prompts. Durable predictive
   advantage and general language capability are separate results.
10. The 64M falsification confirmed the crossover. Organism v1 reached 3.8949
    loss versus 3.8924, 31.6% strict free relation versus 32.0%, and only 33,963
    tokens/s versus 110,345. Its early advantage was a low-data inductive bias,
    not a scalable replacement. Dense event computation is retired from the
    base token mixer.
11. Sparse event v2 preserved the exact stream and recovered a relation benefit:
    random one-of-four specialists reached 27.0% strict free relation versus
    14.5% exact-only at tied heldout loss. The first utility router fell back to
    14.8% despite slightly better scalar loss. Chosen-expert-only short-horizon
    feedback is insufficient to learn expert ranking.
12. Comparative all-expert probes repaired utility free relation from 14.8% to
    25.8%, but random remained better at 27.0% and lower loss. Next-event token
    utility is not a sufficient coordination currency for this sidecar.
13. V17 cleanly rejects an all-active grouped GRU sidecar. Its full-rank state
    and complete gradients tie off/local/dense loss after 33.56M tokens per arm,
    while costing about 20% throughput. The V16 synthetic small-bank advantage
    does not transfer to optional within-window language recurrence.
14. V19/V19b show that jointly trained bounded memory tokens can carry source
    information, but not enough of it. Recurrent and partitioned banks reach
    30.1% and 31.4% paired source-following, roughly tied with mean pooling and
    more than sixteen points behind exact history. Normalization, full
    gradients, and bank partitioning do not repair the compression bottleneck.
15. V20 separates addressing from compression. No fixed top-one key passes the
    preregistered retrieval gate, but lexical top-two contains the exact episode
    in 98.83% of cases while reading half of the four-record history. That
    observation admits a language test without promoting TF-IDF as a general
    memory system.
16. V21 is the first admitted memory architecture in this iteration. A jointly
    trained cortex with lexical top-two exact retrieval reaches 51.6% free exact
    and 52.0% paired source-following, beating all-history's 39.5% and 38.0%
    while reading 96 rather than 192 source tokens. General loss stays inside
    the retention bound. Wall time is tied and the task is relation-specific,
    so the result advances to causal general-document streams rather than to
    runtime installation or Base-Language Qualification.
17. V22 finds that an older same-document episode is genuinely useful but an
    unconditional reader is not. Oracle-one improves loss by 0.0341 with a
    positive paired interval. Lexical-one retrieves the episode 75.0% of the
    time yet ties local-only overall: correct reads gain 0.0372, while wrong
    reads lose 0.1050. High lexical margin raises precision to 95.3% at 50%
    coverage. Retrieval therefore needs calibrated abstention, not a larger
    fixed top-k.
18. V22b shows that correctness confidence is transferable but is the wrong
    optimization target. The frozen lexical gate reaches 97.84% precision and
    gains 0.0356 loss, while equal-mask controls lose; always-on lexical still
    gains more at 0.0388. The gate removes useful low-margin evidence and misses
    its preregistered advantage. Detached abstention is retired; the next test
    co-adapts the cortex to selected and distracting document contexts.
19. V23 demonstrates co-adapted source use without a promotable system. Oracle
    gains 0.0417 over off, and true history beats a distractor by 0.0833 inside
    the lexical-trained model. Lexical's aggregate +0.0192 interval still
    crosses zero, target inclusion is 69.92%, and general loss regresses
    0.1200/0.1346. The next bounded test combines top-two recall with 50% general
    replay; another failure ends raw prompt-style document memory.
20. V24 shows that replay balance fixes forgetting but top-two does not fix
    retrieval. Lexical-two is 0.0064 worse than top-one despite strong source use.
    The lexical-one control itself gains 0.0255 over off with a positive interval
    and preserves general loss. Because balanced random-one was absent, one
    fresh-seed top-one replication is required before promotion or retirement.
21. V25 replicates exact top-one memory on both corpora: +0.0430 over off with a
    positive interval, +0.1127 true-vs-wrong source use, and bounded retention.
    Yet all eight free continuations fail anchored review. Raw concatenation is
    closed; the next reader must keep evidence separate and improve generation,
    not merely teacher-forced likelihood.
22. V26 rejects final-layer cross-attention despite complete gradients. Oracle
    gain is only 0.00010, true-vs-wrong evidence is zero, and the learned gate
    stays near 0.119. The separation idea is not falsified, but evidence must
    enter before later cortex computation. The next bounded test interleaves a
    shared reader between early/middle V11 layers.
23. V27 rejects that bounded test. Raw context gains 0.0426 over gate-zero with
    a positive interval, but both lexical and oracle interleaved readers lose
    about 0.0392. Oracle true-vs-wrong gain is only 0.0062 with an interval
    crossing zero. Both gates and every tensor train; the interface, not dead
    machinery or retrieval, fails. Cross-attention document memory is retired.

## Exploratory reset after V27

V27 closes the local evidence-reader neighborhood. The next search changes the
computational substrate rather than moving another gate or attention layer.
Candidates share one matched language contract; novelty does not excuse weaker
heldout loss, free generation, gradient coverage, or compute accounting.
MARULHO is architecture-agnostic: small units, columns, spikes, organisms,
particles, monolithic cores, and hybrids are hypotheses rather than identity.
An idea survives only when its behavior and cost survive matched falsification.

### First branch: particle-field recurrent core

- **Borrowed:** [Dragon Hatchling / BDH-GPU](https://arxiv.org/abs/2509.26507)
  expresses a large population of positive neuron-like activations through three
  low-rank factor matrices shared over recurrent depth. Causal linear attention
  is the parallel form of a Hebbian fast-weight state. The mechanism is closer
  to LCO's many-small-units intuition than a collection of independent neural
  modules: global meaning is a sparse population pattern and its evolving
  correlation state.
- **Evidence boundary:** the paper's matched scaling experiment uses a
  stateful raw-byte Europarl language/translation stream, 1.2B training tokens,
  and Transformer-XL controls. It is not evidence of broad web-language
  coherence. The public reference is a simple quadratic short-context kernel;
  MARULHO must measure the RTX 3060 implementation directly.
- **MARULHO candidate:** width 256, 24,576 nonnegative particles, four heads,
  eight shared recurrent-depth iterations, three particle factor matrices, a
  tied 8,192-token embedding/head, and no external weights. This gives about
  20.972M parameters versus the 20.976M matched Transformer, without inventing a
  capacity advantage.
- **Required truth:** causal parallel/reference agreement, recurrent-state
  agreement on small shapes, complete gradients, observed activation sparsity,
  exact parameter/operation accounting, and CUDA memory/throughput precede the
  language run.
- **Falsifier:** train particle and Transformer arms on the identical frozen
  corpus, tokenizer, batches, token count, optimizer intent, and seed. A branch
  advances only by improving heldout language and free relation behavior
  together at a durable budget. It then faces genuinely unseen generation
  before any checkpoint. A short-budget learning-rate advantage is not enough.
- **Result:** retired. At 16,777,728 tokens the particle/Transformer heldout
  loss was 4.9132/4.3193 and exact free relation generation was 11.33%/40.23%.
  Particle throughput was 11.1k versus 92.6k tokens/s and peak CUDA allocation
  was 5.36 GB versus 0.60 GB. Both arms had complete gradients and the particle
  arm reached 100% metrics-only candidate ranking, so neither dead machinery
  nor task ignorance explains the weak free language. This implementation of
  the population-field hypothesis is not promising enough for local tuning.
  The code and tests are deleted; the durable report retains the result.

### Second branch: learning geometry before another substrate

- **Reason:** changing the architecture while holding a weak optimizer fixed can
  reject useful models for the wrong reason. Muon has primary evidence of
  roughly twofold compute efficiency over AdamW and additional experiments in
  the 30M--200M regime, close enough to MARULHO's 21M scale to justify a direct
  local falsifier.
- **Mechanism:** keep the entire Transformer fixed. For hidden weight matrices,
  replace coordinate-wise Adam moments with momentum whose update is
  approximately orthogonalized by Newton-Schulz iteration. Keep AdamW for the
  tied token embedding and norms. This tests a different geometry of learning,
  not more parameters, labels, modules, or data.
- **Controls:** cross AdamW/Muon with both the historical 3e-4 and reference
  1e-3 peak rates from a common initialization. Compare the best rate per
  optimizer only after every arm sees the same 16.78M tokens. Require complete
  gradients, optimizer-state accounting, loss, and label-free generation.
- **Result:** the 1e-3 Muon arm passes the durable joint gate. Against same-rate
  AdamW at 16.78M tokens, heldout loss is 4.0961 versus 4.2606 and exact free
  relation generation is 17.58% versus 5.47%. Muon uses 40% less optimizer
  state but trains about 42% slower. At 3e-4, Muon improves loss slightly while
  harming generation, so optimizer and learning rate interact rather than
  producing a universal gain. V29 advances to exact reproduction and unseen
  review, not installation; two relation kinds remain at zero free accuracy.
  Reproduction strengthens the result at loss 4.0955 and 26.95% free relation
  and produces a bit-exact strict checkpoint. Unseen review still fails all
  eight source cases: controlled Cosmopedia is readable but generic and
  semantically unstable, while FineWeb is often repetitive or nonsensical.
  Therefore Muon survives as better learning geometry, not as proof that the
  current Transformer/curriculum is sufficient. V30 should remove the synthetic
  relation task from base-language optimization and test a longer paragraph
  context before adding memory or another substrate.

### Third branch: general-first context

- **Question:** did the 20% synthetic relation curriculum and 72-token window
  teach fast template completion at the expense of paragraph language?
- **Falsifier:** train fresh, exactly initialized Muon models with zero relation
  updates at context 72/batch 32 and context 256/batch 9. Both consume 2,304
  tokens per update and the same 16.78M total tokens from identical general
  source ranges. Compare both on V29's common context-72 heldout batches.
- **Selection:** require a 0.05 common general-loss gain over the strict V29
  checkpoint. Prefer context 72 unless context 256 adds at least 0.02 more gain,
  because longer quadratic attention is a cost rather than a capability by
  declaration. Relation behavior is recorded but base-language selection no
  longer optimizes a synthetic task.
- **Boundary:** only a selected, bit-exact checkpoint may face the same unseen
  FineWeb-Edu/Cosmopedia suite. Readable but generic output is still a failure.
- **Result:** general72 wins. Common V29/general72/general256 loss is
  4.0955/4.0093/4.0258, so removing synthetic relation updates helps and longer
  context alone does not. Both candidates lose free relation completely.
  FineWeb-Edu/Cosmopedia source loss improves by 0.1151/0.0387, but all unseen
  cases still fail and text remains unstable. The next scale point uses a
  256 MiB, 16-range sample from each replay shard and stratifies selected token
  windows across the resulting full-source spans for one fresh approximately
  67M-token pass at context 72. Repeating the 16M subset would not be credible
  evidence.

### V31 result: scaling works, base quality is still blocked

- **Mechanical truth:** 29,128 distinct batches process 67,110,912 tokens. Each
  source contributes 14,564 unique batch indices; 16 byte ranges and the
  selected token windows span each source. All parameters receive a final
  gradient, compiled/eager loss differs by 0.0000496, and strict reload is
  bit-exact.
- **Scaling result:** common V30/V31 heldout loss is 4.0093/3.6291 and
  perplexity is 55.11/37.68. The 0.3802 gain decisively clears the 0.15 gate.
  V31 sustains 56.1k tokens/s, uses 96.0 MiB optimizer state, and peaks at
  593.6 MiB CUDA allocation.
- **Unseen result:** FineWeb-Edu loss improves 4.4801→4.2053 and Cosmopedia
  3.8488→3.4896, but both greedy suites remain 0/4. Repetition controls raise
  Cosmopedia distinct-bigram fraction from 0.667 to 0.960 without grounding the
  continuation. Direct prose is more locally coherent but remains generic,
  repetitive, and prone to invented or unstable facts.
- **Decision:** `retain_v31_scaling_curve_expand_unique_data_not_base_quality`.
  At 3.2 update tokens per parameter, the model is still far from a decisive
  data-scaling test. Build a much larger unique-data point before judging this
  base, but keep architecture search independent: this result does not make the
  Transformer, small units, or any metaphor part of MARULHO's identity.

### V32 preregistration: third data-scaling point

- **Why scale again:** V31 used only 3.2 update tokens per parameter. The
  direction predicted by [compute-optimal scaling](https://arxiv.org/abs/2203.15556)
  and the heavy overtraining used by modern small models such as
  [SmolLM2](https://arxiv.org/abs/2502.02737) both make an architecture verdict
  at that ratio premature. These references motivate the direction, not an
  imported universal constant for MARULHO's data or hardware.
- **Frozen model:** 20,976,128 parameters, context 72, tied 8,192-token BPE,
  Muon 1e-3, exact V31 initialization seed, general-only causal loss, and the
  same FineWeb-Edu/Cosmopedia holdout. V31 is evaluation-only.
- **Fresh data:** five disjoint parquet shards supply 201,323,520 scheduled
  tokens in 87,380 steps. Each source contributes exactly 17,476 unique batches
  and 40,264,704 tokens. Every byte selection and stratified token-window set
  must span its source; any repeated index invalidates the run.
- **Kill rule:** require at least 0.20 heldout-loss gain over V31, every
  parameter receiving a gradient, compiled/eager parity, and bit-exact strict
  reload. Only then run the unchanged unseen suite. Better loss with unstable
  prose remains a scaling result, not base qualification.
- **Architecture boundary:** the parallel candidate remains a current editable-
  state hybrid such as Gated DeltaNet plus local attention. It is not a return
  to the retired delta loop and does not require modular units. V32 establishes
  the stronger control that such a replacement must beat.

### V32 result: diminishing return closes fixed-21M data scaling

- **Valid run:** all 87,380 steps complete, processing 201,323,520 tokens. Each
  of five disjoint sources contributes 17,476 unique batches; byte and token
  coverage audits pass. Every parameter receives a final gradient, V31 loss
  reproduces exactly, and compiled/eager loss differs by 0.000103.
- **Result:** V31/V32 heldout loss is 3.6291/3.4983 and perplexity is
  37.68/33.06. The 0.1308 gain is positive but misses the preregistered 0.20
  requirement. Throughput remains 56.2k tokens/s, optimizer state 96.0 MiB, and
  peak CUDA allocation 593.6 MiB. Candidate likelihood rises to 48.8%, but
  metrics-only relation behavior is not a selection criterion and free exact
  generation remains 0%.
- **Decision:** `stop_v32_general_scaling_no_durable_loss_gain`. Changing the
  gate after observing 0.1308 would be post-hoc. No checkpoint and no unseen
  review are admitted. The compact report remains as the third scaling point;
  rematerialized raw text is deleted because its manifests can reproduce it.
- **Next branch:** redesign the fixed 21M core. The leading bet is a genuinely
  current chunk-parallel editable-state/local-attention hybrid, compared against
  V31 under matched parameters, tokens, initialization, optimizer, and data.
  This is neither a defense of Transformers nor a return to small-unit
  modularity.

### Other orthogonal branches

- **Modern editable matrix state:**
  [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) separates channel-wise
  decay, erase, and write, while [Mamba-3](https://arxiv.org/abs/2603.15569)
  adds complex oscillatory state and multiple inputs/outputs. MARULHO's retired
  delta v1 already implements almost the same asymmetric erase/write equation,
  but with a serial reference loop and a small-state recipe. This family is a
  control candidate only if implemented with a current chunk-parallel block and
  current training recipe; the old path will not be restored under a new name.
- **Adaptive recursive computation:**
  [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) shares parameters
  over depth while assigning tokens different iteration counts. This is
  materially different from MARULHO's rejected static depth allocation and
  shallow depth-reuse weights. Its falsifier must show that routed extra
  iterations beat fixed-recursion and shuffled-routing controls at matched
  training FLOPs, not merely at matched parameter count. It is not V29: at the
  paper's smallest 135M equal-FLOP comparison, MoR remains slightly worse than
  the full Transformer, making a 21M implementation a lower-priority bet than
  first repairing the shared training geometry.
- **Dynamic byte patches:**
  [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) is relevant to a
  future tokenizer replacement, but its reported advantage emerges at much
  larger model/data scales and its local byte encoder/decoder is a large fixed
  overhead for MARULHO's current 21M--36M regime. It is not the next 3060 test.

### Self-extending causal computation

The attached Autogenic Causal Compiler discussion contributes a useful long-term
hypothesis: preserve explicit execution receipts, localize contradictions by
counterfactual replay, introduce a new latent distinction only when it repairs
heldout interventions, and compile repeated transferable traces into reusable
operations. This connects LCO's causal-object work with LCWM's strongest V9
diagnosis: candidate programs should be selected because executing them works.

It is not a credible replacement for base language modeling yet. A first test
belongs in a grounded interactive world where observations and interventions can
falsify a newly invented predicate. Text-only next-token loss cannot establish
concept birth, causal truth, or safe self-modification. MARULHO therefore keeps
this as a later execution-coupled causal organ, with an immutable ledger,
versioned edits, shadow evaluation, and rollback—not as V28's token mixer.

## Provisional scaling diagnosis

The 4.20M and 16.79M fresh matched points imply different local slopes against
log update tokens. Transformer loss fell from 6.0113 to 4.6130, about -1.009 per
natural-log token unit. Organism loss fell from 5.5257 to 4.5101, about -0.733.
The organism remains ahead, but its margin shrank from 0.4857 to 0.1029. Extending
these two straight lines predicts a crossover near 24.4M tokens.

This is deliberately not called a scaling law: two points cannot establish
curvature or asymptotic behavior, and each budget used a fresh frozen schedule.
The 64M experiment resolved that prediction: the losses tied within 0.0025, free
relation also tied within 0.4 percentage points, and organism throughput fell to
30.8% of the baseline. This falsifies v1 as the scalable base mixer. It does not
falsify a sparse memory specialist that leaves the exact stream intact, because
v1 never tested that interface: its population consumed about 60% of every
block's mix and almost every unit remained active.

## Current systems opportunity

V1's eager implementation was launch-bound rather than arithmetic-bound: a profiled
training step spent about 737 ms in CPU/dispatch work versus 181 ms of CUDA
kernel time, across hundreds of small matrix, multiply, copy, and batch-matrix
operations. Its optimized path materialized auxiliary tensors only for probes
or requested telemetry and had parity tests for loss, state, and every gradient.

A diagnostic `torch.compile` run is substantially more promising but is not yet
scientific quality evidence. The first uncached full candidate took about 343
seconds to compile once; after warm-up, a synthetic fixed-shape
forward/backward loop reached roughly 104k tokens/s. A later real-data backend
smoke compiled one full graph per arm, used an explicit probe schedule, and kept
probe steps eager. The Transformer compiled in 36.2 seconds and sustained 112.7k
tokens/s; the organism compiled in 144.8 seconds and sustained 41.4k tokens/s
across seven compiled steps and one eager probe. Full-model BF16 compiled/eager
loss deltas were 0.000037 and 0.000155, below the 0.001 rejection tolerance.
Compile cost, steady training time, and amortized throughput are reported
separately, and full-graph mode fails rather than silently falling back. The
matched 4.20M compiled reproduction retained the loss conclusion: 5.5257 versus
6.0113, within 0.0007 and 0.0021 of the eager results. Organism steady throughput
rose from 20.4k to 50.3k tokens/s; compile-amortized throughput was 45.8k versus
105.2k for the Transformer. At 64M the candidate fell to 34.0k versus 110.3k,
showing that compile fixed dispatch overhead but could not fix dense event-path
cost. The v1 runner is retired; strict full-graph parity and separate
compile/steady/amortized timing remain requirements for v2.

## What LCO contributes

LCO tests whether many locally interacting, individually limited units can
organize useful behavior. Its current module map is:

```text
experiment config
    -> scripts/run_experiment.py
    -> lco/evals/benchmarks.py
    -> lco/envs/gridworld.py
    -> PopulationCPU / PopulationTorch
       -> local state, traces, memory, traits, communication, lifecycle
```

The key LCO evidence is narrower than the inspiration:

- local communication helps when the world actually contains neighbor-required
  information, but is harmful as a universal default;
- fast/slow local traces are a useful substrate baseline;
- a protected shared identity organ can be hosted stably, but the former living
  repair advantage was withdrawn after matched controls;
- lifecycle, mutation, and specialization are not yet a broad capability win;
- the new nonliving causal table validates separated movement/intervention and
  object-action binding, while the living causal organ remains unimplemented.

The transferable idea is therefore **distributed specialization under measured
local utility**, not simulated metabolism for its own sake. MARULHO should not
copy LCO's deaths, births, energy, or grid topology into language until each one
has a language-specific purpose.

## Research fronts worth combining

### Exact and compressed sequence state

- **Borrowed:** attention gives strong content-addressed exact retrieval but its
  working memory grows with context.
- **Borrowed:** recurrent fast weights and state-space models keep bounded state.
  [Fast Weights](https://arxiv.org/abs/1610.06258),
  [Mamba-3](https://arxiv.org/abs/2603.15569), and
  [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) offer different update
  rules.
- **Borrowed:** large-scale hybrid evidence from
  [Kimi Linear](https://arxiv.org/abs/2510.26692) and the July 2026
  [linear-attention comparison](https://arxiv.org/abs/2607.07953) argues against
  assuming that pure recurrence should replace every attention layer.
- **Observed:** MARULHO's serial delta/attention hybrid showed a real early
  learning gain but lost it later. Alternating specialists in series may damage
  information before another specialist can use it.
- **Tested requirement:** the memory must solve an information dependency the
  exact local window cannot solve. V18 removed raw source tokens before a query;
  V19/V19b then trained bounded states jointly with the cortex. Both designs
  carried measurable source information, but exact history remained decisively
  better.
- **First result:** V18a's exact-history reader barely beats a source-independent
  local adapter on greedy answers, while the learned slots collapse to effective
  rank 2.01. Candidate ranking is contaminated by answer-template clues. V18b
  keeps the negative report, normalizes every learned write, and evaluates
  identical-question/source-swap pairs. Only source-following behavior can now
  advance the branch.
- **Final bounded-state result:** V18b repairs state scale but not organization
  or use. V19's jointly trained recurrent state and V19b's partitioned banks
  also lose to simple pooling and exact history. The frozen bridge and latent
  memory-token interface are retired.
- **Selected direction:** V20/V21 move compression out of episode content and
  into the index. An exact-token archive plus bounded lexical top-two selection
  beats both local-only and indiscriminate all-history on the controlled binding
  task. The next falsifier replaces relation templates with causal,
  document-disjoint language and must improve continuation loss and anchored
  generation together.

### Explicit read/write memory

- **Borrowed:** Neural Turing Machines and the
  [Differentiable Neural Computer](https://www.nature.com/articles/nature20101)
  show that learned controllers can address and modify external memory.
- **Borrowed:** object-centric
  [Slot Attention](https://arxiv.org/abs/2006.15055) shows that competitive slots
  can bind to entities and generalize to new compositions.
- **Observed:** MARULHO's surprise-selected prompt memory, PMRM slots, and later
  learned latent-token banks failed. The missing ingredient was not merely
  capacity; lossy writes discarded distinctions that the downstream query
  needed.
- **Current design:** the local cortex remains bounded, while an external
  append-only episode archive retains exact token spans, provenance, and compact
  retrieval keys. V21 validates this division of labor on controlled relation
  binding. V22 shows that true general-document episodes help but retrieval
  errors are asymmetrically costly. V22b then shows that same-document
  confidence does not predict marginal utility well enough to improve on
  always-read lexical selection. V23 co-training creates genuine source use but
  loses retention and lacks a significant aggregate win. V24 restores retention
  but rejects top-two distraction; its lexical-one control is significant. The
  final raw-context replication wins likelihood but fails every anchored sample.
  Exact memory therefore remains promising while raw concatenation is retired.
  V26 shows that final-layer reading is too late, and V27 shows that two earlier
  gated reads still cannot exploit even oracle evidence. Exact history remains
  useful under raw context, but no read interface is admitted and no
  checkpoint/index contract exists. Further memory work waits for a stronger
  base-language architecture or a fundamentally different execution mechanism.

### Multiple learning timescales

- **Borrowed:** complementary learning-systems research separates fast episodic
  learning from slow generalization and consolidation. The computational case
  dates at least to
  [McClelland, McNaughton, and O'Reilly (1995)](https://doi.org/10.1037/0033-295X.102.3.419).
- **Borrowed:** [Nested Learning](https://arxiv.org/abs/2512.24695) interprets
  activations, fast weights, optimizer state, and ordinary weights as nested
  memories operating at different update rates.
- **Borrowed:** [In-Place TTT](https://arxiv.org/abs/2604.06169) reports that a
  next-token-aligned fast-weight objective works better than generic
  reconstruction for test-time adaptation.
- **Borrowed:** [FOREVER](https://arxiv.org/abs/2601.03938) schedules continual
  replay using model-change signals rather than a fixed wall-clock rhythm.

### Predictive state and world models

- **Borrowed:** predictive-state representations define hidden state through
  predictions of future observable events instead of an unconstrained latent.
  See [Hilbert-space PSRs](https://arxiv.org/abs/1309.6819).
- **Observed:** LCWM's retained evidence suggests typed roles and paths can help
  structured composition, but its V9 result remained below the promotion gate.
  Its strongest lesson is to select a latent program by whether executing it
  improves the downstream result.
- **Hypothesis:** a unit's state will be more useful and auditable if it is
  trained to predict several future horizons or answer classes, rather than only
  contributing an opaque residual to the next token.

### Conditional computation

- **Borrowed:** [Adaptive Computation Time](https://arxiv.org/abs/1603.08983)
  lets a model spend more internal steps on difficult transitions.
- **Borrowed:** sparse experts activate different parameters for different
  inputs, but routing can collapse or become a load-balancing exercise rather
  than a capability mechanism.
- **Observed:** MARULHO's old routed columns did not earn survival. A router must
  be trained on marginal usefulness, not only similarity or balanced traffic.

### Behavioral memory evaluation

- **Borrowed:** the July 2026
  [deployment-memory evaluation](https://arxiv.org/abs/2607.00368) demonstrates
  that one-step updates can lower losses while free-form recall remains zero.
- **Observed:** this mirrors MARULHO's ranking/generation gap. Every future
  memory claim must test later recall, paraphrase, conflict replacement,
  locality, retention, and downstream use after support context is removed.

## Retired architecture hypothesis: distributed predictive organism v1

V1 tested a single end-to-end language system made from many
small predictive units and several memory timescales. No single unit contains
the meaning. Meaning is the coordinated pattern of unit states, exact memories,
messages, disagreements, and actions.

The name is descriptive, not a claim that the model is alive.

### 1. Exact recent workspace

A bounded local-attention workspace preserves exact recent tokens and latent
events. It is the lossless notebook. It prevents the failure seen when a small
recurrent state is forced to compress every name, location, and relation.

The workspace is deliberately bounded. It is not the system's long-term memory.

### 2. Population of small predictive units

Each unit owns:

- a small recurrent state;
- fast and slow traces;
- a specialist projection or update rule;
- a prediction proposal at one or more future horizons;
- a confidence estimate;
- a learned utility trace;
- a small communication budget.

Units operate in vectorized groups, not Python objects. Groups can specialize in
syntax, entities, temporal change, causal relations, uncertainty, or other
regularities, but no specialization label is supplied. Specialization must be
diagnosed after learning.

Unlike the rejected serial hybrid, exact and recurrent paths receive the same
input in parallel. A learned mixer combines their proposals. This lets the
recurrent path compress persistent regularities without forcing exact retrieval
through the compression bottleneck.

### 3. Sparse shared episodic organ

A bounded latent key/value store keeps a small number of exact older events. It
is not prompt text and is not an unbounded database. Writes include context,
time, source, and a version/conflict trace so that “the key is in the drawer” can
later be superseded by “the key is in the jar” without deleting the fact that a
move occurred.

The organ reads into hidden state. It never inserts oracle labels or answers into
the prediction path.

### 4. Utility is the common currency

The central new idea is **counterfactual utility credit**.

During a small fraction of training batches, the system compares future loss
and behavioral predictions with a unit, message, read, or proposed write present
versus masked. The difference becomes a delayed target for a cheap utility
predictor. Normal execution then uses the predictor once, without running both
counterfactual branches.

This changes the questions asked by the router:

- not “was this token surprising?”;
- not “is this memory similar?”;
- not “has every expert received equal traffic?”;
- but “will preserving or computing this reduce future error or improve a later
  action?”

Utility is measured over several horizons. Immediate next-token gain alone would
discard facts whose value appears much later.

### 5. Learning at four rates

1. **Token rate:** exact attention and unit proposals change every token.
2. **Event rate:** predictive state, episodic slots, and fast weights change
   after a causal token chunk when predicted utility is high. The initial
   reference uses 24-token events; learned boundaries remain a later hypothesis.
3. **Consolidation rate:** replay distills repeatedly useful episodes into slow
   weights while measuring old-domain retention.
4. **Structural rate:** units become dormant, split, or are retired only when a
   persistent residual-error cluster and a counterfactual capacity audit justify
   the change.

LCO-style energy becomes a compute/credit budget grounded in marginal task
value. Units do not survive because of an arbitrary metabolism score.

### 6. Deliberation after language competence

For difficult inputs, the system may run extra latent steps before emitting a
token or action. The halt decision is trained against expected improvement minus
compute cost. LCWM-like typed path execution and LCO-like causal interventions
can later become specialist units in this deliberative phase. They are not the
base token mixer and are not enabled before coherent language exists.

## Why v1 was worth testing

Every ingredient has ancestors. The proposed research contribution is their
coupling:

1. many small predictive units compete and cooperate;
2. exact, recurrent, episodic, and slow memories coexist at explicit timescales;
3. the same delayed counterfactual utility signal trains communication, memory
   writes, compute allocation, and eventual structural change;
4. unit state is constrained by multi-horizon future predictions;
5. behavioral memory tests, not proxy loss alone, determine survival.

This is a hypothesis of novelty, not a novelty claim. A broader literature and
prior-art review is required before publication language.

## V1 decisive evidence and retirement

The vectorized 20,971,120-parameter reference passed causal, gradient,
counterfactual, generation, checkpoint, and compiled/eager parity tests. It beat
the matched Transformer at 4.20M and 16.79M tokens. It then failed all twelve
source-absent semantic continuations and tied/lost the matched 67.11M point while
using 4.22 GiB and about three times more training time. This satisfies the
predeclared radical-redesign condition. Code and rejected checkpoints are
deleted; compact reports retain the curve.

## Retired v2 hypothesis: full exact stream plus sparse event memory

V1 made the exact path and event population compete for the same per-layer
capacity. Its smaller feed-forward path and always-active population helped
early learning but became an asymptotic tax. V2 changes the ownership boundary:

1. **Exact stream remains whole.** Start from the complete matched Transformer
   block, including its full feed-forward capacity. Event memory cannot replace
   the normal token path.
2. **Sidecar is residual and initially neutral.** One or two event-memory
   sidecars read event summaries and may add a bounded low-rank residual. Their
   output scale begins at zero, so the exact-only model is embedded in v2.
3. **Activation is genuinely sparse.** A fixed event budget selects top-utility
   events and specialists before recurrent updates. Unselected specialists do
   not execute; telemetry must measure actual skipped FLOPs, not small gates.
4. **Utility pays a compute price.** The target is future-loss reduction minus
   an explicit compute/write cost. A gate near 0.5 is a failure, not a useful
   soft mixture.
5. **Memory is tested as a specialist, not a universal mixer.** The first run
   compares exact-only, dense sidecar, random-budget sidecar, and utility-sparse
   sidecar. This separates extra parameters, extra compute, and selection value.
6. **Parameter and compute fairness are both reported.** The primary arm may use
   a small parameter overhead to preserve the exact stream, but must also face a
   parameter-matched control. Claims state which budget is being compared.

The kill criterion is fast: if utility selection does not beat random under the
same activation/write budget, or if the neutral sidecar harms exact-only loss,
retire the selector/interface before another long language run. If it survives,
the first material language point is 16M followed directly by 64M; no repeated
sub-million sweeps.

The first reference now exists. It embeds the complete 20,976,128-parameter
Transformer and adds 133,124 parameters (0.635%) for four rank-32 specialists,
a router, and residual scales. A completed 24-token event can affect only the
following event. Normal execution gathers one specialist rather than evaluating
all four; telemetry reports 25% active specialist compute. Exact neutrality,
causal scan/step equality, dense-versus-sparse accounting, and counterfactual
gradient coverage pass. On one warm RTX 3060 eager diagnostic it sustained 81.1k
forward/backward tokens/s versus 88.4k for the Transformer. Before quality
testing, this established machinery only.

That comparison is now complete at 16.79M tokens. Exact-only, dense, random, and
utility losses were 4.6140, 4.6146, 4.6128, and 4.6116. Strict free relation was
14.5%, 25.4%, 27.0%, and 14.8%. Random and utility both executed one of four
specialists; utility ran 202 counterfactual probes with mean target +0.0073.
Therefore sidecar diversity remains promising, but the initial utility credit is
retired. It observed only the chosen expert and supplied no comparative targets
for alternatives. V2.1 then evaluated all four alternatives on probe steps and
trained centered relative utility. Free relation recovered to 25.8%, but random
remained better at 27.0% with lower loss. The selector/interface therefore met
its kill criterion. V2 code is deleted; the reports retain both stages.

## Beyond a monolithic model

The failure of dense organism v1 does not imply that intelligence must remain
one static monolith. It refutes making every small unit compete with the full
language path on every token. A more plausible decomposition is an ecology built
around a shared substrate:

- the substrate supplies stable language, routing context, and common latent
  coordinates;
- small units own bounded memories, domains, tools, causal models, or temporal
  scales rather than miniature copies of the whole model;
- only units with predicted marginal value execute for a given event;
- new units can be added, consolidated, split, made dormant, or retired without
  rewriting all shared knowledge;
- communication uses a narrow learned interface and is judged by downstream
  behavior, not by biological analogy or balanced traffic.

V2 is only the first narrow test of this direction: one exact shared substrate
and sparse event specialists. If utility selection fails to beat random, that
falsifies this selector/credit interface, not the larger possibility of a
non-monolithic system. Conversely, a small sidecar win is not proof that the
substrate should immediately be decomposed. Modularizing syntax and general
representation comes only after sparse coordination demonstrates repeatable
benefit, stable checkpoint composition, and bounded interference.

## V3 result: a modular predictive society (retired)

V3 directly tested four complete small causal language models with independent
weights/state, delayed 32-dimensional messages, and a token-level coordinator.
Its 21,000,608 parameters matched the 20,976,128-parameter monolith within
0.12%. All arms used the same 16.79M-token schedule and every trainable parameter
received a final gradient.

| Arm | Heldout loss | Strict free relation |
| --- | ---: | ---: |
| Monolith | 4.6140 | 14.5% |
| Uniform average, no message | 5.0261 | 5.1% |
| Learned coordinator, no message | 5.0460 | 2.0% |
| Shuffled messages | 5.0973 | 0.4% |
| Real messages | 5.1073 | 0.0% |

**Observed:** four complete miniature models did not organize into a better
predictor. Real messages were worse than no messages and shuffled messages.
Compiled society controls stayed within a 1.003x steady-throughput band, so the
result is not explained by unequal execution among controls.

**Diagnosis:** this decomposition duplicated 9.70M vocabulary-embedding
parameters, leaving each prediction path only two layers. Processing 24-token
events through the streaming state block also detached the exact attention-cache
gradient at event boundaries; only the compressed message path crossed them.
The coarse mean message was then added uniformly to every token in the next
event. This refutes the implemented full-model society and bus, not every system
made from small units.

## V4 result: a depth-preserving modular workspace (no scale)

V4 treated cells as internal latent processors inside one
language system. It keeps a shared token embedding and readout, retains a full
differentiable context path, and alternates parallel local processing with a
narrow causal workspace. The parameter budget moves from duplicated vocabulary
tables into depth and latent computation.

At an 8,192-token vocabulary, the reference has 20,970,448 parameters versus
20,976,128 for the monolith. A token follows two 368-wide shared layers, one
256-wide layer in each of four cells, a 64-dimensional same-token causal
exchange, a second layer in each cell, and two shared integration layers. The
same shared embedding owns the tied full-vocabulary readout. Unlike v3, no
training-context boundary detaches the gradient inside a 72-token example.

The matched result was:

| Arm | Heldout loss | Strict free relation |
| --- | ---: | ---: |
| Compiled monolith | 4.6147 | 32.0% |
| Parallel cells, no exchange | 4.8549 | 10.2% |
| Shuffled workspace | 4.8518 | 11.7% |
| Real workspace | 4.8507 | 21.5% |

Real exchange nearly doubled free behavior relative to both controls, while loss
remained tied. It did not meet the predeclared 0.005 loss margin and remained
behind the monolith, so v4 is not scaled. The result supports meaningful message
content but not the unweighted mean workspace as a base architecture. The
compiled monolith's 32.0% free score also differed from the prior eager 14.5%
control at tied loss; any future positive needs a second seed or eager replicate.

## V5 result: selective content-addressed workspace (retired)

V5 retained the shared interface and full-gradient paths, but cells competed to
write into one 64-wide causal latent stream. A narrow attention layer retrieves
prior workspace states before broadcasting the result to every cell. This is the
minimal faithful connection to modern Hopfield networks: the retrieval update is
attention, while the hypothesis under test is persistent, bandwidth-limited
shared state.

At an 8,192-token vocabulary v5 had 21,012,624 parameters, 0.174% above the
monolith. Its result was:

| Arm | Heldout loss | Strict free relation |
| --- | ---: | ---: |
| Compiled monolith | 4.6142 | 17.2% |
| No exchange | 4.8526 | 24.6% |
| Shuffled associative workspace | 4.8479 | 22.7% |
| Real associative workspace | 4.8494 | 6.6% |

Real writes failed both behavior controls while loss remained tied. Mean write
entropy fell to 1.097 from roughly 1.35 in controls (maximum for four cells is
1.386), so the writer learned selectivity rather than remaining inert. The
failure is that selected temporal content was harmful. V5 and the current
modular language line are retired and deleted.

## V6 result: hyperspherical convergence retired

[nGPT](https://arxiv.org/abs/2410.01131) normalizes embeddings, hidden states,
attention/MLP vectors, and weight rows/columns onto hyperspheres. The reported
effect is substantially faster convergence, a direct fit for constrained local
training. Its [official implementation](https://github.com/NVIDIA/ngpt) also
warns that gains are smaller for shorter runs and that public low-precision
details may overstate the baseline gap.

The local test used 20.988M normalized parameters versus 20.976M frozen-control
parameters, context 72, the exact 16.79M-token schedule, full-vocabulary loss,
compiled/eager parity, and the same free behavior audit. Its 2x2 separated
architecture from recipe: both models ran the MARULHO recipe and the public nGPT
high-LR/no-warmup/no-decay recipe.

| Arm | Heldout loss | Candidate likelihood | Strict free relation | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Transformer + MARULHO recipe | 4.6144 | 96.5% | 14.8% | 129.0k |
| Transformer + native recipe | 4.6448 | 79.3% | 0% | 130.1k |
| Normalized + MARULHO recipe | 6.2844 | 67.6% | 0% | 128.4k |
| Normalized + native recipe | 4.7092 | 94.1% | 0% | 128.8k |

The native recipe was not a hidden Transformer improvement, and normalized-native
lost both its same-recipe control and the frozen baseline. Every parameter
received a final gradient. Compiled/eager warm-up loss deltas were 0.000518 and
0.000186; final normalized matrix error was at most 1.79e-7. Compiled projection
removed the eager projection slowdown, leaving all arms near 128--130k tokens/s.
One loss graph per architecture served both recipes, avoiding two redundant
compiles, and the full run completed in 731 seconds. This is a clean local
replacement failure, not a general refutation of the published long-context
nGPT result. No checkpoint was saved; v6 code is deleted and only the report is
retained at
`reports/language_scaling/hyperspherical-transformer-v6-falsification-16m-20260710.json`.

The modern Hopfield result does not independently replace attention: its
continuous one-step retrieval update is equivalent to key-value attention. V5
tested a causal latent-bank version and real retrieval was harmful, so the live
modular/Hopfield/column language code is deleted. Heterogeneous columns remain a
possible future grounded hypothesis only after a base model earns sufficient
language quality; they were not part of v6.

## V7 hypothesis: gated multiscale dynamical memory

V7 keeps the four-layer attention path and inserts four fixed-stable rotating
memory banks between layers two and three. MLP hidden widths shrink from 2048 to
1920 to hold the total at 20.977M versus the 20.976M control. Decays of
0.50/0.875/0.96875/0.9921875 provide different state horizons; a content gate
controls writes. This is an attention-recurrence hybrid, not a claim that linear
recurrence is new. LRU, Mamba, HGRN, Griffin, and the recall failures measured by
Zoology are the closest constraints on the design.

The first unrolled implementation was computationally rejected: 258.3 seconds to
compile and 63.7k tokens/s. The same fixed recurrence is now evaluated during
training as grouped causal convolutions and during generation as a one-token
recurrent update. The two forms match with nonzero prior state. First compile fell
to 67.6 seconds, compiled/eager loss delta was 0.000261, and 20 steady updates
reached 114.1k tokens/s. The matched runner then reused one graph across five
controls in a CUDA smoke, avoiding four compiles and passing parity at 0.000040.

The full decision compares Transformer, memory-off, single-scale, multiscale
always-write, fixed-random-write, and learned-write arms on the frozen 16.79M
schedule. Learned multiscale memory must beat every control by at least 0.005
loss and two strict-free points. Any first positive is replicated before scale;
no checkpoint exists before survival.

## Retired ideas

- SNN or GRU language recurrence as the active language core.
- Fixed routed columns as a base-language architecture.
- Surprise-selected prompt-prepending memory as evidence against all exact
  episodic retrieval.
- Raw token surprise as assumed write utility.
- Delta-memory v1 as the next scalable core.
- Dense distributed-organism v1 as the base token mixer.
- Sparse event-memory v2's next-token utility selector.
- Modular predictive society v3's duplicated full-language cells and delayed
  mean-message bus.
- Modular workspace v4's unweighted same-token mean as the final communication
  operator.
- Content-addressed modular workspace v5's selective causal memory stream.
- Multiple-choice or loss improvement as proof of memory.
- Biological vocabulary without a measurable computational role.

## Open creative questions

1. Can disagreement among units provide a better uncertainty signal than one
   model-wide confidence scalar?
2. Can episodic conflicts be represented as transitions instead of destructive
   overwrites?
3. Can unit utility be estimated cheaply from randomized masks and distilled
   well enough to control real inference compute?
4. Can multi-horizon predictive state produce emergent entity/event specialists
   without supervised slot labels?
5. Can consolidation replay be generated from useful internal episodes without
   amplifying the model's own errors?
6. Can a dormant/split/retire lifecycle improve capacity under a fixed parameter
   budget, or will it merely destabilize optimization?
7. Can grounded LCO/LCWM units later share the same utility currency with
   language units so that words, predictions, and actions become one learned
   system?
