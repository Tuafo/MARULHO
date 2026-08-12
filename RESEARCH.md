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
- **Next branch at that boundary:** redesign the fixed 21M core with a current
  chunk-parallel editable-state/local-attention hybrid, compared against V31
  under matched parameters, tokens, initialization, optimizer, and data.

V33 instantiated that bet without changing the installed runtime. It alternated
two exact bounded local-attention layers with two continuous editable
matrix-state layers. Training used an exact parallel diagonal-affine scan rather
than the retired token-by-token delta loop; decoding used the equivalent
recurrent update. Separate decay and write channels allowed
old addresses to fade without forcing the same gate to scale new values. The
production candidate and Transformer each contained exactly 20,976,128 useful
parameters. Causality, recurrent agreement, bounded state, complete nonzero
gradients, and BF16 compiled/eager parity passed. A local cached-graph Muon smoke
with separated optimizer warmup retained roughly 88% of Transformer training
throughput at roughly 1.41 times its peak allocation. These are execution facts,
not evidence of better language. A strict isolated V33 artifact would have been
saved and roundtrip-audited only after a durable loss/throughput win.

The user's SNN/deep-learning synthesis became a second, conditional claim:
continuous state would carry language while learned events decided whether an
optional expensive branch executed. V33 first tested the fully active branch.
Only a useful branch was allowed to earn always-on, fixed-budget, and learned
delta-event controls.

### V33 result: editable state ties quality but loses the joint contest

- **Valid matched run:** both 20,976,128-parameter arms process the same
  16,777,728 general-only tokens in 7,282 steps. The 16,780,800 elements shared
  by name and shape start bit-exact. Parameter delta is zero, every parameter
  receives a final gradient, BF16 compiled/eager parity passes, and relation
  training fraction is exactly zero.
- **Result:** Transformer/V33 heldout loss is 4.0082/4.0056. The candidate gain
  is only 0.0025 against the frozen 0.02 requirement. Candidate throughput is
  41.1k versus 54.3k tokens/s (75.5%), and peak CUDA allocation is 1.030 versus
  0.733 GB (1.41 times). Both free relation scores remain zero.
- **Decision:** `retire_v33_editable_state_no_heldout_language_win`. This is not
  a pure quality loss, but it is a dominated architecture: near-identical loss
  with worse time and memory. No checkpoint or event-controller screen is
  admitted. Live V33 model, falsifier, tests, and checkpoint surface are deleted.
  The compact report is
  `reports/language_scaling/v33-editable-state-falsification-16m-20260810.json`,
  SHA-256 `3eb9ae16cc5c039c69b0a8a68bb4c743314f02ac463c441361cae954943b9082`.

### V34 hypothesis: qualify a stronger local semantic cortex

The architecture search has repeatedly confounded two questions: whether a new
mechanism is useful, and whether a 21M model trained on modest data can produce a
semantic substrate worth augmenting. V34 separates them. It is a fresh
100,679,424-parameter dense Transformer—width 768, ten layers, twelve heads—on
V31's 8,192-token BPE, context 72, Muon 1e-3 recipe, two-source 67.11M unique
schedule, and common holdout. V31 remains evaluation-only. The larger state is
randomly initialized from the same seed policy but cannot share an initial tensor
hash with a different shape.

The null was that five times more parameters at the same data budget were too
undertrained or too inefficient to improve local language enough. Advancement
requires at least 0.20 lower heldout loss than V31's reproduced 3.6291, complete
gradients, compiled/eager parity, full source/unique-schedule audits, and strict
checkpoint fidelity. Only then run the unchanged unseen suite. A pass means the
local pipeline can build a stronger cortex; it does not mean a Transformer is the
successor architecture. A miss means local capacity scaling at 67M tokens is a
poor substrate strategy and should not be rescued by an expensive 1B-token run.

The full production shape passes a disposable CUDA/Inductor preflight. Full-graph
compile takes 52.5 seconds, compiled/eager BF16 loss differs by 0.00063, steady
two-step training measures 11.3k tokens/s, and peak CUDA allocation is 3.32 GB.
Muon shape warmup and three optimizer warm steps are excluded from measurement;
weights, optimizer, and RNG are reset before counted updates. V31 holdout loss
reproduces exactly. The smoke establishes feasibility only; its candidate loss
and report are deleted.

### V34 result: capacity produces the first clearly stronger local cortex

- **Valid run:** the fresh 100,679,424-parameter model consumes V31's exact
  67,110,912-token unique schedule in 29,128 steps. V31 holdout reproduces
  exactly; source coverage, schedule uniqueness, complete gradients, and BF16
  parity at 0.000414 all pass.
- **Likelihood result:** V31/V34 loss is 3.6291/3.3902 and perplexity is
  37.68/29.67. The +0.2389 gain passes the frozen +0.20 gate. Training sustains
  11.14k tokens/s and peaks at 3.319 GB CUDA. The strict 428,148,134-byte
  checkpoint reloads tensors, tied weights, tokenizer, config, and sample logits
  bit-exactly.
- **Unseen result:** FineWeb-Edu/Cosmopedia source loss improves to
  4.0012/3.2831. Greedy output is grammatical and multi-sentence but generic,
  repetitive, and 0/8 anchored. Repetition controls improve Cosmopedia
  distinct-bigram fraction 0.817 to 0.952 but do not change grounding.
- **Decision:** `retain_v34_capacity_checkpoint_continue_unique_data_not_base_quality`.
  This is the strongest base progress so far, not Base-Language Qualification.
  Training report SHA-256 is
  `8d623cee476b35f3f8e19168838417f748ab2610b3df69625d6eaa5f5021b5e6`;
  checkpoint SHA-256 is
  `69ce5856b8b34d8579d034375c4e1206501c6a4e44b81ff4bee951437636c79c`.

### V35 hypothesis: continue the survivor on non-overlapping data

V35 starts from the strict V34 checkpoint rather than discarding a successful
trajectory. It admits exactly three new training files excluded from V34:
FineWeb-Edu train shard 0, Cosmopedia 150k shard 1, and Cosmopedia 75k shard 3.
The phase adds 134,219,520 tokens for 201,330,432 cumulative updates. It rebuilds
fresh Muon state with a 3e-4 peak and warmup; the parent checkpoint state must
load bit-exactly before any update. The null is that V34's improvement was mostly
capacity-at-fixed-data and additional data yields less than 0.15 loss gain.
Only a pass writes a new checkpoint and repeats the unchanged unseen suite.

### V35 result: promising loss, invalid schedule coverage

V35 reaches heldout loss 3.1654 from V34's reproduced 3.3902, a diagnostic
0.2248 gain above the frozen 0.15 requirement. It processes exactly 134,219,520
unique tokens at 11.20k tokens/s, peaks at 3.330 GB CUDA, matches the V34 initial
state, passes BF16 parity and complete gradients, and never repeats a scheduled
source index. This is not promotable evidence. The prepared pools contain
19,419 batches per source, but the token budget schedules counts
19,419/19,418/19,418. The existing requirement that every prepared batch be
consumed therefore fails. Decision: `invalid_v35_capacity_continuation_evidence`.
No checkpoint or unseen-generation review exists. Report SHA-256 is
`de18b99e21d89fd9741d6c27a4d3c89612b72c00075b82dae6419c9a7b53657f`.

V35R corrects the manifest rather than weakening the gate after observing the
result. It restarts from the exact V34 checkpoint and schedules all 58,257
prepared batches, adding only the two previously omitted batches. The corrected
budget is 134,224,128 new and 201,335,040 cumulative tokens. V34 report and
checkpoint hashes, the three source hashes, 19,419/19,419/19,419 prepared counts,
model shape, 3e-4 learning rate, and +0.15 quality requirement are locked. This
is a fresh confirmatory run; V35's final loss is diagnostic and cannot initialize
or select it.

### V35R result: the continuous base earns the next research phase

- **Valid continuation:** all 19,419 batches from each of the three pinned
  sources are consumed exactly once: 58,257 optimizer updates, 134,224,128 new
  tokens, and 201,335,040 cumulative tokens. Initial-state identity, full
  coverage, schedule uniqueness, complete gradients, and BF16 compiled/eager
  parity all pass.
- **Quality curve:** heldout loss/perplexity improves from V34's reproduced
  3.3902/29.67 to 3.1649/23.69. The 0.2253 gain passes the unchanged 0.15 gate.
  Training sustains 10.65k tokens/s and peaks at 3.330 GB CUDA allocation.
- **Checkpoint:** the 428,148,198-byte checkpoint strict-reloads model tensors,
  tied weights, tokenizer, config, and sample logits. Training report/checkpoint
  SHA-256 values are
  `132555f649483c19999c16a66688691872b37ee316e5561b8b741e87eed34bb9`
  and `48bfe82a70d9c537f10dc6d898c3cf18906716bd90acfefb7089ccd30477d9df`.
- **Unseen language:** FineWeb/Cosmopedia continuation loss improves from
  4.0012/3.2831 to 3.8020/2.9282. Controlled Cosmopedia generation reaches
  0.968 distinct-bigram fraction and produces coherent multi-sentence English.
  It still copies too little of the hidden source continuation and remains 0/8
  anchored.
- **Decision:** `save_v35r_capacity_continuation_201m_for_unseen_generation`.
  V35R is the first checkpoint strong enough to reopen continual-learning and
  conditional-compute tests. It is not evidence that grounding, memory, sparse
  compute, or runtime installation is solved.

Before another long run, profile the exact V35R CUDA training step. The measured
3.330 GB allocation leaves substantial physical capacity on the 12 GB RTX 3060,
but memory headroom alone does not prove under-utilization. Batch shape, forward/
backward, gradient clipping, Muon orthogonalization, parameter updates, and host
staging must be timed separately. A faster path is admitted only after numerical
parity and a matched short quality trajectory; tokens per second cannot replace
loss quality.

### Frontier open-model audit: mechanisms, not borrowed cognition

Kimi K3, the current open Qwen3.5/3.6 line, and DeepSeek V4 are useful here as
large-scale evidence, not as MARULHO components. MARULHO will not load their
weights, distill their outputs, call their APIs, or delegate cognition to them.
Any surviving idea must be implemented locally, initialized independently, and
beat the current MARULHO checkpoint under matched data, token, quality, memory,
and wall-clock controls.

The requested `Qwen 3.8` name is not supported by a current official open report
or repository. The official public repository presently identifies Qwen3.6 and
Qwen3.5, so this audit uses those inspectable releases rather than a possible API
preview or informal name.

#### What the independent frontier designs agree on

- **Do not make one mixer solve every sequence problem.**
  [Kimi K3](https://github.com/MoonshotAI/Kimi-K3) repeats three Kimi Delta
  Attention layers followed by one global Gated MLA layer. The official
  [Qwen3.6 configuration](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)
  likewise repeats three Gated DeltaNet layers followed by one full-attention
  layer. [DeepSeek V4](https://arxiv.org/abs/2606.19348) instead mixes two forms
  of learned sequence-compressed attention, one sparse and one dense. The common
  result is not "delta recurrence won" or "attention won"; it is that cheap
  lossy mixing and expensive exact/global mixing have distinct jobs.
- **Sparse width is paired with a dense/common path.** Kimi K3's LatentMoE sends
  routed experts a half-width latent while two shared experts retain a full-width
  path, activates 16 of 896 routed experts, normalizes the routed aggregate, and
  balances dispatch without putting the balancing bias into mixture weights.
  Qwen3.6-35B-A3B similarly exposes 256 routed experts, eight selected experts,
  and one shared expert in its official configuration. This is materially
  different from MARULHO's retired micro-model societies: the experts specialize
  inside one shared representation rather than duplicating the language model.
- **Depth is becoming addressable state.** Kimi K3 Attention Residuals retrieve
  from learned block-level representations across depth. DeepSeek V4 mHC widens
  the residual stream into a few channels and constrains their learned mixing to
  a non-expansive doubly stochastic matrix. These are credible modern relatives
  of the user's columns/small-units intuition, but the units are latent routes
  inside one jointly trained cortex, not autonomous miniature LMs.
- **Efficiency is trained and kernel-shaped, not inferred from operation
  counts.** Kimi lower-bounds KDA decay so all 16-token tiles can use dense Tensor
  Core matrix multiplication. Qwen provides fused Gated DeltaNet kernels.
  DeepSeek compresses sequence entries before sparse selection and groups the
  output projection. All three retain hardware-regular batches and specialized
  dense kernels. This agrees with MARULHO's finding that irregular nominal
  sparsity can be slower than dense execution on the RTX 3060.
- **Batch size, learning rate, schedule, and model shape are coupled.** Kimi K3
  independently retunes batch size and learning rate for each learning-rate
  schedule and reports cosine beating WSD only after that retuning. DeepSeek V4
  explicitly grows its token batch through training. Therefore MARULHO cannot
  adopt the measured 8x larger batch as a speed switch while holding the old
  optimizer-step semantics fixed; it needs a short token-matched learning-rate
  screen.

#### Immediate 3060 implications

The highest-value near-term transfer is optimizer geometry, because profiling
the exact V35R shape attributes roughly 57% of step time to the optimizer and
shows that Newton--Schulz orthogonalization dominates that cost. Kimi K3 applies
Muon independently to each attention-head projection rather than one coupled
Q/K/V matrix; its report says this balances head update scales and slightly
reduces optimizer overhead. This maps exactly onto MARULHO's combined
`qkv.weight`, but changes learning geometry and therefore needs a loss-parity
test, not just a timing benchmark. DeepSeek V4 also batches same-shaped Muon
matrices and uses BF16 orthogonalization, both already present in MARULHO, which
independently supports the current implementation choices.

The next speed gate therefore compares the unchanged V35R optimizer with a
MARULHO-owned per-head Q/K/V Muon path and evaluates larger physical batches with
token-progress-aligned schedules. Every arm starts from the same V35R checkpoint,
consumes the same ordered tokens, and is judged jointly on heldout loss,
throughput, finite updates, and peak VRAM. A throughput win with worse loss is
rejected.

The first optimizer-only implementation check is positive but deliberately not
promoted: on the exact 100.68M-parameter V35R shape, 20 measured CUDA updates take
117.02 ms each with whole-QKV Muon and 104.91 ms with per-head Q/K/V Muon, a
10.3% optimizer reduction. Given the full-step phase profile, the isolated gain
projects to only about 6% end-to-end. It therefore remains one arm in the larger
batch/learning-rate screen rather than replacing the baseline by timing alone.

#### V36 result: use the GPU harder without paying a quality penalty

The six-arm CUDA screen completes in 1,361.27 seconds and passes all source,
schedule, gradient, and compiled/eager parity checks. The batch-32 whole-QKV
control reaches 3.24553 heldout loss at 11,080 tokens/s. Physical batch 256 at
the same 3e-4 learning rate reaches 3.14227 at 25,065 tokens/s, a 2.262x speedup
and a 0.10326 loss improvement. It uses 7.72 GiB peak CUDA allocation, leaving
enough headroom on the 12 GB card for a durable run. Larger-batch learning-rate
scaling is not useful here: 8.5e-4 reaches 3.22290 and 1.2e-3 reaches 3.27892,
with the latter failing the frozen +0.01 loss bound.

Per-head Q/K/V Muon passes its independent batch-32 gate: 11,886 versus 11,080
tokens/s, a 7.27% gain, for +0.00238 heldout loss. At batch 256, however, the
same-rate 8.5e-4 comparison gains only 1.76% and loses 0.00381 loss. The frozen
selector chooses the fastest qualifying arm, batch-256/per-head/8.5e-4, but that
arm is scientifically dominated by batch-256/whole-QKV/3e-4: the latter gives
0.08443 better loss for only 1.68% less throughput. The immutable V36 artifact
keeps the preregistered selector output; MARULHO's standing quality-first rule
chooses batch 256, whole-QKV Muon, and 3e-4 for the next durable stage. No V36
checkpoint is saved. The raw report SHA-256 is
`e57ec348e588c073712c6c1a03613a6fc7b3400c205a7fae8e28fcc42f346719`.

### V37: full-width depth assembly is too expensive

The next architecture test asks whether a monolithic residual chain is losing
useful intermediate representations. MARULHO Depth Assembly gives block `i` a
triangular set of bounded scalar corrections toward every representation before
its current input. Forty-five learned scalars augment the ten-layer V35R model;
there are no imported weights, external inference calls, additional token
mixers, or hidden labels. Zero initialization is exactly the existing network,
including cached incremental decoding, so the comparison starts from identical
function and tensors rather than a fresh candidate initialization.

The frozen control and candidate were assigned 16,773,120 identical unique
ordered tokens at physical batch 256, whole-QKV Muon 3e-4, context 72, and BF16
compiled execution. V37 does not complete within the fixed 3,600-second command
budget. Live samples reach 11,744/12,288 MiB device allocation, and the process
must be explicitly terminated after its parent timeout. No terminal report is
emitted, so there is no admissible heldout or route-value result and no claim
that the mechanism helps or hurts language quality.

This is nevertheless a terminal systems result: many scalar-weighted full-width
history tensors are a poor GPU primitive and fail the consumer-hardware goal.
The candidate, runner, and tests are deleted. Any future depth cooperation must
be a new hypothesis expressed through a few fused dense or low-rank operations,
not a repaired or longer-running V37. Timeout artifact SHA-256 is
`cf7465cdeec25be68bbb75af07096e2ae420d233260aaa1a985496c2cee86442`.

### V38: continual learning on the qualified base

V38 returns to MARULHO's central claim now that V35R has coherent general
language and V36 supplies a quality-safe fast recipe. Three equal-compute arms
continue the exact V35R tensors for 16,773,120 tokens at physical batch 256 and
Muon 3e-4. The focused arm is 100% new procedural relations. Two replay arms use
50/50 and 20/80 relation/general schedules; replay comes only from FineWeb shard
2 and Cosmopedia shard 4, which V35R did not train on. The new domain uses the
larger 800k-document relation corpus and a separate 256-case compositional
holdout.

Success is autonomous behavior, not candidate ranking alone. A replay arm must
reach at least 50% strict free-answer accuracy, at least 80% label-safe ranking,
and no more than +0.10 regression on the old general holdout. The relation-only
arm diagnoses whether the base can acquire the domain under the same compute;
it cannot be promoted as continual learning. A joint replay pass saves the one
best arm and requires exact model/tokenizer reload. If focused learning works
but replay does not, the next branch is parameter isolation; if focused learning
also fails, the relation objective/interface is redesigned.

V38 completes in 2,633 seconds. The 50/50 replay arm is the clear Pareto point:
100% candidate accuracy, 46.875% strict free generation, general loss 3.11237,
25.02k tokens/s, and 7.73 GiB peak allocation. The initial state is 39.06% / 0%
candidate/free at loss 3.16492. Relation-only reaches 100% / 37.50% but destroys
general language at loss 15.83729. The 20/80 arm reaches 100% / 37.11% and loss
3.07688. No arm meets the frozen 50% free gate, so no checkpoint exists.

The failure distribution localizes the objective/interface gap. For 50/50,
free event-order/property/container/ownership accuracy is
85.94%/76.56%/20.31%/4.69%. Many misses express the right relation with malformed
or fused lexical forms (`coins`, `orangecoin`) that fail exact answer matching.
More replay is not indicated: replay already improves general loss and 50/50
outperforms focused relation training on free answers. V39 must emphasize the
answer-bearing tokens or structured lexical realization under the same replay
and compute budget; lowering the evaluator or promoting 100% ranking is not an
acceptable repair. Raw report SHA-256 is
`e356bf9a44ccb7fd1986be256c41128c2bb79a086c903d1be7bd110a841cc1d2`.

### V39: emphasize answer-bearing tokens

V39 changes only how the fixed 50/50 V38 token stream contributes loss. A
compiled causal mask finds the full tokenizer-owned ` Answer:` marker and
weights subsequent target tokens through the document EOS by 2x or 4x. Loss is
renormalized by total token weight, general rows without the marker remain
ordinary next-token training, and the exact V38 report is hash-pinned as the
unweighted control. Both arms restart from V35R and receive the same 16,773,120
tokens, Muon 3e-4 schedule, and strict evaluation.

The mechanism is intentionally between ordinary loss and the previously failed
answer-only objective: V38 shows that full prompt modeling supplies useful
relation representations, while its 100% ranking/46.88% free split says answer
tokens need more gradient share. Promotion still requires at least 50% strict
free accuracy, 80% ranking, no more than +0.10 general-loss regression, complete
gradients, and exact checkpoint/tokenizer reload. Decoding controls, model
capacity, replay fraction, and evaluator normalization are frozen.

V39 passes narrowly and saves the first continual-qualified checkpoint from the
100M base. The 2x arm reaches 46.09% free / 98.44% ranked at general loss
3.11275. The selected 4x arm reaches exactly 50.00% free / 98.44% ranked at loss
3.11336, processing 24.86k tokens/s with 7.99 GiB peak allocation. Marked answer
targets occupy 7.48% of the representative mixed batch. The checkpoint reloads
with exact model and tokenizer hashes after 218,108,160 cumulative tokens.

This is a real but bounded continual result. The selected arm gets 58/64
property and 51/64 event-order cases freely correct, versus only 15/64 container
and 4/64 ownership. Reloaded open prompts remain grammatical and multi-sentence,
although repetition and semantic oddities remain at this scale. V39 proves that
moderate causal credit assignment crosses the joint learning/retention boundary;
it does not prove general relational reasoning or lifelong learning. The
maintained answer objective moves to the training package, while the one-shot
runner is deleted. Report/checkpoint SHA-256 are
`3b64d702ed2db458587c78316d34fe826138bef8d4d72b8093dc861d11289127` and
`6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d`.

V40 is preregistered as the same-checkpoint sustained-runtime and active-compute
qualification. Its immutable parent is the V39 checkpoint with SHA-256
`6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d`.
The primary run generates 256 independent streams of 2,048 consecutive tokens,
for exactly 524,288 aggregate new tokens. This is not described as one
524,288-token context or one continuous stream. Success requires CUDA execution,
exact pre/post tensor-state identity, bounded 72-token KV and decode-control
history, finite full-vocabulary logits, in-vocabulary outputs, no external LLM,
and completion inside 600 seconds. Output storage is bounded to hashes and
selected previews. A one-step module-hook audit records unique parameter
coverage, executed layers, and dense attention/MLP ownership; if all maintained
components execute, V40 records zero structural sparsity instead of inventing a
sparse claim. A diagnostic on the exact checkpoint measured about 8.17k
aggregate tokens/s for 256 BF16 streams and 2.94 GiB peak allocation, so this
shape should finish in roughly one minute before decode-control overhead.

V40 passes its frozen gate. The exact V39 checkpoint emits all 524,288 tokens
as 256 independently prompted 2,048-token streams in 74.8408 seconds, or
7,005.38 aggregate tokens/s, with 3,165,870,592 bytes peak CUDA allocation.
Every pre/post model tensor hash matches; final KV length is 72; raw non-finite
logits, out-of-vocabulary IDs, and decode fallbacks are all zero. Full-stream
hashes are distinct for 248/256 streams. Hooks observe all 100,679,424 unique
parameters and every attention/MLP block, providing a measured negative result
for sparsity: V39 executes 100% densely. Preview inspection also preserves the
boundary—some long streams become repetitive, symbolic, or semantically
unstable, so this is runtime qualification rather than long-context quality.
Decision: `qualify_v40_same_checkpoint_sustained_runtime`. Report SHA-256 is
`4757c0a0f0972fabe1de3e0b742f91a049f166994a9421d141c117a7ddcf2331`.

V41 tests direct hidden-state episodic memory rather than another cortex or
cross-attention rewrite. [kNN-LM](https://arxiv.org/abs/1911.00172) establishes
that final-context representations can index next-token values and adapt a
language model without weight training. Later analysis finds that retrieval
quality should control interpolation
([Drozdov et al.](https://arxiv.org/abs/2210.15859)), while open-ended generation
often degrades under unconditional kNN interpolation
([Wang et al.](https://arxiv.org/abs/2305.14625)). Those are design constraints,
not borrowed weights or a capability claim.

The V41 parent is the exact V39 checkpoint. Frozen training-only relation
documents supply normalized final hidden keys and their next answer-token IDs.
A disjoint calibration set selects k, cosine threshold, and bounded logit bias;
the frozen 256 V39 cases remain untouched until the terminal audit. The memory
is active only after the complete `Answer:` marker, and gate-off logits must be
bit exact to V39. Controls are base V39 and identical keys with deterministically
shuffled values. True memory must reach 65% free / 98% ranked, raise ownership
and container to 40% each, beat shuffled values by ten free points, leave model
tensors unchanged, and exact-reload keys, values, normalization, tokenizer hash,
and provenance. The report separates active top-k values from the dense full-key
search and measures latency/VRAM; no nominal sparsity claim can pass.

V41 fails terminally. The 65,536-entry store is built from 8,192 training-only
documents in 6.72 seconds; disjoint calibration selects top-1, threshold 0.85,
weight 0.8. Frozen base/true/shuffled free accuracy is
50.00%/51.56%/1.17%, while ranked accuracy is 98.44%/98.44%/91.80%. True memory
raises property from 90.62% to 100% but ownership remains 6.25% and container
falls from 23.44% to 20.31%. The same +1.56-point result appeared with only
4,096 entries, so datastore scale does not repair the representation. General
loss is exactly unchanged at 3.53110 over 18,432 sampled tokens, gate-off logits
are bit-identical, and the frozen cortex hash is unchanged. Shuffled values
establish that logit fusion is causal, but causal intervention is not useful
binding. Search compares 740,950,016 keys for the terminal true arm and remains
dense. Decision: `retire_v41_hidden_state_memory_no_joint_free_binding_win`.
No memory artifact is saved; model, runner, and tests are deleted. Report
SHA-256 is `96a34833e573638b4bcbe06c2fba47b99b709c671a16c47b93089eb9302c0e2a`.

V42 follows the evidence rather than reopening memory. V39 ranks the correct
candidate at 98.44% but freely emits only 50%; V41 changes next-token
probabilities causally yet cannot identify ownership/container values. The next
test makes the training objective contrast role-confusable answer tokens.
[Unlikelihood training](https://arxiv.org/abs/1908.04319) shows that explicitly
lowering unwanted token/sequence probabilities can improve generation without
sacrificing perplexity, and
[sequence likelihood calibration](https://arxiv.org/abs/2210.00045) shows that
conditional generation can improve when likelihood ordering is trained rather
than left to decoding heuristics. V42 borrows neither model nor data; these are
objective precedents.

All arms restore the exact V39 state and use its frozen 50/50 schedule, Muon
recipe, 4x normalized answer emphasis, tokenizer, sources, and evaluator. The
candidate adds unlikelihood only when the correct next token belongs to an
explicit entity/container/color/event-polarity group; all other answer and
general tokens are unchanged. A 2,359,296-token pilot compares weights 0, 0.25,
and 1.0. Successive halving requires +5 free points, +5 ownership or container
points, and no more than +0.03 general-loss regression before the selected
candidate and control receive the full 16,773,120-token confirmation. Terminal
promotion requires 65% free, 98% ranked, 40% ownership, 40% container, bounded
general loss, full gradients, and exact reload. A pilot miss deletes the entire
objective rather than tuning more weights.

V42 closes without a quality result. Its checkpoint-owned BPE trie, fused
cross-entropy normalizer, vectorized negative lookup, and 64-case evaluation
chunks all passed focused checks; one batch-256 forward/backward took 2.75
seconds, peaked at 11.72 GB, and reached all 62 parameter tensors. The exact
32x8 eager run nevertheless remained saturated for 16,507.6 seconds and wrote
no arm artifact. Continuing would reward sunk cost and block fast research, so
the process was stopped and all live V42 machinery deleted. This does not show
that role contrast is bad; it shows this falsifier cannot answer the question
credibly at acceptable cadence. Decision:
`stop_v42_execution_infeasible_no_quality_conclusion`. Report SHA-256 is
`9ecd6e1e4ba8e603624eb15797f9fe4f5a534388e2221401f9537c98286f7808`.
The shared experiment runner now times complete warmup optimizer steps, rejects
projected over-budget arms before counted training, and atomically persists each
exact-contract arm with optional model state. Projection and artifact behavior
pass focused unit checks. The exact V39 100.68M checkpoint also passes integrated
BF16/Muon GPU rejection and exact restoration. Effective batch 224 is the best
tested safe point at 19.22k training tokens/s and 10.49 GB peak allocation;
batch 256 falls to 3.83k tokens/s under memory pressure and is rejected. No
counted training or quality comparison occurred, so every later mechanism must
still use the same frozen batch contract across arms. Runtime-preflight report
SHA-256 is
`284b35710e6b59572459a35ff9d79dd9f3a8b02921fbc7a6e3f4bb3d43884c15`.

### V43 preregistration: grounded prompt-copy readout

V39 exposes a narrower failure than missing candidate knowledge: it ranks the
correct complete answer in 98.44% of heldout cases but generates it freely in
only 50.00%. V41 could perturb next-token probabilities but did not recover the
weak ownership/container bindings, and V42 could not execute its many negative
branches at useful cadence. V43 therefore tests whether the ordinary vocabulary
readout is failing to assemble evidence that already exists in the current
causal window.

The candidate adds one low-rank learned pointer from each output hidden state to
earlier hidden states. Pointer mass is scattered only onto token IDs that
actually occur earlier in the same causal window, then enters the existing
vocabulary logits through one zero-initialized bounded residual scale. At reset,
the candidate is exactly V39. It does not read answer labels, candidates,
metrics, a retrieved archive, or future tokens. It is a readout experiment, not
a long-term-memory claim. Pointer-generators have improved factual reproduction
in sequence generation, while later work warns that copy mechanisms may fail to
use their intended path; copying ability can also emerge late in ordinary
Transformers ([See et al.](https://arxiv.org/abs/1704.04368),
[Bafna et al.](https://arxiv.org/abs/2403.10963),
[Lv et al.](https://arxiv.org/abs/2409.09281)). Those results motivate the
mechanism and the skeptical controls; they do not validate MARULHO's candidate.

Before training, a metrics-only audit must report how many answer BPE tokens and
complete answer spans are actually copyable from each prompt. If fewer than 85%
of answer tokens occur in the prompt, stop: this interface cannot explain the
target gap. Otherwise freeze one 4,194,304-token 50/50 general/relation schedule
and compare three exact-reset arms at one candidate-safe effective batch:

1. unchanged V39 answer-weight-4 continuation;
2. the learned content pointer;
3. an equal-parameter pointer whose source token identities are deterministically
   shuffled only inside the copy readout while the Transformer sees the true
   prompt.

The integrated three-step BF16/Muon preflight selects the fastest common batch
whose peak allocation stays at or below 10.5 GB and whose projected per-arm
wall time stays at or below 1,200 seconds. Every completed arm is persisted
atomically under one frozen contract hash. Any projection failure stops before
counted training; no partial arm becomes evidence.

The pointer advances to a 16,773,120-token confirmation only if it beats both
controls by at least 10 free-generation points, reaches at least 60% strict free
accuracy, keeps general heldout loss within +0.05 of the unchanged arm, keeps
ranked accuracy at or above 98%, and raises both ownership and container by at
least 10 points. On answer-changing source-swap pairs it must follow the changed
source at least 65% of the time and beat the unchanged arm by 10 points. Shuffling
pointer token identities must remove at least half of the candidate's free-
generation gain. Active scale, gradients, or attention entropy alone cannot
pass. Failure deletes the V43 implementation and retains only the compact report
and decision.

V43 stops at that first gate without creating an implementation. Under the exact
V39 BPE tokenizer, only 66.53% of correct-answer token IDs occur anywhere in the
prompt, below the frozen 85% requirement, and no complete answer token sequence
occurs contiguously. Container/ownership/property/event-order coverage is
68.42%/68.68%/71.49%/57.84%. The generator must synthesize substantial relational
language rather than merely point to source tokens, so this copy-only residual
cannot explain the measured gap as proposed. Decision:
`stop_v43_prompt_copy_insufficient_answer_token_coverage`. No model, runner,
tests, or checkpoint exist. Compact report SHA-256 is
`6b9580d3097d34fbd28b3edc49965ec0851026743ab98fba77fabc95fe9afc70`.

### V44: the 50% binding wall was partly a decoder-policy bug

A read-only audit of V39's retained rows shows correct candidate sequences with
near-zero teacher-forced loss beside free generations containing small plural,
template, or entity deviations. The maintained decoder applied repetition 1.1
and no-repeat-3 to prompt plus generated tokens, while candidate scoring applied
neither. This made any three-token phrase already present in the source illegal
to repeat in a factual answer.

The same frozen checkpoint and 256 cases isolate the effect. Old default,
no-controls, old repetition-only, and old no-repeat-only strict free accuracy is
50.00%, 88.67%, 87.11%, and 51.56%. No-repeat prompt history is therefore the
dominant cause. V44 changes no weights: decode policy v4 bounds repetition and
no-repeat history to generated continuation tokens only. The maintained batched
evaluator reaches 88.67% strict free accuracy at the same 98.44% ranking, with
container/ownership/property/event-order at 60.94%/100%/100%/93.75%. Model hashes
remain exact.

Decision: `promote_v44_generated_only_decode_controls_requalify_v39`. This is a
real capability correction but not a learning gain or general-grounding result.
It also retrospectively removes the main motivation for V42 and V43; their
execution and copyability conclusions remain valid, but the presumed 48-point
ranking/free deficit does not. The changed runtime policy invalidates V40 as
current behavioral evidence, so the exact 524,288-token sustained test must run
again before runtime is requalified. Report SHA-256 is
`e413abd919fb25ea546046b76652c7e011666fa0b7c8ecda8e7a454bdb0b2315`.

### V45: generated-only sustained runtime requalifies

The generalized v4 sustained contract reruns the unchanged V39 artifact with
V44 decoding. It completes 256 streams times 2,048 tokens: 524,288 tokens in
73.1564 seconds at 7,166.67 tokens/s, with 3,165,493,760 bytes peak allocation.
Checkpoint and pre/post tensor hashes are exact, all logits are finite, every
token is in vocabulary, KV and decode-control state stay bounded, and hooks
observe all 100,679,424 parameters. There are 247 distinct full-continuation
hashes across 256 streams. The current runtime is therefore qualified and still
measured as 100% dense. Previews remain locally grammatical but drift, repeat
themes, and invent facts, so V45 does not promote long-generation quality.
Decision: `qualify_same_checkpoint_sustained_runtime`. Report SHA-256 is
`51eefbbd66c8869217c4ca5a53fa1e5006f44887de028c654a1a3995d0572175`.

### V46: correct the unseen diagnostic, then stop calling it grounding

The old unseen evaluator had two independent defects: it encoded the entire
remaining 37–51 MB source text to retain at most 64 continuation tokens, and it
inserted a new BOS token at the start of that continuation. The replacement
scans progressively larger bounded character prefixes until the requested BPE
prefix is stable and explicitly encodes continuation tokens without BOS. Focused
tests cover both properties; the three-report V39 suite falls from a timeout to
roughly one minute total.

The corrected matched result remains negative. FineWeb-Edu V35R/V39 loss is
3.60076/3.64029; Cosmopedia is 2.71844/2.64983. V39 passes none of four FineWeb,
four Cosmopedia greedy, or four corrected-control cases. Its relation update is
therefore mixed but bounded on these sources, not catastrophic forgetting, and
does not create exact unseen continuation.

This suite is not actually a grounding test: the model receives only a three-word
heldout prefix while the source document remains metrics-only. Exact agreement
with the hidden author's continuation measures predictive specificity, not use
of visible evidence. The result remains useful under that name, but it cannot
decide the architecture needed for grounding. Decision:
`redesign_unseen_grounding_benchmark_keep_exact_continuation_diagnostic`. The
next test uses a source-visible, heldout extractive QA set with question-only and
corrupted-source controls, followed by matched continual training only if the
frozen V39 audit shows headroom. Composite report SHA-256 is
`9df4477f806f99f46892ca828e3e1b058588f2a8e6501e5d94ae15d6f43914e2`.

V47 preregisters that replacement against `rajpurkar/squad` validation. The
manifest selects 64 deterministic rows, caps each source/question prompt at 64
V39 BPE tokens, restricts answers to eight tokens, limits repeated article
titles, and excludes questions that contain an accepted answer. Each intact
case is paired with question-only and answer-absent mismatched-source controls;
all prompts fit the same context bound. The frozen V39 checkpoint sees no labels
or candidate list. A base grounding pass requires at least 25% strict answer
accuracy and +10 points over the stronger control. A weaker +5-point causal
source gain may still admit matched continual training, but no activity or loss
proxy can substitute for generated answers. The official training split remains
untouched until this validation audit is terminal.

V47 is valid and terminal. The unchanged V39 checkpoint produces 3/64 strict
answers with intact source and 0/64 for both question-only and answer-absent
mismatched source. All prompts fit, answers are visible only in the intended
condition, and pre/post model hashes are exact. The 4.69-point causal gain is
below the preregistered 5-point weak-use branch by one case and far below the
25% absolute/+10-point promotion gate. Decision:
`v39_no_visible_source_use_train_grounding_with_replay`. This establishes a
credible capability target rather than another likelihood proxy. The disjoint
official training split may now train two matched arms—ordinary causal loss and
answer-weighted loss—while replay preserves V39 relations and general text.
Manifest/report SHA-256 are
`9b3392f137a2ca467bc329815810581a98169da170f74f50e8ccb41cb06e12d6`
and `5a4d36afec1f20f8bf777e7f5eaef35e171e07c2e238bbd7001e028113477b71`.

### V48: learning source use is possible, shared plasticity is the blocker

V48 freezes the V47 validation contract and materializes 512 disjoint official
SQuAD training cases through one hash-pinned Parquet shard. Both exact-reset
arms process 4,193,280 tokens on the same schedule: 50% SQuAD, one-sixth prior
relation replay, and one-third FineWeb-Edu/Cosmopedia replay. The only scientific
difference is ordinary causal loss versus V39's normalized 4x answer emphasis.
The small SQuAD corpus owns 44 full source microbatches and is deliberately
repeated during this short learnability screen; each of the three replay sources
contributes 1,214 prepared batches.

Both arms learn a causal source signal. Ordinary loss moves intact/question-only/
mismatched accuracy from V47's 3/0/0 to 9/1/0; answer weighting reaches 14/1/0.
That is a +7.81-point matched gain for answer weighting and a +20.31-point gain
over its stronger control. The objective is useful, but not sufficient: answer
weighting misses the 16/64 grounding floor by two cases, and both arms overwrite
old behavior. On a deterministic kind-stratified panel, strict relation recall
falls from 57/64 to 26/64 and 28/64. General heldout loss rises from 3.13964 to
3.24415 and 3.24195, more than twice the allowed +0.05 regression.

The execution result matters too. The first physical-batch path assembled 28
host microbatches as separate CUDA temporaries and fell into memory paging; its
three-step preflight projected 232,231 seconds and correctly executed zero
counted updates. True gradient accumulation preserves effective batch 224 while
using physical batch 8. It completes both 260-step arms at 5,429/5,365 tokens/s,
with 3.19/2.20 GiB measured peak allocation, and deletes both 428 MB temporary
states after the terminal decision.

Decision: `retire_v48_objective_only_grounding_repair`. This does not refute
source grounding or answer emphasis. It refutes the claim that one shared weight
state plus random replay is enough to add a second broad capability locally.
V49 will test isolated plasticity: freeze V39, add a small MARULHO-owned residual
adapter behind a source-visible condition, require bit-exact old-path logits when
inactive, and retain the same V47 grounding/control gate when active. A success
would justify learned routing later; a failure retires adapter-style modularity
before a larger architectural investment. Report SHA-256 is
`834e1bce825675f0c18cac77c39e30b8403fcb5368e3937b9c91a46b5b9fb968`.

### V49 preregistration: frozen cortex, conditional residual plasticity

V49 asks one question: can MARULHO add source-visible QA without writing into
the weights that already own language and relation behavior? The candidate keeps
every V39 tensor frozen and adds one small causal Transformer sidecar after the
base output normalization. When inactive, the sidecar is skipped as a Python
branch rather than multiplied by zero; base logits, streaming state, relation
answers, and general loss must therefore be bit-exact. When active, the sidecar
gets the frozen contextual states, owns one bounded KV cache, and may learn a
residual representation before the unchanged tied vocabulary head.

This screen gives the sidecar exactly 2,096,640 SQuAD tokens, equal to V48's new-
domain exposure, and retains V48's 4x answer objective, training manifest, V47
validation manifest, decode policy, and answer-absent controls. Replay is removed
because inactive behavior is protected structurally, not statistically. The
activation flag is an explicit benchmark condition derived from the source-QA
interface; V49 makes no learned-routing claim. The module must remain below 5%
of base parameters and every trainable tensor must receive a final gradient.

Promotion requires at least 18/64 intact answers (28.125%), at least +10 points
over the stronger control, and at least +5 points over V48's 14/64 weighted arm.
It also requires exact frozen-base hashes, bit-exact inactive logits, unchanged
inactive relation/general metrics, bounded adapter state, and exact adapter
checkpoint reload. A pass advances the sidecar to a learned-router falsifier.
A miss retires this final-layer sidecar and deletes its live implementation; it
does not license another replay-ratio sweep.

V49 is terminal and negative. The sidecar contains 4,130,304 trainable
parameters (4.10% of V39), every tensor receives a final gradient, and its
answer-weighted training loss falls from about 3.53 to 3.20 across 2,096,640
tokens. The frozen-base design is unusually efficient on the RTX 3060: 130
physical-batch-224 updates complete in 37.81 seconds at 55,447 tokens/s, peak
allocation is 2,358,689,280 bytes, and Muon state is only 16,527,360 bytes.

Isolation passes all meaningful checks. Inactive parent state and sample logits
are bit-exact before/after training; general loss is exactly 3.149025917 both
times; relation ranking/free generation remains 63/64 and 57/64; adapter cache
length stays within 72. Yet the active sidecar answers only 1/64 intact SQuAD
cases and 0/64 for both controls. It is worse than unchanged V39's 3/64 and far
below V48's 14/64 weighted arm, so more steps would be scaling a mechanism that
has already lost the matched representational comparison.

Decision: `retire_v49_final_sidecar_insufficient_grounding`. The result separates
two problems cleanly: modular parameters solve forgetting by construction, but
plasticity only after the final frozen representation is too late to learn
source-grounded behavior. A future modular candidate must interact within the
representation hierarchy or own its own source encoder; it cannot be another
final-state adapter or replay-ratio sweep. The V49 model, runner, checkpoint
surface, and tests are deleted. Report SHA-256 is
`204bbd170158834017fe5b52c0874491a02112c257ca912586fecc77d3aef7a1`.

### V50 preregistration: isolated plasticity throughout the hierarchy

V50 keeps V49's successful isolation contract and moves the trainable path to
the place V49 could not reach. Every V39 attention QKV/output projection and
SwiGLU gate-up/down projection receives a conditional rank-16 low-rank delta.
The original linear remains frozen and is called directly when the condition is
off, so inactive logits, state, loss, and relations must remain bit-exact. When
on, the deltas can change source selection and feature construction inside all
ten layers rather than trying to reinterpret only the final frozen state.

The run keeps the V48/V49 SQuAD train and validation manifests, 4x answer loss,
2,096,640 new-domain tokens, generated-only decode controls, and 18/64 grounding
floor. It uses no replay because old behavior is structurally isolated. The
active path must beat V48's 14/64 by at least five points, exceed the stronger
source control by ten points, keep added parameters below 5% of V39, give every
delta tensor a final gradient, and strict-reload if promoted. A pass advances
hierarchical modular plasticity toward a learned router. A miss deletes the
implementation and rejects low-rank conditional adaptation at this scale; the
next architecture must own a separate source encoder or change the base cortex.

V50 is terminal and negative. The all-layer rank-16 deltas add 2,457,600
parameters (2.44% of V39), and every tensor receives a nonzero final gradient.
Training loss falls steadily from about 3.55 to 2.76 over the matched 2,096,640
new-domain tokens. The optimized physical-batch-224 path completes 130 updates
in 88.76 seconds at 23,622 tokens/s, with 8,967,276,544 bytes peak allocation
and only 9,830,400 bytes of optimizer state.

The inactive path again passes perfectly: parent hashes and sample logits are
bit-exact, general loss stays exactly 3.149025917, and relation ranking/free
recall stays 63/64 and 57/64. Active source grounding improves from V49's 1/64
to 5/64, with 0/64 question-only and mismatched controls. This proves that
plasticity throughout depth accesses more useful source information than a final
sidecar, but it remains far below V48's 14/64 full-weight arm and misses the
ten-point source-gain gate at only 7.81 points.

Decision: `retire_v50_hierarchical_lora_insufficient_grounding`. Low-rank deltas
solve retention but do not provide enough functional freedom for this capability
at matched exposure. The next modular architecture must use higher-capacity
residual functions inside depth or own a separate source encoder; another LoRA
rank, replay ratio, or final adapter is not justified. The model, runner,
checkpoint surface, and tests are deleted. Report SHA-256 is
`c97ba0505aa06c3976802430851abc8a3f321f110960ac437320a26307d46541`.

### V51 preregistration: full specialist fork as the modular upper bound

V48, V49, and V50 leave one ambiguity: modular isolation may be correct while
the tested modules are simply too constrained. V51 removes that ambiguity by
copying the complete V39 cortex into a source-QA specialist. The original V39
checkpoint is immutable and owns the inactive route; the specialist begins
tensor-identical, receives exactly 2,096,640 SQuAD tokens with 4x answer loss,
and gets no replay. Only one 100.68M model executes per route, but durable storage
doubles. This is a capacity upper bound, not the desired efficient endpoint.

The specialist faces the same immutable V47 intact/question-only/mismatched
validation and must reach at least 18/64 intact answers, exceed the stronger
control by ten points, and beat V48's 14/64 by at least five points. The original
route must retain exact checkpoint hashes and live general/relation metrics; a
passing specialist must strict-reload. A pass advances modular checkpoint banks
to copy-on-write compression and learned routing. A miss means neither compact
nor full isolated plasticity learns enough at matched exposure, so the next work
changes training data/objective or the source-processing architecture rather
than inventing another adapter.

The first counted execution exposed a GPU-memory cliff rather than a scientific
result: concatenating 28 source microbatches into a physical batch of 224 kept
the RTX 3060 at 100% utilization and about 11.9 GB allocated, but produced no
terminal artifact after roughly 41 minutes. It was terminated and is not counted
as V51 evidence. The replacement uses exact gradient accumulation across the
same 28 ordered microbatches, token budget, optimizer-step count, objective, and
seed. This is an execution correction only; it changes neither the frozen
hypothesis nor its gates.

V51 is terminal and negative. The corrected run trains all 100,679,424
specialist parameters for exactly 2,096,640 tokens in 395.80 seconds at 5.30k
tokens/s with 3.19 GiB measured peak allocation. Intact source grounding reaches
12/64 (18.75%) while question-only and mismatched-source controls each reach
1/64, so the 17.19-point causal source gain passes. It nevertheless loses two
cases to V48's answer-weighted shared model and misses the preregistered 18/64
floor. Training loss collapses to 0.0101 while the specialist's general heldout
loss regresses from 3.1490 to 5.1532. The parent checkpoint and state remain
exact; no candidate is saved because the capability gate fails.

Decision: `retire_v51_full_specialist_insufficient_grounding`. Increasing the
isolated path from 2.46M low-rank parameters to a full 100.68M model does not fix
the grounding failure in the executed stream-packed curriculum. The runner and
tests are deleted. Report SHA-256 is
`9b74355e9c287270e28a6fa5b9c54ad79bd25428983d3c5e61a6bd10ea033fad`;
its immutable source audit SHA-256 is
`92d444b8ab5c097a66102ac42d2ea7488a2e81f46ee5f32a68889b81d0c07cb5`.

### V52 preregistration: repair record alignment before architecture

A post-V51 token audit found a stronger and cheaper falsifier than another
module. All 512 encoded SQuAD training records are at most 73 tokens and can fit
the V39 causal window, but the corpus builder concatenates documents and slices
the global stream every 72 tokens. Only 80/512 records (15.625%) retain their
complete context, question, and answer in one training window. The answer marker
and full answer remain together in 466/512, which explains how answer loss can
collapse while the source evidence needed to answer is often truncated. This
does not invalidate the measured V48--V51 outputs; it invalidates the broader
inference that their failure is independent of training-record alignment.

V52 changes one variable. Each grounding record becomes one document-aligned
73-token row: BOS, complete prompt, first answer, EOS, then right padding after
EOS. Pad targets contribute zero loss. The real-token objective remains V48's 4x
answer-weighted causal loss. V39 initialization, 4,193,280 processed positions,
50% grounding/50% identical replay schedule, batch 8, 28-way exact gradient
accumulation, 130 optimizer updates, Muon/AdamW settings, seeds, V47 controls,
relation panel, and general holdout are frozen. Runtime projection must remain
below 1,200 seconds on the RTX 3060.

The source gate requires at least 18/64 intact answers (28.125%), at least ten
points over the stronger question-only or mismatched-source control, and at least
five points over V48's 14/64. The retention gate permits at most five points of
free-relation regression and +0.05 general-loss regression. If both pass, save a
strict-reload candidate for confirmation. If source passes but retention fails,
do not promote the checkpoint; retain the aligned-record contract and test an
isolated pointer/copy or span-supervised source path. If source fails, alignment
alone is insufficient: delete the V52 runner, tests, and task-specific alignment
path before moving to an explicit source-answer interaction architecture.

V52 passes capability and fails retention. Document alignment raises intact
grounding from V48's 14/64 to 19/64 while question-only and mismatched-source
controls both fall to 0/64. The resulting 29.69-point source gain, 7.81-point
gain over V48, exact token budget, and full parameter gradients pass every source
gate. This validates the packing audit as causal rather than cosmetic. It also
partially improves forgetting versus V48, but not enough: strict free relation
recall is 56.25% versus the 89.06% baseline, and general loss is 3.229799509
versus 3.139640808 (+0.09016). Training processes 4,193,280 positions in 802.53
seconds at 5.23k tokens/s with 3.19 GiB peak allocation.

Decision: `advance_v52_aligned_signal_to_isolated_copy`. The shared-weight
checkpoint is rejected and deleted. The aligned-record builder and post-EOS pad
mask survive because they caused a preregistered capability pass; the one-off
V52 runner and gate tests are deleted. V53 must freeze V39 and add an explicit
source-token copy/span path trained from the validated aligned rows. Inactive
parent logits, state, general loss, and relations must remain exact. Report
SHA-256 is `23ef805fae825cd3bd46dd5a85c1deebc3eaabe38db59a9f3750657b6557e33d`;
source audit SHA-256 is
`ff2b6aec6c8a000cddea1004c1f8124b1edb01b75f6fe89550fbd15001d290bd`.

### V53 preregistration: frozen-base source pointer

V52 proves that intact causal records can teach source-grounded answering, while
its retention failure proves that the signal must not rewrite the shared cortex.
V53 freezes the exact V39 checkpoint and adds a MARULHO-owned rank-64 pointer
head. Queries come from the frozen final hidden state; keys come only from token
states structurally located between `Context:` and `Question:`. A causal
attention distribution is scattered through the checkpoint tokenizer IDs to
form a copy distribution, then a learned scalar gate mixes copy probability with
the frozen vocabulary distribution. The head can therefore point to rare source
tokens without storing a second language model. When the source route is
inactive, V39 executes unchanged.

The head receives exactly 2,096,640 processed positions from V52's 512 aligned
SQuAD records, no replay, batch 8, 28-way gradient accumulation, and 130 optimizer
updates. Only answer targets contribute loss; post-EOS pads are excluded. The
parent checkpoint, state, tokenizer, logits, general loss, and relation panel
must remain exact. Every pointer parameter must receive a final gradient, added
parameters must stay below 0.25% of V39, the runtime projection must stay below
1,200 seconds, and a passing artifact must strict-reload against the exact parent
hash.

Capability requires at least 18/64 intact answers, at least +20 points over the
stronger question-only or mismatched-source control, and no more than one case
below V52's 19/64. A pass advances the frozen-base copy route to learned routing,
broader domains, and checkpoint-bank integration. A miss deletes the head,
runner, tests, and checkpoint surface; it means frozen final states do not expose
enough answer-localization geometry, so V54 must add a trainable source encoder
or explicit span supervision rather than another residual adapter.

The first V53 execution completed all 130 counted optimizer updates in about
3.5 minutes, then stopped before heldout evaluation or arm serialization because
the experimental wrapper omitted the protocol's `next_token_loss` method. It
produced no report, checkpoint, or capability evidence. The frozen contract is
unchanged; the required inactive-route evaluation method is added and tested
before a clean rerun.

The corrected rerun completed training and all capability/retention evaluation,
then stopped while serializing two tensor-valued telemetry scalars to JSON. The
exact-contract arm artifact and independent source audit were already durable;
no metric was inspected or tuning changed. Telemetry now converts those values
to host numbers only when collection is requested, and the runner resumes the
completed arm to repeat evaluation/reporting without retraining.

The attempted resume showed that the cached arm row itself retained the old
tensor-valued telemetry. Rather than mutate an exact-contract artifact in place,
the temporary arm is discarded. A clean rerun under the already committed
telemetry fix is required; seeds, schedule, thresholds, and scientific contract
remain unchanged.

The clean V53 run is terminal and negative by the frozen gate. The 99,073-
parameter pointer (0.0984% of V39) trains every parameter across exactly
2,096,640 aligned positions in 185.39 seconds at 11.31k tokens/s with 0.56 GiB
peak allocation. Intact/question-only/mismatched grounding is 17/64, 0/64, and
0/64. The 26.56-point source gain is real and beats V48, but misses the 18/64
floor by one case and is two cases below V52, violating both capability checks.
Parent checkpoint, state, logits, general loss, and relation evidence remain
exact, demonstrating that structural isolation works.

Decision: `retire_v53_frozen_source_pointer_insufficient_grounding`. The result
does not justify post-hoc rank, learning-rate, or threshold tuning. Frozen final
states plus a pointer are slightly too weak; V54 must train how source tokens are
represented or supervise answer spans directly while preserving an immutable
language base. The pointer model, runner, tests, compact checkpoint surface, and
compatibility path are deleted. Report SHA-256 is
`3af6ebad988b2844d83b91f73fe3f7c22443dab933e5f6fcbf9a1bbf48ae4620`;
source audit SHA-256 is
`72c89a9b714c4297913ca4f8b0e99a6c3fefb1930e74c8b7df28825fad30ca2e`.

### V54 preregistration: trainable source encoder with direct span supervision

V53 demonstrates that exact isolation is cheap and effective, but a pointer over
frozen final language states stops at 17/64. V54 changes the representation and
learning signal together, as preregistered after that terminal miss. The exact
V39 checkpoint remains immutable. Its checkpoint-owned token embeddings are
read-only inputs to a separate width-128, two-layer, four-head bidirectional
source/question encoder. A trainable projection, positional/type embeddings,
and start/end scorers learn the answer span directly. At inference the module
copies the best source-contained span of at most eight tokens. If no explicit
`Context:` field exists, V39 owns generation unchanged.

This combines established span-QA and pointer ideas rather than claiming a new
primitive. [Pointer Networks](https://arxiv.org/abs/1506.03134) established
selection over input positions; [BiDAF](https://arxiv.org/abs/1611.01603)
established explicit context-question interaction; and
[BERT](https://arxiv.org/abs/1810.04805) popularized direct start/end span
supervision. MARULHO's falsifiable claim is narrower and architectural: such a
source organ can be trained continually, stored separately, and activated
without modifying or replaying the general language cortex.

The matched V54 screen uses exactly the same 512 SQuAD training cases and fixed
64-case V47 holdout as V52/V53. It processes 2,096,640 padded positions in 455
batch-64 updates; every example keeps the complete prompt and gold span. Only
the source encoder trains, with AdamW, cosine decay, BF16 CUDA, seed 54131, and
no replay. Added parameters must remain below 0.75% of V39. Parent checkpoint,
state, tokenizer, sample logits, general loss, and relation evidence must remain
exact; the module checkpoint is bound to the parent SHA-256 and must strict-
reload.

Capability requires at least 19/64 intact answers, at least +25 points over the
stronger question-only or mismatched-source control, complete trainable-gradient
coverage, and observed training below 1,200 seconds. A pass advances immediately
to a larger disjoint SQuAD training set, multi-source span/copy data, and learned
route integration. A miss deletes the encoder, runner, tests, and checkpoint
surface, and moves to joint generative-plus-span training or a longer-context
source architecture rather than tuning width/depth after the result.

The terminal V54 run is negative. All 373,506 trainable parameters receive
nonzero gradients and span loss falls from 3.4005 to 1.5942. The exact 2,096,640
positions finish in 12.05 seconds at 173,965 positions/s with 498,480,128 bytes
peak allocation. Parent checkpoint, tensors, logits, general loss, relation
behavior, tokenizer, and compact reload remain exact. Heldout intact/question-
only/mismatched-source accuracy is 16/64, 0/64, and 0/64. The source gain passes
at exactly 25 points, but capability misses the 19/64 gate by three and trails
V53/V52 by one/three cases. Decision:
`retire_v54_span_encoder_insufficient_grounding`.

The errors change the next design. Ten of 48 misses contain a real but truncated
answer fragment, including `Denver` for `Denver Broncos`, `174` for `1745`, and
`18 February 154` for `18 February 1546`. More importantly, V54 shares only
7 successes with V53 and 6 with V52. The separate V52/V53/V54 success sets have
an oracle union of 35/64 while only three cases are common to all three. This is
not a runnable ensemble or a quality result: the older failed checkpoints do not
exist, and oracle selection uses labels. It is evidence that causal language
states, bidirectional source states, and autoregressive realization make
different errors. The next falsifier should learn across those views and produce
answer tokens autoregressively with span supervision as an auxiliary signal,
not tune V54's width, depth, or boundary threshold. The V54 model, runner, tests,
checkpoint surface, and temporary checkpoint are deleted. The generic tokenizer
offset contract remains because it is still required for the auxiliary span
labels. Report SHA-256 is
`32f2c700c8168c6fdccb4c681afda978f9113645b0686ca44253b81aed04d0e0`;
source audit SHA-256 is
`3eefb53d6a448fb024a79b0f84469f4ee1f9d198d2cd9876decd19f4e2923f9c`.

### V55 preregistration: multi-view autoregressive answer transducer

V55 follows the V54 branch rule directly: it replaces terminal span copying
with joint generative-plus-span training and scales the evidence, rather than
tuning the failed encoder width or threshold. The exact V39 language model is
immutable. For each source/question prompt, V55 reads two distinct frozen-parent
interfaces:

1. V39 causal final states, which carry the language model's learned sequential
   representation and reproduce the useful V53 interface;
2. frozen V39 token embeddings passed through a new width-192, two-layer,
   six-head bidirectional source/question encoder, which retains V54's direct
   access to both sides of the question-context relation.

A learned normalized fusion feeds a width-192, two-layer autoregressive pointer
decoder. Starting from BOS, it predicts one legal source-token position at a
time and finally EOS, so answer length and boundaries are learned as a sequence
rather than inferred from two independent endpoint scores. An auxiliary
start/end loss keeps exact localization pressure. A frozen-token embedding of
the previously selected source token feeds the next decoder step. This remains
fully MARULHO-owned and can output only evidence-visible tokens in this
extractive falsifier. Source-absent prompts route to unchanged V39. A
deterministic 70/15/15 schedule trains both views, bidirectional-only, and
causal-only examples so neither path can be dead while the fused result appears
strong.

The corrected V55b immutable manifest uses the official SQuAD training split but
excludes all 512 prior V48 training IDs and all 64 fixed V47 validation IDs. It
contains 8,192 cases from 134 titles; each complete prompt fits 64 tokens and
each answer occupies one to eight tokens at its actual position inside the
source. Preflight rejected the first manifest before training because two
answers that encoded to eight tokens alone occupied nine contextual BPE tokens.
That invalid live artifact is deleted and V55b records both rejected IDs and the
superseded contract. The validation panel and its question-only and mismatched-
source controls remain unchanged. V55b manifest contract SHA-256 is
`d58ac8d0b5b8337ab9cf5577991cb3f9ca015d86f44aecb6e9379e6db2fcf395`;
file SHA-256 is
`d51c56125e5e3a31e857b90b00de8def45fdf56b8afc1077a7a9c793dfc508fc`.
This is a 16-times larger unique curriculum than V54. Fifteen exact epochs at
batch 64 and context 72 process 8,847,360 padded source positions in 1,920
optimizer updates. Frozen V39 causal states may be cached in host BF16 memory;
cache construction time and bytes remain part of the report and total runtime.
The optimizer is AdamW at 3e-4 with BF16 CUDA, 5% warmup, cosine decay, 0.1
minimum fraction, 0.1 weight decay, gradient clipping at 1.0, data seed 55121,
and model seed 55131. No replay is needed because the parent cannot mutate.

The architecture may add at most 2.5% of V39 parameters and must finish causal-
state caching plus training within 1,200 seconds. It must give every trainable
tensor a nonzero gradient and strict-reload a compact checkpoint bound to the
V39 file hash. Parent checkpoint, state, tokenizer, sample logits, heldout
general loss, and relation behavior must remain exact. Capability promotion
requires at least 32/64 intact answers, at least +45 points over the stronger
question-only or mismatched-source control, and a minimum four-case advantage
over each trained single-view inference ablation. The 32-case bar doubles V54
and materially exceeds V52; the ablation bar prevents an attractive multi-view
story from surviving if one view actually owns the result.

A pass advances the compact source organ to multi-document retrieval, larger
mixed extractive/generative corpora, and an owned learned route from the base
language model. A miss deletes the transducer, runner, tests, checkpoint surface,
and caches, then moves to a longer-context retrieval architecture rather than
another SQuAD-only source head.

The terminal V55 result is a clean negative with one retained architectural
signal. All 2,130,819 trainable tensors receive nonzero gradients. Total loss
falls from 4.1729 to 1.1699; final pointer/span losses are 0.8435/1.3055. The
8,847,360-position run trains in 145.45 seconds at 60,827.8 positions/s. Frozen
V39 causal-state caching adds 12.15 seconds and 905,969,664 host bytes, giving
157.60 seconds total and 56,137.4 cache-amortized positions/s. Peak CUDA
allocation is 891,945,472 bytes. Parent checkpoint, state, logits, general loss,
relation behavior, tokenizer, and compact reload are exact.

Both-view intact/question-only/mismatched-source accuracy is 20/64, 0/64, and
0/64. The causal-only ablation reaches 16/64 while bidirectional-only reaches
2/64. Fusion therefore adds exactly four cases over the stronger single view and
passes the preregistered synergy check. Capability still misses 32/64 by twelve,
and a 31.25-point causal source gain misses the 45-point bar by nine cases. The
result is a real but insufficient new best for an isolated frozen source organ,
not a promotion.

Failure inspection explains why lower training loss did not translate. Fourteen
of 44 misses contain some gold answer words, but the autoregressive position
decoder can select noncontiguous BPE pieces. It produces malformed assemblies
such as `Carololina`, `William the Conquor`, `historical divisionsisions`, and
invalid UTF-8 replacement characters. Sequential position prediction solved
neither token-boundary safety nor general answer realization. V55 has five
successes absent from every V48/V52/V53/V54 report; the oracle union across all
five separate systems is 43/64. That union uses labels, has no surviving older
checkpoints, and is not ensemble accuracy. It reinforces heterogeneous errors
but does not justify retaining dead implementations.

Decision: `retire_v55_multiview_transducer_capability_or_ablation_failure`.
The model, runner, tests, compact checkpoint surface, temporary state, and
nondurable 906 MB cache are deleted. The corrected V55b data manifest remains
as immutable evidence. The next falsifier must leave the repeated context-72
SQuAD head family: retrieve or compress longer source segments and realize only
token-safe contiguous segments before testing synthesis. Report SHA-256 is
`d1d30b6aec1237277d57be534c7029c66364546b16439ce4ad9830d59bfc6911`;
both/bidirectional/causal source-audit SHA-256 values are
`2e10e523f77a8f446c2599c816d554c554259929938d616778761c9973dfd96d`,
`0922f5e9b82c0bf0e4baae370d255c822c1fc0e2733785493b16dd5a0b3f38c4`, and
`0d37b0493c16fdcb471fd407e4ecb21eb366622e6c8cadfcec9dc95ea16cfdfd`.

### V56 preregistration: landmark retrieval inside the causal language path

V54 and V55 show that source isolation is cheap and preserves the parent, but a
separate copy decoder does not inherit enough language realization. V56 changes
the interface rather than enlarging the head: retrieved evidence enters the
causal representation whose own vocabulary head already produces coherent text.
This follows two relevant primary results. [RETRO](https://arxiv.org/abs/2112.04426)
conditions an autoregressive LM on retrieved chunks through a differentiable
encoder and chunked cross-attention, and reports that pretrained Transformers can
be retrofitted rather than always trained from scratch. [Landmark Attention](https://arxiv.org/abs/2305.16300)
learns block representatives and retrieves selected blocks inside attention.
[RAG](https://arxiv.org/abs/2005.11401) and [Memorizing Transformers](https://arxiv.org/abs/2203.08913)
support the broader separation of parametric language from editable external
evidence. V56 tests only the smallest locally falsifiable core of that family;
it does not import their weights, corpora, retrievers, or capability claims.

The exact V39 model is frozen. Each 96–228-token source is tokenized into up to
five non-overlapping 48-token blocks and encoded blockwise by frozen V39. The
question-only causal prompt is encoded separately. A trainable 128-wide query/
landmark projection scores blocks from the question's final hidden state and
mean-pooled source states. Multi-label binary retrieval loss marks every block
overlapping the answer. The answer never enters the retrieval query. Gold top-
two evidence preserves boundary-spanning answers; a one-block answer receives a
deterministic adjacent block as context. Retrieval scores gate each block's
cross-attention contribution so the second block need not become an equal-weight
distractor. Predicted top-one is reported as a matched diagnostic because older
V24 evidence showed that indiscriminate top-two context can be harmful.

The implementation freezes one additional tokenization invariant before the
run. Retrieval encodes the bare question, excluding structural `Question:` and
`Answer:` text. The generator encodes its question/answer prefix separately from
the accepted answer, with a virtual trailing space after `Answer:`. This prevents
whole-string BPE merges from making the teacher-forced prefix differ from the
runtime prefix. All train/validation sequences still fit: the separated prefix,
answer, and EOS use at most 73 IDs. Standalone validation answers use at most ten
generated IDs, so evaluation requests twelve tokens; V55's eight-token pointer
limit does not carry into this full-vocabulary experiment.

For generation, frozen V39 encodes the question plus teacher-forced answer prefix
inside its existing 72-token window. A trainable 256-wide projection and two
causal Transformer-decoder layers cross-attend to the two selected frozen
evidence blocks. A projected gated residual returns to V39's 768-wide hidden
stream, and V39's tied full-vocabulary head predicts answer tokens plus EOS. This
removes V55's noncontiguous BPE assembly surface: generated IDs are ordinary
checkpoint-vocabulary outputs. Source-absent prompts call V39 directly and must
remain bit-exact. The adapter may add at most 3% of V39 parameters and its compact
checkpoint is strictly bound to the parent file SHA-256.

The corrected V56c train manifest contains 8,192 new official-train cases from
170 titles, excludes every V48/V55 case, and keeps question+answer+EOS within 72
tokens. Source lengths are 96–228 tokens and every case has 2–5 fixed blocks.
Contract/file SHA-256 values are
`efd56051f98ea32fad2474e3f9504d33bec6aac4d7e69978378f3c9547d5552d` and
`ebc512f0a1d680ce3c9b0f11b52ed9a86395f035be125c5470d4b326e902a5e3`.
The new 128-case official-validation panel excludes V47 and all train IDs, spans
127–246 prompt tokens, and has contract/file SHA-256 values
`feca4f4088d3452265f2fc35240f7aa45de68dfc856e0be80af7f45a9e470a84` and
`fa609b4c6c381d1d0c347fc3286dc2ed5e35daea4c57da8400b20056f0facbc6`.

Frozen source and query states may be cached once in nondurable host BF16 memory.
Cache time, token count, bytes, content hash, CUDA peak, and amortized throughput
remain part of the result. Fifteen exact epochs at batch 32 produce 3,840 updates
and 20,643,840 padded adapter positions: 72 query positions plus 96 selected-
evidence positions per case. AdamW uses peak 3e-4, 5% warmup, cosine decay to 0.1
of peak, weight decay 0.1, BF16 CUDA, clip 1.0, data seed 56121, and model seed
56131. Generator answer/EOS loss and retriever loss have equal weight. Gold
evidence trains generation; predicted, oracle, and shuffled evidence remain
separate heldout interventions so retrieval failure cannot be hidden by label
access.

Promotion requires all of the following:

- the concatenated predicted top-two block union contains a full gold answer in
  at least 80% of validation cases;
- predicted-evidence exact answer accuracy is at least 64/128 and at least 45
  points above the stronger question-only or mismatched-source control;
- oracle-evidence generation reaches at least 72/128, predicted evidence remains
  within ten cases of oracle, and shuffled evidence stays at or below 8/128;
- every trainable tensor receives a nonzero final gradient, added parameters stay
  below 3%, and cache plus training remains below 1,200 seconds;
- parent checkpoint/state/tokenizer/sample logits/general loss/relation behavior
  remain exact, and the compact parent-bound checkpoint strict-reloads exactly.

V47 short-source accuracy is reported for continuity but cannot pass V56. A pass
advances the evidence interface into a durable block store, continual writes,
retrieval provenance, and learned event-controlled activation. A miss deletes
the retrofit and moves to context-length expansion or recurrent segment memory;
it does not reopen endpoint/pointer tuning.

#### V56 terminal result

V56 is a valid negative and is retired. All 2,383,361 trainable parameters
receive nonzero final gradients and loss falls 6.4843 to 2.8132. The full
20,643,840-position schedule trains in 268.02 seconds at 77.02k positions/s;
nondurable train/validation caching adds 64.91 seconds and 4.91 GB host storage.
Parent and compact-checkpoint fidelity pass exactly.

The retriever reaches 91/128 top-two answer coverage and 55/128 top-one coverage,
missing the 80% gate. That is not the primary failure: oracle answer-containing
evidence still yields 0/128 exact generations, tied with predicted top-two,
top-one, and shuffled evidence. Question-only yields 1/128. Thus the frozen
cross-attention residual learns its supervised losses but cannot reliably steer
V39's autoregressive realization. This rejects the V56 interface, not retrieval
or long-context memory in general. Do not tune it on this validation panel. Move
to a base-native context mechanism or recurrent segment state where evidence can
participate throughout the language computation.

### V57 terminal result: native context is insufficient and destructive

V56 separates two hypotheses cleanly. Learned top-two retrieval reaches 71.09%,
but oracle evidence still produces no correct answers. The next experiment must
therefore stop freezing the language computation. V57 asks whether V39 can learn
long evidence use when the source participates in every causal layer, while a
matched oracle-localized arm determines whether answer realization itself is
still the blocker.

The exact 100,679,424 V39 tensors are strict-loaded into the same Transformer
with rotary context expanded from 72 to 320. No parameter, tokenizer token, head,
sidecar, pointer, or external model is added. Before counted training, common
prefixes up to 72 tokens must remain logit-exact. Both arms reset to that state:

- `native_full` receives each complete 128–278-token source;
- `oracle_short` receives the stored answer-bearing source region, with its
  location supplied by the benchmark rather than learned.

Both inputs are right-padded to 320, so parameter count, tensor shape, attention
compute, cases, answers, initialization, optimizer, and schedule remain matched.
The oracle arm is a control only. Each causal prefix ends in the same encoded
trailing space used at generation; the standalone answer and EOS are appended
without a cross-boundary BPE merge. Ordinary next-token loss covers the complete
record and the existing V39 answer objective raises answer/EOS targets to weight
4. General and relation replay use ordinary full-vocabulary loss.

The new train manifest has 8,192 cases from 171 titles and excludes every prior
V48/V55/V56 train case. The new validation panel has 256 cases from 22 titles and
excludes V47/V56 validation. Train contract/file SHA-256 values are
`fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7` and
`aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133`;
validation values are
`9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c` and
`b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80`.

Training uses four complete grounding epochs at batch 32. Exactly half of 2,048
updates are grounding, one quarter are document-aligned relation replay, and one
quarter are balanced across the two general sources. Each arm processes exactly
20,971,520 padded positions; together they process 41,943,040. Muon uses 3e-4,
5% warmup, cosine decay to 0.1 of peak, weight decay 0.1, BF16, clip 1.0, data
seed 57121, and model seed 57131. Eager execution is frozen. Its real full-step
preflight reaches 16,129.8 positions/s at 7.15 GB. Inductor is rejected because
loss drift is 0.001868 above the 0.001 tolerance and steady speed collapses to
328.5 positions/s after a 165.24-second compile.

Promotion requires all of the following:

- oracle-short exact generation at least 128/256;
- native-full exact generation at least 128/256 and at least 45 points above the
  stronger question-only or mismatched-source control;
- native-full no more than 16 cases behind oracle-short, and mismatched source at
  most 16/256;
- matched general heldout loss regression at most 0.10 and stratified relation
  exact-generation regression at most five percentage points from V39;
- exact 20,971,520 positions and 2,048 updates per arm, unchanged parameters,
  complete nonzero final gradients, and at most 1,800 counted training seconds
  per arm;
- initial short-prefix parity, tokenizer identity, parent-file immutability, and
  strict candidate tensor/logit/checkpoint reload.

Both trained models are evaluated on full and oracle prompts. If oracle passes
and native fails, V57 identifies long-range localization/integration as the
blocker and pivots to recurrent segment state. If oracle fails, context length
is exonerated and the base/task objective must change. If native capability
passes but retention fails, replay/objective design is the blocker. Only a joint
native pass advances to larger unique data, continual evidence writes, and the
same-checkpoint sustained runtime qualification.

Runner profiling before a terminal arm exposed and removed two non-scientific
costs: unused parent grounding generation, and per-parameter nonzero-gradient
host readbacks on every update. The latter forced roughly 160,000 CUDA
synchronizations. V57 now retains the preregistered final-step gradient truth,
parent general/relation baselines, and every trained-arm source control without
those costs. Interrupted profiling invocations emitted no report or checkpoint
and are not experiment results.

The terminal matched result is valid and negative. Oracle-short generates
122/256 exact answers; native-full generates 43/256. The native model reaches
90/256 when the same heldout evidence is oracle-localized at evaluation, so
long-range localization/integration is independently real. But oracle itself
misses the 128-case capability floor, while general loss regresses from 3.1490
to 3.3712/3.3553 and relation exact generation from 89.06% to 34.38%/75.00%.
Both arms pass parameter, schedule, final-gradient, time, parent, tokenizer, and
strict reload truth at 16.77k/17.03k positions/s. Decision:
`retire_v57_context_exonerated_base_or_objective_failure`. Do not repeat full
base-tensor fine-tuning with a longer window. The next architecture must provide
a protected plastic pathway that participates throughout depth while leaving
the qualified semantic cortex exactly recoverable, and must address evidence
localization explicitly rather than relying on dense attention to discover it.
Report SHA-256 is
`fe93519ca693837796c76ba8e1161e68e7f4d210ad31a47341f854f90660cb99`.

The frontier architectures suggest later, separate falsifiers rather than one
large hybrid rewrite:

1. test block-level attention residuals or a very small constrained residual
   workspace before another new token mixer, because this adds alternative depth
   paths without deleting exact token attention;
2. test a shared-dense plus latent-routed FFN only after the baseline throughput
   gate, with expert utilization, effective active parameters, and wall time
   measured explicitly;
3. revisit delta state only as a 3:1 hybrid with exact attention, bounded decay,
   parallel chunk execution, and a long-context dependency that context-72
   attention cannot already solve;
4. defer compressed/sparse attention, KV quantization such as TurboQuant, and
   million-token curricula until context length makes KV traffic a measured
   bottleneck. They do not address today's context-72 training cost or grounding
   failure.

### V58 preregistration: protected bidirectional evidence organ

V57 removes the context-length excuse but leaves two different failures:
full-source localization is weak, and updating the common language cortex
destroys retained behavior. V58 therefore stops asking one causal model to own
both stable language and exact document lookup. It tests a deliberately large
capacity ceiling before spending more runs on compact adapters.

The V39 causal cortex and checkpoint remain immutable and continue to own every
source-absent language request. A separate MARULHO-owned evidence organ clones
V39's checkpoint-trained token embedding, ten full-width Transformer blocks,
and final norm, but runs the cloned blocks bidirectionally over one bounded
`Context` plus `Question` record. It has no vocabulary head. Two new linear
scorers predict the start and end of one contiguous Unicode-character source
span from token hidden states plus bounded within-token boundary features.
Inference copies that consecutive source interval exactly; it cannot join
noncontiguous positions or split a Unicode character, so V55's malformed BPE
assembly is structurally impossible. The organ is protected rather than
residual: it never alters the causal cortex, and source-absent generation
bypasses it exactly.

This is not presented as a novel span-QA primitive. V54 already tested a
373,506-parameter, two-layer span encoder and reached only 16/64; V55 tested a
2.13M-parameter multi-view pointer and reached 20/64. V58 asks the missing
capacity question with approximately one full V39 body and the title-disjoint
V57 data boundary. A failure therefore closes this SQuAD-style extractive-organ
line rather than causing another width, depth, endpoint, or pointer sweep.

The primary `v39_initialized` arm uses all 8,192 frozen V57 training cases and
the 256 frozen validation cases, whose 171/22 titles are disjoint. Each
source/question record is padded to 320 positions. Eight exact epochs at batch
32 give 2,048 optimizer steps and 20,971,520 padded positions. All organ
parameters train with BF16 AdamW, peak learning rate 1e-4, 5% warmup, cosine
decay to 10% of peak, weight decay 0.1, gradient clipping at 1.0, data seed
58121, and model seed 58131. Counted training must finish within 1,800 seconds
on the RTX 3060. No general or relation replay enters the organ because the
causal parent is structurally unreachable from its optimizer.

Before training, every gold answer must map to an exact contiguous source-
character span whose copied text matches an accepted normalized answer; this
makes the mechanical oracle 256/256. V57's stored source offsets are only hints:
the immutable answer text resolves the nearest exact occurrence because 34
training and 2 validation offsets are off by one or more characters after source
bounding. No label is consulted at evaluation after these training targets are
materialized. Promotion requires at least 192/256 learned exact
spans, at least 70 percentage points over mismatched-source extraction, at most
8/256 mismatched exact answers, all 2,048 steps and positions, nonzero final
gradients for every organ tensor, parent file/tokenizer/state/logit identity,
and exact organ tensor/logit reload. Question-only requests have no episodic
source and therefore exercise the unchanged causal route rather than receiving
an invented empty-document classifier.

If the initialized arm passes capability, one mandatory `random_initialized`
arm runs the same architecture, records, order, optimizer, and budget. A
16-case initialized advantage supports transfer from the language cortex. If
both pass without that margin, the exact evidence organ may still advance as a
specialized memory mechanism, but not as evidence that language pretraining
created the localization skill. If the primary arm fails, the random control is
unnecessary and the entire extractive-organ line is retired. A passing organ
advances to learned routing, mixed extractive/generative evidence, and then
compression against this ceiling; it does not by itself qualify general
grounded generation or a frontier model.

#### V58 terminal result

V58 is a valid, decisive negative. The 100,686,146-parameter initialized organ
trains all 68 tensors for the exact 2,048 updates and 20,971,520 padded
positions. Counted training takes 871.75 seconds at 24,056.8 positions/s with
5,386,713,600 peak CUDA bytes; final sampled span loss is 1.9638. The mechanical
copy oracle remains 256/256 and V39's checkpoint file, tokenizer, tensor state,
and sampled logits remain exact.

Heldout title-disjoint accuracy is only 20/256, versus 0/256 with mismatched
sources. The 7.81-point source gain is real but far below the required 70
points, and capability misses the 192-case floor by 172. Sixty-one predictions
have a substring relation to an accepted answer and 76 contain a digit, while
failure examples repeatedly choose answer-type-looking dates, numbers, or
`Stadium` fragments. Full capacity and falling training loss therefore do not
produce transferable query-conditioned localization under this data boundary.
The random-initialization control is correctly skipped because the primary gate
fails.

Decision: `retire_v58_extractive_evidence_organ_capacity_failure`. This closes
the V53--V58 SQuAD extractive-organ family: frozen pointers, small span encoders,
autoregressive position decoders, residual readers, unrestricted native-context
fine-tuning, and now one full-depth protected span organ all fail their frozen
capability gates. No V58 checkpoint survives. The runner, model, tests, and
loading surface are deleted; only the report remains. Report SHA-256 is
`761f3f385d0524a880f568f956aaafbf0f520b4124c4a2a525302707229331e2`.

The next branch must leave answer-span supervision rather than widening this
organ again. The leading research direction is protected write-time learning:
evidence should change a bounded temporary learner through a source-native
self-supervised objective, then answer through that learned state, while the
slow causal cortex remains immutable. This connects MARULHO's continual-memory
goal to test-time training and nested learning without claiming those mechanisms
work locally before a matched falsifier exists.

### V59 preregistration: source-native write-time learning ceiling

V59 tests whether gradient descent itself can act as an episodic write before
MARULHO invents a compact learned update rule. This follows the central TTT idea
that a model can compress test context into weights through a self-supervised
objective, while keeping the local claim much narrower than
[TTT](https://arxiv.org/abs/2407.04620),
[end-to-end TTT](https://arxiv.org/abs/2512.23675),
[Titans](https://arxiv.org/abs/2501.00663), or
[Nested Learning](https://arxiv.org/abs/2512.24695). V59 has no meta-learned
update rule and makes no long-context speed claim.

For each heldout document, a temporary BF16 copy of all V39 parameters resets
exactly to the qualified parent. It sees only `Context: {source}` under ordinary
next-token prediction: no question, answer, span, answer weight, validation
label, retrieval oracle, or external model enters the write loss. Four causal
epochs traverse contiguous context-72 windows in source order. AdamW uses peak
learning rate 1e-4, betas 0.9/0.95, no weight decay, and gradient clipping at
1.0. The adapted copy then greedily answers the source-absent
`Question: ...\nAnswer: ` prompt with V44 generated-only repetition 1.1 and
no-repeat-3 controls. Its weights are discarded before the next case.

The panel is a frozen 64-case round-robin sample from all 22 V57 validation
titles: cases sort by title and case ID, then selection cycles across title
depths until 64. Ordered case-ID SHA-256 is
`185a9963bd28d53f04d075cc54937e0d6ca75ffc7719ac5979359ca1ee84e94f`.
Four evaluation arms share questions and decoding:

1. `no_write` uses exact V39;
2. `mismatched_write` learns from the manifest's wrong document;
3. `true_write` learns from the intact document;
4. `oracle_short_write` is a diagnostic that learns only from V57's stored
   answer-bearing source region.

`true_write` is the only candidate. It must reach at least 16/64 exact answers
and beat the stronger no-write or mismatched control by at least 12 cases.
Mismatched write must stay at or below 8/64, while oracle-short must reach at
least 24/64 to show that source-native learning can expose an answer when
localization is removed. At least 90% of true-write cases must lower their own
source next-token loss from first to final epoch. Every adaptive parameter
tensor must receive a nonzero gradient, total terminal wall time must stay below
2,400 seconds, every per-case reset must start from the exact common BF16 state,
and the parent checkpoint file, tokenizer, CPU state, and sampled logits must
remain exact.

A joint pass advances to a compact per-example fast learner and meta-learned
initialization. Oracle-only success localizes the next design to learned source
selection. Oracle failure rejects naive source-only full-model adaptation as a
usable QA memory and requires a meta-learned write objective rather than more
inner epochs or learning-rate tuning. No transient adapted checkpoint is ever
durable.

The reports also reinforce a negative conclusion: frontier quality still comes
with enormous data, capacity, careful curation, and post-training. Their
architecture choices can improve MARULHO's compute frontier, but none provides a
shortcut from 201M local tokens to frontier knowledge. The scientific opportunity
is a better quality-per-token and quality-per-second curve on consumer hardware,
followed by continual adaptation that static frontier checkpoints do not offer.

### Post-V35 direction: spikes control semantic machinery, not semantic bandwidth

Recent results strengthen the hybrid hypothesis but narrow its credible form.
[SpikeLM](https://arxiv.org/abs/2406.03287) explicitly reports that binary spikes
do not carry enough semantic information and restores capacity with signed,
elastic amplitudes and frequencies. That is useful evidence that converting the
language stream itself into binary events gives up the part deep continuous
models already do well. [SMixer](https://openreview.net/forum?id=78glEsQB0v)
also separates the label *spiking* from real asynchronous execution: high-performing
spiking Transformers may retain unsupported synchronous operations, while temporal
unrolling makes GPU training expensive. MARULHO will not claim efficiency from
spike counts or addition-only arithmetic without measured wall-clock execution.

The stronger role for spikes is temporal control. [Mixture-of-Depths](https://arxiv.org/abs/2404.02258)
shows that a Transformer can route context-dependent tokens through expensive
layers while retaining a fixed compute capacity and GPU-friendly tensor shapes.
[Titans](https://arxiv.org/abs/2501.00663) shows that attention and learned
persistent memory can have distinct short- and long-term roles. MARULHO's own
prompt-memory experiments already showed that raw surprise is not synonymous
with write utility, so neither result is copied as a solution.

If V35 earns a coherent base, the next falsifier will keep continuous hidden
states and add a small stateful event controller around an independently useful
slow branch. Its membrane integrates predicted marginal utility across chunks;
threshold crossings request extra compute, retrieval, writing, or bounded
plasticity. Actions execute in fixed-capacity chunk batches so nominal sparsity
has a chance to become real CUDA speed. The slow branch must first beat `off`
when always enabled. A temporal spike gate then competes with parameter-matched
dense-sigmoid, independent top-k, random rate-matched, always-on, and always-off
controls. It survives only if it preserves or improves heldout language and
long-range behavior at lower measured active-branch rate and wall-clock cost.

The prospective novelty is not conditional computation alone. It is persistent
event state and hysteresis across chunks, trained against the *marginal utility*
of actions, with multiple action types and durable memory receipts. A per-token
score followed by top-k is a Mixture-of-Depths control, not evidence for an SNN.
Likewise, irregular per-token spikes that save analytic operations but slow the
RTX 3060 are a systems failure, regardless of biological appeal.

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
  overhead for MARULHO's current regime. [H-Net](https://arxiv.org/abs/2507.07955)
  strengthens the case: learned causal dynamic chunks decide when the expensive
  inner model runs and beat fixed/heuristic chunking under matched FLOPs. But its
  controlled English models start around 680M parameters, and crossovers require
  tens of billions of training bytes. Applying the idea after BPE at 21M would
  remove its tokenizer-free advantage. Keep it as a later scale-aware direction,
  not the next 3060 test.

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
