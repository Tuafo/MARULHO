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

#### V59 terminal result

V59 is mechanically valid and terminally negative. All 64 true-source cases
lower their own next-token loss between the first and fourth write epoch. The
true arm performs 844 full-model optimizer steps over 52,012 positions in 91.23
counted write seconds at 570.1 positions/s. Every one of 62 parameter tensors
receives a final nonzero gradient. All 192 adapted-case resets are exact, peak
CUDA allocation is 1,125,761,536 bytes, total wall time is 388.30 seconds, and
the parent checkpoint, tokenizer, CPU tensors, and sampled logits remain exact.

No-write, mismatched-write, true-write, and oracle-short-write strict accuracy
is 0/64 for every arm. True and oracle writes each place an accepted answer
somewhere inside 5/64 continuations, versus zero for both controls, but extra or
corrupted text makes every output fail the frozen exact-answer metric. Examples
include an answer token such as `Apollo` followed by an unrelated continuation.
This is a weak source-dependent bias, not a usable memory read interface.

Decision: `retire_v59_naive_source_only_gradient_memory`. Four source-only
epochs can memorize document tokens but cannot organize that change for clean
question readout, even when the written source is answer-localized. Do not tune
the inner learning rate or epoch count. The transient model, runner, and tests
are deleted and no checkpoint exists. Report SHA-256 is
`388c43f79c10cc306fc12b1f1d7ad245ba42c317e40d18007e11d357d18247f0`.
The surviving hypothesis is end-to-end meta-learning: train the initialization
or write rule so that a source-native inner update becomes useful to later
language loss, with no validation answer available during the write itself.

### V60 preregistration: meta-gradient episodic matrix

V59 shows why ordinary test-time fine-tuning is insufficient: the source loss
collapses, and source-specific answer words occasionally leak into generation,
but the gradient update is not organized for later question readout. V60 tests
the smallest end-to-end remedy. It is a protected fast learner, not another span
head, source pointer, cross-attention reader, or durable fine-tune.

The exact V39 checkpoint remains immutable. Its causal hidden states encode five
64-token source chunks under `torch.no_grad`. V57 permits question-plus-answer
records up to 96 tokens, so the read path reconstructs the same tensors with a
parameter-free rotary context of 96; common prefixes through 72 must remain
logit-exact. A
MARULHO-owned slow controller contains learned key and query projections from
width 768 into eight 16-wide heads, a learned positive write rate per head, one
width-768 read projection, and bounded read gates. It adds fewer than 1% of
V39's parameters. Frozen next-token embeddings from each source are split into
eight 96-wide value heads.

Each document starts with eight zero fast matrices of shape 16x96. One exact
gradient step from zero on masked source next-token embedding reconstruction is
equivalent to the batched outer-product write used by the implementation. Thus
the fast state is produced only from source causal states and their actual next
tokens; the question, answer, span, labels, and validation data never enter the
write. A question-prefix query reads all eight matrices, concatenates the 96-
wide results, and adds a bounded learned residual to V39's final hidden state
before V39's unchanged tied vocabulary head. Teacher-forced answer loss trains
only the slow controller through this write/read computation. At inference the
fast matrices are rebuilt from a new source and discarded after the request.

The frozen V57 title-disjoint boundary is reused: 8,192 training cases from 171
titles and 256 validation cases from 22 unseen titles. Source and query hidden
states may be cached in host BF16 memory; cache construction time and bytes are
reported, and all caches are deleted after the terminal result. Eight exact
epochs at batch 32 give 2,048 controller updates and 20,971,520 padded source-
memory positions. AdamW uses 3e-4, betas 0.9/0.95, weight decay 0.1, 5% warmup,
cosine decay to 10% of peak, BF16, clip 1.0, data seed 60121, and model seed
60131. Counted training must stay below 1,800 seconds and cache plus training
below 2,400 seconds on the RTX 3060.

Before training, the random controller's true-source accuracy is recorded but
cannot promote. After training, zero-memory, shuffled-source, true-source, and
oracle-short-memory views share the same 256 questions and generated-only V44
decode policy. Promotion requires true memory at least 64/256, at least 20
percentage points above the stronger zero or shuffled control, shuffled at most
16/256, oracle at least 128/256, and true no more than 64 cases behind oracle.
Every controller tensor must receive a final nonzero gradient; parameter count,
updates, positions, cache identity, parent file/tokenizer/state/logits, and a
compact controller strict tensor/logit reload must pass exactly.

A joint pass advances this per-document matrix to multi-source routing,
conflict/version writes, and continual checkpoint ownership. Oracle-only success
means the write/read rule works but source localization still needs hierarchy.
If oracle fails, one linear meta-gradient step is too weak and the branch moves
to an iterative MLP fast learner; it does not return to extractive spans or raw
full-model AdamW.

The first terminal invocation completed all 2,048 updates and all 1,280
generation cases, with every learned zero/shuffled/true/oracle view at 0/256,
then stopped before writing a report because the evidence serializer attempted
to byte-view a scalar BF16 gate without first flattening it. This is an
execution-only reporting defect after all behavioral computation. No report or
checkpoint is admitted. The scalar is flattened before hashing and the exact
frozen run must repeat; seeds, data, model, schedule, optimizer, gates, and
interpretation remain unchanged.

### V60 terminal result: retire one-step linear fast memory

The corrected exact rerun is terminally negative. The controller contains
786,449 parameters, 0.7811% of frozen V39, and all six tensors receive final
nonzero gradients. All 2,048 optimizer steps and 20,971,520 padded source
positions complete in 372.55 seconds at 56,292 positions/s; setup plus training
takes 383.82 seconds and peak CUDA allocation is 1,272,701,952 bytes. Frozen V39
state, tokenizer, and common-prefix logits remain exact.

None of that machinery produces usable source-conditioned language. Untrained
true memory, learned zero memory, shuffled memory, true memory, and oracle-short
memory each score 0/256 exact answers and contain zero accepted answers. The
minimum true, source-gain, and oracle gates fail, so no checkpoint is saved.
Decision: `retire_v60_one_step_linear_meta_gradient_memory`. Report SHA-256 is
`76becda7f4d4986eb0bfca1056d2dd14f074c4d348bf5cf0f735c6125e9718fb`.
The failed runner and tests are deleted. This result rejects a single linear
gradient-equivalent outer-product write plus late residual read. It does not
reject end-to-end fast learning; the preregistered next branch is an iterative
nonlinear MLP state trained through downstream answer loss.

### V61 preregistration: iterative nonlinear fast learner

V60 changed a document-specific linear matrix only once. Its exact oracle failure
leaves two coupled hypotheses: the memory update may need nonlinear iterative
learning, or a final-hidden residual may be too late to control V39. V61 changes
only the first factor. This is the strongest fair test of the frozen branch
declared before V60 ran; a later all-depth test remains separate.

The exact V39 checkpoint and tokenizer remain immutable. Five causal context-64
source chunks are encoded under `torch.no_grad`. A slow MARULHO controller learns
three projections from width 768: source keys, source reconstruction targets,
and question queries. Keys and queries split into eight width-32 heads; targets
split into eight width-96 heads. Each head owns a meta-learned width-32 hidden
MLP initialization. For every document, two explicit differentiable gradient
steps minimize masked source target reconstruction with learned positive step
sizes. Those temporary weights are then queried by the question stream, joined,
projected to width 768, and added through bounded per-head and output gates before
V39's unchanged vocabulary head. The slow controller must remain below 2% of
V39. Temporary weights never persist between documents.

The source write sees source causal states and exact next-token embeddings only;
a learned target view maps those frozen embeddings into the eight value heads.
It never sees the question, answer, accepted strings, answer span, validation
title, or metric labels. Outer teacher-forced answer loss is the sole downstream
signal and updates only the shared initialization, learned views, positive inner
step sizes, and readout. Manual per-example gradient formulas keep the two inner
steps exact and differentiable without cloning a full model or aggregating state
across batch members.

V61 freezes the V57 title-disjoint manifests: 8,192 train cases over 171 titles
and 256 validation cases over 22 unseen titles. It uses five context-64 source
chunks and the parameter-free V39 context-96 read path, with exact common-prefix
logits through context 72. Eight epochs at batch 32 give exactly 2,048 outer
AdamW updates and 20,971,520 padded source positions. The outer schedule remains
3e-4, betas 0.9/0.95, weight decay 0.1, 5% warmup, cosine decay to 10%, BF16,
clip 1.0, data seed 61121, and model seed 61131. Inner reconstruction uses
float32 accumulation and two learned softplus-positive rates. Training must stay
below 1,800 seconds and setup plus training below 2,400 seconds.

Before training, true-source accuracy is diagnostic only. Terminal views are
no-write shared initialization, shuffled-source adaptation, true-source
adaptation, and oracle-short adaptation on the same 256 questions and generated-
only V44 decoding. Promotion requires true at least 64/256, at least 20
percentage points above the stronger no-write or shuffled control, shuffled no
more than 16/256, oracle at least 128/256, and true no more than 64 cases behind
oracle. Both inner steps must reduce masked reconstruction loss, all slow tensors
must have final nonzero gradients, slow parameters must remain below 2%, and
data/schedule hashes, isolation, finite state, CUDA allocation, timing, parent
file/tokenizer/state/logit fidelity, and optional qualified checkpoint strict
tensor/logit reload must pass.

A joint pass advances to routed multi-document state, conflict replacement, and
continual consolidation. Oracle-only success means the fast learner is viable
but source localization needs hierarchy. If oracle fails, iterative nonlinear
fast state at the final residual is retired; the next controlled variable is a
protected read injected throughout V39's depth. Widening the MLP, adding more
epochs, or returning to spans is not an admissible interpretation of failure.

### V61 terminal result: retire nonlinear final-residual fast memory

V61 completes the exact frozen run and fails both behavior and mechanism. The
slow controller has 1,605,657 parameters, 1.5948% of V39, and every one of its
12 trainable tensors receives a final nonzero gradient. The 2,048 outer updates
process 20,971,520 padded source positions in 416.16 seconds at 50,392
positions/s; setup plus training is 428.90 seconds, total wall time is 1,092.60
seconds, and peak CUDA allocation is 1,410,242,048 bytes. Parent checkpoint,
state, tokenizer, and shared context-prefix logits remain exact.

At the final batch, inner reconstruction loss changes 35.7097 to 18.2811 after
step one, then explodes to 5,640.9917 after step two. The positive learned rates
therefore do not form a stable iterative learner. More importantly, untrained
true, learned no-write, shuffled, true, and oracle-short views each score 0/256
strict answers. Their accepted-answer containment counts are 1, 1, 0, 0, and 1;
the isolated hits are not source-conditioned capability. Capability, source-gain,
oracle, and both-inner-step gates fail. No checkpoint is saved.

Decision: `retire_v61_final_residual_nonlinear_fast_learner`. Report SHA-256 is
`12d3cc8b3a1aa14937e68f8323607c9fb1322645b24aec4a3710c8a680b9c358`.
The failed runner and tests are deleted. V60 versus V61 changed the fast learner
from one linear write to two nonlinear MLP updates while holding the late read
interface fixed; both oracle views remained zero. The next falsifier therefore
changes the read locus: a bounded protected memory signal must participate at
multiple frozen V39 depths. It may not widen V61 or reinterpret lower outer loss
as retrieval.

### V62 preregistration: protected three-depth shared memory

V60 and V61 held the read interface at the final hidden state and both failed
even with oracle-short evidence. V57's near-pass, in contrast, let evidence
participate throughout all ten causal blocks but damaged the shared cortex.
V62 isolates that difference while retaining structural protection. It does not
reopen V27's raw cross-attention or V50's rank-16 weight deltas.

The design is informed by [Memory Layers at Scale](https://arxiv.org/abs/2412.09764),
which reports that a shared memory pool used at several spaced Transformer
layers is materially better than a single layer, while adding too many memory
layers eventually degrades performance. The paper studies static trainable
parameter memory at far larger scale, so it is motivation rather than evidence
for MARULHO. [Titans](https://arxiv.org/abs/2501.00663) and
[Learning to Learn at Test Time](https://arxiv.org/abs/2407.04620) separately
support learned fast state as a long-context mechanism, but V60/V61 show their
claims do not transfer automatically to this checkpoint or task.

V62 restores V60's source write without modification. Frozen V39 final source
states are projected into eight normalized width-16 keys; exact frozen next-token
embeddings split into eight width-96 values. Their masked normalized outer
product produces eight temporary 16x96 matrices with learned positive per-head
rates. No question, answer, accepted string, span, metric label, or validation
record enters the write, and each document state is discarded after generation.

The read interface changes. The same matrix is queried immediately before V39
blocks 2, 5, and 8, leaving respectively eight, five, and two frozen blocks to
transform its contribution. Each site has a width-128 query projection, an
eight-value token-dependent sigmoid gate, and a bounded scalar residual gate.
The three sites share one width-768 output projection and one set of source-write
parameters. Added slow state must remain below 1.25% of V39. V39 remains outside
the optimizer, yet answer-loss gradients propagate through its frozen operations
to every earlier memory read. The inactive forward must be tensor- and KV-state-
exact to the ordinary context-96 V39 forward.

The immutable V57 boundary remains 8,192 training cases over 171 titles and 256
validation cases over 22 unseen titles. Five context-64 source chunks and the
parameter-free context-96 question/answer path are unchanged. Eight epochs at
batch 32 give exactly 2,048 AdamW updates and 20,971,520 padded source positions.
Outer optimization uses 3e-4, betas 0.9/0.95, weight decay 0.1, 5% warmup, cosine
decay to 10%, BF16, clip 1.0, data seed 62121, and model seed 62131. Counted
training must stay below 1,800 seconds and setup plus training below 2,400 seconds.

Untrained true memory is diagnostic. Terminal inactive, shuffled-source, true-
source, and oracle-short views share the same 256 questions and V44 decode
policy. Promotion requires true at least 64/256, at least 20 percentage points
above the stronger inactive or shuffled control, shuffled no more than 16,
oracle at least 128, and true no more than 64 cases behind oracle. Exact data and
schedule hashes, all-controller gradient coverage, finite fast state, parameter
budget, CUDA allocation, timing, parent checkpoint/tokenizer/state/common-prefix
logits, inactive forward/KV parity, and strict compact reload on a behavioral
pass are mandatory.

A joint pass advances to routed multi-document fast state and conflict/version
writes. Oracle-only success localizes the remaining problem to full-source
compression or selection. Oracle failure closes this compressed matrix plus
multi-depth read family; adding a fourth site, wider keys, or more epochs is not
an admissible response. The next branch must retain exact source tokens or
change the base language computation.

Implementation preflight passes on the RTX 3060. The controller has 1,001,483
parameters, 0.9947% of V39, and every tensor receives a nonzero gradient on a
real batch-32 forward/backward. The inactive custom forward is bit-exact to V39
for hidden states, logits, and all 21 streaming-state tensors. One warm-up batch
peaks at 1,208,395,264 CUDA bytes. Twenty steady optimizer steps sustain 46,442
source positions/s and project the frozen 2,048-step training phase to about 452
seconds, with 1,214,325,248 bytes peak allocation. These are feasibility facts,
not quality evidence.

### V62 terminal result: retire compressed multi-depth memory

The exact frozen run completes mechanically and fails behavior. The 1,001,483-
parameter controller is 0.9947% of V39; every one of its 13 tensors receives a
final nonzero gradient. All 2,048 updates and 20,971,520 padded source positions
finish in 473.36 seconds at 44,303 positions/s. Setup plus training is 482.80
seconds, total wall time is 1,067.76 seconds, peak CUDA allocation is
1,460,438,528 bytes, and final answer loss is 3.1527. Parent checkpoint,
tokenizer, common-prefix logits, and inactive hidden/logit/all-21-state outputs
remain exact before and after training.

The three BF16 scalar site gates stay at 0.1192, although their gradients and
all higher-dimensional write, query, token-gate, and shared-read gradients are
nonzero. That quantization is a limitation but not the behavioral explanation:
untrained true, inactive, shuffled, true, and oracle-short views score 0, 0, 1,
1, and 1 of 256. Every view contains an accepted answer exactly once. Correct
and wrong sources are therefore indistinguishable, and oracle evidence does not
rescue the interface. Capability, causal source-gain, and oracle gates fail.

Decision: `retire_v62_compressed_three_depth_fast_memory`. Report SHA-256 is
`7742199d52ed13c11cf20816fc4e593500dec7ee99486fd41f44bf416cf5e5b1`.
No checkpoint survives; failed code, tests, and logs are deleted. V60 to V62
tested linear versus nonlinear writes and final versus three-depth reads. The
shared failure is compressed reconstructed memory. The next falsifier must keep
the source's exact token-level KV state available to every frozen attention
layer, or leave the V39 substrate entirely; it may not tune this matrix family.

### V63 exact-token adaptive KV memory: terminal RETIRE

V52 proved that aligned exact causal records produce source-conditioned answers,
but unrestricted shared-weight learning forgot old capabilities. V57 showed the
same broader pattern at context 320: native full context reached 43/256 and
oracle-short reached 122/256, while retention collapsed. V60--V62 protected the
base but compressed evidence and failed even oracle controls. V63 combines only
the parts that survived: exact token-level evidence, native causal attention at
all ten layers, and structural protection of V39.

The direction is consistent with [Memorizing Transformers](https://arxiv.org/abs/2203.08913),
which retains internal key/value representations of past inputs, and with
[LongMem](https://arxiv.org/abs/2306.07174), which freezes a backbone memory
encoder while adapting a decoupled reader. Neither paper validates this small
checkpoint or title-disjoint QA task; MARULHO's controls decide that. Unlike
those systems, V63 performs no approximate retrieval and keeps only the current
document's bounded exact causal state.

For every manifest record, the exact V57 `causal_prompt` or
`oracle_causal_prompt` is tokenized with BOS and the first accepted answer plus
EOS is appended. The tokenizer's exact character offsets must place a clean
boundary after `Context: {source_text}`; the remaining delimiter/question/answer
text must equal the corresponding question-only prompt after removing only its
boundary whitespace. This offset contract is required because causal prompts
intentionally contain a final space after `Answer:` while question-only prompts
do not, so literal suffix-token equality would be false. Inputs are right-padded
only after EOS to context 320; source masks mark only the native `Context:`
prefix, while answer loss marks answer tokens plus EOS. The shuffled control uses
the manifest's immutable mismatched causal prompt with the same question. No
answer, span, accepted string, label, or validation record can alter a source
mask or correction.

V39 is reconstructed at context 320 and strict-loads the exact same tensors.
Each frozen attention layer computes its ordinary 12 heads of width 64. A
MARULHO controller owns one FP32 64x64 key-correction matrix and one FP32 64x64
value-correction matrix per layer and head. Each is passed through an elementwise
tanh, scaled by 0.25/sqrt(64), applied only to source-token keys/values, and added
residually before ordinary causal attention. All matrices initialize exactly
zero, giving 983,040 parameters (0.9764% of V39). Question and answer keys/values,
attention outputs, MLPs, norms, embeddings, and vocabulary head remain frozen.
The active-zero custom forward must be bit-exact to ordinary V39 for hidden
states, logits, and all KV-state tensors; inactive question-only execution calls
the unmodified parent directly.

The immutable V57 8,192/256 title-disjoint split remains frozen. Eight batch-32
epochs give 2,048 AdamW updates and 20,971,520 padded context-320 positions.
Controller parameters and optimizer state remain FP32; V39 activations and
weights remain BF16. AdamW uses 3e-4, betas 0.9/0.95, weight decay 0.1, 5%
warmup, cosine decay to 10%, clip 1.0, data seed 63121, and model seed 63131.
Training must finish below 1,800 seconds and setup plus training below 2,400
seconds on the RTX 3060.

Frozen raw true and raw oracle prefix accuracy are recorded before optimization
but cannot promote. Terminal question-only, shuffled, true, and oracle-short
views share the same 256 questions and V44 generated-only decode policy.
Promotion requires true at least 64/256, at least 20 percentage points above the
stronger question-only or shuffled control, shuffled at most 16, oracle at least
128, and true no more than 64 cases behind oracle. Every correction tensor must
receive a final nonzero gradient. Source/suffix/mask/data/schedule hashes,
finite state, parameter budget, CUDA/time evidence, original parent identity,
active-zero parity, inactive fidelity, and strict compact tensor/logit reload on
a behavioral pass are mandatory.

A joint pass advances exact KV state to selective archival storage, multiple
documents, and version/conflict writes. Oracle-only success isolates full-source
localization and justifies bounded selection over exact tokens. Oracle failure
retires protected memory adaptation around V39; another rank, gate, read site,
or SQuAD replay mix is forbidden. The following architecture experiment must
change the base language computation or learning objective.

The terminal CUDA run is mechanically valid and behaviorally negative. All
25,344 tokenizer-boundary views, zero hidden/logit/all-21-state parity, immutable
V39 checks, finite FP32 controller state, exact schedule/position budget, and
240/240 final correction-matrix gradients pass. The 2,048 updates process
20,971,520 positions in 751.066 seconds at 27,922 positions/s with 4.315 GB peak
allocation; setup plus training is 780.932 seconds. Answer loss falls to 3.2313,
but raw true/oracle both score 0/256 and learned question-only, shuffled, true,
and oracle score 0/0/0/1. Correct exact evidence does not beat wrong or absent
evidence, and even the localized oracle misses its 128/256 floor by 127 cases.

Decision: `retire_v39_protected_memory_adaptation`. No checkpoint survives.
The failed runner and tests are deleted after retaining
`exact-token-kv-v63-20m-20260812.json`, SHA-256
`08baf18c9b203c85fe6a2e8ef1913e31cbf025173be3789e64ef789033cd5e43`.
This closes adapter-level rescue around V39; V64 must replace the base language
computation or its learning objective, not alter memory rank, gates, sites, or
replay proportions.

### V64 delta-state cortex: preregistered base replacement

**Question.** Can a bounded recurrent matrix state integrated during base
pretraining learn source-conditioned language more effectively than a matched
Transformer without giving up general continuation quality or consumer-GPU
feasibility? This is the first post-V63 base replacement. It starts from random
weights, imports no external model, and uses `external_llm_used=false`.

**Null hypothesis.** At roughly 100M parameters and 83.89M matched training
positions, the candidate does not jointly improve source-conditioned exact
generation and preserve heldout general language. A loss-only, recognition-only,
oracle-only, or throughput-only advantage is a failure.

**Candidate.** The MARULHO delta-state cortex has twelve width-640, ten-head
pre-norm blocks and tied 8,192-token embeddings. Blocks repeat three delta-state
mixers followed by one local-attention mixer, giving nine recurrent and three
attention blocks. Each head owns a 64x64 FP32 state. The recurrent update has
independent learned channel decay, key-side erase, and value-side write; query,
key, value, erase, write, decay, output gate, and output projections are trained
with the base. Local attention has a hard 64-token window. Delta training uses
an owned chunk-32 parallel form and generation uses the algebraically identical
recurrent form. The SwiGLU hidden width is 2,624 so the measured candidate count
must fall within 0.99--1.01 times the control's 100,679,424 parameters.

This is not the retired V33/delta-v1 path under a new name. V33 used two serial,
small-state diagonal-affine recurrent blocks and two attention blocks, lost its
early advantage by 16.78M tokens, and ran at roughly one tenth of the control.
V64 uses a current full per-head matrix state, separately controllable
decay/erase/write, chunk-parallel training, three bounded exact-attention
specialists, a 100M-class base, and a mixed real-language/source-use objective.
The mechanism is informed by
[Gated Delta Networks](https://arxiv.org/abs/2412.06464),
[Kimi Linear](https://arxiv.org/abs/2510.26692), and
[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), but MARULHO owns the
implementation, runtime state, weights, generation, checkpoint, and report.
The Flash Linear Attention package may be inspected as reference code but is
not a runtime dependency and supplies no kernel or learned tensor.

**Matched control and schedule.** The control is a fresh V34-shape Transformer:
width 768, ten layers, twelve heads, 100,679,424 parameters. Both arms use the
same existing 8,192-token BPE, context 320, batch 32, BF16 dense projections,
FP32 recurrent algebra where required, fused AdamW, 4e-4 peak learning rate,
5% warmup, cosine decay to 10%, identical seeds-by-role, and the same immutable
batch order. The frozen vocabulary hash is
`faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715`.
There are 8,192 optimizer steps and 83,886,080 padded positions:

- 6,144 unique general batches, 3,072 from each frozen FineWeb-Edu and
  Cosmopedia source, totaling 62,914,560 positions;
- 2,048 document-aligned source-QA batches, eight exact passes over 8,192
  title-disjoint training records, totaling 20,971,520 positions;
- deterministic `general, general, general, QA` interleaving; ordinary causal
  loss on general batches and the retained renormalized four-times answer
  emphasis on QA answer spans while every non-pad token remains trained.

The immutable inputs are the existing `fineweb-edu-replay-75k-shard2` and
`cosmopedia-v2-replay-75k-shard4` corpora with SHA-256
`034a3a00ea86ec097b913f6002485a6081c6adb2b66c14ddc82be7d57b13751c` and
`7b6f41e3b3d2c1871d0124dc19f212713e3c8136e9f66cb462c845354e267aa7`;
their disjoint eval corpora hash to
`a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a`
and `e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491`.
The QA train/validation manifests hash to
`aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133`
and `b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80`.

The validation side is never sampled for training. General evaluation repacks
the existing disjoint FineWeb-Edu and Cosmopedia holdouts at context 320. Source
evaluation uses the frozen 256-record title-disjoint manifest and reports
question-only, shuffled-source, true-source, and oracle-localized generation.
It also records answer containment and token-boundary validity so a tokenizer
or prompt bug cannot masquerade as a mechanism result. Fixed unseen general
prompts are generated before and after training for both arms under one frozen
decode policy.

**Kernel truth before quality.** The sequential FP32 reference is the oracle for
the owned chunkwise implementation. Random and adversarial short sequences must
match forward outputs, final state, every input/parameter gradient, chunk
composition, token-by-token streaming, and state reset. CUDA BF16/FP32 mixed
execution must remain within preregistered numerical tolerances and all trainable
tensors must receive a finite nonzero gradient. Compiled execution is admitted
only after eager/compiled loss, state, and gradient parity; compile time is
reported separately and eager remains valid. A real batch-32/context-320 step
must fit below 11.5 GiB. If candidate preflight throughput is below half the
control, the terminal run stops for kernel redesign without a quality verdict.

The frozen numerical tolerances are `atol=rtol=3e-6` for FP64 chunk/reference
forward, state, and gradients; `atol=rtol=2e-5` for FP32 chunk composition and
streaming logits; compiled/eager BF16 loss absolute delta at most `1e-3`; and
compiled/eager gradients finite for every tensor with global cosine at least
`0.999` and maximum absolute element delta at most `0.01`. The first observed
all-BF16 batch-32 measurement was rejected as a recipe mismatch rather than
promoted. Under the frozen FP32-master/BF16-autocast recipe, eager candidate/
control throughput is 8.09k/24.30k positions/s, while full-graph candidate
compilation takes 587.20 seconds once and then reaches 19.97k positions/s with
0.000150 scalar-loss delta and 7.36 GB peak allocation. This clears only the
50% preflight floor. A backend-matched optimizer-inclusive comparison and
gradient parity still decide terminal admission; the 70% promotion gate is
unchanged.

The Inductor execution branch is terminally rejected. A graph that fused the
answer-weighted cross-entropy hit the 1,204-second per-process ceiling twice; a
model-only retry and a shared compact-WY recurrence retry each hit the same
bound. None produced an atomic artifact, and the repeated compiler load froze
and crashed the Windows host. The generated cache and V64 Inductor backend are
deleted. The exact eager candidate remains only a mathematical oracle: its
8.09k versus 24.30k positions/s is 33.3% of control and fails the frozen 50%
preflight floor. Decision: `stop_v64_for_kernel_redesign_no_quality_verdict`.
This makes no language-quality claim because terminal training never started.
The next admissible V64 execution path is a directly owned CUDA/Triton operator
tested first as a short isolated kernel; another full-model or recurrence
`torch.compile` attempt is forbidden. The compact stop report SHA-256 is
`7060aba8aa591e50ba9cb7673811dd7502b369dd3e4ecddf8307c6a674589be6`.

The first direct-kernel replacement passes its operator and full-model parity
gates. One explicit Triton recurrence owns forward execution. The first
unrestricted inverse backward correctly failed with non-finite early gradients;
an eight-token inverse left four of 2,560 adversarial elements outside tolerance,
and a four-token inverse still rotated stacked-model gradients. None survives.
The admitted backward instead loads an exact state every four tokens and replays
at most three forward updates before applying the reverse equations. A separate
layout audit then caught strided `empty_like` gradient buffers shuffling head
views; explicitly contiguous logical gradients close that integration bug.

At batch 32, ten heads, context 320, and width 64, direct/eager operator forward
is 1.020M/486.1k positions/s (2.10x) and complete forward/backward is
182.2k/121.9k (1.49x). Global gradient cosine is 0.999999991 and maximum element
delta is 3.73e-9; incremental peak is 1.033 GB versus 0.544 GB. Exact stacked-
model tests pass. The real 100,202,970-parameter BF16 model then passes loss and
all-146-gradient parity at physical batches 2, 8, 16, and 24. Batch 16 is
selected at 9.16k positions/s and 7.46 GB versus identical eager 4.75k, with
0.000021 loss delta, 0.999976 gradient cosine, and 0.000061 maximum delta. Batch
24 gains only 0.21% measured speed while adding 3.39 GB; batch 32 reaches the
120-second memory-pressure bound without an artifact and is retired. Effective
batch 32 uses two physical-16 microbatches. A process-tree watchdog now kills
the real Python child on deadline; its deliberate timeout test passes. No
`torch.compile` or Inductor cache participates. The next gate is optimizer-
inclusive CUDA Graph replay against the fresh Transformer control; no language-
quality run has started. Operator/model/sweep report SHA-256 values are
`3f812d7dca0ab6a48aa038f4e3c7f9da350ca1c06669e28a5d7114175ee33f7f`,
`9c38cd4dcd4f03389d07cf22f107958d04a705a7562ca4f7174105605aa5177f`, and
`ae5bed5540703e7e22e1ba9f339bcc49fb7bf6003a233e291717005004d59f0c`.

The optimizer-inclusive CUDA Graph gate closes V64. Capture of the selected
physical-16 by accumulation-2 step takes 0.51 seconds, peaks at 8.70 GB, and
matches eager Triton exactly in scalar loss, clipped gradient norm, update-vector
cosine, and every one of 100,202,970 updated parameters. Three replayed fused-
AdamW steps reach 9.24k positions/s. The fresh 100,679,424-parameter Transformer
at physical batch 32 reaches 21.03k positions/s, so the candidate achieves only
43.93% of control and fails the frozen 50% admission floor. Decision:
`stop_v64_for_kernel_redesign_no_quality_verdict`. This is an execution result,
not a language-quality verdict: the 8,192-step curriculum never starts and no
V64 checkpoint exists. Report SHA-256 is
`c3648f55b49b1bc58b2e677c79059f661414092a2e9bb2a217234536daa5f4c1`.
Another launch-capture layer cannot repair a compute path whose complete graph
is already 2.28 times slower than control; the next architecture must change the
parallel training computation rather than wrap this recurrence again.

### V65 parallel editable-state cortex: preregistration

**Question.** Did V64 fail because finite editable state is a weak language
mechanism, or because its training kernel performed the recurrence token by
token? The null is that a true chunk-parallel algorithm either remains too slow
on the RTX 3060 or reaches speed by changing the recurrence enough to lose exact
state and gradient truth.

The current literature makes this distinction testable. [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
uses the update

`S_t = (I - k_t (b_t * k_t)^T) D_t S_(t-1) + k_t (w_t * v_t)^T`,

where channel-wise decay, erase, and write have separate jobs. This is close to
V64's intended state semantics, but the reported system derives an asymmetric
WY form and gate-aware backward so training is parallel inside chunks. At 1.3B
parameters and 100B FineWeb-Edu tokens it reports stronger aggregate language,
commonsense, and retrieval results than its matched recurrent competitors.
[Mamba-3](https://arxiv.org/abs/2603.15569) is the strongest alternative
state-space control, while [Raven](https://arxiv.org/abs/2607.25357) motivates
sparse slot writes for later interference tests. Neither becomes V65 by name or
dependency. The official Gated DeltaNet-2 code is non-commercially licensed;
MARULHO will not copy it. Published mathematics and independently written
oracles/kernels are the allowed inputs. `external_llm_used` remains false.

**Stage A — kernel admission.** Implement the exact sequential equation in FP64,
then independently derive a chunkwise forward/backward with direct CUDA/Triton
execution. Test chunk sizes 16, 32, and 64; random and adversarial decay/erase/
write regimes; non-contiguous head layouts; final-state continuation; and every
input/state gradient. Global gradient cosine must be at least 0.999 and maximum
absolute gradient delta at most 0.01. No `torch.compile`, Inductor cache, hidden
external package, inverse-state reconstruction, or unbounded process is allowed.
The watchdog and disposable compiler cache are mandatory.

At V64's batch-32, ten-head, context-320, key/value-64 operator shape, complete
forward/backward must exceed 300k positions/s and incremental peak allocation
must stay below 0.80 GB. These are mechanism-admission thresholds chosen to be
materially beyond V64's 182.2k and 1.033 GB, not claims of model speed. A miss
deletes the kernel and stops V65 before model construction.

**Stage B — full-model admission.** If Stage A passes, build a fresh roughly
100M-parameter model with repeated two-state/one-bounded-attention cells. State
training uses only the admitted parallel form; recurrent form is generation and
continuation truth. Compare with the fresh 100,679,424-parameter Transformer at
matched tokenizer, data, optimizer, precision, effective batch, and context.
Sweep context 320, 1,024, and the largest safe longer point so the result shows
both the short-context constant cost and the intended scaling regime. The
candidate must reach at least 50% of control at its actual training context,
retain at least 70% for terminal promotion, remain below 11.5 GiB, and pass exact
optimizer/checkpoint parity before language training.

**Stage C — capability.** Reuse V64's frozen 8,192-step mixed general/source-QA
curriculum only after Stage B. The candidate must remain within 0.02 heldout
general loss of control, reach at least 64/256 exact true-source answers, beat
the Transformer by at least 20 cases and the stronger absent/shuffled control by
at least 51, preserve coherent unseen prose, and exactly reload. A useful base
then earns continual-domain retention and the 524,288-token sustained run.

Cross-layer value routing, sparse Raven-like slots, MoE, reservoirs, wavelets,
and structural growth are excluded from V65's base gate. They become isolated
ablations only after the parallel state path proves both speed and language
value. This prevents an attractive collection of ideas from hiding which
mechanism worked.

**Outcome.** Stage A is a rigorous systems negative. The independently derived
change of coordinates writes `S_t = G_t Z_t`, transforms the key/query/erase
vectors by cumulative channel decay, and solves every corrected write in a chunk
through one causal lower-triangular system. Exact FP64 recurrent/parallel output,
final state, split-stream continuation, and all seven gradients pass for chunks
16/32/64. Owned Triton cumulative-coordinate forward/backward passes the FP32
output/state tolerance of 2e-4 and every-gradient tolerance of 3e-3.

Chunk 64 is decisively best. The PyTorch CUDA reference reaches 240.4k complete
forward/backward positions/s with 460 MB incremental allocation. One-warp Triton
coordinate fusion reaches the overall best 249.7k at 518 MB; fusing the gated
write falls to 215.7k, combined fusion reaches 226.4k, and two/four coordinate
warps reach 229.2k/228.1k. CUDA Graph replay remains 249.7k, showing launch
capture is not the missing speed. The best path is 1.37x V64's 182.2k and halves
its 1.033 GB workspace, but reaches only 83.24% of the frozen 300k Stage-A floor.

Decision: `stop_v65_stage_a_parallel_kernel_misses_throughput`. No 100M model,
checkpoint, or language-quality claim exists. The reference, kernels, evaluator,
tests, and transient reports are deleted. Compact stop report SHA-256 is
`dc141e7d6df1a25f1f238bfe68d37865556ef3c875e3523a84a67a77bddff755`.
The result does not contradict Gated DeltaNet-2's H100/1.3B evidence; it says
this exact matrix-state training shape does not clear MARULHO's consumer-GPU
admission threshold. The next candidate must reduce the state-edit arithmetic
or expose substantially larger dense tiles, not merely refactor the same WY
system or add another launch wrapper.

### V66 causal micro-macro exchange: preregistration

**Bet.** Replace compressed recurrent matrices with compressed *token
representations* and use only dense attention. Split a causal sequence into
64-token neighborhoods. Each neighborhood appends four learned summary tokens
after its real tokens, so a standard causal mask lets summaries read the whole
completed neighborhood while preventing real tokens from reading summaries.
All neighborhoods execute in parallel by folding the neighborhood index into
the batch dimension. The four summaries from every neighborhood then enter a
small global causal mixer. Block `j` receives the completed global output from
block `j-1` as four prefix tokens during the next local stage. Block zero uses a
learned start prefix. Two or more exchanges can repeat across depth.

This is not a novelty claim for hierarchy. [MEGABYTE](https://arxiv.org/abs/2305.07185)
uses local and global decoders across fixed patches; [Block Transformer](https://arxiv.org/abs/2406.02657)
pretrains a global-to-local hierarchy and reports 10--20x inference throughput
with equivalent perplexity/zero-shot behavior; [H-Net](https://arxiv.org/abs/2507.07955)
learns dynamic byte chunks and reports compute/data-matched gains. V66's isolated
question is whether repeated **completed-summary global exchange followed by a
one-block causal shift back into the token stream** gives MARULHO a fast,
source-sensitive macro path on this consumer GPU. The implementation, model
weights, tokenizer, and checkpoint remain MARULHO-owned; no external model or
cognition package participates.

**Stage A — attention-core truth and speed.** Implement one exchange with direct
PyTorch CUDA SDPA, no Inductor. At batch 32, width 640, ten heads, four summaries,
and neighborhood 64, compare forward/backward at context 320 and 1,024 against
ordinary causal SDPA over the same input tensor. Perturb all tokens after a
chosen boundary and require all earlier outputs to remain bitwise equal in FP64
or within 1e-6 in FP32. Perturb a completed earlier block and require a nonzero
change in later-block outputs. Every input and learned-summary gradient must be
present and finite. Context-1,024 candidate throughput must exceed control and
context-320 throughput must remain at least 70% of control; peak allocation must
stay below control at 1,024 and below 2.0 GB incremental at 320. The process-tree
watchdog bounds every CUDA run. A miss deletes Stage-A code before model work.

**Controls.** The causal control is one ordinary full-attention exchange. A
local-only ablation removes global summaries; a shuffled-summary return keeps
compute but sends each block another document's macro prefix; a no-shift invalid
diagnostic must fail anti-leakage and is never trainable. Stage A measures the
operator, not language quality, and cannot promote the architecture alone.

**Stage B — matched base model.** On a Stage-A pass, build an approximately 100M
model with the same embedding, tied vocabulary head, SwiGLU budget, tokenizer,
optimizer, and data as the fresh 100,679,424-parameter Transformer. Match total
parameters by adjusting local/global depth, not by giving V66 an extra decoder.
Train first on a short unique-data loss curve; only a real heldout signal admits
V64's frozen 8,192-step general/source-QA curriculum. Terminal promotion still
requires general loss within 0.02 of control, at least 64/256 true-source exact
answers, +20 cases over Transformer, +51 over absent/shuffled source, coherent
unseen prose, at least 70% control training throughput, strict checkpoint reload,
continual retention, and the 524,288-token sustained run.

Dynamic boundaries, extra summary counts, sparse experts, persistent archives,
and structural growth are later experiments. V66 fixes 64/4 so an attractive
hierarchy cannot hide an unmeasured routing or segmentation effect.

**Outcome.** V66 is mechanically correct but stops at Stage A. FP32 future
perturbation changes earlier outputs by exactly zero, a completed block changes
the following block by up to 0.4835, and all 414,720 audited input/summary/start
gradient elements are finite and nonzero. At context 1,024 the candidate reaches
1.678M positions/s versus 1.574M for full attention (1.066x), proving the
compressed global path has a real crossover. It nevertheless uses 694 MB versus
505 MB peak allocation. At context 320 it reaches 1.516M versus 2.729M
positions/s (0.555x), failing the frozen 0.70 floor. No language model or
checkpoint is built. The implementation, evaluator, partial artifacts, and
tests are deleted; `micro-macro-v66-stage-a-stop-20260813.json` owns the durable
evidence (SHA-256
`ade21fd08089e576d2369f2a187e1081cc4ab6c3e3885974365f0375091843fd`).

### V67 queried-summary exchange: preregistration

**Isolated variable.** Replace V66's first length-68 local self-attention pass
with four learned queries cross-attending to the 64 raw hidden vectors of each
completed block. Fold blocks into the batch exactly as in V66. Flatten the four
outputs per block into the same causal global summary stream. Shift block
`j-1`'s completed global outputs into block `j` as four prefix vectors; block
zero receives a learned start prefix. Run exactly one length-68 local causal
self-attention pass over prefix plus raw block and return its 64 real outputs.
There are no summary projections, gates, convolutions, recurrence, dynamic
boundaries, or additional local pass. Neighborhood size remains 64 and summary
count remains four.

**Stage A frozen protocol.** Use direct PyTorch CUDA SDPA without Inductor on
the same RTX 3060. At batch 32, width 640, ten heads, BF16, and contexts 320 and
1,024, run two warmup steps and time five complete forward/backward steps. The
candidate loss is mean-squared local output plus mean-squared global-summary
output; the ordinary full causal-attention control uses mean-squared output.
Report median positions/s, all samples, absolute and incremental peak CUDA
allocation, device/software identity, and whether every input/query/start
gradient is present, finite, and nonzero. A separate FP32 context-320 truth arm
must keep outputs before a future perturbation boundary within 1e-6 and must
show a nonzero change in the immediately following block when a completed block
is perturbed.

The unchanged admission gates are candidate/control throughput greater than
1.00 at context 1,024, at least 0.70 at context 320, candidate peak allocation
below control at 1,024, and candidate incremental allocation below 2.0 GB at
320. Every CUDA process uses the process-tree watchdog. This exact implementation
gets one measurement; no query count, block size, precision, loss, or attention
layout changes are allowed after observing it. Any miss records a compact report
and deletes all V67 code before model work.

**Stage B only after a complete pass.** Build an approximately 100M owned model
against the fresh 100,679,424-parameter Transformer with the same tokenizer,
embedding/head, SwiGLU budget, Muon/AdamW optimizer, data, and causal-language
protocol. Adjust the number of local/global layers to reach parameter ratio
0.99--1.01. A short unique-data curve must show a real heldout signal before
the frozen 8,192-step general/source-QA curriculum. Local-only, shuffled-summary
return, and question/source controls remain mandatory mechanism falsifiers.

**Outcome.** V67 passes causal and compute gates but stops on memory before
model construction. FP32 future perturbation changes earlier output by exactly
zero; perturbing a completed block changes the following block by up to 2.2634;
all 414,720 audited gradient elements are finite and nonzero. At context 320 it
reaches 1.729M positions/s versus 2.339M (0.739x), clearing the 0.70 floor. At
context 1,024 it reaches 2.260M versus 1.652M (1.368x), but peaks at 646 MB
versus 505 MB (1.279x). The explicit head-major to folded-block transforms and
prefix/output materialization are the leading allocation hypothesis, not a
validated cause. V67's code, runner, partials, and tests are deleted; the compact
report `queried-summary-v67-stage-a-stop-20260813.json` owns the evidence
(SHA-256
`7cfcfdade10aa7789191dc1b54f82bf9ff7cba6958aa0775d8f309d7bdbb45a9`).

V68 may isolate that hypothesis by retaining block as a native leading SDPA
dimension and otherwise preserving V67's equations, seeds, shapes, losses,
warmup/timing counts, hardware, controls, and gates. It must be preregistered
separately before implementation.

### V68 block-native queried exchange: preregistration

**Isolated variable.** Preserve V67's exact queried-summary/global-causal/
shifted-prefix equations, four summaries, 64-token blocks, losses, and precision.
Change only the post-projection candidate layout from `[B,H,T,D]` to contiguous
`[B,block,H,token,D]`. Folding `B*block` for the 4-D Flash SDPA calls must be a
storage-aliasing reshape, not a permute-copy. Return local outputs in the same
block-native layout; do not materialize a time-major candidate output. Only the
small summary stream may permute/copy between block-major and head-major forms.
The ordinary control remains natively contiguous `[B,H,T,D]`. Both contain the
same number, dtype, and statistical distribution of post-projection elements.

This is a core-layout falsifier, not permission to omit real model costs. A
Stage-A pass says only that the hierarchy's attention core can be represented
efficiently. Any later approximately 100M model comparison must include its
embedding, QKV/output projections, all layout operations, MLP, optimizer, and
full forward/backward step against the Transformer.

**Frozen measurement.** Reuse V67's CUDA truth arm, batch 32, width 640, ten
heads, BF16 timing, contexts 320/1,024, two warmups, five measured steps, squared
local-plus-summary candidate loss, squared control loss, and process-tree
watchdog. Add an explicit storage-pointer assertion that candidate block folding
aliases the native input. Admission still requires zero/1e-6 future leakage,
nonzero completed-block influence, complete finite nonzero gradients, context-
320 throughput at least 0.70x control with less than 2.0 GB incremental
allocation, and context-1,024 throughput above control with peak allocation
strictly below control. No kernel, checkpointing, block size, summary count,
precision, or custom backward sweep is allowed. A miss deletes V68; a complete
pass admits the parameter-matched Stage-B model defined by V67.

**Outcome.** V68 stops before model construction. Block folding aliases input
storage, future perturbation changes earlier output by exactly zero, completed-
block perturbation changes the following block by up to 2.6794, and all 414,720
audited gradient elements are finite and nonzero. At context 1,024, native
layout reaches 2.451M positions/s versus 1.604M (1.528x) and removes 84,541,440
peak bytes relative to V67, validating real copy overhead. It still peaks at
561 MB versus 505 MB (1.112x). At context 320 it reaches 2.029M versus 2.936M,
or 0.691x against the 0.70 floor. The exact frozen result is not rerun or tuned.
The remaining prefix concatenation is now the leading allocation hypothesis,
not a validated cause. V68 code, runner, partials, and tests are deleted; the
compact report `block-native-v68-stage-a-stop-20260813.json` owns the evidence
(SHA-256
`14cf9cc1dca10d2e049bb3f7d1ff3121d26df5434d8fc7b49ee652b8c72a85c5`).

### V69 macro-conditioned full attention block: preregistration

**Bet.** Preserve the causal completed-block hierarchy while deleting macro
prefix tokens entirely. Both candidate and control own separate width-640 query,
key/value, and output projections initialized bit-identically. Candidate hidden
is a storage-aliasing `[B,block,64,width]` view of the same logical input. Its
projected K/V values serve both four learned per-head summary queries and local
attention. The summaries mix through the same short causal global stream and
are averaged within each completed block. Block `j-1`'s macro vector is shifted
to block `j`; block zero uses one learned per-head start vector.

Two learned per-head elementwise scales inject that shifted macro vector without
creating tokens: one adds it in place to local queries before length-64 causal
attention, and one is passed through the shared output weight then added in
place to the projected block output. The latter is algebraically identical to
adding the macro before the bias-free output projection, but does not mutate the
SDPA output saved for backward. Query conditioning can select local information
while output conditioning carries macro content itself. No prefix concatenation,
macro projection matrix, recurrence, dynamic boundary, MLP, normalization,
checkpointing, custom kernel, or Inductor is present. This is one complete
attention sublayer, not a raw post-projection operator.

**Stage A frozen protocol.** At FP32 batch 2/context 320, require candidate
outputs before a future-block perturbation to remain within 1e-6, require a
completed block to change the immediately following block, require the folded
candidate input to alias storage, and require finite nonzero gradients for
hidden, Q/KV/output weights, summary queries, start vector, and both macro
scales. At BF16 batch 32, width 640, ten heads, contexts 320/1,024, run two
warmups and five measured forward/backward steps. Candidate loss is squared
local output plus squared global summaries; control loss is squared output.
Record all timing samples, medians, absolute/incremental peak CUDA allocation,
parameter counts, exact common-parameter initialization hashes, and hardware.
Every arm uses the process-tree watchdog.

Admission remains deliberately unchanged: candidate throughput at least 0.70x
control and incremental allocation below 2.0 GB at context 320; candidate
throughput greater than control and absolute peak allocation strictly below
control at context 1,024. The full-block comparison supersedes raw-core speed
interpretation but does not relax any threshold. One exact measurement is
allowed. A miss records compact evidence and deletes V69. A complete pass admits
the approximately 100M parameter-matched Stage-B language comparison and all
behavioral controls preregistered under V67.

**Outcome.** V69 stops under its frozen memory gate and receives no quality
verdict. Candidate/control common projection hashes match exactly. FP32 future
perturbation changes earlier output by zero, completed-block perturbation changes
the next block by up to 8.0925, and all 2,052,480 audited input/parameter gradient
elements are finite and nonzero. At context 320, candidate/control throughput is
1.075M/1.174M positions/s (0.915x). At 1,024 it is 1.198M/0.941M (1.273x), while
peak allocation is 703/652 MB (1.078x). Both speed gates pass; below-control
long-context memory does not. V69 is not retroactively promoted. Its code,
runner, partials, and tests are deleted; compact report
`macro-conditioned-v69-stage-a-stop-20260813.json` owns the evidence (SHA-256
`96bfc6fd4f6b7e3b23af815dd6813fc4938bb437d7a46bf107dcb38e8c40d8aa`).

Across V66--V69, FlashAttention's linear activation storage changes the systems
interpretation: an additional learned macro path can reduce attention compute
yet retain slightly more total state than a control with no macro pathway. The
next experiment may transparently ask whether that state earns its cost at the
full-model boundary, using the existing at-least-70% end-to-end throughput and
at-most-11.5-GiB hardware gates. It must receive a new version and preregistration;
V69's failed result remains immutable.

### V70 parameter-matched macro cortex: preregistration

**Question.** Does V69's completed-block macro pathway improve language learning
enough to earn its measured 7.81% full-block memory overhead? V70 reuses the
mechanism under a new version; it does not reclassify V69. Candidate and control
are fresh approximately 100M decoder-only models with vocabulary 8,192, width
768, ten layers, twelve 64-wide heads, SwiGLU ratio 4, RMSNorm, tied embeddings,
and context 320. Every unchanged embedding, norm, MLP, QKV, output, and head
tensor is initialized exactly equal and hash-audited. Candidate adds four
summary queries, one start macro, and query/output elementwise macro scales per
layer; expected parameter ratio must remain 0.99--1.01.

**Candidate layer.** Each layer applies RMSNorm, combined QKV projection, local
RoPE, four queried summaries per completed 64-token block, causal RoPE over the
short global summary stream, a one-block shift, macro-conditioned local query
and output, residual, then the same normalized SwiGLU residual as control.
Candidate input stays `[B,block,64,width]` within each attention operation and
returns logical `[B,time,width]` for the residual/MLP. A block never receives
its own summary. No recurrence, prefix token, dynamic boundary, expert, sparse
kernel, Inductor, compiled optimizer, external weight, or pretrained model
participates.

**Phase 0 — full-step admission.** Transfer exact common tensors and verify
their named hashes. On the RTX 3060, BF16, batch 32, context 320, and whole-
matrix MARULHO Muon at learning rate 3e-4 with
`compile_orthogonalizer=False`, run exact candidate/control warmup
and at least three complete forward, cross-entropy backward, gradient clipping,
and optimizer steps inside the process-tree watchdog. Both must keep all
gradients finite and nonzero, candidate peak allocation must remain below 11.5
GiB, and candidate median positions/s must be at least 70% of control. Also
require FP64/FP32 future-block isolation, completed-block influence, and exact
inactive common initialization. A miss deletes V70 before quality training.

**Phase 1 — frozen short quality screen.** From exact fresh resets, use one
tokenizer, immutable unique-data general-language schedule, heldout split, Muon
recipe, seed, BF16 policy, and evaluation code. Each arm receives exactly 512
batch-32 context-320 optimizer updates, or 5,242,880 target positions, with no
relation/source-QA records and no repeated selected window. Persist schedule,
tokenizer, split, initial-state, and final-state hashes plus per-arm partial
rows atomically. Learning rate warms linearly for 26 steps from 0.1x to 3e-4,
then cosines to 3e-5; heldout loss is recorded before training and after steps
128, 256, and 512. Candidate must finish at least 0.02 heldout loss below control,
retain at least 70% measured end-to-end throughput, remain below 11.5 GiB, and
keep complete finite gradients. A tie or loss deletes the model/runner/tests and
records `retire_v70_no_short_language_signal`; it cannot be rescued by more
tokens, another learning rate, or summary/block-size tuning.

**Phase 2 only after a Phase-1 win.** Add strict training-only checkpoint reload,
owned incremental generation state, local-only and shuffled-macro controls, then
run the frozen 8,192-step general/source-QA curriculum. Terminal promotion still
requires general loss within 0.02 of Transformer, true-source exact at least
64/256 and +20 over Transformer and +51 over absent/shuffled, oracle at least
128/256, coherent unseen prose, at least 70% control throughput, less than 11.5
GiB, exact reload, continual retention, and the 524,288-token sustained run.
Failure deletes the candidate surface; success may replace the installed base
only after every gate, never from the short loss screen alone.

**Phase-0 outcome.** V70 advances to the frozen 512-update quality screen.
Candidate/control parameters are 100,733,184/100,679,424 (ratio 1.000534) and
their 106,970,880 common named state elements hash identically. Complete eager-
Muon forward/cross-entropy/backward/clip/step throughput is 18.57k/19.41k
positions/s, or 95.67% retention. Peak allocation is 5.53/5.48 GB; the 46.9 MB
candidate overhead is only 0.86%, and both are far below 11.5 GiB. Every tensor
gradient is present, finite, and nonzero. No compilation occurred, both bounded
processes exited normally, and Phase 0 contains no language-quality evidence.
Compact report `macro-cortex-v70-phase0-pass-20260813.json` owns the exact
timings and hashes (SHA-256
`48747a3e2df299560f755d82223d9f222f9427dc0a48bb9632f3e271482ab7b3`).

**Phase-1 outcome.** V70 terminates with `retire_v70_no_short_language_signal`.
Candidate/control start at identical 9.1875 heldout loss. At steps 128/256/512,
candidate loss is 6.9063/6.0000/5.7188 versus control's 6.5000/5.7500/5.5000.
The required candidate gain was +0.02; actual control-minus-candidate loss is
-0.21875. This is not explained by systems failure: candidate retains 94.61%
end-to-end throughput (17.28k versus 18.27k positions/s), uses 5.14 versus 5.07
GiB peak allocation, and both arms observe every parameter gradient. Tokenizer,
schedule, common-initialization, source-selection, and unique-window contracts
match exactly; relation updates and repeated selected windows are zero.

The conclusion is narrow: ten layers of 64-token local attention with a four-
summary completed-block channel lose too much detailed cross-block information
for short general-language learning. It does not reject hierarchy, summaries,
or macro conditioning in architectures that periodically restore exact token-
level communication. No V70 checkpoint or generation implementation is built.
The model, preflight, quality runner, tests, and partial arms are deleted; the
compact quality report `macro-cortex-v70-quality-stop-20260813.json` owns the
terminal evidence (SHA-256
`5ba460b305c7a2f661e6bbf570efe6a55a217eaacac0a9c2957641544bda1e6b`).

### V71 periodic global-reset hierarchy: preregistration

**Hypothesis.** V70 fails because four summaries are the only cross-block path
at every depth. V71 keeps width 768, ten layers, twelve heads, context 320,
64-token blocks, SwiGLU, RMSNorm, tied 8,192 vocabulary, and all training
contracts, but freezes the layer topology to `L,L,L,L,G,L,L,L,L,G`. `L` is
V70's local attention; in the macro arm it also extracts four summaries, mixes
them causally, and shifts the completed previous-block macro into local query
and output. `G` is the exact ordinary full-token Transformer attention layer.
Thus no information must survive summary compression for more than four layers.

**Controls.** `periodic_macro` and `periodic_local` start from identical common
embedding, norm, QKV/output, MLP, and head tensors. `periodic_local` runs the
same frozen L/L/L/L/G topology but contains no summary/macro parameters or
computation. It tests whether periodic locality alone explains a gain. The
ordinary ten-global-layer Transformer control is the immutable V70 control row
from `macro-cortex-v70-quality-stop-20260813.json`; reuse is valid only if V71
reproduces its 100,679,424 parameters, common-state hash
`700f403ac0405b11cc25262f87434b9a00174d4ed10bc46198e778b7ad84127a`,
tokenizer hash `faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715`,
schedule hash `8342013bb10d842f136c28338664e24db3132c13f5f160ea1eb94065b99daa07`,
initial loss 9.1875, and the exact optimizer/data/evaluation recipe. Any mismatch
requires rerunning the Transformer; it cannot be ignored.

**Phase 0.** FP64/FP32 tests require future-block isolation before every global
reset, completed-block influence in macro L layers, exact common initialization,
macro-off parameter absence, output shapes, and finite nonzero gradients. A
bounded BF16 batch-32 context-320 100M eager-Muon preflight measures at least
three complete steps for both V71 arms. Parameter ratios must remain 0.99--1.01,
peak allocation below 11.5 GiB, and each arm at least 70% of V70's immutable
18.267k Transformer positions/s. A miss deletes V71 before quality training.

**Phase 1.** From fresh resets, both V71 arms consume the exact immutable V70
512-entry indexed-host schedule: 5,242,880 unique context-320 target positions,
zero relation records, no repeated selected window, batch 32, BF16, eager whole-
matrix Muon, 26-step linear warmup to 3e-4 then cosine to 3e-5, clip 1.0, and
heldout evaluation at 0/128/256/512. Atomic arm rows persist state/split/source/
schedule/tokenizer hashes, gradients, throughput, and memory. `periodic_macro`
must finish at least 0.02 loss below both the 5.5000 Transformer and
`periodic_local`, retain at least 70% Transformer throughput, and stay below
11.5 GiB. If local wins but macro does not, retire the macro channel and retain
periodic exact reset as a separate future hypothesis. If neither beats the
Transformer, retire this topology. No learning-rate, layer-position, summary,
block-size, seed, or budget sweep follows the result.

**Phase 2 only after joint win.** The same checkpoint/generation, shuffled-
macro, source-QA, general retention, unseen prose, continual learning, and
524,288-token sustained gates defined under V70/V67 apply. Phase 1 alone cannot
promote runtime.

**Phase-0 outcome.** Both V71 arms advance. `periodic_macro` has 100,722,432
parameters, reaches 18.24k complete-step positions/s (99.86% of immutable
Transformer), and peaks at 5.52 GB. `periodic_local` exactly matches the
100,679,424 control parameter count, reaches 19.32k positions/s (105.76%), and
peaks at 5.48 GB. Both reproduce common hash
`700f403ac0405b11cc25262f87434b9a00174d4ed10bc46198e778b7ad84127a`,
stay below 11.5 GiB, and have complete finite nonzero gradients. The macro
channel costs 5.58% throughput relative to periodic-local before quality is
known. Phase 0 admits only the two frozen 512-update arms. Compact report
`periodic-v71-phase0-pass-20260813.json` owns the timings and gates (SHA-256
`3f8b2b27f3d5a8f66200b844089f192a7afde6d33de6daa3899b479a49557142`).

**Phase-1 outcome.** V71 terminates with
`retire_v71_macro_and_periodic_local_no_language_win`. All arms start at 9.1875
loss. At steps 128/256/512, macro reaches 6.9063/6.0000/5.6875, local reaches
6.6563/5.8438/5.5625, and immutable Transformer is 6.5000/5.7500/5.5000.
Exact resets recover most of V70's deficit but do not win. Macro worsens its
matched topology by 0.125 and runs at 93.83% of local throughput. Both arms
retain at least 95% of Transformer throughput, stay near 5.1 GiB, and observe
every gradient; all hash, data, and uniqueness contracts match.

This closes V66--V71's four-summary completed-block channel as a base-language
mechanism. It does not reject all hierarchy or latent state. A successor must
preserve intact token communication and test whether persistent internal state
adds information under zero, shuffled, and same-compute controls. V71 code and
partials are deleted; compact report `periodic-v71-quality-stop-20260813.json`
owns the evidence (SHA-256
`776d710784f1fc128005cf36c675eade8ec008b7dfac52ea5094aac1b3aa3e0e`).

### V72 persistent cross-segment workspace: preregistration

**Why this is not V4/V5 again.** V4 passed a transient mean among depth-aligned
modules and V5 used a within-window content-addressed workspace; neither owned
durable document state, both reduced exact-stream capacity, and V5 lost to
shuffled/no-exchange. V70/V71 additionally show that compressed summaries must
not replace token communication. V72 therefore leaves a complete ten-layer
width-768 Transformer intact and tests only whether eight persistent latent
tokens add useful information beyond its 320-token window.

**Mechanism.** Documents are split into ordered nonoverlapping 320-token
segments. Each layer still performs ordinary full-token causal attention and
SwiGLU. After layers 3 and 7, segment tokens read the *previous completed*
eight-token workspace through residual cross-attention. After the final layer,
eight learned write queries cross-attend to the completed segment and update the
workspace with a gated residual plus RMS normalization. The updated state is
detached between segments in Stage A, bounding training memory and isolating
state content rather than long backpropagation. It resets at document boundaries
and is never shared across documents. No source labels, future segment, target
tokens, retrieval index, external model, summary replacement, recurrence over
individual tokens, Inductor, or custom kernel participates.

**Stage A1 — mechanism truth.** Use a small owned model on delayed associative
recall where key/value evidence appears in segment zero, distractors occupy at
least one full segment, and the query appears in the final segment. Compare
`persistent`, `reset_each_segment`, `shuffled_document_state`, and
`nonpersistent_same_compute`; all have identical parameters and execute the
same reads/writes, while controls change only state identity/lifetime. Require
future-segment perturbation not to change earlier outputs, exact state reset,
finite nonzero gradients, persistent accuracy at least 80%, at least +20 points
over every control, and three fixed seeds. Failure deletes V72 before real data.

The frozen A1 recipe uses seeds `7201`, `7202`, and `7203`; batches of 128
synthetic documents; three nonoverlapping 64-token segments; 16 possible keys,
16 possible values, and four independently sampled key/value bindings per
document. Segment zero exposes the four bindings in shuffled pair order, one
entire middle segment contains only distractors, and the final segment exposes
only a query key. The owned width-64, two-head, two-token-block model carries
eight width-64 workspace tokens. Every arm uses the same learned initial state,
two token-to-workspace reads, one eight-query workspace write, local
key/value reconstruction of segment-zero writes, and a visible-input write-gate
target; no query answer supervises an earlier segment. State is detached at
each boundary. `reset_each_segment` restores the learned initial bank,
`shuffled_document_state` rotates state to another batch member, and
`nonpersistent_same_compute` replaces document state with the batch-mean bank;
all still execute the write/read tensors. Train 600 AdamW updates at `3e-4`
with no weight decay, gradient norm 1.0, and evaluate 4,096 fresh documents per
seed. The frozen loss is final-query value cross-entropy plus `0.5` times each
of the segment-zero workspace key and value reconstruction losses plus `0.1`
times visible write-marker binary cross-entropy. Accuracy is the only
behavioral selection metric. No width, step, learning-rate, auxiliary weight,
or dataset sweep is allowed after observing an arm result.

**Stage-A1 result: PASS.** All three frozen persistent arms reach 4,096/4,096
correct (100%). The largest control accuracy is 7.1533%; the smallest is 6.2256%,
around the 6.25% chance level. The minimum persistent margin is therefore
92.8467 percentage points. Initial parameter hashes and complete synthetic
schedules match within every seed, future perturbation and reset errors are
exactly zero, every parameter receives a finite nonzero gradient, and maximum
peak CUDA allocation is 333.51 MiB. The run uses eager PyTorch without
compilation and completes safely in 958.1 seconds. Decision:
`advance_v72_to_sequential_real_language`. This validates durable state content,
not general language quality or runtime promotion. Compact report
`persistent-workspace-v72-stage-a1-20260813.json` owns the evidence (SHA-256
`bfbd03cdad6079fe8952596f194431c93384b818a57ce4949dbed3e997357a40`).

**Stage A2 — sequential real-language admission.** Materialize disjoint long
FineWeb-Edu and Cosmopedia documents with at least three complete segments; do
not stream-pack across documents. Use the same tokenizer and match candidate/
Transformer parameters within 0.99--1.01 by reducing candidate MLP width, not
removing attention depth. Train fresh `persistent`, `reset`, `shuffled`, and
Transformer arms on identical ordered document/segment schedules. Loss on the
first segment is diagnostic; the frozen promotion metric is loss on segments
two and later, where durable state can matter. Persistent must beat Transformer
and every workspace control by at least 0.02, retain at least 70% throughput,
stay below 11.5 GiB, and show a positive state-swap counterfactual: replacing a
document's state with another document's must worsen its next-segment loss by at
least 0.02. Any miss retires the mechanism; no scale or source-QA run follows.

The frozen A2 recipe uses the qualified MARULHO 8,192-token BPE vocabulary and
the already materialized, mutually disjoint FineWeb-Edu/Cosmopedia replay and
evaluation shards. Select the first 4,096 eligible training documents from each
replay shard and the first 512 eligible heldout documents from each evaluation
shard in file order; eligibility is at least 961 encoded tokens including BOS.
Keep exactly the first 961 tokens and form three ordered 320-input/320-target
segments at token offsets 0, 320, and 640. Training seed `72121` shuffles the
8,192-document schedule once; model seed `72131` initializes every arm. Use 256
unique-document optimizer updates at batch 32 (7,864,320 input positions per
arm), no relation data, and evaluate the fixed 1,024 heldout documents. No
document repeats and no cross-document packing are allowed.

The Transformer fixes width 768, ten layers, twelve heads, context 320, and
SwiGLU hidden width 3,072. Workspace arms retain all ten full-attention layers,
use SwiGLU hidden width 2,768 to pay for two width-768 cross-attention reads and
one eight-query write, and must land within parameter ratio 0.99--1.01. Reads
occur after layers 3 and 7. Eight width-768 write queries attend to the completed
segment, then a learned scalar content gate, residual update, and RMS norm form
the next detached state. To train the writer without cross-boundary gradients,
each workspace slot predicts the already-observed token ending its fixed
40-token segment partition (positions 39, 79, ..., 319) through the tied
vocabulary head. Its local reconstruction loss weight is 0.1; every workspace
control uses the identical objective and modules. The Transformer uses ordinary
next-token loss only. Therefore persistent-versus-reset/shuffled isolates state
lifetime, while persistent-versus-Transformer remains the joint system test.

Use owned uncompiled Muon/AdamW, weight decay 0.1, gradient norm 1.0, thirteen
linear warmup updates from `3e-5` to `3e-4`, then cosine decay to `3e-5` at
update 256. Backpropagate each segment's one-third-scaled loss before advancing
to the next segment, update once per three-segment document batch, and detach
state exactly at boundaries. Frozen arms are `transformer`, `persistent`,
`reset`, and `shuffled`; they execute in separate bounded processes. Report
first-segment and combined later-segment heldout losses, corpus-specific later
loss, state-swap next-segment delta, complete gradients, exact data/model hashes,
training positions/s, and peak CUDA allocation. No optimizer, width, auxiliary,
schedule, data-selection, or state rule changes are allowed after an arm result.

**A2 full-step preflight: PASS.** On the RTX 3060, the
100,769,281-parameter workspace candidate is 1.000893 times the
100,679,424-parameter Transformer. Both complete a real batch-32, three-segment
Muon/AdamW update with finite nonzero gradients for every parameter. Candidate/
control peak CUDA allocation is 5.691/5.650 GB and one-step throughput is
17.55k/16.43k positions/s (1.068 ratio). The exact long-document contract hash
is `eb56d6828e9a89ec7a0a7092663694e5c27c4c1d29dc1104b15ad29d10739d27`;
the tokenizer hash remains
`faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715`.
No compilation is used. This admits the frozen quality arms and makes no
language-quality claim. Compact report
`persistent-workspace-v72-a2-preflight-20260813.json` owns the evidence
(SHA-256 `aeccb9a6ce0f4eae2c40e7fa63633fc567d8c5b589ce9d1d8152b7f18299df40`).

**A2 terminal result: STOP.** Persistent state is mechanically real but loses
the joint language gate. After the frozen 256 unique-document updates,
Transformer/persistent later-segment heldout loss is 5.85117/5.95039. The
candidate is worse by 0.09922 overall, 0.09141 on FineWeb-Edu, and 0.10859 on
Cosmopedia. First-segment loss is also worse by 0.09297, evidence that the
slightly narrowed SwiGLU path and auxiliary write objective reduce ordinary
language learning before memory could help. This is not dead state: replacing a
document's workspace with another document's worsens later loss by 0.03867,
above the causal 0.02 gate. Candidate throughput is 21.45k versus 22.20k
positions/s (96.61%), peak allocation is 5.89 GB, parameter ratio is 1.000893,
and every gradient is finite and nonzero. Useful state is therefore insufficient
for a better complete model.

The Transformer comparison is already terminally failed, so the reset and
shuffled training arms cannot rescue the joint gate and are not run. This is a
preregistered early stop, not a positive control claim. Decision:
`retire_v72_persistent_workspace_real_language_failure`. All V72 implementation,
runners, and tests are deleted; the compact A1, preflight, Transformer,
persistent, and early-stop reports retain the evidence. The terminal report is
`persistent-workspace-v72-a2-early-stop-20260813.json` (SHA-256
`c81b525ec612886785302f0d3c7e318e113eb3ec47fd5685ff827279209c5db8`). V72 rules out this
compressed detached-workspace interface, not every form of long context.

### Post-V72 synthesis: keep the Transformer, adapt around it

The user's correction is adopted: MARULHO should stop repeatedly replacing the
only local language cortex that survives matched scaling. The original
[TTT layers](https://arxiv.org/abs/2407.04620) make their hidden state a linear
model or MLP and update it with self-supervised gradient descent; they were
evaluated as sequence layers from 125M to 1.3B parameters. The later
[end-to-end TTT](https://arxiv.org/abs/2512.23675) is conceptually closer to
MARULHO's target: a standard sliding-window Transformer continues next-token
learning on test context, and training meta-learns an initialization for those
future updates. Its reported 3B-parameter, 164B-token regime cannot be treated
as locally validated on one RTX 3060. [Titans](https://arxiv.org/abs/2501.00663)
similarly separates precise short-term attention from a neural memory updated
at test time, but V72 shows that real state utility does not automatically make
a better complete model.

MARULHO has already tested non-equivalent shortcuts. V59's ordinary source-time
full-model update memorized source loss but produced 0/64 answers. V60--V63
meta-trained linear, nonlinear, multi-depth, and exact-token protected adapters
around frozen V39; none produced usable source-conditioned language. Those
failures do not reproduce end-to-end TTT, because the base Transformer was not
jointly trained from scratch to make its own future update useful. They do show
that another post-hoc frozen-model adapter is not an admissible novelty claim.

### V73 preregistration: exact-cortex adaptive sidecar

V72 isolates a sharper variable before paying second-order meta-gradient cost:
its state-swap delta is +0.03867, but the candidate loses 0.09922 overall and is
already 0.09297 worse on the first segment, where no document history exists.
The plausible cause is not absent memory; it is paying for memory by reducing
every Transformer SwiGLU from 3,072 to 2,768 and allowing the auxiliary write
objective to alter the shared cortex. V73 removes exactly those two costs. This
is a new isolated test, not a V72 width sweep.

The full 100,679,424-parameter width-768, ten-layer, twelve-head Transformer is
kept unchanged. After layers 3 and 7, a shared four-head width-256 sidecar reads
eight width-256 document-state tokens and adds a residual controlled by one
zero-initialized scalar per read site. The sidecar has its own token-query,
state-key/value, and width-768 output projections. After the final Transformer
layer, eight learned width-256 queries attend to a **detached** view of the
completed segment through separate width-256 key/value projections. A learned
content gate, residual update, and RMS normalization form the next state. The
writer reconstructs the eight already-observed 40-token landmarks through its
own width-256-to-768 projection and the unchanged tied vocabulary head at weight
0.1. Detaching its source hidden states prevents this local objective from
changing the Transformer. State is detached between segments. No target from a
future segment, question, answer, retrieval index, external model, compilation,
or custom kernel participates.

V73 reuses V72's immutable A2 tokenizer, first-eligible 8,192 train and 1,024
heldout documents, hashes, seed `72121` schedule, seed `72131` base initialization,
batch 32, three 320-token segments, 256 unique-document Muon/AdamW updates, and
7,864,320 positions. The retained V72 Transformer arm at later loss 5.851171875,
22,203.68 positions/s, and 5,852,532,736 peak bytes is the exact baseline; it is
not rerun. Fresh `persistent` and, only after admission, `reset` and `shuffled`
arms use identical sidecar parameters, operations, objective, and initialization.
Reset restores the learned initial state at every boundary; shuffled rotates
the completed state across documents. No static adapter or extra parameter can
explain persistent-versus-reset.

Before quality, V73 must prove: disabled read gates reproduce the complete
Transformer hidden states and logits bit-exactly; future-segment perturbations
cannot change earlier outputs/state; reset is exact; candidate parameters are
at most 1.02 times the Transformer; every parameter receives a finite nonzero
gradient after a real optimizer step; throughput is at least 70% of the retained
control; and peak allocation is below 11.5 GiB. Persistent then runs first. It
must reach later-segment loss at most 5.831171875, state swap must worsen loss by
at least 0.02, and both corpus-specific later losses must improve over V72's
Transformer values. Any miss is terminal and deletes V73 without running controls
that cannot rescue it. If admitted, reset and shuffled run; persistent must beat
each by at least 0.02 under the same joint gates. A full pass advances only to
checkpoint/generation and continual validation. It does not install a runtime.

**V73 preflight: PASS.** The candidate has 101,932,035 parameters, 1.012442
times the exact 100,679,424-parameter Transformer. Disabled logits are bit-exact
with maximum error zero and the copied Transformer hash equals V72's retained
initial hash. After two real three-segment optimizer updates, all base and all 18
sidecar tensors have received finite nonzero gradients. The second update reaches
20.63k positions/s versus the retained 16.43k one-step Transformer preflight;
two-update aggregate is 17.25k. Peak CUDA allocation is 6.068 GB, and no
compilation occurs. The persistent quality arm is admitted. Compact report
`exact-cortex-sidecar-v73-preflight-20260813.json` owns the evidence (SHA-256
`5f368b0c639986e75abbcc7b3446f9ff8d42fbf1c8ee182948071e720d3c4bf3`).

**V73 terminal result: STOP.** Preserving the Transformer repairs V72's deficit
but the document state contributes no measurable causal utility. Persistent/
Transformer later-segment loss is 5.85078125/5.851171875, a gain of only
0.000390625 against the required 0.02. FineWeb-Edu and Cosmopedia losses are
each numerically identical to the retained control. More decisively, rotating
the incoming state across documents changes later loss by exactly 0.0. Read
gates do train to 0.02319 and 0.01819, the mean content gate is 0.51170, all
parameters receive finite nonzero gradients, and disabled parity remains exact;
this is learned but document-insensitive residual behavior, not dead machinery.

Candidate throughput is 20.41k versus 22.20k positions/s (91.91%), peak CUDA
allocation is 6.068 GB, and parameter ratio is 1.012442. Systems gates pass.
Because persistent already misses both Transformer improvement and state-swap
admission, reset and shuffled cannot rescue the branch and are not run. Decision:
`retire_v73_no_document_state_utility`. All model, runner, and tests are deleted;
compact preflight, persistent, and stop reports retain the evidence. V72--V73
together establish that local reconstruction can make state causal on synthetic
data or harmless on real text, but does not optimize state for future language.
The terminal report is `exact-cortex-sidecar-v73-stop-20260813.json` (SHA-256
`0f787ab8d7ec9674b19ff4764c7957f5469e51a9a48361ef7e76680a5ff3c157`). The next
admissible branch must meta-train the update on future loss.

### V74 preregistration: end-to-end fast-MLP test-time learning

The official TTT-E2E derivation identifies the decisive difference from V73:
the inner test-time objective is the same next-token loss as the outer task, and
the initialization is optimized for performance *after* those updates. Its final
large-model design updates regular MLPs in the last quarter of the Transformer,
because preparing upstream gradients makes many small fast layers less efficient
than fewer large ones. The paper also reports that exact gradient-of-gradient
training remains 3.4 times slower than full attention at context 8K and cannot
use cuDNN FlashAttention. MARULHO cannot silently claim that implementation on a
3060. A newer [ICML 2026 analysis](https://arxiv.org/abs/2602.21204) further
shows that broad KV-binding TTT layers reduce to learned linear attention. V74
therefore does not rename another key/value or reconstruction memory as TTT.

**Owned consumer formulation.** A complete causal Transformer remains the slow
model. Only the down projection of each last-quarter SwiGLU block receives a
temporary rank-8 LoRA delta. Each document begins from shared meta-parameters
`A0` (normal standard deviation 0.02) and `B0` (zero). For a completed segment,
ordinary full-vocabulary next-token cross-entropy produces an independent
per-document gradient for those fast parameters. A learned positive per-layer
inner rate, initialized to 0.1, applies one SGD update. The following segment is
predicted with the updated weights. The slow Transformer, `A0`, `B0`, and rates
are optimized from losses observed with the evolving fast weights.

To bound memory and avoid unsupported second derivatives, Stage A uses an
explicit first-order meta-gradient: inner gradients are detached, while a
straight-through identity connects each updated fast value to the shared
initialization and rate for the following segment. This is a FOMAML-style
approximation of TTT-E2E, not the paper's exact gradient-of-gradient algorithm.
It retains the essential falsifier—future next-token loss trains an
initialization for earlier next-token updates—without carrying Transformer
activation graphs through time. Disabled LoRA must be bit-exact to the ordinary
Transformer. Fast weights reset at document boundaries and never persist across
batch members. No reconstruction target, future token, query answer, label,
retrieval index, external model, compilation, or custom kernel enters an inner
update.

**Stage A0 — gradient-memory truth.** Freeze seeds `7401`, `7402`, and `7403`;
width 128, four Transformer layers, four attention heads, context 64, SwiGLU
width 512, and rank-8 fast LoRA only in the fourth block. Each batch contains
128 independent three-segment documents drawn online from 16 keys, 16 values,
and 32 distractors. Segment zero repeatedly exposes four random key/value pairs;
one complete middle segment is distractors; segment two repeatedly queries all
four keys, so 16 ordinary next-token targets depend on the old evidence. Train
800 slow AdamW updates at 3e-4, no weight decay, clip 1.0. Every completed
segment computes the identical inner gradient in all arms. `persistent_update`
applies its own gradient, `no_update_same_compute` discards it, and
`shuffled_update` applies another batch member's gradient. All arms share exact
initial tensors and document batches. Evaluate 4,096 fresh documents per seed.

Mechanical admission requires disabled bit parity, future perturbation leaving
earlier logits/updates exact, document-reset exactness, finite fast weights and
inner rates, and finite nonzero gradients for every slow and fast-initialization
parameter by update two. Behavioral admission requires persistent query accuracy
at least 80% and at least 20 percentage points above both controls in all three
seeds. Any miss deletes V74 before full language. No rank, layer, rate, step,
loss-weight, or dataset sweep follows a failed result.

**Stage A1 only after A0 passes.** Insert rank-8 fast deltas in the final three
MLPs of the exact 100.679M Transformer, retain the V72 long-document contract,
and first measure a two-update batch-8/16/32 safety ladder. Advance only if a
safe physical batch sustains at least 50% of the Transformer throughput and
projects below a one-hour three-arm screen. Quality then compares persistent,
no-update, shuffled-update, and the immutable Transformer on later-segment loss,
with the same 0.02 language and state-update counterfactual gates. Exact
second-order TTT remains a later systems experiment only if the first-order
mechanism establishes future utility.

**Stage B only after both passes.** Add strict state/checkpoint reload and owned
incremental generation, then test sequential-domain learning, source grounding,
state retention, shuffled/zero state, unseen long prose, and the 524,288-token
sustained contract. Base-runtime promotion still requires general quality not
worse than Transformer, a material long-context/source gain, at least 70%
throughput, bounded state, and every prior terminal safety gate. Stage A cannot
install a model.

### V74 terminal result: useful gradients, insufficient retention

Seed 7401 mechanically passes: disabled LoRA has zero output error, perturbing a
future segment changes no earlier loss or update, initial tensors and schedules
match, and all required gradients are finite and nonzero. Persistent updates
reach 11,255/16,384 delayed queries, or 68.695%. Discarding the identically
computed updates reaches 6.293%; applying another document's updates reaches
6.342%, both effectively the 6.25% chance rate. The 62-point causal separation
shows that ordinary next-token gradients can write document-specific facts into
temporary Transformer parameters. It is not a renamed key/value store.

The frozen gate nevertheless requires at least 80% in every seed. Seed 7401
misses it by 11.305 points, so the preregistered terminal condition fires before
seeds 7402/7403 or the 100M language stage. Decision:
`retire_v74_stage_a0_failure`. The most plausible next falsifier is not a rate,
rank, layer, or step sweep. It is an adaptive retention rule that can learn when
a new local gradient should overwrite accumulated fast state, with fixed-update
V74 and wrong-document updates retained as matched controls. The full report is
`end-to-end-ttt-v74-seed7401-20260813.json` (SHA-256
`e5573feb65e594dcf2840a5add5f6516e63dfcf56d0e0ee4aa48b48a0a25e7a8`).

### V75 preregistration: adaptive gradient retention

**Hypothesis.** V74's ordinary next-token gradients contain usable facts, but a
single fixed-rate update cannot distinguish evidence worth retaining from later
interference. V75 asks whether the Transformer can meta-learn that decision from
causal local statistics. This borrows the general idea of learned forgetting and
surprise from [Titans](https://arxiv.org/abs/2501.00663), while retaining V74's
standard Transformer, next-token objective, bounded first-order training, and
MARULHO-owned implementation. It is not Titans, an exact reproduction of
[TTT-E2E](https://arxiv.org/abs/2512.23675), or the unrelated classifier method
also named adaptive retention.

**Mechanism.** Restore V74's four-layer width-128 causal Transformer only inside
the temporary falsifier. Its final SwiGLU down projection again owns per-document
rank-8 fast weights. After each completed segment, compute the same independent
ordinary next-token gradient as V74. A shared two-layer gate maps four detached
per-document scalars—`log1p(loss)`, log gradient RMS, cosine alignment between
the candidate update and accumulated accepted update, and fast-state RMS—to one
sigmoid acceptance value. The accepted update is `fast <- fast - gate * rate *
gradient`. Gate parameters, shared fast initialization, rate, and the slow model
learn only from future next-token loss through the same explicit first-order
straight-through path. The gate cannot inspect token IDs, explicit segment
position, a query answer, future text, document identity, or a retrieval index.
It is initialized close to open, so V74 is the starting behavior rather than a
privileged hand-coded skip rule.

**Frozen Stage A0.** Reuse seeds 7401/7402/7403, exact online data distribution,
model dimensions, rank, batch 128, 800 AdamW updates at 3e-4, clipping, and 4,096
fresh evaluation documents from V74. Train one adaptive model per seed, then
evaluate five interventions on its exact slow weights and documents:
`adaptive_own`, `forced_open_own`, `matched_constant_own`,
`discard_same_compute`, and `adaptive_shuffled`. The matched constant is the
mean of adaptive gates over the two pre-query updates on the frozen evaluation
documents; it matches average update strength without per-segment or per-document
selection and does not inspect labels. All arms compute identical per-document gradients. Report
accuracy, segment losses, gate distributions, update norms, throughput, peak
CUDA allocation, complete gradients, schedule/data hashes, and final parameter
hashes. Disabled fast weights must remain bit-exact; future-token perturbation
must leave all earlier outputs, statistics, gates, and updates exact; state must
reset exactly at document boundaries.

**Decision.** Every seed must achieve at least 80% with `adaptive_own`, at least
10 percentage points above both `forced_open_own` and
`matched_constant_own`, and at least 20 points above both discarded and shuffled
controls. Seed 7401 is an early terminal: any miss stops the remaining seeds. No
rate, rank, depth, width, step, feature, threshold, or dataset sweep follows
failure. Passing all seeds admits only a bounded 100M long-document safety
ladder and matched language screen; it does not install a runtime. Failure
retains a compact report and deletes all V75 code and tests.

### V75 terminal result: retention gate reduces to step size

Seed 7401 passes every mechanical check with exact zeros: disabled fast weights,
future perturbation of earlier loss, update norm, gate, and gate feature. All
1,063,650 parameters receive finite nonzero gradients, training sustains 815.3
documents/s, and peak CUDA allocation is 1,526,546,944 bytes. The evidence is
safe and interpretable.

`adaptive_own` reaches 6,882/16,384 queries, or 42.004%. Its gate's mean
acceptance over the two pre-query updates is 0.259. Holding acceptance fixed at
that exact mean reaches 42.261%, 0.256 points better than adaptive selection;
forcing it to one reaches 36.188%. Discarded and wrong-document updates reach
6.927% and 6.891%. Thus gradients remain causally useful, but the learned gate
does not select useful documents or segments. Its pre-query 5th--95th percentile
is only 0.240--0.285, consistent with a nearly global rate reduction.

The 80% accuracy and 10-point matched-control gates fail, so seeds 7402/7403 and
all 100M work stop. Decision: `retire_v75_stage_a0_failure`. The result closes
four-statistic first-order retention gating under this contract; no gate feature,
threshold, rate, or width sweep is allowed. The report is
`adaptive-ttt-v75-seed7401-20260813.json` (SHA-256
`1f30112c2f31b8a6d926d149d596e1650d0b72e2d526bde7ca90f819d205136a`).
All model, runner, and tests are deleted. The scientifically distinct unresolved
TTT question is whether exact differentiation through the inner next-token
gradient can meta-shape the Transformer into a better learner than V74's
first-order approximation.

**Terminal gates.** Mechanical validity requires schedule/tokenizer hashes,
parameter ratio 0.99--1.01, exact no-leakage contracts, complete gradients,
finite state, owned generation, checkpoint tensor/logit/state reload, and
observed CUDA accounting. Behavioral promotion requires all of:

1. candidate disjoint general loss at most `control + 0.02`;
2. candidate true-source exact answers at least 64/256, at least 20 cases above
   the Transformer, and at least 51 cases above
   `max(question_only, shuffled_source)`;
3. shuffled-source exact answers at most 16/256, oracle at least 128/256, and
   true-to-oracle gap at most 64 cases;
4. no obvious collapse on the frozen unseen prose panel;
5. candidate steady training throughput at least 70% of the control and peak
   CUDA allocation no greater than 11.5 GiB.

If neither arm reaches 128/256 oracle, decide
`redesign_v67_training_objective_no_architecture_verdict`. If the Transformer
passes source/general gates and the candidate does not, decide
`retire_v67_queried_summary_exchange`. If the candidate passes every joint gate
and materially beats the control on true-source behavior, decide
`scale_v67_queried_summary_exchange_to_continual_validation`. No extra query
count, layer-order, optimizer, replay, or decode sweep is allowed after seeing
terminal results. A failed candidate leaves only compact evidence; its model,
runner, tests, and checkpoint are deleted.

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
