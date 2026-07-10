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

### Explicit read/write memory

- **Borrowed:** Neural Turing Machines and the
  [Differentiable Neural Computer](https://www.nature.com/articles/nature20101)
  show that learned controllers can address and modify external memory.
- **Borrowed:** object-centric
  [Slot Attention](https://arxiv.org/abs/2006.15055) shows that competitive slots
  can bind to entities and generalize to new compositions.
- **Observed:** MARULHO's prompt-prepending memory and PMRM slots failed. The
  missing ingredient was not merely capacity; write selection, binding, and
  downstream use were weak.

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

## New architecture hypothesis: a distributed predictive organism

The working hypothesis is a single end-to-end language system made from many
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

## Why this could be genuinely new

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

## First decisive experiment

Build one vectorized reference architecture at roughly 21M parameters with:

- parallel bounded attention and multi-rate predictive units in every block;
- a learned mixer;
- a small latent episodic store;
- delayed counterfactual utility targets for unit output and memory writes;
- the existing strict language/checkpoint protocol.

Compare it with a fresh matched Transformer using one frozen data schedule. The
experiment must include:

- real heldout loss curves at early and durable token budgets;
- exact and free relation/conflict replacement;
- unseen multi-sentence generation with human semantic review;
- prompt-absence verification;
- every-parameter gradient coverage;
- fixed and dynamic state bytes;
- training and generation throughput;
- checkpoint/resume equality;
- ablations for exact workspace, utility credit, and episodic memory.

The architecture is killed or radically redesigned if it repeats delta v1:
an early loss win that disappears at a durable budget, poor free recall, failed
unseen semantics, or sequential execution so slow that scaling is implausible.

If the base model survives, the next experiment is sequential-domain learning
with conflict updates, bounded forgetting, replay consolidation, and exact
restore. Structural growth and grounded causal units follow only after that.

## Retired ideas

- SNN or GRU language recurrence as the active language core.
- Fixed routed columns as a base-language architecture.
- Prompt-text episodic retrieval as memory.
- Raw token surprise as assumed write utility.
- Delta-memory v1 as the next scalable core.
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
