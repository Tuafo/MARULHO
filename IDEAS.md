# MARULHO Architecture Idea Ledger

This is a living set of falsifiable research directions, not a promise to keep
any mechanism. Ideas survive only when a matched experiment improves language,
memory, continual learning, or measured compute. Broad metaphors are translated
into an operation, prediction, control, and kill condition before implementation.

## Current synthesis

The most promising connection across the new ideas is not a pure reservoir,
cellular automaton, torus, or free-energy model. It is a **multiscale gated
dynamical memory** attached to a quality-capable token predictor:

1. split the hidden stream into exact fast/medium/slow temporal resolutions;
2. update several small recurrent dynamical states with different stable memory
   horizons instead of duplicating complete language models;
3. let a learned gate write only when the candidate state improves future
   prediction, with always-write, random-write, and shuffled controls;
4. keep state trajectories geometrically bounded and measure their intrinsic
   dimension, perturbation decay, and specialization;
5. synthesize the scales back into one full-vocabulary prediction path.

This combines the useful part of small autonomous units, reservoir dynamics,
memory gates, neural manifolds, wavelets, and micro-to-macro organization while
avoiding the failed v3-v5 pattern of several weaker language paths exchanging
messages. V6 has now failed its full recipe-separated comparison, so this is the
active replacement hypothesis rather than a side branch.

## Ranked directions

### 1. Neural-manifold instrumentation — do soon

Geometry is immediately useful as measurement, even if it is not a new model.
For the frozen Transformer and each serious replacement, capture bounded hidden
samples per layer and measure:

- participation ratio and effective rank;
- a nonlinear intrinsic-dimension estimate such as TwoNN;
- local neighborhood preservation across layers and time;
- temporal autocorrelation and perturbation decay;
- state collapse, anisotropy, and unused dimensions;
- whether relation/entity examples occupy separable but compositional regions.

The testable question is not “does the model have a manifold?” Every finite
network does. The question is whether a candidate produces a representation
geometry that predicts lower heldout loss, stronger binding, or longer useful
memory. V6 demonstrated why the distinction matters: it maintained matrix norms
to 1.79e-7 yet still lost language quality and free behavior.

Relevant evidence:

- [Intrinsic dimension of data representations in deep neural networks](https://arxiv.org/abs/1905.12784)
  reports low-dimensional, nonlinear representation structure associated with
  generalization.
- [The geometry of hidden representations of large transformers](https://arxiv.org/abs/2302.00294)
  finds an expansion/compression/decoding profile across layers. This is a
  diagnostic pattern to test, not an architectural law to assume.
- [nGPT](https://arxiv.org/abs/2410.01131) is already MARULHO v6's concrete
  geometry experiment: bounded directions on a hypersphere, tested by language
  evidence rather than geometry alone.

### 2. Gated multiscale dynamical memory — strongest next architecture

Use a single capable language interface plus several small state organs. Each
organ receives a different temporal scale and uses a constrained recurrent map
with a distinct decay horizon. A gate decides whether and how much to update.
The state can be fixed-reservoir, lightly trainable, or fully trainable as an
ablation; “reservoir” is not assumed to be the winner.

Why this is plausible:

- a recurrent state costs linear time and can persist beyond a short attention
  window;
- different leak rates or stable spectra naturally create multiple clocks;
- gating protects old information from every-token overwrite;
- several small states provide diversity without duplicating embeddings,
  decoders, or entire language models;
- geometric constraints make stability measurable rather than rhetorical.

The closest existing families constrain the novelty claim. [Linear Recurrent
Units](https://arxiv.org/abs/2303.06349) show how diagonal stable recurrences can
train well; [Mamba](https://arxiv.org/abs/2312.00752) makes state transitions
input-selective; [HGRN](https://arxiv.org/abs/2311.04823) assigns increasing
memory horizons across layers; and [Griffin](https://arxiv.org/abs/2402.19427)
mixes gated recurrences with local attention. MARULHO is not claiming those
primitives as new. Its test is whether several small fixed-stable memories with
controlled writes can add useful temporal organization without replacing or
duplicating the strong language path.

[Zoology](https://arxiv.org/abs/2312.04927) is the strongest warning: much of the
real-language gap for efficient recurrent/convolutional models comes from
associative recall, while attention-recurrence hybrids recover most of it. That
is why the first MARULHO candidate keeps all four attention layers and treats
memory as a middle organ rather than an attention-free backbone.

Why a pure reservoir is not the plan:

- [Reservoir Computing as a Language Model](https://arxiv.org/abs/2507.15779)
  finds efficient character-level training/inference but clearly better
  prediction quality from Transformers in its matched comparisons.
- [Towards a Comprehensive Theory of Reservoir Computing](https://arxiv.org/abs/2511.14484)
  is useful for predicting memory capacity and readout geometry, but capacity on
  synthetic memory tasks is not proof of useful language abstraction.
- [Do Reservoir Computers Work Best at the Edge of Chaos?](https://arxiv.org/abs/2012.01409)
  shows that the edge of instability is not generally optimal. MARULHO should
  measure stability and task fit, not maximize chaos.

Controls required:

- same parameters and update tokens as the current quality baseline;
- no memory, fixed random memory, shuffled temporal-scale assignment,
  always-write, random-write, learned gate, and fully trainable recurrence;
- labels remain metrics-only and cannot enter a write decision;
- heldout loss and strict free behavior decide survival;
- perturbation growth, state rank, gate entropy, write frequency, active FLOPs,
  and state bytes explain the result but do not promote it.

Kill the direction if the learned gate cannot beat random/always-write controls,
if the memory state is ignored, if stability requires erasing useful history, or
if language loss remains below the monolith at the full matched budget.

### 3. Wavelet-style temporal resolution — promising component

Wavelets provide an exact way to separate slow approximation from fast detail.
The first implementation should use a fixed orthogonal Haar transform so that
information is neither invented nor silently discarded. Small state organs can
then process different scales before exact inverse synthesis.

This is more precise than assigning informal “short-term” and “long-term”
columns. It produces explicit scale bands, predictable sequence reduction, and
a shuffled-scale control. A later experiment may learn the filters only after a
fixed transform earns a result.

[Learnable Multi-Scale Wavelet Transformer](https://arxiv.org/abs/2504.08801)
is relevant inspiration for linear-cost multiresolution sequence mixing, but its
claims are not treated as MARULHO evidence. The local falsifier remains matched
next-token loss, behavior, memory, and wall time on MARULHO data.

### 4. Memory gates — high-value mechanism, not a complete architecture

A memory gate answers a concrete problem: which new information is worth
overwriting persistent state? The gate should be trained through future
prediction utility rather than surprise alone. Surprise can trigger a candidate
write, but the write earns promotion only when it improves downstream loss.

Modern evidence makes gating worth testing:

- [xLSTM](https://arxiv.org/abs/2405.04517) revisits recurrent memory with
  exponential gating and matrix memory.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) combine gated state
  decay with delta-rule updates for expressive linear-time sequence memory.

These works do not prove that a MARULHO gate will help. They justify a direct
learned-versus-random-versus-always-write comparison.

### 5. Toroidal phase memory — narrow use for time and order

MARULHO already uses RoPE, which represents position through products of planar
rotations; mathematically, this already introduces circular/toroidal phase
geometry. A torus is therefore not a novel language architecture by itself.

A genuinely different use would be a persistent bank of phase variables with
several incommensurate periods. Their joint phase can encode elapsed time, event
order, and multiple temporal scales compactly. Test it first on markerless event
order and long-delay retrieval, then require a full language win before keeping
it. Do not use toroidal geometry as a semantic claim without evidence.

[RoFormer / Rotary Position Embedding](https://arxiv.org/abs/2104.09864) is the
relevant baseline: any toroidal proposal must add something beyond the rotations
already in the active model.

### 6. Hyperdimensional or vector-symbolic memory — binding organ only

High-dimensional distributed vectors can bind roles, entities, and relations by
algebraic operations and store several items in superposition. This connects to
PMRM and relation memory more than to base token mixing. A useful experiment
would write label-free entity/event structures into a vector-symbolic episodic
store and test execution-grounded retrieval against dense learned memory.

[Vector Symbolic Architectures as a Computing Framework](https://arxiv.org/abs/2106.05268)
surveys the relevant binding and superposition algebra. High ambient dimension
alone is not a contribution; the binding operation, retrieval contract, and
capacity/interference curve must be explicit.

### 7. Autonomous local pattern generation — grounded research, not next LM

Neural or adaptive cellular automata demonstrate that shared local rules can
produce robust global organization. That is a real example of micro rules making
macro structure, but language supplies no obvious spatial neighborhood or local
repair target. Forcing it into the base LM now would repeat the unearned-column
problem.

The idea is better suited to MARULHO's grounded subsystem: self-organizing maps,
sensorimotor state, structural growth, or repair under damage. Relevant work
includes [Goal-Guided Neural Cellular Automata](https://arxiv.org/abs/2205.06806)
and [Locally adaptive cellular automata for goal-oriented self-organization](https://arxiv.org/abs/2306.07067).
Any language use must define the neighborhood, conserved information, global
objective, and credit path first.

### 8. Free-energy principle — translate or reject

The free-energy principle is too broad to install as an architecture. In passive
text training, full-vocabulary next-token cross-entropy already minimizes a
prediction error. A distinct contribution needs latent-state inference,
uncertainty, an action policy, and observations whose distribution the system can
change. That belongs naturally to grounded active learning, not the present
fixed-corpus base-LM comparison.

Predictive-coding formulations may still inspire local error signals or iterative
latent inference; [Predictive Coding Approximates Backprop Along Arbitrary
Computation Graphs](https://arxiv.org/abs/2006.04182) provides a concrete bridge.
The proposal is rejected if it only renames cross-entropy or backpropagation.

## Fast experiment order

1. Add a read-only geometry diagnostic to the frozen baseline and retained v6
   evidence; this is cheap and can explain the negative v6 result.
2. Implement one parameter-matched multiscale gated
   dynamical-memory candidate. Use exact Haar scales, three stable recurrent
   horizons, and a learned prediction-utility gate.
3. Run the decisive full-budget controls in one compiled harness. Share data and
   compiled graphs where mathematically identical; never share weights,
   optimizer state, or labels.
4. Keep toroidal phase, vector-symbolic binding, cellular self-organization, and
   active-inference ideas scoped to the memory/grounded problems they actually
   address unless evidence earns broader use.

The core creative bet is therefore specific: **many small dynamical memories may
organize useful temporal structure around one strong predictive interface**.
That is meaningfully different from both a monolithic Transformer and the
retired design of several incomplete language models exchanging messages.
