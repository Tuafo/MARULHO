# MARULHO Architecture Idea Ledger

This is a living set of falsifiable research directions, not a promise to keep
any mechanism. Ideas survive only when a matched experiment improves language,
memory, continual learning, or measured compute. Broad metaphors are translated
into an operation, prediction, control, and kill condition before implementation.

## Current synthesis

The first synthesis from this ledger has now been tested and refuted in its
implemented form. V7 attached four fixed-stable rotating memory banks and a
content write gate to a full-attention Transformer. At 16.79M matched tokens,
learned multiscale memory reached loss 4.6066 / 4.7% strict free relation versus
the Transformer's 4.6137 / 21.5%, and it failed to beat the simpler single-scale
control's 4.6061 / 10.5%. The gate was active and all bank states were used, so
the result kills this sidecar rather than merely exposing a dead implementation.

The surviving connection is narrower and more useful: many small units should
be tested as **conditional capacity inside the main predictive path**, not as
several incomplete language models or a weak memory attached beside one. V8
tested the simpler structural question of where a fixed feed-forward budget
should live across depth. Its short-budget early-heavy win reversed at the
durable budget, so no static profile becomes the baseline for micro-experts. V9
now tests whether direct reuse of earlier representations is more durable than
moving capacity toward them.

Geometry remains high-value instrumentation. Wavelets, toroidal phase,
reservoirs, cellular self-organization, and active inference remain scoped
components or grounded hypotheses until a precise experiment shows what they
add beyond the retired v7 mechanism.

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

### 2. Depth-shaped capacity — v8 retired

The uniform four-layer Transformer spends the same 2048-neuron SwiGLU budget in
every layer. V8 compares exact-total profiles of 2048/2048/2048/2048,
3072/2560/1536/1024, and 1024/1536/2560/3072. Total parameters, summed MLP
matmul work, embeddings, attention, normalization, data, and training schedule
remain matched. Only the depth location of nonlinear capacity changes.

This is inspired by [OpenELM](https://arxiv.org/abs/2404.14619), which reports
better parameter use from non-uniform layer-wise width/head allocation, but its
billion-parameter result is not assumed to transfer to 21M. Newer work such as
[Mixture-of-Depths Attention](https://arxiv.org/abs/2603.15619) also suggests
that preserving and selecting representations across depth can matter, while
[Mixture-of-Recursions](https://arxiv.org/abs/2507.10524) combines shared depth
with adaptive token computation. V8 deliberately tests the cheapest premise
before introducing depth routing or recursion.

Promote a profile only if it beats uniform heldout loss and strict free
generation by the frozen margins, then replicate before scale. If all profiles
tie or lose, retire static depth allocation and do not use it to justify a more
complex router.

V8 cleared its first two gates before failing durability. At
16.79M tokens, early-heavy beat uniform from two independent model/schedule seeds:
loss 4.5843 versus 4.6067 and 4.5839 versus 4.6021; strict free relation 25.4%
versus 7.0% and 30.9% versus 9.0%. Late-heavy lost both times. Compute, memory,
gradient, parity, and common-initialization audits are matched. The next
falsifier was a 67.11M-token uniform/early-heavy comparison because MARULHO v1
already showed that an early architectural win can disappear with more training.
That warning was correct: uniform/early-heavy finished at loss 3.8861/3.8957 and
tied free relation at 20.3%. Static early-heavy allocation is retired. The result
does not prove a training-stage crossover because the short and long experiments
used budget-sized cosine schedules; it motivates explicit curve measurement or
adaptive depth only if those mechanisms receive their own controls.

### 3. Depth-weighted representation reuse — active V9

V8 suggests that shallow computation can help under a short horizon but does not
justify permanently widening shallow layers. A smaller intervention is to keep
all uniform blocks and let later depths directly reuse earlier representations.
[DenseFormer](https://arxiv.org/abs/2402.02622) does this with learned
Depth-Weighted-Average connections and reports better language-model perplexity;
its weights include small and negative connections, so a convex mean is not an
equivalent implementation.

V9 uses only 14 new scalars at four layers and requires these controls on one
exact-reset graph:

- identity, which must exactly reproduce the Transformer;
- fixed mean, testing access without learned selection;
- fixed seeded random convex weights;
- learned unconstrained weights initialized to identity, matching the
  DenseFormer mechanism;
- learned simplex weights initialized near identity, testing whether an
  identity-preserving bounded manifold is sufficient.

This also tests the useful core of [Hyper-Connections](https://arxiv.org/abs/2409.19606)
and [mHC](https://arxiv.org/abs/2512.24880) without multiplying residual streams.
That restraint matters because a 2026
[stream-collapse analysis](https://arxiv.org/abs/2606.03483) finds that
multi-stream systems can concentrate information in one dominant stream. V9
measures depth-weight entropy, negative weights, embedding reuse, effective rank,
participation ratio, and adjacent-depth cosine, but only loss and free generation
can select a branch.

Kill V9 if learned reuse cannot beat identity plus fixed controls, if its gain is
only short-budget, or if data movement erases the quality/compute tradeoff.

### 4. Sparse shared-core micro-experts — research candidate

Replace each monolithic feed-forward transformation with a dense shared path
plus several small sparse expert residuals. All tokens retain the same attention
and vocabulary interface; the units add conditional capacity rather than
becoming separate language models. This directly tests the user's micro-to-macro
intuition in the part of a Transformer that naturally decomposes by token.

The first falsifier needs more than learned routing versus a small dense model:

- compare at both equal active FLOPs and equal wall-clock, while reporting total
  parameters and optimizer/VRAM cost;
- include shared-only, fixed-random, token-hash, and learned top-k routing;
- keep expert capacity and dispatch counts identical across routing controls;
- report router load, entropy, expert gradient coverage, active parameters, and
  specialization, but promote only heldout loss and free generation;
- preserve one dense residual path so a bad early router cannot delete basic
  language computation;
- compare against a dense model with similar total parameters when it fits, so
  sparsity is not credited for capacity that dense scaling would use better.

Kill the line if learned routing does not beat random/hash routing, if experts
collapse to interchangeable functions, or if dispatch overhead erases the
active-compute benefit on the RTX 3060. Even a win would establish a useful
conditional-compute substrate, not yet a new theory of cognition.

Relevant constraints are already known. [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
combines fine-grained routed experts with shared experts;
[PEER](https://arxiv.org/abs/2407.04153) retrieves from very large pools of tiny
experts; and [OLMoE](https://arxiv.org/abs/2409.02060) provides open routing and
specialization evidence at much larger scale. A 2026
[counterfactual routing audit](https://arxiv.org/abs/2605.07260) finds that
standard routers often miss better equal-compute routes on fragile tokens. That
warning matches MARULHO v2's learned-selector failure and makes random/hash and
counterfactual controls mandatory.

### 5. Gated multiscale dynamical memory — v7 retired

V7 performed the required memory-off, single-scale, always-write,
fixed-random-write, and learned-write comparison with one exact-reset parameter
graph. Learned memory did not beat the single-scale control and sharply harmed
strict free generation. No checkpoint or live code remains. A future memory
experiment must materially change the information/credit mechanism—for example,
content-addressed matrix updates or causal multiresolution attention—and must
not simply retune v7's decay constants.

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

V7 met the kill condition because its learned gate could not beat the simpler
control on both loss and free behavior. The broader literature remains useful,
but this fixed-stable sidecar is no longer an active architecture direction.

### 6. Wavelet-style temporal resolution — promising component

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

### 7. Memory gates — high-value mechanism, not a complete architecture

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

### 8. Toroidal phase memory — narrow use for time and order

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

### 9. Hyperdimensional or vector-symbolic memory — binding organ only

High-dimensional distributed vectors can bind roles, entities, and relations by
algebraic operations and store several items in superposition. This connects to
PMRM and relation memory more than to base token mixing. A useful experiment
would write label-free entity/event structures into a vector-symbolic episodic
store and test execution-grounded retrieval against dense learned memory.

[Vector Symbolic Architectures as a Computing Framework](https://arxiv.org/abs/2106.05268)
surveys the relevant binding and superposition algebra. High ambient dimension
alone is not a contribution; the binding operation, retrieval contract, and
capacity/interference curve must be explicit.

### 10. Autonomous local pattern generation — grounded research, not next LM

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

### 11. Free-energy principle — translate or reject

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

1. Completed: build and run the matched v7 dynamical-memory controls; retire the
   line after learned memory loses free generation and the simpler control.
2. Completed: run and retire V8 after its replicated short-budget early-heavy
   win reverses in the 67.11M-token durability comparison.
3. Run V9 identity/fixed/learned depth-reuse controls with read-only geometry
   diagnostics; require durability before promotion.
4. Add a read-only neural-manifold diagnostic to the maintained Transformer and
   every future survivor. Use it to explain results, never to promote a model.
5. Research and cost a shared-core sparse micro-expert candidate, then run
   shared-only, random, hash, and learned routing in one matched harness. Share
   data and compiled graphs where mathematically identical; never share weights,
   optimizer state, or labels.
6. Test wavelet-style compression only as a causal old-context mechanism on a
   task where context exceeds the local attention window, followed by a base
   language retention guard.
7. Keep toroidal phase, vector-symbolic binding, cellular self-organization, and
   active-inference ideas scoped to the memory/grounded problems they actually
   address unless evidence earns broader use.

The next creative bet is therefore specific: **many small conditional units may
increase useful capacity inside one full-strength predictive interface without
requiring every parameter for every token**. This is meaningfully different from
both a fully dense monolith and the retired design of several incomplete language
models or recurrent sidecars exchanging messages.
