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
then tested direct reuse of earlier representations: its small loss signal
replicated, but joint language behavior did not, so it is also retired.

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

### 3. Depth-weighted representation reuse — v9 retired

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

Two 16.79M-token seeds close the question. Learned-unconstrained reuse improved
heldout loss over the Transformer by 0.0092 and 0.0075, but strict free generation
moved by +14.8 and -0.4 points and the arm never beat identity plus every fixed
control on both metrics. Fixed-mean's first 0.0277 loss gain shrank to 0.0021;
fixed-random hurt loss; simplex remained almost identity. All candidate controls
shared one graph, matched throughput within 0.14%, and passed initialization,
parity, and gradient audits. The repeated signed pattern is informative: later
rows learned small negative weights on earlier depths and reduced their diagonal
weight to roughly 0.94-0.96. That looks more like residual attenuation or
decorrelation than useful content retrieval. Preserve that clue in future
normalization and micro-expert design, but do not preserve this architecture.

This tested the cheapest useful core of [Hyper-Connections](https://arxiv.org/abs/2409.19606)
and [mHC](https://arxiv.org/abs/2512.24880) without multiplying streams. A 2026
[stream-collapse analysis](https://arxiv.org/abs/2606.03483) remains a warning
against promoting a larger multi-stream variant without stronger evidence.

### 4. Product-key singleton micro-experts — retired V10 evidence

V10 tests the micro-to-macro idea inside one language model without creating
separate little language models. It replaces the dense MLP in one middle
Transformer block with two cooperating parts:

- a 1024-wide shared SwiGLU path that every token uses;
- a pool of 16,384 singleton experts, where each expert is one learned input
  vector, one nonlinearity, and one learned output vector.

Four independent query heads use a 128 x 128 product-key index and retrieve two
experts each, so only eight singleton functions are active for a token. Product
keys reduce routing from an exhaustive 16,384-way search to two 128-way searches
per head. Query normalization is per-token rather than batch normalization so
routes remain causal and full-forward generation can equal streaming generation.
The target model stores about 37.3M parameters, but the shared path, query
projection, key search, and eight active experts require roughly 92% of the
baseline dense MLP multiplies in the replaced block before top-k overhead. The
remaining three Transformer blocks stay unchanged.

This design replaces an earlier coarse eight-MLP-expert sketch. Conventional
expert-choice routing depends on other tokens in the batch and therefore breaks
MARULHO's streaming contract. Token-choice routing for full MLP experts requires
dynamic grouped dispatch that is poorly matched to a single RTX 3060. Singleton
experts can instead be gathered and combined with fixed-shape tensor operations,
making the causal scientific test practical on local hardware.

The exact-reset candidate graph must compare:

- shared-only, with expert work computed but its residual contribution zeroed;
- fixed-random product-key routing, with the initial router frozen;
- deterministic token-hash routing, with no learned selector;
- learned product-key routing.

All expert vectors train in the routed controls; labels remain metrics-only and
cannot affect route selection. Reports must include total and active parameters,
theoretical work, observed throughput, peak VRAM, used-expert fraction, routing
unevenness, per-head duplication, routing entropy, router gradients, expert-row
gradient coverage, and expert-vector diversity. Those measurements explain the
mechanism; heldout loss and strict free generation select the branch. Learned
routing must beat shared-only, random, hash, and the Transformer. A fixed router
may replicate without a learned-routing claim. No checkpoint exists before a
replicated survivor.

The CUDA/Inductor mechanism smoke passed after replacing two hash constants that
exceeded Triton's inferred int32 range. One candidate graph served all four
controls, avoiding three redundant compilations. Candidate modes stayed within
0.97% steady throughput, peaked at 1.80 GB, and one evaluation batch touched
roughly 60-69% of the pool with exactly eight assignments per token. These facts
admit the full falsifier; the two-step losses are discarded.

The first full seed selects token-hash for replication. At 16.79M tokens,
Transformer/shared-only/frozen-random/token-hash/learned-router loss was
4.6143/4.6166/4.6134/4.5388/4.6118 and strict free relation was
10.2%/15.6%/21.1%/29.7%/29.3%. Token-hash touched 68.5% of experts in the
diagnostic batch, gave 69.7% of expert rows final gradients, and improved its
training trace from the first checkpoint onward. Learned routing had full router
gradients but used only 9.5% of the pool, exposing specialization collapse rather
than a dead implementation. The current claim is narrow: stable token-indexed
micro-capacity may help; learned organization has failed this seed.

Fresh-seed replication confirms the important part without clearing the whole
gate. Token-hash again reaches loss 4.5372 versus Transformer 4.5990 and strict
free relation 34.4% versus 31.6%; shared-only is 4.6088 / 32.4%. Thus token-hash
repeats its general-loss gain and beats the Transformer behaviorally, but its
five-case behavior advantage over shared-only is 1.95 points, just below the
predeclared 2.0-point margin. The formal result remains redesign. Learned routing
again collapses, now to 8.9% pool usage, and is retired. The next falsifier must
delete query projection, product keys, and unused routing modes, retain only the
stable token-indexed singleton functions, verify equivalence to the winning arm,
and test durability at a larger budget.

That pruning is now implemented as V11. The hash-only candidate stores
36,180,480 parameters and deletes 1,114,112 query/key parameters plus all top-k
search. Its shared path and eight active singleton functions require 1,581,056
theoretical multiplies per token in the replaced block, 50.26% of the dense MLP.
With surviving weights copied, its logits exactly match V10 token-hash. The next
evidence is a larger-budget Transformer/shared-only/token-hash durability run,
not another learned-router repair.

The V11 CUDA smoke confirms that pruning removes real cost: candidate compile
falls from 39.4s to 22.8s, peak memory from 1.80 GB to 1.70 GB, and token-hash
steady rate rises from roughly 114k to 124.2k tokens/s. Shared-only is 122.7k,
within 1.21%, and parity plus expert-row gradients pass. The two-step scores are
discarded; these facts only admit the durability run.

Durability passes. At 67,112,064 tokens, Transformer/shared-only/token-hash loss
is 3.8951/3.9088/3.8747 and strict free relation is 19.1%/25.8%/35.9%.
Token-hash runs at 125.2k tokens/s versus 130.4k, peaks at 2.70 GB including the
staged schedule, and improves throughout the long training trace. The result
promotes V11 to checkpoint fidelity and unseen generation. It does not yet prove
continual learning, general coherence, or a superior scaling law.

Checkpoint fidelity now passes without pretending BF16/Inductor training is
bit-identical across long runs. An independent exact-recipe hash arm reaches
loss 3.8738 and 30.9% strict free relation, which still re-passes the original
fixed margins against both qualified controls. The strict 154.3 MiB checkpoint
reloads the complete token-hash model, tied weights, tokenizer hash, ownership,
and qualification record. The next falsifier is source-held-out continuation
and genuinely unseen multi-sentence generation from that reloaded artifact.

That falsifier now separates three facts. First, the model does generate
grammatical multi-sentence language, so the architecture is no longer at the
noise/repetition-only stage. Second, FineWeb-Edu/Cosmopedia source loss of
4.3092/3.6194 and zero of eight source-prefix passes show that language quality
is still below qualification. Third, repetition penalty plus no-repeat decoding
raises Cosmopedia bigram diversity from 0.675 to 0.948 without moving source
loss or prefix agreement. Decode collapse is therefore real but downstream of a
larger representation/training deficit. The fastest credible next connection is
not another memory mechanism: extend the same V11 checkpoint to the existing
~251M-token Transformer comparison point, then use the curve and repeated unseen
texts to decide whether fixed micro-capacity scales or should be redesigned.

The 251.66M point answers that question narrowly. V11 scales productively:
heldout loss drops 3.8709 to 3.4865, FineWeb-Edu prompt loss 4.3092 to 4.0272,
and Cosmopedia 3.6194 to 3.3689. It ties the local 251M Transformer on FineWeb
loss and is more diverse, but trails on Cosmopedia (3.3689 versus 3.2047); both
models still pass zero of eight source gates. A fixed token hash therefore looks
like a useful lexical-capacity primitive, not a complete replacement for
contextual computation. The same token always reaches the same singleton
functions even when its meaning changes. The next hypothesis should retain a
stable token-hash anchor while adding a causal context/manifold code without
learned top-k competition. Separately test 256-token training sequences. The
current checkpoint is configured for 72 tokens; first expand its rotary context
with identical learned tensors and require exact old-prefix output parity
at runtime.

General-only scaling also exposes the future continual-learning target cleanly:
relation candidate accuracy falls 95.7% to 32.8% and free relation generation
30.9% to 0%. Do not add replay or memory before the context-sensitive base branch
is tested; later mechanisms must recover this capability without losing the new
general-language curve.

### 5. Counterfactual-gated micro-assemblies — next V12 falsifier

A direct hidden-state hash is not the next experiment. [Hash Layers for Large
Sparse Models](https://arxiv.org/abs/2106.04426) already found that balanced or
random hashes over local token features beat clustering and longer-range context
features. That prior result agrees with MARULHO's fixed-hash win and learned
context-router collapse. Repeating a random-projection manifold hash would be a
known-risk variant, not a new mechanism.

The sharper opportunity comes from [counterfactual routing
analysis](https://arxiv.org/abs/2605.07260): standard MoE training scores only
the executed route, while equal-compute alternatives can assign higher
next-token probability on fragile tokens. MARULHO can test this unusually
cleanly because V11 has one rank-one expert pool, deterministic routes, complete
gradient coverage, and a strict 251M checkpoint.

The precursor experiment freezes the entire 251M model and evaluates several
alternative deterministic eight-expert assemblies for the same heldout token.
It records route regret without changing weights:

- fraction of tokens where an alternative lowers exact next-token loss;
- mean and tail loss improvement over V11's installed token hash;
- separation by model confidence and corpus;
- expert-pool usage, duplicate routes, and equal active compute;
- labels used only to score alternatives, never to choose an evaluation route.

If alternatives do not offer a material opportunity, retire the V12 gate before
training it. If they do, train a tiny causal gate from the pre-expert hidden state
to predict the best assembly using detached counterfactual targets. At inference
the gate sees hidden state only. Keep half of the eight active singleton slots as
the stable V11 token anchor and let the gate choose the other half from a small
fixed route bank. This turns the user's memory-gate idea into a concrete rule:
micro-units earn future selection because their executed consequence reduced
downstream prediction loss, not because they were merely similar or active.

[Equifinality in Mixture of Experts](https://arxiv.org/abs/2604.14419) is a
warning that elaborate routing topology may change asymptotic perplexity only
slightly; geometry is diagnostic unless counterfactual utility predicts a real
gain. Its companion [geometric-routing
study](https://arxiv.org/abs/2604.14434) makes rank-one expert semantics
inspectable, which may later help audit V11/V12 specialization, but does not by
itself solve quality.

Other user-proposed fronts remain tracked but are not promoted into the base:

- [Reservoir Computing as a Language Model](https://arxiv.org/abs/2507.15779)
  finds efficient training/inference but better Transformer prediction quality;
  reservoir dynamics remain a later fast temporal state or readout candidate.
- [Wavelet Logic Machines](https://arxiv.org/abs/2507.19514) reports compact
  classification evidence, not competitive causal generation; wavelets remain a
  possible multiscale memory-compression primitive rather than a language core.
- Free-energy/active-inference language frameworks remain high-level objectives.
  MARULHO uses their useful operational part—prediction error, uncertainty, and
  resource cost—but does not add an unfalsifiable free-energy layer.
- Torus, higher-dimensional geometry, neural manifolds, autonomous patterns, and
  micro-chaos/macro-order remain inspiration until each names a measurable state,
  intervention, and heldout advantage.

[PEER](https://arxiv.org/abs/2407.04153) establishes product-key retrieval and
single-neuron experts as the closest prior architecture; V10 is a small-scale,
causal, controlled test rather than a novelty claim for those primitives.
[DeepSeekMoE](https://arxiv.org/abs/2401.06066) supports retaining a shared path,
while [OLMoE](https://arxiv.org/abs/2409.02060) shows specialization only at much
larger scale. A 2026 [counterfactual routing audit](https://arxiv.org/abs/2605.07260)
finds that standard routers often miss better equal-compute routes on fragile
tokens. That warning matches MARULHO v2 and makes the fixed routing controls
mandatory.

The frozen audit passes its admission gate on a materially larger sample. Across
4,608 contexts, mean oracle improvement is 0.1911 and 40.5% of tokens have at
least 0.05 route regret. FineWeb-Edu/Cosmopedia mean gains are 0.2020/0.1802.
Fragile contexts carry most of the opportunity: 0.3159/0.2963 mean gain versus
0.0882/0.0641 on the confident halves. Every alternative hash is globally
0.62–0.66 loss worse than V11, while per-token oracle choices spread almost
equally across all four alternatives. This rules out a lucky global seed and
supports a context-prediction problem. Decision: `train_v12_counterfactual_gate`.

The next test trains both a linear and a small nonlinear utility predictor on
counterfactual labels from the general *training* corpora, while the 251M model
and expert pool remain frozen. Evaluation uses the separate FineWeb-Edu and
Cosmopedia holdouts. The predictor receives only the causal pre-expert hidden
state and estimates each alternative's loss improvement over the stable V11
route. At inference it chooses an alternative only when predicted gain is
positive. It must lower realized heldout loss by at least 0.02 on both corpora,
beat a baseline-always policy, preserve equal active compute, and show noncollapsed
selection. Failure retires the gate even though oracle opportunity exists.

That prediction test fails. The linear gate cannot fit useful route structure:
its best training threshold still loses 0.0205 and its FineWeb-Edu/Cosmopedia
heldout gains are -0.0381/-0.0334. The MLP fits training (+0.1126) but overfits,
reversing to -0.0757 combined heldout gain. Both choose harmful alternatives
more often than helpful ones. The parent is frozen and hash-identical, evaluation
selection never reads targets, and no artifact is saved. Decision:
`retire_v12_gate_cannot_predict_counterfactual_utility`.

The insight is narrower than “routing cannot work.” Equal-compute alternatives
contain oracle wins, but in this frozen random route bank the identity of a win
is not stable enough to predict from the local causal manifold. Counterfactual
regret may be idiosyncratic noise or require co-adapted experts, a richer state,
or direct execution feedback over time. Do not tune the same gate on the seen
holdouts. Preserve the audit as a constraint for a future memory gate, delete the
failed predictor, and move to the orthogonal long-context training ablation.

### 6. Multi-horizon future prediction — V13 retired

The long-context control passes its mechanical and likelihood gates without
solving the observed behavior. Expanding 72 to 256 rotary positions is
state-exact before training. After 67.11M matched tokens, 256-token heldout loss
improves 4.2033 to 3.3243, but FineWeb-Edu/Cosmopedia source losses move only
4.0272 to 3.9951 and 3.3689 to 3.3586. All eight anchored cases still fail.
Controlled decoding removes many literal loops but leaves generic topic drift.
Decision: `retain_long_context_infrastructure_reject_context_only_quality_explanation`.

This isolates a sharper hypothesis: next-token loss rewards local plausibility
without requiring one hidden state to represent a useful span of the future.
[Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
reports improved sample efficiency from independent future-token heads, while
[Future Token Prediction](https://arxiv.org/abs/2410.18160) specifically reports
smoother semantic state and better topic coherence at matched next-token
perplexity. The evidence is not universal: [Predicting the Order of Upcoming
Tokens](https://arxiv.org/abs/2508.19228) finds ordinary exact MTP inconsistent
and argues that distant-token targets can be too hard. This makes a controlled
local falsifier more useful than assuming MTP works.

V13 starts from the same strict 251M parent, 256-token expansion, seed, data
schedule, token budget, batch size, and learning rate as the retained 318M
next-token-only control. The base predicts the next token as before. Three small
independent shared-vocabulary heads predict dyadic horizons 2, 4, and 8 with a
bounded auxiliary weight. These heads exist only during training; the saved
inference graph remains one MARULHO-owned causal model. Survival requires at
least 0.02 lower matched heldout loss than the 3.3243 control, no corpus-level
source-loss regression, and visibly less topic drift under the same prompts.
Otherwise delete the heads and retire the objective. This is a cheap test of
multiscale predictive state before adding persistent matrices or columns.

The result is decisively negative. The temporary heads receive gradient and
their measured +2/+4/+8 losses fall from 7.8794/8.1342/8.1896 to
6.0638/6.8102/6.9785, so the task is learnable. Yet the exact same parent,
schedule, 67.11M tokens, and long-context recipe produce 4.9522 standard heldout
loss after the heads are removed, versus 3.3243 for next-token-only training.
Head attachment and removal are both bit-exact, all 36,180,480 inference
parameters persist, and no checkpoint is saved. Decision:
`retire_v13_future_prediction_no_control_gain`. The failed code is deleted.
Do not retune horizon weights on the same holdout. Exact future-token supervision
at these horizons creates gradient interference rather than the desired macro
state; a later semantic-state target must be materially different and tested on
fresh evidence.

### 7. Gated multiscale dynamical memory — v7 retired

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

### 8. Wavelet-style temporal resolution — promising component

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

### 9. Memory gates — high-value mechanism, not a complete architecture

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

### 10. Toroidal phase memory — narrow use for time and order

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

### 11. Hyperdimensional or vector-symbolic memory — binding organ only

High-dimensional distributed vectors can bind roles, entities, and relations by
algebraic operations and store several items in superposition. This connects to
PMRM and relation memory more than to base token mixing. A useful experiment
would write label-free entity/event structures into a vector-symbolic episodic
store and test execution-grounded retrieval against dense learned memory.

[Vector Symbolic Architectures as a Computing Framework](https://arxiv.org/abs/2106.05268)
surveys the relevant binding and superposition algebra. High ambient dimension
alone is not a contribution; the binding operation, retrieval contract, and
capacity/interference curve must be explicit.

### 12. Autonomous local pattern generation — grounded research, not next LM

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

### 13. Free-energy principle — translate or reject

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
3. Completed: run and retire V9 after two seeds show a replicated small loss
   signal but no joint win over identity and fixed controls.
4. Completed: V10/V11 isolate the surviving shared plus fixed token-hash
   micro-expert path; V11 remains the strongest sparse base evidence.
5. Completed: the V12 counterfactual opportunity is real, but both causal utility
   gates fail heldout prediction and are retired.
6. Completed: parity-gated 256-token continuation strongly improves its matched
   loss but does not solve unseen topic stability; retain the infrastructure and
   checkpoint only as a control.
7. Completed: dyadic future-token heads learn their auxiliary losses but
   catastrophically lose to the matched next-token control; delete and retire.
8. Completed: indexed-host scheduling preserves exact batch values/order and
   runs at 121.8k tokens/s while removing linear token-budget CUDA storage.
9. Next: scale the retained 318M next-token checkpoint to a materially larger
   token/parameter point before claiming an architectural ceiling.
10. Add a read-only neural-manifold diagnostic only when it can explain a branch;
   it never promotes a model.
11. Test wavelet-style compression only as a causal old-context mechanism on a
   task where context exceeds the local attention window, followed by a base
   language retention guard.
12. Keep toroidal phase, vector-symbolic binding, cellular self-organization, and
   active-inference ideas scoped to the memory/grounded problems they actually
   address unless evidence earns broader use.

The next creative bet is therefore specific: **many small conditional units may
increase useful capacity inside one full-strength predictive interface without
requiring every parameter for every token**. This is meaningfully different from
both a fully dense monolith and the retired design of several incomplete language
models or recurrent sidecars exchanging messages.
