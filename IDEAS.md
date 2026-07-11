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

## Cross-front synthesis: dynamics need a job

The user's newer list—autonomous pattern generation, modern reservoir
computing, memory gates, neural manifolds, wavelets, geometry, tori,
higher-dimensional computation, micro-chaos/macro-organization, and the
free-energy principle—is not one architecture. It separates into five layers
that should not be conflated:

1. **A dynamical substrate** transforms a stream through recurrent local
   interactions. Reservoirs and autonomous attractors belong here.
2. **A memory-control rule** decides what persists, changes, or is erased.
   Delta correction and learned write/retention gates belong here.
3. **A temporal coordinate system** exposes fast and slow structure. Wavelet
   bands, learned delays, and periodic phase variables belong here.
4. **A representation diagnostic** asks what collective state was actually
   formed. Neural-manifold dimension, topology, rank, and perturbation decay
   belong here.
5. **An agent-level objective** trades prediction, uncertainty, complexity,
   and action. The free-energy principle belongs here, not in a replacement
   matrix multiplication.

This decomposition suggests a coherent non-monolithic destination: one capable
token interface supported by several small state organs, each with an explicit
timescale and write contract. Their local activity may be rich or even mildly
chaotic, but the shared readout must expose a stable, lower-dimensional macro
state that improves future language. “Emergence” is then measurable: microstate
perturbations may vary while paragraph-level predictions, retrieved entities,
and state geometry remain stable. Without that invariance, micro-chaos is just
noise.

Recent evidence sharpens the bets:

- [Reservoir Computing as a Language Model](https://arxiv.org/abs/2507.15779)
  directly compares character language models and finds the expected split:
  reservoirs are efficient, while Transformers retain the prediction-quality
  advantage. A pure reservoir is therefore not the active base-LM bet.
- [Reservoir-computing associative memory and itinerancy](https://www.nature.com/articles/s41467-024-49190-4)
  shows that reservoirs can store and revisit whole dynamical attractors rather
  than only static patterns. That supports a bounded trajectory-memory organ,
  not a claim that autonomous dynamics automatically acquire semantics.
- [Reshaping reservoirs with unsupervised Hebbian adaptation](https://www.nature.com/articles/s41467-025-67137-1)
  improves fixed reservoirs by adapting their connectivity from activity. If a
  fixed reservoir later earns a language signal, a label-free local-plasticity
  arm is more credible than endless random spectral-radius tuning.
- [Towards a Comprehensive Theory of Reservoir Computing](https://arxiv.org/abs/2511.14484)
  predicts memory capacity and readout geometry across many echo-state variants.
  Its practical use for MARULHO is a cheap preflight: reject reservoirs whose
  measured capacity/stability cannot cover the target delay before language
  training.
- [W4S4](https://arxiv.org/abs/2506.07920) constructs stable state dynamics from
  redundant wavelet frames and reports better long-horizon retention than
  HiPPO-based states on delay and long-range tasks. This makes wavelets more than
  a compression metaphor: they are a candidate initialization and coordinate
  system for slow state.
- [mGRADE](https://arxiv.org/abs/2507.01829) combines learnable delay embeddings
  for fast local dynamics with a minimal gated recurrence for global context.
  Its latest evidence supports testing separated fast/slow paths under a fixed
  memory budget, not adding many identical recurrent columns.
- [Toroidal topology of population activity in grid cells](https://www.nature.com/articles/s41586-021-04268-7)
  is strong evidence for toroidal collective geometry when the latent variable
  is periodic two-dimensional position. It supports phase/order organs and
  topology diagnostics; it does not imply that unrestricted semantics is a
  torus.
- [Dynamics of specialization in neural modules under resource constraints](https://www.nature.com/articles/s41467-024-55188-9)
  finds that structural modularity alone does not guarantee functional
  specialization. This is a direct warning for the “many small units” idea:
  differentiated causal contribution must be measured, not inferred from the
  wiring diagram.
- [Self-orthogonalizing attractor networks from the free-energy principle](https://arxiv.org/abs/2505.22749)
  offers an interesting theoretical bridge between prediction/complexity and
  separated attractors. It is not yet competitive language evidence. MARULHO
  may test its operational prediction—less-interfering state memories—not install
  “free energy” as an unfalsifiable label.

V14 has now tested and rejected the most immediate connection: a gated,
error-correcting associative state fed by mean segment summaries. The next
admissible hypothesis is a **causal dyadic state pyramid**: completed token pairs
produce Haar approximation/detail coefficients, completed pairs of
approximations form the next scale, and separate small states receive those
exact bands. V14 showed no language state advantage, so wavelets must not be
wrapped around its failed mechanism and sent directly into another 268M-token
comparison. First test the dyadic mechanism on long-delay retrieval against a
same-size gated recurrence, with raw averages, fixed Haar bands, and shuffled
band-to-state assignments. Only a clear memory-capacity and interference win
admits it to base language. Reservoir topology adaptation, toroidal phase, and
free-energy objectives remain later arms whose prerequisite problem is
respectively identified dynamics, genuinely periodic state, and an environment
the model can act on.

The V15 preflight closes that boundary on a harder-than-delay task. Streams
contain multiple key/value writes, random structured distractors, later key-only
queries, and an overwrite profile. Training length is 128; heldout lengths reach
512. Every arm has 112 state floats. Across three fresh seeds, length-512 mean
accuracy is 6.4% for a stronger flat GRU, 20.0% for raw dyadic averages, 19.1%
for random balanced contrasts, and 18.8% for ordered Haar contrasts. Raw averages
also win the 128, 256, and overwrite profiles. Haar never earns its required
three-point advantage, so ordered wavelet detail and the live V15 code are
deleted.

The geometry result prevents a wrong explanation: raw's 512-token effective
rank is 16.6, below Haar's 18.9 and shuffled's 20.8, despite better accuracy and
loss. Richer/high-dimensional activity is not automatically more useful. What
survives is a causal-organization question. The next matched preflight compares
raw dyadic averages with (a) seven same-size banks updated every token, (b) seven
banks all using one uniform block size with the same total update count, and
(c) dyadic banks receiving only each block's last token. That separates small
units, low-pass filtering, and genuinely different clocks before language.

V16 replicates the small-unit connection. With identical candidate tensors and
state bytes, seven token-rate banks reach mean 512-token recall of 37.9% versus
21.5% for uniform low-pass, 20.2% for dyadic last-token sampling, 19.1% for
dyadic low-pass, and 6.25% for a larger monolithic GRU. Every seed preserves the
ordering, and token-bank accuracy barely changes from length 128 to 512. The
overwrite profile falls to 22.7%, so updating facts remains a weakness. Token
banks receive seven times more recurrent updates and run 2.35x slower than flat;
this is a capability/organization result, not an efficiency result.

The mechanism is a grouped or block-diagonal GRU: every group reads the same
input, recurrent mixing stays inside the group, and one readout uses all groups.
It is related to [Recurrent Independent Mechanisms](https://arxiv.org/abs/1909.10893)
and [Slot State Space Models](https://arxiv.org/abs/2406.12272), but initially
omits their learned competition and sparse communication because MARULHO's
learned routers already failed. [Block-Diagonal LRUs](https://arxiv.org/abs/2602.12021)
independently support the narrower principle that state-mixing structure, not
width alone, shapes recurrent expressivity. MARULHO does not claim the primitive
as new; its next test asks whether this grouping adds language value to the
qualified V11 sparse token path.

The V17 screen attaches eight small all-active GRU groups after a middle V11
layer through a zero-initialized residual. Required controls are exact V11/off,
the same grouped parameters applied locally with no carried state, and one dense
GRU with the same total state width and more parameters. There is no router,
label-dependent update, semantic role assignment, or communication between
groups. First require exact attachment, streaming equivalence, full gradients,
and viable CUDA throughput; then use a bounded matched language screen before a
67M-token durability run.

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

### 7. Causal segment associative state — V14 retired

The 1B scale point closes the “just train the small model properly” question
without closing the architecture question. At 1,000,001,664 cumulative tokens,
V11 reaches 3.0805 heldout loss and 3.9678/3.1405 FineWeb-Edu/Cosmopedia source
loss. Controlled Cosmopedia generation is readable and diverse, but remains
generic; FineWeb loops whole propositions, and all eight anchored cases still
fail grounding. More tokens improve likelihood, not a persistent representation
of what the paragraph is about. Decision:
`retain_v11_1b_sparse_base_redesign_persistent_semantic_state`.

V14 preserved V11's full token path and added one materially different organ:
a bank of small causal associative states updated once per segment. Lower-layer
tokens form a segment key/value summary. Later segments query matrices containing
only earlier summaries. A gated delta rule corrects stored values instead of
merely adding or exponentially averaging another vector. The readout is injected
through a zero-initialized residual, so attaching the organ must leave every V11
logit exact before training. Training remains ordinary next-token cross-entropy;
there is no V13-style distant-token loss and no label-dependent write policy.

The primitives have close prior work and are not claimed as independently new.
[Gated Delta Networks](https://arxiv.org/abs/2412.06464) show that gated erasure
and error-correcting delta updates complement one another. [TransformerFAM](https://arxiv.org/abs/2404.09173)
shows that feedback latent representations can act as working memory, while
[Titans](https://arxiv.org/abs/2501.00663) combines attention with longer-lived
neural memory. [Large Concept Models](https://arxiv.org/abs/2412.08821) support
the broader need for representations above individual tokens, but rely on an
external sentence space and trillion-token scale; V14 learned its segment state
end-to-end inside MARULHO.

The matched arms started from the exact 1B checkpoint and one shared
67.11M-token schedule:

- ordinary V11 continuation;
- a parameter-matched local residual using the same projections but no state
  carried between segments;
- causal delta-state read/write without a learned gate;
- causal delta-state with a learned, label-safe write/erase gate.

All candidate arms reused one parameter graph and exact initial tensors. The
predeclared first gate required gated state to improve heldout loss by at least
0.03 over both V11 and the local residual while staying within 0.005 of ungated
delta. State bytes, matrix rank, gate entropy, write frequency, perturbation
growth, throughput, and active multiplies explained the result but could not
promote it.

The implementation preflight passes. V14 adds 102,912 parameters to the
36,180,480-parameter V11 parent, uses four 8-by-16 associative matrices, and
updates them every 32 tokens. Exact attachment, causal equivalence, streaming
equivalence, gradient reachability, strict checkpoint round-trip, and decision
logic pass 17 tests. The first compiled design accidentally executed every
control path and achieved only 26.7k tokens/s at 3.18 GB; that graph was
replaced, not excused. Mode-specific compiled graphs now measure 117.3k/off,
118.7k/local, 115.0k/ungated-delta, and 110.6k/gated-delta tokens/s at about
2.0 GB in a 20-step hot smoke. Gated state therefore retains roughly 91% of the
1B V11 continuation rate and is admitted to the full matched falsifier. The
smoke losses are discarded and its report is deleted.

The full falsifier is a decisive tie. Every arm receives 67,112,960 tokens on
the same 6,554-step indexed-host schedule. Off/local/ungated-delta/gated-delta
heldout losses are 3.0746086/3.0745938/3.0746429/3.0746036. The gated arm gains
only 0.0000050 over off, loses 0.0000098 to local, and misses the 0.03 gate by
roughly four orders of magnitude. Relation ranking is 48.4%/48.0%/48.4%/48.4%
and free relation generation stays 0% in every arm. No checkpoint is saved.

The organ is not dead. Gated-delta gives every associative parameter tensor a
gradient, produces full matrix rank 8 and trajectory rank 8, and learns mean
write/retention gates of 0.082/0.826. Yet no write exceeds 0.5, effective matrix
rank contracts to 2.97, perturbation gain is 0.0018, and the residual RMS is only
0.00035. Ungated delta keeps rank 8, writes fully, produces a larger 0.00081
residual, and slightly worsens loss. The model learned to suppress a segment
summary that carried no useful predictive advantage. Decision:
`retire_v14_no_segment_state_gain`. The 42.6-minute report is retained at
`reports/language_scaling/v14-segment-associative-67m-20260711.json` with
SHA-256 `ee20dbb54769845ec60e9564ddc2525c00d432c32b2edec36ab4626204111190`;
the model, runner, tests, and checkpoint surface are deleted.

### 8. Gated multiscale dynamical memory — v7 retired

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

### 9. Wavelet-style temporal resolution — promising component

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

[W4S4](https://arxiv.org/abs/2506.07920) is the closer state-space connection:
redundant wavelet frames define stable recurrent dynamics and retain long-delay
information. [mGRADE](https://arxiv.org/abs/2507.01829) supplies a complementary
fast/slow decomposition through learned delays plus a small global gate. The
first MARULHO use should still be fixed causal Haar bands because they are exact,
inspectable, and admit shuffled-scale controls; learned filters or delays come
only after the fixed decomposition earns a signal.

### 10. Memory gates — high-value mechanism, not a complete architecture

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

V14 performed that comparison in the current language setting. Its gate read
causal hidden summaries, received complete gradients, and separated retention
from delta correction without targets. It learned to suppress writing and tied
the off/local controls rather than improving them. This rejects mean-segment
associative gating, not every possible memory gate. A later gate must receive a
state whose ungated form first proves useful.

### 11. Toroidal phase memory — narrow use for time and order

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

The biological case is narrower but stronger than a generic geometry analogy:
[grid-cell population activity lies on a toroidal manifold](https://www.nature.com/articles/s41586-021-04268-7)
across environments and behavioral states. MARULHO should therefore use a torus
when it needs a stable periodic coordinate—elapsed phase, nested event order, or
cyclic sensorimotor position—and compare it with ordinary RoPE and unconstrained
state. It should not force entity or proposition meaning onto periodic
coordinates without an empirical topology result.

### 12. Hyperdimensional or vector-symbolic memory — binding organ only

High-dimensional distributed vectors can bind roles, entities, and relations by
algebraic operations and store several items in superposition. This connects to
PMRM and relation memory more than to base token mixing. A useful experiment
would write label-free entity/event structures into a vector-symbolic episodic
store and test execution-grounded retrieval against dense learned memory.

[Vector Symbolic Architectures as a Computing Framework](https://arxiv.org/abs/2106.05268)
surveys the relevant binding and superposition algebra. High ambient dimension
alone is not a contribution; the binding operation, retrieval contract, and
capacity/interference curve must be explicit.

### 13. Autonomous local pattern generation — grounded research, not next LM

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

Reservoir attractor work adds a second grounded interpretation: a local system
can autonomously regenerate a learned *trajectory*, not just a static shape.
That may eventually support internal simulation or reusable skills. For base
language, require a teacher-forced state to improve prediction before testing
closed-loop autonomy; otherwise the organ can generate elaborate dynamics that
the language interface cannot use.

### 14. Free-energy principle — translate or reject

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

[Self-orthogonalizing attractor networks from the free-energy principle](https://arxiv.org/abs/2505.22749)
suggest that prediction-plus-complexity pressure can separate attractor memories.
The falsifiable MARULHO translation is an interference test: at equal capacity,
does the proposed local objective preserve more independently retrievable states
and less forgetting than ordinary next-token training? That belongs after a
useful state organ exists. Until then, free energy remains a theory-level lens.

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
9. Completed: scale V11 to 1.000B tokens. Likelihood still improves, but fixed
   prompts expose persistent proposition loops and topic drift.
10. Completed: V14's local, ungated, and gated segment-state arms tie V11 at
    67.11M tokens; the active full-rank memory is suppressed and deleted.
11. Completed: V15 rejects ordered Haar detail but finds a replicated raw
    multiscale-bank advantage over a same-state-byte flat GRU.
12. Completed: V16 shows that seven independent token-rate banks, not clocks or
    averaging, drive the replicated recall and length-generalization win.
13. Next: attach an all-active grouped recurrent organ to V11 and screen off,
    local, dense, and grouped controls for parity, throughput, and language loss.
14. Use read-only state geometry and group ablations to explain that branch; they
    never promote a model.
15. Run 67M-token durability and unseen generation only after the bounded grouped
    language screen wins.
16. Keep toroidal phase, vector-symbolic binding, cellular self-organization, and
   active-inference ideas scoped to the memory/grounded problems they actually
   address unless evidence earns broader use.

The next creative bet is therefore specific: **several all-active independent
recurrent groups may organize one shared language state better than a dense
monolith, without duplicating the language model**. Synthetic evidence earns a
language screen, not belief; off, local, and dense controls decide whether the
effect survives inside V11.
