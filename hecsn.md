# HECSN — Hierarchical Emergent Concept Spiking Networks

**A scalable, biologically-grounded architecture for autonomous knowledge accumulation.**

Emergent concept formation via competitive learning with Kohonen/SOM updates, columnar organization with hierarchical approximate routing, adaptive sleep-replay consolidation, and predictive-coding bootstrap. Uses online-learned representations with Rate-Temporal Fusion encoding. GPU-accelerated sparse tensors with dual sparsity strategy (2:4 structured for fixed layers, CSR for dynamic connectivity) and hierarchical routing.

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Domain:** Computational Neuroscience · Unsupervised Learning
**Status:** Executable Research Scaffold — maintained path covers mechanism validation, memory consolidation, contextual routing, checkpoint/query tooling, the Terminus runtime surface with live observability and a richer multi-source control plane, HF active acquisition, a deterministic curated-web acquisition smoke, public-service registry-backed plus `live_remote_search` candidate pools with a default autonomous remote-search bootstrap, and a maintained grounded-semantics benchmark under episode-level evidence retrieval, query-focused concept separation, multi-evidence synthesis, surprise-weighted context integration, a mixed-world open-world grounding slice, recent-query-driven autonomy focus, live gap-driven candidate uptake, broader default candidate-pool stability, and a partial continuous abstraction proxy over the running Terminus stream; `autonomy_hf_smoke`, the feedback-calibrated `autonomy_hf_baseline` rerun, `memory_consolidation_hf_scale_robust`, `memory_consolidation_hf_baseline`, `autonomy_acquisition_hf_catalog`, `autonomy_acquisition_curated_web_smoke`, and `phase6_meaning_grounding_20260407` are green on the current tree, while HF scout plus broader live open-web acquisition remain exploratory and broader grounded semantic meaning remains below the project target
**Stack:** PyTorch 2.1+ · CUDA · 2:4 Structured Sparsity · CSR Sparse Tensors · FAISS (CPU)

**Executable status (2026-04-07):** The maintained smoke gate still passes on the mechanism-validation benchmark (`silhouette ~= 0.675`, `DBI ~= 0.304`, `trained_eval_recon_error 0.0619 < random_assignment 0.0907`), and the maintained runtime still exposes local plasticity, multiscale context, explicit tag/PRP replay-consolidation state, query/respond tooling, and active acquisition over maintained HF and curated-web source banks. The public service/UI surface is now the **Terminus** runtime control plane (`/terminus`, `/terminus/configure`, `/terminus/start`, `/terminus/stop`, `/terminus/tick`) rather than the earlier brain-named API, and that control plane now exposes materially better runtime observability: `/terminus` and `/stream/status` report recent runtime events, per-source tick visits / last activity / share of background tokens, next-source routing state, last-tick duration and token delta, autonomy trigger readiness with candidate-bank names, and now the recent user-observed gap history plus the aggregated autonomy `focus_plan` that will steer the next acquisition pass. The control plane itself is also less bottlenecked than before: the Ask workspace can now configure a multi-source `source_bank` plus an optional autonomy `candidate_bank` directly against the public Terminus API instead of collapsing the runtime back to a single manually entered source, and that `candidate_bank` now accepts registry-backed catalog specs (`semantic_registry`) plus live remote discovery specs (`live_remote_search`) instead of only flat source lists. If autonomy is enabled without a user-supplied candidate bank, the service now defaults to a small `live_remote_search` pool (`wikipedia` + `arxiv`) plus a strongly focus-biased shortlist, so recent unsupported queries can trigger remote discovery without first hand-authoring a source list. Fresh phase-6 reruns on the current tree now separate the healthy and unhealthy mechanisms more honestly: `reports/phase6_memory_scale_rerun_20260406/summary.json` remains green as a preservation stress test when final replay cycles are disabled; `reports/phase6_hf_catalog_rerun_20260406/summary.json` remains green, with active acquisition selecting `reviews`, stopping early when projected further acquisition is negative, and finishing with much lower residual candidate gap than round-robin; and the recalibrated autonomy baseline `reports/phase6_autonomy_baseline_feedback_tiebreak_20260407/summary.json` is green after two targeted fixes. First, bank-frontier queries now filter out tiny prefix fragments, so the controller stops treating junk terms like `wa / wal` as genuine knowledge gaps. Second, active selection now learns empirical marginal payoff only for near-tie decisions instead of letting feedback override a clear heuristic leader. On that maintained rerun, active finishes with lower held-out gap than round-robin (`active_final_mean_gap ~= 1.97e-08` vs `2.76e-08`, `active_final_max_gap ~= 2.19e-08` vs `2.98e-08`), source visits are no longer extremely lopsided (`news=4`, `wiki=5`, `reviews=3` rather than the earlier `9/1/2` pattern), and the diagnostic/info-gain surfaces now move in the same direction as the real held-out metric even though the older raw `gap_score` remains less faithful. The previously dangerous replay-heavy sleep path has now been recovered on the maintained baseline: after isolating sleep replay from global online state updates and reducing replay plasticity, `reports/phase6_memory_baseline_replay_fix_20260407/summary.json` passes with `task_a_relative_degradation_after_consolidation ~= -0.0787`, `task_a_overlap_after_consolidation ~= 0.999993`, and `replay_updates = 2400`, so the current replay path is no longer catastrophically destructive under the maintained sequential A/B benchmark. The semantic readout path has also advanced materially on the maintained slice: memory updates now persist episode-level text alongside raw windows, query summaries build `memory_episodes` from a wider raw match pool, diversify retrieval toward salient query terms, and split mixed windows down to query-relevant clauses so complementary support is not lost behind a dominant tail phrase. Shared grounding-text normalization now runs across retrieval, response selection, concept observation, and gap planning, which removes a class of false knowledge-gap detections caused by trivial inflection mismatches like `opens/open`, `solves/solve`, or `forms/form`. The concept layer trims mixed retrieved episodes down to query-relevant clauses and only merges stable concepts when they retain some lexical overlap, and the responder treats native decode as a fallback when grounded evidence is strong, extracting grounded clauses from selected evidence and synthesizing multi-fact answers when support is distributed across memories. The layer hierarchy is also closer to the paper path than before: maintained meaning/service checkpoints now enable the Context and Binding layers directly, and surprise-derived precision now scales context integration during both online training and replay so low-confidence inputs contaminate context less aggressively. That maintained path now covers two benchmark slices. The original `simple_animals` slice still answers grounded cat and dog queries, handles the compositional cat query, abstains on the unsupported ocean query, and keeps cat/dog concept evidence split. A new maintained `mixed_world` slice now keeps its footing in a noisier open-world-like memory store containing unrelated astronomy, geology, animal, weather, library, and music facts: it answers `"What opens jars and solves puzzles?"` with grounded octopus evidence, answers `"What forms when sunlight passes through water droplets?"` with grounded rainbow evidence, answers `"Which place lends books and what kind of rooms does it provide?"` with grounded library evidence, synthesizes `"What is closest to the sun and what do volcanoes release?"` across two distinct facts, and abstains on the unsupported submarine query. The next maintained slice is now partially wired into autonomy rather than staying query-only: `/query` and `/respond` persist recent unsupported terms, retrieval prompts, and follow-up questions into Terminus runtime state; live acquisition merges that recent-query focus with the frontier planner before candidate discovery, shortlist scoring, and active source selection; prefix-chain frontier noise like `rs / rt / ort` is now filtered before those plans are merged; active stop-gating now tolerates slight negative projected frontier scores when a candidate is strongly aligned to the current semantic focus instead of treating every tiny negative projection as a hard stop; the service runtime now auto-enables a focus-biased semantic shortlist for broader candidate banks when recent-query focus is present, so medium candidate pools do not depend on hand-tuned shortlist settings to stay on target; and projected commit selection now breaks near-tied frontier outcomes in favor of clearly semantically aligned candidates instead of letting a microscopic projected-gap delta override strong frontier relevance. The catalog discovery path is now tighter too: semantic ranking for mapping-backed remote candidates now uses the actual candidate windows rather than treating dict specs as empty banks, and `query_text` no longer leaks artificial semantic credit across every discovered remote result when real summary/title/description content is available. New direction tests now guard that path at multiple levels: remote-search ranking; shortlist scoring that keeps explicit recent-query focus ahead of ambient frontier/context noise; service-manager handoff/shortlist estimation including the default live-remote bootstrap when autonomy is enabled without a candidate bank; and end-to-end Terminus HTTP/TestClient paths that both commit `submarine_source` from a `submarine buoyancy ballast` gap and then recover a grounded answer on that same query after acquisition. The abstraction path is also no longer query-only on the maintained runtime: the current `ConceptStore` / `OnlineSlowFeatureMap` proxy now observes stored episode text and routing signatures during manual `feed`, respond-time learn-back, background Terminus ticks, and live acquisition training, so slow-feature concept state can accumulate while the system runs rather than only when a query asks for concepts. On the current tree, a live Terminus HTTP smoke through the public server surface still returns the mixed-world grounded synthesis `"Based on grounded evidence, mercury is the closest planet to the sun. Also, volcanoes release ash and lava during eruptions."` after `/terminus/configure`, `/terminus/tick`, and `/respond`, while the same live HTTP path keeps octopus concept terms grounded (`octopuses`, `solve`, `puzzles`, `open`) without spurious unsupported-term noise. A fresh live Terminus HTTP smoke from `checkpoints/terminus.pt` on a tiny synthetic setup now starts with `query_unsupported_terms=submarine,buoyancy,ballast`, carries `ballast, buoyancy, submarine` into the runtime autonomy `focus_plan`, and then commits `submarine_source` with `tokens_trained_total=24` and `stopped_early=false` after a single `/terminus/tick`, showing that recent user-observed gaps can now trigger real source uptake rather than only changing the internal target summary. A second fresh live HTTP smoke with **four** candidate sources and default shortlist settings still commits `submarine_source` with `tokens_trained_total=32` and `stopped_early=false`, showing that the maintained focus path now survives a broader candidate pool instead of only the smallest hand-tuned smoke. A third fresh live Terminus HTTP smoke with a public-service `semantic_registry` candidate bank backed by four local web sources now also commits `submarine_source` with `tokens_trained_total=192` and `stopped_early=false`, showing that the maintained Terminus surface can expose registry-backed candidate pools without dropping recent-query focus at the final commit step. A fourth fresh maintained Terminus HTTP smoke with a stubbed public `live_remote_search` provider on `checkpoints/terminus.pt` now also commits `submarine_source` with `tokens_trained_total=96` and `stopped_early=false`, showing that the public runtime can turn a live remote-discovery candidate pool into focused source uptake rather than only handling deterministic pre-curated entries. A fifth fresh maintained Terminus HTTP smoke with **no user-supplied autonomy candidate bank** now defaults to that remote-search bootstrap, commits `submarine_source` with `tokens_trained_total=96` and `stopped_early=false`, and then answers the same `submarine buoyancy ballast` query with grounded evidence (`"ballast water shifts pressure and buoyancy inside a submarine."`) and no unsupported terms. The UI control plane also received another stability pass: editable Terminus source-bank rows now use stable draft IDs instead of array indexes, and long evidence/raw-window text now wraps rather than overflowing the dashboard tables. Repo validation is green again on the current tree (`PYTHONPATH=src; python -m pytest -q` -> `121 passed, 3 warnings, 7 subtests passed`; `HECSN_UI`: `npm run build`). However, the broader semantic target is still not solved: the paper's Abstraction Layer is no longer only query-time, but it is still a proxy built around Memory Store episodes rather than a first-class dedicated core layer, and broader open-world concept abstraction / generalization at scale remain limited. On the current maintained path, Terminus should still be treated as a continuously learning retrieval-and-acquisition runtime with partial grounded semantics, not yet as a solved semantic agent.

---

## Table of Contents

1. [Philosophy: Emergence and Autonomy](#1-philosophy-emergence-and-autonomy)
2. [System Architecture](#2-system-architecture)
3. [Critical Mechanisms](#3-critical-mechanisms)
4. [Scalability Architecture](#4-scalability-architecture)
5. [Implementation Structure](#5-implementation-structure)
6. [Data Pipeline & Encoding](#6-data-pipeline--encoding)
7. [Development Roadmap](#7-development-roadmap)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Critical Risks & Mitigations](#9-critical-risks--mitigations)
10. [References](#10-references)

---

## 1. Philosophy: Emergence and Autonomy

The objective is a **self-sustaining knowledge accumulation system**: a network that observes a stream of unlabeled raw input (character sequences), extracts statistical regularities, forms stable concept representations through competitive learning, detects its own knowledge gaps, and actively seeks information to fill them — without labeled supervision, pre-encoded semantic priors, or pre-defined token boundaries.

> **What "Emergent" Means**
>
> Knowledge is emergent when representations arise from the *dynamics of interaction* between the network and its input stream, not from explicit labels, pre-trained static embeddings, or pre-computed distributional statistics. Semantic similarity, compositional structure, and hierarchical abstraction must self-organize from spike-timing correlations and prediction errors. The network begins as a tabula rasa at the level of symbols; structure at the sub-symbolic level (character patterns) is gradually wrought into symbolic abstractions through experience via local plasticity rules.

### Core Principles

**1. Local Learning Only:** No backpropagation through time. No global loss functions. Synaptic updates depend only on pre-synaptic spikes, post-synaptic spikes, and local neuromodulatory signals (three-factor rule). Continuous local learning enables online training without pausing for gradient computation.

**2. Scalability from Day One:** The architecture must accommodate 1,000 neurons or 100,000 neurons without rewriting the core logic. This mandates dual sparsity (2:4 structured for fixed-dimension weight matrices, CSR for dynamic connectivity graphs), hierarchical routing (not all-to-all), columnar organization with restricted connectivity, and distributed indices from the first commit. Scaling to 1M+ neurons requires distributed training architecture.

**3. Biological Plausibility as Constraint:** The design is constrained by what is known of neural microcircuits — excitatory/inhibitory ratios, sparse connectivity (10–20%), log-STDP learning windows (sublinear LTD prevents saturation), inhibitory STDP for E/I balance maintenance, synaptic scaling for homeostasis, and sleep-replay consolidation — not because biological mimicry is the goal, but because these constraints have evolved to solve the stability-plasticity dilemma. Layer naming uses functional descriptions (Memory Store, Competitive Layer, etc.) rather than cortical numbering (L1–L6) to avoid false mapping to specific cortical laminae, since this is a computational architecture inspired by — but not isomorphic to — cortical microcircuits.

> **Critical Constraint: Energy Reality**
>
> Energy efficiency claims for SNNs only hold on specialized neuromorphic hardware (Loihi: ~3 pJ/bit/hop). On GPUs, sparse operations incur overhead that often eliminates advantages. HECSN targets GPU implementation for accessibility, accepting that energy efficiency is a secondary concern to scalability and biological fidelity in this implementation phase. True energy efficiency requires future neuromorphic deployment.

### What HECSN Gains

| Category | Detail |
|---|---|
| Temporal Precision | Rate-Temporal Fusion encoding preserves both spike timing and rate information for sequential data. |
| Temporal Precision | Log-STDP with sublinear LTD naturally produces stable log-normal weight distributions without hard bounds. |
| Biological Fidelity | Cell assembly theory (Hebb 1949) maps directly to SNN dynamics with synaptic scaling and intrinsic plasticity. |
| Biological Fidelity | Adaptive sleep-replay is intended to reduce catastrophic forgetting via homeostatic renormalization and synaptic tagging. |

### What HECSN Sacrifices

| Category | Detail |
|---|---|
| Training Speed | Assembly formation requires hundreds to thousands of exposures. No gradient acceleration. |
| Training Speed | Sleep-replay phases add 10–20% overhead to training time (but prevent catastrophic forgetting). |
| Benchmark Performance | Cannot match transformer accuracy on standard NLP benchmarks at current network sizes. |
| Benchmark Performance | Character-level input requires longer training than word-level pre-tokenized approaches. |

---

## 2. System Architecture

### Layer Hierarchy & Data Flow

```
╔═══════════════════════════════════════════════════════════════════╗
║ [PARTIAL: Abstraction proxy — Stage 3 precursor]                  ║
║ • Current: online slow-feature concept memory over stored episodes║
║ • Full dedicated abstraction layer remains future                  ║
╠═══════════════════════════════════════════════════════════════════╣
║ SURPRISE MONITOR [Precision-Weighted Error]                        ║
║ • Layer-specific prediction error computation                      ║
║ • Precision (inverse variance) for attention weighting             ║
║ • Internally-derived neuromodulation gates plasticity              ║
║ • Generates dopaminergic/cholinergic/noradrenergic signals        ║
╠═══════════════════════════════════════════════════════════════════╣
║ BINDING LAYER [Event/Conjunction Detection]                        ║
║ • Binding neurons with coincidence detection (>threshold)          ║
║ • Short-term plasticity (facilitation/depression)                  ║
║ • PV+ interneuron-mediated fast inhibition                         ║
╠═══════════════════════════════════════════════════════════════════╣
║ CONTEXT LAYER [Temporal Integration]                               ║
║ • Approximate attractor dynamics (slow-transmission model)         ║
║ • ~15 token context window via recurrent connectivity              ║
║ • SST+ interneuron feedback inhibition for gain control            ║
║ • Context→Competitive multiplicative gain modulation               ║
╠═══════════════════════════════════════════════════════════════════╣
║ COMPETITIVE LAYER [Competitive Learning]                           ║
║ • Two-stage routing: HNSW (k candidates) → WTA (single winner)     ║
║ • Kohonen/SOM competitive update for prototype learning            ║
║ • Log-STDP + excitatory/inhibitory STDP + synaptic scaling         ║
║ • Structural plasticity via spike correlation (activity-dependent) ║
╠═══════════════════════════════════════════════════════════════════╣
║ MEMORY STORE [Dual Buffer: Reservoir + EMA + Capture State]        ║
║ • Slow buffer: reservoir-sampled assemblies (unbiased history)     ║
║ • Fast EMA: recent assemblies for drift / novelty baselining       ║
║ • Adaptive sleep from drift + replay-pressure signals              ║
║ • STC-like capture tags, replay spacing, consolidation level       ║
╚═══════════════════════════════════════════════════════════════════╝
▲ feedforward (spikes) ▼ feedback (modulation)
```

*Surprise Monitor implements precision-weighted predictive coding with internally-derived neuromodulation. Memory Store uses a dual-buffer architecture with an unbiased reservoir slow buffer, explicit tag/PRP/consolidation replay state on stored memories, and a fast EMA for drift / novelty baselining. Competitive Layer uses Kohonen/SOM competitive learning (not Oja's PCA), with both excitatory and inhibitory STDP for log-normal weight distributions and E/I balance. Two-stage hierarchical routing: HNSW (CPU-based) for O(log n) candidate selection → WTA inhibition for single winner selection.*

> **Executable Scope Note**
>
> The maintained runtime implements the Competitive Layer, surprise modulation, a maintained local plasticity path (`plasticity_mode = local_stdp`) with log-STDP-style eligibility traces, iSTDP-style inhibitory balancing, synaptic scaling, and plastic latent projections, explicit tag/PRP/consolidation replay state in the memory store, and a multiscale recurrent Context/Binding path over column assemblies. Surprise-derived precision weighting now also scales live Context Layer integration, so the paper's Surprise -> Context pathway is no longer only descriptive. The full neuron-level recurrent AdEx / molecular-STC circuit described later in this document remains reference architecture rather than the current regression target, and the paper's Abstraction Layer is now only partially represented by the maintained `ConceptStore` / `OnlineSlowFeatureMap` proxy: it observes live runtime memory episodes continuously, but it is not yet a first-class dedicated core layer.

### Feedback Pathways (Top-Down Modulation)

Top-down modulation is critical and operates through three defined pathways:

1. **Surprise Monitor → Competitive Layer (plasticity gating):** The Surprise Monitor computes layer-specific modulatory signals M(t) that multiplicatively gate the eligibility traces in the Competitive Layer's log-STDP rule. High surprise increases plasticity; low surprise stabilizes existing representations. This is implemented as the `modulator` parameter in the three-factor learning rule: `delta_w = lr * modulator * eligibility_trace`.

2. **Context Layer → Competitive Layer (gain modulation):** The Context Layer provides multiplicative gain to Competitive Layer activations: `activation_competitive = input * (1 + gain_context)`. This is intended to enable context-dependent routing, where the same input pattern can activate different columns depending on preceding context (polysemy disambiguation). The gain signal is the sigmoid-transformed Context Layer state vector, projected through learned weights.

3. **Surprise Monitor → Context Layer (precision weighting):** Precision estimates from the Surprise Monitor scale the Context Layer's integration rate. Low precision (noisy input) reduces the integration time constant, preventing unreliable patterns from contaminating the temporal context. High precision accelerates integration.

### Rate-Temporal Fusion (RTF) Encoding

Pure TTFS (Time-to-First-Spike) encoding is inadequate for sequential text data because it can only encode static patterns and cannot represent temporal context changes. HECSN uses **Rate-Temporal Fusion (RTF)** encoding, which combines:

- **Temporal component:** Spike latency encodes pattern identity (earlier = higher confidence)
- **Rate component:** Spike count encodes context confidence (more spikes = more certain context)
- **Burst coding:** Multiple spikes per pattern carry contextual information beyond single spike

RTF encoding is adapted from Li et al. (2024) [34], who demonstrated that combining rate and temporal coding in hardware-based artificial visual neurons achieves 94.4% accuracy on facial recognition and 91.3% on MNIST using NbOx Mott memristors. Their work validates the principle that dual-coded spike trains outperform either pure rate or pure temporal coding alone.

> **Adaptation Gap**
>
> Li et al.'s results are from hardware optical/visual encoding with physical Mott memristors, not software-based text processing. The accuracy figures above apply to visual pattern recognition on their specific hardware. HECSN adapts the *principle* of rate-temporal fusion to software-based character encoding. Performance on text data requires independent validation and should not be assumed equivalent to the visual domain results.

> **Encoding Limitation**
>
> RTF requires careful calibration: too many spikes increase energy consumption without proportional information gain; too few spikes lose temporal precision. The burst count (n_bursts) must adapt to input complexity, not be fixed.

> **RTF Input Definition**
>
> The encoder produces two aligned views from each character window:
> - `feature_vec` (shape `[input_dim]`, default `[128]`): active routing features from `RTFEncoder.feature_vector(chars)`. The maintained default is `order_weighted_ascii`; `unigram_ascii` and `hashed_ngram` remain explicit ablations behind the same latent routing interface.
> - `rtf_spikes` (shape `[128, n_bursts_max]`): burst-latency spike map retained for future neuron-level simulation and ablation work; the maintained scaffold does not route or learn directly from this tensor in its mainline path.
>
> This keeps the system tabula rasa (no labels, no pre-trained embeddings) while keeping one active routing contract.

### Representation Contract

To remove ambiguity between encoding, routing, and learning paths, HECSN adopts a single authoritative representation contract:

| Symbol | Shape | Producer | Consumer | Purpose |
|---|---|---|---|---|
| `feature_vec` | `[input_dim]` (default `[128]`) | `RTFEncoder.feature_vector(chars)` | Routing + competitive learning | Active character-window routing features; maintained default is `order_weighted_ascii` |
| `routing_key` | `[column_latent_dim]` (default `[256]`) | `W_project @ feature_vec` | HNSW Stage 1 | Candidate retrieval key |
| `prototype_i` | `[column_latent_dim]` | Competitive layer | HNSW + WTA Stage 2 | Column centroid in latent space |
| `assembly_act` | `[n_columns]` | Competitive layer | Context, Binding, Memory logic | Sparse column activation pattern |

Rules:
1. HNSW search input is always `routing_key` (latent), not raw RTF burst tensors.
2. Canonical maintained training uses `routing_key = W_project @ feature_vec` with `input_representation = order_weighted_ascii`.
3. Alternate routing representations (`unigram_ascii`, `hashed_ngram`) are allowed only through the same `feature_vec -> W_project -> routing_key` contract and are treated as explicit ablations rather than separate runtime paths.
4. `assembly_act` is a population activation vector and must not be used as a routing key unless explicitly projected to latent space.
5. All implementation blocks must state which contract symbol they consume/emit.

> **Maintained Representation Smoke Benchmark (2026-03-31)**
>
> `reports/refactor_representation_smoke/summary.json` compares `order_weighted_ascii`, `unigram_ascii`, and `hashed_ngram` on 4K training / 1K evaluation windows from WikiText-103. On this smoke slice, HECSN competitive-only routing beats the `OnlineKMeans` baseline on silhouette for all three representations (`0.0603 > 0.0515`, `0.0628 > 0.0156`, `0.0718 > 0.0372`). `hashed_ngram` is strongest numerically on this narrow slice, but `order_weighted_ascii` remains the maintained default because it is the representation exercised by the live service and phase benchmarks.

### Emergent Column Formation (Competitive Layer)

HECSN forms columns through **competitive learning** on raw input statistics.

**Problem with distributional semantics:** Pre-computing PMI over anchor vocabularies requires pre-defined tokens and global corpus statistics, violating the tabula rasa principle.

**Competitive learning solution:** Columns self-organize via online clustering of prediction error vectors. Winning columns update their prototypes via **Kohonen/SOM competitive update** (not Oja's PCA rule). The maintained scaffold currently uses a fixed column budget with dead-column revival and homeostatic thresholds; true capacity growth / spawn-on-error neurogenesis remains reference work. This ensures representations emerge from prediction dynamics, not pre-computed statistics.

> **Critical: Competitive Learning Rule**
>
> Standard Oja's rule converges to the first principal component — it performs online PCA, not clustering. HECSN uses the **Kohonen/SOM competitive update**:
>
> ```
> # WRONG (Oja's rule — PCA, not clustering):
> # dw = eta * (y * x - y^2 * w)
>
> # CORRECT (Kohonen competitive update):
> # dw = eta * (x - w)  ← moves prototype toward winning input
> ```
>
> This update moves the winning prototype toward the input that activated it, forming true cluster centroids rather than PCA projectors.

> **Character-Level Scalability Warning**
>
> The maintained routing input is a fixed-width character-window feature vector (default 128-dimensional order-weighted ASCII, with unigram ASCII and hashed n-grams retained as ablations). Even at this scale, competitive learning needs sufficient columns and long exposure to reach stable prototypes. Convergence still requires on the order of 10^4-10^5 character exposures per column, so the 5000-token bootstrap phase only initializes proto-assemblies.

### Log-STDP with Synaptic Scaling

Standard STDP leads to bimodal weight saturation (peaks at 0 and 1). The full HECSN circuit target uses **log-STDP** with synaptic scaling to avoid that collapse. The executable scaffold does not yet expose this full synaptic circuit; the maintained path instead uses surprise-scaled competitive prototype updates, direct column input weights, homeostatic thresholds, and dead-column revival. The equations below describe the planned full-circuit target rather than the current Stage-0 / Phase-3 regression path:

```
delta_w_ij = M(t) * [A+ * exp(-delta_t / tau+) * w_ij^mu+ (if delta_t > 0, LTP)
                      - A- * exp(+delta_t / tau-) * w_ij^mu- (if delta_t < 0, LTD)]

where mu+ ~ 0, mu- ~ 1 (sublinear LTD)
delta_t = t_post - t_pre (actual spike timing difference)
M(t) is the layer-specific modulatory signal from Surprise Monitor

Synaptic scaling: w_ij <- w_ij * (target_firing_rate / actual_firing_rate)^alpha
```

Sublinear LTD (mu- ~ 1) prevents runaway depression of strong synapses, while synaptic scaling maintains firing rate homeostasis. Together these produce stable log-normal weight distributions centered at intermediate values, avoiding the bimodal saturation that plagues standard STDP. Research (Effenberger et al., 2015 [35]; Carlson et al., 2013 [5]) shows that both excitatory STDP and inhibitory STDP, combined with synaptic scaling, are needed to reliably produce log-normal weight distributions and maintain E/I balance through plasticity.

> **Homeostasis Warning**
>
> Synaptic scaling requires firing rate statistics that partially sacrifice strict locality — each neuron tracks its own exponential moving average of firing rate, which is local, but the target rate is a global parameter. Recent research shows improper homeostasis can fragment networks: the "biphasic Gaussian rule" with non-zero setpoints at zero activity can silence half the network. HECSN uses the "zero Gaussian rule" where eta=0 for the silent setpoint, ensuring silent neurons can recover.

### Inhibitory STDP (iSTDP) for E/I Balance

Excitatory STDP alone cannot maintain stable E/I ratios. The full-circuit target adds **inhibitory STDP** on inhibitory-to-excitatory synapses:

```
delta_w_inh = eta_inh * [x_pre * (sigma_post - rho_target)]

where x_pre is the presynaptic (inhibitory neuron) trace
sigma_post is the postsynaptic (excitatory neuron) firing rate
rho_target is the target firing rate for excitatory neurons

This rule increases inhibition when excitatory neurons fire above target
and decreases inhibition when they fire below target.
```

iSTDP is intended to maintain E/I balance as a dynamical equilibrium through plasticity, not as a fixed parameter. Without iSTDP, E/I ratios are expected to drift as the network learns — excitatory STDP can progressively shift the balance toward excitation, eventually requiring compensatory mechanisms. In the current scaffold, this role is approximated by threshold homeostasis and competitive rebalancing; explicit inhibitory synapse plasticity remains reference work.

### Layer-Specific Surprise Monitoring

Instead of a single global modulator, the full HECSN design computes **precision-weighted prediction errors per layer**. The maintained executable scaffold now exposes this mechanism on the competitive path and uses the same surprise-derived precision signal to scale Context Layer integration on the live online/replay path; fully separate context- and binding-specific error channels are still benchmarked behaviorally rather than gated by distinct live plasticity channels:

- **Competitive Layer surprise:** Reconstruction error from competitive learning (prediction − actual)
- **Context Layer surprise:** Attractor state deviation (distance from expected attractor manifold)
- **Binding Layer surprise:** Binding failure rate (conjunctions that don't form)

Each layer's modulator M_layer(t) gates plasticity only in that layer, preventing global learning rate corruption. Additionally, **precision** (inverse variance of prediction errors) weights the surprise signals — high precision indicates reliable predictions, low precision indicates noisy input (reduce learning rates).

**Internal Neuromodulation (No External Reward):** All neuromodulatory signals are derived internally from the network's own prediction dynamics:

- **Dopamine analog:** Computed as **Reward Prediction Error (RPE)** — the signed difference between predicted and actual prediction error. `RPE = predicted_error - current_error`. Positive RPE = better than expected → LTP boost. Negative RPE = worse than expected → LTD boost. This is the Schultz (2015) formulation, not a sliding window average.
- **Acetylcholine analog:** Computed as the novelty of current input, measured by distance to nearest known assembly prototype. High novelty = increase learning rate. Low novelty = stabilize.
- **Norepinephrine analog:** Computed as the magnitude of surprise across all layers. High = global network reset/exploration. Low = normal operation.

### Hierarchical Approximate Routing

Instead of O(n) k-NN search, HECSN uses **HNSW (Hierarchical Navigable Small World) indexing** for O(log n) column routing:

- Two-level hierarchy: coarse clusters (navigable graph) → fine columns within cluster
- Cosine similarity for normalized assembly vectors (see note on Jaccard below)
- **CPU-based via FAISS** — HNSW indices are not GPU-acceleratable in FAISS; routing is CPU-bound with asynchronous prefetch to overlap with GPU computation
- Dynamic index updates as new columns form

> **On Jaccard vs. Cosine Similarity**
>
> The original design specified Jaccard similarity approximated via cosine on binary vectors. For sparse binary vectors, cosine = |A∩B| / sqrt(|A|*|B|) while Jaccard = |A∩B| / |A∪B|. These diverge significantly for vectors of unequal cardinality. Since HECSN's assembly vectors are real-valued (continuous prototype vectors from competitive learning, not binary), cosine similarity is the appropriate metric. Jaccard applies only if assemblies are discretized to binary activation patterns for the HNSW index — in which case, use MinHash approximation (not cosine) for accurate Jaccard search at scale.

> **Critical: HNSW Dynamic Update Limitations**
>
> HNSW can suffer from the "unreachable points phenomenon" during dynamic updates [3]. Frequent column insertion/deletion degrades graph connectivity, creating unreachable nodes. HECSN mitigates this via: (1) batched updates (rebuild index every N insertions), (2) tombstone marking instead of deletion, (3) periodic graph repair during sleep phases. Without these, routing fails at scale.

### Two-Stage Selection Protocol

**Problem:** HNSW finds the nearest column to an input (winner selection). WTA inhibition also selects a winner within the column population. Two selection mechanisms existed for the same role with no stated interaction.

**Solution:** Define a strict two-stage selection protocol:

```
Stage 1 — Coarse routing (HNSW):
  Input: current pattern vector
  Output: top-k candidate columns (k = 5–20, configurable)
  Purpose: reduce computation from O(n_columns) to O(k)

Stage 2 — Fine selection (WTA):
  Input: top-k columns from Stage 1
  Output: single winning column
  Mechanism: WTA inhibition among only the k candidates
  Purpose: enforce competitive learning within the relevant neighborhood

Learning rule updates ONLY the Stage 2 WTA winner.
HNSW index is updated ONLY after Stage 2 winner is determined.
```

Add `k_routing: int = 10` to `HECSNConfig` as an explicit hyperparameter. Smaller k = faster but more routing errors; larger k = slower but more accurate competition.

---

## 3. Critical Mechanisms

### 1. Adaptive Sleep-Replay Consolidation

Catastrophic forgetting occurs when new training overwrites old synaptic weights. The Stage-0 executable path now uses a **two-tier sleep controller** rather than a single drift-triggered replay loop.

**Critical finding:** Sleep must be interleaved, not batched. Catastrophic forgetting occurs when training continues without consolidation — memories can become irrecoverable [1]. Sleep ratio of 10–20% of training time is optimal — too little prevents consolidation, too much reduces learning efficiency.

**Active phase (80–90%):** Network processes input stream, updates Competitive Layer via log-STDP, writes to Memory Store at rate alpha.

**Sleep controller:**

- **Micro-sleep:** every ~200 tokens, run a cheap replay burst over the top 5 replay candidates for 10 steps. This keeps newly learned assemblies from being ignored without paying the cost of a full consolidation pass.
- **Deep sleep:** run scheduled maintenance over the top 100 replay candidates, and trigger emergency deep sleep from closed drift-floor windows rather than the rolling minimum. The recovered 100-column baseline uses a scheduled deep interval of `2500` tokens with `150` replay steps.
- Replay priority is now `importance * log(1 + tokens_since_last_replay)`, so newly learned items are reviewed soon while well-consolidated items are revisited less often.
- Replay also nudges the fast memory trace toward the replay centroid. This closes the control loop between replay and the drift proxy; the older implementation spent replay compute without changing the drift signal itself.
- Temporal decay on `slow_mean` is implemented as an experimental knob. The motivation is valid: a flat reservoir mean can anchor drift to the pre-specialization era. The original 1K sweeps on the global slow mean did not improve drift-floor convergence. In the winner-local controller, however, a later audit showed that `slow_mean_decay` had not yet been wired into the local slow traces, so the first full-length 100-column decay retest was not a valid evaluation of temporal decay under winner-local drift.
- Warm-starting slow memory after bootstrap remains an experimental knob, not a mainline default. It reduces average drift and replay cost in some runs, but by itself did not improve drift-floor slope in the first fixed-token sweeps.
- **Validated Stage-0 fix:** winner-local drift. Replay storage remains global, but the drift controller compares fast and slow traces within the active winner bucket so changes in overall winner mix do not masquerade as overwrite. On the 1K-neuron-equivalent Stage-0 run, this reduced drift-floor slope from `+0.00136` to approximately `+8.9e-7` while keeping the full gate green and reducing replay updates from about `10.5K` to `6.9K`.
- **Validated Phase-1 hardening:** emergency deep sleep is now triggered from closed drift-floor windows rather than the rolling minimum, scheduled and emergency deep cooldowns are separated, and replay priority is penalized by replay count so the same high-importance items do not monopolize maintenance replay.
- Competitive prototypes use momentum during both wake and replay updates so columns do not oscillate between transient states as sleep cadence widens.
- Synaptic weights undergo homeostatic renormalization: `w <- w_target * (w / w_target)^lambda`
- **Activity-dependent structural plasticity:** Prune weights below theta_prune; form new connections between neurons with correlated **spike activity** (not sub-threshold voltage), using spike record already stored for STDP. This eliminates the need for separate sub-threshold voltage tracking.
- HNSW index repair: rebuild graph connectivity to fix unreachable points

Mechanism: Micro-sleep handles recency, deep sleep handles drift-floor recovery, and spaced replay prevents the controller from wasting budget on already refreshed assemblies. Renormalization preserves relative weight ordering while preventing saturation.

### 2. Predictive Coding Bootstrap (Cold Start Solution)

During the initial phase, the network has no stable representations. HECSN uses **predictive coding** at the character level:

The network predicts the next character from previous context and uses prediction error to drive initial structure formation. This creates "proto-assemblies" that refine as statistics accumulate. Prediction error serves as the initial teaching signal before competitive learning stabilizes.

**Architecture:** The Competitive Layer generates predictions via its learned prototypes; the Surprise Monitor computes prediction errors. Errors propagate via precision-weighted connections back to modulate Competitive Layer plasticity.

> **Tabula Rasa Clarification**
>
> The predictive coding bootstrap constitutes a **minimal structural bias**, not a semantic label. Next-character prediction is a temporal regularity extraction mechanism — any Markov process does this, and it requires no semantic priors, no vocabulary, no labeled data. What "tabula rasa" means in HECSN is: no pre-encoded semantic categories, no pre-trained embeddings, no pre-defined token boundaries. The network *does* begin with the structural capacity for temporal prediction (this is architectural, like the brain's cortical hierarchy existing before learning), but it does *not* begin with any knowledge of what sequences mean. This distinction is important: "tabula rasa" refers to the content of representations, not the computational architecture.

> **Cold Start Duration**
>
> The bootstrap phase requires ~5000 tokens minimum before competitive learning transitions from random to structured clustering. However, full convergence requires 10^4–10^5 exposures per prototype. During bootstrap, representations are unstable and highly sensitive to initial input statistics. Training should begin with diverse, representative data to avoid pathological attractor states. The 5000-token threshold is empirical and should be validated by monitoring reconstruction error convergence.

### 3. Approximate Attractor Dynamics (Context Layer)

The reference HECSN design targets **approximate attractor dynamics** for temporal integration. The maintained executable scaffold now uses a multiscale recurrent `ContextLayer` over column assemblies, with fast/medium/slow traces and inhibitory stabilization, and validates it via Phase-3 separation / winner-switch / `bank`-probe metrics rather than via a neuron-level recurrent spiking attractor loop. Recent neuroscience research (Nair et al., Nature 2024 [36]; Sagodi et al., 2024 [37]) still constrains the full target design:

**Structural instability of continuous attractors:** Perfect continuous attractors (line attractors, ring attractors) are structurally unstable — infinitesimal parameter changes destroy them [37]. Real neural systems implement *approximate* attractors: slow manifolds that approximate attractor dynamics for functionally relevant timescales without requiring the infinite precision that true continuous attractors demand.

**Slow neurotransmission requirement:** Nair et al. (2024) showed that line attractor dynamics in the mammalian hypothalamus require slow neurotransmission (tau ~ 20s) with fast feedback inhibition, not fast glutamatergic excitation [36]. Dense subnetwork connectivity (~36%) among attractor-contributing neurons is also required.

The reference circuit design uses:

- **Slow excitatory time constant:** `tau_slow = T_per_token * context_tokens = 25 ms * 15 = 375 ms` (not 5000ms). This provides the same functional ~15 token context window with 13x less simulation time. The 5000ms figure was derived from human reading speed in biological circuits — not appropriate for GPU simulation where "time" is defined by `T_per_token`.
- **Dense local connectivity** among context neurons (30–40% within the Context Layer, vs. 10–20% elsewhere)
- **SST+ interneuron feedback inhibition** for gain control (replacing the over-simplified center-surround model)
- Acceptance that the manifold is *approximately* stable, not perfectly continuous — drift along the manifold is expected and tolerable for context integration tasks

### 4. Binding Neurons with Short-Term Plasticity (Binding Layer)

The reference binding design uses **binding neurons** that detect coincidences across columns with short-term plasticity. The maintained scaffold now uses a stronger binding circuit over column assemblies and context predictions, with facilitation/depression state and PV-like inhibition, rather than only a static conjunction-memory proxy:

- Threshold theta_binding requires convergent input from >=2 columns within tau_binding = 50ms
- Short-term facilitation enhances transmission with repeated activation (temporal filtering)
- Short-term depression provides adaptation to sustained input
- PV+ interneurons provide fast feedforward inhibition for temporal precision
- Successful binding creates new compositional assembly via polychronous firing

### 5. Synaptic Tagging and Capture (STC)

The reference long-term consolidation design targets **Synaptic Tagging and Capture**. The maintained scaffold now uses drift-aware replay scheduling plus an explicit phenomenological tag/PRP state stack: each stored memory carries a decaying capture tag, a local PRP trace, access to shared global/bucket PRP pools, consolidation level, replay spacing, and replay counts. Maintenance sleep recruits PRP into tagged memories before replay converts capture pressure into longer-lived consolidation:

- Early-LTP (1–3 hours): Protein-synthesis independent, reversible
- Late-LTP (>3 hours): Requires protein synthesis, stabilized by persistent kinases
- Weak stimuli set "synaptic tags" that capture proteins from strong stimuli
- Behavioral tagging: Novel experiences can capture plasticity-related proteins

**Temporal parameters (functional time units):**

In the full-circuit reference design, STC timescales use **functional time units** tied to network activity, not wall-clock time:

```python
# config/model_config.py
functional_minute: int = 500  # tokens — tunable hyperparameter
# Rationale: a cortical "experience" at ~10 Hz firing rate processes
# roughly 600 spike bursts per biological minute.

# STC timescales in functional minutes (preserves biological ratios)
stc_tag_duration_weak: float = 30.0      # functional minutes = 15,000 tokens
stc_tag_duration_strong: float = 120.0   # functional minutes = 60,000 tokens
stc_prp_tau_weak: float = 60.0           # functional minutes = 30,000 tokens
stc_prp_tau_strong: float = 240.0        # functional minutes = 120,000 tokens
```

**Note on reduced maximum:** Chong et al. (2025)'s 9-hour figure is for strong stimulation in hippocampal slices under optimal conditions. In a continuously learning network, using 9 biological hours would mean a tag set at token 1 is still active at token 270,000. 240 functional minutes (120,000 tokens) tests whether the mechanism adds value before committing to maximum biological bounds.

> **STC Implementation Constraints**
>
> Full molecular STC models require sub-100us timestepping and dozens of molecular species. HECSN uses a simplified phenomenological model: tags are continuous-valued (0–1) with exponential decay, PRPs have concentration dynamics with first-order synthesis and degradation, and capture occurs when tag * PRP exceeds threshold. This sacrifices molecular fidelity for computational tractability. The model follows the approach of Luboeinski & Tetzlaff (2021) [39] who validated STC in recurrent spiking networks with similar simplifications.

### 6. Intrinsic Plasticity (IP)

Each neuron adapts its excitability based on firing history:

```
theta_threshold(t+1) = theta_threshold(t) + eta_IP * (f_actual(t) - f_target)

where f_target is the desired firing rate (e.g., 5 Hz)
eta_IP is the intrinsic plasticity learning rate
```

IP prevents silent neurons (no plasticity) and runaway excitation, maintaining network sensitivity to input changes.

> **Precision-Weighted Safeguard**
>
> If precision (inverse variance of prediction errors) falls below threshold theta_low = 0.1, input is flagged as noisy and learning rates are reduced. If precision exceeds theta_high = 10.0 for 1000 tokens, network is overconfident — exploration noise is injected to prevent local minima.

---

## 4. Scalability Architecture

Every component is designed to scale from 1K to 100K neurons on single GPU, with distributed architecture for 1M+ neurons.

### Dual Sparsity Strategy

HECSN uses two distinct sparsity paradigms for different components:

**1. 2:4 Structured Sparsity (fixed-dimension weight matrices):**

Applied to weight matrices with known, fixed dimensions — e.g., feedforward projection matrices between layers, binding neuron input weights. PyTorch supports semi-structured (2:4) sparsity with hardware acceleration on Ampere+ GPUs via cuSPARSELt:

- 2:4 pattern: 2 non-zero values per 4-element block (50% sparsity)
- Memory reduction: 50% of dense storage
- Speedup: ~1.6x on A100 GPUs (not 2x due to metadata overhead) [15]
- Requires Ampere+ GPUs (A100, RTX 30xx+). Older GPUs show no speedup or slowdown.

**2. CSR Sparse Tensors (dynamic connectivity):**

Applied to synaptic weight matrices in the Competitive Layer and Context Layer where connectivity is dynamic (structural plasticity adds/removes connections). CSR (Compressed Sparse Row) format handles arbitrary sparsity patterns:

- Natural fit for 10–20% connectivity density
- Supports dynamic insertion/removal of connections
- **Limitation:** PyTorch CSR support is beta. Basic operations like `cat`, `stack` do not work on CSR tensors (pytorch/pytorch#129843). Workaround: convert to COO for operations requiring concatenation, then back to CSR.
- **No hardware acceleration** — CSR operations run on standard CUDA cores, not Tensor Cores. Performance advantage comes from reduced memory and arithmetic, not from hardware-specific kernels.

> **Sparsity Reality Check**
>
> 2:4 structured sparsity delivers real speedups only on Ampere+ GPUs and only for large matrices. For the typical HECSN scale (10K–100K neurons, 10–20% connectivity), CSR sparse operations may actually be slower than dense operations due to indexing overhead. Profile before committing to sparse format at small scale. At 100K+ neurons, sparse becomes mandatory for memory.

### Hierarchical Assembly Index (HNSW on CPU)

Pattern completion requires nearest-neighbor search over Memory Store assemblies. HECSN uses **HNSW** (CPU-based) with cosine similarity:

```python
# retrieval/hnsw_index.py

import faiss
import torch
import numpy as np
from typing import List, Tuple, Optional


class HierarchicalAssemblyIndex:
    """CPU-based approximate nearest neighbor index using HNSW.

    CRITICAL: In upstream FAISS, HNSW search is CPU-bound in standard builds.
    GPU acceleration is available for other index families; for HNSW-like GPU
    paths, use the FAISS+cuVS route with separate validation and feature checks.
    Routing is CPU-bound. Use asynchronous prefetch to overlap
    with GPU spike computation.

    Maintains a parallel _vector_store for rebuild capability.
    Tombstone strategy: mark deletions, rebuild during sleep phases.
    """

    def __init__(self, dim: int, rebuild_threshold: int = 1000):
        self.dim = dim
        self.rebuild_threshold = rebuild_threshold
        self.insertion_count = 0
        self.tombstones: set = set()

        # Parallel store for rebuild capability
        self._vector_store: dict[int, np.ndarray] = {}

        # HNSW index for O(log n) search — CPU only
        self.index = faiss.IndexIDMap(faiss.IndexHNSWFlat(dim, 32))
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128

    def add(self, assemblies: torch.Tensor, ids: np.ndarray) -> None:
        """Add assemblies to index. Vectors are L2-normalized for cosine."""
        norms = torch.norm(assemblies, dim=1, keepdim=True)
        normalized = (assemblies / (norms + 1e-8)).cpu().numpy()

        self.index.add_with_ids(normalized, ids)

        # Store every vector for future rebuild
        for i, id_ in enumerate(ids):
            self._vector_store[int(id_)] = normalized[i].copy()

        self.insertion_count += len(ids)

        if self.insertion_count >= self.rebuild_threshold:
            self.rebuild()

    def remove(self, id_to_remove: int) -> None:
        """Mark for deletion (tombstone). Also remove from vector store."""
        self.tombstones.add(id_to_remove)
        self._vector_store.pop(id_to_remove, None)

    def rebuild(self) -> None:
        """Rebuild index from vector store, excluding tombstoned entries.

        O(n) but runs only during sleep phases — acceptable cost.
        """
        valid_ids = [id_ for id_ in self._vector_store
                     if id_ not in self.tombstones]

        if not valid_ids:
            self.index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.dim, 32))
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 128
            self.insertion_count = 0
            self.tombstones.clear()
            return

        vectors = np.stack([self._vector_store[id_] for id_ in valid_ids])
        ids_array = np.array(valid_ids, dtype=np.int64)

        # Rebuild fresh index
        new_index = faiss.IndexIDMap(faiss.IndexHNSWFlat(self.dim, 32))
        new_index.hnsw.efConstruction = 200
        new_index.hnsw.efSearch = 128
        new_index.add_with_ids(vectors, ids_array)

        self.index = new_index
        self.insertion_count = 0
        self.tombstones.clear()

    def search(self, query: torch.Tensor, k: int = 5
               ) -> Tuple[List[List[int]], np.ndarray]:
        """Return k nearest assembly IDs and distances. O(log n)."""
        if self.index.ntotal == 0:
            empty_ids = [[] for _ in range(query.shape[0])]
            empty_dist = np.empty((query.shape[0], 0), dtype=np.float32)
            return empty_ids, empty_dist

        norms = torch.norm(query, dim=1, keepdim=True)
        normalized = (query / (norms + 1e-8)).cpu().numpy()
        D, I = self.index.search(normalized, k + len(self.tombstones))

        # Filter tombstones from results
        valid = [
            [int(i) for i in row
             if i >= 0 and i not in self.tombstones][:k]
            for row in I
        ]
        return valid, D
```

### Hierarchical Column Routing (Two-Stage)

Incoming patterns route via the two-stage protocol:

1. **Stage 1 — HNSW routing:** Select top-k candidate columns (O(log n))
2. **Stage 2 — WTA selection:** Single winner from k candidates (O(k))

This ensures O(log n) routing complexity regardless of network size, enabling million-neuron scales. CPU→GPU data transfer for routing results uses pinned memory and CUDA streams for overlap.

### Memory Budget Analysis

Estimated GPU memory for key scales, including all hidden state:

| Scale | Neurons | Synapses (15%) | Weights (fp16) | Eligibility (fp16, sparse) | STC State | Total GPU (est.) |
|---|---|---|---|---|---|---|
| Small | 10K | 15M | 30 MB | ~15 MB | 120 MB | ~500 MB |
| Medium | 100K | 1.5B | 3 GB | ~150 MB | 12 GB | ~20–30 GB |
| Large | 1M | 150B | 300 GB | ~15 GB | 1.2 TB | Distributed (16+ GPUs) |

**Hidden components now included:**

| Component | Formula | Cost at 100K neurons, 15% density |
|---|---|---|
| Eligibility traces | fp16 × active_neurons × trace_window | 5% × 1.5B × 2B = ~150 MB (sparse) |
| Spike traces (pre+post) | 2 × n_neurons × trace_window | 2 × 100K × 20 × 4B = 16 MB (negligible) |
| Homeostatic state | 6 mechanisms × n_neurons | 6 × 100K × 4B = 2.4 MB (negligible) |
| STC state | tag + PRP per synapse (fp16) | 2 × 1.5B × 2B = 6 GB (fp16) |
| Slow buffer (reservoir) | capacity × prototype_dim | 10K × 512 × 4B = 20 MB |
| W_inh (sparse) | 20K × 80K × 20% × ~12B | ~38 MB (vs 6.4 GB dense) |

**Key optimizations:**
- Eligibility traces use fp16 precision (sufficient for learning)
- Sparse eligibility: only neurons that fired have active traces
- STC state can use fp16 (continuous values 0–1)
- Medium scale fits on single A100 80GB with careful memory management

---

## 5. Implementation Structure

```
hecsn/
├── config/
│   ├── __init__.py
│   ├── model_config.py    # HECSNConfig dataclass (includes functional_minute)
│   └── training_config.py # Adaptive sleep schedule
├── data/
│   ├── __init__.py
│   ├── corpus_loader.py   # Streaming character pipeline + file/HF/web loaders
│   ├── rtf_encoder.py     # Rate-Temporal Fusion encoding
│   ├── source_catalog.py  # Semantic expansion of remote HF registries (+ exploratory open-web registries)
│   └── tokenizer.py       # Character-level (no sub-word)
├── core/
│   ├── __init__.py
│   ├── columns.py         # Dynamic Competitive Layer with Kohonen/SOM update
│   ├── context.py         # Context Layer + Binding Layer executable proxies
│   └── surprise.py        # Surprise Monitor (precision-weighted, RPE-based)
├── interaction/
│   ├── __init__.py
│   └── responder.py       # Strict-evidence response synthesis over retrieved memories
├── consolidation/
│   ├── __init__.py
│   ├── memory_store.py    # DualMemoryStore (reservoir + EMA + capture/consolidation state)
├── retrieval/
│   ├── __init__.py
│   ├── hnsw_index.py      # Hierarchical ANN index (CPU, fixed rebuild)
│   └── decoder.py         # Native assembly-window stitching / overlap decode
├── reporting/
│   ├── __init__.py
│   ├── io.py              # Shared JSON artifact writers
│   ├── stage0.py          # Stage-0 CSV + diagnostic plot rendering
│   ├── benchmark_plots.py # Memory, routing, and scale diagnostic plot rendering
│   ├── mechanism_validation.py # Mechanism-validation CSV + diagnostic plot rendering
│   └── autonomy.py        # Autonomy benchmark plot rendering
├── service/
│   ├── __init__.py
│   ├── api.py             # FastAPI app over checkpoint/query/respond/Terminus-runtime flows
│   ├── concepts.py        # Learned concept memory over retrieved routing traces
│   ├── manager.py         # Serialized mutable runtime wrapper + persisted concept/Terminus state
│   ├── schemas.py         # Request/response models for the local service
│   ├── server.py          # Uvicorn launcher for the local service
│   └── benchmark_reports.py # Archived benchmark loader retained for internal validation
├── training/
│   ├── __init__.py
│   ├── bootstrap.py      # PredictiveBootstrap
│   ├── checkpointing.py  # Save/load executable HECSN checkpoints
│   ├── runner_utils.py   # Shared runner seed/setup helpers
│   ├── trainer.py        # Main training loop for the executable scaffold
│   ├── mechanism_validation_runner.py  # Mechanism-validation benchmark runner
│   ├── memory_consolidation_runner.py  # Sequential A/B consolidation benchmark
│   ├── contextual_routing_runner.py  # Contextual-routing benchmark
│   ├── hierarchical_scale_runner.py  # Hierarchical-scale benchmark with sharded routing
│   ├── query_runner.py   # Raw-text input + retrieval + native decode interface
│   ├── autonomy_runner.py # Concept-frontier + gap active information-seeking benchmark
│   └── autonomy_acquisition_runner.py # Active acquisition of unseen source banks with concept-aware scoring
├── web/
│   ├── package.json       # React/Vite operator interface
│   ├── vite.config.js     # Frontend dev/build config
│   └── src/
│       ├── App.jsx        # Live chat, concept/Terminus controls, telemetry, checkpoint controls
│       ├── main.jsx       # Frontend bootstrap
│       └── index.css      # Motion + visual system for the local operator UI
└── monitoring/

Implementation note:
- The executable scaffold currently available in this repository covers config, encoder, competitive routing, surprise modulation, memory drift tracking, memory-consolidation benchmarking, a runnable contextual-routing path, a runnable autonomy benchmark for concept-frontier-aware source selection, a runnable acquisition benchmark for selecting which unseen source bank to ingest next under `src/hecsn/`, and a maintained meaning-grounding benchmark over episode-level evidence retrieval, with maintained HF source banks and exploratory curated open-web URL source banks routed through the unified corpus loader plus semantic expansion over remote source registries.
- The executable scaffold now also supports checkpoint save/load plus a direct raw-text query path: `checkpointing.py` persists the current network state; the maintained `mechanism_validation_runner.py`, `memory_consolidation_runner.py`, `contextual_routing_runner.py`, `hierarchical_scale_runner.py`, `autonomy_runner.py`, `autonomy_acquisition_runner.py`, and `meaning_grounding_runner.py` can emit or evaluate queryable state; and `query_runner.py` can load that state, feed additional unlabeled text online, retrieve the nearest stored memory windows plus larger `memory_episodes` for a new text query, emit an initial native assembly-to-text decode summary by stitching overlapping remembered windows, or compare the same query under two different context primes.
- The repository now also includes a local interactive surface: `interaction/responder.py` implements a strict-evidence response mode over retrieved memories, now preferring complete sentence-level episode evidence and abstaining when support has zero lexical grounding, plus a guarded native-decode response mode when the stitched continuation is confident enough; `service/api.py` + `service/manager.py` wrap checkpoint-backed HECSN state behind serialized FastAPI endpoints for status, feed, query, respond, checkpoint save/restore, persisted recent turn traces under `reports/service/traces/`, a live `/stream/status` telemetry feed, and a checkpoint-persisted Terminus control plane (`/terminus`, `/terminus/configure`, `/terminus/start`, `/terminus/stop`, `/terminus/tick`) that continuously feeds unlabeled character streams into the active checkpoint.
- `semantics/concepts.py` now implements a shared learned concept-memory layer over retrieved routing traces: query/respond flows attach `concept_summary`, concepts are grouped in slow-feature space rather than by raw lexical clustering, summaries expose uncertainty/drift/temporal-coherence/abstraction metrics, and `service/manager.py` persists this concept state through service-managed checkpoint save/restore.
- `web/` now provides a React/Vite operator interface for chat, evidence inspection, checkpoint control, live telemetry, concept-summary/concept-grounding inspection, and Terminus runtime control over continuously learned source streams; the concept panels now surface learned-cluster support, observation count, uncertainty, and drift.
- Reporting and artifact rendering now live under `src/hecsn/reporting/` instead of inside the training runners. The runner files are now primarily responsible for corpus loading, training/evaluation orchestration, checkpoint export, and CLI handling, while plot generation and JSON/CSV artifact formatting are dispatched to shared reporting helpers.
- The current runtime still uses phenomenological proxies for molecular STC, full attractor circuitry, and binding microcircuits; however, the consolidation proxy is now explicit capture/replay/consolidation state rather than a single replay tag, and the service concept layer now uses an online slow-feature abstraction over retrieved routing signatures rather than a purely transient grouping heuristic or a raw lexical clusterer.
- Production code must route device selection through config and avoid hardcoded `'cuda'` literals.

> **Reference-Design Appendix**
>
> The Step 0-11 code blocks below are design sketches for the planned full circuit, not the current executable mainline under `src/hecsn/`. When these sketches disagree with the runnable scaffold, the scaffold and the correction log in Section 6.1 take precedence.

### Reference Step 0: AdEx Neuron Model

```python
# core/neuron.py

import torch


class AdExNeuron:
    """Adaptive Exponential Integrate-and-Fire neuron.

    State per neuron: membrane voltage V, adaptation variable w.
    Supports bursting and spike-frequency adaptation.
    Two state variables — tractable at 100K+ scale.

    Parameters (Brette & Gerstner 2005 standard values):
    C_m: membrane capacitance (200 pF)
    g_L: leak conductance (10 nS)
    E_L: leak reversal (-70 mV)
    V_T: threshold (-50 mV)
    delta_T: slope factor (2 mV) ← exponential upswing
    tau_w: adaptation time constant (100 ms)
    a: subthreshold adaptation (2 nS)
    b: spike-triggered adaptation increment (0 pA for tonic,
       ~100 pA for bursting)
    V_reset: reset voltage (-58 mV)
    V_peak: spike cutoff (20 mV)
    """

    def __init__(self, n_neurons: int, dt: float = 0.5,
                 device: str = 'cuda', burst_mode: bool = True):
        self.n = n_neurons
        self.dt = dt  # ms — 0.5ms for balance of accuracy and speed
        self.device = device

        # Biophysical parameters
        self.C_m = 200e-3  # nF (scaled for mV/ms units)
        self.g_L = 10e-3   # µS
        self.E_L = -70.0   # mV
        self.V_T = -50.0   # mV
        self.delta_T = 2.0 # mV
        self.tau_w = 100.0 # ms
        self.a = 2e-3      # µS
        # b=0: tonic spiking; b=80e-3: bursting — matches RTF requirement
        self.b = 80e-3 if burst_mode else 0.0
        self.V_reset = -58.0
        self.V_peak = 20.0
        self.V_spike = 0.0  # mV recorded as spike time marker

        # State vectors (one value per neuron)
        self.V = torch.full((n_neurons,), self.E_L, device=device)
        self.w = torch.zeros(n_neurons, device=device)
        self.spike_times = torch.full((n_neurons,), -1.0, device=device)

    def step(self, I_syn: torch.Tensor, t: float) -> torch.Tensor:
        """Advance one timestep using exponential Euler for stability.

        I_syn: [n_neurons] synaptic current in pA

        Uses exponential Euler for the leak term (analytically stable)
        and forward Euler for adaptation and exponential upswing.
        """
        # Exponential term with tighter clamp for stability
        exp_term = self.delta_T * torch.exp(
            torch.clamp((self.V - self.V_T) / self.delta_T, max=5.0)
        )

        # Exponential Euler for leak: V_inf = E_L + (exp_term - w + I_syn) / g_L
        V_inf = self.E_L + (exp_term - self.w + I_syn) / self.g_L
        exp_factor = torch.exp(-self.g_L / self.C_m * self.dt)

        # Voltage update: V = V_inf + (V - V_inf) * exp_factor
        self.V = V_inf + (self.V - V_inf) * exp_factor

        # Adaptation update (forward Euler)
        dw = ((self.a * (self.V - self.E_L) - self.w) / self.tau_w) * self.dt
        self.w = self.w + dw

        # Spike detection — stays on GPU, no sync
        spikes = (self.V >= self.V_peak)
        spikes_float = spikes.float()

        # Fully vectorized reset
        self.V = torch.where(spikes, torch.full_like(self.V, self.V_reset), self.V)
        self.w = self.w + spikes_float * self.b

        # Update spike times
        self.spike_times = torch.where(
            spikes,
            torch.full_like(self.spike_times, t),
            self.spike_times
        )

        return spikes

    def reset_state(self) -> None:
        self.V.fill_(self.E_L)
        self.w.zero_()
        self.spike_times.fill_(-1.0)

    @classmethod
    def inhibitory(cls, n_neurons: int, dt: float = 0.5,
                   device: str = 'cuda') -> 'AdExNeuron':
        """Create fast-spiking inhibitory neuron population.

        Parameters tuned for PV+ interneuron dynamics:
        - No adaptation (b=0, a=0): tonic fast spiking
        - Shorter membrane time constant (C_m/g_L ~ 10ms vs 20ms for exc)
        - Lower threshold (-45mV vs -50mV for exc)

        Reference: Brette & Gerstner (2005), Wang & Buzsáki (1996).

        Inhibitory neurons must be simulated separately for iSTDP.
        """
        neuron = cls(n_neurons, dt=dt, device=device, burst_mode=False)
        # Fast-spiking PV+ parameters
        neuron.C_m = 100e-3  # nF — smaller = faster (tau_m = C/g_L = 10ms)
        neuron.g_L = 10e-3  # µS
        neuron.V_T = -45.0  # mV — lower threshold
        neuron.tau_w = 20.0  # ms — faster adaptation
        neuron.a = 0.0  # No subthreshold adaptation
        neuron.b = 0.0  # No spike-triggered adaptation (tonic)
        neuron.V_reset = -65.0  # mV
        return neuron


# Simulation timestep configuration
# config/model_config.py

from dataclasses import dataclass, field


@dataclass
class HECSNConfig:
    """Configuration dataclass with enforced architectural constraints."""

    # Fixed architectural constants
    N_ASCII: int = 128  # Fixed ASCII range — do not change
    window_size: int = 10  # Character window size
    input_dim: int = field(init=False)  # Routing input dimension under the maintained feature_vec contract

    # Network size
    n_neurons: int = 1_000  # Stage 0 default
    n_columns: int = 10  # Stage 0 default
    n_context: int = 200  # Context Layer neurons
    memory_capacity: int = 1_000  # Slow buffer capacity

    # Latent dimension for column prototypes
    column_latent_dim: int = 256  # Projects input_dim -> latent_dim (dimensionality reduction allowed)

    # Simulation
    dt: float = 0.5  # ms per timestep
    T_per_token: float = 25.0  # ms simulated per character
    context_tokens: int = 15  # Context window in tokens
    device: str = 'cuda'

    # Learning
    connectivity: float = 0.1  # Excitatory connection density
    inh_connectivity: float = 0.2
    functional_minute: int = 500  # Tokens per functional minute
    drift_threshold: float = 0.05
    sleep_ratio: float = 0.15
    k_routing: int = 10
    ema_alpha: float = 0.01

    # Derived constraints — not settable
    column_dim: int = field(init=False)
    n_bindings: int = field(init=False)

    def __post_init__(self):
        self.input_dim = self.N_ASCII
        self.column_dim = self.column_latent_dim  # Prototypes operate in latent space
        self.n_bindings = self.n_columns
        assert self.n_neurons > self.n_columns
        assert self.dt <= self.T_per_token
        assert self.column_latent_dim > 0  # Dimensionality reduction is allowed
```

### Reference Step 1: Model Wiring and Spike Loop

```python
# core/model.py

import torch
from typing import Any, Dict


class HECSNModel:
    """Wires all HECSN components into a single forward-passable model.

    This class owns all component instances and exposes them under
    the names expected by HECSNTrainer.

    This reference forward pass includes a full spike simulation loop connecting
    AdEx neurons to synaptic weights and STDP learning.
    """

    def __init__(self, config: Any):
        n_exc = int(config.n_neurons * 0.8)
        n_inh = int(config.n_neurons * 0.2)

        from core.neuron import AdExNeuron
        from core.columns import CompetitiveColumnLayer
        from core.attractor import ContextLayer
        from core.binding import BindingLayer
        from core.surprise import SurpriseMonitor
        from consolidation.memory_store import DualMemoryStore
        from plasticity.log_stdp import LogSTDP
        from plasticity.istdp import InhibitorySTDP
        from retrieval.hnsw_index import HierarchicalAssemblyIndex

        self.config = config
        self.n_exc = n_exc
        self.n_inh = n_inh

        self.exc_neurons = AdExNeuron(n_exc, dt=config.dt, device=config.device)
        self.inh_neurons = AdExNeuron.inhibitory(n_inh, dt=config.dt, device=config.device)

        self.competitive = CompetitiveColumnLayer(
            n_columns=config.n_columns,
            column_dim=config.column_dim,
            input_dim=config.input_dim,
            k_routing=config.k_routing,
        )
        self.columns = [self.competitive]

        self.hnsw_index = HierarchicalAssemblyIndex(
            dim=config.column_dim,
            rebuild_threshold=1000,
        )

        # Seed routing index with initial prototypes to avoid empty-index failures.
        import numpy as np
        initial_ids = np.arange(config.n_columns, dtype=np.int64)
        self.hnsw_index.add(self.competitive.prototypes.detach(), initial_ids)

        self.log_stdp = LogSTDP(n_exc, n_exc, density=config.connectivity)
        self.istdp = InhibitorySTDP(n_inh, n_exc, density=config.inh_connectivity)

        # Feedforward excitation->inhibition drive with static sparse topology.
        n_ei_edges = int(n_exc * n_inh * 0.1)
        self.W_ei_pre = torch.randint(0, n_exc, (n_ei_edges,), device=config.device)
        self.W_ei_post = torch.randint(0, n_inh, (n_ei_edges,), device=config.device)
        self.W_ei_values = torch.rand(n_ei_edges, device=config.device) * 0.01

        self.context = ContextLayer(
            n_neurons=config.n_context,
            n_input=config.n_columns,
            T_per_token=config.T_per_token,
            context_tokens=config.context_tokens,
        )

        self.binding = BindingLayer(
            n_bindings=config.n_columns,
            n_columns=config.n_columns,
        )

        self.surprise = SurpriseMonitor(
            layer_names=['competitive', 'context', 'binding'],
        )

        self.memory_store = DualMemoryStore(
            capacity=config.memory_capacity,
            ema_alpha=config.ema_alpha,
        )

        # Firing rate estimation for homeostasis
        self.firing_rate_ema = torch.zeros(n_exc, device=config.device)
        self.firing_rate_tau = 1000.0  # ms
        self.last_exc_spikes = torch.zeros(n_exc, device=config.device)
        self.last_inh_spikes = torch.zeros(n_inh, device=config.device)

        # Assembly-to-latent projection so competitive updates follow spike-derived assemblies
        self.W_assembly_project = torch.randn(
            config.n_columns,
            config.column_dim,
            device=config.device,
        ) * 0.01

    def simulate_timestep(self, t: float,
                          input_current_exc: torch.Tensor,
                          modulator: float,
                          dt: float) -> None:
        """Run one simulation timestep: current injection → spikes → STDP.

        This connects AdEx neurons to the weight matrices and learning rules.
        """
        # Recurrent currents are based on the previous timestep spikes.
        I_syn_exc = self.log_stdp.compute_synaptic_current(self.last_exc_spikes)

        # iSTDP inhibition also uses previous inhibitory spikes.
        I_inh = self.istdp.compute_inhibition(self.last_inh_spikes)

        I_total = I_syn_exc - I_inh + input_current_exc

        exc_spikes = self.exc_neurons.step(I_total, t)

        # Inhibitory population receives structured sparse E->I feedforward drive.
        inh_drive = torch.zeros(self.n_inh, device=self.config.device)
        inh_drive.scatter_add_(
            0,
            self.W_ei_post,
            self.W_ei_values * self.last_exc_spikes[self.W_ei_pre],
        )
        inh_spikes_tensor = self.inh_neurons.step(inh_drive, t)

        # Update firing rate estimate (exponential moving average)
        spike_rate = exc_spikes.float() / dt * 1000  # Hz
        self.firing_rate_ema += dt / self.firing_rate_tau * (spike_rate - self.firing_rate_ema)

        # Three-factor STDP: pre=previous spikes, post=current spikes.
        self.log_stdp.update(
            self.last_exc_spikes,
            exc_spikes.float(),
            modulator,
            self.firing_rate_ema,
            dt,
        )
        self.istdp.update(inh_spikes_tensor.float(), self.firing_rate_ema, dt)

        self.last_exc_spikes = exc_spikes.float()
        self.last_inh_spikes = inh_spikes_tensor.float()

    def forward(self, routing_vec: torch.Tensor,
                rtf_vec: torch.Tensor,
                modulator: float,
                is_bootstrap: bool = False,
                n_timesteps: int = 50) -> Dict[str, Any]:
        """Single token forward pass for the reference full spike simulation.

        Args:
            routing_vec: canonical encoder feature vector
            rtf_vec: [128, n_bursts_max] full spike train for neurons
            modulator: current plasticity gate from SurpriseMonitor
            is_bootstrap: if True, allow O(n) fallback in compete()
            n_timesteps: number of simulation steps (default 50 = 25ms/0.5ms)

        Returns:
            dict with assembly, context_gain, binding_act, winner_indices
        """
        # Spike simulation loop
        dt = self.config.dt

        # Convert RTF event map to a fixed external current vector for this token.
        # A full implementation can replace this with a learned encoder projection.
        token_drive = torch.zeros(self.n_exc, device=self.config.device)
        if rtf_vec.numel() > 0:
            char_drive = (rtf_vec >= 0).float().sum(dim=1)
            n = min(char_drive.shape[0], token_drive.shape[0])
            token_drive[:n] = char_drive[:n]

        for step in range(n_timesteps):
            t = step * dt
            self.simulate_timestep(t, token_drive, modulator, dt)

        # Derive assembly from final firing rates (time-averaged)
        # Each column maps to a sub-population of excitatory neurons
        neurons_per_column = self.n_exc // self.config.n_columns
        firing_rates = self.firing_rate_ema
        assembly = torch.zeros(self.config.n_columns, device=self.config.device)
        for col in range(self.config.n_columns):
            start = col * neurons_per_column
            end = min((col + 1) * neurons_per_column, self.n_exc)
            assembly[col] = firing_rates[start:end].mean()

        # Contract: route using assembly-derived latent key
        routing_key = torch.mv(self.W_assembly_project.t(), assembly)
        routing_key = F.normalize(routing_key, dim=0)

        # Stage 1: HNSW routing (CPU)
        candidate_list, _ = self.hnsw_index.search(
            routing_key.unsqueeze(0), k=self.config.k_routing
        )
        candidates = torch.tensor(candidate_list[0], device=self.config.device) \
            if candidate_list and candidate_list[0] else None

        # Stage 2: WTA selection
        winner_indices, strengths, _ = self.competitive.compete(
            routing_key, candidates,
            fallback_allowed=is_bootstrap
        )

        # Update prototypes based on spike-derived assembly, projected to latent space.
        assembly_latent = torch.mv(self.W_assembly_project.t(), assembly)
        assembly_latent = torch.nn.functional.normalize(assembly_latent, dim=0)
        self.competitive.process(assembly_latent, winner_indices, modulator)

        # Context integration
        context_gain = self.context.step(assembly)
        binding_act = self.binding.detect_conjunctions(assembly)

        return {
            'assembly': assembly,
            'context_gain': context_gain,
            'binding_act': binding_act,
            'winner_indices': winner_indices,
        }
```

### Reference Step 2: Rate-Temporal Fusion Encoder

```python
# data/rtf_encoder.py

import torch
import math


class RTFEncoder:
    """Rate-Temporal Fusion encoding for character sequences.

    Produces two aligned views from the same character window:
    1) feature vector [128] for routing/competitive learning
    2) spike-time tensor [128, n_bursts_max] for the reference neuron simulation
    """

    def __init__(self, t_max: float = 20.0, n_bursts_max: int = 5,
                 window_size: int = 10, dt: float = 1.0):
        self.t_max = t_max
        self.n_bursts_max = n_bursts_max
        self.window_size = window_size
        self.dt = dt

        # Position-to-time mapping: position 0 fires at t=0, position 9 at t=9*dt_spacing
        # This ensures earlier characters spike earlier, preserving order
        self.t_spacing = t_max / (window_size + 1)

    def character_window_to_pattern(self, char_sequence: list[int],
                                     window_size: int = 10) -> torch.Tensor:
        """Convert a character window to a simple frequency-based feature vector [128]."""
        window = char_sequence[-window_size:]
        pattern = torch.zeros(128, dtype=torch.float32)
        if len(window) == 0:
            return pattern

        for c in window:
            if 0 <= c < 128:
                pattern[c] += 1.0

        return pattern / max(1, len(window))

    def routing_vector(self, char_sequence: list[int]) -> torch.Tensor:
        """Example routing representation for the reference circuit sketch."""
        return self.character_window_to_pattern(char_sequence, self.window_size)

    def encode(self, char_sequence: list[int],
               context_confidence: float) -> torch.Tensor:
        """Encode character window as spike train with order-aware latency.

        Each character fires at time t = position * t_spacing.
        Earlier positions = earlier spikes = preserved sequential order.

        Returns: spike_times [128, n_bursts_max] with actual spike times.
        """
        window = char_sequence[-self.window_size:]
        while len(window) < self.window_size:
            window = [0] + window

        spike_times = torch.full((128, self.n_bursts_max), -1.0, dtype=torch.float32)

        for pos, c in enumerate(window):
            if 0 <= c < 128:
                # Latency encoding: earlier position → earlier spike
                first_spike_time = pos * self.t_spacing

                # Burst count depends on context confidence
                n_spikes = max(1, int(self.n_bursts_max * context_confidence))

                for i in range(n_spikes):
                    spike_times[c, i] = first_spike_time + i * 3.0

        return spike_times

    def compute_complexity(self, char_sequence: list[int]) -> float:
        """Compute complexity based on unique characters in window."""
        window = char_sequence[-self.window_size:]
        unique = len(set(window))
        return unique / self.window_size  # normalized complexity
```

### Reference Step 3: Competitive Column Formation (Kohonen/SOM Update)

```python
# core/columns.py

import torch
import torch.nn.functional as F
from typing import Tuple


class CompetitiveColumnLayer:
    """Competitive Layer: Emergent columns via competitive learning.

    Self-organizing representation formation without pre-computed statistics.
    Uses Kohonen/SOM competitive update, NOT Oja's PCA rule.

    Projects input_dim to column_latent_dim for abstraction.
    Two-stage selection protocol:
    1. HNSW routing → top-k candidates
    2. WTA inhibition → single winner
    """

    def __init__(self, n_columns: int, column_dim: int,
                 input_dim: int = 128,  # example feature-vector input
                 k_routing: int = 10, n_winners: int = 1):
        self.n_columns = n_columns
        self.column_dim = column_dim  # latent dimension (e.g., 256)
        self.input_dim = input_dim
        self.k_routing = k_routing
        self.n_winners = n_winners
        self.update_count = 0

        # Random projection from input_dim to latent_dim (fixed, not learned)
        # This provides the abstraction: input -> latent space
        self.W_project = torch.randn(input_dim, column_dim, device='cuda') * 0.01
        # Normalize projection columns for stable routing magnitudes
        self.W_project = F.normalize(self.W_project, dim=0) * 0.1

        # Column prototypes in latent space (learned via competition)
        self.prototypes = torch.randn(n_columns, column_dim, device='cuda')
        self.prototypes = F.normalize(self.prototypes, dim=1)

        # Adaptive thresholds for intrinsic plasticity
        self.thresholds = torch.ones(n_columns, device='cuda') * 0.5
        self.target_firing_rate = 0.05

        # Learning rate schedule
        self.lr_initial = 0.01
        self.lr_decay = 1e-6
        self.max_weight_norm = 10.0

    def project_input(self, input_vec: torch.Tensor) -> torch.Tensor:
        """Project input to latent space for routing."""
        latent = torch.mv(self.W_project.t(), input_vec)
        return F.normalize(latent, dim=0)

    def get_lr(self) -> float:
        """Annealed learning rate for stability."""
        return self.lr_initial / (1 + self.lr_decay * self.update_count)

    def compete(self, input_vec: torch.Tensor,
                candidate_indices: torch.Tensor = None,
                fallback_allowed: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Competitive selection of columns with two-stage protocol.

        Stage 1: If candidate_indices provided (from HNSW), use those.
        Otherwise, compute similarities over all columns.
        Stage 2: WTA among candidates → single winner.

        fallback_allowed controls whether O(n) fallback is
        permitted. Set True only during bootstrap (token_count < 5000).
        At scale, fallback is never acceptable.

        Args:
            input_vec: Pattern vector to route
            candidate_indices: Pre-computed candidates from HNSW (or None)
            fallback_allowed: If True, permit O(n) fallback when HNSW
                             has fewer than k_routing entries

        Returns: (winner_indices, activation_strengths, candidate_indices)
        """
        comp_vec = input_vec
        if input_vec.shape[0] != self.column_dim:
            comp_vec = self.project_input(input_vec)
        else:
            comp_vec = F.normalize(input_vec, dim=0)

        if candidate_indices is not None and len(candidate_indices) > 0:
            # Stage 1 already done (HNSW provided candidates)
            prototypes_subset = self.prototypes[candidate_indices]
            similarities = torch.mm(
                comp_vec.unsqueeze(0), prototypes_subset.t()
            ).squeeze(0)
            indices_in_subset = torch.arange(
                len(candidate_indices), device=comp_vec.device
            )
        else:
            # Explicit cold-start policy with error
            if not fallback_allowed:
                raise RuntimeError(
                    "compete() called with no candidates and fallback disabled. "
                    "HNSW index may be empty or corrupt. "
                    "Pass fallback_allowed=True only during bootstrap."
                )
            # Explicit cold-start: full search, but logged
            similarities = torch.mm(
                comp_vec.unsqueeze(0), self.prototypes.t()
            ).squeeze(0)
            candidate_indices = torch.arange(
                self.n_columns, device=comp_vec.device
            )
            indices_in_subset = candidate_indices

        # Subtract adaptive thresholds (intrinsic plasticity)
        thresholds_subset = self.thresholds[candidate_indices]
        activations = torch.relu(similarities - thresholds_subset)

        # Vectorized winner selection — no sync
        values, local_indices = torch.topk(activations, min(self.n_winners, len(activations)))
        has_winner = (values.max() > 0).float()

        self.thresholds = torch.where(
            (has_winner < 0.5),
            self.thresholds * 0.99,
            self.thresholds
        )

        global_indices = torch.where(
            has_winner.bool(),
            candidate_indices[local_indices],
            torch.zeros_like(candidate_indices[local_indices])
        )
        strengths = torch.where(
            has_winner.bool(),
            values / (values.sum() + 1e-8),
            torch.ones_like(values)
        )

        return global_indices, strengths, candidate_indices

    def process(self, input_vec: torch.Tensor,
                winner_indices: torch.Tensor,
                modulator: float) -> torch.Tensor:
        """Generate assembly from pre-selected winner.

        Does NOT re-run compete(). Winner is provided by caller,
        ensuring single routing pass per token.

        Args:
            input_vec: Pattern vector
            winner_indices: Pre-computed winners from compete()
            modulator: Neuromodulatory signal for plasticity gating

        Returns:
            assembly: Activation vector over columns
        """
        comp_vec = input_vec
        if input_vec.shape[0] != self.column_dim:
            comp_vec = self.project_input(input_vec)
        else:
            comp_vec = F.normalize(input_vec, dim=0)

        winner_proto = self.prototypes[winner_indices]
        assembly = torch.zeros(self.n_columns, device=comp_vec.device)
        assembly[winner_indices] = torch.cosine_similarity(
            comp_vec.unsqueeze(0), winner_proto, dim=1
        )

        # Vectorized Kohonen update — no .tolist(), no sync
        lr = self.get_lr()
        delta = comp_vec.unsqueeze(0) - winner_proto  # [n_winners, dim]
        update = lr * delta
        norms = torch.norm(update, dim=1, keepdim=True)
        update = update * torch.clamp(self.max_weight_norm / (norms + 1e-8), max=1.0)
        self.prototypes[winner_indices] += update
        self.prototypes[winner_indices] = F.normalize(
            self.prototypes[winner_indices], dim=1
        )
        self.update_count += len(winner_indices)

        return assembly

    def replay_assembly(self, assembly: torch.Tensor) -> None:
        """Drive column activity from a replayed assembly vector.

        During sleep, the Memory Store injects stored assemblies.
        The winning column for this assembly updates its prototype
        using unsupervised STDP (no modulator — pure Hebbian).
        """
        if assembly.shape[0] != self.n_columns:
            return
        winner_idx = int(assembly.argmax().item())
        sleep_lr = self.get_lr() * 0.1
        delta = assembly - self.prototypes[winner_idx]
        self.prototypes[winner_idx] += sleep_lr * delta
        self.prototypes[winner_idx] = F.normalize(
            self.prototypes[winner_idx].unsqueeze(0), dim=1
        ).squeeze(0)

    def strengthen_tagged_synapses(self, capture_strength: float) -> None:
        """Apply STC capture: multiplicatively strengthen tagged weights.

        capture_strength: product of tag * PRP (0–1 range)
        """
        if not hasattr(self, '_tagged_synapse_mask'):
            return
        scale = 1.0 + capture_strength * 0.1
        values = self.W.values() if hasattr(self, 'W') else None
        if values is not None:
            self.W = torch.sparse_csr_tensor(
                self.W.crow_indices(),
                self.W.col_indices(),
                torch.clamp(values * scale, 0.001, 0.999),
                self.W.shape, device='cuda'
            )

    def renormalize_weights(self, target: float = 0.5,
                            power: float = 0.999) -> None:
        """Homeostatic renormalization during sleep.

        w_new = target * (w / target)^power
        Preserves relative weight ordering while preventing saturation.
        """
        if not hasattr(self, 'W'):
            return
        values = self.W.values()
        renorm = target * ((values / (target + 1e-8)) ** power)
        renorm = torch.clamp(renorm, 0.001, 0.999)
        self.W = torch.sparse_csr_tensor(
            self.W.crow_indices(),
            self.W.col_indices(),
            renorm, self.W.shape, device='cuda'
        )

    def structural_plasticity_spike_correlation(
        self, prune_threshold: float = 0.05,
        correlation_threshold: float = 0.3) -> None:
        """Prune weak synapses; form new ones between correlated columns.

        Uses prototype similarity as proxy for spike correlation during sleep.
        Operates on the scatter-based weight representation.
        """
        if not hasattr(self, 'W_values'):
            return

        # Pruning: remove connections below threshold
        keep_mask = self.W_values.abs() > prune_threshold

        self.W_values = self.W_values[keep_mask]
        self.pre_idx = self.pre_idx[keep_mask]
        self.post_idx = self.post_idx[keep_mask]

        # Growth: add connections between highly correlated columns
        # Use prototype similarity as correlation proxy
        if self.n_columns > 1:
            # Compute correlation matrix from prototypes
            corr = torch.mm(self.prototypes, self.prototypes.t())
            corr.fill_diagonal_(0)  # no self-connections

            # Find highly correlated pairs not already connected
            new_pairs = (corr > correlation_threshold).nonzero(as_tuple=False)

            for pair in new_pairs[:100]:  # limit growth per sleep
                pre, post = pair[0].item(), pair[1].item()
                # Check if already connected
                existing = ((self.pre_idx == pre) & (self.post_idx == post)).any()
                if not existing:
                    # Add new connection with small initial weight
                    self.pre_idx = torch.cat([self.pre_idx, torch.tensor([pre], device='cuda')])
                    self.post_idx = torch.cat([self.post_idx, torch.tensor([post], device='cuda')])
                    self.W_values = torch.cat([self.W_values, torch.tensor([0.1], dtype=torch.float16, device='cuda')])

    def get_current_assemblies(self) -> dict:
        """Return current prototype state for drift detection."""
        return {
            'prototypes': self.prototypes.detach().clone(),
            'n_columns': self.n_columns,
            'thresholds': self.thresholds.detach().clone()
        }

    def nearest_prototype_distance(self, input_vec: torch.Tensor) -> float:
        """Return cosine distance to nearest prototype (novelty signal)."""
        comp_vec = input_vec
        if input_vec.shape[0] != self.column_dim:
            comp_vec = self.project_input(input_vec)
        else:
            comp_vec = F.normalize(input_vec, dim=0)
        sims = torch.mv(self.prototypes, comp_vec)
        nearest_sim = sims.max().item()
        return float(1.0 - nearest_sim)

    def update_thresholds(self, firing_rates: torch.Tensor,
                          ip_lr: float = 0.001) -> None:
        """Intrinsic plasticity: adapt thresholds to target firing rate."""
        self.thresholds += ip_lr * (firing_rates - self.target_firing_rate)
        self.thresholds = torch.clamp(self.thresholds, 0.1, 2.0)

    def spawn_column(self, input_vec: torch.Tensor,
                     hnsw_index: 'HierarchicalAssemblyIndex') -> int:
        """Neurogenesis: spawn new column and register in routing index.

        hnsw_index is REQUIRED. A column not registered here is permanently
        unreachable through the two-stage routing protocol.
        """
        new_prototype = input_vec + torch.randn_like(input_vec) * 0.01
        new_prototype = F.normalize(
            new_prototype.unsqueeze(0), dim=1
        ).squeeze(0)

        self.prototypes = torch.cat(
            [self.prototypes, new_prototype.unsqueeze(0)], dim=0
        )
        self.thresholds = torch.cat(
            [self.thresholds,
            torch.tensor([0.5], device=input_vec.device)]
        )
        self.n_columns += 1
        new_id = self.n_columns - 1

        import numpy as np
        hnsw_index.add(
            new_prototype.unsqueeze(0),
            np.array([new_id], dtype=np.int64)
        )

        return new_id
```

### Reference Step 4: Log-STDP with Synaptic Scaling (Vectorized)

```python
# plasticity/log_stdp.py

import torch
import numpy as np
from typing import Optional


class LogSTDP:
    """Log-STDP with sublinear LTD and vectorized synaptic scaling.

    Produces log-normal weight distributions. Uses zero Gaussian rule
    for homeostasis: silent neurons (eta=0) can recover, preventing
    network fragmentation.

    Uses scatter-based weight storage for GPU efficiency. No CSR rebuild.
    Static COO indices (pre_idx, post_idx) + mutable value tensor.
    """

    def __init__(self, n_pre: int, n_post: int, density: float = 0.1):
        self.n_pre = n_pre
        self.n_post = n_post
        n_connections = int(n_pre * n_post * density)

        # Static COO indices - connection topology never changes during update
        self.pre_idx = torch.randint(0, n_pre, (n_connections,), device='cuda')
        self.post_idx = torch.randint(0, n_post, (n_connections,), device='cuda')

        # Deduplicate connections
        pairs = torch.stack([self.pre_idx, self.post_idx]).unique(dim=1)
        self.pre_idx = pairs[0]
        self.post_idx = pairs[1]
        n_connections = self.pre_idx.shape[0]

        # Mutable weight values (fp16 for memory efficiency)
        self.W_values = torch.empty(n_connections, dtype=torch.float16, device='cuda')
        self.W_values.log_normal_(mean=np.log(0.5), std=0.5)
        self.W_values.clamp_(0.01, 0.99)

        # Eligibility traces (fp16)
        self.eligibility = torch.zeros(n_connections, dtype=torch.float16, device='cuda')
        self.tau_e = 200.0

        # Pre/post synaptic traces for STDP timing
        self.pre_trace = torch.zeros(n_pre, device='cuda')
        self.post_trace = torch.zeros(n_post, device='cuda')
        self.tau_trace = 20.0

        # Log-STDP parameters
        self.A_plus = 0.01
        self.A_minus = 0.012
        self.mu_plus = 0.0
        self.mu_minus = 1.0

        # Synaptic scaling — zero Gaussian rule
        self.target_firing_rate = 5.0
        self.scaling_alpha = 0.1
        self.silent_threshold = 0.01

    def compute_synaptic_current(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute I_syn = scatter_add(W_values * pre_spikes[pre_idx], post_idx)."""
        I_syn = torch.zeros(self.n_post, device='cuda')
        I_syn.scatter_add_(0, self.post_idx, self.W_values.float() * pre_spikes[self.pre_idx].float())
        return I_syn

    def update(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
               modulator: float, firing_rates: torch.Tensor,
               dt: float = 1.0) -> None:
        """Log-STDP update with scatter-based weight modification."""
        # Update pre/post traces
        self.pre_trace *= (1 - dt / self.tau_trace)
        self.pre_trace += pre_spikes
        self.post_trace *= (1 - dt / self.tau_trace)
        self.post_trace += post_spikes

        # STDP via traces
        ltp = (self.A_plus
               * (self.W_values.float() ** self.mu_plus)
               * post_spikes[self.post_idx].float()
               * self.pre_trace[self.pre_idx])
        ltd = (self.A_minus
               * (self.W_values.float() ** self.mu_minus)
               * pre_spikes[self.pre_idx].float()
               * self.post_trace[self.post_idx])

        # Update eligibility traces
        self.eligibility *= (1 - dt / self.tau_e)
        self.eligibility += (ltp.half() - ltd.half())

        # Three-factor update: always compute, even at low modulator
        # Initialize from current weights (avoids NameError bug)
        delta_w = 0.01 * modulator * self.eligibility.float()
        new_values = self.W_values.float() + delta_w

        # Synaptic scaling
        is_silent = firing_rates < self.silent_threshold
        raw_scale = (self.target_firing_rate / (firing_rates + 1e-8)) ** self.scaling_alpha
        raw_scale = torch.clamp(raw_scale, 0.5, 2.0)
        scale_per_post = torch.where(is_silent, torch.ones_like(raw_scale), raw_scale)
        connection_scale = scale_per_post[self.post_idx]
        new_values *= connection_scale

        # Soft bound and store
        self.W_values = torch.clamp(new_values, 0.001, 0.999).half()

    def resize_eligibility(self) -> None:
        """Resize eligibility trace to match current number of connections."""
        n_current = self.W_values.shape[0]
        n_trace = self.eligibility.shape[0]

        if n_current == n_trace:
            return

        if n_current > n_trace:
            padding = torch.zeros(n_current - n_trace, dtype=torch.float16, device='cuda')
            self.eligibility = torch.cat([self.eligibility, padding])
        else:
            self.eligibility = self.eligibility[:n_current]
```

### Reference Step 5: Inhibitory STDP

```python
# plasticity/istdp.py

import torch


class InhibitorySTDP:
    """Inhibitory STDP for E/I balance maintenance.

    Updates inhibitory->excitatory synapses to maintain target
    excitatory firing rate. When excitatory neurons fire too much,
    inhibition increases; when too little, inhibition decreases.

    Reference: Vogels et al. (2011), Effenberger et al. (2015) [35].

    Uses sparse COO/CSR representation. Memory: ~38 MB at 100K neurons.
    Stored as W[exc, inh] so compute is direct multiply (no transpose).
    """

    def __init__(self, n_inh: int, n_exc: int, density: float = 0.2):
        self.n_inh = n_inh
        self.n_exc = n_exc
        self.rho_target = 5.0  # Target excitatory firing rate (Hz)
        self.eta_inh = 0.01  # Inhibitory learning rate

        n_connections = int(n_inh * n_exc * density)
        # Indices: row = exc neuron, col = inh neuron
        # Store as W[exc, inh] so that W @ inh_spikes → exc_current
        exc_idx = torch.randint(0, n_exc, (n_connections,))
        inh_idx = torch.randint(0, n_inh, (n_connections,))

        pairs = torch.stack([exc_idx, inh_idx]).unique(dim=1)
        self._coo_indices = pairs  # [exc_idx, inh_idx]

        # Mutable weights with static topology (no CSR rebuild in update())
        self.W_values = torch.ones(pairs.shape[1], device='cuda') * 0.05

        self.pre_trace = torch.zeros(n_inh, device='cuda')
        self.tau_trace = 20.0

    def update(self, inh_spikes: torch.Tensor,
               exc_firing_rates: torch.Tensor,
               dt: float = 1.0) -> None:
        """Update inhibitory weights to track target excitatory rate."""
        self.pre_trace *= (1 - dt / self.tau_trace)
        self.pre_trace += inh_spikes

        error = exc_firing_rates - self.rho_target  # [n_exc]
        row_idx = self._coo_indices[0]  # exc indices
        col_idx = self._coo_indices[1]  # inh indices

        # delta per connection: error[exc] * trace[inh]
        delta_values = self.eta_inh * error[row_idx] * self.pre_trace[col_idx]
        self.W_values = torch.clamp(self.W_values + delta_values, 0.0, 1.0)

    def compute_inhibition(self, inh_spikes: torch.Tensor) -> torch.Tensor:
        """Returns [n_exc] inhibitory current via scatter-add over static COO edges."""
        row_idx = self._coo_indices[0]
        col_idx = self._coo_indices[1]
        current = torch.zeros(self.n_exc, device='cuda')
        current.scatter_add_(0, row_idx, self.W_values * inh_spikes[col_idx].float())
        return current
```

### Reference Step 6: Context Layer with Approximate Attractor

```python
# core/attractor.py

import torch


class ContextLayer:
    """Context Layer: Approximate attractor for temporal integration.

    Uses slow excitatory dynamics with fast inhibitory feedback,
    following Nair et al. (2024) finding that line attractors require
    slow neurotransmission (~20s time constants in biology).

    tau_slow = T_per_token * context_tokens = 375 ms.
    W_rec is a FIXED random reservoir (echo state property).
    """

    def __init__(self, n_neurons: int, n_input: int,
                 T_per_token: float = 25.0, context_tokens: int = 15,
                 tau_fast: float = 10.0, local_density: float = 0.35):
        self.n = n_neurons
        self.n_input = n_input
        self.tau_slow = T_per_token * context_tokens
        self.tau_fast = tau_fast

        # Feedforward projection from competitive layer to context
        self.W_ff = torch.randn(n_neurons, n_input, device='cuda') * 0.01
        ff_norm = torch.linalg.norm(self.W_ff, ord=2)
        if ff_norm > 0:
            self.W_ff *= (0.5 / ff_norm.item())

        self.W_rec = torch.randn(n_neurons, n_neurons, device='cuda') * 0.01
        self.connectivity_mask = (
            torch.rand(n_neurons, n_neurons, device='cuda') < local_density
        ).float()
        self.W_rec *= self.connectivity_mask

        spectral_radius = torch.linalg.norm(self.W_rec, ord=2)
        if spectral_radius > 0:
            self.W_rec *= (0.9 / spectral_radius.item())

        self.W_sst = torch.ones(n_neurons, device='cuda') * 0.1
        self.state = torch.zeros(n_neurons, device='cuda')
        self.inh_state = torch.zeros(n_neurons, device='cuda')

    def step(self, competitive_assembly: torch.Tensor,
             dt: float = 1.0) -> torch.Tensor:
        """Evolve attractor state with SST-mediated inhibition.

        competitive_assembly: [n_columns] real-valued activation (NOT spikes)
        """
        ff_input = torch.mv(self.W_ff, competitive_assembly)
        recurrent_input = torch.mv(self.W_rec, self.state)
        excitatory_drive = ff_input + recurrent_input

        mean_activity = self.state.mean()
        self.inh_state += dt * (
            -self.inh_state / self.tau_fast
            + self.W_sst * mean_activity
        )

        self.state += dt * (
            -self.state / self.tau_slow
            + excitatory_drive
            - self.inh_state
        )
        self.state = torch.relu(self.state)

        # Vectorized soft normalization — no sync
        state_max = self.state.max()
        scale = torch.where(
            state_max > 10.0,
            torch.tensor(10.0, device=self.state.device) / (state_max + 1e-8),
            torch.ones(1, device=self.state.device)
        )
        self.state = self.state * scale

        context_gain = torch.sigmoid(self.state)
        return context_gain
```

### Reference Step 7: Precision-Weighted Surprise Monitor (RPE-Based)

```python
# core/surprise.py

import torch
from collections import deque
from typing import Dict, List


class SurpriseMonitor:
    """Surprise Monitor: Precision-weighted prediction error per layer.

    Implements three-factor learning gate: pre, post, and modulator.
    All neuromodulatory signals are internally derived — no external
    reward signal.

        Dopamine = Reward Prediction Error (RPE), not sliding average.
    """

    def __init__(self, layer_names: List[str],
                 history_len: int = 100):
        self.layers: Dict[str, dict] = {
            name: {
                'errors': deque(maxlen=history_len),
                'precision': 1.0,
            }
            for name in layer_names
        }
        self.history_len = history_len

        # Internally-derived neuromodulator levels
        self.dopamine = 0.5      # RPE signal (signed)
        self.acetylcholine = 0.5 # Input novelty / arousal
        self.norepinephrine = 0.5 # Global surprise magnitude

        # Predicted error baseline for RPE computation
        self.predicted_error = 0.5

    def update(self, layer_name: str, prediction: torch.Tensor,
               actual: torch.Tensor) -> None:
        """Update prediction error and precision for specific layer."""
        error = torch.norm(prediction - actual).item()
        self.layers[layer_name]['errors'].append(error)

        # Update precision (inverse variance)
        errors = self.layers[layer_name]['errors']
        if len(errors) >= 10:
            error_tensor = torch.tensor(list(errors))
            variance = torch.var(error_tensor).item()
            self.layers[layer_name]['precision'] = 1.0 / (variance + 1e-6)

    def get_modulator(self, layer_name: str) -> float:
        """Layer-specific modulator M(t) with precision weighting."""
        errors = list(self.layers[layer_name]['errors'])
        if len(errors) < 10:
            return 0.5

        recent_error = errors[-1]
        mean_error = sum(errors) / len(errors)

        surprise = recent_error - mean_error

        # Normalize precision via sigmoid to bounded [0, 1]
        raw_precision = self.layers[layer_name]['precision']
        precision_weight = torch.sigmoid(
            torch.tensor(0.1 * (raw_precision - 10.0))
        ).item()

        dopamine_factor = self.dopamine * 2 - 1
        ach_factor = self.acetylcholine

        modulator = surprise * precision_weight * dopamine_factor * ach_factor

        if abs(surprise) > 0.1:
            self.norepinephrine = min(1.0, self.norepinephrine + 0.1)
        else:
            self.norepinephrine *= 0.95

        return max(-1.0, min(1.0, modulator))

    def compute_dopamine_rpe(self, current_error: float,
                             predicted_error: float) -> float:
        """Compute dopamine as true Reward Prediction Error.

        RPE = predicted_error - current_error
        Positive RPE (error < predicted): better than expected → LTP boost
        Negative RPE (error > predicted): worse than expected → LTD boost

        Normalized by baseline error magnitude. The tanh input is the
        fractional improvement, not the absolute difference.
        """
        baseline = predicted_error + 1e-6
        fractional_rpe = (predicted_error - current_error) / baseline
        return float(torch.tanh(torch.tensor(fractional_rpe * 3.0)))

    def update_predicted_error(self, actual_error: float,
                                alpha: float = 0.01) -> None:
        """Update the baseline prediction of expected error.

        Slow EMA — this IS where the sliding window belongs:
        as the baseline, not as the signal.
        """
        self.predicted_error = (alpha * actual_error
                                + (1 - alpha) * self.predicted_error)

    def update_neuromodulators(self, current_error: float,
                                novelty: float) -> None:
        """Update neuromodulators from internal prediction dynamics.

        Args:
            current_error: Current prediction error.
            novelty: Distance of current input to nearest prototype.
        """
        # Dopamine = RPE
        self.dopamine = self.compute_dopamine_rpe(
            current_error, self.predicted_error
        )
        self.dopamine = (self.dopamine + 1) / 2  # map [-1,1] to [0,1]

        # Update predicted error baseline
        self.update_predicted_error(current_error)

        # Acetylcholine: novelty / arousal
        self.acetylcholine = 0.9 * self.acetylcholine + 0.1 * novelty
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine))

    def should_reset_network(self) -> bool:
        """Check if norepinephrine-triggered reset is needed."""
        return self.norepinephrine > 0.8
```

### Reference Step 8: Dual Memory Store (Reservoir + EMA)

```python
# consolidation/memory_store.py

import torch


class DualMemoryStore:
    """Two-track memory: slow stable buffer + fast EMA baseline.

        Solves the bias problem in drift-triggered sleep.
    - slow_buffer: reservoir-sampled assemblies from all training history.
    Unbiased by recency. Used for sleep replay; the executable scaffold
    further annotates these memories with capture and consolidation state.
    - fast_ema: EMA of recent assemblies. Used for drift detection
    and novelty baselining, while learned concept memory lives separately
    in service/concepts.py.

        Maintains running mean of slow buffer for O(1) drift
    computation instead of O(capacity) tensor stacking per call.
    """

    def __init__(self, capacity: int, ema_alpha: float = 0.01):
        self.capacity = capacity
        self.ema_alpha = ema_alpha

        # Slow buffer: reservoir sampling (Algorithm R — Vitter 1985)
        # Every assembly has equal probability of being in the buffer
        # regardless of when it was seen. This is unbiased.
        self.slow_buffer: list[torch.Tensor] = []
        self.slow_buffer_importance: list[float] = []
        self.n_seen = 0

        # Fast EMA: biased toward recent, used for drift / novelty baselining
        self.fast_ema: torch.Tensor | None = None

        # Running mean for O(1) drift computation
        self._slow_mean: torch.Tensor | None = None
        self._slow_mean_n: int = 0

    def update(self, assembly: torch.Tensor,
               importance: float = 1.0) -> None:
        """Update both buffers with new assembly."""
        # Update fast EMA (drift / novelty baseline)
        if self.fast_ema is None:
            self.fast_ema = assembly.clone()
        else:
            self.fast_ema = (self.ema_alpha * assembly
                             + (1 - self.ema_alpha) * self.fast_ema)

        # Reservoir sampling for slow buffer (unbiased history)
        self.n_seen += 1
        if len(self.slow_buffer) < self.capacity:
            self.slow_buffer.append(assembly.clone())
            self.slow_buffer_importance.append(importance)

            # Update running mean incrementally
            if self._slow_mean is None:
                self._slow_mean = assembly.clone()
                self._slow_mean_n = 1
            else:
                self._slow_mean_n += 1
                alpha = 1.0 / self._slow_mean_n
                self._slow_mean = (alpha * assembly
                                   + (1 - alpha) * self._slow_mean)
        else:
            # Replace with probability capacity/n_seen
            j = torch.randint(0, self.n_seen, (1,)).item()
            if j < self.capacity:
                # Update running mean approximately
                old = self.slow_buffer[j]
                self.slow_buffer[j] = assembly.clone()
                self.slow_buffer_importance[j] = importance
                self._slow_mean = (self._slow_mean
                                   - old / self.capacity
                                   + assembly / self.capacity)

    def sample_for_replay(self, n: int) -> list[torch.Tensor]:
        """Sample from slow buffer weighted by importance.

        High-importance assemblies replay more often — but ALL
        history is representable, not just recent history.
        """
        if not self.slow_buffer:
            return []
        weights = torch.tensor(self.slow_buffer_importance, dtype=torch.float32)
        weights = torch.clamp(weights, min=1e-6)
        weights = weights / weights.sum()
        n_sample = min(n, len(self.slow_buffer))
        indices = torch.multinomial(weights, n_sample, replacement=False)
        return [self.slow_buffer[i] for i in indices]

    def compute_drift(self) -> float:
        """O(1) drift computation using running mean."""
        if self.fast_ema is None or self._slow_mean is None:
            return 0.0
        return (self.fast_ema - self._slow_mean).norm().item()
```

### Reference Step 9: Adaptive Sleep Replay with STC (Functional Time)

```python
# consolidation/sleep_replay.py

import torch
import numpy as np
from typing import Any, List, Optional


class AdaptiveSleepReplay:
    """Two-tier sleep controller with spaced replay priority.

    Micro-sleep protects recent assemblies cheaply.
    Deep sleep is reserved for scheduled maintenance or a rising drift floor.
    """

    def __init__(self, memory_store: Any, drift_threshold: float = 0.05,
                 micro_interval: int = 200, deep_interval: int = 5000,
                 drift_floor_window: int = 1000,
                 momentum: float = 0.85):
        self.memory_store = memory_store
        self.drift_threshold = drift_threshold
        self.micro_interval = micro_interval
        self.deep_interval = deep_interval
        self.drift_floor_window = drift_floor_window
        self.momentum = momentum

        self.token_counter = 0
        self.micro_sleep_counter = 0
        self.deep_sleep_counter = 0
        self.drift_history: List[float] = []
        self.last_micro_sleep = -10**9
        self.last_deep_sleep = -10**9
        self.previous_drift_floor: Optional[float] = None

    def check_drift(self, current_assemblies: dict) -> float:
        """Compute mean drift rate across assemblies."""
        drift = self.memory_store.compute_drift()
        self.drift_history.append(drift)
        return drift

    def replay_priority(self, importance: float, tokens_since_last_replay: int) -> float:
        return importance * np.log1p(tokens_since_last_replay)

    def should_deep_sleep(self) -> bool:
        if len(self.drift_history) < self.drift_floor_window:
            return False

        drift_floor = min(self.drift_history[-self.drift_floor_window:])
        scheduled = (self.token_counter - self.last_deep_sleep) >= self.deep_interval
        rising_floor = (
            self.previous_drift_floor is not None
            and drift_floor > self.previous_drift_floor
            and drift_floor > self.drift_threshold
        )
        self.previous_drift_floor = drift_floor
        return scheduled or rising_floor

    def sleep_phase(self, columns: list, hnsw_index: Any,
                    duration_ms: float = 100.0,
                    dt: float = 0.5) -> int:
        """Deep sleep: replay + renormalization + index repair.

        Runtime mainline uses the same replay path for micro and deep sleep,
        but deep sleep runs a much larger candidate set and rebuilds HNSW.

        Args:
            duration_ms: Sleep duration in milliseconds
            dt: Timestep in milliseconds (default 0.5ms)
        """
        # Convert ms to steps: 100ms / 0.5ms = 200 steps
        sleep_steps = int(duration_ms / dt)

        # Increase protein synthesis during sleep
        self.protein_synthesis_level = 1.0

        # Sample assemblies from slow buffer (unbiased)
        replay_batch = self.memory_store.sample_for_replay(n=100)

        for step in range(sleep_steps):
            if step % 10 == 0 and len(replay_batch) > 0:
                assembly = replay_batch[
                    np.random.randint(len(replay_batch))
                ]
                for col in columns:
                    col.replay_assembly(assembly)

                    # STC block moved INSIDE the inner loop
                    # (was using 'col' from last iteration — scope bug)
                    col_id = getattr(col, 'id', None)
                    if col_id is not None and col_id in self.synaptic_tags:
                        tag = self.synaptic_tags[col_id]
                        capture = tag * self.protein_synthesis_level
                        if capture > 0.1:
                            col.strengthen_tagged_synapses(capture)

            # Homeostatic renormalization stays at step level (not per-column)
            if step % 5 == 0:
                for col in columns:
                    col.renormalize_weights(
                        target=self.target_weight,
                        power=self.decay_power
                    )

            self.protein_synthesis_level *= 0.995

        # Spike correlation-based structural plasticity
        for col in columns:
            col.structural_plasticity_spike_correlation(
                prune_threshold=0.05,
                correlation_threshold=0.3
            )

        # Repair HNSW index
        hnsw_index.rebuild()

        # Decay tags (using functional time)
        current_fm = self.tokens_to_functional_minutes(self.token_counter)
        expired = []
        for col_id, timestamp in self.tag_timestamps.items():
            age_fm = self.tokens_to_functional_minutes(
                self.token_counter - timestamp
            )
            if age_fm > self.tau_tag_default_fm:
                self.synaptic_tags[col_id] *= 0.95
                if self.synaptic_tags[col_id] < 0.01:
                    expired.append(col_id)

        for col_id in expired:
            del self.synaptic_tags[col_id]
            del self.tag_timestamps[col_id]

        self.sleep_counter += 1
        self.tokens_since_sleep = 0
        return self.sleep_counter

    def tag_synapses(self, column_id: int, strength: float = 1.0) -> None:
        """Tag synapses for later consolidation (STC)."""
        self.synaptic_tags[column_id] = min(1.0, strength)
        self.tag_timestamps[column_id] = self.token_counter

    def increment_token(self) -> None:
        self.token_counter += 1
        self.tokens_since_sleep += 1
```

### Reference Step 10: Binding Layer with Short-Term Plasticity

```python
# core/binding.py

import torch
import math
from typing import List, Tuple


class BindingLayer:
    """Binding Layer: Coincidence detection with short-term plasticity.

    Detects convergent activation from multiple Competitive Layer
    columns. PV+ interneurons provide fast feedforward inhibition.

    n_bindings MUST equal n_columns.
    """

    def __init__(self, n_bindings: int, n_columns: int,
                 threshold: float = 2.5, tau_binding: float = 50.0):
        assert n_bindings == n_columns, (
            f"n_bindings ({n_bindings}) must equal n_columns ({n_columns})."
        )
        self.n_bindings = n_bindings
        self.n_columns = n_columns
        self.threshold = threshold
        self.tau_binding = tau_binding

        self.stp_u_inc = 0.15
        self.stp_tau_f = 1500.0
        self.stp_tau_d = 200.0

        self.x = torch.ones(n_bindings, device='cuda')
        self.u = torch.zeros(n_bindings, device='cuda')

        self.pv_inhibition = torch.tensor(0.0, device='cuda')
        self.pv_threshold = 5.0

        self.recent_inputs: List[Tuple[float, torch.Tensor]] = []
        self.current_time = 0.0

    def update_stp(self, pre_spikes: torch.Tensor,
                   dt: float = 1.0) -> torch.Tensor:
        """Update short-term plasticity variables."""
        self.u += dt * (
            -self.u / self.stp_tau_f
            + self.stp_u_inc * pre_spikes * (1 - self.u)
        )
        release = self.u * self.x * pre_spikes
        self.x += dt * ((1 - self.x) / self.stp_tau_d - release)
        return release

    def detect_conjunctions(self, competitive_activations: torch.Tensor,
                            dt: float = 1.0) -> torch.Tensor:
        """Detect coincidences across Competitive Layer assemblies."""
        self.current_time += dt

        effective_input = self.update_stp(competitive_activations, dt)

        self.recent_inputs.append((self.current_time, effective_input))

        cutoff = self.current_time - self.tau_binding
        self.recent_inputs = [
            (t, act) for t, act in self.recent_inputs if t > cutoff
        ]

        if len(self.recent_inputs) == 0:
            return torch.zeros(self.n_bindings,
                               device=competitive_activations.device)

        weighted_sum = torch.zeros_like(competitive_activations)
        for t, act in self.recent_inputs:
            age = self.current_time - t
            weight = math.exp(-age / (self.tau_binding / 3))
            weighted_sum += weight * act

        # Vectorized PV inhibition — no sync, no .item()
        activity_sum = weighted_sum.sum()
        exceeded = (activity_sum > self.pv_threshold).float()
        self.pv_inhibition = (exceeded * 0.3 * activity_sum
                              + (1 - exceeded) * self.pv_inhibition * 0.9)

        binding_act = torch.relu(
            weighted_sum - self.threshold - self.pv_inhibition
        )
        return (binding_act > 0).float()
```

### Reference Step 11: Complete Training Loop

```python
# training/bootstrap.py

import torch
import torch.nn.functional as F


class PredictiveBootstrap:
    """Minimal predictive coding bootstrap for cold start.

    No vocabulary. Predicts the next routing vector from the current one.

    This is a delta rule (not pure Hebbian) over routing vectors.
    Uses KL divergence for error measurement — appropriate for
    probability distributions.
    """

    def __init__(self, input_dim: int, lr: float = 0.01):
        self.W = torch.zeros(input_dim, input_dim, device='cuda')
        self.lr = lr
        self.prev_pattern: torch.Tensor | None = None

    def update(self, current_pattern: torch.Tensor) -> float:
        """Update predictor and return prediction error.

        Args:
            current_pattern: [input_dim] routing representation vector

        Returns:
            prediction_error: KL divergence between prediction and actual
        """
        if self.prev_pattern is None:
            self.prev_pattern = current_pattern.clone()
            return 0.0

        prediction = F.softmax(torch.mv(self.W, self.prev_pattern), dim=0)

        current_pattern = current_pattern / (current_pattern.sum() + 1e-8)

        # KL divergence: more appropriate for normalized distributions
        error = F.kl_div(
            torch.log(prediction + 1e-8),
            current_pattern,
            reduction='sum'
        ).item()

        # Delta rule update (labeled accurately — not pure Hebbian)
        delta = (current_pattern - prediction).unsqueeze(1)
        self.W += self.lr * torch.mm(delta, self.prev_pattern.unsqueeze(0))
        self.W = torch.clamp(self.W, -1.0, 1.0)

        self.prev_pattern = current_pattern.clone()
        return error
```

```python
# training/trainer.py

import torch
from typing import Any, Dict
from consolidation.sleep_replay import AdaptiveSleepReplay
from training.bootstrap import PredictiveBootstrap


class HECSNTrainer:
    """Main training loop for HECSN.

    Manages bootstrap phase, sleep scheduling, and metric collection.
    All reward/modulation signals are internally derived.

        Uses PredictiveBootstrap without vocab_size.
        Passes winner_indices to process() to avoid double routing.
    """

    def __init__(self, model: Any, config: Any):
        self.model = model
        self.config = config

        self.sleep_manager = AdaptiveSleepReplay(
            model.memory_store,
            drift_threshold=config.drift_threshold,
            sleep_ratio=config.sleep_ratio,
            functional_minute=config.functional_minute,
        )
        # No vocab_size — tabula rasa system
        self.bootstrap = PredictiveBootstrap(input_dim=config.input_dim)

        self.token_count = 0
        self.is_bootstrap = True

    def train_step(self, char: str,
                   pattern_vec: torch.Tensor) -> Dict[str, Any]:
        """Single training step. Returns metrics dict."""
        metrics: Dict[str, Any] = {}

        # Check drift and sleep
        current_assemblies = self.model.competitive.get_current_assemblies()
        drift = self.sleep_manager.check_drift(current_assemblies)

        if self.sleep_manager.should_sleep(drift):
            sleep_n = self.sleep_manager.sleep_phase(
                self.model.columns,
                self.model.hnsw_index,
            )
            metrics['sleep_phase'] = sleep_n
            metrics['drift_triggered'] = (
                drift > self.sleep_manager.drift_threshold
            )

        # Bootstrap phase (first ~5000 tokens)
        if self.token_count < 5000:
            pred_error = self.bootstrap.update(pattern_vec)
            metrics['pred_error'] = pred_error

            # RPE-based dopamine
            self.model.surprise.update_neuromodulators(
                current_error=pred_error,
                novelty=pred_error,
            )
            modulator = self.model.surprise.get_modulator('competitive')
        else:
            self.is_bootstrap = False
            modulator = self.model.surprise.get_modulator('competitive')

        # Two-stage routing
        # Contract: project the current feature vector to a latent key
        routing_key = self.model.competitive.project_input(pattern_vec)

        # Stage 1: HNSW routing for candidate columns
        candidate_indices, _ = self.model.hnsw_index.search(
            routing_key.unsqueeze(0), k=self.config.k_routing
        )
        candidate_indices = (
            torch.tensor(candidate_indices[0], device=self.config.device)
            if candidate_indices and candidate_indices[0] else None
        )

        # Pass fallback_allowed during bootstrap
        winner_indices, strengths, _ = self.model.competitive.compete(
            routing_key, candidate_indices,
            fallback_allowed=self.is_bootstrap
        )

        # Pass winner to process() to avoid double routing
        comp_assembly = self.model.competitive.process(
            routing_key, winner_indices, modulator
        )
        context_gain = self.model.context.step(comp_assembly)
        binding_act = self.model.binding.detect_conjunctions(comp_assembly)

        # Update Memory Store
        self.model.memory_store.update(comp_assembly)

        # Tag synapses for STC if high surprise
        surprise_val = abs(modulator)
        if surprise_val > 0.5:
            for col_id in winner_indices.tolist():
                self.sleep_manager.tag_synapses(
                    col_id, strength=surprise_val
                )

        # Update internal neuromodulators
        layer_errors = []
        for name in self.model.surprise.layers:
            errs = list(self.model.surprise.layers[name]['errors'])
            if errs:
                layer_errors.append(errs[-1])
        if layer_errors:
            mean_err = sum(layer_errors) / len(layer_errors)
            nearest_dist = self.model.competitive.nearest_prototype_distance(
                routing_key
            )
            self.model.surprise.update_neuromodulators(
                current_error=mean_err,
                novelty=min(1.0, nearest_dist),
            )

        # Increment counters
        self.token_count += 1
        self.sleep_manager.increment_token()

        metrics['token'] = self.token_count
        metrics['sparsity'] = (comp_assembly > 0).float().mean().item()
        metrics['surprise'] = modulator
        metrics['drift'] = drift

        return metrics
```

---

## 6. Data Pipeline & Encoding

### Streaming Character Pipeline

HECSN processes text as a raw character stream, not pre-tokenized words. This enables truly emergent representation formation from sub-symbolic patterns.

```python
# data/corpus_loader.py

import gzip
import torch
import math
from typing import Iterator, Tuple


class StreamingCorpusLoader:
    """Streaming character-level data pipeline.

    Supports local files (.txt, .gz) and HuggingFace streaming datasets.
    Character window encoding follows the representation contract:
    feature_vec + optional RTF spike trains.
    """

    def __init__(self, source: str, window_size: int = 10):
        self.source = source
        self.window_size = window_size
        self.token_count = 0

        if source.endswith('.txt') or source.endswith('.gz'):
            self.stream = self._file_stream()
        else:
            from datasets import load_dataset
            dataset = load_dataset(source, split='train', streaming=True)
            self.stream = self._hf_stream(dataset)

    def _file_stream(self) -> Iterator[str]:
        """Stream characters from large text file."""
        opener = gzip.open if self.source.endswith('.gz') else open
        with opener(self.source, 'rt', encoding='utf-8') as f:
            while True:
                char = f.read(1)
                if not char:
                    break
                if char.isprintable() or char.isspace():
                    yield char

    def _hf_stream(self, dataset) -> Iterator[str]:
        """Stream characters from HuggingFace dataset."""
        for example in dataset:
            text = example.get('text', '')
            for char in text:
                if char.isprintable() or char.isspace():
                    yield char

    def char_stream(self) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor, bool]]:
        """Yields (char, pattern_for_routing, rtf_vec, is_bootstrap) tuples.
        """
        window: list = []
        rtf = RTFEncoder(t_max=20.0, n_bursts_max=5)

        for char in self.stream:
            self.token_count += 1
            is_bootstrap = self.token_count <= 5000

            window.append(char)
            if len(window) > self.window_size:
                window.pop(0)

            feature_vec = rtf.routing_vector([ord(c) for c in window])
            context_conf = min(1.0, len(window) / self.window_size)

            rtf_vec = rtf.encode([ord(c) for c in window], context_conf)
            pattern_for_routing = feature_vec  # current feature-vector input

            yield char, pattern_for_routing, rtf_vec, is_bootstrap
            # pattern_for_routing → competitive layer (routing)
            # rtf_vec → optional neuron layer in the reference full circuit
            # Maintained scaffold routes on feature_vec, not a separate unigram-only path
```

---

## 6.1 Correction Log and Current Observations (2026-03-31)

The following corrections are mandatory for implementation validity and now define the canonical path:

1. **C1 (representation contract mismatch) corrected:**
    - Routing/prototype learning now consistently consumes the active encoder `feature_vec`, with `order_weighted_ascii` as the maintained default and `unigram_ascii` / `hashed_ngram` retained as explicit ablations behind the same latent routing interface.
    - Rationale: paper contract and executable path must be identical.

2. **S2 (split teacher path) corrected:**
    - Competitive updates now share the same active feature-vector contract used by routing, rather than a disconnected unigram teacher path.
    - This keeps the executable training path aligned with the runtime query and benchmark path.

3. **C4 pattern (sparse rebuild in update loops) corrected for iSTDP:**
    - iSTDP now uses static sparse topology + mutable value tensors (`W_values`) with scatter-based current computation.
    - No per-step CSR reconstruction in the update path.

4. **Inhibitory drive semantics corrected:**
    - Inhibitory neurons receive structured feedforward `E->I` drive (`W_ei @ exc_spikes`) instead of a global scalar mean drive.
    - This makes inhibitory activity depend on excitatory population structure, consistent with plastic inhibitory pathways [44, 35].

5. **Current Stage-0 empirical observation:**
    - The maintained smoke gate now passes on the executable path (`reports/refactor_stage0_smoke/summary.json`).
    - Passing smoke metrics: `silhouette ~= 0.675`, `DBI ~= 0.304`, `winner_entropy ~= 2.574 bits`, and trained-vs-random eval reconstruction `0.0619 < 0.0907`.
    - Interpretation: Stage 0 is now a maintained behavioral regression gate, not an open clustering blocker. Larger-scale replay/drift tuning still matters, but Stage-0 promotion is no longer blocked on the old failure note.

6. **Research alignment notes (open-access cross-check):**
    - Effenberger et al. (2015) supports the claim that STDP + homeostatic mechanisms can stabilize dynamics and produce structured long-tailed synaptic organization [35].
    - Sagodi et al. (2024/2025) supports using approximate attractor language under perturbations rather than idealized perfectly continuous attractors [37].
    - The predictive-coding SNN survey emphasizes explicit error pathways and representational consistency, matching this document's contract hardening direction [43].

7. **Focused Stage-0 execution policy (best-known path):**
    - Mainline smoke benchmark now uses the maintained `order_weighted_ascii` default with active `column_input_weights`, winner-local drift, and the executable Stage-0 behavioral gate.
    - Alternate representations (`unigram_ascii`, `hashed_ngram`) remain supported as ablations, not as the default maintained claim.

8. **Representation benchmark added:**
    - `reports/refactor_representation_smoke/summary.json` now records a maintained smoke comparison between `order_weighted_ascii`, `unigram_ascii`, and `hashed_ngram` against an `OnlineKMeans` baseline on the same raw-window protocol.
    - Current result: all three HECSN competitive-only runs beat the baseline on silhouette, `hashed_ngram` is numerically strongest on this small slice, and `order_weighted_ascii` remains the maintained default because it is the live service and phase-runner path.

These updates align the architecture with the predictive-coding/SNN literature emphasis on explicit error pathways and consistent representation interfaces [43, 35, 44].

---

## 7. Development Roadmap

### Stage 0 — Mechanism Validation (MANDATORY)

**This gate remains mandatory for regressions, but it is no longer the current blocker: the maintained smoke path passes and now serves as the minimum executable floor for later phases.**

```
═══════════════════════════════════════════════════════════════════
STAGE 0 — Mechanism Validation (mandatory before Stage 1)
═══════════════════════════════════════════════════════════════════
Network size: 1,000 neurons, 10 columns
Input: 100,000 characters of plain English text (Project Gutenberg)

Goal: Demonstrate that column prototypes converge to distinct
      character n-gram clusters (measurable via silhouette score
      on prototype vectors).

Success gate:
    - Silhouette score > 0.20 on prototype clustering OR Davies-Bouldin < 1.5
    - Reconstruction error trend negative over last 10,000 tokens
    - Stage-0 ablation superiority: competitive update beats random-assignment baseline

Failure mode:
  If not achieved in 500,000 characters, the competitive learning
  mechanism requires redesign before scaling.

Exit criteria:
  - Complete Stage 0 validation report
  - Pass behavioral tests B1–B4 (see Section 8)
  - Document findings before proceeding to Phase 1
═══════════════════════════════════════════════════════════════════
```

Maintained executable note: the day-to-day regression gate currently runs on `reports/refactor_stage0_smoke/summary.json` (`800` train / `200` eval tokens from `wikitext-103-raw-v1`) and enforces the same behavioral checks. The larger 100k-character target remains a scale-up stress benchmark, not the current blocker.

**Rationale:** No STDP-based SNN has demonstrated word-level concept emergence from raw character sequences without labeled readout or surrogate gradients. This is a research frontier, not a solved engineering problem. Stage 0 validates the mechanism at minimal cost.

### Phase 1: Core Assembly Formation — *Foundation*

Implement Competitive Layer with competitive learning (Kohonen/SOM), log-STDP, excitatory and inhibitory STDP, synaptic scaling, and RTF encoding. Verify log-normal weight distributions (not bimodal).

**Target:** 10K neurons (10 columns x 1K), character-level input, assembly drift rate < 0.05, log-normal weight distribution (kurtosis 3–6).

**Prerequisite:** Stage 0 completed successfully.

**Phase 1 carry-over constraints (from Stage 0):**
- Sleep aggressiveness must be treated as a scaling cost metric, not just a stability knob. Stage-0 pass required frequent replay; at 10K-neuron scale this replay cadence can become a dominant compute cost.
- Report sleep aggressiveness telemetry every run: sleep events per 1K tokens, mean sleep interval (tokens), and replay updates per event.
- Track drift floor, not only drift slope. Compute minimum drift in each 10K-token window and require the floor trend to be non-increasing over time; if floor rises, trigger deep sleep from the drift-floor controller before continuing scale-up.
- Revalidate drift threshold at each scale increment (1K -> 10K -> 50K), because a threshold that is safe at 1K may over-trigger at higher neuron counts.
- Current Phase-1 profile evidence: a 100-column run (10K-neuron equivalent by the reporting assumption) preserved Stage-0 gate pass, but baseline sleep cadence remained expensive at roughly 5 events per 1K tokens. A cheaper schedule with wider cooldown and deeper replay reduced sleep frequency to roughly 2 events per 1K tokens while keeping the gate green in short sweeps.
- Schedule-only tuning was insufficient: widening cooldowns, adaptive replay-depth boosting, and uniform replay mixing all preserved the gate in some short sweeps but did not stabilize the drift floor.
- Current mainline mitigation is structural rather than parametric: micro-sleep every `200` tokens (`10` replay steps from top-`5` spaced-priority assemblies), winner-local drift, replay-count-aware spacing, scheduled deep maintenance with emergency deep sleep driven by closed drift-floor windows, prototype momentum `0.85`, and replay-to-memory blending so consolidation changes the drift proxy instead of only the prototypes. The recovered 100-column baseline uses scheduled deep maintenance every `2500` tokens with `150` replay steps from the top-`100` candidates.
- Temporal-decay sweep result at 1K scale: `slow_mean_decay` values `{0.9995, 0.9999, 0.99995}` all preserved the Stage-0 gate, but drift-floor slope stayed positive and worsened versus the non-decayed baseline. Observed slopes were approximately `+0.00412`, `+0.00207`, and `+0.00190`, versus the prior non-decayed two-tier baseline at `+0.00136`.
- Fixed-token warm-start result at 1K scale: starting slow memory at `5K`, `10K`, or `20K` tokens preserved the Stage-0 gate and reduced average drift, but did not improve drift-floor slope in the first sweeps. This suggests the remaining issue is not just cold-start contamination; a single global mean is still too sensitive to winner-distribution shift.
- Winner-local drift result at 1K scale: with `slow_memory_start_tokens = 0` and winner-local drift enabled, the Stage-0 gate remained fully green, drift-floor slope dropped to approximately `+8.9e-7` (effectively flat), mean drift fell to about `0.0787`, and replay updates fell to about `6.9K`. This is the first tested configuration that fixes the practical blocker without sacrificing Stage-0 validity.
- First full-length Phase-1 decay retest at 100 columns: the original `100K` comparison with `slow_mean_decay = 0.9999` was invalid because a later code audit showed the decay knob had not yet been applied to the winner-local slow traces used by the controller.
- Corrected full-length Phase-1 retest at 100 columns: after wiring `slow_mean_decay` into the winner-local path and increasing scheduled deep replay from `150` to `200` steps, the `100K` run passed the full gate. The corrected profile reached drift mean approximately `0.0433`, drift-floor slope approximately `-8e-8` (decreasing), `silhouette ~= 0.748`, `DBI ~= 0.759`, reconstruction slope approximately `-6.5e-6`, and trained-vs-random reconstruction `0.116 < 0.648`.
- First 10K-scale winner-local-drift profile: the drift blocker stayed solved at 100 columns, with drift-floor slope near-flat at approximately `+1.5e-5` and replay updates only `3.39K`, but clustering fell short (`silhouette ~= 0.127`, `DBI ~= 1.57`) because deep maintenance had become too sparse.
- Recovered 10K-scale baseline: keeping winner-local drift and the hardening changes, but increasing scheduled deep maintenance to every `2500` tokens with `150` replay steps, restored the full gate at 100 columns. The recovered profile reached `silhouette ~= 0.779`, `DBI ~= 0.831`, drift-floor slope approximately `+2.36e-6`, and replay updates about `5.34K` over 50K tokens.
- Recommended Phase-1 baseline at 10K-neuron-equivalent scale: winner-local drift enabled, `slow_mean_decay = 0.9999`, micro-sleep `200/10/top-5`, scheduled deep maintenance every `2500` tokens with `200` replay steps from the top-`100` candidates, replay-count-aware spacing, and emergency deep sleep driven by closed drift-floor windows rather than by the rolling minimum.
- Active input-weight integration update: a direct `column_input_weights` tensor is now part of live competition in `HECSNModelLite`. The first naive `input_weight_blend = 0.10` integration regressed the 100-column / `100K` gate, but reducing the live blend to `0.02` preserved the direct weight path and restored the full gate. The tuned profile reached drift mean approximately `0.0444`, drift-floor slope approximately `-1.78e-6`, `silhouette ~= 0.707`, `DBI ~= 0.616`, reconstruction slope approximately `-1.45e-6`, and trained-vs-random reconstruction `0.092 < 0.600`.
- Scope note on this result: the runnable path in this repository is still the `HECSNModelLite` competitive-routing scaffold, so the corrected `100K` Phase-1 pass validates the active winner-local drift and replay controller at 10K-neuron-equivalent scale. The executable scaffold now also includes active column-specific input weights that directly participate in competition and can be measured in runtime summaries, but it still does not expose the full log-STDP / iSTDP synaptic circuit from the full architecture.
- Runtime reporting note: summary JSON files from the active scaffold now include a `runtime_scope` block that explicitly marks full synaptic weight-distribution validation as unsupported in `HECSNModelLite` while reporting direct distribution statistics for the active `column_input_weights`, plus supporting prototype/projection diagnostics.

### Phase 2: Consolidation & Sleep — *Memory*

Implement DualMemoryStore and adaptive sleep-replay triggered by drift detection with STC consolidation. Test catastrophic forgetting protection: train on Task A, then Task B, verify A is not forgotten after sleep. Verify renormalization maintains log-normal distributions. Test HNSW index repair.

- Current executable Phase-2 mainline now uses an explicit STC-style proxy in the runnable scaffold: the slow buffer tracks capture tags, last-capture timestamps, consolidation level, and consolidation events per memory. Task-boundary tagging raises capture state for recently learned assemblies, replay priority is driven by unmet capture plus replay spacing, and sleep replay converts that capture pressure into increasing consolidation level while anchored winner columns preserve prototype/input-weight states during later wake and replay updates.
- Historical HF Phase-2 baseline in the executable path: `ag_news` (Task A) -> `wikitext-103-raw-v1` (Task B), `10K` train tokens each, `2K` eval tokens, memory capacity `1000`, winner-local drift enabled, micro-sleep `200/10/top-5`, scheduled deep replay `2500/200/top-100`, task-boundary tag strength `3.0`, task-boundary anchor strength `8.0`, boundary deep consolidation `6` cycles, and post-Task-B deep consolidation `12` cycles.
- Historical baseline result from the runnable scaffold: Task-A reconstruction improved from approximately `0.410` after Task A to approximately `0.330` after consolidation, Task-A activation overlap versus the frozen post-Task-A reference was approximately `0.524`, and the executable Phase-2 gate passed under the paper-aligned criteria of `<= 5%` relative Task-A degradation plus `>= 0.50` Task-A activation overlap.
- Higher-budget confirmation status: the same locked baseline was stress-tested at `50K/50K` train tokens with `5K` eval tokens. Reconstruction still improved after consolidation for both tasks, but Task-A overlap dropped to approximately `0.461`, so the executable Phase-2 gate did **not** pass at that budget. In other words, the current Phase-2 path is validated at the baseline budget but not yet robust to the larger sequential stress test.
- Root-cause clarification from the 50K stress test: Task-A overlap after Task B remained approximately `0.620`, but the fixed post-Task-B consolidation block drove it down. Replay-balance retunes and stronger replay-time anchor restoration did not repair that collapse, which means the dominant failure was over-eager late consolidation rather than Task-B wake overwrite.
- Fresh no-final-consolidation 50K-scale confirmation on the current tree: using the same `consolidation_cycles = 0` regime as the maintained scale-robust path, the corrected `50K/50K` rerun now passes again (`reports/phase2_scale_stress_after_metric_fix/summary.json`, `task_a_relative_degradation_after_consolidation = 0.0`, `task_a_overlap_after_consolidation ~= 0.99999`). The immediately prior local fail (`reports/phase2_scale_stress_after/summary.json`) was numerical: tiny negative reconstruction errors had been amplified into false positive relative degradation until prototype distance was clamped non-negative. The current `memory_consolidation_hf_scale_robust` preset is still a maintained `10K/10K` no-final-consolidation variant rather than a stored `50K/50K` preset.
- Exploratory late-consolidation branches (`balanced`, `protected`, `frozen`, and emergency-only `adaptive`) were removed from the runnable Phase-2 surface after they failed the 50K overlap gate. The maintained executable path now keeps only the validated `micro`/`deep` replay schedule plus the explicit no-final-consolidation scale-robust preset.
- Superseded rerun note: an older local rerun (`reports/recheck_phase2_scale_robust_20260331/summary.json`) had reported `task_a_overlap_after_consolidation ~= 0.498` and `task_a_relative_degradation_after_consolidation ~= 0.0620`, but that no longer describes the current tree.
- Current Phase-2 recovery on this tree: aligning `capture_tag_decay` with the documented functional-minute time base restored durable capture pressure through later replay. The post-fix smoke rerun (`reports/phase2_smoke_after/summary.json`) now passes with `task_a_relative_degradation_after_consolidation ~= -0.985` and `task_a_overlap_after_consolidation ~= 0.980`, and the maintained `memory_consolidation_hf_scale_robust` rerun (`reports/phase2_scale_robust_after/summary.json`) now passes clearly with `task_a_relative_degradation_after_consolidation ~= 0.0103` and `task_a_overlap_after_consolidation ~= 0.99999`.
- Scope note on this result: this validates the current executable consolidation path for the `HECSNModelLite` scaffold, including explicit capture-to-consolidation state, replay-priority spacing, and anchored column maintenance, but it is still a phenomenological proxy for full molecular STC and does not yet implement the full log-STDP / iSTDP / PRP state stack described for the long-term architecture.

### Phase 3: Autonomy & Context — *Metacognition*

Implement Surprise Monitor with RPE-based internally-derived neuromodulation. Activate Context Layer approximate attractor with SST+ inhibition and Binding Layer with STP. Network begins context-dependent routing for polysemy disambiguation.

- Current executable Phase-3 mainline adds a runnable multiscale recurrent `ContextLayer` and short-term-plasticity `BindingLayer` around the existing `SurpriseMonitor`: the context state integrates fast/medium/slow traces over column assemblies, emits multiplicative competitive gain, and the binding path tracks coincidence with facilitation/depression-style short-term dynamics plus PV-like inhibition.
- The executable validation path is `contextual_routing_runner.py`, which trains on interleaved HF blocks from `ag_news` and `wikitext-103-raw-v1`, then primes the model with task-specific context windows and measures whether the same probe bank routes differently under Task-A versus Task-B context.
- The maintained Phase-3 runner now also exports checkpoints directly with `--checkpoint-out`, so the same context-enabled model can be reopened later by `query_runner.py` for raw-text probing instead of being usable only inside the benchmark process.
- Current maintained smoke validation on the executable Phase-3 path (`order_weighted_ascii`, contextual routing on, binding-conjunction memory on) passes with context-state separation `1.000`, probe winner switch rate approximately `0.154`, mean probe assembly distance approximately `0.122`, and a direct B3 `bank` lexical-bundle probe family accuracy `0.875` with positive signature margin approximately `0.00028`. On the current maintained path the top-column bank winner sequence no longer separates reliably across contexts, so the signature-level family separation rather than winner-switching is the honest regression point for contextual disambiguation.
- Baseline validation at the larger executable budget also passed: with `10K/10K` interleaved training tokens and `2K` eval tokens, context-state separation remained `1.000`, probe winner switch rate increased to approximately `0.169`, and mean probe assembly distance was approximately `0.071`. This supports that the current contextual-routing effect is not limited to the smoke budget.
- Direct context-conditioned query validation now also exists outside the benchmark metric itself: on a saved Phase-3 smoke checkpoint, `query_runner.py` with `--compare-context-a/--compare-context-b` routed the same raw query `union` to different winner columns (`48` under a news-like prime and `9` under an encyclopedia-like prime). This is still a proxy-scale demonstration, but it confirms that contextual routing can be inspected through the maintained query interface rather than only through offline evaluation code.
- The missing autonomy piece is now executable through `autonomy_runner.py`: the runner reserves a small probe bank per HF source, measures source-level reconstruction-error gaps on those probes, and then compares an active source-selection policy against equal-budget round-robin over the same source bank. The maintained controller is no longer only residual-gap-driven: the runner now also derives concept-frontier novelty and concept uncertainty from the current memory trace, logs those values per source, and emits an `info_gain_score` that augments the older diagnostic gap score during active selection.
- Current maintained smoke autonomy result on this tree: with explicit `ag_news`, `wikitext-103-raw-v1`, and `imdb`, `500`-token seek episodes, `128` probe tokens per source, and `8` adaptive seek episodes after one warmup round, the active controller again finishes with lower residual gaps than round-robin (`reports/phase3_autonomy_smoke_after/summary.json`, `final_mean_gap ~= 1.57e-5` vs `2.55e-5`, `final_max_gap ~= 1.57e-5` vs `2.55e-5`, `autonomy_gate_pass = true`).
- Current maintained larger-budget autonomy result on this tree: with the `autonomy_hf_baseline` preset (`1000`-token seek episodes, `256` probe tokens per source, `9` adaptive seek episodes, `coverage_balance_penalty = 0.02`, `gap_focus_margin = 0.02`), the active controller also again beats equal-budget round-robin (`reports/phase3_autonomy_baseline_after/summary.json`, `final_mean_gap ~= 9.23e-7` vs `1.37e-6`, `final_max_gap ~= 9.29e-7` vs `1.39e-6`, `autonomy_gate_pass = true`).
- Regression root cause and recovery on the current tree: the maintained autonomy presets had drifted from the intended diverse `news/wiki/reviews` trio to a registry-backed `wiki/dbpedia/news` bank, and the old fixed `-0.01` improvement gate no longer matched the present `1e-5`/`1e-6` gap regime. Restoring the explicit maintained source bank and using a scale-aware improvement target (3% of the round-robin gap, capped at `0.01`) recovered both smoke and baseline autonomy gates on fresh live reruns.
- Dynamic source acquisition is now also executable through `autonomy_acquisition_runner.py`: the runner first trains on a seed source bank, then derives a frontier gap plan from that seed-conditioned model state, expands any configured semantic registries of remote candidate sources against that plan, evaluates the resulting held-out candidate banks by probe gap, concept-frontier novelty, and concept uncertainty, and compares active candidate acquisition against fixed-order acquisition under the same token budget. On catalog-backed runs this refresh now happens slot by slot rather than only once up front, and already acquired sources are excluded from later catalog refreshes so the curiosity loop explores new candidates instead of repeatedly reselecting the same source.
- Validated acquisition smoke result: starting from `ag_news` plus `wikitext-103-raw-v1`, with held-out candidates `yelp_polarity` and `imdb`, one `1000`-token acquisition slot, and `128` probe tokens per candidate, the active policy chose `imdb` rather than the fixed-order `yelp_polarity` candidate and finished with lower residual candidate gaps (`final_mean_candidate_gap ~= 0.516` vs `0.595`, `final_max_candidate_gap ~= 0.536` vs `0.603`).
- Historical larger acquisition result: with the `autonomy_acquisition_hf_baseline` preset (`8000` seed tokens per source, `256` probe tokens per candidate, one `2000`-token acquisition slot), active acquisition once chose `imdb` over the fixed-order `yelp_polarity` candidate and reduced the held-out candidate gaps to `final_mean_candidate_gap ~= 0.317` vs `0.470` and `final_max_candidate_gap ~= 0.320` vs `0.481`.
- Historical multi-slot acquisition allocation result: the maintained `autonomy_acquisition_hf_allocation` preset uses held-out candidates `{yelp_polarity, dbpedia_14, imdb}` with two `2000`-token acquisition slots. An earlier run chose `imdb` and then `dbpedia_14`, while fixed-order acquisition spent its budget on `yelp_polarity` and then `dbpedia_14`; that run reported lower residual held-out gaps for active allocation (`final_mean_candidate_gap ~= 0.389` vs `0.421`, `final_max_candidate_gap ~= 0.414` vs `0.441`).
- Late-March failure note (`reports/recheck_autonomy_acquisition_hf_allocation_20260331/summary.json`): the older active path could still select `dbpedia` twice and finish worse than fixed-order acquisition, which exposed two structural problems rather than a small ranking bug: benchmark and live acquisition execution had drifted into duplicated code paths, and isolated lookahead was not preserving the live RNG trajectory.
- Current maintained rerun (`reports/refactor_autonomy_acquisition_hf_allocation_rng_20260401/summary.json`): after unifying offline/live execution, projecting against copied candidate-bank state, and restoring RNG state before each trial projection, `autonomy_acquisition_hf_allocation` again passes strongly. On this environment active allocation chose `reviews` and then `dbpedia` and beat fixed-order acquisition by a wide margin (`active_final_mean_candidate_gap ~= 0.137` vs `0.244`, `active_final_max_candidate_gap ~= 0.148` vs `0.275`, `acquisition_gate_pass = true`).
- An exploratory scout-and-commit acquisition policy is also executable through `autonomy_acquisition_runner.py`: on each slot it evaluates candidate scouts on copied trainer and candidate-bank state, restores RNG state before each isolated trial so the projected frontier matches the live commit path, and only then replays the chosen scout chunk on the live trainer before spending the rest of the slot budget. Scout previews remain non-destructive: rejected candidates no longer advance the live candidate-bank cursor or visit count, so isolated lookahead does not silently discard uncommitted source tokens.
- Current HF scout note (`reports/refactor_autonomy_acquisition_hf_scout_projected_20260401/summary.json`): under the corrected shared executor, the visible exploratory preset is now named `autonomy_acquisition_hf_scout_exploratory`. With `scout_commit_tokens = 100` and `scout_top_k = 1` the scout path matches the active result on the three-candidate HF frontier (`scout_commit_final_mean_candidate_gap ~= 0.137`, `scout_commit_final_max_candidate_gap ~= 0.148`) but still fails the scout gate because it does not improve over active (`scout_commit_gate_pass = false`). A small follow-up sweep with `scout_top_k = 2` and `scout_commit_tokens in {100, 250}` also remained below the active baseline (`reports/refactor_autonomy_acquisition_hf_scout_top2_20260401/summary.json`, `reports/refactor_autonomy_acquisition_hf_scout_250_top2_20260401/summary.json`), so HF scout should currently be treated as exploratory rather than maintained.
- Current maintained catalog rerun (`reports/phase4_benchmark_holdout_shift_full/summary.json`): after normalizing semantic metadata, adding a probe-first scout over tiny real HF samples, and ranking the finalist cut by maintained selection score, `autonomy_acquisition_hf_catalog` now also passes at full budget. On this environment active acquisition chooses `reviews`, then `dbpedia`, then `yelp` and leaves `amazon` held out, while round-robin acquires `amazon`, then `yelp`, then `dbpedia` and leaves `reviews`; active therefore finishes with lower residual held-out gaps (`active_final_mean_candidate_gap ~= 3.61e-7` vs `5.47e-7`, `active_final_max_candidate_gap ~= 3.61e-7` vs `5.47e-7`, `acquisition_gate_pass = true`).
- The larger HF catalog path is therefore back on the maintained validation surface. `autonomy_acquisition_hf_catalog_semantic` and all current HF scout presets still remain exploratory because they do not yet produce reliable wins over the active maintained baselines.
- The same acquisition machinery can also ingest curated web source banks. `corpus_loader.py` can auto-detect `https://...` sources, extract visible page text from the main article content, and stream that text through the same character windowing path used for HF datasets; semantic registries can still rank those remote web entries against the current frontier before the acquisition policy spends any token budget.
- Current deterministic curated-web rerun (`reports/phase5_curated_web_smoke_domain_coherent/summary.json`): after replacing the earlier mixed HF/news setup with pinned `Neuroscience` + `Memory` web seeds, widening the plasticity candidate registry to `{Synaptic plasticity, Hebbian theory, Long-term potentiation, Long-term depression, Spike-timing-dependent plasticity}`, and adding a stop gate for non-positive projected acquisitions, `autonomy_acquisition_curated_web_smoke` now passes on the current environment. Active acquisition chooses `Hebbian theory` and then `Synaptic plasticity`, while round-robin acquires `Spike-timing-dependent plasticity` and then `Hebbian theory`; active therefore finishes with much lower residual held-out gaps (`active_final_mean_candidate_gap ~= 3.96e-7` vs `1.11e-6`, `active_final_max_candidate_gap ~= 4.12e-7` vs `1.12e-6`, `acquisition_gate_pass = true`).
- Historical broader open-web scout result: the exploratory `autonomy_acquisition_open_web_scout` preset seeds the model with Wikipedia pages for `Hebbian theory` and `Synaptic plasticity`, then evaluates held-out candidates `{Neuroplasticity, Predictive coding, Spiking neural network}` with `4000` seed tokens per source, `128` probe tokens per candidate, two `1200`-token acquisition slots, and a `200`-token scout budget over the top `2` candidates per slot. An earlier run reported active allocation beating fixed-order acquisition and the isolated-lookahead scout path beating both baselines.
- Fresh corrected rerun note (`reports/refactor_autonomy_acquisition_open_web_scout_projected_20260402/summary.json`, confirmed by `reports/refactor_autonomy_acquisition_open_web_scout_confirm_20260402/summary.json`): on the current codebase neither active allocation nor scout-and-commit beats round-robin on the broader curated open-web frontier. Active acquisition now selects `snn` and then `predictive` but still finishes worse than round-robin (`active_final_mean_candidate_gap ~= 0.318` vs `0.277`, `active_final_max_candidate_gap ~= 0.352` vs `0.309`, `acquisition_gate_pass = false`), while scout-and-commit selects `neuroplasticity` and then `predictive` and also remains below the baseline (`scout_commit_final_mean_candidate_gap ~= 0.328`, `scout_commit_final_max_candidate_gap ~= 0.344`, `scout_commit_gate_pass = false`). The broader live open-web acquisition path should therefore currently be treated as exploratory end-to-end rather than as a maintained active-allocation success.
- Benchmark note: the earlier two-candidate, three-slot acquisition-allocation setup was dropped from the maintained path because brute-force enumeration showed its best achievable order already matched the fixed-order baseline, so it was a poor benchmark for validating concept-aware active allocation.
- Scope note on the autonomy benchmark: this is still a source-selection proxy for "detects its own knowledge gaps, and actively seeks information to fill them". It now validates closed-loop gap measurement, explicit probe/frontier question generation, semantic gap planning over retrieved memory traces, and adaptive corpus choice in the executable scaffold across maintained HF datasets plus the deterministic curated-web smoke slice, while broader curated open-web URL banks use the same machinery only in exploratory mode. The maintained runner can now expand remote HF semantic registries against the frontier before forming the candidate bank, and it can score pinned curated-web registries in the same loop. It still does not implement unconstrained web search or broader open retrieval beyond the configured remote registries. The richer ambiguity/switch diagnostics are now logged, projected frontier lookahead is the maintained acquisition path, and both HF scout plus the broader live open-web acquisition presets remain exploratory until they reproduce wins over their baselines on fresh reruns.
- Scope note on this result: the executable Phase-3 path is now a stronger approximate attractor / binding circuit implemented over column assemblies, but it is still not a full neuron-level recurrent circuit with explicit SST/PV cell populations and spike-time simulation. However, it no longer lacks explicit polysemy measurement: the maintained runner now includes direct B3 `bank` probes and reports accuracy, signature-margin, and winner-sequence-difference metrics on the live runtime path.
- Transition note: Phase 3 is not 100% complete relative to the long-term paper goal. The current executable path is strong enough to move forward, and the remaining unconstrained live-search / broader open-retrieval work can be revisited after the scale path is locked; explicit question generation and a maintained semantic gap-planning path now exist in the executable scaffold, but broader retrieval and benchmark robustness remain open.

### Phase 4: Hierarchy & Scale — *Scale*

Scale to 100K+ neurons (100+ columns) with dual sparsity strategy and HNSW routing. Full Wikipedia corpus. Distributed training across multiple GPUs with column-level sharding. Vocabulary emerges from character statistics.

- The executable hierarchical-scale path now lives in `hierarchical_scale_runner.py`. It builds a large-column benchmark over the same `HECSNModelLite` scaffold, enables logical column sharding in the routing index, trains on a streaming Wikipedia proxy (`wikitext-103-raw-v1`), and then measures routing recall, routing latency, index integrity, shard balance, throughput, and the paper-aligned memory budget estimate.
- Phase-4 implementation detail that mattered in practice: the first scale pass exposed a real routing-maintenance bug. Dead-column revival updated prototypes in the competitive layer without synchronizing those revived columns back into the routing index, which made the ANN integrity check fail after long runs. The maintained fix now tracks revived column IDs, feeds them through the same index-update path as winners, avoids duplicate ID accumulation inside FAISS HNSW, and forces a clean rebuild before the Phase-4 evaluation sweep.
- Validated smoke result: the maintained `hierarchical_scale_hf_smoke` preset uses `12K` training tokens, `1.5K` eval tokens, `256` columns, `4` routing shards, and `k=12`. It now reaches perfect routing quality on the sharded FAISS HNSW path (`recall@k = 1.000`, `top1_recall = 1.000`, `unreachable_fraction = 0.000`) with mean routing latency approximately `1.04 ms`; the only failing gate is the intended scale threshold because the smoke preset represents only `25.6K` neuron-equivalent capacity.
- Validated baseline result: the maintained `hierarchical_scale_hf_baseline` preset uses `30K` training tokens, `4K` eval tokens, `1024` columns, `8` routing shards, `k=16`, rebuild threshold `128`, and shard candidate factor `2`. Under the reporting assumption of `100` neurons per column this corresponds to `102,400` neuron-equivalent scale, and the executable hierarchical-scale gate passed on the FAISS-backed sharded HNSW path: `recall@k = 1.000`, `top1_recall = 1.000`, mean routing latency approximately `2.28 ms` (`p95 ~= 4.40 ms`), `unreachable_fraction = 0.000`, shard balance ratio `1.0`, winner coverage `1.0`, throughput approximately `156 tokens/s` (`~1560 chars/s`), and estimated total GPU budget approximately `8.94 GB`.
- The maintained Phase-4 baseline also kept the routing compression ratio in the intended sparse regime: only `16 / 1024 = 1.5625%` of columns participate in Stage-2 competition for a given query.
- Practical input path: the current executable interface accepts raw text directly. `query_runner.py` converts the input stream into rolling character windows, encodes each window with the same `RTFEncoder.routing_vector` contract used during training, and can either feed those windows through `trainer.train_step(...)` for continued online learning or query the current model state without changing weights.
- This input/retrieval path is no longer Phase-4-only: the maintained Stage-0, Phase-2, Phase-3, Phase-4, autonomy, and acquisition runners can all emit checkpoints, so the same raw-text query interface can probe the base learner, the consolidation path, the context-enabled learner, the scale benchmark, or the active-information-seeking variants without needing a separate conversion step.
- Practical retrieval path: the current executable system retrieves *evidence* and now also exposes a small native decode summary rather than generating free text. A raw-text query is encoded into a routing key, Stage 1 HNSW returns the nearest candidate columns, Stage 2 identifies the strongest winner, and the query interface then searches the Memory Store for the closest stored routing keys / input windows. When checkpoints were created from a runner that recorded raw windows, retrieval can return literal remembered character windows such as `character ` for the query suffix `characters`; the same path can also stitch overlapping remembered windows into a best-effort `native_decode` fragment / continuation with a confidence score. This keeps retrieval aligned with the unlabeled, character-level objective instead of introducing a supervised or autoregressive decoder.
- Example executable query result on a saved Phase-4 smoke checkpoint: after feeding the additional raw text `autonomous systems learn from raw character streams without labels.`, a query for `knowledge accumulation from raw characters` reduced to the final window `characters`, routed to winner column `241` on shard `1`, and retrieved the nearest remembered window `character ` from the stored memory bank with cosine similarity approximately `0.633`. This demonstrates the current retrieval contract: nearest internal evidence and column routing support, not language-model generation.
- Example executable base-checkpoint result on a saved Stage-0 smoke run: querying `character stream` reduced to the final window `ter stream`, routed to winner column `1`, and retrieved the remembered window `t is the t` from the stored memory bank. This shows that even the pre-context base learner now participates in the same checkpoint/query contract.
- Initial native-decode validation: reopening an exploratory open-web scout checkpoint through `query_runner.py` also yields populated `native_decode` summaries for probe suffixes such as `plasticity`, with confidence approximately `0.785` and overlap ratio approximately `0.667`. The output is still intentionally fragmentary, which is consistent with the intended scope of memory-backed continuation rather than fluent generation.
- Additional integration check: saved Phase-2 and autonomy-acquisition smoke checkpoints were also reopened successfully through `query_runner.py`, confirming that the shared checkpoint/query contract now spans sequential consolidation and active source-acquisition runs as well, not only the base/context/scale demos.
- Scope note on this result: within the executable scaffold, Phase 4 is now complete as a hierarchy-and-scale proxy. The maintained path validates sharded HNSW routing, large-column scaling, and the memory/latency gate in one process. It is still a logical sharding implementation rather than a full multi-process NCCL trainer, but the routing contract and scale diagnostics needed for the current roadmap are now in place.

### Stage 3 (Future): Abstraction Layer

**Partial maintained precursor exists, but the full dedicated layer is still future work.**

Current maintained precursor:
- `ConceptStore` + `OnlineSlowFeatureMap` now observe stored Memory Store episodes not only during query-time concept readout but also during live Terminus `feed`, respond-time learn-back, background ticks, and acquisition training.
- This makes abstraction accumulation partially continuous on the maintained runtime without introducing labels or a separate pretrained semantic module.
- However, this precursor still rides on Memory Store episode text plus stored routing signatures; it is not yet the paper's full dedicated abstraction layer with its own long-timescale update schedule and direct top-down influence.

Planned implementation using Slow Feature Analysis (Wiskott & Sejnowski 2002):
- Input: Binding Layer activation sequence over N episodes
- Learns: linear projection W such that W*x changes minimally across time
- Update: minimize d/dt(W*x)^2 subject to unit variance constraint
- Learning: gradient-free, solvable by eigendecomposition of temporal derivative covariance matrix (offline, during sleep)

---

## 8. Evaluation Protocol

### 1. Assembly Quality Metrics

**Drift Rate:** Mean Jaccard distance between consecutive exposures to similar patterns. Target < 0.04 for stable concepts.

**Drift Floor Trend (mandatory at Phase 1+):** Minimum drift per 10K-token window. This must be tracked as a time series and should be non-increasing; rising floor indicates replay is no longer compensating ongoing overwrite.

**Sparsity:** Percentage of active neurons per assembly. Target: 2–5% of column population.

**E/I Balance:** Ratio of mean excitatory to inhibitory weights. Maintained dynamically by iSTDP.

### 2. Synaptic Weight Distribution (Critical)

**Distribution Shape:** Must be log-normal (not bimodal, not uniform). Log-STDP + iSTDP + synaptic scaling together produce this.

**Kurtosis:** Target 3–6 (log-normal range). Bimodal saturation shows kurtosis > 10.

**Renormalization Efficacy:** After sleep, distribution should remain log-normal with mean within 0.1 of target.

### 3. Catastrophic Forgetting Resistance

Sequential learning benchmark: Train on Science corpus (20K tokens) → Test pattern completion. Train on Politics corpus (20K tokens) → Re-test Science. **Failure criterion:** >15% degradation without sleep. **Success:** <5% degradation with adaptive sleep.

### 4. Emergence Verification

**Character-to-Word Transition:** Network should spontaneously form multi-character assemblies (proto-words) after sufficient exposure (~50K–100K tokens, not 5K).

**Context Dependence:** Same character sequence activates different assemblies based on preceding context (measured via assembly overlap).

**Compression Efficiency:** Bits per spike should decrease as structure emerges (better compression = better representations).

### 5. Scalability Benchmarks

**Memory Usage:** Verify CSR sparse storage < 4 GB for 50K neurons including HNSW index, eligibility traces, and all auxiliary state.

**Throughput:** Characters processed per second. Record measured throughput with hardware metadata (GPU model, precision mode, batch size, sparsity mode). Use this as a baseline trace, not as a fixed pass/fail gate.

**Routing Latency:** HNSW query time for 100K columns. Target: <5ms (CPU).

**Index Integrity:** After 10K dynamic insertions, <1% unreachable points.

### 6. Behavioral Verification

**Test B1 — Character N-gram Recovery**

```
Procedure:
  - Present partial character sequence (e.g., "hel" from "hello")
  - Measure which column activates
  - Present "hel" → measure nearest-neighbor assembly in Memory Store
  - Present "hell" → measure nearest-neighbor assembly

Success criterion:
  - Assembly distance between "hel" and "hell" < assembly distance
    between "hel" and "xyz" (completion coherence)
```

**Test B2 — Distributional Clustering**

```
Procedure:
  - Collect column activation patterns for 1000 common English words
  - Compute silhouette score on activation vectors

Success criterion:
  - Silhouette score > 0.25 (assemblies for similar words cluster)
```

**Test B3 — Context Disambiguation**

```
Procedure:
  - Present the character sequence "bank" after "river" context
  - Present the character sequence "bank" after "money" context
  - Measure which binding assembly activates

Success criterion:
  - Different assemblies activate for the same characters in
    different contexts at a rate > chance (>60% consistent)

Note: This is the hardest and most important test. It directly
validates the claim that context modulation produces polysemy
disambiguation. If B3 fails, the Context Layer is not functioning.
```

**Test B4 — Forgetting Resistance**

```
Procedure:
  - Train on corpus A for 100K tokens
  - Train on corpus B for 100K tokens
  - Return to corpus A patterns
  - Measure activation pattern overlap with pre-B patterns

Success criterion:
  - Activation patterns for corpus A stimuli after B training
    overlap >50% with pre-B patterns (measured by cosine similarity)
```

---

## 9. Critical Risks & Mitigations

> **RISK 1: Competitive Learning Divergence**
>
> **Issue:** Learning can diverge with non-stationary inputs common in online learning.
> **Mitigation:** Kohonen/SOM update with learning rate annealing (eta(t) = eta_0/(1 + alpha*t)), gradient clipping (max norm 10.0), weight normalization after each update, and prototype reset if norm exceeds threshold.

> **RISK 2: HNSW Graph Degradation**
>
> **Issue:** Dynamic column insertion/deletion creates unreachable nodes, breaking routing [3].
> **Mitigation:** Tombstone marking instead of deletion, parallel `_vector_store` for rebuild capability, periodic index rebuild during sleep phases (every 1000 insertions), and graph connectivity repair via reverse edge addition.

> **RISK 3: Homeostatic Network Fragmentation**
>
> **Issue:** Synaptic scaling with improper setpoints can silence half the network permanently.
> **Mitigation:** Zero Gaussian rule (eta=0 for silent setpoint), minimum weight floor (0.001), iSTDP for E/I balance, and recovery protocol for neurons silent >1000 tokens.

> **RISK 4: Sleep Timing Sensitivity**
>
> **Issue:** Delayed sleep causes irreversible memory erasure; too frequent sleep prevents learning [1].
> **Mitigation:** Two-tier controller: micro-sleep every `200` tokens for cheap recency replay, scheduled deep maintenance for structural replay, and emergency deep sleep triggered from closed drift-floor windows instead of the rolling minimum. The recovered 100-column baseline uses scheduled deep maintenance every `2500` tokens with `150` replay steps.

> **RISK 5: Cold Start Pathology**
>
> **Issue:** First 5000 tokens determine initial attractor states; poor data produces pathological representations.
> **Mitigation:** Diverse bootstrap corpus, predictive coding warm-up, fixed-budget competition with dead-column revival, and explicit Stage-0 validation before scaling. Monitor reconstruction error convergence rather than assuming early specialization is healthy.

> **RISK 6: Attractor Structural Instability**
>
> **Issue:** Continuous attractors are structurally unstable — infinitesimal parameter changes destroy them [37]. The Context Layer cannot rely on perfect attractor dynamics.
> **Mitigation:** Use approximate attractor with slow time constants (robust to perturbation). tau_slow = T_per_token * context_tokens (functional mapping). Accept manifold drift as functional feature, not bug. Monitor attractor quality via autocorrelation half-width of context state.

> **RISK 7: CSR Tensor Limitations**
>
> **Issue:** PyTorch CSR tensor support is beta. Concatenation, stacking, and many standard operations fail. This affects dynamic connectivity operations.
> **Mitigation:** Use static index tensors with mutable value vectors for update paths. Avoid per-step CSR/COO conversion in learning loops; apply structural plasticity through masked index/value edits, then refresh sparse views only at maintenance boundaries.

> **RISK 8: Mechanism Unvalidated**
>
> **Issue:** Character-level emergent concept formation via STDP is a research frontier, not a solved problem. Building at scale before Stage 0 validation risks wasting months on an unvalidated premise.
> **Mitigation:** Mandatory Stage 0 gate with measurable success criteria (clustering quality threshold, baseline comparison, behavioral tests B1–B4). Document findings before scaling. Redesign mechanism if Stage 0 fails.

---

## 10. References

Citation-integrity note (2026-03-31): earlier draft placeholder preprints with unresolved topic verification were removed from the maintained claim path below. Numbering gaps are intentional until replacement sources are checked.

[1] Massey, F. et al. (2026). Sleep-Based Homeostatic Regularization for Stabilizing STDP in Recurrent SNNs. *arXiv:2601.08447*. Interleaved sleep (10-20% ratio) stabilizes STDP in recurrent networks.

[46] Ratcliff, R. (1990). Connectionist models of recognition memory: Constraints imposed by learning and forgetting functions. *Psychological Review*, 97(2), 285-308. Catastrophic interference in distributed memory systems.

[2] Malkov, Y. A. & Yashunin, D. (2020). Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. *IEEE TPAMI*, 42(4), 824–836.

[3] Li, J. et al. (2025). Dynamic HNSW: A Dynamic Indexing Method for HNSW Based on Subgraph Reconstruction. *Neurocomputing*, 127338. Unreachable points phenomenon.

[5] Carlson, K. D. et al. (2013). A biological plausible self-organizing spiking neural network with synaptic scaling. *IJCNN 2013*.

[6] Buzsaki, G. (2010). Neural syntax: Cell assemblies, synapsembles, and readers. *Neuron*, 68(3), 362–385.

[7] Redondo, R. L. & Morris, R. G. (2011). Making memories last: The synaptic tagging and capture hypothesis. *Nature Reviews Neuroscience*, 12(1), 17–30.

[8] Markram, H. et al. (2015). Reconstruction and simulation of neocortical microcircuitry. *Cell*, 163(2), 456–492.

[9] Davies, M. et al. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. *IEEE Micro*, 38(1), 82–99.

[10] Bogacz, R. (2017). A tutorial on the free energy framework for modelling perception and learning. *Journal of Mathematical Psychology*, 76, 198–211.

[11] Clopath, C. et al. (2010). Connectivity reflects coding: A model of voltage-based STDP with homeostasis. *Nature Neuroscience*, 13(3), 344–352.

[12] Turrigiano, G. G. & Nelson, S. B. (2004). Homeostatic plasticity in the developing nervous system. *Nature Reviews Neuroscience*, 5(2), 97–107.

[13] Turrigiano, G. G. (2008). The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell*, 135(3), 422–435.

[14] Tremblay, R., Lee, S., & Rudy, B. (2016). GABAergic interneurons in the neocortex. *Trends in Neurosciences*, 39(11), 775–794.

[15] Nvidia (2024). Structured Sparsity in PyTorch. *Technical documentation*. 2:4 sparsity ~1.6x speedup on Ampere+.

[16] Kohonen, T. (1990). The self-organizing map. *Proceedings of the IEEE*, 78(9), 1464–1480.

[17] Schultz, W. (2015). Neuronal reward and decision signals: From theories to data. *Physiological Reviews*, 95(3), 853–951.

[18] Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. *Current Opinion in Neurobiology*, 16(6), 710–715.

[19] Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity. *Neuron*, 81(1), 12–34.

[20] Desai, N. S. et al. (1999). Plasticity in the intrinsic excitability of cortical pyramidal neurons. *Nature Neuroscience*, 2(6), 515–520.

[21] Sajikumar, S. & Frey, J. U. (2004). Late-associativity, synaptic tagging and the role of dopamine. *Neurobiology of Learning and Memory*, 82(1), 12–25.

[22] Frey, U. & Morris, R. G. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533–536.

[23] Markram, H., Lubke, J., & Frotscher, M. (1997). Regulation of synaptic efficacy by coincidence. *Science*, 275(5297), 213–215.

[24] Tetzlaff, C. et al. (2011). The dynamics of memory consolidation in Hebbian networks. *Frontiers in Computational Neuroscience*, 5:15.

[25] Wiskott, L. & Sejnowski, T. J. (2002). Slow Feature Analysis: Unsupervised Learning of Invariances. *Neural Computation*, 14(4), 715–770.

[26] Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

[28] Schuman, C. D. et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*, 2(1), 10–19.

[29] Zenke, F. & Vogels, T. P. (2021). The Remarkable Robustness of Surrogate Gradient Learning. *Neural Computation*, 33(4), 899–925.

[30] Gilson, M. & Fukai, T. (2011). Stability versus neuronal specialization for STDP. *Neural Computation*, 23(6), 1514–1529.

[31] Roy, D. et al. (2019). Lifelong Learning of Spatiotemporal Representations. *Frontiers in Neural Networks*, 12:765.

[33] Yang, W. et al. (2014). Differences in E/I balance between cortical layers. *Journal of Neuroscience*, 34(34), 11206–11213.

[34] Li, Y. et al. (2024). Artificial visual neurons with NbOx Mott memristors for rate-temporal fusion encoding. *Nature Communications*, 15, 6027. Origin of RTF encoding principle; hardware visual domain.

[35] Effenberger, F., Jost, J., & Levina, A. (2015). Self-organization in balanced state networks by STDP and homeostatic plasticity. *PLOS Computational Biology*, 11(9), e1004420. Excitatory STDP + inhibitory STDP + synaptic scaling produce log-normal weight distributions.

[36] Nair, A. et al. (2024). Causal evidence of a line attractor encoding an affective state. *Nature*, 634, 394–401. Line attractors require slow neurotransmission and dense subnetwork connectivity.

[37] Sagodi, A. et al. (2024). Back to the Continuous Attractor. *arXiv:2408.00109*. Continuous attractors are structurally unstable; approximate attractors persist via slow manifolds.

[38] Chong, Y. S., Ang, S. R., & Sajikumar, S. (2025). Beyond boundaries: extended temporal flexibility in synaptic tagging and capture. *Communications Biology*, 8, 475. Tags persist up to 9 hours in strong-before-weak paradigms.

[39] Luboeinski, J. & Tetzlaff, C. (2021). Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks. *Communications Biology*, 4(275). STC in recurrent spiking networks.

[40] Karlsson, V., Fianda, N., & Kamarainen, J.-K. (2026). Difference Predictive Coding for Training Spiking Neural Networks. *ICLR 2026*. Spike-native predictive coding with sparse ternary messages — relevant to Surprise Monitor design.

[41] NeuronSpark (2026). A 0.9B-parameter spiking language model trained from scratch with next-token prediction and surrogate gradients. Demonstrates feasibility of SNNs for language at scale. *arXiv:2603.16148*.

[42] Naderi, R. et al. (2025). Unsupervised post-training learning in spiking neural networks. *Scientific Reports*, 15, 17647. Short-term plasticity enables post-training adaptation without weight changes.

[43] N'dri, A. W. et al. (2024). Predictive Coding with Spiking Neural Networks: a Survey. *arXiv:2409.05386*. Comprehensive review of spiking predictive coding approaches.

[44] Vogels, T. P. et al. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, 334(6062), 1569–1573. Foundation for iSTDP.

[45] Brette, R. & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model as an effective description of neuronal dynamics. *Journal of Neurophysiology*, 94, 3637–3642. AdEx neuron model.

---

*Thiago Maceno Rocha Goulart · Brasil · github.com/Tuafo*

*HECSN — Hierarchical Emergent Concept Spiking Networks*

*PyTorch 2.1+ · CUDA · 2:4 Structured Sparsity · CSR Sparse · FAISS (CPU)*

*Research Architecture — Stage 0 Validation Required*

*All verification targets are falsifiable predictions. No experimental results reported.*
