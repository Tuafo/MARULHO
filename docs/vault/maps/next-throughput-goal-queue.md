---
type: map
status: active
related_code:
  - ../../../src/marulho/training/cuda_graph_route_transition.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/core/column_runtime.py
related_docs:
  - ../../goals/marulho-living-runtime-goal.md
  - ../../research-living-brain.md
  - ../../adr/0006-persistent-text-tick-executor.md
related_papers:
  - ../papers/cuda-triton-snn-optimization.md
  - ../papers/neuronspark-v1.md
  - ../papers/predictive-coding.md
  - ../papers/structural-plasticity.md
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
---

# MARULHO Next Throughput Goals

Generated: 2026-06-15

This note turns the current throughput and living-brain research direction into copy-ready `/goal` prompts. It follows the Codex goal pattern: one durable objective, a verifiable stopping condition, required context to inspect first, validation commands/artifacts, and constraints that must remain intact.

Each prompt below is intentionally multi-iteration. A completed iteration should commit and report evidence, but the `/goal` itself should remain active until its durable completion condition is actually met, the user redirects it, or the agent is genuinely blocked.

Use the full north-star spec at `docs/goals/marulho-living-runtime-goal.md` as shared context for every goal below. Use `docs/vault/` as the refined Obsidian/Graphify navigation layer; treat `graphify-out/` as an ignored generated cache. Current source, tests, benchmark reports, and Runtime Truth remain more authoritative than stale docs or generated graph output.

## Current Evidence

- Retained best sustained CUDA evidence: `reports/base_comparison_20260615/current-native-131072-i32.json` reached `4992.049 tokens/sec` over `131072` warm source tokens on RTX 3060 with `tick_tokens=128`, `quantum_tokens=16`, exact eight-token native parent graph replay, `131056` native-covered burst tokens, and zero native graph/burst failures.
- The same shape with native replay disabled reached `4530.883 tokens/sec`, so exact eight-token native parent replay remains useful.
- A later rerun under contention reached `4489.641 tokens/sec`, so current "right now" speed must be interpreted with `velocity_environment.v1`.
- Direct one-block Triton route/vote fusion is retired. It passed parity but regressed complete runtime, so the next speed work should not revive it as a local top-k wrapper.
- Partial native parent graph replay for non-eight-token tails is opt-in diagnostic only. The `tick_tokens=130` experiment regressed and exposed host-truth misalignment.
- The next credible speed path is a true device-owned multi-tick or persistent sequence executor that reduces the real per-token graph/kernel launch boundary while preserving sequential state, Runtime Truth cadence, and rollback gates.

## Research Anchors

- CUDA Graphs reduce repeated launch/setup overhead only when the graphed range is the actual bottleneck: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html and https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html
- FlashRNN and persistent RNN work support keeping recurrent time loops and weights on chip/device for low-batch sequential models: https://arxiv.org/html/2412.07752v2 and https://proceedings.mlr.press/v48/diamos16.html
- NeuronSpark shows pure SNN language is plausible with selective state-space spiking, adaptive timesteps, and fused Triton PLIF kernels, but remains inspiration-only until MARULHO owns training, grounding, and checkpoints: https://arxiv.org/html/2603.16148v1
- Thousand Brains/Monty supports many independent learning modules communicating through bounded protocol messages and voting rather than one monolithic loop: https://arxiv.org/html/2412.18354v1
- Expert Choice MoE supports capacity-bounded specialist activation: https://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html
- Modern Hopfield networks support bounded associative recall inside columns, not as the whole mind: https://openreview.net/forum?id=tL89RnzIiCd
- Predictive Coding Light supports suppressing predictable spikes and transmitting compressed representations as a metabolism rule: https://www.nature.com/articles/s41467-025-64234-z
- Difference Predictive Coding supports sparse spike-compatible local learning signals: https://openreview.net/forum?id=iu9dbz2lB9
- Structural plasticity research supports growth/pruning with local evidence, homeostasis, reward/error pressure, and sparse GPU-aware topology, not hot-path mutation: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1224752/full and https://arxiv.org/html/2510.19764v1
- Nemotron 3 Ultra/Super data and architecture notes reinforce sparse MoE, hybrid recurrence, multi-token prediction, and open high-quality data as slow-path training ingredients, not as a hidden runtime LLM: https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/ and https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v3

## Ranked Paths

1. Device-Owned Multi-Tick Executor v0. This is the highest-leverage throughput path because current evidence still pays one persistent tick replay per token and Python sequence orchestration around eight-token bursts.
2. Column-Society Execution Scheduler. Make many-column growth useful by ensuring awake columns per tick stay small and inactive specialists do not tax the executor.
3. Spike-Native Language/Readout v0. Build a MARULHO-owned language path as grounded sparse sequence prediction, not a hidden LLM.
4. Growth/Pruning Promotion Pipeline. Turn repeated surprise into reversible candidate-column/synapse trials with evidence, budgets, and rollback.
5. Local Recall and Replay Windows. Use bounded modern-Hopfield recall inside columns and replay/consolidation slow paths to improve memory without taxing the live tick.
6. Autonomous Research/Data Acquisition Slow Loop. Replace brittle source assumptions with explicit, licensed, high-value data acquisition and evaluation windows.
7. Runtime Truth and Control Room Parity. Ensure operator-visible control room evidence tracks the true promoted path without adding hot-path reporting tax.

## Goal 1: Device-Owned Multi-Tick Executor v0

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward a true device-owned multi-tick executor for the promoted CUDA text path. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: reduce or bypass the remaining per-token host/graph-launch boundary in the promoted text runtime without skipping sequential SNN state updates. Start from the current exact fast shape: 1024 columns, 64 column dim, k=10, text-only CUDA checkpoint, `tick_tokens=128`, `execution_quantum_tokens=16`, host-truth cadence 32, exact eight-token native parent graph replay. The goal is not another local wrapper; it is a larger device-owned sequence boundary or a defensible prototype proving why the next lower-level executor must be C++/CUDA/Triton/hybrid.

Inspect first: `CONTEXT.md`, ADR 0005 and ADR 0006, `docs/retired-paths.md`, `docs/research-living-brain.md`, `docs/vault/benchmarks/hot-path-latency.md`, `docs/vault/modules/training.md`, `docs/vault/concepts/column-runtime.md`, `src/marulho/training/cuda_graph_route_transition.py`, `src/marulho/training/trainer.py`, `src/marulho/training/column_transition_runtime.py`, `src/marulho/core/fused_route_vote_cuda.py`, `src/marulho/core/inplace_column_cuda.py`, `src/marulho/core/native_cuda_graph_replay.cpp`, and the current long-run reports under `reports/base_comparison_20260615/`.

Research requirements: use current CUDA Graph, persistent kernel/RNN, FlashRNN, Triton persistent-kernel, NeuronSpark fused PLIF, and PyTorch CUDA Graph guidance before choosing the executor boundary. Document research only if it changes implementation direction.

Hard constraints: protect the hot path; do not reintroduce direct one-block route/vote fusion or partial-native tail replay as defaults; do not move algorithms into `service`; do not trade throughput for decorative scalar reports; preserve pre-mutation fallback and fail-closed post-launch behavior; keep startup/capture cost visible but outside measured warm throughput.

Iteration success criteria: produce a contained implementation or benchmark-grade prototype that either beats `4992.049 tokens/sec` on a clean `131072` token sustained CUDA run or proves with profiling that the next executor must move lower than the current Python/CUDA Graph boundary. Runtime Truth must expose which executor ran, token coverage, failures/fallbacks, device evidence, host-truth cadence, startup compile/capture cost, and whether sequential state parity or bounded cognitive-quality gates passed.

Durable completion condition: complete this goal only when the promoted runtime has a measured device-owned sequence executor that consistently beats the retained sustained ceiling on comparable long CUDA runs, or when profiling and prototype evidence commits a lower-level executor ADR proving no safe local path remains inside the current architecture.

Verification: run focused parity tests for state mutation order, CUDA failure/fallback tests, py_compile, relevant `pytest` tests, and at least one long warm sustained run comparable to `reports/base_comparison_20260615/current-native-131072-i32.json`. Include `velocity_environment.v1` or explain why unavailable.

Docs: update `CONTEXT.md`, ADR 0006 or a new ADR if the executor boundary is hard to reverse, `docs/research-living-brain.md`, `docs/retired-paths.md` for rejected executor attempts, and `docs/vault/benchmarks/hot-path-latency.md`. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report instead of inventing progress if the native extension/Triton path, CUDA evidence, tests, or long-run benchmark cannot be executed safely.
```

## Goal 2: Column-Society Execution Scheduler

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward a Thousand-Brains-style column-society execution scheduler. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: move Column Runtime from mostly report-only evidence toward a real scheduler boundary where many columns may exist, but only a small relevant subset wakes per tick. Each column should have role, state, prediction, surprise, usefulness, cost, memory pressure, cached vote, and wake/sleep reason. Awake columns per tick must stay bounded even as total columns grow.

Inspect first: `CONTEXT.md`, `docs/vault/concepts/column-runtime.md`, `docs/vault/maps/runtime-truth-surface-map.md`, `docs/research-living-brain.md`, `docs/retired-paths.md`, `src/marulho/core/column_runtime.py`, `src/marulho/core/predictive_columns.py`, `src/marulho/core/columns.py`, `src/marulho/training/trainer.py`, `src/marulho/training/cognitive_boundary_controller.py`, `src/marulho/service/status_read_model.py`, and column runtime tests/benchmarks.

Research requirements: use Thousand Brains/Monty/CMP, sparse MoE routing, Expert Choice capacity routing, predictive coding, and neuromorphic metabolism to choose the scheduler contract. Hopfield recall may be used only inside a column as bounded recall, not as a global mind.

Hard constraints: cached-vote/sleep/deep-sleep must not be fake status if execution still runs all columns; scheduler decisions must be training/core owned, while service only projects Runtime Truth; growth/pruning remains checkpoint-backed and operator-reviewed; do not add all-column scans to decide sleep.

Iteration success criteria: promote one real execution effect beyond existing candidate scoring/homeostasis, such as a scheduler-owned awake mask or cached-vote path that demonstrably skips non-relevant specialist work while preserving state correctness. Runtime Truth must expose active/idle/sleep/deep-sleep/candidate/retired counts, vote disagreement, wake reasons, cached vote use, latency/cost, fallback reasons, and `runs_all_columns` truth.

Durable completion condition: complete this goal only when the scheduler is the promoted execution owner for bounded awake masks or cached votes, benchmarks prove total column count can grow without proportional awake work, Runtime Truth exposes the full wake/sleep/vote contract, and the next growth/pruning continuation is explicitly queued.

Verification: add focused tests for scheduler decisions, cached-vote correctness, sleep/wake transitions, fallback behavior, and no service algorithm ownership. Run a CUDA or CPU benchmark showing awake-count remains bounded and complete tick/runtime cost is neutral or better. If CUDA scoped sparse update is slower, report and retain dense CUDA with explicit fallback reason.

Docs: update `CONTEXT.md` for any resolved column-state terms, `docs/vault/concepts/column-runtime.md`, `docs/vault/benchmarks/hot-path-latency.md`, and `docs/retired-paths.md` for rejected scheduler variants. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if scheduler truth cannot be proven without adding a hot-path all-column tax.
```

## Goal 3: Spike-Native Language and Verified Readout v0

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward MARULHO-owned spike-native language/readout. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: build the next real language-from-spikes step without importing a hidden LLM or external checkpoint as cognition. The target is a bounded MARULHO-owned sparse sequence/readout module that can learn or evaluate next-symbol/readout trajectories from Subcortex evidence, with grounding and Runtime Truth gates. Multi-token/speculative readout may be explored only as a verified draft mechanism: cheap local spike-readout drafts, grounded Subcortex verification, accepted prefix evidence.

Inspect first: `CONTEXT.md`, `docs/vault/concepts/language-from-spikes.md`, `docs/research-living-brain.md`, ADR 0005, SNN language/readout service ledger/executor modules, `src/marulho/semantics/spike_language_neurons.py`, `src/marulho/semantics/language_surface.py`, language/readout tests, and any SNN language capacity mutation/replay gates.

Research requirements: use NeuronSpark, SpikeGPT, SpikingSSMs, Nord-AI as inspiration; use Medusa/MTP/speculative decoding only as a verification pattern; use Nemotron datasets only as optional slow-path training/evaluation data sources, never as hidden runtime cognition.

Hard constraints: do not satisfy language readiness by loading NeuronSpark/Nord/Nemotron checkpoints; no hidden ThoughtLoop, Cortex LLM, prompt template, or external generation path; language generation/training stays slow-path until promoted; runtime readout must be grounded in MARULHO-owned tensors, support terms, and evidence.

Iteration success criteria: produce a code/test/evaluation improvement that moves beyond display-only readiness toward a measurable sparse sequence/readout capability, or creates a rigorous training/evaluation harness with dataset provenance. Runtime Truth must expose available/trained/grounded/device status, mutation absence or gate status, latency, memory/VRAM cost, and why the path is or is not promotable.

Durable completion condition: complete this goal only when MARULHO owns a tested spike-native language/readout training or evaluation path with grounding, device evidence, readiness gates, checkpoint/rollback behavior, and at least one explicit promote/reject decision for a real readout capability.

Verification: run focused tests for non-generative/read-only boundaries or trainer gates, an evaluation runner over a bounded corpus, CUDA/device evidence if tensors are used, grounding support checks, and a speed/latency report. If using Hugging Face/Nemotron data, record dataset name, license/terms, split/sample size, and cache path.

Docs: update `CONTEXT.md`, `docs/research-living-brain.md`, `docs/vault/concepts/language-from-spikes.md`, `docs/vault/papers/neuronspark-v1.md`, and retired-path notes if old language vocabulary/code is removed. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if data, GPU memory, grounding gates, or dependency setup block a defensible local run.
```

## Goal 4: Growth/Pruning Promotion Pipeline

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward reversible self-growth and pruning. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: turn repeated prediction failure, surprise, uselessness, redundancy, cost, or instability into a checkpoint-backed candidate growth/pruning pipeline. Candidate columns/synapses must prove usefulness before trust, and weak/slow/noisy structures must sleep, weaken, archive, or become prune-eligible without mutating the always-on runtime unsafely.

Inspect first: `CONTEXT.md`, `docs/vault/concepts/column-runtime.md`, `docs/vault/papers/structural-plasticity.md`, `docs/research-living-brain.md`, `docs/retired-paths.md`, `src/marulho/core/predictive_columns.py`, `src/marulho/core/hypercube.py`, `src/marulho/core/plasticity.py`, `src/marulho/training/developmental_runner.py`, `src/marulho/evaluation/binding_growth_trial.py`, structural mutation ledgers, checkpointing, and service read-only trial endpoints.

Research requirements: use structural plasticity, reward-modulated STDP, homeostatic plasticity, sparse GPU structural plasticity, event-based delay learning, and self-growing neural-network work to design local evidence and budgets. Research does not override MARULHO's operator-reviewed rollback gate.

Hard constraints: no structural mutation in the hot tick; no service-owned edge selection; no growth from one-shot surprise; no pruning without tombstone/provenance/rollback; no CUDA claim unless tensor/device placement and benchmark evidence prove it.

Iteration success criteria: implement or strengthen one end-to-end gate from evidence collection to isolated evaluation or reviewed transaction. It must include exact baseline hash, candidate reason, cost/usefulness metrics, latency/VRAM/RAM impact, rollback artifact, Runtime Truth summary, and a no-mutation proof until the approved executor runs.

Durable completion condition: complete this goal only when an evidence-to-isolated-evaluation-to-reviewed-transaction path exists, checkpoint rollback/tombstone behavior is tested, and at least one candidate growth or pruning case is safely promoted, rejected, or retired with recorded Runtime Truth evidence.

Verification: add tests for repeated-failure gating, no one-shot mutation, checkpoint roundtrip, rollback/tombstone evidence, and service read-only boundaries. Run a focused benchmark/evaluation showing cost and quality before/after in a clone or explicit slow path.

Docs: update `CONTEXT.md` for any growth/pruning terms, `docs/research-living-brain.md`, `docs/vault/concepts/column-runtime.md`, `docs/vault/papers/structural-plasticity.md`, and `docs/retired-paths.md` for removed unsafe growth paths. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if mutation safety, checkpointing, rollback, or evaluation evidence is unavailable.
```

## Goal 5: Local Recall and Replay Windows

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward bounded local recall and replay/consolidation that improves cognition without taxing the live tick. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: make memory more useful while preserving velocity. Use modern-Hopfield-style recall only as bounded associative recall inside a column or replay window. Keep archival storage CPU-resident unless active replay computation benefits from CUDA. Replay must be selected, measured, and cadenced, not every-token background work.

Inspect first: `CONTEXT.md`, `docs/vault/concepts/column-runtime.md`, `docs/vault/papers/replay-consolidation.md`, `docs/vault/benchmarks/hot-path-latency.md`, `docs/research-living-brain.md`, `src/marulho/core/column_runtime.py`, memory store/consolidation modules, replay runners, `src/marulho/training/trainer.py`, and replay/consolidation tests.

Research requirements: use modern Hopfield networks, complementary learning systems, continual-learning replay, synaptic tagging/capture, and sparse replay literature. Treat attention-like recall as a local memory operator, not as a transformer mind.

Hard constraints: no full-memory scan in the live tick; no every-token slow-memory admission; no GPU-resident archival metadata unless complete-runtime evidence wins; no hidden language reasoning through replay text.

Iteration success criteria: implement or improve one bounded replay/recall mechanism with clear selection criteria, memory budget, device placement, quality metric, latency cost, and Runtime Truth evidence. It should either improve prediction/grounding/reconstruction under a bounded benchmark or retire a memory path that costs speed without evidence.

Durable completion condition: complete this goal only when bounded recall/replay is integrated into explicit slow-path windows, improves a measured prediction/grounding/reconstruction target or retires a costly dead path, and long-run evidence proves the live tick remains protected.

Verification: run focused recall/replay tests, checkpoint/reload tests, a replay quality benchmark, and a hot-path or long-run check proving the live tick is not slower. Report CPU/GPU/RAM/VRAM behavior.

Docs: update `docs/research-living-brain.md`, `docs/vault/papers/replay-consolidation.md`, `docs/vault/concepts/column-runtime.md`, `docs/vault/benchmarks/hot-path-latency.md`, and retired paths as needed. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if quality evidence cannot be gathered or the recall path would enter the always-on runtime without promotion.
```

## Goal 6: Autonomous Research/Data Acquisition Slow Loop

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward an autonomous research and data acquisition slow loop. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: replace brittle/default source assumptions with an explicit slow-path pipeline that can search, select, cache, inspect, and evaluate useful local data sources for MARULHO training/evaluation. Prioritize licensed, high-signal corpora and MARULHO-relevant tasks. Hugging Face/Nemotron datasets may be researched and sampled, but ingestion must remain explicit and not part of the always-on tick.

Inspect first: `CONTEXT.md`, source catalog/data modules, runtime source cache, autonomy acquisition runners, evaluation data loaders, docs/vault data/source notes, `docs/research-living-brain.md`, and current source benchmark reports.

Research requirements: use Hugging Face dataset metadata, NVIDIA Nemotron dataset notes, SNN language datasets, replay/consolidation datasets, and local autonomy/self-evolving agent research where relevant. Record license, intended use, size, sample strategy, and why the data helps MARULHO rather than a generic LLM.

Hard constraints: no training data download in the hot path; no hidden external LLM data generation as cognition; no Wikipedia fallback by habit; no unbounded cache growth; no capability claim from dataset presence alone.

Iteration success criteria: implement or improve a slow-path source/data runner that can select, sample, cache, and report a bounded dataset/source with provenance, license, schema, token count, encoding/device behavior, and an evaluation hook. Runtime Truth or benchmark currency may expose dataset readiness, but must not run acquisition from status reads.

Durable completion condition: complete this goal only when MARULHO has a repeatable slow-path acquisition loop that can inspect, select, cache, and evaluate at least one licensed high-value source with provenance, bounded storage, evaluation hooks, and no hot-path dependency.

Verification: run source/data tests, a small dataset smoke or dry-run, cache integrity checks, and a benchmark/evaluation proving ingestion cost is slow-path. If Hugging Face access is unavailable, report the attempted command/API and blocker.

Docs: update `docs/research-living-brain.md`, vault paper/data notes, source/capability notes, and retired paths for removed stale source assumptions. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if licensing, network, disk, authentication, or dependency setup prevents a defensible local run.
```

## Goal 7: Runtime Truth and Control Room Parity

```text
/goal Advance MARULHO through repeated evidence-backed work cycles toward Runtime Truth and control-room parity with the promoted high-throughput path. Use $marulho-grill-with-docs as the operating skill.

Use `docs/goals/marulho-living-runtime-goal.md` as north-star context and `docs/vault/` as the refined Graphify/Obsidian navigation layer. Treat this goal as the active completion contract across multiple iterations. Do not mark it complete after one implementation/report cycle; after each completed iteration, commit, report evidence, then continue to the next highest-leverage continuation inside this same goal unless blocked, redirected by the user, or the durable completion condition below is actually met.

Objective: make the operator-facing control room show the same promoted runtime path that benchmarks prove, without adding hot-path reporting tax. The UI/status layer should expose true current throughput, long-run evidence, executor mode, CUDA/device truth, contention, fallback reasons, host-truth cadence, and startup cost. It must not imply slow UI demos are the real cognitive velocity, and it must not poll in a way that reduces speed.

Inspect first: `CONTEXT.md`, `docs/vault/concepts/runtime-truth.md`, `docs/vault/maps/runtime-truth-surface-map.md`, `docs/vault/benchmarks/hot-path-latency.md`, service Runtime Truth projection modules, status read model, API schemas/tests, MARULHO_UI only if UI parity is part of the work, and current benchmark reports.

Research requirements: no major external research is required unless changing CUDA/reporting architecture. Use OpenAI goal guidance for completion criteria and MARULHO Runtime Truth docs for evidence boundaries.

Hard constraints: do not synchronize CUDA just to draw UI; do not report decorative CUDA; do not run benchmarks from status reads; do not make `service` absorb trainer algorithms; keep report caches bounded and stale/fresh status explicit.

Iteration success criteria: control room/API evidence must distinguish retained top evidence, current measured speed, contention, active executor, fallback reasons, and cold-start overhead. If a long test is running, status must show progress/lifecycle without blocking or pretending the UI owns the run.

Durable completion condition: complete this goal only when the control room and API consistently expose the promoted high-throughput path, retained long-run evidence, active executor, contention, fallback reasons, cold-start cost, and benchmark lifecycle without adding measurable polling/reporting tax.

Verification: run focused API/status tests, frontend build if UI changed, a smoke service/control-room run if feasible, and one benchmark or report replay proving the displayed data matches report files/Runtime Truth. Measure status latency or justify why unavailable.

Docs: update `CONTEXT.md` if Runtime Truth terms change, `docs/vault/concepts/runtime-truth.md`, benchmark notes, and retired UI/API paths if old status cards/endpoints are removed. Rebuild and validate the vault when docs change.

Commit after each meaningful completed iteration. Stop and report if the server/UI/test environment cannot run or if parity would require adding hot-path report tax.
```

## Recommended Order

Run Goal 1 first. It targets the real throughput ceiling. Run Goal 2 next or in parallel only if the scheduler can feed a compact awake mask into the executor without slowing the fast text path. Goals 3-6 should be slow-path or evaluation-path until they have evidence. Goal 7 is the operator-truth cleanup that keeps the control room aligned with whatever the promoted path actually proves.
