# HECSN / Terminus Living Brain Roadmap

## Summary And Current Truth

The project currently stands as an auditable Terminus runtime: a grounded SNN-style substrate, lazy NVIDIA NIM cortex path, sensory/action/replay evidence APIs, and a refactored service manager facade split into modules. It is not yet a self-improving autonomous brain; replay bundle generation is still preview/export only and does not train, mutate memory, promote facts, execute actions, call tools, or start sleep.

Fresh validation found the core is mostly intact after the refactor: service API, action loop, cortex/thought loop, query/memory primitives, service benchmark, trace export, and UI build all pass in targeted slices. Two regressions block calling `main` green: one stale/refactor-broken query-focus test, and one acceptance harness mock-cortex failure where idle gating returns `partial`.

System DNA to preserve: grounded evidence first, language cortex second, explicit safety gates, measurable liveness, reproducible traces, and no autonomous learning without offline evaluation plus operator approval.

## Key Changes

1. Stabilization PR first
   - Status: complete for the current baseline. The service manager and API suites now pass after the refactor stabilization work.
   - The lazy cortex/mock cortex contract is explicit enough for status/API calls to avoid eager cortex initialization while still reporting cortex availability through runtime truth.
   - Keep the refactor shape: `HECSNServiceManager` remains the composition root; service behavior stays in focused mixins.
   - Top-level docs now state what is true now, what is stale, and which APIs are safety-gated.

2. Runtime truth contract
   - Status: first maintained contract and reporting slices complete.
   - `/status` and `/terminus` now expose `runtime_truth` with `verdict`, `evidence`, `cortex_available`, `memory_pressure`, `replay_role`, `safety_flags`, `latency_ms`, and `recommended_action`.
   - Every “alive/degraded/partial/failed” verdict now includes evidence and a recommended next action on the main operator status surfaces.
   - Service benchmark JSON and long-test JSON/Markdown reports now preserve runtime truth so endpoint latency, liveness, replay safety, and operator action are visible in the same validation artifacts.
   - Keep public endpoints stable unless a field is clearly misleading; add fields instead of breaking existing clients.

3. Replay-to-learning ladder
   - Keep current replay dataset bundle API preview-only by default.
   - Add a separate offline learning gate later: versioned dataset manifest, dedupe, train/eval/holdout split, contamination check, regression benchmark, and explicit operator approval.
   - No memory promotion, adapter training, or behavior policy update may happen from replay data until the offline evaluation passes.

4. Autonomy loop
   - Policy actuator may propose actions; a separate executor validates permissions, expected outcome, rollback plan, and delayed consequence tracking.
   - Actions must be sandboxed, audited, and replayable from traces.
   - Autonomous behavior graduates in levels: observe-only, propose-only, approved execution, constrained recurring execution, then adaptive policy updates.

5. Research-backed direction
   - Use predictive coding and free-energy ideas as the core control model: prediction, error, update, action to reduce expected error.
   - Use hippocampal replay and continual-learning work to guide memory consolidation instead of naive fine-tuning.
   - Use world-model work for future imagination/planning, but only after the current evidence and benchmark layer is reliable.
   - Treat LLM-agent benchmarks as external pressure tests, not as the project’s identity.

## Public Interfaces And Benchmarks

Public API defaults:
- Existing Terminus endpoints remain the main surface: `/terminus/living-loop`, `/terminus/policy-actuator`, `/terminus/replay-plan`, `/terminus/replay-sample`, `/terminus/runtime-traces/export`, `/terminus/replay-dataset/preview`, `/terminus/replay-dataset/bundle`, `/terminus/action`, `/terminus/runtime-feedback`, and cortex endpoints.
- Add or standardize status fields for `verdict`, `evidence`, `cortex_available`, `memory_pressure`, `replay_role`, `safety_flags`, `latency_ms`, and `recommended_action`.
- Add benchmark result records that capture command, git commit, environment, endpoint latencies, pass/fail verdicts, and known limitations.

Benchmark ladder:
- Internal required: unit slices, service API, long acceptance harness, replay bundle safety, runtime trace export, action audit loop, latency benchmark, UI build.
- Live required before major autonomy claims: real NIM cortex long run with cost/credential approval, saved report, trace export, and acceptance verdict.
- External optional pressure tests: [GAIA](https://arxiv.org/abs/2311.12983) for tool-use assistants, [WebArena](https://arxiv.org/abs/2307.13854) for web action reliability, [AgentBench](https://arxiv.org/abs/2308.03688) for multi-environment agents, [ARC-AGI-3](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf) for adaptive reasoning, and [MLE-bench](https://arxiv.org/abs/2410.07095) only if the project becomes an ML-engineering agent.

## Test Plan

Every PR must include:
- A short hypothesis: what behavior should improve and why.
- Focused unit tests for touched modules.
- At least one live or in-process validation command with saved result.
- Benchmark delta for latency, memory pressure, replay quality, or liveness when relevant.
- Documentation update when behavior, API shape, safety policy, or benchmark meaning changes.
- A performance note: current cost, bottleneck, and whether a simpler or faster design was considered.

Immediate green bar:
- `tests/test_service_manager.py`: 127 passed after the runtime truth contract slice
- `tests/test_service_api.py`: 45 passed after the runtime truth contract slice
- `tests/test_long_test_runner.py`: 7 passed after the runtime truth reporting slice
- `tests/test_service_benchmark.py`: 2 passed after the runtime truth reporting slice
- action/cortex/thought/query/memory slices
- service benchmark and trace export tests
- `npm run build` in `HECSN_UI`

## 2026-04-30 Runtime Truth Contract Slice

Implemented:
- Added additive `runtime_truth` contract fields to `/status` and `/terminus`.
- Added focused manager/API tests so the contract cannot silently disappear.
- Kept existing public fields stable.
- Tightened delayed-consequence supportive recovery so well-grounded non-adverse follow-up recovery deterministically reaches the existing support threshold under strict recovery conditions.

Validation:
- `python -m pytest tests/test_service_manager.py -q` — 127 passed
- `python -m pytest tests/test_service_api.py -q` — 45 passed

Follow-up slice:
- Extend `runtime_truth` into benchmark artifacts and long-test reports so endpoint latency, liveness verdicts, replay safety, and recommended operator actions are recorded in the same validation output.

## 2026-04-30 Runtime Truth Reporting Slice

Implemented:
- Service benchmark now probes `/status` and `/terminus`, records endpoint timings, and writes `status_runtime_truth_summary` plus `terminus_runtime_truth_summary`.
- Long-test runner now stores final `runtime_truth`, writes it into JSON/Markdown reports, and treats `partial`, `degraded`, or `failed` truth verdicts as health signals.
- Added maintained tests for both reporting surfaces.

Validated:
- `python -m pytest tests/test_service_benchmark.py -q` - 2 passed
- `python -m pytest tests/test_long_test_runner.py -q` - 7 passed
- `python -m hecsn.evaluation.service_benchmark --create-synthetic-checkpoint ...` - success, `reports/runtime_truth_benchmark_validation.json`, 15 endpoint timings, `status_runtime_truth_summary.verdict=partial`, `recommended_action=configure_terminus_sources`, `training_gate.status=blocked_preview_only`

Next useful roadmap step:
- Start the replay-to-learning gate as a separate offline path: versioned dataset manifest, dedupe, train/holdout/eval split, contamination checks, explicit no-mutation preview tests, then a benchmark gate before any adapter/memory update.

## 2026-04-30 Replay-To-Learning Gate Slice

Implemented:
- Replay dataset bundle preview now exposes a machine-readable `training_gate`.
- The gate is explicitly `blocked_preview_only`, `eligible_for_training=false`, and lists the conditions required before any future training path may consume replay data.
- Service benchmark bundle summaries preserve `training_gate` so benchmark artifacts can show whether replay data is still blocked from learning.

Validated:
- Focused API and benchmark tests cover the blocked training gate and no-training safety boundary.
- Broader affected suites: `tests/test_service_api.py tests/test_service_benchmark.py` - 47 passed; `tests/test_service_manager.py tests/test_long_test_runner.py` - 134 passed.
- Regenerated `reports/runtime_truth_benchmark_validation.json` includes `training_gate.status=blocked_preview_only` and `eligible_for_training=false`.

Next useful roadmap step:
- Build the offline evaluation command behind `run_offline_replay_training_eval_gate`; it should consume a saved bundle, run decontamination/regression checks, write a gate report, and still avoid adapter training or memory mutation until separately approved.

## Assumptions And Sources

Defaults chosen:
- Priority is “stabilize first,” then autonomy.
- No paid/external live NIM validation is required before the first stabilization PR, but it is required before claiming current live-cortex liveness.
- The project keeps NVIDIA NIM as the production cortex path.
- Replay remains preview/export only until a separate learning gate exists.

Research anchors:
- Predictive coding: [Rao & Ballard 1999](https://www.nature.com/articles/nn0199_79)
- Free-energy / active inference: [Friston 2010](https://www.nature.com/articles/nrn2787), [Active Inference Survey](https://arxiv.org/abs/2112.01871)
- Replay and consolidation: [Wilson & McNaughton 1994](https://pubmed.ncbi.nlm.nih.gov/8036517/), [Continual Lifelong Learning Review](https://arxiv.org/abs/1802.07569)
- World models: [Ha & Schmidhuber 2018](https://arxiv.org/abs/1803.10122), [DreamerV3](https://arxiv.org/abs/2301.04104)
