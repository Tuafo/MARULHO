# HECSN Tutorial — Current Operator & Developer Guide

> **Current runtime truth:** HECSN/Terminus is a Subcortex-first SNN system. The former external LLM/NIM path is retired and must not be used as a runtime, mock path, language engine, or fallback.

---

## Table of Contents

1. [What HECSN is now](#1-what-hecsn-is-now)
2. [Current runtime stack](#2-current-runtime-stack)
3. [Installation and environment setup](#3-installation-and-environment-setup)
4. [Training a checkpoint](#4-training-a-checkpoint)
5. [Running the service](#5-running-the-service)
6. [Building and serving the dashboard](#6-building-and-serving-the-dashboard)
7. [Terminus runtime](#7-terminus-runtime)
8. [Retired path, action loop, and sleep control](#8-retired-path-action-loop-and-sleep-control)
9. [API reference](#9-api-reference)
10. [Validation and testing](#10-validation-and-testing)
11. [Extension points](#11-extension-points)
12. [Current limitations](#12-current-limitations)

---

## 1. What HECSN is now

HECSN is a **Subcortex-first cognitive architecture**:

- a **predictive spiking substrate** handles grounding, prediction error, salience, replay, buffering, and local online adaptation
- language, reasoning, narrative continuity, question formation, answer generation, and dream-style recombination must move through HECSN-owned Subcortex/SNN surfaces with explicit grounding and device evidence

The architectural rule is:

> Subcortex owns **when**, **why**, and **how deeply** cognition runs. Retired external LLM paths do not own thought, language, memory, or sleep.

The maintained system is now centered around five backend truths:

1. **runtime correctness**
2. **grounded evidence flow**
3. **real gating**
4. **maintained action/verification loops**
5. **explicit acceptance/health classification**

---

## 2. Current runtime stack

### Retired external LLM side

The old external LLM path is retired and is not part of the active runtime stack.
Do not configure NIM models, LLM factories, or mock language substitutes.

Important:
- production cognition is **Subcortex/Living Loop first**
- the top-level `hecsn.cortex` package is deleted
- historical Cortex submodules are not extension points
- LLM construction must not occur in active service startup, `/health`, `/status`, action execution, or language surfaces

### Text / multimodal runtime side

The maintained quick-start runtime uses one canonical preset:
- **`curriculum`**

Current runtime sources:
- `wikimedia/wikipedia` (`20231101.en`)
- `AlgorithmicResearchGroup/s2orc_arxiv` (`abstract`)
- `HuggingFaceFW/fineweb-edu` (`sample-10BT`)
- `ScienceOne-AI/S1-MMAlign`
- `OpenSound/AudioCaps`
- focus-aware allocation across maintained background text sources
- autonomy-guided real-source acquisition over maintained source catalogs with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility
- real multimodal grounding stays on the Hugging Face sensory streams

### Maintained action loop

The maintained digital action surface currently supports:
- `workspace_search`
- `workspace_read`
- `web_fetch`
- `api_request`

These all share the same maintained path for:
- execution
- verification
- provenance
- persistence
- replay

### Maintained sleep control

Sleep/consolidation belongs to Subcortex/Living Loop maintenance. Retired
Retired-path sleep endpoints and retired-intent sleep paths are not maintained product
surfaces.

### Acceptance / long-test path

The maintained long-test runner now:
- runs a deterministic local acceptance harness
- classifies runs as:
  - `alive`
  - `degraded`
  - `dead`
- emits non-zero exit codes for degraded/dead runs

---

## 3. Installation and environment setup

### Prerequisites

- Python 3.10+
- Node.js 20+
- PyTorch 2.x
- Hugging Face access if you want the live HF-backed runtime to behave well

### Python dependencies

The minimal maintained dependency surface is defined in `pyproject.toml`.

Typical install:

```bash
pip install -e .
```

or, if you need a quick direct install:

```bash
pip install torch numpy fastapi uvicorn[standard] datasets httpx
```

### Environment

Create a `.env` in the repo root:

```bash
HF_TOKEN=your_hf_token_here
```

Recommended optional overrides:

```bash
HECSN_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

### Runtime behavior when env is missing

- if `HF_TOKEN` is missing, the live HF runtime may still work, but with worse limits / slower behavior

This is different from older documentation that implied local Ollama/Gemma or
NVIDIA NIM operation. Those are no longer production paths.

---

## 4. Training a checkpoint

Training still begins from the SNN side. The normal entrypoint is the developmental runner.

### Developmental training

```bash
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir checkpoints/terminus \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-field text \
  --max-tokens 500000
```

### Developmental validation protocol

```bash
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir reports/developmental \
  --n-tokens 5000
```

### Supported evaluation and health checks

```bash
PYTHONPATH=src python -m hecsn.training.meaning_grounding_runner \
  --output-dir reports/meaning_grounding
```

For runtime acceptance/health reporting, use:

```bash
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/long_test
```

The old `hecsn.training.emergence_evaluation_runner` module is no longer available.
Checkpoint output is a `.pt` file used by the local service.

---

## 5. Running the service

### Start the API service

```bash
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt \
  --port 8000 \
  --trace-dir reports/service/traces
```

### Useful flags

| Flag | Meaning |
|---|---|
| `--checkpoint` | checkpoint `.pt` file to load |
| `--host` | bind host |
| `--port` | bind port |
| `--trace-history-limit` | in-memory trace cap |
| `--trace-dir` | persisted trace output directory |
| `--web-dist-dir` | built frontend directory to serve at `/app`; defaults to `HECSN_UI/dist` |
| `--log-level` | uvicorn log level |
| `--reload` | dev autoreload |

### Verify it is up

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## 6. Building and serving the dashboard

The active frontend source currently lives in:
- `HECSN_UI/`

### Development mode

```bash
cd HECSN_UI
npm install
npm run dev
```

### Production build

```bash
cd HECSN_UI
npm run build
```

The service defaults to this build directory:

```bash
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt
```

If the build directory exists, the service mounts it at:
- `/app`

---

## 7. Terminus runtime

Terminus is the maintained long-running backend runtime. The canonical current path is:

`observe -> predict -> error/salience/drives -> reason/act -> verify -> typed memory -> replay/consolidation -> self-model update`

In practice this means:

1. observe configured sources, sensory previews, operator questions, and action outcomes
2. predict likely outcomes before acting or answering
3. convert prediction error, salience, uncertainty, novelty, fatigue, and drive pressure into control signals
4. deliberate through Subcortex/Living Loop surfaces and act through the maintained digital action loop
5. verify action/answer outcomes against grounded evidence
6. write typed memory with provenance such as `observed`, `inferred`, `dreamed`, `verified`, and `contradicted`
7. replay and consolidate memories, delayed consequences, source utility, and long-horizon evidence
8. update the operational self-model so capabilities, limits, budgets, memory health, and grounding health stay visible

### Current quick-start preset

There is currently **one maintained quick-start preset**:
- `curriculum`

It configures:
- 3 HF background text sources
- 2 live sensory sources
- focus-aware allocation across the maintained background text registry
- autonomy-guided targeted acquisition over the maintained real-source registry with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility
- bounded `tick_tokens` for better short-run visibility
- hypercube binding + cross-modal runtime support via model overrides

### Terminus runtime commands

#### List presets

```bash
curl http://localhost:8000/terminus/presets
```

#### Quick-start the maintained runtime

```bash
curl -X POST "http://localhost:8000/terminus/quick-start?preset=curriculum"
```

#### Inspect runtime state

```bash
curl http://localhost:8000/terminus
```

#### Inspect the living loop/self-model surface

```bash
curl http://localhost:8000/terminus/living-loop
```

`/terminus/living-loop` returns the same living-loop snapshot embedded in `/status` under `terminus_runtime.living_loop`, plus wrapper status fields such as `dirty_state`, `state_revision`, and `token_count`.

Key telemetry fields:
- `prediction_count` — number of surfaced predictions in the current snapshot
- `action_count` — number of surfaced action records in the current snapshot
- `world_model_lite` — derived prediction/verification scoring, contradiction rate, policy score, and recommended next action
- `skill_memories` — action/tool memories derived from verified, contradicted, and mixed action history
- `capabilities` — currently observed operational abilities such as action execution, prediction tracking, verification tracking, and world-model-lite policy scoring
- `limits` — supported actions, snapshot/history counts, truncation state, memory capacity/fill, revision, and runtime flags
- `budgets` — action-history use, snapshot use, world-model policy budget/cost/risk/uncertainty, and memory use
- `memory_health` — capacity/fill/provenance health for the surfaced memory snapshot
- `grounding_health` — verification, evidence, contradiction, feedback counts, and grounding status for the current loop state
- `feedback_summary`, `feedback_count`, `verified_feedback_count`, `contradicted_feedback_count`, `unverified_feedback_count`, and `recent_feedback` — operator/runtime review state folded back into the living loop
- `replay_plan` — advisory prioritized replay/consolidation candidates derived from feedback, uncertainty, memory pressure, latency, and policy signals
- `replay_sample_summary` and `replay_executor_summary` — identical audit summaries for the operator-gated replay sampler/executor history, including selected counts, mode/status counts, safety flags, and endpoint aliases

### Policy actuator

`GET /terminus/policy-actuator` exposes the current living-loop policy recommendation as a small, advisory-only actuator surface. It returns the same decision embedded in `/terminus/living-loop` as `policy_decision`, but without requiring clients to parse the full self-model snapshot.

Typical usage:

```bash
curl http://localhost:8000/terminus/policy-actuator
```

Response fields:
- `schema_version` — policy-actuator response schema
- `action` — normalized recommendation key such as `investigate_contradictions`, `verify_pending_evidence`, `consolidate_or_sleep`, `reduce_scope_or_wait`, `collect_more_evidence`, or `continue_current_policy`
- `recommendation` — operator-readable explanation for the selected action
- `reasons` — ordered reason objects with stable `code` and human-readable `detail`
- `risk`, `expected_information_gain`, `expected_goal_progress`, `expected_cost`, and `uncertainty` — bounded scores used for monitoring and downstream benchmark summaries
- `target_episode_id`, `target_action_id`, and `action_id` — optional target references; `action_id` mirrors the targeted action when present
- `suggested_endpoint`, `suggested_input`, and `input` — a suggested operator/API handoff payload, not an automatic execution request
- `advisory=true`, `executable=false`, and `created_at`

Recommendation priority is intentionally conservative:

1. **Contradictions first**: contradicted feedback, predictions, actions, episodes, or `grounding_health.status=contradictions_present` select `investigate_contradictions`.
2. **Pending evidence next**: unverified feedback, pending predictions, unverified actions, or pending/unverified runtime episodes select `verify_pending_evidence`.
3. **Maintenance pressure before new evidence**: memory fill at or above `0.90`, retired-runtime fatigue at or above `0.70`, or an already-sleeping retired runtime path selects `consolidate_or_sleep`.
4. **Cost/latency pressure before exploration**: high endpoint latency, policy cost at or above `0.80`, or budget use at or above `0.80` selects `reduce_scope_or_wait`.
5. **Uncertainty after safety/cost checks**: high uncertainty, unknown predictions, or surfaced uncertain domains select `collect_more_evidence`.
6. **Healthy state fallback**: if no pressure crosses a threshold, the actuator returns `continue_current_policy`.

The safety boundary is strict: the policy actuator **does not execute actions, mutate action history, advance state revision, start sleep, post feedback, call the retired runtime path, or change runtime configuration**. `suggested_endpoint` and `suggested_input` are operator guidance for a separate deliberate call. This keeps policy selection visible and benchmarkable while preserving the existing verification-first action path.

Policy-actuator fields are surfaced consistently across telemetry/export/benchmark paths:
- telemetry: `/terminus/living-loop` includes `policy_decision`, and `benchmark_telemetry.policy_recommendations` includes `total`, `latest`, `counts`, and outcome scores such as `information_gain`, `goal_progress`, `risk`, and `uncertainty`
- export: `GET /terminus/runtime-traces/export` includes a sanitized top-level `policy_decision`; each exported example also includes sanitized `policy_decision` with `action`, `recommendation`, `reason_codes`, score fields, target IDs, `advisory`, `executable`, and `suggested_endpoint`; the CLI export metadata mirrors that summary
- benchmark: `src\hecsn\evaluation\service_benchmark.py` exercises `GET /terminus/policy-actuator`, records it in `endpoint_timings` / `endpoints_by_name.policy_actuator`, and writes `policy_actuator_summary` with the actuator scores, target IDs, `advisory`, `executable`, `suggested_endpoint`, and `reason_codes`

This is the next living-loop step because Terminus can now turn observed outcomes, feedback, grounding health, memory pressure, latency/cost pressure, and uncertainty into an explicit policy recommendation without crossing the safety boundary into autonomous actuation. Operators and tests can inspect why the loop would investigate, verify, sleep, reduce scope, collect evidence, or continue before any separate action is taken.

### Replay and consolidation planner

`GET /terminus/replay-plan?limit=20` exposes the current prioritized replay/consolidation plan. It is the bridge between feedback-grounded traces and later learning work: runtime episodes, action outcomes, predictions, uncertain domains, memory pressure, latency pressure, and policy decisions are converted into ranked candidates that an operator or future training sampler can inspect.

Typical usage:

```bash
curl "http://localhost:8000/terminus/replay-plan?limit=10"
```

Response fields:
- `schema_version`, `generated_at`, `endpoint`, `limit`, `count`, `state_revision`, and `token_count`
- `advisory=true` and `executable=false`
- `priority_rules_version=deterministic-v1`
- `priority_weights` for `safety`, `feedback`, `uncertainty`, `memory_pressure`, `latency_pressure`, `policy_pressure`, `provenance_gap`, and `recency_rank`
- `plan_reason_codes` — the reason-code union across returned candidates
- `snapshot_counts` — counts for runtime episodes, actions, predictions, feedback, and uncertain domains used to build the plan
- `candidates` — ranked replay targets with `candidate_id`, `rank`, `target_type`, `target_id`, `target_ids`, `operation`, `reason_codes`, `priority_score`, `priority_components`, `suggested_consolidation_action`, `suggested_endpoint`, `suggested_input`, `summary`, `provenance`, `risk`, `uncertainty`, `latency`, `memory_health`, `feedback`, and `policy`

Candidate `target_type` can be `runtime_episode`, `action`, `prediction`, `feedback`, `memory_health`, `uncertain_domain`, or `policy_decision`. Stable reason codes include `contradicted_feedback`, `contradicted_runtime_episode`, `contradicted_action`, `failed_runtime_episode`, `corrected_output_available`, `unverified_feedback`, `pending_prediction`, `unverified_action`, `high_uncertainty`, `uncertain_domain`, `memory_capacity_pressure`, `fatigue_sleep_pressure`, `high_latency`, `high_cost`, `high_budget_use`, and `healthy_grounded_state`.

Suggested consolidation actions are also advisory-only: `review_contradiction`, `verify_pending_evidence`, `replay_episode_for_grounding`, `sleep_consolidation_advisory`, `reduce_scope_or_wait`, `collect_more_evidence`, or `continue_observing`. The endpoint **does not start sleep, replay memories, train adapters, mutate runtime state, post feedback, or execute actions**. It ranks what should be reviewed or consolidated next.

Research rationale: this wave follows complementary learning systems and hippocampal replay (McClelland et al. 1995; Wilson & McNaughton 1994), reward/salience-biased replay (Singer & Frank 2009; Yang et al. 2024), prioritized experience replay and value-of-computation memory access (Schaul et al. 2015; Mattar & Daw 2018), world-model replay (Ha & Schmidhuber 2018; Hafner et al. 2019/2020), active-inference uncertainty reduction (Friston et al. 2015), and agent skill/reflection replay patterns such as Voyager and Reflexion. The current endpoint uses deterministic ranking for auditability; a future trainer should sample replay stochastically with capped `P(i) proportional to (epsilon + score)^alpha`, a uniform diversity floor, and strict separation between observed/verified memories and dreamed or synthetic hypotheses.

Replay-plan fields are surfaced consistently across telemetry/export/benchmark paths:
- telemetry: `/terminus/living-loop` and `/status` include `replay_plan`; `benchmark_telemetry.replay_plan_summary` mirrors the count, reason codes, top candidate, and advisory/executable posture
- export: `GET /terminus/runtime-traces/export` includes a sanitized top-level `replay_plan_summary`; each exported example also includes sanitized `replay_plan_summary`; the CLI export metadata mirrors that summary
- benchmark: `src\hecsn\evaluation\service_benchmark.py` exercises `GET /terminus/replay-plan`, records it in `endpoint_timings` / `endpoints_by_name.replay_plan`, and writes `replay_plan_summary`

What remains before autonomous learning: replay candidates still require an operator or separate training process to select, validate, and apply them. Contradicted memories should become negative lessons or corrected examples, not facts. Dreamed/imagination memories must remain provenance-tagged and cannot be consolidated as observed/verified without external validation.

### Operator-gated replay sampler/executor

`POST /terminus/replay-sample` records an operator-confirmed replay sample from the current replay plan. `POST /terminus/replay-execute` is an alias for the same guarded path; when `mode="execute"` the response also sets `execution_id`, but execution still means **audit execution only**. The endpoint revalidates the live `GET /terminus/replay-plan` candidates, samples or selects candidates, records a bounded sanitized history item, and returns before/after counters proving that learning/action state did not move.

Typical sample request:

```bash
curl -X POST http://localhost:8000/terminus/replay-sample \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "sample",
    "operator_id": "operator-a",
    "operator_note": "Review the highest-value replay candidate before training.",
    "confirmation": true,
    "count": 2,
    "alpha": 1.0,
    "seed": 42
  }'
```

Typical audit-execute request for one already inspected candidate:

```bash
curl -X POST http://localhost:8000/terminus/replay-execute \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "execute",
    "candidate_id": "candidate-id-from-replay-plan",
    "target_type": "runtime_episode",
    "target_id": "episode-id-from-replay-plan",
    "operator_id": "operator-a",
    "operator_note": "Confirmed this candidate should become a reviewed lesson later.",
    "confirmation": true
  }'
```

Request fields:
- `mode` — `dry_run`, `sample`, or `execute`; all modes are audit/sample only, and `execute` does not train, sleep, promote memory, post feedback, execute digital actions, or call outside services
- `candidate_id` — optional exact replay-plan candidate to revalidate; stale or missing candidate IDs return HTTP 422
- `target_type` and `target_id` — optional guards that must match every selected candidate before a record is accepted
- `operator_id` — required non-empty operator identifier for the audit trail
- `operator_note` — optional bounded note explaining the review intent
- `confirmation` — must be `true`; this is the human-in-the-loop gate
- `limit` / `count` — requested number of candidates, clamped to the maintained replay-sample maximum
- `alpha` — PER-style priority exponent clamped to `0.0..4.0`; `0.0` is close to uniform, larger values favor high `priority_score`
- `seed` — optional deterministic seed for reproducible stochastic sampling

Response fields:
- `schema_version`, `replay_sample_id`, optional `execution_id`, `created_at`, `mode`, `status`, `reason`, and `endpoint`
- `operator_id`, `operator_note`, `requested_candidate_id`, `target_type`, `target_id`, `requested_count`, `alpha`, and `seed`
- `candidate_ids` — current replay-plan candidate IDs considered by the sampler
- `selected_candidate_ids` and `selected_candidates` — sanitized selected candidate payloads; each selected candidate includes `safety` flags such as `audit_only`, `not_promoted`, `non_factual`, `negative_lesson`, and `dreamed_or_synthetic`
- `safety_checks` — confirmation, live candidate revalidation, optional target-guard result, bounded-count check, maximum count, and the safety boundaries
- `safety_flags` — audit-level proof flags including `training_started=false`, `sleep_started=false`, `memory_verification_promoted=false`, `feedback_posted=false`, `digital_action_executed=false`, `external_calls_made=false`, and `not_promoted=true`
- `before` / `after` — `token_count`, `state_revision`, `action_history_count`, and `feedback_count` snapshots used to verify no learning/action state changed
- `plan_summary` — sanitized summary of the replay plan used for the selection

History endpoints are `GET /terminus/replay-sample/history` and `GET /terminus/replay-execute/history`:

```bash
curl "http://localhost:8000/terminus/replay-sample/history?limit=20"
curl "http://localhost:8000/terminus/replay-execute/history?limit=20"
```

Both history endpoints return `schema_version`, `endpoint`, total `count`, requested `limit`, and recent `history` records. `/terminus/replay-execute/history` is an alias over the same replay-sample audit history.

Sampling semantics and research rationale: when `candidate_id` is omitted, Terminus uses a seeded prioritized-experience-replay sampler without replacement. Candidate weight is `P(i) proportional to (epsilon + score)^alpha`, where `score` is the replay-plan `priority_score`; a diversity dampening step reduces repeat target-type dominance so review batches do not collapse onto one candidate family. This keeps the deterministic replay plan auditable while allowing reproducible stochastic review batches. The rationale combines PER with human-in-the-loop safe RL and active learning: surface high-value, high-uncertainty, or contradiction-heavy candidates to an operator, but keep training and promotion in a separate verified process.

Safety boundary: this endpoint is **operator-gated audit/sample only**. It does not start training, start sleep, promote memory verification, post feedback, execute digital actions, make external calls, or convert dreamed/synthetic material into facts. Contradicted candidates are negative lessons or correction targets, not verified knowledge. Dreamed, synthetic, simulated, contradicted, or failed candidates remain provenance-tagged and must be externally validated before any future training or memory-promotion path can use them as positive examples. The only persistent effect is the sanitized replay-sample history record and its audit summary.

Replay-sample/executor fields are surfaced consistently across telemetry/export/benchmark paths:
- telemetry: `/terminus/living-loop` and `/status` include `living_loop.replay_sample_summary` and `living_loop.replay_executor_summary`; `benchmark_telemetry.replay_sample_summary` and `benchmark_telemetry.replay_executor_summary` mirror endpoint aliases, counts, latest selected item, mode/status counts, safety boundaries, and audit-only/executable posture
- export: `GET /terminus/runtime-traces/export` includes sanitized top-level `replay_sample_summary` and `replay_executor_summary`; each exported example includes sanitized `replay_sample_summary`; `python -m hecsn.service.trace_export_runner` mirrors both summaries in CLI metadata
- benchmark: `src\hecsn\evaluation\service_benchmark.py` exercises `GET /terminus/replay-sample/history`, records it in `endpoint_timings` / `endpoints_by_name.replay_sample_history`, and writes both `replay_sample_summary` and `replay_executor_summary` with history-derived counts and safety flags
- UI: the current dashboard remains read-only for replay. It renders the advisory replay-plan card and does not post to `/terminus/replay-sample` or `/terminus/replay-execute`; operators should use the API/CLI until a dedicated gated sampler form and history panel are added

Remaining work before autonomous replay learning: add an operator UI for candidate selection/history review, define a separate offline trainer or adapter-distillation job that consumes only externally validated examples, add promotion rules that distinguish negative lessons from positive facts, enforce retention/audit review for replay-sample history, and keep benchmark claims limited to sampling/telemetry until a verified training path exists.

### Curated replay dataset preview

`GET /terminus/replay-dataset/preview` turns sanitized runtime traces, evaluator feedback, replay-plan context, and replay-sample history into a read-only dataset preview for future adapter/distillation work. It is intentionally an export surface, not a trainer.

Typical usage:

```bash
curl "http://localhost:8000/terminus/replay-dataset/preview?limit=20"
curl "http://localhost:8000/terminus/replay-dataset/candidates?limit=20"
curl "http://localhost:8000/terminus/replay-dataset/history?limit=20"
curl -X POST "http://localhost:8000/terminus/replay-dataset/bundle" \
  -H "Content-Type: application/json" \
  -d '{"operator_id":"operator-a","confirmation":true,"limit":20,"holdout_fraction":0.2,"eval_fraction":0.2,"seed":17}'
```

The equivalent CLI helper writes the same preview shape to disk:

```bash
PYTHONPATH=src python -m hecsn.service.replay_dataset_runner \
  --checkpoint checkpoints/terminus/model.pt \
  --output reports/replay_dataset.json \
  --limit 20

PYTHONPATH=src python -m hecsn.service.replay_dataset_bundle_runner \
  --checkpoint checkpoints/terminus/model.pt \
  --output reports/replay_dataset_bundle.json \
  --operator-id operator-a \
  --confirm \
  --limit 20
```

Preview response fields:
- `schema_version`, `export_kind=terminus_replay_dataset_preview`, `training_role=replay_dataset_preview_only_not_training_no_mutation`, `description`, `created_at`, `latest_export_timestamp`, `latest_history_timestamp`, `endpoint`, `limit`, `max_limit`, `filter_endpoint`, and `count`
- `positive_count` and `negative_count` so verified/corrected examples stay separate from contradicted/failed lessons
- `provenance_counts` and `example_type_counts` for auditability
- `policy_decision`, `replay_plan_summary`, `replay_sample_summary`, and `replay_executor_summary` linking examples back to the current living-loop context
- `safety_flags`, `before`, and `after` proving the export did not start training, mutate memory, post feedback, execute digital actions, or make external calls
- `items` containing sanitized SFT/DPO-style preview records with provenance labels, target IDs, replay-plan reasons, sample links, positive/negative role markers, and excluded raw fields
- `empty_reason` when the checkpoint has no eligible sanitized runtime traces

`GET /terminus/replay-dataset/candidates` exposes the replay-plan candidates that could feed a dataset preview. `GET /terminus/replay-dataset/history` exposes replay-sample history through the same preview-only safety boundary, using `/terminus/replay-sample/history` as its source.

`POST /terminus/replay-dataset/bundle` is the first packaging gate. It requires `operator_id` plus `confirmation=true`, then converts the preview into a versioned bundle artifact with deterministic deduplication, contamination filters, retention filtering, and train/holdout/eval splits. Response fields include `bundle_id`, `bundle_version`, `bundle_hash`, `source_preview_hash`, `operator_approval`, `packaging_policy`, `split_counts`, `split_summaries`, `manifest`, `splits`, and `excluded_items`. The bundle endpoint remains preview/export only: `safety_flags.training_started=false`, `memory_mutated=false`, `feedback_posted=false`, `digital_action_executed=false`, `external_calls_made=false`, and `requires_separate_training_approval=true`.

Provenance and training-role rules:
- `operator_verified` and `corrected` examples can become positive SFT-style candidates only after external review.
- `contradicted`, `failed`, and rejected examples remain negative lessons or DPO rejected-side candidates; they are not facts.
- `observed` examples without verification remain evidence candidates, not positive training truth.
- `dreamed_synthetic` or simulated content must stay provenance-tagged and must not enter verified fact channels without external validation.
- Benchmark, ARC, and held-out evaluation data must be excluded from future training datasets to avoid contamination.

Replay-dataset fields are surfaced consistently across telemetry/export/benchmark/UI paths:
- telemetry: `/terminus/living-loop`, `/status`, and `/terminus` include `replay_dataset_summary`; `benchmark_telemetry.replay_dataset_summary` mirrors count, positive/negative counts, provenance/type counts, latest timestamps, endpoint, safety flags, and empty reason
- export: `GET /terminus/runtime-traces/export` includes top-level `replay_dataset_summary`; `python -m hecsn.service.trace_export_runner` mirrors the summary in CLI metadata
- benchmark: `src\hecsn\evaluation\service_benchmark.py` exercises `/terminus/replay-dataset/preview`, `/terminus/replay-dataset/bundle`, `/terminus/replay-dataset/candidates`, and `/terminus/replay-dataset/history`, then writes `replay_dataset_summary`, `replay_dataset_bundle_summary`, `replay_dataset_candidates_summary`, and `replay_dataset_history_summary`
- UI: the Runtime dashboard shows a read-only Curated replay dataset card with example counts, positive/negative split, provenance/type counts, endpoint, timestamps, empty reason, and mutation-boundary safety flags

Safety boundary: replay datasets and bundles are **preview/export artifacts only**. They do not train adapters, rewrite or promote memories, post feedback, execute digital actions, call external tools, start sleep, or convert contradicted/dreamed content into verified facts. The packaging gate validates schema, decontamination, retention, deduplication, and holdout/eval splits, but a separate offline trainer approval is still required before any LoRA/QLoRA adapter job runs.

### Feedback-grounded living loop

`POST /terminus/runtime-feedback` is the maintained operator/evaluator path for feeding review results back into runtime episodes and action records. It closes the loop after `/feed`, `/respond`, `/query`, or `/terminus/action` by letting external verification update the target instead of leaving evidence quality as a static trace artifact.

Typical usage:

```bash
curl -X POST http://localhost:8000/terminus/runtime-feedback \
  -H "Content-Type: application/json" \
  -d '{
    "target_type": "runtime_episode",
    "target_id": "episode-id-from-runtime-response",
    "verdict": "verified",
    "confidence": 0.91,
    "summary": "Manual review found the answer grounded in cited evidence.",
    "evidence": [{"note": "reviewed citation span"}],
    "tags": ["reviewed"],
    "evaluator_id": "operator-a"
  }'
```

Action records use the same endpoint with `"target_type": "action"` and an `action_id` from `/terminus/action` or `/terminus/actions`. Optional `corrected_output` can hold the corrected answer/action result; when present, the applied status is treated as `contradicted` so the correction is auditable. The response returns `accepted`, the normalized `feedback`, the updated `target`, `dirty_state`, `state_revision`, and a fresh `terminus_runtime` snapshot. Unknown targets or unsupported verdicts return HTTP 422.

Verdict semantics:
- `verified` means the target survived review. The target verification status becomes `verified`, `verification.success=true`, confidence/summary/evidence are retained, and provenance is promoted to `verified`.
- `contradicted` means the target was wrong, unsupported by the cited evidence, or was submitted with `corrected_output`. The verification status and provenance become `contradicted`, `verification.success=false`, `verification.contradiction=true`, and the correction is stored when supplied.
- `unverified` means the target is not accepted as grounded yet. The verification status/provenance become `unverified`, `verification.success=false`, and the item remains visible for later review without claiming support or contradiction.

Feedback changes the surfaced living-loop state immediately:
- provenance: targets receive `feedback_status`, `feedback_provenance`, `last_feedback_at`, and `verification.last_feedback_id`; action targets also carry top-level feedback provenance, while runtime episodes are promoted at top level when verified or contradicted
- grounding health: `/terminus/living-loop` and `/status` include feedback counts in `grounding_health`; contradictions set `status=contradictions_present`, while unresolved unverified feedback can move otherwise grounded state to `needs_verification`
- export: `GET /terminus/runtime-traces/export` includes sanitized per-example `feedback` and `feedback_summary` fields so offline dataset previews can distinguish reviewed, contradicted, and pending traces without exposing raw secrets
- benchmarks: `benchmark_telemetry.feedback`, `living_loop_benchmark_telemetry.feedback`, and service-benchmark `feedback_telemetry` expose `feedback_count`, verdict/status counts, recent feedback, and `grounding_impact` alongside endpoint latency and policy telemetry

This advances the living-loop goal because Terminus no longer only observes, predicts, acts, and logs. It can ingest review outcomes, revise provenance/grounding health, carry corrections into trace export and benchmark telemetry, and make the next self-model snapshot reflect what was actually verified, contradicted, or still unresolved.

### Optimized latency behavior

The maintained service has one canonical optimized query path:

- `POST /query` is the routing/evidence endpoint. There is **no separate fast query API**, `/fast-query` route, or alternate backend path to keep in sync.
- Query latency is optimized inside `src\hecsn\training\query_runner.py`: semantic query-term matching uses bounded in-request caches for evidence units, token forms, and term-pair matches. Callers should use `/query`; they should not bypass it for a special "fast" mode.
- `POST /feed` uses the normal trainer `train_step` path, but the request handler passes `allow_sleep_maintenance=False`. Due micro/deep sleep maintenance is counted as `sleep_maintenance_deferred` and surfaced with `sleep_maintenance_allowed=false` in the feed result instead of blocking the request. Background/runtime trainer behavior remains unchanged when sleep maintenance is allowed.
- Service startup plus `/health` and `/status` must not construct retired external LLM, NIM, or external embedder paths. Query and action latency belongs to maintained Subcortex/Living Loop surfaces.
- Benchmark telemetry is embedded in the living-loop snapshot. Inspect `benchmark_telemetry` via `GET /terminus/living-loop` or under `terminus_runtime.living_loop.benchmark_telemetry` in `GET /status`. Useful fields include `endpoint_latency_ms`, `tokens_per_second`, `retired_external_adapter`, `cache`, `action_success`, `verification_success`, and `policy_recommendations`.

#### Stop runtime

```bash
curl -X POST http://localhost:8000/terminus/stop
```

### Custom runtime configuration

If you do not want the canonical preset, use `/terminus/configure` with explicit `source_bank`, `ingestion`, `sensory`, and `autonomy` payloads.

Example local file configuration:

```bash
curl -X POST http://localhost:8000/terminus/configure \
  -H "Content-Type: application/json" \
  -d '{
    "source_bank": [
      {
        "name": "local_text",
        "source": "C:/data/notes.txt",
        "source_type": "file",
        "text_field": "text",
        "topic_terms": ["systems neuroscience memory plasticity"],
        "metadata": {"label": "systems neuroscience notes"}
      }
    ],
    "tick_tokens": 20,
    "sleep_interval_seconds": 0.01,
    "repeat_sources": false,
    "ingestion": {
      "enabled": true,
      "queue_target_tokens": 40,
      "prewarm_on_startup": false,
      "prewarm_max_seconds": 0.2
    }
  }'
```

Manual tick:

```bash
curl -X POST http://localhost:8000/terminus/tick \
  -H "Content-Type: application/json" \
  -d '{"steps": 2}'
```

---

## 8. Retired path, action loop, and sleep control

### Retired external LLM runtime

The old external LLM runtime path is deleted. These names are not active product API
or extension points:
- `ThoughtLoop`
- `DriveSystem`
- `ThalamicGate`
- `WorkingMemory`
- `NarrativeSelf`
- `EpisodicMemory`

### Programmatic cognition boundary

Production code should use Subcortex/Living Loop service surfaces. Do not
create a ThoughtLoop or external LLM runtime:

```python
from hecsn.service.manager import HECSNServiceManager

manager = HECSNServiceManager("checkpoints/terminus/model.pt")
status = manager.runtime_facade.living_loop_status()
```

### Maintained action loop

The maintained action loop is backend-first and verification-first.

Supported actions:
- `workspace_search`
- `workspace_read`
- `web_fetch`
- `api_request`

API examples:

```bash
curl -X POST http://localhost:8000/terminus/action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "workspace_search",
    "query_text": "cats chase mice",
    "predicted_outcome": "I expect grounded workspace evidence about cats chasing mice."
  }'
```

```bash
curl http://localhost:8000/terminus/actions
```

### Maintained sleep control

Use Living Loop and replay surfaces for maintenance review. Retired
ThoughtLoop query, thought-stream, and sleep endpoints are not part of the
maintained path.

---

## 9. API reference

### Core service endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness check |
| `GET` | `/status` | full service/runtime snapshot |
| `GET` | `/architecture` | architecture summary |
| `GET` | `/traces` | recent trace history |
| `GET` | `/datasets` | current runtime datasets |
| `GET` | `/stream/status` | SSE telemetry stream |
| `POST` | `/feed` | feed text into the checkpoint-backed runtime, deferring due sleep maintenance for request latency |
| `POST` | `/query` | canonical optimized routing/evidence query without answer synthesis |
| `POST` | `/respond` | grounded answer generation |
| `POST` | `/grounding-probe/run` | grounding probe |
| `GET` | `/checkpoints` | list checkpoints |
| `POST` | `/checkpoint/save` | save checkpoint |
| `POST` | `/checkpoint/restore` | restore checkpoint |

### Terminus runtime endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/terminus` | current Terminus runtime snapshot |
| `GET` | `/terminus/living-loop` | living-loop telemetry and operational self-model snapshot |
| `GET` | `/terminus/policy-actuator` | advisory-only living-loop policy recommendation |
| `GET` | `/terminus/replay-plan` | advisory replay/consolidation candidate plan |
| `POST` | `/terminus/replay-sample` | operator-gated audit sampler over replay-plan candidates |
| `POST` | `/terminus/replay-execute` | alias for replay-sample audit execution mode; no training/action side effects |
| `GET` | `/terminus/replay-sample/history` | replay sampler audit history |
| `GET` | `/terminus/replay-execute/history` | replay executor alias over the same audit history |
| `GET` | `/terminus/presets` | list maintained quick-start presets |
| `POST` | `/terminus/quick-start` | apply preset and start runtime |
| `POST` | `/terminus/configure` | explicit runtime configuration |
| `POST` | `/terminus/start` | start background runtime |
| `POST` | `/terminus/stop` | stop background runtime |
| `POST` | `/terminus/tick` | manual step-based tick |
| `GET` | `/terminus/actions` | action history |
| `POST` | `/terminus/action` | execute maintained digital action |
| `POST` | `/terminus/runtime-feedback` | record verified/contradicted/unverified feedback for runtime episodes or actions |
| `GET` | `/terminus/sensory/recent` | recent sensory previews |
| `GET` | `/terminus/runtime-traces/export` | sanitized runtime trace dataset preview |
| `GET` | `/terminus/replay-dataset/preview` | curated replay dataset preview |
| `POST` | `/terminus/replay-dataset/bundle` | operator-approved preview-only dataset packaging gate |
| `GET` | `/terminus/replay-dataset/candidates` | replay-plan candidates for dataset preview |
| `GET` | `/terminus/replay-dataset/history` | replay-sample history for dataset preview |

---

## 10. Validation and testing

### Full suite

```bash
python -m pytest -q
```

### Useful slices

```bash
# service + runtime
python -m pytest tests/test_service_api.py tests/test_service_manager.py -q

# retired cortex boundary and language packet/result contracts
python -m pytest tests/test_language_contracts.py tests/test_p1_improvements.py -q

# action loop
python -m pytest tests/test_action_loop.py -q

# latency posture / benchmark telemetry docs
python -m pytest tests/test_living_loop_primitives.py tests/test_query_runner.py tests/test_memory_consolidation.py -q

# long-test / acceptance path
python -m pytest tests/test_long_test_runner.py -q
```

### Long-test runner

```bash
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/wp06_validation
```

The maintained long-test path now:
- runs the deterministic local acceptance harness
- emits `alive` / `degraded` / `dead`
- writes markdown + JSON reports

### Runtime trace dataset export

`GET /terminus/runtime-traces/export` and `python -m hecsn.service.trace_export_runner` expose bounded, sanitized runtime episode traces for offline evaluation-data inspection. The export is a **dataset preview only** (`training_role=adapter_distillation_dataset_preview_only_not_training`); it does not train adapters and should not include raw environment, path, secret, credential, password, cookie, authorization, or unbounded token fields.

CLI usage:

```powershell
python -m hecsn.service.trace_export_runner --checkpoint checkpoints\terminus\model.pt --output reports\runtime_trace_examples.json --limit 20 --endpoint respond
```

Useful options:
- `--checkpoint` — checkpoint containing persisted Terminus runtime episode traces
- `--output` — JSON output path; omit it or pass `-` to write JSON to stdout
- `--limit` — maximum examples to export, clamped to the maintained export limit
- `--endpoint` / `--type` — optional operation filter such as `respond`, `query`, `/respond`, or `/terminus/tick`
- `--trace-dir` and `--env-root` — service-manager construction paths used while loading the checkpoint
- `--indent` — JSON indentation

Top-level trace dataset fields include:
- `export_kind=terminus_runtime_trace_dataset_preview`
- `schema_version`
- `training_role`
- `description`
- `limit`, `max_limit`, `endpoint`, and `count`
- `policy_decision`, `replay_plan_summary`, `replay_sample_summary`, and `replay_executor_summary`
- `examples`
- `excluded_fields`
- CLI-only `metadata` with `source`, `generated_by`, `sanitization`, `contains_examples`, and, for empty checkpoints, `empty_reason=checkpoint_contains_no_persisted_runtime_episode_traces`

Each example is intentionally small and sanitized. Expected example fields include `example_id`, `trace_id`, `dataset_role=adapter_distillation_example_preview`, `endpoint`, `type`, `operation`, `status`, timestamps, `state_revision`, `token_count`, `context.request`, `prediction`, `proposed_answer`, `proposed_action`, `action`, `actual_output`, `verification`, `feedback`, `feedback_summary`, `policy_decision`, `replay_plan_summary`, `replay_sample_summary`, `provenance`, `latency_ms`, `failure`, and `error`. A checkpoint with no persisted runtime traces is a valid empty dataset with `count=0` and `examples=[]`.

### Service benchmark harness

`src\hecsn\evaluation\service_benchmark.py` measures the local FastAPI service in-process with `TestClient`. It is a smoke/latency harness for the maintained service surface, not a live load generator. The harness creates or loads a checkpoint, exercises `/health`, `/feed`, `/query`, `/respond`, `/terminus/living-loop`, `GET /terminus/policy-actuator`, `GET /terminus/replay-plan`, `GET /terminus/replay-sample/history`, and `/terminus/runtime-traces/export`, and writes one JSON result file.

CLI usage with a tiny deterministic local checkpoint:

```powershell
python -m hecsn.evaluation.service_benchmark --checkpoint reports\service_benchmark\synthetic.pt --output reports\service_benchmark\result.json --create-synthetic-checkpoint
```

Useful options:
- `--checkpoint` — checkpoint to load; with `--create-synthetic-checkpoint`, a tiny deterministic checkpoint is created there if missing
- `--output` — benchmark JSON output path
- `--trace-dir`, `--web-dist-dir`, and `--env-root` — optional service construction paths
- `--feed-text`, `--query-text`, and `--export-limit` — request payload controls for the smoke run

The output JSON includes `benchmark=hecsn_service_endpoint_latency`, `schema_version`, `generated_at`, `checkpoint_path`, `success`, `total_latency_ms`, `endpoint_timings`, `endpoints_by_name`, `living_loop_benchmark_telemetry`, `feedback_telemetry`, `policy_actuator_summary`, `replay_plan_summary`, `replay_sample_summary`, `replay_executor_summary`, `trace_export_summary`, and `output_path`. Each endpoint timing records the endpoint `name`, HTTP `method`, `path`, `latency_ms`, `success`, `status_code`, optional `params`, `response_size_bytes`, and JSON response keys, including `replay_sample_history` timing for `GET /terminus/replay-sample/history`. `policy_actuator_summary` captures `schema_version`, `action`, `recommendation`, `risk`, `expected_information_gain`, `expected_goal_progress`, `expected_cost`, `uncertainty`, `advisory`, `executable`, target IDs, `suggested_endpoint`, and `reason_codes`. `replay_plan_summary` captures replay `schema_version`, `endpoint`, `count`, `priority_rules_version`, `plan_reason_codes`, `snapshot_counts`, and the top candidate summary. `replay_sample_summary` / `replay_executor_summary` capture sampler endpoints, history endpoints, count/history count, selected counts, mode/status counts, latest history item, safety flags, safety boundaries, and audit-only/executable posture.

### Benchmark posture: ARC-AGI

ARC-AGI should remain a **separate benchmark path**, not a claim attached to the current Terminus runtime. A credible ARC-AGI path would need, at minimum:
- an object parser for grids and relational scene structure
- a DSL/program synthesis layer for candidate transformations
- a verifier that executes candidates against examples
- search/refinement over candidates and partial programs
- optional Subcortex candidates as proposals, not as the scorer
- exact-match scoring against held-out task outputs

The initial ARC-specific code lives under `src\hecsn\evaluation\arc_agi.py`. It is intentionally limited to deterministic benchmark plumbing: grid validation, JSON-like task loading, exact-match scoring, an evaluation skeleton, an object parser, and a tiny deterministic DSL/search scaffold explicitly labeled as **not** an ARC solver.

The object parser plus tiny DSL/search are benchmark plumbing for ARC experiments, not a solver, not program synthesis evidence, and not core living-loop evidence. Current Terminus telemetry validates living-loop, grounding, action verification, memory, and operational self-model posture. It does **not** imply that Terminus already solves ARC-AGI, and the ARC benchmark path should not be used as evidence for core living-loop claims.

### Current validation reports

Key backend reports live in:
- `reports/runtime_correctness_validation.md`
- `reports/grounding_bridge_validation.md`
- `reports/gating_validation.md`
- `reports/ingestion_latency_validation.md`
- `reports/action_loop_validation.md`
- `reports/acceptance_harness_validation.md`

---

## 11. Extension points

### Adding a new maintained action

1. extend `src/hecsn/service/action_loop.py`
2. extend `src/hecsn/service/schemas.py`
3. keep execution/verification/provenance on the same maintained path
4. add manager/service/API tests
5. add controlled validation and update reports

### Adding a new source type

1. add the source runtime/stream logic in `src/hecsn/service/manager.py`
2. keep it on the same ingestion/warm-queue/runtime path
3. extend schemas/API only if needed
4. add tests for configuration, runtime progress, and shutdown behavior

### Adding a new sensory modality

1. add encoder/runtime support in `src/hecsn/data/` and service wiring
2. integrate with the same sensory buffering and observation path
3. add tests and validation

### Adding a new API endpoint

1. define schemas in `src/hecsn/service/schemas.py`
2. add manager method
3. wire route in `src/hecsn/service/api.py`
4. add tests in `tests/test_service_api.py`

---

## 12. Current limitations

The active docs should state current runtime limitations plainly.

### Runtime limitations
- live Hugging Face sources can still introduce real latency and transient stalls
- short smoke runs are now correctly classified, but they remain sensitive to real remote performance
- replay sampling/execution is currently an operator-gated audit path only; it records review history but does not perform autonomous training, memory promotion, feedback posting, sleep, digital action execution, or external calls

### Cognitive limitations
- replay validation remains weak in short runs
- some language/readout surfaces are still awkward or generic
- language/readout quality is not yet consistent across domains
- contradicted replay candidates are negative lessons until a separate correction/training process validates them; dreamed or synthetic candidates are never promoted as facts by the sampler/executor

### Data limitations
- current multimodal sources are materially better than old digit-only scaffolding, but they are still not broad embodied experience
- focus-aware background source allocation plus autonomy-guided real-source acquisition with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility is still only a first step, not a substitute for richer real-world grounding data

### Documentation truth
If you find any operator-facing documentation that still tells you to:
- start Ollama
- pull Gemma locally
- configure NVIDIA NIM / `NVIDIA_API_KEY`
- use retired external LLM mock backends
- call ThoughtLoop query, thought, or sleep endpoints as maintained paths
- use removed quick-start presets like `multimodal`

then that documentation is stale and should be treated as incorrect.

---

## Quick reference

```bash
# Train a checkpoint
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir checkpoints/terminus \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-field text

# Start service
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt

# Build frontend
cd HECSN_UI && npm install && npm run build

# Run tests
python -m pytest -q

# Run long-test acceptance/health path
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/wp06_validation

# Quick-start Terminus
curl -X POST "http://localhost:8000/terminus/quick-start?preset=curriculum"

# Review living-loop state
curl http://localhost:8000/terminus/living-loop
```
