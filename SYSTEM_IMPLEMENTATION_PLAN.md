# Terminus System Audit and Implementation Plan

**Date:** 2026-04-22  
**Status:** Active execution plan  
**Scope:** System-level recovery and maturation plan. The UI is treated as an operator interface, not the core of the work.

---

## 2026-04-30 Stabilization Update

The service refactor is now treated as the current implementation baseline: `HECSNServiceManager` is a composition root over focused service mixins, cortex/NIM construction remains lazy, and replay dataset bundle generation is still preview/export only.

Current validation after the stabilization pass:

- `tests/test_service_manager.py`: **123 passed**
- `tests/test_long_test_runner.py tests/test_service_benchmark.py tests/test_trace_export_runner.py`: **16 passed**
- `tests/test_service_api.py`: **44 passed**
- action, cortex, thought-loop, living-loop, query-runner, and memory-consolidation slices: **194 passed, 3 skipped**
- `HECSN_UI` production build: **passed**, with the existing large `three` chunk warning still present
- fresh in-process service benchmark: **success**, total latency **7155.884 ms**
- post-feed-sampling service benchmark: **success**, total latency **2201.261 ms**, `/feed` **1584.017 ms**
- post-feed-segmentation service benchmark: **success**, total latency **1070.362 ms**, `/feed` **619.942 ms**
- post-feed-lexical/query-cache service benchmark: **success**, total latency **537.965 ms**, `/feed` **265.337 ms**, `/query` **84.106 ms**, `/respond` **93.741 ms**

Important current truth:

- The acceptance harness now initializes the lazy cortex before judging idle gating, so mock-cortex and live-cortex paths exercise the same initialization contract.
- Query-conditioned retrieval focus is owned by `hecsn.service.interaction_runtime`; tests should patch that module when they need to intercept `build_query_result`.
- Replay dataset preview and bundle APIs remain safety-gated: no training, no memory promotion, no action execution, no sleep, no external calls, and no state mutation beyond packaging metadata.
- `/feed` remains the dominant benchmark cost, but request-time feed now uses lexical rolling segment windows instead of character windows or learned-boundary segmentation. It still preserves phrase-level concept grounding, samples runtime concept observation every 8 feed units plus the final pending unit, and the current synthetic benchmark processed 44 feed units with 7 concept observations.
- Query/respond now cache pure grounding term expansion and semantic unit similarity, cutting repeated memory-episode matching work without changing matching semantics.
- Duplicate rolling-window pruning was tested and rejected because repeated focused evidence is still needed for concept-store growth and autonomy focus.
- Service benchmark JSON now exposes `feed_summary`, so future performance PRs can inspect endpoint latency and feed observation pressure in the same artifact.

Tests worth having toward the living-brain goal:

- **Truth/liveness tests:** acceptance harness must pass with a valid cortex and must report `partial` or `failed` when cortex initialization is unavailable; empty or dead runs must never be classified as alive.
- **Grounding tests:** query, response, and cortex thought paths must prove that source evidence influences output instead of allowing generic language drift.
- **Safety-boundary tests:** replay sample, replay execute, replay dataset preview, and replay dataset bundle must prove they do not train, mutate memory, promote facts, post feedback, execute actions, start sleep, or make external calls.
- **Autonomy tests:** policy actuator and delayed-consequence tests must verify proposal, contradiction, recovery, split/remerge lineage, and operator-gated execution without hiding side effects.
- **Performance tests:** service benchmark output must keep endpoint timings and `feed_summary` visible, with `/feed` tracked as the current first optimization target rather than hidden inside total latency.
- **Operator-surface tests:** API and UI build checks protect the dashboard as an observation/control surface, but they are secondary to runtime truth, safety, and evidence quality.

---

## 1. Purpose

This document turns the recent audit into an implementation roadmap.

The goal is to move Terminus from:

- a **well-structured research codebase with many good ideas**
- into a **reliable, testable, grounded system**
- where the **UI is only a window into the runtime**, not the runtime itself.

This plan is intentionally specific. For each major issue it states:

1. **what is wrong now**
2. **what the system should do instead**
3. **how we will implement the change**
4. **how we will test and validate it**
5. **what documentation must be updated before moving on**

---

## 2. Audit Baseline: What was verified

The following facts were established during direct inspection and execution:

### 2.1 Repository and build health
- Full Python test suite passed:
  - **923 passed**
  - **3 skipped**
  - **4 warnings**
  - **7 subtests passed**
- UI production build succeeded.
- A real NVIDIA NIM cortex call succeeded with a valid structured answer in roughly **970 ms**.

### 2.2 Live runtime findings
- The real curriculum preset can start, but real background ingestion is **slow to become useful**.
- In a real quick-start run using the live Hugging Face + NIM path, the **first meaningful background tick took ~43.7 seconds**.
- A short real long-test run produced a report with:
  - **0 thoughts**
  - **0 topics**
  - **0 memory fill**
  - **0 prediction error**
  - **0 dream verification**
- In a local-file runtime test, the system processed source text quickly, but the cortex still produced thoughts unrelated to the source content.
  - Example source content: octopus-related text
  - Example generated thoughts: aurora, glaciers, seed dispersal
- Calling manual `terminus_tick()` while the background runtime was active caused a real crash:
  - `ValueError: generator already executing`

### 2.3 Documentation and test coverage findings
- Production code is NIM-only, but stale docs still reference:
  - Ollama
  - Gemma
  - FakeCortex
- The test suite is strong at component and API level, but many runtime tests use:
  - mocked start paths
  - local file sources
  - synthetic or simplified conditions
- The current long-test runner validates report generation, but it does **not fail loudly when the runtime is effectively dead**.

---

## 3. Non-Negotiable Target Architecture

The system we are trying to build is **not**:

- a dashboard that shows metrics from an LLM
- a passive corpus stream with cognitive-style labels
- a free-running thought loop decorated with neuroscience terms

The system we are trying to build **is**:

```text
Observation / Sensory Input / External Query
        ↓
State Update in SNN / Predictive Substrate
        ↓
Prediction Error / Salience / Active Tensions
        ↓
Gate: decide whether cortex should think or act
        ↓
Cortex reasoning over real grounded evidence
        ↓
Action selection (digital now, physical later)
        ↓
Action execution
        ↓
Verification / success detection / contradiction
        ↓
Memory update + replay + planning for next step
        ↓
UI only presents the above state to the operator
```

### Required properties of the target system
1. The cortex must not self-start without real pressure or explicit query.
2. The cortex must receive **actual evidence**, not just metadata about evidence.
3. The live runtime must not block cognition on cold remote dataset startup.
4. Manual and background execution paths must not compete for the same stream state.
5. Reports must distinguish **healthy runs** from **dead runs**.
6. The architecture must include **action and verification**, not only passive thought.

---

## 4. What Is Wrong Now / What It Should Be / How We Fix It

## Issue A — The cortex can think before the SNN has earned the right to trigger it

### What is wrong now
In `src/hecsn/cortex/drives.py`:
- `DriveState.curiosity` starts at `0.5`
- `DriveState.uncertainty` starts at `0.5`
- `DriveSystem.should_think()` allows thought when curiosity is `> 0.4`

That means the thought loop can start generating thoughts **before** meaningful grounded evidence arrives.

### What it should be
The cortex should only fire when at least one of these is true:
- real external query exists
- real observation was injected
- prediction error crossed threshold
- verified unresolved tension exists
- action result requires interpretation

### How we fix it
- Lower default drive values for startup.
- Introduce explicit startup quiet state.
- Separate **passive idle state** from **deliberation-ready state**.
- Make thought generation require one of:
  - query pending
  - grounded observation pending
  - predictive signal above threshold
  - unresolved working-memory tension above threshold

### Primary code targets
- `src/hecsn/cortex/drives.py`
- `src/hecsn/cortex/thought_loop.py`

---

## Issue B — External queries are not strongly guaranteed to trigger an answer

### What is wrong now
`ThalamicGate.submit_query()` calls `DriveSystem.update_from_external_input()`, which raises social drive by `0.3`, while `DriveSystem.should_think()` checks `social > 0.3`.

That means a query can land exactly on the boundary and still fail to trigger thought unless some other drive also pushes the system.

### What it should be
A valid external query must reliably trigger an answer cycle within a bounded time window.

### How we fix it
- Make query submission an explicit high-priority wake event.
- Do not rely on a fragile threshold coincidence.
- Add direct query-pending logic in the thought loop.
- Add test coverage that proves a submitted query causes a new thought/answer.

### Primary code targets
- `src/hecsn/cortex/drives.py`
- `src/hecsn/cortex/thought_loop.py`

---

## Issue C — The cortex is being fed metadata instead of grounded evidence

### What is wrong now
In `src/hecsn/service/manager.py`, the observation injected into the cortex is effectively:
- how many tokens were processed
- which source name was used
- a few concept labels

This is not the same as injecting the **actual semantic content** that was just processed.

### What it should be
The cortex should receive structured grounded observations such as:
- source excerpt or compressed proposition list
- sensory observation summary
- current prediction mismatch
- verified recent evidence
- provenance and confidence

### How we fix it
- Introduce a structured observation payload.
- Carry forward the actual recent raw window or a controlled semantic summary derived from it.
- Add provenance tags to what enters working memory and episodic memory.
- Ensure the thalamic gate prefers recent verified observations over generic source metadata.

### Primary code targets
- `src/hecsn/service/manager.py`
- `src/hecsn/cortex/thought_loop.py`
- `src/hecsn/cortex/drives.py`
- `src/hecsn/cortex/core.py`
- `src/hecsn/cortex/episodic_memory.py`

---

## Issue D — Real-source startup latency is too high

### What is wrong now
The live system depends on remote Hugging Face streaming/parquet startup during cognition time. In practice this caused very slow first progress.

### What it should be
The runtime should support:
- **cold start** that becomes useful in a reasonable time
- **warm start** that becomes useful quickly
- decoupled ingestion and cognition

### How we fix it
- Introduce an ingestion/prewarm layer.
- Pre-fetch and normalize source material before the main cognitive loop depends on it.
- Cache or locally spool normalized episodes/chunks.
- Keep the live loop consuming from a local queue, not directly from remote startup paths.
- Where supported, restrict remote reads to only required columns/fields.

### Primary code targets
- `src/hecsn/service/manager.py`
- `src/hecsn/data/corpus_loader.py`
- `src/hecsn/service/terminus_hf_sources.py`
- `src/hecsn/service/terminus_sensory.py`

---

## Issue E — There are two competing runtime execution models

### What is wrong now
The background runtime uses `_brain_loop()` / `_collect_chunk_unlocked()`, but manual `terminus_tick()` still uses `_brain_tick_locked()` and can touch the same generator-backed stream state.

This produced a real crash:
- `ValueError: generator already executing`

### What it should be
There must be a single owner of stream consumption.

### How we fix it
Pick one of the following and enforce it consistently:
1. **Reject manual tick while background runtime is active** with a clear error/HTTP response; or
2. Route manual tick through the same scheduler/queue as the background loop.

The simplest safe first step is option 1.

### Primary code targets
- `src/hecsn/service/manager.py`
- `src/hecsn/service/api.py`
- `src/hecsn/service/schemas.py`

---

## Issue F — Shutdown and runtime control are not yet fully robust

### What is wrong now
Shutdown can be delayed or fail cleanly when background work is blocked in remote operations.

### What it should be
Stop/start behavior must be reliable and bounded.

### How we fix it
- Audit all background blocking points.
- Ensure network operations in the runtime are bounded by timeout and respond to stop requests cleanly.
- Improve stop semantics and logging around stop failure.
- Add tests that assert graceful stop under live-ish conditions.

### Primary code targets
- `src/hecsn/service/manager.py`
- possibly `src/hecsn/data/corpus_loader.py`
- possibly `src/hecsn/service/terminus_sensory.py`

---

## Issue G — The long-test runner can produce a “clean” report for a dead run

### What is wrong now
`src/hecsn/training/long_test_runner.py` can complete and generate a report even if the system performed no useful work.

### What it should be
A long test should be considered failed or degraded when:
- no thoughts were generated
- no meaningful ticks completed
- no memory changed
- no sensory episodes appeared when expected
- no grounded evidence entered cortex context

### How we fix it
- Add health assertions and run classification:
  - `healthy`
  - `degraded`
  - `failed`
- Include explicit dead-run reasons in JSON and markdown.
- Add minimum-activity thresholds.
- Fail CI or validation steps for dead runs when appropriate.

### Primary code targets
- `src/hecsn/training/long_test_runner.py`
- new acceptance helpers/tests under `tests/`

---

## Issue H — The tests are strong, but they do not yet prove the live system

### What is wrong now
The suite heavily validates components, mocks, and local-file paths, but the most important live path is under-tested:
- real NIM cortex
- real source ingestion behavior
- live grounding behavior
- query-to-answer liveness
- dead-run detection

### What it should be
We need a layered validation strategy:
- fast unit tests
- deterministic integration tests
- optional live smoke tests
- periodic acceptance runs against the real stack

### How we fix it
Add explicit system-level tests for:
- query triggers answer
- no ungated thought at idle startup
- local source affects thought topic distribution
- background + manual tick cannot conflict
- long-test dead runs are flagged

### Primary code targets
- `tests/test_service_manager.py`
- `tests/test_service_api.py`
- new targeted integration tests under `tests/`

---

## Issue I — Documentation is stale and misdescribes the system

### What is wrong now
`TERMINUS_Tutorial.md` still describes old Ollama/Gemma/FakeCortex flows that do not match the current production path.

### What it should be
Docs must describe the actual runtime:
- NIM-only production cortex
- current dataset strategy
- current limitations
- true operator workflow
- what is experimental vs production

### How we fix it
- Update stale docs after each completed work package.
- Keep one canonical architecture description synced with the code.
- Remove obsolete instructions and terminology.

### Primary code targets
- `TERMINUS_Tutorial.md`
- `GPCSN.md`
- possibly `PRD.md` and report docs

---

## Issue J — Some research features are implemented as heuristic proxies, not system-defining capabilities

### What is wrong now
Examples:
- source/provider selection now includes age-sensitive consequence-state cooling/retirement, repeated-record compaction/aggregation, trajectory-sensitive family summaries, divergence-sensitive family splitting, lineage-aware family remerge, and grounded family-summary calibration, and the final WP-08 closure review did not expose another explicit maintained-path retuning slice
- outcome-linked utility is now calibrated against richer long-horizon grounded family summaries; provider/autonomy weighting remains heuristic, but no further WP-08 slice is justified without a new measured maintained-path failure

### What it should be
Research features should be tuned **after** the base system loop is correct.

### How we fix it
- Delay research tuning until runtime correctness, grounding, and action-verification are stable.
- Then revisit feature fidelity and measure impact with real benchmarks.

### Primary code targets
- `src/hecsn/core/hypercube.py`
- `src/hecsn/consolidation/memory_store.py`
- `src/hecsn/training/trainer.py`

---

## Issue K — The system is still mostly passive, not action-verified

### What is wrong now
The current architecture still leans heavily on:
- passive corpus exposure
- thought production
- internal replay

It does not yet have a strong action → outcome → verification loop.

### What it should be
A grounded system must learn from consequence.

### How we fix it
Introduce a digital embodiment/action layer before chasing physical embodiment:
- browser actions
- API actions
- file/system actions
- retrieval/search actions with verification

Memory must then include:
- observation
- intended action
- predicted result
- actual result
- verification status
- downstream utility

### Primary code targets
- new action/execution modules
- `src/hecsn/service/manager.py`
- `src/hecsn/cortex/thought_loop.py`
- `src/hecsn/cortex/episodic_memory.py`
- `src/hecsn/interaction/`
- `src/hecsn/reporting/`

---

## 5. What Is Palliative vs What Is Structural

## Palliative / temporary scaffolding
These are useful, but they do **not** solve the core system problem by themselves:
- passive corpus streaming used as “grounding”
- dream/replay/narrative layers without action verification
- topology tweaks before system correctness
- UI polish before runtime correctness

## Structural / core work
These are the changes that actually move the system forward:
- true SNN gating of cortex activity
- evidence-rich observation injection
- safe and unified runtime scheduling
- decoupled ingestion and cognition
- action + verification + memory update loop
- dead-run detection and acceptance testing
- documentation aligned with the real system

---

## 6. Ordered Work Packages

The work packages below are the implementation order we should follow.
We should **not** jump ahead unless an earlier package is blocked for a concrete reason.

---

# WP-01 — Runtime Correctness and Operator Safety

**Priority:** P0  
**Why first:** Without runtime correctness, every later experiment is unreliable.

## Problems addressed
- dual scheduler conflict
- unsafe manual tick during background runtime
- weak query triggering
- inconsistent startup/shutdown behavior
- inconsistent env/bootstrap behavior

## Desired end state
- one safe runtime control model
- manual tick is either serialized or explicitly rejected while running
- submitted query reliably produces an answer cycle
- start/stop behavior is deterministic and bounded
- manager initialization behavior is consistent across entrypoints

## Implementation tasks
1. In `src/hecsn/service/manager.py`:
   - detect background-running state inside `terminus_tick()`
   - reject with a clear structured error or reroute through one scheduler
2. In `src/hecsn/cortex/drives.py` and `src/hecsn/cortex/thought_loop.py`:
   - make query pending an explicit wake condition
3. Review stop/join behavior in `src/hecsn/service/manager.py`:
   - improve logging
   - improve timeout behavior
   - ensure remote operations do not keep shutdown hostage longer than expected
4. Standardize environment loading:
   - create a single helper or explicit contract for `.env` loading
   - avoid entrypoint-specific surprises

## Tests to add/update
- `tests/test_service_manager.py`
  - manual tick while background runtime active returns safe failure, not crash
  - submitted query triggers new thought
  - runtime start/stop remains clean
- `tests/test_service_api.py`
  - API behavior for unsafe tick while running
- optional small integration test for consistent env/bootstrap

## Validation
- targeted pytest for affected files
- local-file runtime smoke test
- real query-to-answer smoke test using NIM
- verify no `generator already executing` path remains

## Documentation deliverables
- add `reports/runtime_correctness_validation.md`
- update this plan status after completion

## Exit criteria
- no concurrent stream-consumer crash path exists
- query reliably yields an answer cycle
- safe start/stop works repeatedly
- bootstrap behavior is documented and consistent

---

# WP-02 — Evidence-Rich Grounding Bridge

**Priority:** P0  
**Why second:** The system cannot be called grounded while the cortex mainly sees metadata.

## Problems addressed
- cortex seeing token counts/source labels instead of actual evidence
- thought content drifting away from source content

## Desired end state
When the system processes a source window or sensory episode, the cortex receives a compact but real grounded representation of that evidence.

## Implementation tasks
1. Define a structured observation event shape containing at least:
   - source ID
   - provenance
   - raw excerpt or compressed proposition summary
   - sensory summary if present
   - confidence
   - salience
   - prediction mismatch if available
2. Replace generic observation injection in `src/hecsn/service/manager.py` with evidence-rich injection.
3. Ensure `ThoughtLoop` and `ThalamicGate` prioritize recent grounded observations.
4. Preserve provenance in episodic memory.

## Tests to add/update
- local-source integration test:
  - if source text is about octopuses, recent thoughts should show lexical/semantic overlap with octopus-related content
- sensory observation injection tests
- memory provenance tests

## Validation
- local-file runtime with domain-specific source text
- query the system about the active source after a few ticks
- compare thought topics to source keywords using a simple overlap metric

## Documentation deliverables
- `reports/grounding_bridge_validation.md`
- architecture note explaining observation event flow

## Exit criteria
- source-conditioned thought content is measurably better aligned
- cortex context contains real evidence payloads, not just processing metadata

---

# WP-03 — True SNN-Gated Cognition

**Priority:** P0  
**Why third:** Once grounded evidence is flowing, cortex activation policy must be corrected.

## Problems addressed
- ungated spontaneous thought at startup
- cortex thinking before real evidence or tension exists

## Desired end state
The cortex does not think “just because it is alive.” It thinks because the substrate or operator created a real need.

## Current execution status
- `WP-03.1` complete: startup quiet state and explicit grounded/substrate wake gating are now implemented.
- `WP-03.2` complete: repeated deliberation continuation now depends on renewed substrate pressure or still-active unresolved tension.
- `WP-03.3` complete: sustained-pressure hysteresis and longer-run background wake/inhibit behavior are now implemented and validated.
- `WP-03` package status: **COMPLETE**.

## Implementation tasks
1. Reduce default startup drives.
2. Add explicit idle/quiet startup mode.
3. Make `should_think()` depend on:
   - query pending
   - grounded observation pending
   - predictive error threshold
   - unresolved working-memory tension
   - action verification need
4. Consider separating:
   - `can_think()` from
   - `should_answer_now()`

## Tests to add/update
- no spontaneous thought in a no-input idle window
- query still triggers thought immediately enough
- first autonomous thought happens only after grounded evidence or real prediction pressure

## Validation
- local runtime with no source: quiet
- local runtime with source: first thought only after evidence
- real preset run: thought timing reflects actual runtime activity

## Documentation deliverables
- `reports/gating_validation.md`
- update architecture docs to reflect new gating policy

## Exit criteria
- idle startup no longer free-runs into unrelated thoughts
- cortex firing is traceable to a concrete trigger

---

# WP-04 — Ingestion Plane and Warm Queue

**Priority:** P1  
**Why fourth:** Once behavior is correct, startup latency becomes the next blocker for real use.

## Problems addressed
- cold remote source startup latency
- live loop blocked by HF dataset resolution

## Desired end state
The cognitive loop consumes from a prewarmed local queue of normalized episodes/chunks.

## Current execution status
- `WP-04.1` complete: text-source warm queue buffering and ingestion telemetry are now implemented.
- `WP-04.2` complete: background prewarm path and cold-vs-warm startup instrumentation are now implemented.
- `WP-04.3` complete: sensory warm queue parity and longer-run stall tolerance validation are now implemented.
- `WP-04.4` complete: bounded remote prewarm budgets and real cold-vs-warm remote-start validation are now implemented.
- `WP-04.5` complete: detached startup prewarm state and automated isolation checks are implemented and validated.
- `WP-04.6` complete: startup prewarm now yields to active execution during the remote cold-start window.
- `WP-04.7` complete: startup prewarm waiting uses event-driven active-idle signaling; delayed prewarm can stand down after active progress; active remote ticks/episodes are time-bounded; and remote bootstrap helpers now obey explicit budget envelopes during close/reconfigure.
- `WP-04.8` complete: remote text and sensory warm-result cache restore now makes later startup materially faster; first-ever cold text, audio, and visual remote startup now bootstrap without repeated retry pressure; loader-level column projection is in place; and the S1 visual path now uses fast fallback bootstrapping while richer recaption metadata backfills on the canonical stream path.
- `WP-04` package status: **COMPLETE**.
- `WP-05.1` partial: a deterministic `workspace_search` digital action now executes, verifies success vs contradiction, persists structured action history, replays into cortex memory on restore, and is exposed through the backend API.
- `WP-05.2` partial: `respond()` can now auto-trigger `workspace_search` for under-grounded queries, reuse recent verified action results on repeated related queries, and feed that evidence back into the same response pass.
- `WP-05.3` partial: cortex-side `search` action intent now triggers the same maintained `workspace_search` path, records trigger provenance, and avoids self-sealing on runtime-generated artifacts.
- `WP-05.4` partial: the maintained action surface now includes `workspace_read`; query-gap routing and cortex-side `search` intent can choose file-read actions when the query names a concrete workspace file.
- `WP-05.5` partial: the maintained action surface now includes `web_fetch`; query-gap routing and cortex-side `search` intent can choose explicit-URL fetch actions while reusing the same verification / persistence / replay path.
- `WP-05.6` partial: the maintained action surface now includes `api_request`; query-gap routing and cortex-side `search` intent can choose explicit JSON/API endpoints while reusing the same verification / persistence / replay path.
- `WP-05.7` partial: `api_request` now supports maintained parameterized request forms (`GET`/`POST`, `params`, `json_body`, bounded remote budgets) while preserving the same verification / persistence / replay path and preventing explicit-URL reuse from collapsing POST/body-shaped requests into simple GET-style reuse.
- `WP-05.8` partial: cortex-triggered maintained routing now accepts `search`, `ask`, `remember`, and `explore`; these intents either execute or reuse the same maintained workspace/fetch/API action path while preserving action provenance in runtime events and persisted action history.
- `WP-05.9` partial: `api_request` verification now lifts grouped object/array evidence from structured JSON payloads so multi-field facts can be verified on the same maintained execution / persistence / replay path rather than only through isolated scalar leaf matches.
- `WP-05.10` partial: `api_request` now supports explicit `expected_json_paths` and `expected_response_shape` assertions, allowing the maintained verification path to succeed or contradict on operator-visible structural expectations rather than only lexical query matching.
- `WP-05.11` partial: `api_request` now supports explicit `expected_json_values` assertions, allowing the maintained verification path to succeed or contradict on exact expected values at JSON paths rather than only structural presence or lexical relevance.
- `WP-05.12` partial: `api_request` now supports explicit `expected_json_predicates` assertions, allowing the maintained verification path to succeed or contradict on predicate-style conditions such as contains, regex, and numeric thresholds at JSON paths.
- `WP-05.13` partial: `api_request` now supports richer composite predicates such as numeric ranges, startswith/endswith checks, and quantifier-style array predicates while preserving the same maintained verification / persistence / replay path.
- `WP-05.14` partial: `api_request` now supports logical predicate groups (`all` / `any` / `none`) plus stronger array/object quantifier semantics such as `all_contains`, `all_regex`, `none_contains`, and `none_regex`, while preserving the same maintained verification / persistence / replay path.
- `WP-05.15` partial: `api_request` now supports wildcard-style nested path expansion across path/value/predicate assertions plus nested predicate groups, allowing repeated child structures to verify on the same maintained verification / persistence / replay path without hard-coded indices.
- `WP-05.16` partial: cortex-side `sleep` now routes through an explicit maintained manager-side control path, with operator/API-triggered and cortex-intent-triggered requests using the same queued sleep-cycle control surface and exposing structured request/completion provenance in runtime state.
- `WP-05` package status: **COMPLETE**.
- `WP-06.1` partial: `src/hecsn/training/long_test_runner.py` now emits explicit `alive` / `degraded` / `dead` run classification, includes a deterministic maintained acceptance harness (idle gating, query answer, grounded source influence, runtime progress), and returns non-zero CLI exit codes for degraded/dead runs.
- `WP-06.2` partial: `src/hecsn/service/manager.py` now supports fresh diagnostic snapshots through `status(fresh_wait_seconds=...)` and `terminus_status(fresh_wait_seconds=...)`, and the long-test runner uses that maintained path to avoid dead-by-stale-status smoke classification.
- `WP-06.3` partial: the canonical `curriculum` quick-start preset now uses a bounded `tick_tokens` budget so the maintained live smoke path reaches observable thought activity sooner, and the real short smoke run now classifies as `alive` instead of dead/degraded-by-latency.
- `WP-06` package status: **COMPLETE**.
- `WP-07` partial: active docs were synced to the maintained runtime by rewriting `TERMINUS_Tutorial.md`, updating `GPCSN.md`, and recording the cleanup/validation in `reports/docs_sync_validation.md`.
- `WP-07` package status: **COMPLETE**.
- `WP-08.1` partial: `DualMemoryStore` awake-ripple replay priority was retuned from a flat boolean/3x boost to graded DA- and recency-sensitive ripple strength with measured 3–5x replay multipliers, documented in `reports/mechanism_retuning_validation.md`.
- `WP-08.2` partial: `HypercubeTopology` shortcut selection was retuned from random long-range picks to deterministic maximal-span shortcut masks, improving reproducibility and reducing average shortest-path length without leaving the old random policy in place.
- `WP-08.3` partial: `HypercubeBindingLayer` hub boosting was retuned from a floor-rounded binary hub mask plus flat 1.5x factor and activity-sum refresh heuristic to a per-bind, ceil-based graded 1.5x–2.0x hub profile with persisted hub activation state, without keeping the old flat hub-boost path.
- `WP-08.4` partial: `HypercubeTopology` shortcut budgeting was retuned from a fixed per-node shortcut count to a deterministic target-degree compensation policy, so masked boundary nodes recover lost bit-flip degree on the same maintained topology path instead of keeping a lower-degree legacy shape.
- `WP-08.5` partial: `HypercubeBindingLayer` hub influence is now expressed through deterministic structural hub outreach instead of a direct source-multiplier path, so selected hubs can literally expand outgoing connectivity on the maintained graph without leaving the old weighted hub path behind.
- `WP-08.6` partial: long-range shortcut and hub-edge weighting was retuned from a fixed raw 0.5 rule to a bounded relative-mass calibration policy, so over-augmented rows stop letting long-range links dominate normalized row mass without keeping the old fixed-weight path.
- `WP-08.7` partial: curriculum injection was retuned from text plus synthetic visual/audio hint channels to text-only curriculum episodes on the maintained path, so multimodal grounding now lives only on the real Hugging Face sensory runtime instead of leaving the old hint path behind.
- `WP-08.8` partial: the maintained runtime now removes NIM-generated curriculum text episodes entirely in favor of autonomy-guided real-source acquisition over maintained catalogs, without leaving the old curriculum-text path behind.
- `WP-08.9` partial: the maintained runtime now adaptively retunes autonomy trigger cadence, acquisition budget, and provider priority weighting under strong focus pressure, so targeted acquisition can displace passive background exposure more aggressively on the same maintained path without leaving a second scheduling mode behind.
- `WP-08.10` partial: the maintained runtime now replaces passive round-robin background source scheduling with focus-aware source allocation on the same text-ingestion path, so aligned maintained sources can win passive exposure without spawning a second background-routing implementation.
- `WP-08.11` partial: the maintained runtime now calibrates persistent background-source and provider utility on the maintained path itself, so repeated useful sources/providers can influence future selection without spawning a second curriculum or routing implementation.
- `WP-08.12` partial: the maintained runtime now calibrates source/provider utility against grounded answer and verified action outcomes on the same maintained path, so persistent utility is no longer driven only by local alignment/gain heuristics.
- `WP-08.13` partial: query/response evidence now preserves source/provider provenance and credits grounded response outcomes back to the exact maintained evidence provenance when available, so utility attribution no longer depends only on topic-level heuristics.
- `WP-08.14` partial: delayed multi-turn query improvement now credits grounded response evidence provenance back to the same maintained source/provider path, so utility no longer depends only on immediate-turn outcomes.
- `WP-08.15` partial: later unsupported/regressed queries and relevant contradicted maintained actions now impose contradiction/decay-aware long-horizon utility penalties on that same maintained source/provider path, so long-run utility is no longer positive-credit-only.
- `WP-08.16` partial: later grounded improvement can now explicitly forgive earlier long-horizon penalties on that same maintained source/provider path via unresolved-penalty recovery scheduling, so mixed evidence is no longer handled only by raw EMA blending.
- `WP-08.17` partial: stale long-horizon consequence records now cool and retire through explicit token-age-sensitive maintenance on that same maintained source/provider path, so old consequence state is no longer governed only by bounded deque limits and generic EMAs.
- `WP-08.18` partial: repeated matched long-horizon consequence records now compact/aggregate by query family and provenance on that same maintained source/provider path, so near-duplicate consequence evidence no longer survives only as separate records.
- `WP-08.19` partial: aggregated long-horizon consequence families now retain trajectory-sensitive summaries on that same maintained source/provider path, so mixed family behavior is no longer summarized only through bounded maxima and counts.
- `WP-08.20` partial: mixed long-horizon consequence families can now split by divergent query branches on that same maintained source/provider path, so separated family behavior is no longer forced to remain inside one stable merged cluster.
- `WP-08.21` partial: split long-horizon consequence families can now remerge after aligned recovery on that same maintained source/provider path, so split lineage behavior is no longer treated as mostly irreversible.
- `WP-08.22` partial: long-horizon source/provider utility now calibrates against grounded family-summary signals on that same maintained source/provider path, so family utility no longer depends only on bounded family-state scalars plus event EMAs.
- `WP-08` closure review: targeted and broader maintained suites passed, real NIM integration passed, and a one-minute live `curriculum` smoke run reached `alive` classification in `reports/wp08_closure_validation/long_test_20260425_021934.md` without exposing another explicit maintained-path retuning slice.
- `WP-08` package status: **COMPLETE**.
- Next step: no further WP-08 slice is planned; only reopen the package if a new measurement exposes a concrete maintained-path bottleneck.

## Implementation tasks
1. Create a background ingestion/prewarm path.
2. Normalize remote text into cached chunks.
3. Normalize sensory streams into cached preview/episode units.
4. Add warm/cold start instrumentation.
5. Keep the main runtime loop reading from queue/cached stream, not directly from remote startup paths.

## Tests to add/update
- queue fill tests
- cold/warm startup smoke tests
- timeout handling tests

## Validation
- measure cold start to first useful tick
- measure warm start to first useful tick
- verify cognition continues even if remote source momentarily stalls after queue is filled

## Documentation deliverables
- `reports/ingestion_latency_validation.md`
- operator note on warm vs cold runtime startup

## Exit criteria
- warm runtime becomes useful quickly
- cold runtime is bounded and observable
- cognition is no longer directly hostage to first remote read

---

# WP-05 — Action and Verification Loop (Digital Embodiment First)

**Priority:** P1  
**Why fifth:** This is the transition from passive “thinking about things” to grounded learning through consequence.

## Problems addressed
- no true action-outcome loop
- weak grounding beyond passive corpora

## Desired end state
The system can:
- choose a digital action
- execute it
- verify the result
- remember the result with provenance
- use replay and curiosity over action outcomes, not only text streams

## Initial digital action set
Start small and verifiable:
- search/retrieve
- fetch/inspect content
- API request
- file read/query over known workspace data

## Implementation tasks
1. Define an action schema:
   - action type
   - inputs
   - predicted outcome
   - actual outcome
   - verification result
2. Add a verification schema:
   - success/failure
   - confidence
   - contradiction
   - evidence links
3. Extend episodic memory to store action episodes.
4. Feed verified outcomes back into working memory and planning.

## Tests to add/update
- deterministic action simulation tests
- verification storage tests
- action-memory-recall tests

## Validation
- run fixed digital tasks
- measure success rate, correction rate, and memory use across repeated attempts

## Documentation deliverables
- `reports/action_loop_validation.md`
- design note for action episode schema

## Exit criteria
- at least one end-to-end action-verification loop exists and is measurable

---

# WP-06 — Real Acceptance Testing and Long-Test Reform

**Priority:** P1  
**Why sixth:** We need a way to tell whether the system is alive, degraded, or dead.

## Problems addressed
- dead runs passing as valid reports
- insufficient live acceptance criteria

## Desired end state
Every long run and smoke test is explicitly classified.

## Implementation tasks
1. Update `src/hecsn/training/long_test_runner.py` to emit run health classification.
2. Define minimum live activity thresholds.
3. Fail or flag runs when core conditions are not met.
4. Add a small acceptance harness for:
   - idle gating
   - query answer
   - grounded source influence
   - runtime progress

## Tests to add/update
- runner classifies dead runs as failed/degraded
- report includes health verdict and reasons

## Validation
- run short local acceptance tests
- run one real NIM/HF smoke acceptance test

## Documentation deliverables
- `reports/acceptance_harness_validation.md`
- update operator instructions for interpreting reports

## Exit criteria
- no more “empty success” long-test reports

---

# WP-07 — Documentation, Terminology, and Operator Truthfulness

**Priority:** P2  
**Why seventh:** Docs should lag implementation slightly, but must eventually match reality.

## Problems addressed
- stale Ollama/Gemma/FakeCortex docs
- mismatch between research story and runtime truth

## Desired end state
All key docs accurately describe:
- current runtime stack
- current limitations
- current action loop
- what is experimental vs established

## Implementation tasks
1. Rewrite stale sections in `TERMINUS_Tutorial.md`.
2. Sync `GPCSN.md` with the actually implemented system.
3. Update or retire outdated notes.
4. Add a clear “current system limitations” section.

## Tests to add/update
- none automated, but docs review required before closing package

## Validation
- manual consistency pass against code
- ensure operator instructions match actual commands/paths

## Documentation deliverables
- updated canonical docs
- `reports/docs_sync_validation.md`

## Exit criteria
- no stale Ollama/Gemma/FakeCortex operational guidance remains in active docs

---

# WP-08 — Research Fidelity and Mechanism Retuning

**Priority:** P3  
**Why last:** No point tuning research details before the base system works correctly.

## Problems addressed
- spec drift in topology/mechanism implementations
- unclear impact of heuristics on real behavior

## Desired end state
Research features are revisited after the core loop is stable and measurable.

## Initial targets
- hypercube shortcut policy
- hypercube hub structure and long-range edge weighting
- awake ripple priority weighting
- acquisition-balance tuning after the action loop exists

## Implementation tasks
1. Reconcile code with intended mechanism descriptions.
2. Decide whether to keep heuristic approximations or align more literally.
3. Measure impact on throughput and grounded task performance.

## Tests to add/update
- targeted unit tests per mechanism
- targeted throughput benchmarks
- effect-on-behavior tests once action loop exists

## Validation
- benchmark before/after each mechanism change

## Documentation deliverables
- mechanism-specific validation reports

## Exit criteria
- research-tuning changes are justified by measured behavior, not only elegance

---

## 7. Execution Protocol for Every Work Package

For **every** package above, we will use the same process:

1. **Implement the smallest coherent slice**.
2. **Add or update automated tests**.
3. **Run targeted tests first**.
4. **Run at least one live validation check** if the package affects runtime behavior.
5. **Write a validation report** in `reports/`.
6. **Update this plan** with status and any scope changes.
7. **Only then move to the next package**.

### Minimum proof required before advancing
- code change merged locally
- targeted tests pass
- live validation completed
- documentation/report written
- open risks explicitly recorded

---

## 8. Status Tracker

| ID | Work Package | Priority | Status | Depends On | Deliverable |
|----|--------------|----------|--------|------------|-------------|
| WP-01 | Runtime correctness and operator safety | P0 | COMPLETE | none | `reports/runtime_correctness_validation.md` |
| WP-02 | Evidence-rich grounding bridge | P0 | COMPLETE | WP-01 | `reports/grounding_bridge_validation.md` |
| WP-03 | True SNN-gated cognition | P0 | COMPLETE | WP-02 | `reports/gating_validation.md` |
| WP-04 | Ingestion plane and warm queue | P1 | COMPLETE | WP-01 | `reports/ingestion_latency_validation.md` |
| WP-05 | Action and verification loop | P1 | COMPLETE (WP-05.1 partial: deterministic workspace search action + verification memory loop; WP-05.2 partial: query-gap-triggered action selection + reuse; WP-05.3 partial: cortex search-intent bridge; WP-05.4 partial: workspace-read action widening; WP-05.5 partial: explicit-URL web-fetch action widening; WP-05.6 partial: structured API-request action widening; WP-05.7 partial: parameterized API-request path widening; WP-05.8 partial: broadened cortex intent routing beyond search; WP-05.9 partial: structure-aware API verification widening; WP-05.10 partial: explicit API structural assertions; WP-05.11 partial: explicit API value assertions; WP-05.12 partial: explicit API predicate assertions; WP-05.13 partial: richer composite API predicates; WP-05.14 partial: logical predicate groups and stronger quantifiers; WP-05.15 partial: wildcard nested-path expansion and nested groups; WP-05.16 partial: maintained sleep-control routing) | WP-02, WP-03 | `reports/action_loop_validation.md` |
| WP-06 | Real acceptance testing and long-test reform | P1 | COMPLETE (WP-06.1 partial: long-test health classification, thresholds, maintained acceptance harness, and CLI failure signaling; WP-06.2 partial: fresh diagnostic snapshot sampling removed dead-by-stale-status smoke failures; WP-06.3 partial: bounded curriculum quick-start tick budget made the short live smoke run classify as alive) | WP-01, WP-02, WP-03 | `reports/acceptance_harness_validation.md` |
| WP-07 | Documentation and terminology sync | P2 | COMPLETE (active docs synced to the maintained runtime; stale Ollama/Gemma/FakeCortex operational guidance removed from active operator docs) | WP-01 through WP-06 as relevant | `reports/docs_sync_validation.md` |
| WP-08 | Research fidelity and mechanism retuning | P3 | COMPLETE (WP-08.1 partial: awake ripple priority weighting retuned from flat boolean/3x replay boost to graded DA/recency-sensitive 3–5x weighting; WP-08.2 partial: hypercube shortcuts retuned from random selection to deterministic long-range policy; WP-08.3 partial: hub boosting retuned from a flat 1.5x binary mask heuristic to a persisted graded 1.5x–2.0x hub profile; WP-08.4 partial: shortcut budgeting retuned from a fixed per-node count to a target-degree compensation policy; WP-08.5 partial: hub influence retuned from a direct source multiplier to deterministic structural hub outreach; WP-08.6 partial: long-range edge weighting retuned from a fixed raw 0.5 rule to bounded relative-mass calibration; WP-08.7 partial: curriculum injection retuned from text+synthetic hints to text-only episodes; WP-08.8 partial: NIM curriculum text removed in favor of autonomy-guided real-source acquisition; WP-08.9 partial: autonomy cadence/budget and provider weighting retuned under strong focus pressure; WP-08.10 partial: passive background routing replaced with focus-aware source allocation; WP-08.11 partial: persistent source/provider utility calibration added on the maintained path; WP-08.12 partial: grounded answer/action outcomes now calibrate source/provider utility; WP-08.13 partial: response-evidence provenance now calibrates source/provider utility attribution; WP-08.14 partial: delayed multi-turn query improvement now calibrates source/provider utility attribution; WP-08.15 partial: contradiction/decay-aware long-horizon penalties now calibrate source/provider utility attribution; WP-08.16 partial: explicit recovery/forgiveness scheduling now calibrates mixed long-horizon source/provider utility attribution; WP-08.17 partial: age-sensitive cooling/retirement now calibrates long-horizon consequence state; WP-08.18 partial: repeated long-horizon consequence records now compact/aggregate by query family and provenance; WP-08.19 partial: aggregated long-horizon consequence families now retain trajectory-sensitive summaries; WP-08.20 partial: mixed long-horizon consequence families now split by divergent query branches; WP-08.21 partial: split long-horizon consequence families now remerge after aligned recovery; WP-08.22 partial: long-horizon utility now calibrates against grounded family-summary signals; final closure review passed targeted and broader suites, real NIM integration, and a live one-minute `curriculum` smoke run without exposing another explicit slice) | WP-01 through WP-06 | mechanism-specific reports |

---

## 9. Recommended Next Step

No new work package should be opened speculatively.

### Immediate objective
Keep WP-08 closed unless a new measured maintained-path bottleneck appears.

### Why this is next
The closure review already passed:
- targeted WP-08 validation on the maintained path
- broader maintained regression suites
- real NVIDIA NIM integration tests
- a live one-minute `curriculum` smoke run that reached `alive` classification on the maintained runtime

### First code/document changes
- none are required for WP-08 beyond preserving the closure evidence in `reports/mechanism_retuning_validation.md` and `reports/wp08_closure_validation/`
- only open another package when measurement identifies a concrete maintained-path approximation or regression

### First success condition
- future work is driven by measured bottlenecks rather than speculative retuning
- the single maintained runtime path remains canonical
- WP-08 stays closed unless new evidence justifies reopening it

---

## 10. Final Principle

We should not optimize the UI first.
We should not tune research mechanisms first.
We should not celebrate empty long-test reports.

The order is:

1. make the runtime **correct**
2. make it **grounded**
3. make it **gated by real evidence**
4. make it **act and verify**
5. then make it **faster, richer, and cleaner**

That is how Terminus becomes a system instead of only an interface around interesting components.
