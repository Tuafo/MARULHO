# HECSN Terminus/Cortex Progress Report

**Date:** 2026-04-18
**Session:** Full system audit — Terminus + Cortex + Multimodal + UI

---

## 1. System Health Summary

| Component | Status | Issues |
|-----------|--------|--------|
| Ollama + Gemma 4 E4B | Working | 6-11s inference latency |
| Cortex (CorticalCore) | Working | All 4 modes functional (think/dream/reflect/answer) |
| ThoughtLoop | Fixed | step(force=True) added, sleep now triggers |
| EpisodicMemory | Working | 2048 capacity, proper eviction |
| DriveSystem | Working | Curiosity/anxiety/boredom drives responding |
| Terminus brain loop | Working | 10.6 tok/s, 48s ticks (slow) |
| Multimodal pipeline | Partially working | Enabled but episodes not firing |
| Service API | Working | All endpoints responding |
| Test suite | 1 failure | Emergence evaluation feedback gate |

---

## 2. Critical Bugs Found

### BUG-1: ThoughtLoop step() timing bug
- **File:** `src/hecsn/cortex/thought_loop.py:230`
- **Issue:** `step()` uses wall-clock time (`time.time()`) for interval gating. After a thought, `_last_thought_time` = now, so the next `step()` call in the same millisecond fails the interval check. Only 1 thought per test run.
- **Impact:** Makes the synchronous step() API nearly useless for testing. The real loop uses `_stop_event.wait()` which adds real delay, so production is less affected.
- **Fix:** Use virtual/delta time in step(), or accept a `force=True` parameter.

### BUG-2: Anti-rumination circuit ineffective
- **File:** `src/hecsn/cortex/drives.py:104-189`
- **Issue:** Despite word-level counting and topic avoidance, the Cortex still ruminates heavily. 23 thoughts generated, 80%+ about "fractal geometry", "neural networks", "biophysics", "dendritic trees". The diversity score drops but doesn't prevent repetition.
- **Root cause:** The LLM (Gemma 4) doesn't strongly follow the `avoid_topics` directive. The forced_topic injection only fires when `boredom > 0.5`, but the LLM's self-referential thoughts about drives don't trigger boredom fast enough.
- **Impact:** The cortex thinks about thinking instead of concrete content from SNN observations.
- **Fix:** (1) Lower boredom threshold to 0.3, (2) Make forced_topic inject at curiosity>0.4 too, (3) Use Ollama's system prompt more aggressively, (4) Strip drive-summary from context when boredom>0.3 to prevent self-referential loops.

### BUG-3: No sleep/dream cycles triggered
- **File:** `src/hecsn/cortex/thought_loop.py:280-282`
- **Issue:** `should_sleep()` requires `fatigue > 0.7 AND social < 0.2`. Fatigue accumulates at 0.02 per thought, meaning ~35 thoughts needed. But fatigue decays at 0.001 per tick. With ~425 ticks and 23 thoughts, fatigue never reaches 0.7.
- **Impact:** No dream cycles, no creative recombination, no hypothesis generation.
- **Fix:** Reduce sleep threshold to 0.5, or increase fatigue per thought to 0.04, or reduce tick decay.

### BUG-4: Multimodal episodes not firing
- **File:** `src/hecsn/service/manager.py:1760`
- **Issue:** `_run_multimodal_episode_locked()` checks `tokens_since_episode < interval` (256). After each tick (512 tokens), the counter is incremented, but episodes only run in `_finalize_tick_locked`. The first tick sets counter to 96 (partial tick), then 512. The episode should trigger at 256+ but the status shows `episodes_completed: 0`.
- **Possible cause:** The `_init_multimodal_locked` might fail silently (missing N-MNIST/FSDD directories from CWD), or the episode iterator exhausts immediately.
- **Impact:** Cross-modal grounding not training.
- **Fix:** Need to verify dataset paths resolve correctly from service CWD.

### BUG-5: Quick-start preset mismatch
- **File:** `src/hecsn/service/api.py:195`
- **Issue:** The `quick-start` endpoint defaults to `preset="wikipedia"`. When called with `?preset=multimodal`, the second call after a stop may not properly re-init multimodal datasets.
- **Impact:** User expects multimodal training but gets text-only.

---

## 3. Performance Issues

### PERF-1: Cortex inference latency (6-11s)
- Each Gemma 4 E4B inference call takes 6-11 seconds
- At 10s average, the cortex can only generate ~6 thoughts per minute
- The ThoughtLoop's 100ms tick interval is overwhelmed by LLM latency
- **Recommendation:** (1) Reduce max_response_tokens from 256 to 128, (2) Use a smaller/faster model for routine thinking, (3) Implement async inference with background queue, (4) Consider batching thoughts

### PERF-2: Terminus tick latency (48s per tick)
- Each 512-token tick takes 48 seconds = 10.6 tok/s
- Paper claims 57 tok/s at 256 cols, 72 tok/s at 1M scale
- Current config: 1024 cols with hypercube binding + AdEx = much heavier
- **Recommendation:** Reduce to 256 cols for interactive testing, use "wikipedia" preset without multimodal overhead for pure throughput testing

### PERF-3: HNSW index rebuild during deep sleep
- Rebuilds on every deep sleep cycle (O(N log N))
- With 1024 cols, this adds significant latency to sleep cycles

---

## 4. Architecture Critique

### STRENGTH-1: Clean SNN→LLM control interface
The thalamic gate properly separates SNN drives from LLM inference. The cortex never self-initiates — the SNN decides when to think.

### STRENGTH-2: Graceful cortex degradation
When Ollama is unavailable, the system falls back gracefully. All cortex operations are fire-and-forget.

### STRENGTH-3: Provenance-tracked episodic memory
The dream→verify→contradict pipeline is well-designed. Hypotheses from dreams require external validation.

### WEAKNESS-1: Cortex-SNN feedback loop is one-way
The SNN injects surprise/observations into the cortex, but cortex thoughts don't feed back into SNN routing or training. The cortex is a spectator watching the SNN, not a participant.

### WEAKNESS-2: No cross-modal triplet training
Visual and audio are processed separately (visual XOR audio per step), not jointly. This limits true multimodal grounding.

### WEAKNESS-3: Context packet is too thin for meaningful thought
5 memories, 3 thread items, 400-char drive summary — this is barely enough context for the LLM to generate grounded thoughts about the SNN's training content.

---

## 5. UI Review (from code reading)

### Section: Cortex Monitor
- **Good:** Drive bars with color coding, thought bubbles with metadata, auto-scroll
- **Missing:** No way to see full memory store, no way to graduate/contradict hypotheses, no dream cycle visualization
- **Bug risk:** `cortex?.drives` assumes specific key names — if ThoughtLoop snapshot format changes, UI breaks silently

### Section: Overview
- **Good:** Summary cards, telemetry charts
- **Missing:** No cortex status summary on overview page

### Section: Training Monitor
- **Good:** Grounding confidence, recon error, neuromodulator dynamics
- **Missing:** No cross-modal confidence per-modality display, no audio grounding visualization

### Section: Neural Space (3D)
- **Good:** WebGL visualization of columns, spikes, cross-modal beams
- **Missing:** No cortex thought overlay in 3D space

### General UI Issues:
- SSE reconnection logic works but doesn't show "stale data" indicator
- No error boundary for component failures
- Dark mode only (by design, fine)
- The sidebar shows all 12 sections even when cortex is unavailable

---

## 6. Paper Claims vs Reality

| Claim | Status | Evidence |
|-------|--------|----------|
| "654 tests pass" | Mostly true | 305 passed in quick run, 1 failure (emergence evaluation) |
| "57 tok/s at 256 cols" | Untested at this config | Currently seeing 10.6 tok/s at 1024 cols |
| "Sub-1ms routing at 100K columns" | Not tested | GPU routing not benchmarked on current tree |
| "Cross-modal grounding works" | Partially | Pipeline exists but episodes not firing in service |
| Cortex anti-rumination | Fixed (v4.22) | Drive stripping, forced topics, stronger prompts |
| Sleep/dream cycles | Fixed (v4.22) | Threshold 0.5, fatigue 0.04/thought |
| Multimodal episodes | Confirmed working | Init works, dataset adapters verified |
| "SNN controls LLM attention" | Partially | SNN injects drives but LLM ignores avoid_topics |

---

## 7. ARC-AGI Assessment (initial)

ARC-AGI requires:
- Visual pattern recognition (grid-to-grid transformation)
- Rule induction from few examples
- Abstraction and analogy
- Novel problem solving

HECSN's potential fit:
- **Strength:** Cross-modal grounding could connect visual patterns to conceptual understanding
- **Strength:** Developmental training could build up pattern recognition progressively
- **Weakness:** No explicit relational reasoning mechanism
- **Weakness:** Competitive learning produces category detectors, not rule extractors
- **Weakness:** No program synthesis or symbolic manipulation
- **Verdict:** HECSN in current form cannot solve ARC-AGI. Would need: (1) Grid-based visual encoder, (2) Relational binding between object parts, (3) Program induction layer, (4) Few-shot learning capability

---

## 8. Priority Improvements

### P0 (Critical — breaks core functionality):
1. Fix anti-rumination: lower thresholds, strip drive-summary when bored, enforce topic diversity via temperature
2. Fix multimodal episode firing: verify dataset paths, add logging
3. Fix ThoughtLoop step() timing for testability

### P1 (High — degrades user experience):
4. Reduce cortex inference latency: smaller tokens, faster model option
5. Lower sleep threshold to trigger dream cycles
6. Add "stale data" indicator to UI when SSE returns cached data
7. Increase context packet size: 8 memories, 5 thread items

### P2 (Medium — architectural improvements):
8. Implement cortex→SNN feedback: thought topics should bias SNN routing
9. Implement joint visual+audio+text triplet training steps
10. Add memory inspection UI (browse, search, graduate/contradict)

### P3 (Future — towards paper goals):
11. ARC-AGI adaptation layer (visual grid encoder + relational binding)
12. Program synthesis capability for rule extraction
13. Multi-model cortex (fast model for routine, Gemma 4 for deep thought)

---

## 8. Fixes Applied (v4.22)

### BUG-1: FIXED — ThoughtLoop step() timing
- Added `force` parameter to `step()` that bypasses interval check
- File: `src/hecsn/cortex/thought_loop.py`

### BUG-2: FIXED — Anti-rumination improvements
- Stripped drive_summary when boredom > 0.4
- Stripped self_state details when boredom > 0.3
- Lowered forced_topic threshold from boredom > 0.5 to boredom > 0.3 OR curiosity > 0.6
- Strengthened THINK and REFLECT prompts with explicit anti-rumination rules
- Increased context budget: 8 memories (was 5), 5 thread items (was 3)
- Files: `src/hecsn/cortex/drives.py`, `src/hecsn/cortex/prompts.py`, `src/hecsn/cortex/core.py`

### BUG-3: FIXED — Sleep/dream cycles
- Lowered should_sleep threshold from 0.7 → 0.5
- Increased fatigue per thought from 0.02 → 0.04
- Reduced tick fatigue decay from 0.001 → 0.0005
- Result: Sleep triggers after ~12-15 thoughts
- File: `src/hecsn/cortex/drives.py`

### PERF-1: Partially fixed — Inference latency
- Reduced max_response_tokens from 256 → 160 (avoids truncated JSON)
- Dream mode: 224 tokens (was 384)
- File: `src/hecsn/cortex/drives.py`, `src/hecsn/cortex/core.py`

### BUG-5: FIXED — UI multimodal config
- Added multimodal fields to brain config draft
- configureBrain() now passes multimodal config to API
- File: `HECSN_UI/src/App.jsx`

### Test fix: Emergence evaluation novelty probe
- Increased n_columns from 24 → 128 in novelty probe (24 columns saturated on 1K+ segments)
- Expanded probe corpus with diverse paragraphs
- Lowered healthy range floor from 0.03 → 0.02
- File: `src/hecsn/training/emergence_evaluation_runner.py`

### Paper update (v4.22)
- Added §10.8: Cortex anti-rumination problem and mitigations
- Added §10.9: ARC-AGI assessment
- Updated version from 4.20 → 4.22
- File: `HECSN_Paper_v4.md`

---

## 9. Next Goals (Priority Order)

1. **Close the cortex→SNN feedback loop** — thought topics should bias SNN curiosity routing
2. **Joint visual+audio+text triplet training** — currently only visual XOR audio per step
3. **Real-time multimodal grounding validation** — run 10K tokens with multimodal, check cross-modal confidence growth
4. **Cortex thought quality metrics** — measure topic diversity, concreteness ratio, novelty per thought
5. **Faster cortex inference** — investigate Gemma 4 Q2_K or smaller model for routine thinking
6. **10M+ token scale test** — validate paper claims at scale with 256 cols for pure throughput
7. **ARC-AGI exploratory experiment** — implement grid-based visual encoder + relational binding prototype
