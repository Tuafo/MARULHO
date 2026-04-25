"""Tests for the CorticalCore -- the LLM neocortex wrapper.

Covers: ContextPacket building, ThoughtResult parsing, MockCortex
deterministic behaviour, and transport error handling.
No Ollama -- uses NVIDIA NIM in production, MockCortex in tests.
"""

from __future__ import annotations

import json
import os
import pytest

from hecsn.cortex.core import (
    ContextPacket,
    CorticalCore,
    MockCortex,
    MemoryItem,
    ThinkingMode,
    ThoughtResult,
)
from hecsn.cortex.prompts import MODE_PROMPTS


# ---------------------------------------------------------------------------
# MemoryItem
# ---------------------------------------------------------------------------

class TestMemoryItem:
    def test_defaults(self):
        m = MemoryItem(text="hello world")
        assert m.salience == 0.5
        assert m.source == "observed"

    def test_prompt_str_format(self):
        m = MemoryItem(text="the sky is blue", salience=0.9, source="verified")
        s = m.to_prompt_str()
        assert "[verified|sal=0.90]" in s
        assert "the sky is blue" in s

    def test_all_sources(self):
        for src in ("observed", "inferred", "dreamed", "verified", "external"):
            m = MemoryItem(text="x", source=src)
            assert f"[{src}" in m.to_prompt_str()


# ---------------------------------------------------------------------------
# ContextPacket
# ---------------------------------------------------------------------------

class TestContextPacket:
    def test_empty_packet_gives_free_think(self):
        p = ContextPacket()
        prompt = p.to_user_prompt()
        assert "No context provided" in prompt

    def test_drive_summary_included(self):
        p = ContextPacket(drive_summary="explore astronomy")
        prompt = p.to_user_prompt()
        assert "## Current Drives" in prompt
        assert "explore astronomy" in prompt

    def test_drive_truncated(self):
        p = ContextPacket(drive_summary="x" * 1000)
        prompt = p.to_user_prompt()
        drive_section = prompt.split("## Current Drives\n")[1]
        assert len(drive_section) <= p.MAX_DRIVE_CHARS + 10

    def test_memories_capped(self):
        mems = [MemoryItem(text=f"mem-{i}") for i in range(10)]
        p = ContextPacket(top_memories=mems)
        prompt = p.to_user_prompt()
        count = prompt.count("[observed|sal=")
        assert count == p.MAX_MEMORIES

    def test_narrative_self_included(self):
        p = ContextPacket(
            narrative_self="I've recently been exploring coral reefs and heat stress.",
        )
        prompt = p.to_user_prompt()
        assert "## Ongoing Narrative" in prompt
        assert "coral reefs" in prompt

    def test_working_memory_narrative_included(self):
        p = ContextPacket(
            working_memory_narrative="Currently thinking about coral reefs and heat stress.",
        )
        prompt = p.to_user_prompt()
        assert "## Working Memory" in prompt
        assert "coral reefs" in prompt

    def test_external_query_slot(self):
        p = ContextPacket(
            mode=ThinkingMode.ANSWER,
            external_query="What is photosynthesis?",
        )
        prompt = p.to_user_prompt()
        assert "## External Query" in prompt
        assert "photosynthesis" in prompt

    def test_self_state(self):
        p = ContextPacket(self_state="high curiosity, moderate anxiety")
        prompt = p.to_user_prompt()
        assert "## Internal State" in prompt
        assert "curiosity" in prompt

    def test_full_assembly_order(self):
        p = ContextPacket(
            drive_summary="explore",
            self_state="calm",
            narrative_self="I've recently been exploring bridge stability.",
            working_memory_narrative="Considering bridges and balance.",
            grounded_evidence=[MemoryItem(text="e1")],
            top_memories=[MemoryItem(text="m1")],
            external_query="q1",
        )
        prompt = p.to_user_prompt()
        drives_pos = prompt.index("Current Drives")
        state_pos = prompt.index("Internal State")
        narrative_pos = prompt.index("Ongoing Narrative")
        wm_pos = prompt.index("Working Memory")
        query_pos = prompt.index("External Query")
        evidence_pos = prompt.index("Grounded Evidence")
        mem_pos = prompt.index("Relevant Memories")
        assert drives_pos < state_pos < narrative_pos < wm_pos < query_pos < evidence_pos < mem_pos


# ---------------------------------------------------------------------------
# ThoughtResult parsing
# ---------------------------------------------------------------------------

class TestThoughtResult:
    def test_valid_json(self):
        raw = json.dumps({
            "thought": "The universe is expanding",
            "topics": ["cosmology", "physics"],
            "valence": 0.6,
            "confidence": 0.85,
            "action": "explore",
        })
        r = ThoughtResult.from_json(raw, latency_ms=42.0)
        assert r.parse_success
        assert r.thought == "The universe is expanding"
        assert r.topics == ("cosmology", "physics")
        assert r.emotional_valence == 0.6
        assert r.confidence == 0.85
        assert r.action_intent == "explore"
        assert r.latency_ms == 42.0

    def test_valence_clamped(self):
        raw = json.dumps({"thought": "t", "valence": 5.0, "confidence": -2.0})
        r = ThoughtResult.from_json(raw)
        assert r.emotional_valence == 1.0
        assert r.confidence == 0.0

    def test_invalid_action_ignored(self):
        raw = json.dumps({"thought": "t", "action": "hack_the_planet"})
        r = ThoughtResult.from_json(raw)
        assert r.action_intent is None

    def test_null_action(self):
        raw = json.dumps({"thought": "t", "action": "null"})
        r = ThoughtResult.from_json(raw)
        assert r.action_intent is None

    def test_missing_fields_get_defaults(self):
        raw = json.dumps({"thought": "just a thought"})
        r = ThoughtResult.from_json(raw)
        assert r.parse_success
        assert r.topics == ()
        assert r.emotional_valence == 0.0
        assert r.confidence == 0.5
        assert r.action_intent is None

    def test_garbage_input_fallback(self):
        r = ThoughtResult.from_json("not json at all", latency_ms=5.0)
        assert not r.parse_success
        assert r.thought == "not json at all"
        assert r.confidence == 0.3

    def test_json_embedded_in_text(self):
        raw = 'Here is my answer:\n{"thought": "found it", "valence": 0.5}\nDone.'
        r = ThoughtResult.from_json(raw)
        assert r.parse_success
        assert r.thought == "found it"

    def test_empty_string(self):
        r = ThoughtResult.from_json("")
        assert not r.parse_success

    def test_topics_non_list_ignored(self):
        raw = json.dumps({"thought": "t", "topics": "single string"})
        r = ThoughtResult.from_json(raw)
        assert r.topics == ()

    def test_topics_capped_at_8(self):
        raw = json.dumps({"thought": "t", "topics": [f"t{i}" for i in range(20)]})
        r = ThoughtResult.from_json(raw)
        assert len(r.topics) == 8


# ---------------------------------------------------------------------------
# MockCortex
# ---------------------------------------------------------------------------

class TestMockCortex:
    def test_think_mode_default(self):
        cortex = MockCortex()
        ctx = ContextPacket(drive_summary="explore space")
        result = cortex.generate(ctx)
        assert result.parse_success
        assert "explore space" in result.thought
        assert result.latency_ms == 10.0

    def test_dream_mode(self):
        cortex = MockCortex()
        ctx = ContextPacket(mode=ThinkingMode.DREAM)
        result = cortex.generate(ctx)
        assert "dream" in result.thought.lower() or "connection" in result.thought.lower()

    def test_answer_mode(self):
        cortex = MockCortex()
        ctx = ContextPacket(
            mode=ThinkingMode.ANSWER,
            external_query="What is gravity?",
        )
        result = cortex.generate(ctx)
        assert "gravity" in result.thought.lower() or "query" in result.thought.lower()

    def test_custom_responses_cycle(self):
        responses = [
            {"thought": "first", "valence": 0.1, "confidence": 0.9, "action": "search"},
            {"thought": "second", "valence": -0.2, "confidence": 0.4, "action": None},
        ]
        cortex = MockCortex(responses=responses)
        ctx = ContextPacket()

        r1 = cortex.generate(ctx)
        assert r1.thought == "first"
        assert r1.action_intent == "search"

        r2 = cortex.generate(ctx)
        assert r2.thought == "second"
        assert r2.action_intent is None

        # Cycles back
        r3 = cortex.generate(ctx)
        assert r3.thought == "first"

    def test_generation_count(self):
        cortex = MockCortex()
        assert cortex.generation_count == 0
        cortex.generate(ContextPacket())
        cortex.generate(ContextPacket())
        assert cortex.generation_count == 2

    def test_is_available(self):
        cortex = MockCortex()
        assert cortex.is_available()


# ---------------------------------------------------------------------------
# CorticalCore (abstract base)
# ---------------------------------------------------------------------------

class TestCorticalCoreBase:
    def test_generate_raises_not_implemented(self):
        core = CorticalCore()
        with pytest.raises(NotImplementedError):
            core.generate(ContextPacket())

    def test_is_available_false_by_default(self):
        core = CorticalCore()
        assert not core.is_available()


# ---------------------------------------------------------------------------
# Prompts module
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_all_modes_have_prompts(self):
        for mode in ThinkingMode:
            assert mode.value in MODE_PROMPTS

    def test_json_instruction_in_all_prompts(self):
        for prompt in MODE_PROMPTS.values():
            assert "JSON" in prompt
            assert '"thought"' in prompt

    def test_memory_injection_warning(self):
        for prompt in MODE_PROMPTS.values():
            assert "DATA" in prompt or "data" in prompt


# ---------------------------------------------------------------------------
# Integration test -- real NIM (skipped if no API key)
# ---------------------------------------------------------------------------

@pytest.fixture
def live_nim_cortex():
    """Create a real NIMCortex, skip if API key not available."""
    from hecsn.cortex.multi_cortex import NIMCortex
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        pytest.skip("NVIDIA_API_KEY not set")
    cortex = NIMCortex(api_key=api_key, timeout_seconds=30.0)
    if not cortex.is_available():
        cortex.close()
        pytest.skip("NIM API not reachable")
    yield cortex
    cortex.close()


class TestNIMCortexIntegration:
    @pytest.mark.slow
    def test_basic_thought(self, live_nim_cortex):
        ctx = ContextPacket(
            drive_summary="curiosity about the nature of consciousness",
            self_state="alert, curious",
            mode=ThinkingMode.THINK,
        )
        result = live_nim_cortex.generate(ctx)
        assert result.thought
        assert len(result.thought) > 10
        assert result.latency_ms > 0

    @pytest.mark.slow
    def test_answer_mode(self, live_nim_cortex):
        ctx = ContextPacket(
            mode=ThinkingMode.ANSWER,
            external_query="What is the speed of light?",
            top_memories=[
                MemoryItem(text="Light travels at approximately 300,000 km/s", source="verified"),
            ],
        )
        result = live_nim_cortex.generate(ctx)
        assert result.thought
        assert result.parse_success

    @pytest.mark.slow
    def test_dream_mode(self, live_nim_cortex):
        ctx = ContextPacket(
            mode=ThinkingMode.DREAM,
            top_memories=[
                MemoryItem(text="Trees use photosynthesis for energy", source="observed"),
                MemoryItem(text="Solar panels convert light to electricity", source="observed"),
            ],
        )
        result = live_nim_cortex.generate(ctx)
        assert result.thought


# ---------------------------------------------------------------------------
# Shared Rate Limiter
# ---------------------------------------------------------------------------

class TestSharedRateLimiter:
    """Test that rate limiting is shared across NIMCortex instances."""

    def test_shared_across_same_key(self):
        from hecsn.cortex.rate_limit import SharedRateLimiter
        rl1 = SharedRateLimiter.for_key("test-key-12345678")
        rl2 = SharedRateLimiter.for_key("test-key-12345678")
        assert rl1 is rl2  # Same key = same instance

    def test_different_keys_get_different_limiters(self):
        from hecsn.cortex.rate_limit import SharedRateLimiter
        rl1 = SharedRateLimiter.for_key("aaaa-key-11111111")
        rl2 = SharedRateLimiter.for_key("bbbb-key-22222222")
        assert rl1 is not rl2

    def test_wait_enforces_min_interval(self):
        import time
        from hecsn.cortex.rate_limit import SharedRateLimiter
        rl = SharedRateLimiter(max_rpm=60)  # 1 req/sec
        t0 = time.time()
        rl.wait()
        rl.wait()
        elapsed = time.time() - t0
        assert elapsed >= 0.9  # Should have waited ~1 second

    def test_backoff_adds_shared_cooldown(self):
        import time
        from hecsn.cortex.rate_limit import SharedRateLimiter
        rl = SharedRateLimiter(max_rpm=6000)  # tiny min interval so cooldown dominates
        rl.backoff(0.2)
        t0 = time.time()
        rl.wait()
        elapsed = time.time() - t0
        assert elapsed >= 0.18

    def test_nim_cortex_defaults_to_20_rpm_budget(self):
        from hecsn.cortex.multi_cortex import NIMCortex
        cortex = NIMCortex(model="test-model", api_key="test-key-default-20rpm")
        try:
            assert cortex._rate_limiter._max_rpm == 20
        finally:
            cortex.close()


# ---------------------------------------------------------------------------
# Anti-rumination prompt steering
# ---------------------------------------------------------------------------

class TestAntiRuminationPrompt:
    """Test that avoidance redirects the LLM to new topics."""

    def test_avoid_topics_redirects_to_new_domain(self):
        ctx = ContextPacket(
            drive_summary="",
            avoid_topics=["photosynthesis", "chlorophyll"],
            mode=ThinkingMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        # Should redirect to concrete domains, NOT mention avoided topics
        assert "completely new domain" in prompt or "geology" in prompt
        assert "Do NOT mention" not in prompt  # Don't tell LLM what to avoid

    def test_forced_topic_with_avoidance(self):
        ctx = ContextPacket(
            drive_summary="",
            avoid_topics=["bears", "claws"],
            forced_topic="quantum computing",
            mode=ThinkingMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        assert "quantum computing" in prompt
        # Avoided topics should NOT appear in the prompt
        assert "bears" not in prompt
        assert "claws" not in prompt

    def test_forced_topic_without_avoidance(self):
        ctx = ContextPacket(
            drive_summary="",
            forced_topic="quantum computing",
            mode=ThinkingMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        assert "quantum computing" in prompt
