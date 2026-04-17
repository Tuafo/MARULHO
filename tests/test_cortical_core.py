"""Tests for the CorticalCore — the LLM neocortex wrapper.

Covers: ContextPacket building, ThoughtResult parsing, FakeCortex
deterministic behaviour, transport error handling, and (optionally)
real Ollama integration.
"""

from __future__ import annotations

import json
import pytest

from hecsn.cortex.core import (
    ContextPacket,
    CorticalCore,
    FakeCortex,
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

    def test_thread_capped_to_recent(self):
        thread = [f"thought-{i}" for i in range(10)]
        p = ContextPacket(recent_thread=thread)
        prompt = p.to_user_prompt()
        assert "thought-7" in prompt
        assert "thought-9" in prompt
        assert "thought-0" not in prompt

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
            top_memories=[MemoryItem(text="m1")],
            recent_thread=["t1"],
            external_query="q1",
        )
        prompt = p.to_user_prompt()
        drives_pos = prompt.index("Current Drives")
        state_pos = prompt.index("Internal State")
        mem_pos = prompt.index("Relevant Memories")
        thread_pos = prompt.index("Recent Thoughts")
        query_pos = prompt.index("External Query")
        assert drives_pos < state_pos < mem_pos < thread_pos < query_pos


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
# FakeCortex
# ---------------------------------------------------------------------------

class TestFakeCortex:
    def test_think_mode_default(self):
        cortex = FakeCortex()
        ctx = ContextPacket(drive_summary="explore space")
        result = cortex.generate(ctx)
        assert result.parse_success
        assert "explore space" in result.thought
        assert result.latency_ms == 10.0

    def test_dream_mode(self):
        cortex = FakeCortex()
        ctx = ContextPacket(mode=ThinkingMode.DREAM)
        result = cortex.generate(ctx)
        assert "dream" in result.thought.lower() or "connection" in result.thought.lower()

    def test_answer_mode(self):
        cortex = FakeCortex()
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
        cortex = FakeCortex(responses=responses)
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
        cortex = FakeCortex()
        assert cortex.generation_count == 0
        cortex.generate(ContextPacket())
        cortex.generate(ContextPacket())
        assert cortex.generation_count == 2

    def test_is_available(self):
        cortex = FakeCortex()
        assert cortex.is_available()

    def test_ollama_raises(self):
        cortex = FakeCortex()
        with pytest.raises(NotImplementedError):
            cortex._call_ollama("sys", "usr", 100)


# ---------------------------------------------------------------------------
# CorticalCore (unit — transport mocked)
# ---------------------------------------------------------------------------

class TestCorticalCoreUnit:
    def test_loopback_restriction(self):
        with pytest.raises(ValueError, match="loopback"):
            CorticalCore(base_url="http://evil.com:11434")

    def test_localhost_allowed(self):
        # Should not raise (won't connect, just validates URL)
        core = CorticalCore(base_url="http://localhost:11434")
        assert core.model == "gemma4:e4b"
        core.close()

    def test_127_allowed(self):
        core = CorticalCore(base_url="http://127.0.0.1:11434")
        core.close()

    def test_generate_handles_connection_error(self):
        # Use a port that nothing listens on
        core = CorticalCore(base_url="http://127.0.0.1:19999", timeout_seconds=2.0)
        result = core.generate(ContextPacket(drive_summary="test"))
        assert not result.parse_success
        assert result.confidence == 0.0
        assert "cortex unavailable" in result.thought
        core.close()


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
# Integration test — real Ollama (skipped if unavailable)
# ---------------------------------------------------------------------------

@pytest.fixture
def live_cortex():
    """Create a real CorticalCore, skip if Ollama is down."""
    core = CorticalCore(model="gemma4:e4b", timeout_seconds=120.0)
    if not core.is_available():
        core.close()
        pytest.skip("Ollama not available or model not loaded")
    yield core
    core.close()


class TestCorticalCoreIntegration:
    @pytest.mark.slow
    def test_basic_thought(self, live_cortex: CorticalCore):
        ctx = ContextPacket(
            drive_summary="curiosity about the nature of consciousness",
            self_state="alert, curious",
            mode=ThinkingMode.THINK,
        )
        result = live_cortex.generate(ctx)
        assert result.thought
        assert len(result.thought) > 10
        assert result.latency_ms > 0

    @pytest.mark.slow
    def test_answer_mode(self, live_cortex: CorticalCore):
        ctx = ContextPacket(
            mode=ThinkingMode.ANSWER,
            external_query="What is the speed of light?",
            top_memories=[
                MemoryItem(text="Light travels at approximately 300,000 km/s", source="verified"),
            ],
        )
        result = live_cortex.generate(ctx)
        assert result.thought
        assert result.parse_success

    @pytest.mark.slow
    def test_dream_mode(self, live_cortex: CorticalCore):
        ctx = ContextPacket(
            mode=ThinkingMode.DREAM,
            top_memories=[
                MemoryItem(text="Trees use photosynthesis for energy", source="observed"),
                MemoryItem(text="Solar panels convert light to electricity", source="observed"),
            ],
        )
        result = live_cortex.generate(ctx)
        assert result.thought
