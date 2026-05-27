"""Tests for Subcortex language packet/result primitives.

Covers: ContextPacket building and LanguageResult parsing while retired
prompt templates stay deleted.
"""

from __future__ import annotations

import json

from hecsn.semantics.language_packet import (
    ContextPacket,
    MemoryItem,
    ReadoutMode,
)
from hecsn.semantics.language_result import LanguageResult


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
            mode=ReadoutMode.ANSWER,
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
# LanguageResult parsing
# ---------------------------------------------------------------------------

class TestLanguageResult:
    def test_valid_json(self):
        raw = json.dumps({
            "thought": "The universe is expanding",
            "topics": ["cosmology", "physics"],
            "valence": 0.6,
            "confidence": 0.85,
            "action": "explore",
        })
        r = LanguageResult.from_json(raw, latency_ms=42.0)
        assert r.parse_success
        assert r.thought == "The universe is expanding"
        assert r.topics == ("cosmology", "physics")
        assert r.emotional_valence == 0.6
        assert r.confidence == 0.85
        assert r.action_intent == "explore"
        assert r.latency_ms == 42.0

    def test_valence_clamped(self):
        raw = json.dumps({"thought": "t", "valence": 5.0, "confidence": -2.0})
        r = LanguageResult.from_json(raw)
        assert r.emotional_valence == 1.0
        assert r.confidence == 0.0

    def test_invalid_action_ignored(self):
        raw = json.dumps({"thought": "t", "action": "hack_the_planet"})
        r = LanguageResult.from_json(raw)
        assert r.action_intent is None

    def test_null_action(self):
        raw = json.dumps({"thought": "t", "action": "null"})
        r = LanguageResult.from_json(raw)
        assert r.action_intent is None

    def test_missing_fields_get_defaults(self):
        raw = json.dumps({"thought": "just a thought"})
        r = LanguageResult.from_json(raw)
        assert r.parse_success
        assert r.topics == ()
        assert r.emotional_valence == 0.0
        assert r.confidence == 0.5
        assert r.action_intent is None

    def test_garbage_input_fallback(self):
        r = LanguageResult.from_json("not json at all", latency_ms=5.0)
        assert not r.parse_success
        assert r.thought == "not json at all"
        assert r.confidence == 0.3

    def test_json_embedded_in_text(self):
        raw = 'Here is my answer:\n{"thought": "found it", "valence": 0.5}\nDone.'
        r = LanguageResult.from_json(raw)
        assert r.parse_success
        assert r.thought == "found it"

    def test_empty_string(self):
        r = LanguageResult.from_json("")
        assert not r.parse_success

    def test_topics_non_list_ignored(self):
        raw = json.dumps({"thought": "t", "topics": "single string"})
        r = LanguageResult.from_json(raw)
        assert r.topics == ()

    def test_topics_capped_at_8(self):
        raw = json.dumps({"thought": "t", "topics": [f"t{i}" for i in range(20)]})
        r = LanguageResult.from_json(raw)
        assert len(r.topics) == 8

# ---------------------------------------------------------------------------
# Prompts module
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Anti-rumination prompt steering
# ---------------------------------------------------------------------------

class TestAntiRuminationPrompt:
    """Test that avoidance redirects language/readout context to new topics."""

    def test_avoid_topics_redirects_to_new_domain(self):
        ctx = ContextPacket(
            drive_summary="",
            avoid_topics=["photosynthesis", "chlorophyll"],
            mode=ReadoutMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        # Should redirect to concrete domains, NOT mention avoided topics
        assert "completely new domain" in prompt or "geology" in prompt
        assert "Do NOT mention" not in prompt

    def test_forced_topic_with_avoidance(self):
        ctx = ContextPacket(
            drive_summary="",
            avoid_topics=["bears", "claws"],
            forced_topic="quantum computing",
            mode=ReadoutMode.THINK,
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
            mode=ReadoutMode.THINK,
        )
        prompt = ctx.to_user_prompt()
        assert "quantum computing" in prompt
