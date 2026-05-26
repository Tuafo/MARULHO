"""Text merge helpers for Subcortex deliberation surfaces."""

from __future__ import annotations

import re

from hecsn.cortex.core import ThoughtResult


def text_keywords(text: str) -> set[str]:
    stopwords = {
        "the", "and", "with", "that", "this", "from", "into", "about", "than",
        "their", "there", "because", "while", "where", "which", "have", "has",
        "when", "then", "they", "them", "what", "how", "why", "does", "under",
        "through", "between", "could", "would", "should", "these", "those",
    }
    words: set[str] = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower()):
        word = raw.strip("'")
        if len(word) >= 4 and word not in stopwords:
            words.add(word)
    return words


def text_overlap(left: str, right: str) -> float:
    left_words = text_keywords(left)
    right_words = text_keywords(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(1.0, min(len(left_words), len(right_words)))


def statementize_question(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if lowered.startswith("open question:") or lowered.startswith("question:"):
        _, _, suffix = cleaned.partition(":")
        cleaned = suffix.strip()
        lowered = cleaned.lower()
    if not cleaned.endswith("?"):
        return cleaned

    stem = cleaned[:-1].strip()
    lower_stem = stem.lower()
    replacements = (
        ("how do ", "how "),
        ("how does ", "how "),
        ("how did ", "how "),
        ("why do ", "why "),
        ("why does ", "why "),
        ("why did ", "why "),
        ("what is ", "what "),
        ("what are ", "what "),
    )
    for prefix, replacement in replacements:
        if lower_stem.startswith(prefix):
            tail = stem[len(prefix):].strip()
            if prefix == "what is ":
                stem = f"what {tail} is"
            elif prefix == "what are ":
                stem = f"what {tail} are"
            else:
                stem = f"{replacement}{tail}"
            lower_stem = stem.lower()
            break

    yes_no_prefixes = (
        "can ", "could ", "should ", "would ", "is ", "are ",
        "do ", "does ", "did ", "was ", "were ",
    )
    if any(lower_stem.startswith(prefix) for prefix in yes_no_prefixes):
        _, _, remainder = stem.partition(" ")
        stem = f"whether {remainder.strip()}"
    elif stem and stem.split()[0] in {"What", "How", "Why", "When", "Where", "Which"}:
        stem = stem[0].lower() + stem[1:]

    return f"A key open question is {stem}."


def dedupe_sentences(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]
    if len(parts) <= 1:
        return cleaned

    kept: list[str] = []
    for part in parts:
        matched = False
        for idx, existing in enumerate(kept):
            if text_overlap(part, existing) >= 0.78:
                if len(part) > len(existing):
                    kept[idx] = part
                matched = True
                break
        if not matched:
            kept.append(part)
    return " ".join(kept).strip()


def merge_chain_results(chain: list[ThoughtResult]) -> ThoughtResult:
    """Merge deliberation chain outputs into a single auditable result."""
    if len(chain) == 1:
        return chain[0]

    final = chain[-1]
    all_topics: list[str] = []
    seen: set[str] = set()
    for result in chain:
        for topic in result.topics:
            key = topic.lower().strip()
            if key and key not in seen:
                all_topics.append(topic)
                seen.add(key)

    total_latency = sum(result.latency_ms for result in chain)

    if len(chain) >= 3:
        thought = dedupe_sentences(statementize_question(final.thought))
    else:
        observation = dedupe_sentences(chain[0].thought)
        follow_up = dedupe_sentences(statementize_question(final.thought))
        if not follow_up:
            thought = observation
        elif text_overlap(observation, follow_up) >= 0.78:
            thought = follow_up if len(follow_up) >= len(observation) else observation
        else:
            thought = dedupe_sentences(f"{observation} {follow_up}")
        if len(thought) > 300:
            clipped = thought[:300].rstrip()
            thought = clipped.rsplit(" ", 1)[0].rstrip(" ,;:") + "..."

    return ThoughtResult(
        raw_text=final.raw_text,
        thought=thought,
        topics=tuple(all_topics[:8]),
        emotional_valence=final.emotional_valence,
        confidence=final.confidence,
        action_intent=final.action_intent,
        latency_ms=total_latency,
        parse_success=final.parse_success,
    )
