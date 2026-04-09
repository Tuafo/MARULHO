from __future__ import annotations

from difflib import SequenceMatcher
import math
import re
from collections import Counter
from typing import Any, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_LOW_SIGNAL_UNITS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "kind",
    "me",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "place",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "thing",
    "this",
    "to",
    "type",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "yours",
}
_MAX_COMPOUND_WINDOW = 3


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = normalize_text(value).lower()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _normalize_run(value: str) -> str:
    cleaned = "".join(ch.lower() for ch in str(value) if ch.isalnum() or ch == "'")
    return cleaned.strip("'")


def _compact_stream(value: str) -> str:
    return "".join(ch.lower() for ch in normalize_text(value) if ch.isalnum())


def _compact_unit(value: str) -> str:
    normalized = _normalize_run(value)
    return "".join(ch for ch in normalized if ch.isalnum())


def _append_unit(units: list[str], raw_value: str) -> None:
    unit = _normalize_run(raw_value)
    if unit:
        units.append(unit)


def _raw_stream_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    units: list[str] = []
    current: list[str] = []
    for ch in normalized:
        if ch.isalnum() or ch == "'":
            if (
                current
                and current[-1].islower()
                and ch.isupper()
            ):
                _append_unit(units, "".join(current))
                current = [ch]
                continue
            if (
                current
                and current[-1].isupper()
                and ch.islower()
                and len(current) > 1
                and current[-2].isupper()
            ):
                _append_unit(units, "".join(current[:-1]))
                current = [current[-1], ch]
                continue
            current.append(ch)
            continue
        if current:
            _append_unit(units, "".join(current))
            current = []
    if current:
        _append_unit(units, "".join(current))
    return units


def _compound_stream_units(
    raw_units: Sequence[str],
    *,
    max_window: int = _MAX_COMPOUND_WINDOW,
) -> list[str]:
    compacts: list[str] = []
    for unit in raw_units:
        compact = _compact_unit(unit)
        if compact and not (len(compact) == 1 and not compact.isdigit()):
            compacts.append(compact)
    compounds: list[str] = []
    window_limit = min(max(1, int(max_window)), len(compacts))
    for window_size in range(2, window_limit + 1):
        for start in range(0, len(compacts) - window_size + 1):
            parts = compacts[start: start + window_size]
            if not parts:
                continue
            if all(part in _LOW_SIGNAL_UNITS for part in parts):
                continue
            joined = "".join(parts)
            if len(joined) >= max(6, 3 * window_size):
                compounds.append(joined)

    strong_positions = [
        idx
        for idx, part in enumerate(compacts)
        if part not in _LOW_SIGNAL_UNITS
    ]
    max_skip_span = max(4, int(max_window) + 2)
    for start_idx, raw_start in enumerate(strong_positions):
        for mid_idx in range(start_idx + 1, len(strong_positions)):
            raw_mid = strong_positions[mid_idx]
            if raw_mid - raw_start > max_skip_span:
                break
            pair = compacts[raw_start] + compacts[raw_mid]
            if len(pair) >= 8:
                compounds.append(pair)
            for end_idx in range(mid_idx + 1, len(strong_positions)):
                raw_end = strong_positions[end_idx]
                if raw_end - raw_start > max_skip_span:
                    break
                triple = pair + compacts[raw_end]
                if len(triple) >= 12:
                    compounds.append(triple)
    return _dedupe_keep_order(compounds)


def stream_matching_units(
    text: str,
    *,
    max_compound_size: int = _MAX_COMPOUND_WINDOW,
) -> tuple[str, ...]:
    raw_units = _raw_stream_units(text)
    ordered: list[str] = []
    for unit in raw_units:
        compact = _compact_unit(unit)
        if not compact or (len(compact) == 1 and not compact.isdigit()):
            continue
        ordered.append(compact)
    ordered.extend(_compound_stream_units(raw_units, max_window=max_compound_size))
    compact_stream = _compact_stream(text)
    if compact_stream and len(raw_units) <= max_compound_size and len(compact_stream) >= 8:
        ordered.append(compact_stream)
    return tuple(_dedupe_keep_order(ordered))


def stream_unit_profile(
    text: str,
    *,
    max_compound_size: int = _MAX_COMPOUND_WINDOW,
) -> dict[str, float]:
    counter: Counter[str] = Counter()
    raw_units = _raw_stream_units(text)
    for rank, unit in enumerate(raw_units):
        compact = _compact_unit(unit)
        if not compact or (len(compact) == 1 and not compact.isdigit()):
            continue
        weight = 1.0 / math.sqrt(float(rank + 1))
        if compact in _LOW_SIGNAL_UNITS:
            weight *= 0.35
        counter[compact] += weight
    for rank, unit in enumerate(_compound_stream_units(raw_units, max_window=max_compound_size)):
        counter[unit] += 0.45 / math.sqrt(float(rank + 1))
    return {
        term: float(weight)
        for term, weight in counter.items()
        if float(weight) > 0.0
    }


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for unit in _raw_stream_units(text):
        compact = _compact_unit(unit)
        if not compact:
            continue
        if compact in _LOW_SIGNAL_UNITS:
            continue
        if len(compact) == 1 and not compact.isdigit():
            continue
        tokens.append(compact)
    if not tokens:
        compact_stream = _compact_stream(text)
        if compact_stream and not (len(compact_stream) == 1 and not compact_stream.isdigit()):
            tokens.append(compact_stream)
    return _dedupe_keep_order(tokens)


def salient_query_terms(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _inflection_variants(value: str) -> tuple[str, ...]:
    normalized = _compact_stream(value)
    if not normalized:
        return ()

    variants: list[str] = []

    def add(item: str) -> None:
        compact = _compact_stream(item)
        if not compact or compact in variants:
            return
        variants.append(compact)

    add(normalized)
    if len(normalized) > 4 and normalized.endswith("ies"):
        add(normalized[:-3] + "y")
    if len(normalized) > 4 and normalized.endswith("oes"):
        add(normalized[:-2])
    if len(normalized) > 4 and normalized.endswith(("ches", "shes", "sses", "xes", "zes")):
        add(normalized[:-2])
    elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        add(normalized[:-1])
    if len(normalized) > 5 and normalized.endswith("ing"):
        add(normalized[:-3])
    if len(normalized) > 4 and normalized.endswith("ed"):
        add(normalized[:-2])
    return tuple(variants)


def token_forms(token: str) -> tuple[str, ...]:
    normalized = normalize_text(token)
    if not normalized:
        return ()

    variants: list[str] = []

    def add_forms(value: str) -> None:
        for item in _inflection_variants(value):
            if item not in variants:
                variants.append(item)

    raw_units = _raw_stream_units(normalized)
    if raw_units:
        for unit in raw_units:
            add_forms(unit)
        for unit in _compound_stream_units(raw_units):
            add_forms(unit)
    else:
        add_forms(normalized)
    return tuple(variants)


def _char_ngrams(value: str, *, min_n: int = 3, max_n: int = 5) -> set[str]:
    compact = _compact_stream(value)
    if not compact:
        return set()
    if len(compact) < int(min_n):
        return {compact} if len(compact) >= 2 else set()

    grams: set[str] = set()
    upper = min(max_n, len(compact))
    for size in range(max(1, int(min_n)), upper + 1):
        for start in range(0, len(compact) - size + 1):
            gram = compact[start: start + size]
            if len(set(gram)) <= 1:
                continue
            grams.add(gram)
    return grams


def semantic_unit_similarity(left: str, right: str) -> float:
    left_forms = set(token_forms(left))
    right_forms = set(token_forms(right))
    if left_forms and right_forms and left_forms & right_forms:
        return 1.0

    left_compact = _compact_stream(left)
    right_compact = _compact_stream(right)
    if not left_compact or not right_compact:
        return 0.0
    if left_compact == right_compact:
        return 1.0

    shorter, longer = (
        (left_compact, right_compact)
        if len(left_compact) <= len(right_compact)
        else (right_compact, left_compact)
    )
    if len(shorter) >= 4 and shorter in longer:
        ratio = float(len(shorter) / max(1, len(longer)))
        if ratio >= 0.70:
            return ratio

    if len(shorter) >= 6:
        sequence_ratio = float(SequenceMatcher(a=left_compact, b=right_compact).ratio())
        if sequence_ratio >= 0.88:
            return float(sequence_ratio * float(len(shorter) / max(1, len(longer))))

    min_n = 3 if min(len(left_compact), len(right_compact)) < 10 else 4
    left_grams = _char_ngrams(left_compact, min_n=min_n)
    right_grams = _char_ngrams(right_compact, min_n=min_n)
    if not left_grams or not right_grams:
        return 0.0

    coverage = float(len(left_grams & right_grams)) / float(max(1, min(len(left_grams), len(right_grams))))
    length_ratio = float(len(shorter) / max(1, len(longer)))
    if coverage >= 0.85 and length_ratio >= 0.60:
        return float(coverage * length_ratio)
    if coverage >= 0.75 and length_ratio >= 0.80:
        return float(coverage * length_ratio)
    return 0.0


def term_match_score(term: str, candidates: Sequence[str]) -> float:
    best = 0.0
    for candidate in candidates:
        best = max(best, semantic_unit_similarity(term, candidate))
    return float(best)


def match_terms(query_terms: Sequence[str], text: str) -> list[str]:
    evidence_terms = stream_matching_units(text)
    if not query_terms or not evidence_terms:
        return []

    matches: list[str] = []
    for term in query_terms:
        if any(semantic_unit_similarity(term, evidence_term) >= 0.70 for evidence_term in evidence_terms):
            if term not in matches:
                matches.append(term)
    return matches


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    segments = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(normalized) if segment.strip()]
    return segments or [normalized]


def query_focused_clauses(text: str, query_terms: Sequence[str]) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if not query_terms:
        return [normalized]

    clauses = split_sentences(normalized)
    if len(clauses) <= 1:
        return clauses

    focused = [clause for clause in clauses if match_terms(query_terms, clause)]
    return focused or clauses


def query_focused_text(text: str, query_terms: Sequence[str]) -> str:
    return " ".join(query_focused_clauses(text, query_terms)).strip()
