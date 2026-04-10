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
_MAX_FOCUSED_CHUNK_WINDOW = 8
_MAX_FOCUSED_CHUNKS = 4


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


def _append_unit_span(
    spans: list[tuple[str, int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    unit = _normalize_run(text[start:end])
    if unit:
        spans.append((unit, start, end))


def _raw_stream_unit_spans(text: str) -> list[tuple[str, int, int]]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    spans: list[tuple[str, int, int]] = []
    current_start: int | None = None
    current: list[str] = []
    for idx, ch in enumerate(normalized):
        if ch.isalnum() or ch == "'":
            if current_start is None:
                current_start = idx
                current = [ch]
                continue
            if current and current[-1].islower() and ch.isupper():
                _append_unit_span(spans, normalized, current_start, idx)
                current_start = idx
                current = [ch]
                continue
            if (
                current
                and current[-1].isupper()
                and ch.islower()
                and len(current) > 1
                and current[-2].isupper()
            ):
                split_index = idx - 1
                _append_unit_span(spans, normalized, current_start, split_index)
                current_start = split_index
                current = [normalized[split_index], ch]
                continue
            current.append(ch)
            continue
        if current_start is not None:
            _append_unit_span(spans, normalized, current_start, idx)
            current_start = None
            current = []
    if current_start is not None:
        _append_unit_span(spans, normalized, current_start, len(normalized))
    return spans


def _raw_stream_units(text: str) -> list[str]:
    return [unit for unit, _, _ in _raw_stream_unit_spans(text)]


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


def _chunk_text_slice(text: str, start: int, end: int) -> str:
    finish = end
    while finish < len(text) and text[finish] in "\"')]}":
        finish += 1
    if finish < len(text) and text[finish] in ".,;:!?":
        finish += 1
    return text[start:finish].strip(" ,;:")


def _chunk_signal_unit_count(text: str) -> int:
    count = 0
    for unit in _raw_stream_units(text):
        compact = _compact_unit(unit)
        if not compact:
            continue
        if compact in _LOW_SIGNAL_UNITS:
            continue
        count += 1
    return count


def _query_chunk_candidates(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []

    sentences = list(split_sentences(normalized))
    if len(sentences) > 1:
        return sentences

    spans = _raw_stream_unit_spans(normalized)
    if not spans:
        return [normalized]

    candidates: list[str] = []
    if normalized.endswith((".", "!", "?")) or len(spans) <= 3:
        candidates.append(normalized)

    min_window = 2 if len(spans) <= 4 else 3
    max_window = min(_MAX_FOCUSED_CHUNK_WINDOW, len(spans) - 1)
    if max_window < min_window:
        return candidates or [normalized]
    for window_size in range(min_window, max_window + 1):
        for start_idx in range(0, len(spans) - window_size + 1):
            start = spans[start_idx][1]
            end = spans[start_idx + window_size - 1][2]
            chunk = _chunk_text_slice(normalized, start, end)
            if len(_compact_stream(chunk)) < 8:
                continue
            candidates.append(chunk)
    return _dedupe_keep_order(candidates)


def _chunks_overlap(left: str, right: str) -> bool:
    left_norm = normalize_text(left).lower()
    right_norm = normalize_text(right).lower()
    if not left_norm or not right_norm:
        return False
    return left_norm in right_norm or right_norm in left_norm


def query_focused_clauses(text: str, query_terms: Sequence[str]) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if not query_terms:
        return [normalized]

    sentences = split_sentences(normalized)
    if len(sentences) == 1 and normalized.endswith((".", "!", "?")):
        matched_full_sentence = match_terms(query_terms, normalized)
        if len(matched_full_sentence) >= len(_dedupe_keep_order([str(term) for term in query_terms])):
            return [normalized]

    clause_candidates = _query_chunk_candidates(normalized)
    scored: list[dict[str, float | int | str | tuple[str, ...]]] = []
    for clause in clause_candidates:
        matched_terms = tuple(match_terms(query_terms, clause))
        if not matched_terms:
            continue
        signal_units = max(1, _chunk_signal_unit_count(clause))
        scored.append(
            {
                "text": clause,
                "match_count": int(len(matched_terms)),
                "density": float(len(matched_terms) / float(signal_units)),
                "complete_sentence": int(clause.endswith((".", "!", "?"))),
                "signal_units": int(signal_units),
                "position": int(normalized.lower().find(clause.lower())),
                "matched_terms": matched_terms,
            }
        )

    if not scored:
        clauses = split_sentences(normalized)
        focused = [clause for clause in clauses if match_terms(query_terms, clause)]
        return focused or clauses

    scored.sort(
        key=lambda item: (
            int(item["match_count"]),
            float(item["density"]),
            int(item["complete_sentence"]),
            -int(item["signal_units"]),
        ),
        reverse=True,
    )
    best_match_count = int(scored[0]["match_count"])
    best_density = float(scored[0]["density"])
    covered_terms: set[str] = set()
    selected: list[dict[str, float | int | str | tuple[str, ...]]] = []
    for item in scored:
        matched_terms = set(item["matched_terms"])
        adds_new_terms = bool(matched_terms - covered_terms)
        close_to_best = (
            int(item["match_count"]) >= max(1, best_match_count - 1)
            and float(item["density"]) >= max(0.20, best_density * 0.70)
        )
        if selected and not adds_new_terms and not close_to_best:
            continue
        if any(_chunks_overlap(str(item["text"]), str(existing["text"])) for existing in selected):
            continue
        selected.append(item)
        covered_terms.update(matched_terms)
        if len(selected) >= _MAX_FOCUSED_CHUNKS:
            break

    if not selected:
        clauses = split_sentences(normalized)
        focused = [clause for clause in clauses if match_terms(query_terms, clause)]
        return focused or clauses

    selected.sort(key=lambda item: (int(item["position"]), -len(str(item["text"]))))
    return [str(item["text"]) for item in selected]


def query_focused_text(text: str, query_terms: Sequence[str]) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    focused_clauses = query_focused_clauses(normalized, query_terms)
    focused_text = " ".join(focused_clauses).strip()
    if not focused_text:
        return normalized

    if len(split_sentences(normalized)) == 1 and normalized.endswith((".", "!", "?")):
        if len(match_terms(query_terms, normalized)) >= len(match_terms(query_terms, focused_text)):
            return normalized
    return focused_text
