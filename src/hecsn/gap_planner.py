from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS = {
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
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def tokenize_terms(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        if match in _STOPWORDS:
            continue
        if len(match) == 1 and not match.isdigit():
            continue
        tokens.append(match)
    return tokens


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _frontier_terms(text: str) -> list[str]:
    return [
        term
        for term in tokenize_terms(text)
        if len(term) >= 4 or term.isdigit()
    ]


def _suffix_prefix_overlap(left: str, right: str, *, min_overlap: int = 3) -> int:
    left_norm = _normalize_text(left).lower()
    right_norm = _normalize_text(right).lower()
    max_overlap = min(len(left_norm), len(right_norm))
    for size in range(max_overlap, max(1, int(min_overlap)) - 1, -1):
        if left_norm[-size:] == right_norm[:size]:
            return size
    return 0


def _merge_frontier_windows(texts: Sequence[str], *, min_overlap: int = 3) -> list[str]:
    merged: list[str] = []
    for raw_text in texts:
        text = _normalize_text(raw_text)
        if not text:
            continue
        if not merged:
            merged.append(text)
            continue

        updated = False
        for idx, current in enumerate(list(merged)):
            current_norm = current.lower()
            text_norm = text.lower()
            if text_norm in current_norm:
                updated = True
                break
            if current_norm in text_norm:
                merged[idx] = text
                updated = True
                break

            append_overlap = _suffix_prefix_overlap(current, text, min_overlap=min_overlap)
            prepend_overlap = _suffix_prefix_overlap(text, current, min_overlap=min_overlap)
            if append_overlap <= 0 and prepend_overlap <= 0:
                continue
            if append_overlap >= prepend_overlap:
                merged[idx] = _normalize_text(current + text[append_overlap:])
            else:
                merged[idx] = _normalize_text(text + current[prepend_overlap:])
            updated = True
            break

        if not updated:
            merged.append(text)
    return _dedupe_keep_order(merged)


def _top_term_payload(counter: Counter[str], limit: int) -> list[dict[str, Any]]:
    ranked = counter.most_common(max(1, int(limit)))
    return [
        {
            "term": term,
            "weight": float(weight),
        }
        for term, weight in ranked
        if float(weight) > 0.0
    ]


def _query_term_support(
    query_terms: set[str],
    memory_matches: Sequence[Mapping[str, Any]],
    *,
    max_matches: int = 8,
) -> dict[str, float]:
    support = {term: 0.0 for term in query_terms}
    ranked = sorted(
        memory_matches,
        key=lambda item: float(item.get("similarity", 0.0)),
        reverse=True,
    )[: max(1, int(max_matches))]
    for rank, match in enumerate(ranked):
        text = _normalize_text(match.get("raw_window") or match.get("text"))
        if not text:
            continue
        match_terms = set(tokenize_terms(text))
        if not match_terms:
            continue
        similarity = max(0.0, float(match.get("similarity", 0.0)))
        strength = similarity / math.sqrt(float(rank + 1))
        for term in query_terms & match_terms:
            support[term] = max(float(support.get(term, 0.0)), strength)
    return support


def plan_query_gaps(
    *,
    query_text: str,
    query_summary: Mapping[str, Any] | None,
    concept_summary: Mapping[str, Any] | None,
    max_terms: int = 8,
    max_questions: int = 4,
    max_queries: int = 4,
) -> dict[str, Any]:
    query_text_norm = _normalize_text(query_text)
    query_terms = _dedupe_keep_order(tokenize_terms(query_text_norm))
    query_term_set = set(query_terms)
    query_summary = query_summary or {}
    concept_summary = concept_summary or {}
    memory_matches = list(query_summary.get("memory_matches") or [])
    concepts = list(concept_summary.get("concepts") or [])

    term_support = _query_term_support(query_term_set, memory_matches)
    unsupported_terms = [
        term
        for term in query_terms
        if float(term_support.get(term, 0.0)) < 0.30
    ]

    gap_counter: Counter[str] = Counter()
    weak_concepts: list[dict[str, Any]] = []
    for concept in concepts:
        label = _normalize_text(concept.get("label"))
        top_terms = _dedupe_keep_order(
            [
                *[str(term) for term in concept.get("top_terms") or []],
                *tokenize_terms(label),
            ]
        )
        uncertainty = max(0.0, min(1.0, float(concept.get("uncertainty", 1.0))))
        drift = max(0.0, min(1.0, float(concept.get("drift", 0.0))))
        match_count = max(0, int(concept.get("match_count", 0)))
        observations = max(0, int(concept.get("observations", 0)))
        weakness = 0.55 * uncertainty + 0.45 * drift
        if match_count <= 1:
            weakness += 0.15
        if observations <= 1:
            weakness += 0.10
        weakness = max(0.0, min(1.0, weakness))
        if not label and not top_terms:
            continue
        if weakness < 0.25:
            continue
        weak_concepts.append(
            {
                "label": label,
                "weakness": float(weakness),
                "uncertainty": float(uncertainty),
                "drift": float(drift),
                "top_terms": top_terms[:4],
                "match_count": int(match_count),
            }
        )
        for rank, term in enumerate(top_terms[:4]):
            gap_counter[term] += float(weakness) / float(rank + 1)

    for term in unsupported_terms:
        gap_counter[term] += 2.0

    gap_terms = _top_term_payload(gap_counter, max_terms)
    retrieval_queries = _dedupe_keep_order(
        [
            " ".join(unsupported_terms[:3]).strip(),
            *[
                " ".join(concept["top_terms"][:3]).strip()
                for concept in weak_concepts
            ],
            *[
                _normalize_text(concept["label"])
                for concept in weak_concepts
            ],
        ]
    )[: max(1, int(max_queries))]

    follow_up_questions: list[str] = []
    for term in unsupported_terms[: max(1, int(max_questions))]:
        if weak_concepts:
            follow_up_questions.append(
                f"What grounded evidence connects {term} to {weak_concepts[0]['label']}?"
            )
        else:
            follow_up_questions.append(
                f"What grounded evidence is still missing for {term}?"
            )
    for concept in weak_concepts:
        if len(follow_up_questions) >= max(1, int(max_questions)):
            break
        label = concept["label"] or "/".join(concept["top_terms"])
        if concept["uncertainty"] >= concept["drift"]:
            follow_up_questions.append(
                f"What distinguishes {label} from nearby concepts in memory?"
            )
        else:
            follow_up_questions.append(
                f"What stable evidence would reduce drift for {label}?"
            )

    grounded_fraction = 1.0
    if query_terms:
        grounded_fraction = float(
            sum(1 for term in query_terms if term not in unsupported_terms)
            / max(1, len(query_terms))
        )

    return {
        "planner_mode": "semantic_gap_planner",
        "query_terms": query_terms,
        "unsupported_terms": unsupported_terms,
        "grounded_fraction": float(grounded_fraction),
        "gap_terms": gap_terms,
        "weak_concepts": weak_concepts[: max(1, int(max_questions))],
        "retrieval_queries": retrieval_queries,
        "follow_up_questions": _dedupe_keep_order(follow_up_questions)[: max(1, int(max_questions))],
    }


def _frontier_scored_entries(
    *,
    memory_store: Any,
    current_token: int,
    top_entries: int,
) -> list[tuple[float, str]]:
    if memory_store is None:
        return []

    windows = list(getattr(memory_store, "slow_raw_windows", []) or [])
    if not windows:
        return []

    scored_entries: list[tuple[float, str]] = []
    for idx, raw_window in enumerate(windows):
        text = _normalize_text(raw_window)
        if not text:
            continue
        importance_values = list(getattr(memory_store, "slow_importance", []) or [])
        capture_values = list(getattr(memory_store, "slow_capture_tag", []) or [])
        consolidation_values = list(getattr(memory_store, "slow_consolidation_level", []) or [])
        importance = float(importance_values[idx]) if idx < len(importance_values) else 0.0
        if hasattr(memory_store, "_effective_capture_strength"):
            capture = float(memory_store._effective_capture_strength(idx, current_token))
        else:
            capture = float(capture_values[idx]) if idx < len(capture_values) else 0.0
        consolidation = float(consolidation_values[idx]) if idx < len(consolidation_values) else 0.0
        frontier_pressure = max(0.0, capture - consolidation) + 0.5 * max(0.0, 1.0 - consolidation)
        score = max(1e-6, importance) * (1.0 + frontier_pressure)
        scored_entries.append((score, text))

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    return scored_entries[: max(1, int(top_entries))]


def frontier_gap_terms(
    *,
    memory_store: Any,
    current_token: int,
    limit: int = 8,
    top_entries: int = 24,
) -> list[dict[str, Any]]:
    scored_entries = _frontier_scored_entries(
        memory_store=memory_store,
        current_token=current_token,
        top_entries=top_entries,
    )
    if not scored_entries:
        return []

    counter: Counter[str] = Counter()
    for rank, (score, text) in enumerate(scored_entries):
        for pos, term in enumerate(tokenize_terms(text)[:8]):
            counter[term] += float(score) / (math.sqrt(float(rank + 1)) * float(pos + 1))
    return _top_term_payload(counter, limit)


def frontier_gap_plan(
    *,
    memory_store: Any,
    current_token: int,
    max_terms: int = 8,
    max_queries: int = 4,
    max_questions: int = 4,
    top_entries: int = 24,
) -> dict[str, Any]:
    scored_entries = _frontier_scored_entries(
        memory_store=memory_store,
        current_token=current_token,
        top_entries=top_entries,
    )
    gap_terms = frontier_gap_terms(
        memory_store=memory_store,
        current_token=current_token,
        limit=max_terms,
        top_entries=top_entries,
    )
    if not scored_entries:
        return {
            "planner_mode": "frontier_gap_planner",
            "gap_terms": [],
            "unsupported_terms": [],
            "retrieval_queries": [],
            "follow_up_questions": [],
            "frontier_windows": [],
            "frontier_phrases": [],
        }

    frontier_windows = [text for _, text in scored_entries[: max(1, int(max_questions) * 2)]]
    frontier_phrases = _merge_frontier_windows(frontier_windows)
    retrieval_queries = _dedupe_keep_order(
        [
            " ".join(_frontier_terms(text)[:3]).strip()
            for text in frontier_phrases[: max(1, int(max_queries) * 2)]
        ]
        + [
            " ".join(
                [
                    item["term"]
                    for item in gap_terms[offset: offset + 3]
                    if str(item.get("term", "")).strip()
                ]
            ).strip()
            for offset in range(0, min(len(gap_terms), max(1, int(max_queries) * 2)), 2)
        ]
    )[: max(1, int(max_queries))]

    follow_up_questions: list[str] = []
    for text in frontier_phrases[: max(1, int(max_questions))]:
        terms = _frontier_terms(text)
        if len(terms) >= 2:
            follow_up_questions.append(
                f"What grounded evidence links {terms[0]} and {terms[1]} in current frontier memory?"
            )
        elif terms:
            follow_up_questions.append(
                f"What stable evidence is still missing for {terms[0]}?"
            )
    for item in gap_terms:
        if len(follow_up_questions) >= max(1, int(max_questions)):
            break
        term = str(item.get("term", "")).strip()
        if term:
            follow_up_questions.append(
                f"What grounded evidence would stabilize {term} in memory?"
            )

    return {
        "planner_mode": "frontier_gap_planner",
        "gap_terms": gap_terms,
        "unsupported_terms": [str(item["term"]) for item in gap_terms[: max(1, int(max_questions))]],
        "retrieval_queries": retrieval_queries,
        "follow_up_questions": _dedupe_keep_order(follow_up_questions)[: max(1, int(max_questions))],
        "frontier_windows": [text for _, text in scored_entries[: max(1, int(max_questions))]],
        "frontier_phrases": frontier_phrases[: max(1, int(max_questions))],
    }


def _add_weighted_terms(counter: Counter[str], texts: Sequence[str], *, scale: float) -> None:
    for text_rank, text in enumerate(texts):
        for term_rank, term in enumerate(tokenize_terms(text)[:8]):
            counter[term] += float(scale) / (
                math.sqrt(float(text_rank + 1)) * float(term_rank + 1)
            )


def candidate_bank_term_profile(
    bank: Any,
    *,
    max_terms: int = 64,
    max_windows: int = 96,
) -> dict[str, float]:
    counter: Counter[str] = Counter()
    name = _normalize_text(getattr(bank, "name", ""))
    source = _normalize_text(getattr(bank, "source", ""))
    for rank, term in enumerate(tokenize_terms(name)):
        counter[term] += 2.0 / float(rank + 1)

    if source:
        parsed = urlparse(source)
        source_parts = [
            parsed.netloc.replace(".", " "),
            parsed.path.replace("/", " "),
            source if not parsed.scheme else "",
        ]
        for part in source_parts:
            for rank, term in enumerate(tokenize_terms(part)):
                counter[term] += 1.0 / float(rank + 1)

    probe_windows = list(getattr(bank, "probe_raw_windows", []) or [])
    _add_weighted_terms(counter, probe_windows[: max(1, int(max_windows // 2))], scale=1.25)

    windows = list(getattr(bank, "train_raw_windows", []) or [])
    _add_weighted_terms(counter, windows[: max(1, int(max_windows))], scale=1.0)

    ranked = counter.most_common(max(1, int(max_terms)))
    return {term: float(weight) for term, weight in ranked if float(weight) > 0.0}


def _weighted_text_profile(texts: Sequence[str], *, scale: float = 1.0) -> dict[str, float]:
    counter: Counter[str] = Counter()
    _add_weighted_terms(counter, texts, scale=scale)
    return {term: float(weight) for term, weight in counter.items() if float(weight) > 0.0}


def _profile_overlap_score(
    profile: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    if not profile or not target_weights:
        return 0.0

    def term_match_score(profile_term: str, target_term: str) -> float:
        profile_norm = _normalize_text(profile_term).lower()
        target_norm = _normalize_text(target_term).lower()
        if not profile_norm or not target_norm:
            return 0.0
        if profile_norm == target_norm:
            return 1.0
        shorter = min(len(profile_norm), len(target_norm))
        longer = max(len(profile_norm), len(target_norm))
        if profile_norm in target_norm or target_norm in profile_norm:
            return float(shorter / max(1, longer))
        overlap = max(
            _suffix_prefix_overlap(profile_norm, target_norm, min_overlap=4),
            _suffix_prefix_overlap(target_norm, profile_norm, min_overlap=4),
        )
        if overlap <= 0:
            return 0.0
        return float(overlap / max(1, longer))

    matched = 0.0
    total = 0.0
    max_profile_weight = max(profile.values()) if profile else 1.0
    for term, raw_weight in target_weights.items():
        weight = max(0.0, float(raw_weight))
        total += weight
        best_match = 0.0
        for profile_term, profile_weight in profile.items():
            match_strength = term_match_score(profile_term, term)
            if match_strength <= 0.0:
                continue
            normalized_profile_weight = min(1.0, float(profile_weight) / max(1e-8, max_profile_weight))
            best_match = max(best_match, match_strength * normalized_profile_weight)
        matched += weight * best_match
    if total <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, matched / total)))


def bank_semantic_relevance_score(
    bank: Any,
    plan: Mapping[str, Any] | None,
) -> float:
    if not plan:
        return 0.0
    profile = candidate_bank_term_profile(bank)
    if not profile:
        return 0.0

    gap_targets = {
        str(item.get("term", "")).strip().lower(): float(item.get("weight", 0.0))
        for item in list(plan.get("gap_terms") or [])
        if str(item.get("term", "")).strip()
    }
    query_targets = _weighted_text_profile(
        [str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip()],
        scale=1.0,
    )
    question_targets_raw = _weighted_text_profile(
        [str(item) for item in list(plan.get("follow_up_questions") or []) if str(item).strip()],
        scale=1.0,
    )
    unsupported_targets = {
        str(item).strip().lower(): 1.0
        for item in list(plan.get("unsupported_terms") or [])
        if str(item).strip()
    }
    focus_terms = set(gap_targets) | set(query_targets) | set(unsupported_targets)
    question_targets = {
        term: weight
        for term, weight in question_targets_raw.items()
        if not focus_terms or term in focus_terms
    }

    gap_score = _profile_overlap_score(profile, gap_targets)
    query_score = _profile_overlap_score(profile, query_targets)
    question_score = _profile_overlap_score(profile, question_targets)
    unsupported_score = _profile_overlap_score(profile, unsupported_targets)
    return _clamp01(
        0.45 * gap_score
        + 0.25 * query_score
        + 0.15 * question_score
        + 0.15 * unsupported_score
    )
