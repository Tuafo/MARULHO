from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from marulho.semantics.grounding_text import match_terms
from marulho.semantics.grounding_text import normalize_text as _normalize_text
from marulho.semantics.grounding_text import query_focused_clauses
from marulho.semantics.grounding_text import salient_query_terms
from marulho.semantics.grounding_text import stream_unit_profile
from marulho.semantics.grounding_text import term_match_score
from marulho.semantics.grounding_text import tokenize

_FRONTIER_CANDIDATE_MIN = 32
_FRONTIER_CANDIDATE_MULTIPLIER = 8


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def tokenize_terms(text: str) -> list[str]:
    return tokenize(text)


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


def _add_profile_weights(
    counter: Counter[str],
    profile: Mapping[str, float],
    *,
    scale: float,
    limit: int = 12,
) -> None:
    ranked = sorted(
        profile.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )[: max(1, int(limit))]
    for rank, (term, weight) in enumerate(ranked):
        if len(str(term)) < 2 and not str(term).isdigit():
            continue
        counter[str(term)] += float(scale) * max(0.0, float(weight)) / float(rank + 1)


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
        text = _normalize_text(match.get("text") or match.get("raw_window"))
        if not text:
            continue
        supported_terms = set(match_terms(list(query_terms), text))
        if not supported_terms:
            continue
        similarity = max(0.0, float(match.get("similarity", 0.0)))
        strength = similarity / math.sqrt(float(rank + 1))
        for term in supported_terms:
            support[term] = max(float(support.get(term, 0.0)), strength)
    return support


def _supports_chunked_query_expansion(query_text: str, query_terms: Sequence[str]) -> bool:
    normalized = _normalize_text(query_text)
    if not normalized or " " in normalized:
        return False
    if len(query_terms) != 1:
        return False
    term = str(query_terms[0]).strip().lower()
    return len(term) >= 8 and any(ch.isalpha() for ch in term)


def _supported_chunk_queries(
    *,
    query_text: str,
    query_terms: Sequence[str],
    memory_matches: Sequence[Mapping[str, Any]],
    limit: int,
) -> list[str]:
    if not _supports_chunked_query_expansion(query_text, query_terms):
        return []

    ranked_matches = sorted(
        memory_matches,
        key=lambda item: float(item.get("similarity", 0.0)),
        reverse=True,
    )[: max(1, int(limit) * 3)]

    queries: list[str] = []
    for match in ranked_matches:
        source_text = _normalize_text(match.get("text") or match.get("raw_window"))
        if not source_text:
            continue
        for clause in query_focused_clauses(source_text, query_terms):
            terms = [
                term
                for term in tokenize_terms(clause)
                if len(term) >= 4 or term.isdigit()
            ]
            if len(terms) < 2:
                continue
            queries.append(" ".join(terms[:4]))
            if len(_dedupe_keep_order(queries)) >= max(1, int(limit)):
                return _dedupe_keep_order(queries)[: max(1, int(limit))]
    return _dedupe_keep_order(queries)[: max(1, int(limit))]


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
    query_terms = _dedupe_keep_order(salient_query_terms(query_text_norm))
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
            *_supported_chunk_queries(
                query_text=query_text_norm,
                query_terms=query_terms,
                memory_matches=memory_matches,
                limit=max_queries,
            ),
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


def _sequence_len(value: Any) -> int:
    try:
        return max(0, int(len(value)))
    except (TypeError, ValueError):
        return 0


def _sequence_value(value: Any, index: int, default: Any = None) -> Any:
    try:
        if int(index) < 0 or int(index) >= len(value):
            return default
        return value[int(index)]
    except (TypeError, ValueError, IndexError, KeyError):
        return default


def _empty_frontier_candidate_report(
    *,
    memory_size: int,
    current_token: int,
    requested_count: int,
    fallback_reason: str,
) -> dict[str, Any]:
    return {
        "surface": "bounded_frontier_gap_candidates.v1",
        "status": "empty",
        "scope": "frontier_gap_planner_slow_path",
        "memory_size": int(memory_size),
        "current_token": int(current_token),
        "requested_count": int(requested_count),
        "candidate_window_limit": int(requested_count),
        "candidate_window_policy": "frontier_candidate_window_unavailable",
        "candidate_scope": "frontier_candidate_window_unavailable",
        "candidate_bucket_ids": [],
        "candidate_bucket_count": 0,
        "candidate_index_available_count": 0,
        "candidate_index_available_count_is_lower_bound": False,
        "candidate_index_count": 0,
        "candidate_indices": [],
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "fallback_reason": str(fallback_reason),
    }


def _frontier_candidate_report(
    *,
    memory_store: Any,
    current_token: int,
    requested_count: int,
) -> dict[str, Any]:
    if memory_store is None:
        return _empty_frontier_candidate_report(
            memory_size=0,
            current_token=current_token,
            requested_count=requested_count,
            fallback_reason="missing_memory_store",
        )

    collector = getattr(memory_store, "collect_frontier_gap_indices", None)
    if callable(collector):
        return dict(
            collector(
                current_token=int(current_token),
                max_candidates=max(0, int(requested_count)),
                scope="frontier_gap_planner_slow_path",
            )
        )

    windows = getattr(memory_store, "slow_raw_windows", None)
    memory_size = _sequence_len(windows)
    if memory_size <= 0:
        return _empty_frontier_candidate_report(
            memory_size=0,
            current_token=current_token,
            requested_count=requested_count,
            fallback_reason="empty_memory",
        )

    bounded_count = min(memory_size, max(0, int(requested_count)))
    indices = [int(index) for index in range(bounded_count)]
    return {
        "surface": "bounded_frontier_gap_candidates.v1",
        "status": "collected" if indices else "empty",
        "scope": "frontier_gap_planner_slow_path",
        "memory_size": int(memory_size),
        "current_token": int(current_token),
        "requested_count": int(requested_count),
        "candidate_window_limit": int(requested_count),
        "candidate_window_policy": "bounded_prefix_fixture_window",
        "candidate_scope": "bounded_prefix_fixture_window",
        "candidate_bucket_ids": [],
        "candidate_bucket_count": 0,
        "candidate_index_available_count": int(memory_size),
        "candidate_index_available_count_is_lower_bound": memory_size > bounded_count,
        "candidate_index_count": int(len(indices)),
        "candidate_indices": indices,
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "fallback_reason": "memory_store_missing_bounded_frontier_collector",
    }


def _frontier_scored_entries_with_report(
    *,
    memory_store: Any,
    current_token: int,
    top_entries: int,
) -> tuple[list[tuple[float, str, int]], dict[str, Any]]:
    requested = max(
        max(1, int(top_entries)),
        max(_FRONTIER_CANDIDATE_MIN, int(top_entries) * _FRONTIER_CANDIDATE_MULTIPLIER),
    )
    candidate_report = _frontier_candidate_report(
        memory_store=memory_store,
        current_token=current_token,
        requested_count=requested,
    )
    candidate_indices = [
        int(index)
        for index in list(candidate_report.get("candidate_indices") or [])
        if int(index) >= 0
    ]
    windows = getattr(memory_store, "slow_raw_windows", None) if memory_store is not None else None
    importance_values = getattr(memory_store, "slow_importance", None) if memory_store is not None else None
    capture_values = getattr(memory_store, "slow_capture_tag", None) if memory_store is not None else None
    consolidation_values = (
        getattr(memory_store, "slow_consolidation_level", None)
        if memory_store is not None
        else None
    )

    scored_entries: list[tuple[float, str, int]] = []
    text_payload_count = 0
    for idx in candidate_indices:
        raw_window = _sequence_value(windows, idx)
        text = _normalize_text(raw_window)
        if not text:
            continue
        text_payload_count += 1
        importance = float(_sequence_value(importance_values, idx, 0.0) or 0.0)
        if hasattr(memory_store, "_effective_capture_strength"):
            capture = float(memory_store._effective_capture_strength(idx, current_token))
        else:
            capture = float(_sequence_value(capture_values, idx, 0.0) or 0.0)
        consolidation = float(_sequence_value(consolidation_values, idx, 0.0) or 0.0)
        frontier_pressure = max(0.0, capture - consolidation) + 0.5 * max(0.0, 1.0 - consolidation)
        score = max(1e-6, importance) * (1.0 + frontier_pressure)
        scored_entries.append((score, text, int(idx)))

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    selected_entries = scored_entries[: max(1, int(top_entries))]
    selected_indices = [int(index) for _score, _text, index in selected_entries]
    fallback_reason = candidate_report.get("fallback_reason")
    if candidate_indices and not selected_entries:
        fallback_reason = "frontier_candidates_missing_text_payloads"
    selection_report = {
        **candidate_report,
        "surface": "bounded_frontier_gap_selection.v1",
        "status": "selected" if selected_entries else "empty",
        "score_count": int(len(scored_entries)),
        "selected_indices": selected_indices,
        "selected_count": int(len(selected_indices)),
        "frontier_window_count": int(len(selected_entries)),
        "raw_text_payload_loaded": bool(text_payload_count > 0),
        "raw_text_payload_count": int(text_payload_count),
        "raw_text_payload_policy": "selected_frontier_candidate_indices_only",
        "language_reasoning": False,
        "quality_metric": "frontier_pressure_score_over_bounded_candidates",
        "fallback_reason": fallback_reason,
    }
    return selected_entries, selection_report


def _frontier_gap_terms_from_entries(
    scored_entries: Sequence[tuple[float, str, int]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if not scored_entries:
        return []

    frontier_phrases = _merge_frontier_windows([text for _, text, _ in scored_entries])
    allowed_terms = {
        term
        for text in frontier_phrases
        for term in _frontier_terms(text)
    }
    counter: Counter[str] = Counter()
    for rank, (score, text, _index) in enumerate(scored_entries):
        for pos, term in enumerate(_frontier_terms(text)[:8]):
            if allowed_terms and term not in allowed_terms:
                continue
            counter[term] += float(score) / (math.sqrt(float(rank + 1)) * float(pos + 1))
    return _top_term_payload(counter, limit)


def frontier_gap_terms(
    *,
    memory_store: Any,
    current_token: int,
    limit: int = 8,
    top_entries: int = 24,
) -> list[dict[str, Any]]:
    scored_entries, _report = _frontier_scored_entries_with_report(
        memory_store=memory_store,
        current_token=current_token,
        top_entries=top_entries,
    )
    return _frontier_gap_terms_from_entries(scored_entries, limit=limit)


def frontier_gap_plan(
    *,
    memory_store: Any,
    current_token: int,
    max_terms: int = 8,
    max_queries: int = 4,
    max_questions: int = 4,
    top_entries: int = 24,
) -> dict[str, Any]:
    scored_entries, frontier_selection_report = _frontier_scored_entries_with_report(
        memory_store=memory_store,
        current_token=current_token,
        top_entries=top_entries,
    )
    gap_terms = _frontier_gap_terms_from_entries(scored_entries, limit=max_terms)
    if not scored_entries:
        return {
            "planner_mode": "frontier_gap_planner",
            "gap_terms": [],
            "unsupported_terms": [],
            "retrieval_queries": [],
            "follow_up_questions": [],
            "frontier_windows": [],
            "frontier_phrases": [],
            "frontier_selection_report": frontier_selection_report,
        }

    frontier_windows = [text for _, text, _index in scored_entries[: max(1, int(max_questions) * 2)]]
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
        "frontier_windows": [text for _, text, _index in scored_entries[: max(1, int(max_questions))]],
        "frontier_phrases": frontier_phrases[: max(1, int(max_questions))],
        "frontier_selection_report": frontier_selection_report,
    }


def _add_weighted_terms(counter: Counter[str], texts: Sequence[str], *, scale: float) -> None:
    for text_rank, text in enumerate(texts):
        profile = stream_unit_profile(text)
        ranked_terms = sorted(
            profile.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:24]
        for term_rank, (term, weight) in enumerate(ranked_terms):
            counter[str(term)] += (
                float(scale)
                * max(0.0, float(weight))
                / (math.sqrt(float(text_rank + 1)) * float(term_rank + 1))
            )


def _candidate_bank_field(bank: Any, key: str, default: Any) -> Any:
    if isinstance(bank, Mapping):
        return bank.get(key, default)
    return getattr(bank, key, default)


def candidate_bank_term_profile(
    bank: Any,
    *,
    max_terms: int = 64,
    max_windows: int = 96,
) -> dict[str, float]:
    counter: Counter[str] = Counter()
    name = _normalize_text(_candidate_bank_field(bank, "name", ""))
    source = _normalize_text(_candidate_bank_field(bank, "source", ""))
    metadata = _candidate_bank_field(bank, "metadata", None)
    _add_profile_weights(counter, stream_unit_profile(name), scale=2.0)

    if source:
        parsed = urlparse(source)
        source_parts = [
            parsed.netloc.replace(".", " "),
            parsed.path.replace("/", " "),
            source if not parsed.scheme else "",
        ]
        for part in source_parts:
            _add_profile_weights(counter, stream_unit_profile(part), scale=1.0)

    if isinstance(metadata, Mapping):
        catalog_title = _normalize_text(metadata.get("catalog_title", ""))
        catalog_summary = _normalize_text(metadata.get("catalog_summary", ""))
        if catalog_title:
            _add_profile_weights(counter, stream_unit_profile(catalog_title), scale=1.5)
        if catalog_summary:
            _add_weighted_terms(counter, [catalog_summary], scale=1.25)
        catalog_terms = [
            _normalize_text(item)
            for item in list(metadata.get("catalog_terms") or [])
            if _normalize_text(item)
        ]
        if catalog_terms:
            _add_weighted_terms(counter, catalog_terms[:16], scale=1.0)

    probe_windows = list(_candidate_bank_field(bank, "probe_raw_windows", []) or [])
    _add_weighted_terms(counter, probe_windows[: max(1, int(max_windows // 2))], scale=1.25)

    windows = list(_candidate_bank_field(bank, "train_raw_windows", []) or [])
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

    matched = 0.0
    total = 0.0
    profile_terms = [
        str(term).strip().lower()
        for term, weight in profile.items()
        if str(term).strip() and float(weight) > 0.0
    ]
    for term, raw_weight in target_weights.items():
        weight = max(0.0, float(raw_weight))
        total += weight
        matched += weight * term_match_score(str(term), profile_terms)
    if total <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, matched / total)))


def _candidate_bank_metadata_texts(bank: Any) -> list[str]:
    metadata = _candidate_bank_field(bank, "metadata", None)
    if not isinstance(metadata, Mapping):
        return []

    texts: list[str] = []
    for key in ("catalog_title", "catalog_summary", "catalog_content_preview"):
        value = _normalize_text(metadata.get(key, ""))
        if value:
            texts.append(value)
    texts.extend(
        _normalize_text(item)
        for item in list(metadata.get("catalog_terms") or [])
        if _normalize_text(item)
    )
    return texts


def _metadata_overlap_score(
    bank: Any,
    target_weights: Mapping[str, float],
) -> float:
    if not target_weights:
        return 0.0
    metadata_text = " ".join(_candidate_bank_metadata_texts(bank)).strip()
    if not metadata_text:
        return 0.0
    matched_terms = set(match_terms(list(target_weights), metadata_text))
    if not matched_terms:
        return 0.0

    total = sum(max(0.0, float(weight)) for weight in target_weights.values())
    if total <= 0.0:
        return 0.0
    matched = sum(
        max(0.0, float(weight))
        for term, weight in target_weights.items()
        if str(term).strip().lower() in matched_terms
    )
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

    gap_score = max(
        _profile_overlap_score(profile, gap_targets),
        _metadata_overlap_score(bank, gap_targets),
    )
    query_score = max(
        _profile_overlap_score(profile, query_targets),
        _metadata_overlap_score(bank, query_targets),
    )
    question_score = max(
        _profile_overlap_score(profile, question_targets),
        _metadata_overlap_score(bank, question_targets),
    )
    unsupported_score = max(
        _profile_overlap_score(profile, unsupported_targets),
        _metadata_overlap_score(bank, unsupported_targets),
    )
    return _clamp01(
        0.45 * gap_score
        + 0.25 * query_score
        + 0.15 * question_score
        + 0.15 * unsupported_score
    )
