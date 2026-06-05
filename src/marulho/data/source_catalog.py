from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from html import unescape
import json
import re
from threading import Lock
from time import monotonic
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from marulho.data.corpus_loader import extract_web_text
from marulho.gap_planner import (
    bank_semantic_relevance_score,
    plan_query_gaps,
    salient_query_terms,
    tokenize_terms,
)
from marulho.semantics.grounding_text import match_terms, split_sentences, stream_unit_profile


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_REMOTE_SEARCH_CACHE_TTL_SECONDS = 300.0
_REMOTE_SEARCH_FAILURE_CACHE_TTL_SECONDS = 60.0
_REMOTE_CONTENT_CACHE_TTL_SECONDS = 300.0
_REMOTE_CONTENT_FETCH_MAX_CHARS = 6000
_REMOTE_CONTENT_PREVIEW_MAX_CHARS = 1200
_REMOTE_SEARCH_CACHE_LOCK = Lock()
_TOPIC_TERM_SEPARATORS_RE = re.compile(r"[;,|]")
_TOPIC_TERM_NORMALIZE_RE = re.compile(r"[_./:-]+")
_FOLLOW_UP_QUERY_NOISE_TERMS = {
    "concept",
    "concepts",
    "connect",
    "connects",
    "current",
    "distinguish",
    "distinguishes",
    "drift",
    "evidence",
    "explain",
    "explains",
    "frontier",
    "grounded",
    "link",
    "links",
    "memory",
    "missing",
    "nearby",
    "reduce",
    "stable",
    "stabilize",
    "still",
    "would",
}


@dataclass(frozen=True)
class _RemoteSearchCacheEntry:
    expires_at: float
    entries: list[dict[str, Any]] | None = None
    failure: str | None = None


@dataclass(frozen=True)
class _RemoteContentCacheEntry:
    expires_at: float
    text: str


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    source: str
    source_type: str
    hf_config: str | None
    text_field: str
    semantic_relevance: float
    prior_weight: float
    combined_score: float
    metadata: dict[str, Any]


_REMOTE_SEARCH_CACHE: dict[tuple[Any, str, str, int], _RemoteSearchCacheEntry] = {}
_REMOTE_CONTENT_CACHE: dict[str, _RemoteContentCacheEntry] = {}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _slugify(value: str, *, default: str = "catalog_source") -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value).lower()).strip("_")
    return slug or default


def _strip_html(value: str) -> str:
    return _normalize_text(unescape(_HTML_TAG_RE.sub(" ", value)))


def _focus_plan(
    spec: Mapping[str, Any],
    semantic_plan: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if semantic_plan and (
        semantic_plan.get("gap_terms")
        or semantic_plan.get("retrieval_queries")
        or semantic_plan.get("unsupported_terms")
        or semantic_plan.get("follow_up_questions")
        or semantic_plan.get("weak_concepts")
    ):
        return semantic_plan

    focus_text = _normalize_text(spec.get("catalog_focus_text", ""))
    if not focus_text:
        focus_terms = spec.get("catalog_focus_terms")
        if isinstance(focus_terms, Sequence) and not isinstance(focus_terms, (str, bytes)):
            focus_text = _normalize_text(" ".join(str(term) for term in focus_terms if str(term).strip()))
    if not focus_text:
        return None

    return plan_query_gaps(
        query_text=focus_text,
        query_summary={"memory_matches": []},
        concept_summary={"concepts": []},
        max_terms=max(4, int(spec.get("catalog_limit", 8))),
        max_questions=4,
        max_queries=4,
    )


def _entry_windows(entry: Mapping[str, Any]) -> list[str]:
    windows: list[str] = []
    content_preview = _normalize_text(entry.get("content_preview"))
    content_preview_preferred = bool(entry.get("content_preview_preferred"))
    if content_preview_preferred and content_preview and stream_unit_profile(content_preview):
        windows.append(content_preview)

    for key in ("summary", "title", "description"):
        value = _normalize_text(entry.get(key))
        if value and stream_unit_profile(value):
            windows.append(value)

    if not content_preview_preferred and content_preview and stream_unit_profile(content_preview):
        windows.append(content_preview)

    query_text = _normalize_text(entry.get("query_text"))
    if query_text and not windows and stream_unit_profile(query_text):
        windows.append(query_text)

    tags = entry.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        tag_text = _normalize_text(" ".join(str(tag) for tag in tags if str(tag).strip()))
        if tag_text and stream_unit_profile(tag_text):
            windows.append(tag_text)

    focus_terms = entry.get("terms")
    if isinstance(focus_terms, Sequence) and not isinstance(focus_terms, (str, bytes)):
        term_text = _normalize_text(" ".join(str(term) for term in focus_terms if str(term).strip()))
        if term_text and stream_unit_profile(term_text):
            windows.append(term_text)
    return windows


def _term_profile(texts: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        ranked = sorted(
            stream_unit_profile(text).items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:12]
        for term, weight in ranked:
            if float(weight) > 0.0:
                terms.add(str(term))
    return terms


def _provider_priority_map(spec: Mapping[str, Any]) -> dict[str, float]:
    raw_map = spec.get("catalog_provider_priority_map")
    if not isinstance(raw_map, Mapping):
        return {}
    priorities: dict[str, float] = {}
    for raw_provider, raw_value in raw_map.items():
        provider = _normalize_text(raw_provider).lower()
        if not provider:
            continue
        try:
            priority = float(raw_value)
        except (TypeError, ValueError):
            continue
        priorities[provider] = float(priority)
    return priorities


def _provider_topic_term_map(spec: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_map = spec.get("catalog_provider_topic_terms")
    if not isinstance(raw_map, Mapping):
        return {}
    topic_terms: dict[str, list[str]] = {}
    for raw_provider, raw_terms in raw_map.items():
        provider = _normalize_text(raw_provider).lower()
        if not provider:
            continue
        topic_terms[provider] = _normalized_text_list(raw_terms)
    return topic_terms


def _provider_query_family_map(spec: Mapping[str, Any]) -> dict[str, list[str]]:
    raw_map = spec.get("catalog_provider_query_families")
    if not isinstance(raw_map, Mapping):
        return {}
    query_families: dict[str, list[str]] = {}
    for raw_provider, raw_queries in raw_map.items():
        provider = _normalize_text(raw_provider).lower()
        if not provider:
            continue
        query_families[provider] = _normalized_text_list(raw_queries)
    return query_families


def _diversity_penalty(profile: set[str], selected_profiles: Sequence[set[str]]) -> float:
    if not profile or not selected_profiles:
        return 0.0
    max_overlap = 0.0
    for selected in selected_profiles:
        union = profile | selected
        if not union:
            continue
        overlap = float(len(profile & selected)) / float(len(union))
        max_overlap = max(max_overlap, overlap)
    return max_overlap


def _plan_queries(plan: Mapping[str, Any] | None, spec: Mapping[str, Any], *, limit: int) -> list[str]:
    queries: list[str] = []
    if plan:
        queries.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
        queries.extend(_weak_concept_search_queries(plan))
        queries.extend(
            _follow_up_search_query(str(item))
            for item in list(plan.get("follow_up_questions") or [])
            if str(item).strip()
        )
        gap_terms = [str(item.get("term", "")).strip() for item in list(plan.get("gap_terms") or []) if str(item.get("term", "")).strip()]
        if gap_terms:
            queries.append(" ".join(gap_terms[:3]))
        unsupported = [str(item).strip() for item in list(plan.get("unsupported_terms") or []) if str(item).strip()]
        if unsupported:
            queries.append(" ".join(unsupported[:3]))
    focus_text = _normalize_text(spec.get("catalog_focus_text", ""))
    if focus_text:
        queries.append(focus_text)
    return _dedupe_keep_order(queries)[: max(1, int(limit))]


def _provider_queries(
    provider: str,
    base_queries: Sequence[str],
    spec: Mapping[str, Any],
    *,
    limit: int,
) -> list[str]:
    queries = _dedupe_keep_order(base_queries)
    limit = max(1, int(limit))
    provider_topic_terms = _provider_topic_term_map(spec).get(_normalize_text(provider).lower(), [])
    if not queries:
        return queries[:limit]
    seed_query = queries[0]
    prioritized_queries = list(queries)
    if provider_topic_terms:
        seen_terms = {term.lower() for term in tokenize_terms(seed_query)}
        expansion_terms: list[str] = []
        for raw_phrase in provider_topic_terms:
            phrase = _normalize_text(raw_phrase)
            if not phrase:
                continue
            phrase_terms = tokenize_terms(phrase)
            if phrase_terms and all(term.lower() in seen_terms for term in phrase_terms):
                continue
            expansion_terms.append(phrase)
            for term in phrase_terms:
                seen_terms.add(term.lower())
            if len(expansion_terms) >= 2:
                break
        if expansion_terms:
            expanded_query = _normalize_text(f"{seed_query} {' '.join(expansion_terms)}")
            prioritized_queries = [seed_query, expanded_query]
            prioritized_queries.extend(queries[1:])
    query_families = _provider_query_family_map(spec).get(_normalize_text(provider).lower(), [])
    for raw_query in query_families:
        query = _normalize_text(raw_query)
        if not query:
            continue
        query_terms = tokenize_terms(query)
        if any(
            query == existing
            or (
                query_terms
                and set(query_terms).issubset({term.lower() for term in tokenize_terms(existing)})
            )
            for existing in prioritized_queries
        ):
            continue
        prioritized_queries.append(query)
        if len(prioritized_queries) >= limit:
            break
    return _dedupe_keep_order(prioritized_queries)[:limit]


def _weak_concept_search_queries(plan: Mapping[str, Any]) -> list[str]:
    queries: list[str] = []
    for raw_concept in list(plan.get("weak_concepts") or []):
        if not isinstance(raw_concept, Mapping):
            continue
        top_terms = [
            _normalize_text(term).lower()
            for term in list(raw_concept.get("top_terms") or [])
            if _normalize_text(term)
        ]
        if top_terms:
            queries.append(" ".join(top_terms[:3]))
            continue
        label = _normalize_text(raw_concept.get("label"))
        if label:
            queries.append(label)
    return _dedupe_keep_order(queries)


def _follow_up_search_query(question: str) -> str:
    normalized_question = _normalize_text(question)
    if not normalized_question:
        return ""
    focused_terms = [
        term
        for term in salient_query_terms(normalized_question)
        if term not in _FOLLOW_UP_QUERY_NOISE_TERMS
    ]
    if focused_terms:
        return _normalize_text(" ".join(focused_terms[:4]))
    fallback_terms = salient_query_terms(normalized_question)
    if fallback_terms:
        return _normalize_text(" ".join(fallback_terms[:4]))
    return normalized_question


def _catalog_selection_limit(
    spec: Mapping[str, Any],
    *,
    metadata_prefilter: bool,
    default_limit: int,
) -> int:
    final_limit = max(1, int(spec.get("catalog_limit", default_limit)))
    if not metadata_prefilter:
        return final_limit
    probe_pool_limit = int(spec.get("catalog_probe_pool_limit", final_limit))
    return max(final_limit, probe_pool_limit)


def _ordered_live_remote_providers(spec: Mapping[str, Any]) -> list[str]:
    providers = [
        str(item).strip()
        for item in list(spec.get("catalog_providers") or ["wikipedia"])
        if str(item).strip()
    ]
    if not providers:
        return ["wikipedia"]
    priorities = _provider_priority_map(spec)
    if priorities:
        indexed = list(enumerate(providers))
        indexed.sort(
            key=lambda item: (
                -float(priorities.get(str(item[1]).strip().lower(), 0.0)),
                int(item[0]),
            )
        )
        providers = [provider for _index, provider in indexed]
    provider_limit = max(1, int(spec.get("catalog_provider_limit", len(providers))))
    return providers[: min(len(providers), provider_limit)]


def _dedupe_keep_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = _normalize_text(value)
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
    return ordered


def _normalized_text_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    return _dedupe_keep_order(
        [str(value) for value in values if _normalize_text(value)]
    )


def _looks_like_web_source(source: str, source_type: str) -> bool:
    normalized_source = _normalize_text(source)
    normalized_type = _normalize_text(source_type).lower()
    if normalized_type == "web":
        return True
    parsed = urlparse(normalized_source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _trim_text(text: str, *, max_chars: int) -> str:
    normalized = _normalize_text(text)
    limit = max(1, int(max_chars))
    if len(normalized) <= limit:
        return normalized
    trimmed = normalized[:limit]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed.strip() or normalized[:limit].strip()


def _normalized_topic_phrase(value: Any) -> str:
    return _normalize_text(_TOPIC_TERM_NORMALIZE_RE.sub(" ", str(value)))


def _topic_phrases(value: Any) -> list[str]:
    text = _normalize_text(value)
    if not text:
        return []
    return _dedupe_keep_order(
        _normalized_topic_phrase(part)
        for part in _TOPIC_TERM_SEPARATORS_RE.split(text)
        if _normalized_topic_phrase(part)
    )


def _http_get_json(url: str, *, timeout_seconds: float) -> Any:
    request = Request(
        url,
        headers={
            "User-Agent": "MARULHO/1.0 source discovery",
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        payload = response.read().decode("utf-8", errors="ignore")
    return json.loads(payload)


def _http_get_text(url: str, *, timeout_seconds: float) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "MARULHO/1.0 source discovery",
            "Accept": "application/atom+xml,text/xml;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read().decode("utf-8", errors="ignore")


def _http_get_content_text(
    url: str,
    *,
    timeout_seconds: float,
) -> tuple[str, str | None]:
    request = Request(
        url,
        headers={
            "User-Agent": "MARULHO/1.0 source discovery",
            "Accept": "text/html,text/plain,application/json,text/x-wiki;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        payload = response.read()
        content_type = response.headers.get("Content-Type")
        encoding = response.headers.get_content_charset() or "utf-8"
    try:
        decoded = payload.decode(encoding, errors="ignore")
    except LookupError:
        decoded = payload.decode("utf-8", errors="ignore")
    return decoded, content_type


def _remote_search_cache_key(
    provider_fn: Any,
    provider: str,
    query: str,
    *,
    result_limit: int,
) -> tuple[Any, str, str, int]:
    return (
        provider_fn,
        _normalize_text(provider).lower(),
        _normalize_text(query).lower(),
        max(1, int(result_limit)),
    )


def _prune_remote_search_cache_locked(current_time: float) -> None:
    expired_keys = [
        key
        for key, entry in _REMOTE_SEARCH_CACHE.items()
        if float(current_time) >= float(entry.expires_at)
    ]
    for key in expired_keys:
        _REMOTE_SEARCH_CACHE.pop(key, None)
    expired_content_keys = [
        key
        for key, entry in _REMOTE_CONTENT_CACHE.items()
        if float(current_time) >= float(entry.expires_at)
    ]
    for key in expired_content_keys:
        _REMOTE_CONTENT_CACHE.pop(key, None)


def _search_remote_provider_cached(
    provider: str,
    query: str,
    *,
    result_limit: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    provider_fn = _search_remote_provider
    cache_key = _remote_search_cache_key(
        provider_fn,
        provider,
        query,
        result_limit=result_limit,
    )
    current_time = monotonic()
    with _REMOTE_SEARCH_CACHE_LOCK:
        _prune_remote_search_cache_locked(current_time)
        cached = _REMOTE_SEARCH_CACHE.get(cache_key)
    if cached is not None:
        if cached.failure is not None:
            raise RuntimeError(cached.failure)
        return deepcopy(list(cached.entries or []))

    try:
        entries = list(
            provider_fn(
                provider,
                query,
                result_limit=result_limit,
                timeout_seconds=timeout_seconds,
            )
            or []
        )
    except Exception as exc:
        failure_message = str(exc) or exc.__class__.__name__
        with _REMOTE_SEARCH_CACHE_LOCK:
            _REMOTE_SEARCH_CACHE[cache_key] = _RemoteSearchCacheEntry(
                expires_at=float(current_time + _REMOTE_SEARCH_FAILURE_CACHE_TTL_SECONDS),
                failure=failure_message,
            )
        raise

    cached_entries = deepcopy(entries)
    with _REMOTE_SEARCH_CACHE_LOCK:
        _REMOTE_SEARCH_CACHE[cache_key] = _RemoteSearchCacheEntry(
            expires_at=float(current_time + _REMOTE_SEARCH_CACHE_TTL_SECONDS),
            entries=cached_entries,
        )
    return deepcopy(cached_entries)


def _content_probe_focus_terms(
    plan: Mapping[str, Any] | None,
    spec: Mapping[str, Any],
    *,
    entry: Mapping[str, Any] | None = None,
) -> list[str]:
    ordered_terms: list[str] = []
    seen: set[str] = set()

    def add_text(value: Any) -> None:
        for term in salient_query_terms(_normalize_text(value)):
            lowered = term.lower()
            if not lowered or lowered in seen:
                continue
            seen.add(lowered)
            ordered_terms.append(lowered)

    if entry is not None:
        add_text(entry.get("query_text"))
    for query in _plan_queries(
        plan,
        spec,
        limit=max(1, int(spec.get("catalog_queries_per_provider", 2))),
    ):
        add_text(query)
    if plan:
        for raw_item in list(plan.get("gap_terms") or []):
            if isinstance(raw_item, Mapping):
                add_text(raw_item.get("term"))
        for raw_item in list(plan.get("unsupported_terms") or []):
            add_text(raw_item)
    return ordered_terms[:8]


def _focused_content_preview(
    text: str,
    focus_terms: Sequence[str],
    *,
    max_chars: int,
) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    if not focus_terms:
        return _trim_text(normalized, max_chars=max_chars)

    ranked_sentences: list[tuple[int, int, int, str, list[str]]] = []
    seen_candidates: set[str] = set()
    for sentence in split_sentences(normalized):
        clauses = [
            _normalize_text(fragment)
            for fragment in re.split(r"[,;:()]", sentence)
        ]
        clauses = [fragment for fragment in clauses if fragment]
        if not clauses:
            clauses = [_normalize_text(sentence)]
        candidate_windows = [_normalize_text(sentence)]
        for index, clause in enumerate(clauses):
            candidate_windows.append(clause)
            if index + 1 < len(clauses):
                candidate_windows.append(f"{clause}, {clauses[index + 1]}")
            if index + 2 < len(clauses):
                candidate_windows.append(
                    f"{clause}, {clauses[index + 1]}, {clauses[index + 2]}"
                )
        for candidate in candidate_windows:
            normalized_candidate = _normalize_text(candidate)
            lowered = normalized_candidate.lower()
            if not normalized_candidate or lowered in seen_candidates:
                continue
            seen_candidates.add(lowered)
            matches = match_terms(list(focus_terms), normalized_candidate)
            if not matches:
                continue
            ranked_sentences.append(
                (
                    len(matches),
                    -max(1, len(tokenize_terms(normalized_candidate))),
                    -len(normalized_candidate),
                    normalized_candidate,
                    matches,
                )
            )
    ranked_sentences.sort(reverse=True)
    if not ranked_sentences:
        return _trim_text(normalized, max_chars=max_chars)

    selected_sentences: list[str] = []
    covered_terms: set[str] = set()
    for _match_count, _word_rank, _length_rank, sentence, matches in ranked_sentences:
        new_terms = [term for term in matches if term not in covered_terms]
        if selected_sentences and not new_terms:
            continue
        selected_sentences.append(sentence)
        covered_terms.update(matches)
        if len(selected_sentences) >= 3 or covered_terms.issuperset(focus_terms):
            break
    return _trim_text(" ".join(selected_sentences), max_chars=max_chars)


def _fetch_remote_content_text_cached(
    source: str,
    *,
    timeout_seconds: float,
) -> str:
    cache_key = _normalize_text(source).lower()
    if not cache_key:
        return ""

    current_time = monotonic()
    with _REMOTE_SEARCH_CACHE_LOCK:
        _prune_remote_search_cache_locked(current_time)
        cached = _REMOTE_CONTENT_CACHE.get(cache_key)
    if cached is not None:
        return str(cached.text)

    ttl = _REMOTE_CONTENT_CACHE_TTL_SECONDS
    text = ""
    try:
        payload, content_type = _http_get_content_text(
            source,
            timeout_seconds=timeout_seconds,
        )
        text = extract_web_text(
            payload,
            content_type=content_type,
            max_chars=_REMOTE_CONTENT_FETCH_MAX_CHARS,
        )
    except Exception:
        ttl = _REMOTE_SEARCH_FAILURE_CACHE_TTL_SECONDS
        text = ""

    with _REMOTE_SEARCH_CACHE_LOCK:
        _REMOTE_CONTENT_CACHE[cache_key] = _RemoteContentCacheEntry(
            expires_at=float(current_time + ttl),
            text=str(text),
        )
    return str(text)


def _prefer_content_preview(
    preview: str,
    summary: str,
    *,
    focus_terms: Sequence[str],
) -> bool:
    normalized_preview = _normalize_text(preview)
    normalized_summary = _normalize_text(summary)
    if not normalized_preview:
        return False
    if not normalized_summary:
        return True
    if not focus_terms:
        return len(normalized_preview) <= len(normalized_summary)

    preview_match_count = len(match_terms(list(focus_terms), normalized_preview))
    summary_match_count = len(match_terms(list(focus_terms), normalized_summary))
    if preview_match_count > summary_match_count:
        return True
    if preview_match_count == summary_match_count and preview_match_count > 0:
        return len(normalized_preview) <= len(normalized_summary)
    return False


def _content_probe_entries(
    entries_raw: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    spec: Mapping[str, Any],
    semantic_plan: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    probe_candidates = _rank_catalog_entries(
        mode=mode,
        entries_raw=entries_raw,
        spec=spec,
        semantic_plan=semantic_plan,
        metadata_prefilter=True,
    )
    probe_sources = {
        _normalize_text(item.get("source"))
        for item in probe_candidates
        if _looks_like_web_source(
            _normalize_text(item.get("source")),
            _normalize_text(item.get("source_type", "auto")),
        )
    }
    if not probe_sources:
        return [dict(item) for item in entries_raw]

    timeout_seconds = float(spec.get("catalog_provider_timeout_seconds", 15.0))
    enriched_entries: list[dict[str, Any]] = []
    for raw_entry in entries_raw:
        enriched = dict(raw_entry)
        source = _normalize_text(enriched.get("source"))
        source_type = _normalize_text(enriched.get("source_type", "auto"))
        if source in probe_sources and _looks_like_web_source(source, source_type):
            focus_terms = _content_probe_focus_terms(
                semantic_plan,
                spec,
                entry=enriched,
            )
            content_text = _fetch_remote_content_text_cached(
                source,
                timeout_seconds=timeout_seconds,
            )
            preview = _focused_content_preview(
                content_text,
                focus_terms,
                max_chars=_REMOTE_CONTENT_PREVIEW_MAX_CHARS,
            )
            if preview:
                enriched["content_preview"] = preview
                enriched["content_preview_preferred"] = _prefer_content_preview(
                    preview,
                    _normalize_text(enriched.get("summary")),
                    focus_terms=focus_terms,
                )
        enriched_entries.append(enriched)
    return enriched_entries


def _wikipedia_entries(query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, Any]]:
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max(1, int(result_limit)),
            "format": "json",
            "utf8": 1,
        }
    )
    payload = _http_get_json(url, timeout_seconds=timeout_seconds)
    results = list(((payload or {}).get("query") or {}).get("search") or [])
    page_ids: list[int] = []
    for item in results:
        try:
            page_id = int(item.get("pageid"))
        except (TypeError, ValueError):
            continue
        if page_id > 0:
            page_ids.append(page_id)
    extract_map = _wikipedia_extract_map(page_ids, timeout_seconds=timeout_seconds) if page_ids else {}
    entries: list[dict[str, Any]] = []
    for rank, item in enumerate(results):
        title = _normalize_text(item.get("title"))
        if not title:
            continue
        try:
            page_id = int(item.get("pageid"))
        except (TypeError, ValueError):
            page_id = -1
        page_path = quote(title.replace(" ", "_"), safe="_")
        snippet = _strip_html(str(item.get("snippet") or title))
        extract = _normalize_text(extract_map.get(page_id, ""))
        summary = extract or snippet or title
        description = snippet if snippet and snippet != summary else ""
        entries.append(
            {
                "name": _slugify(title, default=f"wikipedia_{rank + 1}"),
                "title": title,
                "summary": summary,
                "description": description,
                "source": f"https://en.wikipedia.org/wiki/{page_path}",
                "source_type": "web",
                "text_field": "text",
                "query_text": query,
                "tags": ["wikipedia"],
                "catalog_priority": float(1.0 / float(rank + 1)),
                "provider": "wikipedia",
            }
        )
    return entries


def _wikipedia_extract_map(page_ids: Sequence[int], *, timeout_seconds: float) -> dict[int, str]:
    normalized_ids = [int(page_id) for page_id in page_ids if int(page_id) > 0]
    if not normalized_ids:
        return {}
    url = "https://en.wikipedia.org/w/api.php?" + urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "pageids": "|".join(str(page_id) for page_id in _dedupe_keep_order(normalized_ids)),
            "explaintext": 1,
            "exintro": 1,
            "exchars": 1200,
            "format": "json",
            "utf8": 1,
        }
    )
    payload = _http_get_json(url, timeout_seconds=timeout_seconds)
    pages = ((payload or {}).get("query") or {}).get("pages") or {}
    extract_map: dict[int, str] = {}
    if not isinstance(pages, Mapping):
        return extract_map
    for page in pages.values():
        if not isinstance(page, Mapping):
            continue
        try:
            page_id = int(page.get("pageid"))
        except (TypeError, ValueError):
            continue
        extract = _normalize_text(page.get("extract"))
        if page_id > 0 and extract:
            extract_map[page_id] = extract
    return extract_map


def _arxiv_topic_terms(node: ET.Element) -> list[str]:
    terms: list[str] = []
    for category in node.findall("atom:category", _ATOM_NS):
        terms.extend(_topic_phrases(category.get("term", "")))
    primary_category = node.find("arxiv:primary_category", _ATOM_NS)
    if primary_category is not None:
        terms.extend(_topic_phrases(primary_category.get("term", "")))
    comment = _normalize_text(node.findtext("arxiv:comment", default="", namespaces=_ATOM_NS))
    if "keywords:" in comment.lower():
        keyword_text = comment.split(":", 1)[1]
        terms.extend(_topic_phrases(keyword_text))
    return _dedupe_keep_order(terms)[:12]


def _arxiv_entries(query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, Any]]:
    url = "https://export.arxiv.org/api/query?" + urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, int(result_limit)),
        }
    )
    payload = _http_get_text(url, timeout_seconds=timeout_seconds)
    root = ET.fromstring(payload)
    entries: list[dict[str, Any]] = []
    for rank, node in enumerate(root.findall("atom:entry", _ATOM_NS)):
        title = _normalize_text(node.findtext("atom:title", default="", namespaces=_ATOM_NS))
        summary = _normalize_text(node.findtext("atom:summary", default="", namespaces=_ATOM_NS))
        source = _normalize_text(node.findtext("atom:id", default="", namespaces=_ATOM_NS)).replace("http://", "https://")
        if not title or not source:
            continue
        entries.append(
            {
                "name": _slugify(title, default=f"arxiv_{rank + 1}"),
                "title": title,
                "summary": summary,
                "terms": _arxiv_topic_terms(node),
                "source": source,
                "source_type": "web",
                "text_field": "text",
                "query_text": query,
                "tags": ["arxiv"],
                "catalog_priority": float(1.0 / float(rank + 1)),
                "provider": "arxiv",
            }
        )
    return entries


def _openalex_abstract_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    ordered_tokens: dict[int, str] = {}
    for raw_token, raw_positions in value.items():
        token = _normalize_text(raw_token)
        if not token:
            continue
        if not isinstance(raw_positions, Sequence) or isinstance(raw_positions, (str, bytes)):
            continue
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if position >= 0 and position not in ordered_tokens:
                ordered_tokens[position] = token
    if not ordered_tokens:
        return ""
    return _normalize_text(" ".join(token for _position, token in sorted(ordered_tokens.items())))


def _openalex_topic_terms(item: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []

    def _add_topic_mapping(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        display_name = _normalize_text(value.get("display_name"))
        if display_name:
            terms.append(display_name)
        for key in ("subfield", "field", "domain"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                nested_name = _normalize_text(nested.get("display_name"))
                if nested_name:
                    terms.append(nested_name)

    _add_topic_mapping(item.get("primary_topic"))
    for value in list(item.get("topics") or [])[:4]:
        _add_topic_mapping(value)
    for value in list(item.get("keywords") or [])[:8]:
        if isinstance(value, Mapping):
            display_name = _normalize_text(value.get("display_name"))
            if display_name:
                terms.append(display_name)
    for value in list(item.get("concepts") or [])[:8]:
        if isinstance(value, Mapping):
            display_name = _normalize_text(value.get("display_name"))
            if display_name:
                terms.append(display_name)
    return _dedupe_keep_order(terms)[:16]


def _openalex_source_url(item: Mapping[str, Any]) -> str:
    open_access = item.get("open_access")
    if isinstance(open_access, Mapping):
        oa_url = _normalize_text(open_access.get("oa_url"))
        if oa_url:
            return oa_url

    primary_location = item.get("primary_location")
    if isinstance(primary_location, Mapping):
        landing_page_url = _normalize_text(primary_location.get("landing_page_url"))
        if landing_page_url and "doi.org/" not in landing_page_url.lower():
            return landing_page_url

    source = _normalize_text(item.get("id"))
    if source.startswith("https://openalex.org/"):
        return source.replace("https://openalex.org/", "https://api.openalex.org/works/")
    return source


def _openalex_entries(query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, Any]]:
    url = "https://api.openalex.org/works?" + urlencode(
        {
            "search": query,
            "per-page": max(1, int(result_limit)),
        }
    )
    payload = _http_get_json(url, timeout_seconds=timeout_seconds)
    results = list((payload or {}).get("results") or [])
    entries: list[dict[str, Any]] = []
    for rank, item in enumerate(results):
        title = _normalize_text(item.get("display_name") or item.get("title"))
        source = _openalex_source_url(item)
        if not title or not source:
            continue
        summary = _openalex_abstract_text(item.get("abstract_inverted_index")) or title
        entries.append(
            {
                "name": _slugify(title, default=f"openalex_{rank + 1}"),
                "title": title,
                "summary": summary,
                "terms": _openalex_topic_terms(item),
                "source": source,
                "source_type": "web",
                "text_field": "text",
                "query_text": query,
                "tags": ["openalex"],
                "catalog_priority": float(1.0 / float(rank + 1)),
                "provider": "openalex",
            }
        )
    return entries


def _search_remote_provider(
    provider: str,
    query: str,
    *,
    result_limit: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    normalized_provider = _normalize_text(provider).lower()
    if normalized_provider == "wikipedia":
        return _wikipedia_entries(query, result_limit=result_limit, timeout_seconds=timeout_seconds)
    if normalized_provider == "arxiv":
        return _arxiv_entries(query, result_limit=result_limit, timeout_seconds=timeout_seconds)
    if normalized_provider == "openalex":
        return _openalex_entries(query, result_limit=result_limit, timeout_seconds=timeout_seconds)
    raise ValueError(f"Unsupported remote discovery provider: {provider}")


def _rank_catalog_entries(
    *,
    mode: str,
    entries_raw: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    semantic_plan: Mapping[str, Any] | None,
    metadata_prefilter: bool = False,
) -> list[dict[str, Any]]:
    if not entries_raw:
        raise ValueError(f"{mode} catalog requires non-empty entries")

    limit = _catalog_selection_limit(
        spec,
        metadata_prefilter=metadata_prefilter,
        default_limit=len(entries_raw),
    )
    semantic_weight = float(spec.get("catalog_semantic_weight", 1.0))
    prior_weight_scale = float(spec.get("catalog_prior_weight", 1.0))
    provider_priority_weight = float(spec.get("catalog_provider_priority_weight", 0.0))
    provider_priorities = _provider_priority_map(spec)
    diversity_weight = float(spec.get("catalog_diversity_weight", 0.20))
    exclude_sources = {str(item).strip() for item in list(spec.get("catalog_exclude_sources") or []) if str(item).strip()}
    exclude_names = {str(item).strip() for item in list(spec.get("catalog_exclude_names") or []) if str(item).strip()}

    aggregated_entries: dict[str, dict[str, Any]] = {}
    for idx, raw_entry in enumerate(entries_raw):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("catalog entries must be mappings")

        source = _normalize_text(raw_entry.get("source"))
        source_type = _normalize_text(raw_entry.get("source_type", "auto")).lower() or "auto"
        if not source:
            raise ValueError("catalog entry is missing source")
        if source_type == "file":
            raise ValueError(f"{mode} catalog does not allow local file sources")
        if source in exclude_sources:
            continue

        name = _normalize_text(raw_entry.get("name")) or f"catalog_source_{idx + 1}"
        if name in exclude_names:
            continue
        text_field = _normalize_text(raw_entry.get("text_field", "text")) or "text"
        windows = _entry_windows(raw_entry)
        if not windows:
            windows = [name, source]
        prior_weight = float(raw_entry.get("catalog_priority", raw_entry.get("prior_weight", 0.0)))
        title = _normalize_text(raw_entry.get("title")) or name
        provider = _normalize_text(raw_entry.get("provider"))
        query_text = _normalize_text(raw_entry.get("query_text"))
        content_preview = _normalize_text(raw_entry.get("content_preview"))
        content_preview_preferred = bool(raw_entry.get("content_preview_preferred"))
        tags = _normalized_text_list(raw_entry.get("tags"))
        terms = _normalized_text_list(raw_entry.get("terms"))
        aggregated = aggregated_entries.get(source)
        if aggregated is None:
            aggregated_entries[source] = {
                "name": name,
                "source": source,
                "source_type": source_type,
                "hf_config": None if raw_entry.get("hf_config") is None else str(raw_entry.get("hf_config")),
                "text_field": text_field,
                "title": title,
                "primary_summary": windows[0] if windows else name,
                "windows": list(windows),
                "tags": tags,
                "terms": terms,
                "primary_provider": provider,
                "primary_query_text": query_text,
                "primary_content_preview": content_preview,
                "content_preview_preferred": content_preview_preferred,
                "providers": [] if not provider else [provider],
                "query_texts": [] if not query_text else [query_text],
                "prior_weight": float(prior_weight),
                "raw_hits": 1,
            }
            continue

        aggregated["raw_hits"] = int(aggregated["raw_hits"]) + 1
        aggregated["windows"] = _dedupe_keep_order(
            [*list(aggregated["windows"]), *windows]
        )
        aggregated["tags"] = _dedupe_keep_order([*list(aggregated["tags"]), *tags])
        aggregated["terms"] = _dedupe_keep_order([*list(aggregated["terms"]), *terms])
        if provider:
            aggregated["providers"] = _dedupe_keep_order(
                [*list(aggregated["providers"]), provider]
            )
            if not aggregated["primary_provider"]:
                aggregated["primary_provider"] = provider
        if query_text:
            aggregated["query_texts"] = _dedupe_keep_order(
                [*list(aggregated["query_texts"]), query_text]
            )
            if not aggregated["primary_query_text"]:
                aggregated["primary_query_text"] = query_text
        if content_preview:
            if content_preview_preferred or not aggregated["primary_content_preview"]:
                aggregated["primary_content_preview"] = content_preview
            if content_preview_preferred:
                aggregated["content_preview_preferred"] = True
        if float(prior_weight) > float(aggregated["prior_weight"]):
            aggregated["name"] = name
            aggregated["source_type"] = source_type
            aggregated["hf_config"] = None if raw_entry.get("hf_config") is None else str(raw_entry.get("hf_config"))
            aggregated["text_field"] = text_field
            aggregated["title"] = title
            aggregated["primary_summary"] = windows[0] if windows else name
            aggregated["primary_provider"] = provider or str(aggregated["primary_provider"])
            aggregated["primary_query_text"] = query_text or str(aggregated["primary_query_text"])
            aggregated["primary_content_preview"] = content_preview or str(aggregated["primary_content_preview"])
            aggregated["content_preview_preferred"] = content_preview_preferred or bool(
                aggregated["content_preview_preferred"]
            )
            aggregated["prior_weight"] = float(prior_weight)

    catalog_entries: list[tuple[CatalogEntry, set[str]]] = []
    for aggregated in aggregated_entries.values():
        windows = list(aggregated["windows"])
        semantic_relevance = 0.0
        if semantic_plan is not None:
            semantic_relevance = float(
                bank_semantic_relevance_score(
                    {
                        "name": str(aggregated["name"]),
                        "source": str(aggregated["source"]),
                        "probe_raw_windows": windows[: min(4, len(windows))],
                        "train_raw_windows": windows,
                    },
                    semantic_plan,
                )
            )
        prior_weight = float(aggregated["prior_weight"])
        providers = list(aggregated["providers"])
        provider_priority = max(
            (float(provider_priorities.get(str(provider).strip().lower(), 0.0)) for provider in providers if str(provider).strip()),
            default=0.0,
        )
        combined_score = float(
            prior_weight_scale * prior_weight
            + semantic_weight * semantic_relevance
            + provider_priority_weight * provider_priority
        )
        query_texts = list(aggregated["query_texts"])
        metadata = {
            "catalog_mode": mode,
            "catalog_title": str(aggregated["title"]) or str(aggregated["name"]),
            "catalog_summary": str(aggregated["primary_summary"]) or (windows[0] if windows else str(aggregated["name"])),
            "semantic_relevance": float(semantic_relevance),
            "prior_weight": float(prior_weight),
            "combined_score": float(combined_score),
            "tags": list(aggregated["tags"]),
            "catalog_terms": list(aggregated["terms"]),
            "provider": str(aggregated["primary_provider"]) or (providers[0] if providers else ""),
            "query_text": str(aggregated["primary_query_text"]) or (query_texts[0] if query_texts else ""),
            "catalog_content_preview": str(aggregated["primary_content_preview"]),
            "catalog_content_preview_preferred": bool(aggregated["content_preview_preferred"]),
            "providers": providers,
            "query_texts": query_texts,
            "provider_priority": float(provider_priority),
            "duplicate_count": int(aggregated["raw_hits"]),
        }
        catalog_entries.append(
            (
                CatalogEntry(
                    name=str(aggregated["name"]),
                    source=str(aggregated["source"]),
                    source_type=str(aggregated["source_type"]),
                    hf_config=aggregated["hf_config"],
                    text_field=str(aggregated["text_field"]),
                    semantic_relevance=float(semantic_relevance),
                    prior_weight=float(prior_weight),
                    combined_score=float(combined_score),
                    metadata=metadata,
                ),
                _term_profile(windows),
            )
        )

    if not catalog_entries:
        raise ValueError(f"{mode} catalog did not yield any candidate sources")

    selected_entries: list[CatalogEntry] = []
    selected_profiles: list[set[str]] = []
    remaining = catalog_entries[:]
    used_names: set[str] = set()
    while remaining and len(selected_entries) < limit:
        best_index = 0
        best_score = float("-inf")
        for idx, (entry, profile) in enumerate(remaining):
            score = float(entry.combined_score) - diversity_weight * _diversity_penalty(profile, selected_profiles)
            if score > best_score:
                best_index = idx
                best_score = score
        entry, profile = remaining.pop(best_index)
        selected_profiles.append(profile)
        unique_name = entry.name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{entry.name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)
        selected_entries.append(
            CatalogEntry(
                name=unique_name,
                source=entry.source,
                source_type=entry.source_type,
                hf_config=entry.hf_config,
                text_field=entry.text_field,
                semantic_relevance=entry.semantic_relevance,
                prior_weight=entry.prior_weight,
                combined_score=entry.combined_score,
                metadata=entry.metadata,
            )
        )

    return [
        {
            "name": entry.name,
            "source": entry.source,
            "source_type": entry.source_type,
            "hf_config": entry.hf_config,
            "text_field": entry.text_field,
            "metadata": dict(entry.metadata),
        }
        for entry in selected_entries
    ]


def select_catalog_source_specs(
    spec: Mapping[str, Any],
    *,
    semantic_plan: Mapping[str, Any] | None = None,
    metadata_prefilter: bool = False,
) -> list[dict[str, Any]]:
    plan = _focus_plan(spec, semantic_plan)
    entries_raw = list(spec.get("catalog_entries") or [])
    return _rank_catalog_entries(
        mode="semantic_registry",
        entries_raw=entries_raw,
        spec=spec,
        semantic_plan=plan,
        metadata_prefilter=metadata_prefilter,
    )


def discover_remote_search_source_specs(
    spec: Mapping[str, Any],
    *,
    semantic_plan: Mapping[str, Any] | None = None,
    metadata_prefilter: bool = False,
) -> list[dict[str, Any]]:
    plan = _focus_plan(spec, semantic_plan)
    queries = _plan_queries(
        plan,
        spec,
        limit=max(1, int(spec.get("catalog_queries_per_provider", 2))),
    )
    providers = _ordered_live_remote_providers(spec)
    if not queries:
        raise ValueError("live_remote_search catalog requires retrieval queries or catalog_focus_text")

    result_limit = max(1, int(spec.get("catalog_provider_result_limit", 4)))
    timeout_seconds = float(spec.get("catalog_provider_timeout_seconds", 15.0))
    discovered_entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for provider in providers:
        provider_queries = _provider_queries(
            provider,
            queries,
            spec,
            limit=max(1, int(spec.get("catalog_queries_per_provider", 2))),
        )
        for query in provider_queries:
            try:
                discovered_entries.extend(
                    _search_remote_provider_cached(
                        provider,
                        query,
                        result_limit=result_limit,
                        timeout_seconds=timeout_seconds,
                    )
                )
            except Exception as exc:
                failures.append(f"{provider}:{query}:{exc}")
    if not discovered_entries:
        raise ValueError(
            "live_remote_search catalog did not yield any candidate sources"
            + ("" if not failures else f" ({'; '.join(failures[:4])})")
        )
    content_probed_entries = _content_probe_entries(
        discovered_entries,
        mode="live_remote_search",
        spec=spec,
        semantic_plan=plan,
    )

    return _rank_catalog_entries(
        mode="live_remote_search",
        entries_raw=content_probed_entries,
        spec=spec,
        semantic_plan=plan,
        metadata_prefilter=metadata_prefilter,
    )


def expand_source_bank_specs(
    source_bank_specs: Sequence[Mapping[str, Any]],
    *,
    semantic_plan: Mapping[str, Any] | None = None,
    metadata_prefilter: bool = False,
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw_spec in source_bank_specs:
        spec = dict(raw_spec)
        catalog_mode = _normalize_text(spec.get("catalog_mode", "")).lower()
        if not catalog_mode:
            expanded.append(spec)
            continue
        if catalog_mode == "semantic_registry":
            expanded.extend(
                select_catalog_source_specs(
                    spec,
                    semantic_plan=semantic_plan,
                    metadata_prefilter=metadata_prefilter,
                )
            )
            continue
        if catalog_mode == "live_remote_search":
            expanded.extend(
                discover_remote_search_source_specs(
                    spec,
                    semantic_plan=semantic_plan,
                    metadata_prefilter=metadata_prefilter,
                )
            )
            continue
        raise ValueError(f"Unsupported catalog_mode: {catalog_mode}")
    return expanded


__all__ = [
    "CatalogEntry",
    "discover_remote_search_source_specs",
    "expand_source_bank_specs",
    "select_catalog_source_specs",
]
