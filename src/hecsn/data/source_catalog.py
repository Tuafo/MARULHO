from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
import re
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from hecsn.gap_planner import (
    bank_semantic_relevance_score,
    plan_query_gaps,
    tokenize_terms,
)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


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
    for key in ("summary", "title", "description"):
        value = _normalize_text(entry.get(key))
        if value and len(tokenize_terms(value)) >= 2:
            windows.append(value)

    query_text = _normalize_text(entry.get("query_text"))
    if query_text and not windows and len(tokenize_terms(query_text)) >= 2:
        windows.append(query_text)

    tags = entry.get("tags")
    if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)):
        tag_text = _normalize_text(" ".join(str(tag) for tag in tags if str(tag).strip()))
        if tag_text and len(tokenize_terms(tag_text)) >= 2:
            windows.append(tag_text)

    focus_terms = entry.get("terms")
    if isinstance(focus_terms, Sequence) and not isinstance(focus_terms, (str, bytes)):
        term_text = _normalize_text(" ".join(str(term) for term in focus_terms if str(term).strip()))
        if term_text and len(tokenize_terms(term_text)) >= 2:
            windows.append(term_text)
    return windows


def _term_profile(texts: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for text in texts:
        terms.update(tokenize_terms(text))
    return terms


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


def _http_get_json(url: str, *, timeout_seconds: float) -> Any:
    request = Request(
        url,
        headers={
            "User-Agent": "HECSN/1.0 source discovery",
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
            "User-Agent": "HECSN/1.0 source discovery",
            "Accept": "application/atom+xml,text/xml;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        return response.read().decode("utf-8", errors="ignore")


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
    entries: list[dict[str, Any]] = []
    for rank, item in enumerate(results):
        title = _normalize_text(item.get("title"))
        if not title:
            continue
        page_path = quote(title.replace(" ", "_"), safe="_")
        entries.append(
            {
                "name": _slugify(title, default=f"wikipedia_{rank + 1}"),
                "title": title,
                "summary": _strip_html(str(item.get("snippet") or title)),
                "source": f"https://en.wikipedia.org/wiki/{page_path}",
                "source_type": "web",
                "text_field": "text",
                "query_text": query,
                "tags": ["wikipedia", *tokenize_terms(query)[:3]],
                "catalog_priority": float(1.0 / float(rank + 1)),
                "provider": "wikipedia",
            }
        )
    return entries


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
                "source": source,
                "source_type": "web",
                "text_field": "text",
                "query_text": query,
                "tags": ["arxiv", *tokenize_terms(query)[:3]],
                "catalog_priority": float(1.0 / float(rank + 1)),
                "provider": "arxiv",
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
    diversity_weight = float(spec.get("catalog_diversity_weight", 0.20))
    exclude_sources = {str(item).strip() for item in list(spec.get("catalog_exclude_sources") or []) if str(item).strip()}
    exclude_names = {str(item).strip() for item in list(spec.get("catalog_exclude_names") or []) if str(item).strip()}

    catalog_entries: list[tuple[CatalogEntry, set[str]]] = []
    seen_sources: set[str] = set()
    for idx, raw_entry in enumerate(entries_raw):
        if not isinstance(raw_entry, Mapping):
            raise ValueError("catalog entries must be mappings")

        source = _normalize_text(raw_entry.get("source"))
        source_type = _normalize_text(raw_entry.get("source_type", "auto")).lower() or "auto"
        if not source:
            raise ValueError("catalog entry is missing source")
        if source_type == "file":
            raise ValueError(f"{mode} catalog does not allow local file sources")
        if source in exclude_sources or source in seen_sources:
            continue

        name = _normalize_text(raw_entry.get("name")) or f"catalog_source_{idx + 1}"
        if name in exclude_names:
            continue
        seen_sources.add(source)
        text_field = _normalize_text(raw_entry.get("text_field", "text")) or "text"
        windows = _entry_windows(raw_entry)
        if not windows:
            windows = [name, source]

        semantic_relevance = 0.0
        if semantic_plan is not None:
            semantic_relevance = float(
                bank_semantic_relevance_score(
                    {
                        "name": name,
                        "source": source,
                        "probe_raw_windows": windows[: min(4, len(windows))],
                        "train_raw_windows": windows,
                    },
                    semantic_plan,
                )
            )
        prior_weight = float(raw_entry.get("catalog_priority", raw_entry.get("prior_weight", 0.0)))
        combined_score = float(prior_weight_scale * prior_weight + semantic_weight * semantic_relevance)
        metadata = {
            "catalog_mode": mode,
            "catalog_title": _normalize_text(raw_entry.get("title")) or name,
            "catalog_summary": windows[0] if windows else name,
            "semantic_relevance": float(semantic_relevance),
            "prior_weight": float(prior_weight),
            "combined_score": float(combined_score),
            "tags": list(raw_entry.get("tags") or []),
            "provider": _normalize_text(raw_entry.get("provider")),
            "query_text": _normalize_text(raw_entry.get("query_text")),
        }
        catalog_entries.append(
            (
                CatalogEntry(
                    name=name,
                    source=source,
                    source_type=source_type,
                    hf_config=None if raw_entry.get("hf_config") is None else str(raw_entry.get("hf_config")),
                    text_field=text_field,
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
    providers = [str(item) for item in list(spec.get("catalog_providers") or ["wikipedia"])]
    if not queries:
        raise ValueError("live_remote_search catalog requires retrieval queries or catalog_focus_text")

    result_limit = max(1, int(spec.get("catalog_provider_result_limit", 4)))
    timeout_seconds = float(spec.get("catalog_provider_timeout_seconds", 15.0))
    discovered_entries: list[dict[str, Any]] = []
    failures: list[str] = []
    for provider in providers:
        for query in queries:
            try:
                discovered_entries.extend(
                    _search_remote_provider(
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

    return _rank_catalog_entries(
        mode="live_remote_search",
        entries_raw=discovered_entries,
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
