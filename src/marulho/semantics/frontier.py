from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from marulho.gap_planner import (
    bank_semantic_relevance_score,
    frontier_gap_plan,
    plan_query_gaps,
    tokenize_terms,
)
from .concepts import ConceptStore
from marulho.training.query_runner import memory_matches_with_report as retrieve_memory_matches_with_report

SOURCE_BANK_SIGNATURE_PROBE_LIMIT = 16


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def candidate_semantic_signature(
    trainer: Any,
    bank: Any,
    *,
    probe_limit: int = SOURCE_BANK_SIGNATURE_PROBE_LIMIT,
) -> torch.Tensor | None:
    if trainer is None or not getattr(bank, "probe_patterns", None):
        return None
    probe_patterns = list(getattr(bank, "probe_patterns", []) or [])
    probe_indices = _sample_probe_indices(len(probe_patterns), int(probe_limit))
    vectors = [
        trainer.routing_key_for_pattern(probe_patterns[index]).detach().cpu()
        for index in probe_indices
    ]
    if not vectors:
        return None
    mean = torch.stack(vectors).mean(dim=0)
    if float(mean.norm().item()) <= 1e-8:
        return None
    return F.normalize(mean, dim=0)


def current_context_signature(trainer: Any) -> torch.Tensor | None:
    if trainer is None:
        return None
    context_state = trainer.context_state().to(trainer.model.device)
    if float(context_state.sum().item()) <= 0.0:
        return None
    latent = torch.mv(trainer.model.W_assembly_project.t(), context_state)
    if float(latent.norm().item()) <= 1e-8:
        return None
    return F.normalize(latent, dim=0).detach().cpu()


def build_bank_query_text(bank: Any, *, max_windows: int = 6, max_terms: int = 24) -> str:
    windows = list(getattr(bank, "probe_raw_windows", []) or []) or list(getattr(bank, "train_raw_windows", []) or [])
    seen: set[str] = set()
    terms: list[str] = []
    for raw_window in windows[: max(1, int(max_windows))]:
        for term in tokenize_terms(str(raw_window)):
            if len(term) < 4 and not term.isdigit():
                continue
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= max(1, int(max_terms)):
                return " ".join(terms)
    name = str(getattr(bank, "name", "")).strip()
    if terms:
        return " ".join(terms)
    return name


def _sample_probe_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [total // 2]
    step = float(max(1, total - 1)) / float(max(1, limit - 1))
    indices = {int(round(step * float(idx))) for idx in range(limit)}
    return sorted(max(0, min(total - 1, idx)) for idx in indices)


def _empty_bank_memory_match_report(
    *,
    bank_name: str,
    memory_size: int,
    requested_probe_count: int,
    memories_per_probe: int,
    max_matches: int,
    fallback_reason: str,
    latency_ms: float = 0.0,
) -> dict[str, Any]:
    per_probe_limit = max(1, int(memories_per_probe))
    return {
        "surface": "bounded_source_bank_memory_match.v1",
        "status": "empty",
        "scope": "source_bank_semantic_recall_slow_path",
        "bank_name": str(bank_name),
        "memory_size": int(memory_size),
        "requested_probe_count": int(max(0, requested_probe_count)),
        "probe_count": 0,
        "probe_indices": [],
        "memories_per_probe": int(per_probe_limit),
        "max_matches": int(max(1, max_matches)),
        "candidate_surface": "bounded_query_memory_match.v1",
        "candidate_window_policy": "per_probe_bucket_indexed_candidate_window",
        "candidate_scope": "source_bank_probe_memory_recall_window",
        "candidate_bucket_ids": [],
        "candidate_bucket_count": 0,
        "candidate_index_available_count": 0,
        "candidate_index_count": 0,
        "unique_candidate_index_count": 0,
        "similarity_score_count": 0,
        "replay_priority_score_count": 0,
        "match_indices": [],
        "result_count": 0,
        "returned_count": 0,
        "raw_text_payload_loaded": False,
        "raw_text_payload_count": 0,
        "raw_text_payload_cache_hits": 0,
        "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "language_reasoning": False,
        "score_device": "cpu",
        "archival_storage_device": "cpu",
        "quality_metric": "semantic_grounding_gap_inputs",
        "latency_ms": float(latency_ms),
        "fallback_reason": str(fallback_reason),
        "selection_budget": {
            "memory_budget_entries": int(memory_size),
            "probe_budget": int(max(0, requested_probe_count)),
            "per_probe_return_budget": int(per_probe_limit),
            "returned_match_limit": int(max(1, max_matches)),
            "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        },
        "probe_reports": [],
    }


def _record_bank_memory_match_report(trainer: Any, report: dict[str, Any]) -> None:
    store = getattr(getattr(trainer, "model", None), "memory_store", None)
    recorder = getattr(store, "record_bank_memory_match_report", None)
    if callable(recorder):
        recorder(report)


def bank_memory_matches_with_report(
    trainer: Any,
    bank: Any,
    *,
    probe_samples: int = 4,
    memories_per_probe: int = 3,
    max_matches: int = 18,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started_time = time.perf_counter()
    store = getattr(getattr(trainer, "model", None), "memory_store", None)
    memory_size = int(len(getattr(store, "slow_buffer", []))) if store is not None else 0
    bank_name = str(getattr(bank, "name", ""))
    if trainer is None:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason="missing_trainer",
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        return [], report

    probe_patterns = list(getattr(bank, "probe_patterns", []) or [])
    probe_indices = _sample_probe_indices(len(probe_patterns), probe_samples)
    if not probe_indices:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason="empty_source_bank_probe_window",
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        _record_bank_memory_match_report(trainer, report)
        return [], report

    aggregated: dict[int, dict[str, Any]] = {}
    replay_entry_cache: dict[int, dict[str, Any]] = {}
    probe_reports: list[dict[str, Any]] = []
    candidate_buckets: set[int] = set()
    candidate_indices: set[int] = set()
    candidate_available_total = 0
    candidate_count_total = 0
    similarity_score_total = 0
    replay_priority_score_total = 0
    raw_text_payload_count = 0
    raw_text_payload_cache_hits = 0
    global_score_scan = False
    global_candidate_scan = False
    fallback_reasons: list[str] = []

    for probe_idx in probe_indices:
        pattern = probe_patterns[probe_idx]
        routing_key = trainer.routing_key_for_pattern(pattern)
        matches, match_report = retrieve_memory_matches_with_report(
            trainer,
            pattern,
            routing_key,
            top_k=memories_per_probe,
            top_chars=1,
            replay_entry_cache=replay_entry_cache,
        )
        probe_report = {
            "probe_index": int(probe_idx),
            "surface": match_report.get("surface"),
            "candidate_surface": match_report.get("candidate_surface"),
            "candidate_scope": match_report.get("candidate_scope"),
            "candidate_bucket_ids": [
                int(bucket)
                for bucket in match_report.get("candidate_bucket_ids", [])
            ],
            "candidate_index_available_count": int(
                match_report.get("candidate_index_available_count", 0) or 0
            ),
            "candidate_index_count": int(
                match_report.get("candidate_index_count", 0) or 0
            ),
            "similarity_score_count": int(
                match_report.get("similarity_score_count", 0) or 0
            ),
            "replay_priority_score_count": int(
                match_report.get("replay_priority_score_count", 0) or 0
            ),
            "returned_count": int(match_report.get("returned_count", 0) or 0),
            "raw_text_payload_count": int(
                match_report.get("raw_text_payload_count", 0) or 0
            ),
            "raw_text_payload_cache_hits": int(
                match_report.get("raw_text_payload_cache_hits", 0) or 0
            ),
            "fallback_reason": match_report.get("fallback_reason"),
        }
        probe_reports.append(probe_report)
        candidate_available_total += int(probe_report["candidate_index_available_count"])
        candidate_count_total += int(probe_report["candidate_index_count"])
        similarity_score_total += int(probe_report["similarity_score_count"])
        replay_priority_score_total += int(probe_report["replay_priority_score_count"])
        raw_text_payload_count += int(probe_report["raw_text_payload_count"])
        raw_text_payload_cache_hits += int(probe_report["raw_text_payload_cache_hits"])
        global_score_scan = global_score_scan or bool(match_report.get("global_score_scan"))
        global_candidate_scan = global_candidate_scan or bool(match_report.get("global_candidate_scan"))
        fallback_reason = match_report.get("fallback_reason")
        if fallback_reason is not None and str(fallback_reason) not in fallback_reasons:
            fallback_reasons.append(str(fallback_reason))
        for bucket in probe_report["candidate_bucket_ids"]:
            candidate_buckets.add(int(bucket))
        for index in match_report.get("match_indices", []):
            candidate_indices.add(int(index))
        for match in matches:
            memory_index = match.get("memory_index")
            if not isinstance(memory_index, int):
                continue
            existing = aggregated.get(memory_index)
            if existing is None or float(match.get("similarity", 0.0)) > float(existing.get("similarity", 0.0)):
                updated = dict(match)
                updated["probe_indices"] = [int(probe_idx)]
                aggregated[memory_index] = updated
            else:
                indices = list(existing.get("probe_indices") or [])
                if int(probe_idx) not in indices:
                    indices.append(int(probe_idx))
                    existing["probe_indices"] = indices

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item.get("similarity", 0.0)),
            float(item.get("capture_strength", 0.0)),
            float(item.get("importance", 0.0)),
        ),
        reverse=True,
    )
    returned = ranked[: max(1, int(max_matches))]
    latency_ms = (time.perf_counter() - started_time) * 1000.0
    report = {
        "surface": "bounded_source_bank_memory_match.v1",
        "status": "matched" if returned else "empty",
        "scope": "source_bank_semantic_recall_slow_path",
        "bank_name": bank_name,
        "memory_size": int(memory_size),
        "requested_probe_count": int(max(0, probe_samples)),
        "probe_count": int(len(probe_indices)),
        "probe_indices": [int(index) for index in probe_indices],
        "memories_per_probe": int(max(1, memories_per_probe)),
        "max_matches": int(max(1, max_matches)),
        "candidate_surface": "bounded_query_memory_match.v1",
        "candidate_window_policy": "per_probe_bucket_indexed_candidate_window",
        "candidate_scope": "source_bank_probe_memory_recall_window",
        "candidate_bucket_ids": sorted(candidate_buckets),
        "candidate_bucket_count": int(len(candidate_buckets)),
        "candidate_index_available_count": int(candidate_available_total),
        "candidate_index_count": int(candidate_count_total),
        "unique_candidate_index_count": int(len(candidate_indices)),
        "similarity_score_count": int(similarity_score_total),
        "replay_priority_score_count": int(replay_priority_score_total),
        "match_indices": [int(item["memory_index"]) for item in returned],
        "result_count": int(len(ranked)),
        "returned_count": int(len(returned)),
        "raw_text_payload_loaded": bool(raw_text_payload_count > 0),
        "raw_text_payload_count": int(raw_text_payload_count),
        "raw_text_payload_cache_hits": int(raw_text_payload_cache_hits),
        "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        "global_score_scan": bool(global_score_scan),
        "global_candidate_scan": bool(global_candidate_scan),
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "language_reasoning": False,
        "score_device": "cpu",
        "archival_storage_device": "cpu",
        "quality_metric": "semantic_grounding_gap_inputs",
        "latency_ms": float(latency_ms),
        "fallback_reason": None if returned else ";".join(fallback_reasons) or "empty_bank_memory_matches",
        "selection_budget": {
            "memory_budget_entries": int(memory_size),
            "probe_budget": int(max(0, probe_samples)),
            "per_probe_return_budget": int(max(1, memories_per_probe)),
            "returned_match_limit": int(max(1, max_matches)),
            "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        },
        "probe_reports": probe_reports,
    }
    _record_bank_memory_match_report(trainer, report)
    return returned, report


def bank_gap_plan(
    trainer: Any,
    bank: Any,
    *,
    probe_samples: int = 4,
    memories_per_probe: int = 3,
    concept_limit: int = 6,
) -> dict[str, Any]:
    query_text = build_bank_query_text(bank)
    if not query_text:
        store = getattr(getattr(trainer, "model", None), "memory_store", None)
        memory_match_report = _empty_bank_memory_match_report(
            bank_name=str(getattr(bank, "name", "")),
            memory_size=int(len(getattr(store, "slow_buffer", []))) if store is not None else 0,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=1,
            fallback_reason="empty_source_bank_query_text",
        )
        _record_bank_memory_match_report(trainer, memory_match_report)
        return {
            "query_text": "",
            "query_summary": {"memory_matches": [], "memory_match_report": memory_match_report},
            "concept_summary": {"concepts": []},
            "gap_plan": plan_query_gaps(query_text="", query_summary={"memory_matches": []}, concept_summary={"concepts": []}),
            "bank_memory_match_report": memory_match_report,
            "grounding_gap": 1.0,
            "unsupported_ratio": 1.0,
            "weak_concept_pressure": 1.0,
            "answerability": 0.0,
            "semantic_priority": 1.0,
        }

    matches, memory_match_report = bank_memory_matches_with_report(
        trainer,
        bank,
        probe_samples=probe_samples,
        memories_per_probe=memories_per_probe,
    )
    concept_store = ConceptStore()
    concept_summary = concept_store.observe(
        query_text=query_text,
        memory_matches=matches,
        memory_store=None if trainer is None else trainer.model.memory_store,
        limit=concept_limit,
    )
    gap_plan = plan_query_gaps(
        query_text=query_text,
        query_summary={"memory_matches": matches},
        concept_summary=concept_summary,
    )
    query_terms = list(gap_plan.get("query_terms") or [])
    unsupported_terms = list(gap_plan.get("unsupported_terms") or [])
    weak_concepts = list(gap_plan.get("weak_concepts") or [])
    grounding_gap = _clamp01(1.0 - float(gap_plan.get("grounded_fraction", 0.0)))
    unsupported_ratio = _clamp01(float(len(unsupported_terms)) / max(1.0, float(len(query_terms))))
    weak_pressure = _clamp01(
        sum(float(item.get("weakness", 0.0)) for item in weak_concepts) / max(1.0, float(len(weak_concepts)))
    )
    answerability = bank_semantic_relevance_score(bank, gap_plan)
    semantic_priority = _clamp01(
        0.35 * grounding_gap
        + 0.25 * unsupported_ratio
        + 0.20 * weak_pressure
        + 0.20 * answerability
    )
    return {
        "query_text": query_text,
        "query_summary": {
            "memory_matches": matches,
            "memory_match_report": memory_match_report,
        },
        "concept_summary": concept_summary,
        "gap_plan": gap_plan,
        "bank_memory_match_report": memory_match_report,
        "grounding_gap": grounding_gap,
        "unsupported_ratio": unsupported_ratio,
        "weak_concept_pressure": weak_pressure,
        "answerability": answerability,
        "semantic_priority": semantic_priority,
    }


def frontier_semantic_plan(trainer: Any, *, max_terms: int = 8, max_queries: int = 4, max_questions: int = 4) -> dict[str, Any]:
    memory_store = None if trainer is None else getattr(getattr(trainer, "model", None), "memory_store", None)
    current_token = 0 if trainer is None else int(getattr(trainer, "token_count", 0))
    return frontier_gap_plan(
        memory_store=memory_store,
        current_token=current_token,
        max_terms=max_terms,
        max_queries=max_queries,
        max_questions=max_questions,
    )


__all__ = [
    "bank_gap_plan",
    "bank_memory_matches_with_report",
    "build_bank_query_text",
    "candidate_semantic_signature",
    "current_context_signature",
    "frontier_semantic_plan",
]
