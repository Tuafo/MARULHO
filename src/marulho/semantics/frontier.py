from __future__ import annotations

import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from marulho.gap_planner import (
    bank_semantic_relevance_score,
    frontier_gap_plan,
    plan_query_gaps,
    tokenize_terms,
)
from .concepts import ConceptStore
from marulho.training.query_runner import episode_quality, top_feature_details

SOURCE_BANK_SIGNATURE_PROBE_LIMIT = 16
SOURCE_BANK_MEMORY_CANDIDATE_WINDOW_LIMIT = 192


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
        "candidate_window_policy": "merged_probe_bucket_indexed_candidate_window",
        "candidate_scope": "source_bank_merged_probe_memory_recall_window",
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
        "raw_text_payload_policy": "returned_merged_probe_matches_only",
        "merged_probe_candidate_window": True,
        "per_probe_query_match_call_count": 0,
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
            "raw_text_payload_policy": "returned_merged_probe_matches_only",
        },
        "probe_reports": [],
    }


def _record_bank_memory_match_report(trainer: Any, report: dict[str, Any]) -> None:
    store = getattr(getattr(trainer, "model", None), "memory_store", None)
    recorder = getattr(store, "record_bank_memory_match_report", None)
    if callable(recorder):
        recorder(report)


def _bank_candidate_bucket_ids(
    trainer: Any,
    routing_key: torch.Tensor,
    *,
    max_buckets: int,
) -> list[int]:
    routing_index = getattr(getattr(trainer, "model", None), "routing_index", None)
    if routing_index is None or not hasattr(routing_index, "search_tensors"):
        return []
    try:
        candidate_ids, _ = routing_index.search_tensors(
            routing_key.detach().unsqueeze(0),
            k=max(1, int(max_buckets)),
        )
    except Exception:
        return []
    if not isinstance(candidate_ids, torch.Tensor) or candidate_ids.numel() <= 0:
        return []
    return [
        int(value)
        for value in candidate_ids[0].detach().cpu().flatten().tolist()
    ]


def _candidate_evidence_pattern(store: Any, idx: int) -> torch.Tensor | None:
    for values_name in ("slow_routing_keys", "slow_input_patterns", "slow_buffer"):
        values = getattr(store, values_name, [])
        try:
            values_count = len(values)
        except TypeError:
            continue
        if idx < 0 or idx >= values_count:
            continue
        value = values[idx]
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu()
    return None


def _safe_sequence_value(values: Any, idx: int, default: Any = None) -> Any:
    try:
        if idx < 0 or idx >= len(values):
            return default
        return values[idx]
    except (TypeError, IndexError):
        return default


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
    if trainer is None or store is None:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason="missing_trainer_or_memory_store",
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

    if not hasattr(store, "collect_query_memory_match_indices"):
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason="memory_store_missing_bounded_query_match_collector",
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        _record_bank_memory_match_report(trainer, report)
        return [], report

    per_probe_limit = max(1, int(memories_per_probe))
    returned_limit = max(1, int(max_matches))
    max_bucket_count = max(
        per_probe_limit,
        int(getattr(getattr(trainer, "config", None), "k_routing", per_probe_limit)),
    )
    probe_reports: list[dict[str, Any]] = []
    active_probe_reports: list[dict[str, Any]] = []
    probe_routing_keys: list[torch.Tensor] = []
    scored_probe_indices: list[int] = []
    candidate_bucket_ids: list[int] = []
    seen_buckets: set[int] = set()
    fallback_reasons: list[str] = []
    for probe_idx in probe_indices:
        pattern = probe_patterns[probe_idx]
        routing_key = trainer.routing_key_for_pattern(pattern).detach().float().cpu()
        if float(routing_key.norm().item()) <= 1e-8:
            reason = f"empty_probe_routing_key:{probe_idx}"
            fallback_reasons.append(reason)
            probe_reports.append(
                {
                    "probe_index": int(probe_idx),
                    "candidate_bucket_ids": [],
                    "candidate_bucket_count": 0,
                    "candidate_window_policy": "merged_probe_bucket_union_before_scoring",
                    "per_probe_query_match_call": False,
                    "fallback_reason": reason,
                }
            )
            continue
        probe_routing_keys.append(routing_key)
        scored_probe_indices.append(int(probe_idx))
        buckets = _bank_candidate_bucket_ids(
            trainer,
            routing_key,
            max_buckets=max_bucket_count,
        )
        for bucket in buckets:
            bucket_int = int(bucket)
            if bucket_int not in seen_buckets:
                seen_buckets.add(bucket_int)
                candidate_bucket_ids.append(bucket_int)
        probe_report = {
            "probe_index": int(probe_idx),
            "candidate_bucket_ids": [int(bucket) for bucket in buckets],
            "candidate_bucket_count": int(len(buckets)),
            "candidate_window_policy": "merged_probe_bucket_union_before_scoring",
            "per_probe_query_match_call": False,
        }
        probe_reports.append(probe_report)
        active_probe_reports.append(probe_report)

    if not probe_routing_keys or not candidate_bucket_ids:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason=";".join(fallback_reasons) or "empty_merged_probe_bucket_window",
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        report.update(
            {
                "probe_count": int(len(probe_indices)),
                "scored_probe_count": int(len(scored_probe_indices)),
                "probe_indices": [int(index) for index in probe_indices],
                "probe_reports": probe_reports,
                "merged_probe_candidate_window": True,
                "per_probe_query_match_call_count": 0,
            }
        )
        _record_bank_memory_match_report(trainer, report)
        return [], report

    candidate_limit = min(
        SOURCE_BANK_MEMORY_CANDIDATE_WINDOW_LIMIT,
        max(
            32,
            returned_limit * 8,
            per_probe_limit * max(1, len(probe_routing_keys)) * 8,
        ),
    )
    candidate_report = dict(
        store.collect_query_memory_match_indices(
            candidate_bucket_ids=candidate_bucket_ids,
            max_candidates=candidate_limit,
            scope="source_bank_merged_probe_memory_match",
        )
    )
    candidate_indices = [
        int(index)
        for index in candidate_report.get("match_indices", [])
        if 0 <= int(index) < len(getattr(store, "slow_buffer", []))
    ]
    if not candidate_indices:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason=str(candidate_report.get("fallback_reason") or "empty_merged_probe_candidate_window"),
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        report.update(
            {
                "probe_count": int(len(probe_indices)),
                "scored_probe_count": int(len(scored_probe_indices)),
                "probe_indices": [int(index) for index in probe_indices],
                "candidate_surface": candidate_report.get("surface"),
                "candidate_window_policy": "merged_probe_bucket_indexed_candidate_window",
                "candidate_scope": "source_bank_merged_probe_memory_recall_window",
                "candidate_bucket_ids": candidate_bucket_ids,
                "candidate_bucket_count": int(len(candidate_bucket_ids)),
                "candidate_index_available_count": int(candidate_report.get("candidate_index_available_count", 0) or 0),
                "probe_reports": probe_reports,
                "merged_probe_candidate_window": True,
                "per_probe_query_match_call_count": 0,
            }
        )
        _record_bank_memory_match_report(trainer, report)
        return [], report

    candidate_vectors: list[torch.Tensor] = []
    vector_candidate_indices: list[int] = []
    for idx in candidate_indices:
        evidence_pattern = _candidate_evidence_pattern(store, idx)
        if evidence_pattern is None or float(evidence_pattern.norm().item()) <= 1e-8:
            continue
        candidate_vectors.append(evidence_pattern)
        vector_candidate_indices.append(int(idx))

    if not candidate_vectors:
        report = _empty_bank_memory_match_report(
            bank_name=bank_name,
            memory_size=memory_size,
            requested_probe_count=probe_samples,
            memories_per_probe=memories_per_probe,
            max_matches=max_matches,
            fallback_reason="empty_candidate_evidence_vectors",
            latency_ms=(time.perf_counter() - started_time) * 1000.0,
        )
        report.update(
            {
                "probe_count": int(len(probe_indices)),
                "scored_probe_count": int(len(scored_probe_indices)),
                "probe_indices": [int(index) for index in probe_indices],
                "candidate_bucket_ids": candidate_bucket_ids,
                "candidate_bucket_count": int(len(candidate_bucket_ids)),
                "candidate_index_count": int(len(candidate_indices)),
                "probe_reports": probe_reports,
                "merged_probe_candidate_window": True,
                "per_probe_query_match_call_count": 0,
            }
        )
        _record_bank_memory_match_report(trainer, report)
        return [], report

    probe_matrix = F.normalize(torch.stack(probe_routing_keys), dim=1)
    candidate_matrix = F.normalize(torch.stack(candidate_vectors), dim=1)
    similarity_matrix = torch.matmul(probe_matrix, candidate_matrix.t())
    replay_scores = store.replay_scores_for_indices(vector_candidate_indices, trainer.token_count)
    selected_by_index: dict[int, dict[str, Any]] = {}
    for probe_position, probe_idx in enumerate(scored_probe_indices):
        row = similarity_matrix[probe_position]
        ranked_positions = sorted(
            range(len(vector_candidate_indices)),
            key=lambda position: (float(row[position].item()), -int(position)),
            reverse=True,
        )[:per_probe_limit]
        selected_indices: list[int] = []
        for position in ranked_positions:
            idx = int(vector_candidate_indices[position])
            selected_indices.append(idx)
            similarity = float(row[position].item())
            existing = selected_by_index.get(idx)
            if existing is None or similarity > float(existing.get("similarity", 0.0)):
                selected_by_index[idx] = {
                    "memory_index": idx,
                    "similarity": similarity,
                    "evidence_pattern": candidate_vectors[position],
                    "replay_priority": float(replay_scores.get(idx, 0.0)),
                    "probe_indices": [int(probe_idx)],
                }
            else:
                existing_probe_indices = list(existing.get("probe_indices") or [])
                if int(probe_idx) not in existing_probe_indices:
                    existing_probe_indices.append(int(probe_idx))
                    existing["probe_indices"] = existing_probe_indices
        if probe_position < len(active_probe_reports):
            active_probe_reports[probe_position]["selected_candidate_indices"] = selected_indices
            active_probe_reports[probe_position]["selected_candidate_count"] = int(len(selected_indices))

    ranked_rows = sorted(
        selected_by_index.values(),
        key=lambda item: (
            float(item.get("similarity", 0.0)),
            float(item.get("replay_priority", 0.0)),
            float(_safe_sequence_value(getattr(store, "slow_importance", []), int(item["memory_index"]), 0.0) or 0.0),
            -int(item["memory_index"]),
        ),
        reverse=True,
    )
    selected_rows = ranked_rows[:returned_limit]
    representation = getattr(getattr(trainer, "config", None), "input_representation", "order_weighted_ascii")
    raw_text_payload_count = 0
    returned: list[dict[str, Any]] = []
    for row in selected_rows:
        idx = int(row["memory_index"])
        replay_entry = store.replay_entry(idx, current_token=trainer.token_count, include_text_payload=True)
        raw_text_payload_count += 1
        replay_metadata = replay_entry.get("metadata") if isinstance(replay_entry.get("metadata"), Mapping) else {}
        text = replay_entry.get("text") or _safe_sequence_value(getattr(store, "slow_raw_windows", []), idx, "")
        raw_window = replay_entry.get("raw_window") or _safe_sequence_value(getattr(store, "slow_raw_windows", []), idx, "")
        consolidation_level = float(
            replay_entry.get(
                "consolidation_level",
                _safe_sequence_value(getattr(store, "slow_consolidation_level", []), idx, 0.0) or 0.0,
            )
        )
        complete_sentence, clipped_overlap = episode_quality(str(text or "").strip(), raw_window)
        returned.append(
            {
                "memory_index": idx,
                "similarity": float(row.get("similarity", 0.0)),
                "bucket_id": _safe_sequence_value(getattr(store, "slow_bucket_ids", []), idx),
                "raw_window": raw_window,
                "text": text,
                "metadata": dict(replay_metadata),
                "source_name": " ".join(str(replay_metadata.get("source_name", "")).split()).strip(),
                "source_type": " ".join(str(replay_metadata.get("source_type", "")).split()).strip(),
                "provider": " ".join(str(replay_metadata.get("provider", "")).split()).strip().lower(),
                "age_tokens": int(
                    max(
                        0,
                        int(getattr(trainer, "token_count", 0))
                        - int(_safe_sequence_value(getattr(store, "slow_entry_timestamps", []), idx, 0) or 0),
                    )
                ),
                "importance": float(_safe_sequence_value(getattr(store, "slow_importance", []), idx, 0.0) or 0.0),
                "tag_strength": float(replay_entry.get("capture_tag", 0.0)),
                "capture_tag": float(replay_entry.get("capture_tag", 0.0)),
                "prp_level": float(replay_entry.get("prp_level", 0.0)),
                "capture_strength": float(replay_entry.get("capture_strength", 0.0)),
                "consolidation_level": consolidation_level,
                "consolidation_gap": float(max(0.0, 1.0 - consolidation_level)),
                "replay_count": int(_safe_sequence_value(getattr(store, "slow_replay_count", []), idx, 0) or 0),
                "replay_priority": float(row.get("replay_priority", 0.0)),
                "top_chars": top_feature_details(row["evidence_pattern"], 1, representation)
                if isinstance(row.get("evidence_pattern"), torch.Tensor)
                else [],
                "query_overlap": 0,
                "matched_query_terms": [],
                "focus_overlap": 0,
                "matched_focus_terms": [],
                "memory_focus_priority": 0.0,
                "complete_sentence": int(complete_sentence),
                "clipped_overlap": int(clipped_overlap),
                "probe_indices": [int(value) for value in list(row.get("probe_indices") or [])],
            }
        )

    latency_ms = (time.perf_counter() - started_time) * 1000.0
    report = {
        "surface": "bounded_source_bank_memory_match.v1",
        "status": "matched" if returned else "empty",
        "scope": "source_bank_semantic_recall_slow_path",
        "bank_name": bank_name,
        "memory_size": int(memory_size),
        "requested_probe_count": int(max(0, probe_samples)),
        "probe_count": int(len(probe_indices)),
        "scored_probe_count": int(len(scored_probe_indices)),
        "probe_indices": [int(index) for index in probe_indices],
        "memories_per_probe": int(per_probe_limit),
        "max_matches": int(returned_limit),
        "candidate_surface": candidate_report.get("surface"),
        "candidate_window_policy": "merged_probe_bucket_indexed_candidate_window",
        "candidate_scope": "source_bank_merged_probe_memory_recall_window",
        "candidate_bucket_ids": candidate_bucket_ids,
        "candidate_bucket_count": int(len(candidate_bucket_ids)),
        "candidate_index_available_count": int(candidate_report.get("candidate_index_available_count", 0) or 0),
        "candidate_index_count": int(len(candidate_indices)),
        "unique_candidate_index_count": int(len(set(candidate_indices))),
        "similarity_score_count": int(len(vector_candidate_indices) * len(probe_routing_keys)),
        "replay_priority_score_count": int(len(replay_scores)),
        "merged_probe_candidate_window": True,
        "per_probe_query_match_call_count": 0,
        "retired_per_probe_query_match_call_count": int(len(probe_indices)),
        "candidate_window_limit": int(candidate_limit),
        "match_indices": [int(item["memory_index"]) for item in returned],
        "result_count": int(len(ranked_rows)),
        "returned_count": int(len(returned)),
        "raw_text_payload_loaded": bool(raw_text_payload_count > 0),
        "raw_text_payload_count": int(raw_text_payload_count),
        "raw_text_payload_cache_hits": 0,
        "raw_text_payload_policy": "returned_merged_probe_matches_only",
        "global_score_scan": bool(candidate_report.get("global_score_scan")),
        "global_candidate_scan": bool(candidate_report.get("global_candidate_scan")),
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
            "per_probe_return_budget": int(per_probe_limit),
            "returned_match_limit": int(returned_limit),
            "candidate_bucket_budget": int(max_bucket_count),
            "candidate_window_limit": int(candidate_limit),
            "candidate_window_cap": int(SOURCE_BANK_MEMORY_CANDIDATE_WINDOW_LIMIT),
            "raw_text_payload_policy": "returned_merged_probe_matches_only",
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
