from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from hecsn.gap_planner import (
    bank_semantic_relevance_score,
    frontier_gap_plan,
    plan_query_gaps,
    tokenize_terms,
)
from .concepts import ConceptStore
from hecsn.training.query_runner import memory_matches as retrieve_memory_matches


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def candidate_semantic_signature(trainer: Any, bank: Any) -> torch.Tensor | None:
    if trainer is None or not getattr(bank, "probe_patterns", None):
        return None
    vectors = [trainer.routing_key_for_pattern(pattern).detach().cpu() for pattern in bank.probe_patterns]
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


def bank_memory_matches(
    trainer: Any,
    bank: Any,
    *,
    probe_samples: int = 4,
    memories_per_probe: int = 3,
    max_matches: int = 18,
) -> list[dict[str, Any]]:
    if trainer is None:
        return []
    probe_patterns = list(getattr(bank, "probe_patterns", []) or [])
    if not probe_patterns:
        return []

    aggregated: dict[int, dict[str, Any]] = {}
    for probe_idx in _sample_probe_indices(len(probe_patterns), probe_samples):
        pattern = probe_patterns[probe_idx]
        routing_key = trainer.routing_key_for_pattern(pattern)
        for match in retrieve_memory_matches(
            trainer,
            pattern,
            routing_key,
            top_k=memories_per_probe,
            top_chars=1,
        ):
            memory_index = match.get("memory_index")
            if not isinstance(memory_index, int):
                continue
            existing = aggregated.get(memory_index)
            if existing is None or float(match.get("similarity", 0.0)) > float(existing.get("similarity", 0.0)):
                aggregated[memory_index] = dict(match)

    ranked = sorted(
        aggregated.values(),
        key=lambda item: (
            float(item.get("similarity", 0.0)),
            float(item.get("capture_strength", 0.0)),
            float(item.get("importance", 0.0)),
        ),
        reverse=True,
    )
    return ranked[: max(1, int(max_matches))]


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
        return {
            "query_text": "",
            "query_summary": {"memory_matches": []},
            "concept_summary": {"concepts": []},
            "gap_plan": plan_query_gaps(query_text="", query_summary={"memory_matches": []}, concept_summary={"concepts": []}),
            "grounding_gap": 1.0,
            "unsupported_ratio": 1.0,
            "weak_concept_pressure": 1.0,
            "answerability": 0.0,
            "semantic_priority": 1.0,
        }

    matches = bank_memory_matches(
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
        "query_summary": {"memory_matches": matches},
        "concept_summary": concept_summary,
        "gap_plan": gap_plan,
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
    "bank_memory_matches",
    "build_bank_query_text",
    "candidate_semantic_signature",
    "current_context_signature",
    "frontier_semantic_plan",
]
