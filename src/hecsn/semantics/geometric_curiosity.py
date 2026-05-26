from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from hecsn.core.abstraction import AbstractionLayer
from hecsn.semantics.grounding_text import normalize_text, salient_query_terms


def _dedupe_strings(values: Sequence[str], limit: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = normalize_text(raw_value).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
        if len(ordered) >= max(1, int(limit)):
            break
    return ordered


class GeometricCuriosityController:
    """Synthesize retrieval focus from abstraction-space gaps and a learned lexicon."""

    def __init__(
        self,
        abstraction_layer: AbstractionLayer | None,
        *,
        lexicon_limit_per_concept: int = 24,
        top_concepts_per_update: int = 4,
        gap_threshold: float = 0.05,
    ) -> None:
        self.abstraction_layer = abstraction_layer
        self.lexicon_limit_per_concept = max(1, int(lexicon_limit_per_concept))
        self.top_concepts_per_update = max(1, int(top_concepts_per_update))
        self.gap_threshold = max(0.0, float(gap_threshold))
        self.lexicon: dict[int, list[str]] = {}

    def bind(self, abstraction_layer: AbstractionLayer | None) -> None:
        self.abstraction_layer = abstraction_layer

    def _concept_embeddings(self) -> torch.Tensor | None:
        if self.abstraction_layer is None:
            return None
        return F.normalize(self.abstraction_layer.feedforward.detach().float(), dim=1)

    def update_lexicon(
        self,
        concept_activations: torch.Tensor | None,
        active_chunks: Sequence[str] | None,
    ) -> None:
        if self.abstraction_layer is None or concept_activations is None:
            return
        chunks = [normalize_text(chunk) for chunk in list(active_chunks or []) if normalize_text(chunk)]
        if not chunks:
            return
        tokens = _dedupe_strings(
            [token for chunk in chunks for token in salient_query_terms(chunk)],
            limit=self.lexicon_limit_per_concept,
        )
        if not tokens:
            return

        activations = torch.as_tensor(concept_activations, dtype=torch.float32).flatten()
        if activations.numel() != self.abstraction_layer.n_concepts:
            return
        top_k = min(self.top_concepts_per_update, int(activations.numel()))
        values, indices = torch.topk(activations, k=top_k)
        if float(values.max().item()) <= 0.0:
            return

        for raw_index in indices.tolist():
            concept_idx = int(raw_index)
            existing = list(self.lexicon.get(concept_idx, []))
            merged = _dedupe_strings([*tokens, *existing], limit=self.lexicon_limit_per_concept)
            if merged:
                self.lexicon[concept_idx] = merged

    def _neighbor_indices(self, gap_idx: int, *, max_neighbors: int = 2) -> list[int]:
        embeddings = self._concept_embeddings()
        if embeddings is None or gap_idx < 0 or gap_idx >= int(embeddings.shape[0]):
            return []
        if int(embeddings.shape[0]) <= 1:
            return []

        gap_vector = embeddings[gap_idx]
        sims = torch.mv(embeddings, gap_vector)
        sims[gap_idx] = -1.0
        if self.abstraction_layer is not None:
            stability = self.abstraction_layer.concept_stability.detach().float()
            certainty = self.abstraction_layer.concept_certainty.detach().float()
            sims = sims * (0.5 + 0.5 * stability) * (0.5 + 0.5 * certainty)
            sims[gap_idx] = -1.0

        ordered = torch.argsort(sims, descending=True).tolist()
        neighbors: list[int] = []
        for raw_index in ordered:
            idx = int(raw_index)
            if idx == gap_idx or float(sims[idx].item()) <= -0.5:
                continue
            if not self.lexicon.get(idx):
                continue
            neighbors.append(idx)
            if len(neighbors) >= max(1, int(max_neighbors)):
                break
        return neighbors

    def focus_plan(
        self,
        *,
        query_text: str | None = None,
        top_n: int = 3,
    ) -> dict[str, Any] | None:
        if self.abstraction_layer is None:
            return None
        query_terms = _dedupe_strings(salient_query_terms(query_text or ""), limit=8)
        raw_gaps = self.abstraction_layer.curiosity_gaps(top_n=max(1, int(top_n)))
        max_gap = max((float(item.get("gap_score", 0.0)) for item in raw_gaps), default=0.0)
        effective_threshold = min(self.gap_threshold, 0.75 * max_gap) if max_gap > 0.0 else self.gap_threshold
        gaps = [
            item
            for item in raw_gaps
            if float(item.get("gap_score", 0.0)) > 0.0
            and float(item.get("gap_score", 0.0)) >= effective_threshold
        ]
        if not gaps:
            return None

        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concepts: list[dict[str, Any]] = []
        gap_terms: dict[str, float] = {}
        geometric_gaps: list[dict[str, Any]] = []
        aggregate_query_terms: list[str] = list(query_terms)

        for gap in gaps:
            gap_idx = int(gap.get("concept_idx", -1))
            if gap_idx < 0:
                continue
            neighbor_indices = self._neighbor_indices(gap_idx)
            if not neighbor_indices:
                continue
            neighbor_terms = _dedupe_strings(
                [term for idx in neighbor_indices for term in list(self.lexicon.get(idx, []))],
                limit=6,
            )
            if not neighbor_terms:
                continue
            query_basis = _dedupe_strings([*query_terms, *neighbor_terms], limit=6)
            retrieval_query = " ".join(query_basis).strip()
            if not retrieval_query:
                continue
            retrieval_queries.append(retrieval_query)
            aggregate_query_terms.extend(query_basis)
            if query_terms:
                anchor = " ".join(query_terms[:2]).strip() or retrieval_query
                target = " ".join(neighbor_terms[:2]).strip() or retrieval_query
                follow_up_questions.append(f"What grounded evidence connects {anchor} to {target}?")
            else:
                target = " ".join(neighbor_terms[:2]).strip() or retrieval_query
                follow_up_questions.append(f"What grounded evidence would stabilize the gap around {target}?")

            gap_score = float(gap.get("gap_score", 0.0))
            for rank, term in enumerate(neighbor_terms[:3]):
                gap_terms[term] = max(float(gap_terms.get(term, 0.0)), gap_score / float(rank + 1))
            weak_concepts.append(
                {
                    "label": f"gap_concept_{gap_idx}",
                    "weakness": gap_score,
                    "uncertainty": float(1.0 - float(gap.get("certainty", 0.0))),
                    "drift": float(gap.get("stability", 0.0)),
                    "top_terms": neighbor_terms[:4],
                    "match_count": len(neighbor_terms),
                }
            )
            geometric_gaps.append(
                {
                    "concept_idx": gap_idx,
                    "gap_score": gap_score,
                    "neighbor_indices": neighbor_indices,
                    "neighbor_terms": neighbor_terms[:4],
                }
            )

        if not retrieval_queries:
            return None

        return {
            "planner_mode": "geometric_abstraction_gap_focus",
            "query_terms": _dedupe_strings(aggregate_query_terms, limit=8),
            "focus_terms": _dedupe_strings(aggregate_query_terms, limit=8),
            "unsupported_terms": [],
            "gap_terms": [
                {"term": str(term), "weight": float(weight)}
                for term, weight in sorted(gap_terms.items(), key=lambda item: (-float(item[1]), item[0]))[:8]
            ],
            "retrieval_queries": _dedupe_strings(retrieval_queries, limit=4),
            "follow_up_questions": _dedupe_strings(follow_up_questions, limit=4),
            "weak_concepts": weak_concepts[:4],
            "geometric_gaps": geometric_gaps[:4],
        }

    def boost_concept(self, label: str, amount: float = 0.1) -> None:
        """Boost curiosity for a concept used by deliberation feedback.

        Lowers the concept's certainty in the abstraction layer, making it
        a more attractive curiosity target for the next training steps.
        """
        if self.abstraction_layer is None or not label:
            return
        # Find matching concept index by label text
        label_lower = label.lower()
        for idx, words in self.lexicon.items():
            for w in words:
                if label_lower in w.lower() or w.lower() in label_lower:
                    if idx < self.abstraction_layer.n_concepts:
                        self.abstraction_layer.concept_certainty.data[idx] = max(
                            0.0,
                            self.abstraction_layer.concept_certainty.data[idx] - amount,
                        )
                    return

    def summary(self) -> dict[str, Any]:
        lexicon_entries = sum(len(values) for values in self.lexicon.values())
        gap_preview = self.focus_plan(top_n=2)
        return {
            "enabled": bool(self.abstraction_layer is not None),
            "lexicon_concept_count": int(len(self.lexicon)),
            "lexicon_entry_count": int(lexicon_entries),
            "has_focus_plan": bool(gap_preview is not None),
            "planner_mode": None if gap_preview is None else str(gap_preview.get("planner_mode")),
            "top_retrieval_query": None
            if gap_preview is None
            else next(iter(list(gap_preview.get("retrieval_queries") or [])), None),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "lexicon_limit_per_concept": int(self.lexicon_limit_per_concept),
            "top_concepts_per_update": int(self.top_concepts_per_update),
            "gap_threshold": float(self.gap_threshold),
            "lexicon": {str(key): list(values) for key, values in sorted(self.lexicon.items())},
        }

    def load_state_dict(self, snapshot: dict[str, Any] | None) -> None:
        if not snapshot:
            return
        self.lexicon_limit_per_concept = max(1, int(snapshot.get("lexicon_limit_per_concept", self.lexicon_limit_per_concept)))
        self.top_concepts_per_update = max(1, int(snapshot.get("top_concepts_per_update", self.top_concepts_per_update)))
        self.gap_threshold = max(0.0, float(snapshot.get("gap_threshold", self.gap_threshold)))
        raw_lexicon = snapshot.get("lexicon") or {}
        lexicon: dict[int, list[str]] = {}
        if isinstance(raw_lexicon, dict):
            for raw_key, raw_values in raw_lexicon.items():
                try:
                    idx = int(raw_key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                    continue
                values = _dedupe_strings([str(item) for item in list(raw_values)], limit=self.lexicon_limit_per_concept)
                if values:
                    lexicon[idx] = values
        self.lexicon = lexicon

    @classmethod
    def from_state_dict(
        cls,
        abstraction_layer: AbstractionLayer | None,
        snapshot: dict[str, Any] | None,
    ) -> "GeometricCuriosityController":
        controller = cls(abstraction_layer)
        controller.load_state_dict(snapshot)
        return controller

    def snapshot(self) -> dict[str, Any]:
        payload = self.summary()
        payload["lexicon"] = deepcopy(self.lexicon)
        return payload
