from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .grounding_text import normalize_text as _normalize_text
from .grounding_text import query_focused_text as _query_focused_text
from .grounding_text import tokenize as _tokenize


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _dedupe_strings(values: Sequence[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = " ".join(str(value).split()).strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(item)
        if len(ordered) >= max(1, int(limit)):
            break
    return ordered


def _label_terms(tokens: Sequence[str], query_terms: set[str]) -> list[str]:
    seen: list[str] = []
    for token in tokens:
        if token in query_terms and token not in seen:
            seen.append(token)
        if len(seen) >= 2:
            return seen

    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= 2:
            return seen
    return seen[:1]


def _covered_fragment_match(match: dict[str, Any], matches: Sequence[dict[str, Any]]) -> bool:
    focused_text = _normalize_text(match.get("_focused_text"))
    if not focused_text:
        return False
    query_overlap = int(match.get("_query_overlap", 0))
    expansion_chars = float(match.get("expansion_chars", 0.0))
    for other in matches:
        if other is match:
            continue
        other_text = _normalize_text(other.get("_focused_text"))
        if not other_text or other_text == focused_text or len(other_text) <= len(focused_text):
            continue
        if focused_text not in other_text:
            continue
        if int(other.get("_query_overlap", 0)) < query_overlap:
            continue
        if float(other.get("expansion_chars", 0.0)) < expansion_chars:
            continue
        return True
    return False


def _normalize_signature(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    vector = value.detach().clone().cpu().float().reshape(-1)
    if int(vector.numel()) <= 0 or float(vector.norm().item()) <= 1e-8:
        return None
    return F.normalize(vector, dim=0)


def _resize_signature(value: torch.Tensor | None, *, target_dim: int | None) -> torch.Tensor | None:
    vector = _normalize_signature(value)
    if vector is None or target_dim is None:
        return vector
    dim = int(max(1, target_dim))
    current_dim = int(vector.numel())
    if current_dim == dim:
        return vector
    if current_dim < dim:
        resized = F.pad(vector, (0, dim - current_dim))
    else:
        resized = vector[:dim]
    if float(resized.norm().item()) <= 1e-8:
        return None
    return F.normalize(resized, dim=0)


def _align_signatures(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
    *,
    target_dim: int | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    left_norm = _normalize_signature(left)
    right_norm = _normalize_signature(right)
    if left_norm is None or right_norm is None:
        return left_norm, right_norm
    dim = target_dim
    if dim is None:
        dim = max(int(left_norm.numel()), int(right_norm.numel()))
    return _resize_signature(left_norm, target_dim=dim), _resize_signature(right_norm, target_dim=dim)


def _cosine_similarity(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    left_aligned, right_aligned = _align_signatures(left, right)
    if left_aligned is None or right_aligned is None:
        return 0.0
    return float(torch.dot(left_aligned, right_aligned).item())


def _unit_cosine_similarity(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> float:
    """Cosine similarity for already-normalized CPU vectors."""
    if left is None or right is None:
        return 0.0
    left_dim = int(left.numel())
    right_dim = int(right.numel())
    if left_dim <= 0 or right_dim <= 0:
        return 0.0
    if left_dim == right_dim:
        return float(torch.dot(left, right).item())
    target_dim = max(left_dim, right_dim)
    left_aligned = left if left_dim == target_dim else F.pad(left, (0, target_dim - left_dim))
    right_aligned = right if right_dim == target_dim else F.pad(right, (0, target_dim - right_dim))
    return float(torch.dot(left_aligned, right_aligned).item())


def _blend_centroid(
    current: torch.Tensor | None,
    incoming: torch.Tensor | None,
    *,
    weight: float,
) -> torch.Tensor | None:
    current_aligned, incoming_aligned = _align_signatures(current, incoming)
    if incoming_aligned is None:
        return None if current_aligned is None else current_aligned.clone()
    if current_aligned is None:
        return incoming_aligned.clone()
    return F.normalize(current_aligned * float(max(0.0, weight)) + incoming_aligned, dim=0)


def _weighted_centroid(
    left: torch.Tensor | None,
    *,
    left_weight: float,
    right: torch.Tensor | None,
    right_weight: float,
) -> torch.Tensor | None:
    left_aligned, right_aligned = _align_signatures(left, right)
    if right_aligned is None:
        return None if left_aligned is None else left_aligned.clone()
    if left_aligned is None:
        return right_aligned.clone()
    total = left_aligned * float(max(0.0, left_weight)) + right_aligned * float(max(0.0, right_weight))
    if float(total.norm().item()) <= 1e-8:
        return left_aligned.clone()
    return F.normalize(total, dim=0)


class OnlineSlowFeatureMap:
    """Online slow-feature abstraction over routing signatures."""

    def __init__(
        self,
        *,
        output_dim: int = 8,
        slow_lr: float = 0.05,
        variance_lr: float = 0.02,
        mean_lr: float = 0.05,
        var_lr: float = 0.05,
        eps: float = 1e-6,
    ) -> None:
        self.requested_output_dim = int(max(1, output_dim))
        self.slow_lr = float(max(0.0, slow_lr))
        self.variance_lr = float(max(0.0, variance_lr))
        self.mean_lr = float(max(0.0, mean_lr))
        self.var_lr = float(max(0.0, var_lr))
        self.eps = float(max(1e-8, eps))

        self.input_dim: int | None = None
        self.actual_output_dim = 0
        self.mean: torch.Tensor | None = None
        self.var: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.last_whitened: torch.Tensor | None = None
        self.last_projected: torch.Tensor | None = None
        self.updates = 0

    def _ensure_dim(self, input_dim: int) -> None:
        if self.input_dim == int(input_dim) and self.components is not None:
            return

        dim = int(max(1, input_dim))
        out_dim = int(min(self.requested_output_dim, dim))
        self.input_dim = dim
        self.actual_output_dim = out_dim
        self.mean = torch.zeros(dim, dtype=torch.float32)
        self.var = torch.ones(dim, dtype=torch.float32)
        generator = torch.Generator()
        generator.manual_seed(7919 + dim * 131 + out_dim)
        basis = torch.randn(dim, out_dim, dtype=torch.float32, generator=generator)
        q, _ = torch.linalg.qr(basis, mode="reduced")
        self.components = q.t().contiguous()
        self.last_whitened = None
        self.last_projected = None

    def _orthonormalize(self) -> None:
        if self.components is None or int(self.components.numel()) <= 0:
            return
        q, _ = torch.linalg.qr(self.components.t(), mode="reduced")
        self.components = q.t().contiguous()

    def grow_output_dim(self, *, step: int = 1, max_output_dim: int | None = None) -> bool:
        growth_step = max(1, int(step))
        target_requested = int(self.requested_output_dim + growth_step)
        if max_output_dim is not None:
            target_requested = min(target_requested, max(1, int(max_output_dim)))
        if target_requested <= self.requested_output_dim:
            return False
        previous_requested = int(self.requested_output_dim)
        self.requested_output_dim = target_requested
        if self.input_dim is None or self.components is None:
            return True

        target_output_dim = min(self.requested_output_dim, int(self.input_dim))
        if target_output_dim <= self.actual_output_dim:
            return int(self.requested_output_dim) > previous_requested

        additional = int(target_output_dim - self.actual_output_dim)
        generator = torch.Generator()
        generator.manual_seed(12991 + int(self.input_dim) * 173 + int(self.updates) * 17 + int(target_output_dim))
        existing_basis = self.components.t().contiguous()
        new_rows: list[torch.Tensor] = []
        attempts = 0
        while len(new_rows) < additional and attempts < max(8, 4 * additional):
            candidate = torch.randn(int(self.input_dim), dtype=torch.float32, generator=generator)
            if int(existing_basis.numel()) > 0:
                candidate = candidate - torch.mv(existing_basis, torch.mv(existing_basis.t(), candidate))
            for row in new_rows:
                candidate = candidate - torch.dot(candidate, row) * row
            norm = float(candidate.norm().item())
            attempts += 1
            if norm <= self.eps:
                continue
            new_rows.append(candidate / norm)
        if len(new_rows) < additional:
            return False

        self.components = torch.cat([self.components, torch.stack(new_rows, dim=0)], dim=0)
        self.actual_output_dim = int(self.components.shape[0])
        self.last_projected = None
        self._orthonormalize()
        return True

    def project(
        self,
        signature: torch.Tensor | None,
        *,
        update: bool,
    ) -> tuple[torch.Tensor | None, float]:
        raw = _normalize_signature(signature)
        if raw is None:
            return None, 0.0

        self._ensure_dim(int(raw.numel()))
        assert self.mean is not None
        assert self.var is not None
        assert self.components is not None

        previous_whitened = None if self.last_whitened is None else self.last_whitened.clone()
        previous_projected = None if self.last_projected is None else self.last_projected.clone()

        if update:
            self.mean = (1.0 - self.mean_lr) * self.mean + self.mean_lr * raw
        centered = raw - self.mean
        if update:
            self.var = (1.0 - self.var_lr) * self.var + self.var_lr * (centered * centered)

        whitened = centered / torch.sqrt(self.var + self.eps)
        if float(whitened.norm().item()) <= self.eps:
            whitened = raw.clone()
        else:
            whitened = F.normalize(whitened, dim=0)

        projected = torch.mv(self.components, whitened)

        if update and previous_whitened is not None and int(previous_whitened.numel()) == int(whitened.numel()):
            delta_x = whitened - previous_whitened
            if float(delta_x.norm().item()) > self.eps:
                if previous_projected is None or int(previous_projected.numel()) != int(projected.numel()):
                    delta_z = torch.mv(self.components, delta_x)
                else:
                    delta_z = projected - previous_projected
                self.components = self.components - self.slow_lr * torch.outer(delta_z, delta_x)
                projected = torch.mv(self.components, whitened)

        if update:
            z = projected
            residual = whitened.unsqueeze(0) - z.unsqueeze(1) * self.components
            self.components = self.components + self.variance_lr * z.unsqueeze(1) * residual
            self._orthonormalize()
            projected = torch.mv(self.components, whitened)
            self.last_whitened = whitened.clone()
            self.last_projected = projected.clone()
            self.updates += 1

        projected_norm = _normalize_signature(projected)
        temporal_change = 0.0
        if projected_norm is not None and previous_projected is not None:
            temporal_change = _clamp01(
                0.5 * (1.0 - _cosine_similarity(projected_norm, _normalize_signature(previous_projected)))
            )
        return projected_norm if projected_norm is not None else raw, temporal_change

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "online_sfa_proxy",
            "runtime_role": "maintained_abstraction_layer",
            "requested_output_dim": int(self.requested_output_dim),
            "output_dim": int(self.actual_output_dim),
            "input_dim": None if self.input_dim is None else int(self.input_dim),
            "updates": int(self.updates),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "requested_output_dim": int(self.requested_output_dim),
            "slow_lr": float(self.slow_lr),
            "variance_lr": float(self.variance_lr),
            "mean_lr": float(self.mean_lr),
            "var_lr": float(self.var_lr),
            "eps": float(self.eps),
            "input_dim": None if self.input_dim is None else int(self.input_dim),
            "actual_output_dim": int(self.actual_output_dim),
            "mean": None if self.mean is None else self.mean.tolist(),
            "var": None if self.var is None else self.var.tolist(),
            "components": None if self.components is None else self.components.tolist(),
            "last_whitened": None if self.last_whitened is None else self.last_whitened.tolist(),
            "last_projected": None if self.last_projected is None else self.last_projected.tolist(),
            "updates": int(self.updates),
        }

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self.requested_output_dim = int(payload.get("requested_output_dim", self.requested_output_dim))
        self.slow_lr = float(payload.get("slow_lr", self.slow_lr))
        self.variance_lr = float(payload.get("variance_lr", self.variance_lr))
        self.mean_lr = float(payload.get("mean_lr", self.mean_lr))
        self.var_lr = float(payload.get("var_lr", self.var_lr))
        self.eps = float(payload.get("eps", self.eps))
        input_dim = payload.get("input_dim")
        self.input_dim = None if input_dim is None else int(input_dim)
        self.actual_output_dim = int(payload.get("actual_output_dim", 0))

        def _tensor(values: Any) -> torch.Tensor | None:
            if isinstance(values, list) and values:
                return torch.tensor(values, dtype=torch.float32)
            return None

        self.mean = _tensor(payload.get("mean"))
        self.var = _tensor(payload.get("var"))
        self.components = _tensor(payload.get("components"))
        self.last_whitened = _tensor(payload.get("last_whitened"))
        self.last_projected = _tensor(payload.get("last_projected"))
        self.updates = int(payload.get("updates", 0))
        if self.components is None and self.input_dim is not None:
            self._ensure_dim(self.input_dim)


def summarize_concepts(
    *,
    query_text: str,
    memory_matches: Sequence[dict[str, Any]],
    memory_episodes: Sequence[dict[str, Any]] | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    return ConceptStore().observe(
        query_text=query_text,
        memory_matches=memory_matches,
        memory_episodes=memory_episodes,
        limit=limit,
    )


class ConceptStore:
    def __init__(
        self,
        *,
        merge_similarity: float = 0.80,
        lexical_merge_threshold: float = 0.60,
        lexical_weight: float = 0.12,
        slow_feature_dim: int = 8,
        max_slow_feature_dim: int | None = None,
        min_lexical_overlap_for_merge: float = 0.20,
        min_query_overlap_terms: int = 1,
    ) -> None:
        self._merge_similarity = float(merge_similarity)
        self._lexical_merge_threshold = float(lexical_merge_threshold)
        self._lexical_weight = float(lexical_weight)
        self._min_lexical_overlap_for_merge = float(max(0.0, min_lexical_overlap_for_merge))
        self._min_query_overlap_terms = int(max(0, min_query_overlap_terms))
        self._base_slow_feature_dim = int(max(1, slow_feature_dim))
        self._max_slow_feature_dim = int(
            max(
                self._base_slow_feature_dim,
                max_slow_feature_dim if max_slow_feature_dim is not None else self._base_slow_feature_dim + 4,
            )
        )
        self._entries: dict[str, dict[str, Any]] = {}
        self._observations = 0
        self._episode_index = 0
        self._next_id = 1
        self._slow_features = OnlineSlowFeatureMap(output_dim=self._base_slow_feature_dim)
        self._growth_event_count = 0
        self._prune_event_count = 0
        self._growth_events: list[dict[str, Any]] = []
        self._last_growth_episode = 0
        self._last_prune_episode = 0
        self._normalized_centroid_cache: dict[
            str,
            tuple[torch.Tensor | None, torch.Tensor | None],
        ] = {}

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any] | None) -> ConceptStore:
        store = cls()
        if payload:
            store.load_state_dict(payload)
        return store

    def _memory_signature(self, memory_store: Any, memory_index: Any) -> torch.Tensor | None:
        if memory_store is None:
            return None
        if isinstance(memory_index, Sequence) and not isinstance(memory_index, (str, bytes)):
            signatures = [self._memory_signature(memory_store, value) for value in memory_index]
            valid_signatures = [signature for signature in signatures if signature is not None]
            if not valid_signatures:
                return None
            if len(valid_signatures) == 1:
                return valid_signatures[0]
            target_dim = max(int(signature.numel()) for signature in valid_signatures)
            aligned = [
                resized
                for signature in valid_signatures
                if (resized := _resize_signature(signature, target_dim=target_dim)) is not None
            ]
            if not aligned:
                return None
            return _normalize_signature(torch.stack(aligned, dim=0).mean(dim=0))
        try:
            index = int(memory_index)
        except (TypeError, ValueError):
            return None

        for attr in ("slow_routing_keys", "slow_input_patterns", "slow_buffer"):
            values = list(getattr(memory_store, attr, []) or [])
            if index < 0 or index >= len(values):
                continue
            signature = _normalize_signature(values[index])
            if signature is not None:
                return signature
        return None

    def _new_entry(
        self,
        *,
        tokens: Sequence[str],
        raw_signature: torch.Tensor | None,
        slow_signature: torch.Tensor | None,
    ) -> dict[str, Any]:
        concept_id = f"c{self._next_id}"
        self._next_id += 1
        return {
            "concept_id": concept_id,
            "raw_centroid": None if raw_signature is None else raw_signature.clone(),
            "slow_centroid": None if slow_signature is None else slow_signature.clone(),
            "last_slow_signature": None if slow_signature is None else slow_signature.clone(),
            "score_total": 0.0,
            "observations": 0,
            "match_count_total": 0,
            "memory_indices": [],
            "example_windows": [],
            "term_weights": Counter({token: 1.0 for token in tokens[:4]}),
            "uncertainty_ema": 1.0,
            "drift_ema": 0.0,
            "temporal_coherence_ema": 0.0,
            "abstraction_gain_ema": 0.5,
            "growth_pressure_ema": 0.0,
            "growth_streak": 0,
            "split_bias": 0.0,
            "first_episode": int(self._episode_index),
            "last_episode": int(self._episode_index),
        }

    def _top_terms(self, entry: dict[str, Any], limit: int = 4) -> list[str]:
        return [str(term) for term, _ in entry["term_weights"].most_common(limit)]

    def _normalized_entry_centroids(
        self,
        entry: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        concept_id = str(entry["concept_id"])
        cached = self._normalized_centroid_cache.get(concept_id)
        if cached is not None:
            return cached
        normalized = (
            _normalize_signature(entry.get("raw_centroid")),
            _normalize_signature(entry.get("slow_centroid")),
        )
        self._normalized_centroid_cache[concept_id] = normalized
        return normalized

    def _invalidate_normalized_centroids(self, *concept_ids: str) -> None:
        for concept_id in concept_ids:
            self._normalized_centroid_cache.pop(str(concept_id), None)

    def _concept_uncertainty(self, entry: dict[str, Any]) -> float:
        observations = max(1, int(entry.get("observations", 0)))
        dispersion = _clamp01(float(entry.get("uncertainty_ema", 1.0)))
        sample_term = min(1.0, 1.0 / math.sqrt(float(observations)))
        instability = 1.0 - _clamp01(float(entry.get("temporal_coherence_ema", 0.0)))
        return _clamp01(0.45 * dispersion + 0.25 * sample_term + 0.30 * instability)

    def _concept_label(self, entry: dict[str, Any], query_terms: set[str]) -> str:
        top_terms = self._top_terms(entry, limit=6)
        label_terms = _label_terms(top_terms, query_terms)
        if not label_terms:
            return entry["concept_id"]
        return " / ".join(label_terms)

    def _concept_support(self, entry: dict[str, Any]) -> float:
        observations = int(entry.get("observations", 0))
        temporal_coherence = _clamp01(float(entry.get("temporal_coherence_ema", 0.0)))
        abstraction_gain = _clamp01(float(entry.get("abstraction_gain_ema", 0.5)))
        uncertainty = self._concept_uncertainty(entry)
        return _clamp01(
            0.45 * min(1.0, float(observations) / 4.0)
            + 0.25 * temporal_coherence
            + 0.20 * abstraction_gain
            + 0.10 * (1.0 - uncertainty)
        )

    def _concept_weakness(self, entry: dict[str, Any]) -> float:
        uncertainty = self._concept_uncertainty(entry)
        drift = _clamp01(float(entry.get("drift_ema", 0.0)))
        instability = 1.0 - _clamp01(float(entry.get("temporal_coherence_ema", 0.0)))
        abstraction_gain = _clamp01(float(entry.get("abstraction_gain_ema", 0.5)))
        return _clamp01(
            0.40 * uncertainty
            + 0.25 * instability
            + 0.20 * abstraction_gain
            + 0.15 * drift
        )

    def _concept_growth_pressure(self, entry: dict[str, Any]) -> float:
        observations = int(entry.get("observations", 0))
        persistence = min(1.0, float(observations) / 4.0)
        weakness = self._concept_weakness(entry)
        abstraction_gain = _clamp01(float(entry.get("abstraction_gain_ema", 0.5)))
        support = self._concept_support(entry)
        return _clamp01(
            0.35 * weakness
            + 0.25 * abstraction_gain
            + 0.20 * support
            + 0.20 * persistence
        )

    def _refresh_entry_structure(self, entry: dict[str, Any]) -> None:
        observations = int(entry.get("observations", 0))
        growth_pressure = self._concept_growth_pressure(entry)
        previous_growth_pressure = float(entry.get("growth_pressure_ema", 0.0))
        entry["growth_pressure_ema"] = float(
            growth_pressure
            if observations <= 1
            else 0.70 * previous_growth_pressure + 0.30 * growth_pressure
        )
        if observations >= 4 and float(entry["growth_pressure_ema"]) >= 0.45:
            entry["growth_streak"] = int(entry.get("growth_streak", 0)) + 1
        else:
            entry["growth_streak"] = 0
        entry["split_bias"] = _clamp01(
            0.70 * float(entry["growth_pressure_ema"])
            + 0.30 * self._concept_weakness(entry)
        )

    def _structural_growth_report(self) -> dict[str, Any]:
        active_growth_concepts = [
            {
                "concept_id": str(entry["concept_id"]),
                "label": self._concept_label(entry, set()),
                "growth_pressure": float(entry.get("growth_pressure_ema", 0.0)),
                "split_bias": _clamp01(float(entry.get("split_bias", 0.0))),
                "top_terms": self._top_terms(entry),
                "observations": int(entry.get("observations", 0)),
            }
            for entry in sorted(
                self._entries.values(),
                key=lambda item: (
                    float(item.get("growth_pressure_ema", 0.0)),
                    float(item.get("split_bias", 0.0)),
                    int(item.get("observations", 0)),
                ),
                reverse=True,
            )
            if float(entry.get("growth_pressure_ema", 0.0)) >= 0.40
        ]
        growth_ready = bool(
            active_growth_concepts
            and int(self._slow_features.requested_output_dim) < int(self._max_slow_feature_dim)
        )
        return {
            "base_output_dim": int(self._base_slow_feature_dim),
            "max_output_dim": int(self._max_slow_feature_dim),
            "requested_output_dim": int(self._slow_features.requested_output_dim),
            "current_output_dim": int(self._slow_features.actual_output_dim),
            "expansion_events": int(self._growth_event_count),
            "prune_events": int(self._prune_event_count),
            "growth_ready": growth_ready,
            "active_growth_concepts": active_growth_concepts[:4],
            "recent_events": [dict(item) for item in self._growth_events[-6:]],
        }

    def _assignment_score(
        self,
        entry: dict[str, Any],
        token_set: set[str],
        raw_unit_signature: torch.Tensor | None,
        slow_unit_signature: torch.Tensor | None,
    ) -> tuple[float, float, float, float]:
        concept_terms = set(self._top_terms(entry, limit=8))
        lexical_overlap = 0.0 if not token_set else float(len(token_set & concept_terms) / max(1, len(token_set)))
        raw_reference, slow_reference = self._normalized_entry_centroids(entry)
        if slow_reference is None:
            slow_reference = raw_reference
        slow_query = (
            slow_unit_signature
            if slow_unit_signature is not None
            else raw_unit_signature
        )
        slow_similarity = _unit_cosine_similarity(slow_query, slow_reference)
        raw_similarity = _unit_cosine_similarity(raw_unit_signature, raw_reference)
        score = float(0.70 * slow_similarity + 0.18 * raw_similarity + self._lexical_weight * lexical_overlap)
        return score, slow_similarity, raw_similarity, lexical_overlap

    def _assign_concept(
        self,
        *,
        tokens: Sequence[str],
        raw_signature: torch.Tensor | None,
        slow_signature: torch.Tensor | None,
    ) -> str:
        token_set = set(tokens)
        raw_unit_signature = _normalize_signature(raw_signature)
        slow_unit_signature = _normalize_signature(slow_signature)
        best_id: str | None = None
        best_score = -1.0
        best_slow = 0.0
        best_raw = 0.0
        best_lexical = 0.0

        for concept_id, entry in self._entries.items():
            score, slow_similarity, raw_similarity, lexical_overlap = self._assignment_score(
                entry,
                token_set,
                raw_unit_signature,
                slow_unit_signature,
            )
            if score > best_score:
                best_id = concept_id
                best_score = score
                best_slow = slow_similarity
                best_raw = raw_similarity
                best_lexical = lexical_overlap

        if best_id is not None:
            best_entry = self._entries[best_id]
            split_bias = _clamp01(float(best_entry.get("split_bias", 0.0)))
            merge_similarity = min(0.98, self._merge_similarity + 0.12 * split_bias)
            raw_merge_similarity = min(0.99, self._merge_similarity + 0.05 + 0.12 * split_bias)
            lexical_merge_threshold = min(0.98, self._lexical_merge_threshold + 0.10 * split_bias)
            min_lexical_overlap = min(1.0, self._min_lexical_overlap_for_merge + 0.10 * split_bias)
            if (
                best_slow >= merge_similarity
                or best_raw >= raw_merge_similarity
                or best_lexical >= lexical_merge_threshold
            ) and best_lexical >= min_lexical_overlap:
                return best_id

        entry = self._new_entry(
            tokens=tokens,
            raw_signature=raw_signature,
            slow_signature=slow_signature,
        )
        concept_id = str(entry["concept_id"])
        self._entries[concept_id] = entry
        self._invalidate_normalized_centroids(concept_id)
        return concept_id

    def _update_entry(
        self,
        entry: dict[str, Any],
        *,
        tokens: Sequence[str],
        raw_signature: torch.Tensor | None,
        slow_signature: torch.Tensor | None,
        temporal_change: float,
        memory_index: int | None,
        raw_window: str,
        score: float,
    ) -> None:
        observations_before = int(entry.get("observations", 0))
        raw_before = _normalize_signature(entry.get("raw_centroid"))
        slow_before = _normalize_signature(entry.get("slow_centroid"))
        last_slow = _normalize_signature(entry.get("last_slow_signature"))

        dispersion = 1.0
        drift = 0.0
        abstraction_gain = 0.5
        temporal_coherence = _clamp01(1.0 - float(temporal_change))

        if raw_signature is not None:
            entry["raw_centroid"] = _blend_centroid(
                raw_before,
                raw_signature,
                weight=float(observations_before),
            )

        if slow_signature is not None:
            if slow_before is None:
                entry["slow_centroid"] = slow_signature.clone()
                dispersion = 0.0
                abstraction_gain = 0.5
            else:
                slow_similarity = _cosine_similarity(slow_signature, slow_before)
                dispersion = _clamp01(0.5 * (1.0 - slow_similarity))
                entry["slow_centroid"] = _blend_centroid(
                    slow_before,
                    slow_signature,
                    weight=float(observations_before),
                )
                drift = _clamp01(
                    0.5 * (1.0 - _cosine_similarity(_normalize_signature(entry.get("slow_centroid")), slow_before))
                )
                raw_similarity = _cosine_similarity(raw_signature, raw_before)
                abstraction_gain = _clamp01(0.5 + 0.5 * (slow_similarity - raw_similarity))

            if last_slow is not None:
                local_temporal_delta = _clamp01(0.5 * (1.0 - _cosine_similarity(slow_signature, last_slow)))
                temporal_coherence = _clamp01(1.0 - max(float(temporal_change), local_temporal_delta))
            entry["last_slow_signature"] = slow_signature.clone()

        entry["score_total"] += float(score)
        entry["observations"] = observations_before + 1
        entry["match_count_total"] += 1
        entry["last_episode"] = int(self._episode_index)

        entry["uncertainty_ema"] = float(
            dispersion
            if observations_before <= 0
            else 0.70 * float(entry.get("uncertainty_ema", 1.0)) + 0.30 * dispersion
        )
        entry["drift_ema"] = float(
            drift
            if observations_before <= 0
            else 0.80 * float(entry.get("drift_ema", 0.0)) + 0.20 * drift
        )
        entry["temporal_coherence_ema"] = float(
            temporal_coherence
            if observations_before <= 0
            else 0.75 * float(entry.get("temporal_coherence_ema", 0.0)) + 0.25 * temporal_coherence
        )
        entry["abstraction_gain_ema"] = float(
            abstraction_gain
            if observations_before <= 0
            else 0.80 * float(entry.get("abstraction_gain_ema", 0.5)) + 0.20 * abstraction_gain
        )

        if memory_index is not None and memory_index not in entry["memory_indices"]:
            entry["memory_indices"].append(int(memory_index))
            entry["memory_indices"] = entry["memory_indices"][-8:]

        if raw_window and raw_window not in entry["example_windows"]:
            entry["example_windows"].append(raw_window)
            entry["example_windows"] = entry["example_windows"][-3:]

        rank_gain = 1.0 + 0.5 * temporal_coherence + 0.25 * abstraction_gain
        for rank, token in enumerate(tokens[:8]):
            entry["term_weights"][str(token)] += rank_gain * float(score) / max(1.0, float(rank + 1))
        self._refresh_entry_structure(entry)
        self._invalidate_normalized_centroids(str(entry["concept_id"]))

    def _concept_payload(
        self,
        entry: dict[str, Any],
        *,
        query_terms: set[str],
        score: float,
        match_count: int,
        memory_indices: list[int],
        example_windows: list[str],
    ) -> dict[str, Any]:
        first_episode = int(entry.get("first_episode", self._episode_index))
        last_episode = int(entry.get("last_episode", first_episode))
        return {
            "concept_id": str(entry["concept_id"]),
            "label": self._concept_label(entry, query_terms),
            "score": float(score),
            "match_count": int(match_count),
            "observations": int(entry.get("observations", 0)),
            "uncertainty": self._concept_uncertainty(entry),
            "drift": _clamp01(float(entry.get("drift_ema", 0.0))),
            "temporal_coherence": _clamp01(float(entry.get("temporal_coherence_ema", 0.0))),
            "abstraction_gain": _clamp01(float(entry.get("abstraction_gain_ema", 0.5))),
            "growth_pressure": _clamp01(float(entry.get("growth_pressure_ema", 0.0))),
            "split_bias": _clamp01(float(entry.get("split_bias", 0.0))),
            "episode_span": int(max(1, last_episode - first_episode + 1)),
            "memory_indices": memory_indices[:4],
            "example_windows": example_windows[:2],
            "top_terms": self._top_terms(entry),
        }

    def _merge_entries(self, *, keeper_id: str, donor_id: str) -> None:
        keeper = self._entries[keeper_id]
        donor = self._entries[donor_id]
        keeper_observations = int(keeper.get("observations", 0))
        donor_observations = int(donor.get("observations", 0))
        total_observations = max(1, keeper_observations + donor_observations)

        keeper["raw_centroid"] = _weighted_centroid(
            _normalize_signature(keeper.get("raw_centroid")),
            left_weight=float(keeper_observations),
            right=_normalize_signature(donor.get("raw_centroid")),
            right_weight=float(donor_observations),
        )
        keeper["slow_centroid"] = _weighted_centroid(
            _normalize_signature(keeper.get("slow_centroid")),
            left_weight=float(keeper_observations),
            right=_normalize_signature(donor.get("slow_centroid")),
            right_weight=float(donor_observations),
        )
        donor_last_slow = _normalize_signature(donor.get("last_slow_signature"))
        keeper_last_slow = _normalize_signature(keeper.get("last_slow_signature"))
        keeper["last_slow_signature"] = donor_last_slow if donor_last_slow is not None else keeper_last_slow
        keeper["score_total"] = float(keeper.get("score_total", 0.0)) + float(donor.get("score_total", 0.0))
        keeper["observations"] = total_observations
        keeper["match_count_total"] = int(keeper.get("match_count_total", 0)) + int(donor.get("match_count_total", 0))
        keeper["memory_indices"] = list(
            dict.fromkeys([*list(keeper.get("memory_indices") or []), *list(donor.get("memory_indices") or [])])
        )[-8:]
        keeper["example_windows"] = list(
            dict.fromkeys([*list(keeper.get("example_windows") or []), *list(donor.get("example_windows") or [])])
        )[-3:]
        keeper["term_weights"].update(Counter(donor.get("term_weights") or {}))
        for field in (
            "uncertainty_ema",
            "drift_ema",
            "temporal_coherence_ema",
            "abstraction_gain_ema",
            "growth_pressure_ema",
            "split_bias",
        ):
            keeper[field] = float(
                (
                    keeper_observations * float(keeper.get(field, 0.0))
                    + donor_observations * float(donor.get(field, 0.0))
                )
                / float(total_observations)
            )
        keeper["growth_streak"] = max(int(keeper.get("growth_streak", 0)), int(donor.get("growth_streak", 0)))
        keeper["first_episode"] = min(int(keeper.get("first_episode", 0)), int(donor.get("first_episode", 0)))
        keeper["last_episode"] = max(int(keeper.get("last_episode", 0)), int(donor.get("last_episode", 0)))
        self._refresh_entry_structure(keeper)
        self._invalidate_normalized_centroids(keeper_id, donor_id)

    def refresh_structural_capacity(self, *, grouped: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        growth_candidates = [
            entry
            for entry in self._entries.values()
            if int(entry.get("observations", 0)) >= 4
            and int(entry.get("growth_streak", 0)) >= 2
            and float(entry.get("growth_pressure_ema", 0.0)) >= 0.45
        ]
        growth_candidates.sort(
            key=lambda entry: (
                float(entry.get("growth_pressure_ema", 0.0)),
                float(entry.get("split_bias", 0.0)),
                int(entry.get("observations", 0)),
            ),
            reverse=True,
        )
        if (
            growth_candidates
            and int(self._episode_index - self._last_growth_episode) >= 4
            and int(self._slow_features.requested_output_dim) < int(self._max_slow_feature_dim)
        ):
            previous_requested = int(self._slow_features.requested_output_dim)
            if self._slow_features.grow_output_dim(max_output_dim=self._max_slow_feature_dim):
                self._growth_event_count += 1
                self._last_growth_episode = int(self._episode_index)
                self._growth_events.append(
                    {
                        "type": "expand",
                        "episode": int(self._episode_index),
                        "requested_output_dim_before": previous_requested,
                        "requested_output_dim_after": int(self._slow_features.requested_output_dim),
                        "concept_ids": [str(entry["concept_id"]) for entry in growth_candidates[:2]],
                    }
                )
                self._growth_events = self._growth_events[-8:]
                for entry in growth_candidates[:2]:
                    entry["growth_streak"] = 0

        if len(self._entries) >= 3 and int(self._episode_index - self._last_prune_episode) >= 4:
            best_pair: tuple[str, str] | None = None
            best_score = 0.0
            concept_ids = list(self._entries.keys())
            for left_id in concept_ids:
                left = self._entries[left_id]
                left_terms = set(self._top_terms(left, limit=6))
                left_raw_signature, left_signature = self._normalized_entry_centroids(left)
                if left_signature is None:
                    left_signature = left_raw_signature
                if left_signature is None:
                    continue
                for right_id in concept_ids:
                    if left_id == right_id:
                        continue
                    right = self._entries[right_id]
                    if int(right.get("observations", 0)) > 2:
                        continue
                    if float(right.get("growth_pressure_ema", 0.0)) > 0.40:
                        continue
                    right_terms = set(self._top_terms(right, limit=6))
                    lexical_overlap = 0.0 if not left_terms or not right_terms else float(
                        len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))
                    )
                    if lexical_overlap < 0.75:
                        continue
                    right_raw_signature, right_signature = self._normalized_entry_centroids(right)
                    if right_signature is None:
                        right_signature = right_raw_signature
                    similarity = _unit_cosine_similarity(left_signature, right_signature)
                    if similarity < 0.97:
                        continue
                    score = 0.65 * similarity + 0.35 * lexical_overlap
                    if score > best_score:
                        best_score = score
                        best_pair = (left_id, right_id)
            if best_pair is not None:
                keeper_id, donor_id = best_pair
                self._merge_entries(keeper_id=keeper_id, donor_id=donor_id)
                if grouped is not None and donor_id in grouped:
                    donor_group = grouped.pop(donor_id)
                    keeper_group = grouped.setdefault(
                        keeper_id,
                        {"score": 0.0, "match_count": 0, "memory_indices": [], "example_windows": []},
                    )
                    keeper_group["score"] += float(donor_group.get("score", 0.0))
                    keeper_group["match_count"] += int(donor_group.get("match_count", 0))
                    keeper_group["memory_indices"] = list(
                        dict.fromkeys([*list(keeper_group.get("memory_indices", [])), *list(donor_group.get("memory_indices", []))])
                    )
                    keeper_group["example_windows"] = list(
                        dict.fromkeys([*list(keeper_group.get("example_windows", [])), *list(donor_group.get("example_windows", []))])
                    )
                del self._entries[donor_id]
                self._invalidate_normalized_centroids(donor_id)
                self._prune_event_count += 1
                self._last_prune_episode = int(self._episode_index)
                self._growth_events.append(
                    {
                        "type": "prune",
                        "episode": int(self._episode_index),
                        "keeper_id": keeper_id,
                        "donor_id": donor_id,
                    }
                )
                self._growth_events = self._growth_events[-8:]

        return self._structural_growth_report()

    def observe(
        self,
        *,
        query_text: str,
        memory_matches: Sequence[dict[str, Any]],
        memory_episodes: Sequence[dict[str, Any]] | None = None,
        memory_store: Any | None = None,
        limit: int = 6,
    ) -> dict[str, Any]:
        query_terms = set(_tokenize(query_text))
        grouped: dict[str, dict[str, Any]] = {}
        self._observations += 1
        self._episode_index += 1

        evidence_sources = list(memory_episodes or memory_matches)
        prepared_matches: list[dict[str, Any]] = []
        for match in evidence_sources:
            source_text = _normalize_text(match.get("text") or match.get("raw_window"))
            focused_text = _query_focused_text(source_text, query_terms)
            if not focused_text:
                continue

            tokens = _tokenize(focused_text)
            if not tokens:
                continue

            complete_sentence = int(focused_text.endswith((".", "!", "?")))
            clipped_overlap = int(
                bool(
                    _normalize_text(match.get("raw_window"))
                    and not complete_sentence
                    and focused_text.lower() == _normalize_text(match.get("raw_window")).lower()
                )
            )
            prepared_match = dict(match)
            prepared_match["_focused_text"] = focused_text
            prepared_match["_tokens"] = tokens
            prepared_match["_query_overlap"] = len(query_terms & set(tokens))
            prepared_match["_complete_sentence"] = complete_sentence
            prepared_match["_clipped_overlap"] = clipped_overlap
            prepared_matches.append(prepared_match)

        overlapping_matches = [
            match
            for match in prepared_matches
            if int(match.get("_query_overlap", 0)) >= self._min_query_overlap_terms
        ]
        if overlapping_matches:
            prepared_matches = overlapping_matches

        prepared_matches.sort(
            key=lambda match: (
                int(match.get("_query_overlap", 0)),
                int(match.get("_complete_sentence", 0)),
                -int(match.get("_clipped_overlap", 1)),
                float(match.get("similarity", 0.0)),
                float(match.get("importance", 0.0)),
                -float(match.get("age_tokens", 0.0)),
            ),
            reverse=True,
        )
        filtered_matches = [
            match
            for match in prepared_matches
            if not _covered_fragment_match(match, prepared_matches)
        ]
        if filtered_matches:
            prepared_matches = filtered_matches

        for match in prepared_matches:
            raw_window = _normalize_text(match.get("_focused_text") or match.get("text") or match.get("raw_window"))
            tokens = list(match.get("_tokens") or ())

            memory_index = match.get("memory_index")
            try:
                normalized_index = None if memory_index is None else int(memory_index)
            except (TypeError, ValueError):
                normalized_index = None

            signature_reference: Any = match.get("memory_indices")
            if not isinstance(signature_reference, Sequence) or isinstance(signature_reference, (str, bytes)):
                signature_reference = normalized_index

            raw_signature = self._memory_signature(memory_store, signature_reference)
            slow_signature, temporal_change = self._slow_features.project(raw_signature, update=raw_signature is not None)
            concept_id = self._assign_concept(
                tokens=tokens,
                raw_signature=raw_signature,
                slow_signature=slow_signature,
            )
            entry = self._entries[concept_id]

            score = float(match.get("similarity", 0.0))
            score += 0.05 * min(10.0, float(match.get("importance", 0.0)))
            score += 0.03 * max(0.0, float(match.get("capture_tag", match.get("tag_strength", 0.0))))
            score += 0.02 * max(0.0, float(match.get("capture_strength", 0.0)))
            score += 0.02 * max(0.0, float(match.get("consolidation_level", 0.0)))
            score += 0.02 * max(0.0, 1.0 - float(temporal_change))

            self._update_entry(
                entry,
                tokens=tokens,
                raw_signature=raw_signature,
                slow_signature=slow_signature,
                temporal_change=temporal_change,
                memory_index=normalized_index,
                raw_window=raw_window,
                score=score,
            )

            current = grouped.setdefault(
                concept_id,
                {
                    "score": 0.0,
                    "match_count": 0,
                    "memory_indices": [],
                    "example_windows": [],
                },
            )
            current["score"] += float(score)
            current["match_count"] += 1
            if normalized_index is not None and normalized_index not in current["memory_indices"]:
                current["memory_indices"].append(normalized_index)
            if raw_window not in current["example_windows"]:
                current["example_windows"].append(raw_window)

        structural_growth = self.refresh_structural_capacity(grouped=grouped)
        concepts = [
            self._concept_payload(
                self._entries[concept_id],
                query_terms=query_terms,
                score=float(group["score"]),
                match_count=int(group["match_count"]),
                memory_indices=list(group["memory_indices"]),
                example_windows=list(group["example_windows"]),
            )
            for concept_id, group in grouped.items()
        ]
        concepts.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["temporal_coherence"]),
                -float(item["uncertainty"]),
                int(item["match_count"]),
            ),
            reverse=True,
        )
        return {
            "concept_mode": "slow_feature_concept_memory",
            "query_terms": sorted(query_terms),
            "concept_count": int(len(concepts)),
            "source_memory_count": int(len(evidence_sources)),
            "abstraction": self._slow_features.summary(),
            "growth": structural_growth,
            "focus_plan": self.focus_plan(
                query_text=query_text,
                limit=limit,
                min_observations=1,
            ),
            "concepts": concepts[: max(1, int(limit))],
        }

    def focus_plan(
        self,
        *,
        query_text: str | None = None,
        limit: int = 4,
        min_observations: int = 2,
    ) -> dict[str, Any] | None:
        ranked: list[dict[str, Any]] = []
        minimum_observations = max(1, int(min_observations))
        requested_query_terms = _tokenize(query_text or "")
        requested_query_term_set = set(requested_query_terms)
        for entry in self._entries.values():
            observations = int(entry.get("observations", 0))
            if observations < minimum_observations:
                continue
            top_terms = self._top_terms(entry)
            if not top_terms:
                continue
            label = self._concept_label(entry, requested_query_term_set if requested_query_term_set else set())
            concept_terms = _dedupe_strings(
                [*top_terms, *_tokenize(label)],
                limit=6,
            )
            query_overlap_terms = [
                term
                for term in concept_terms
                if term in requested_query_term_set
            ]
            if requested_query_term_set and not query_overlap_terms:
                continue
            query_overlap = (
                0.0
                if not requested_query_term_set
                else float(len(query_overlap_terms)) / max(1.0, float(len(requested_query_term_set)))
            )
            uncertainty = self._concept_uncertainty(entry)
            drift = _clamp01(float(entry.get("drift_ema", 0.0)))
            temporal_coherence = _clamp01(float(entry.get("temporal_coherence_ema", 0.0)))
            abstraction_gain = _clamp01(float(entry.get("abstraction_gain_ema", 0.5)))
            support = self._concept_support(entry)
            weakness = self._concept_weakness(entry)
            focus_weight = _clamp01(
                (
                    0.45 * query_overlap
                    + 0.25 * weakness
                    + 0.15 * abstraction_gain
                    + 0.15 * support
                )
                if requested_query_term_set
                else (
                    0.45 * weakness
                    + 0.20 * abstraction_gain
                    + 0.20 * support
                    + 0.15 * temporal_coherence
                )
            )
            retrieval_terms = _dedupe_strings(
                [*requested_query_terms, *concept_terms],
                limit=4,
            ) if requested_query_terms else top_terms[:3]
            retrieval_query = " ".join(retrieval_terms).strip()
            if not retrieval_query:
                continue
            if requested_query_terms:
                anchor_text = " ".join(requested_query_terms[:2]).strip() or retrieval_query
                target_terms = [
                    term
                    for term in concept_terms
                    if term not in requested_query_term_set
                ]
                target_text = " ".join(target_terms[:2]).strip() or " ".join(top_terms[:2]).strip() or retrieval_query
                follow_up_question = f"What grounded evidence connects {anchor_text} to {target_text}?"
            else:
                follow_up_question = (
                    f"What grounded evidence would stabilize the concept around {retrieval_query}?"
                )
            ranked.append(
                {
                    "label": label,
                    "top_terms": top_terms[:4],
                    "concept_terms": concept_terms,
                    "weakness": weakness,
                    "uncertainty": uncertainty,
                    "drift": drift,
                    "temporal_coherence": temporal_coherence,
                    "abstraction_gain": abstraction_gain,
                    "support": support,
                    "query_overlap": query_overlap,
                    "query_overlap_terms": query_overlap_terms,
                    "focus_weight": focus_weight,
                    "match_count": observations,
                    "memory_indices": [int(value) for value in list(entry.get("memory_indices") or [])][:4],
                    "retrieval_query": retrieval_query,
                    "follow_up_question": follow_up_question,
                }
            )
        if not ranked:
            return None

        ranked.sort(
            key=lambda item: (
                float(item["query_overlap"]),
                float(item["focus_weight"]),
                float(item["weakness"]),
                float(item["abstraction_gain"]),
                float(item["match_count"]),
            ),
            reverse=True,
        )

        gap_weights: Counter[str] = Counter()
        query_terms: list[str] = list(requested_query_terms)
        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concepts: list[dict[str, Any]] = []
        memory_priority: dict[str, float] = {}
        for item in ranked[: max(1, int(limit))]:
            top_terms = list(item["top_terms"])
            focus_terms = list(item["concept_terms"]) if requested_query_terms else top_terms[:3]
            weak_concepts.append(
                {
                    "label": str(item["label"]),
                    "weakness": float(item["weakness"]),
                    "uncertainty": float(item["uncertainty"]),
                    "drift": float(item["drift"]),
                    "abstraction_gain": float(item["abstraction_gain"]),
                    "support": float(item["support"]),
                    "query_overlap": float(item["query_overlap"]),
                    "query_overlap_terms": list(item["query_overlap_terms"]),
                    "top_terms": top_terms,
                    "match_count": int(item["match_count"]),
                    "memory_indices": list(item["memory_indices"]),
                    "focus_weight": float(item["focus_weight"]),
                }
            )
            query_terms.extend(focus_terms[:3])
            retrieval_queries.append(str(item["retrieval_query"]))
            follow_up_questions.append(str(item["follow_up_question"]))
            for memory_index in list(item["memory_indices"]):
                key = str(memory_index)
                memory_priority[key] = max(
                    float(memory_priority.get(key, 0.0)),
                    float(item["focus_weight"]),
                )
            gap_base = float(item["focus_weight"] if requested_query_terms else item["weakness"])
            for rank, term in enumerate(focus_terms[:3]):
                gap_weights[str(term)] += gap_base / float(rank + 1)

        return {
            "planner_mode": "concept_store_abstraction_focus",
            "query_terms": _dedupe_strings(query_terms, limit=8),
            "focus_terms": _dedupe_strings(query_terms, limit=8),
            "unsupported_terms": [],
            "gap_terms": [
                {"term": str(term), "weight": float(weight)}
                for term, weight in gap_weights.most_common(8)
            ],
            "retrieval_queries": _dedupe_strings(retrieval_queries, limit=4),
            "follow_up_questions": _dedupe_strings(follow_up_questions, limit=4),
            "weak_concepts": weak_concepts,
            "memory_priority": memory_priority,
            "structural_growth": self._structural_growth_report(),
        }

    def snapshot(self, limit: int = 8) -> dict[str, Any]:
        concepts = [
            self._concept_payload(
                entry,
                query_terms=set(),
                score=float(entry.get("score_total", 0.0)),
                match_count=int(entry.get("match_count_total", 0)),
                memory_indices=list(entry.get("memory_indices", [])),
                example_windows=list(entry.get("example_windows", [])),
            )
            for entry in self._entries.values()
        ]
        concepts.sort(
            key=lambda item: (
                float(item["score"]),
                float(item["temporal_coherence"]),
                int(item["observations"]),
            ),
            reverse=True,
        )
        return {
            "concept_mode": "slow_feature_concept_memory",
            "observations": int(self._observations),
            "concept_count": int(len(self._entries)),
            "abstraction": self._slow_features.summary(),
            "growth": self._structural_growth_report(),
            "focus_plan": self.focus_plan(limit=min(limit, 4)),
            "top_concepts": concepts[: max(1, int(limit))],
        }

    def state_dict(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for entry in self._entries.values():
            raw_centroid = _normalize_signature(entry.get("raw_centroid"))
            slow_centroid = _normalize_signature(entry.get("slow_centroid"))
            last_slow = _normalize_signature(entry.get("last_slow_signature"))
            entries.append(
                {
                    "concept_id": str(entry["concept_id"]),
                    "raw_centroid": None if raw_centroid is None else raw_centroid.tolist(),
                    "slow_centroid": None if slow_centroid is None else slow_centroid.tolist(),
                    "last_slow_signature": None if last_slow is None else last_slow.tolist(),
                    "score_total": float(entry.get("score_total", 0.0)),
                    "observations": int(entry.get("observations", 0)),
                    "match_count_total": int(entry.get("match_count_total", 0)),
                    "memory_indices": [int(value) for value in entry.get("memory_indices", [])],
                    "example_windows": [str(value) for value in entry.get("example_windows", [])],
                    "term_weights": {str(key): float(value) for key, value in entry["term_weights"].items()},
                    "uncertainty_ema": float(entry.get("uncertainty_ema", 1.0)),
                    "drift_ema": float(entry.get("drift_ema", 0.0)),
                    "temporal_coherence_ema": float(entry.get("temporal_coherence_ema", 0.0)),
                    "abstraction_gain_ema": float(entry.get("abstraction_gain_ema", 0.5)),
                    "growth_pressure_ema": float(entry.get("growth_pressure_ema", 0.0)),
                    "growth_streak": int(entry.get("growth_streak", 0)),
                    "split_bias": float(entry.get("split_bias", 0.0)),
                    "first_episode": int(entry.get("first_episode", 0)),
                    "last_episode": int(entry.get("last_episode", 0)),
                }
            )
        return {
            "concept_mode": "slow_feature_concept_memory",
            "observations": int(self._observations),
            "episode_index": int(self._episode_index),
            "next_id": int(self._next_id),
            "base_slow_feature_dim": int(self._base_slow_feature_dim),
            "max_slow_feature_dim": int(self._max_slow_feature_dim),
            "growth_event_count": int(self._growth_event_count),
            "prune_event_count": int(self._prune_event_count),
            "growth_events": [dict(item) for item in self._growth_events],
            "last_growth_episode": int(self._last_growth_episode),
            "last_prune_episode": int(self._last_prune_episode),
            "slow_features": self._slow_features.state_dict(),
            "entries": entries,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._entries = {}
        self._normalized_centroid_cache = {}
        self._observations = int(payload.get("observations", 0))
        self._episode_index = int(payload.get("episode_index", 0))
        self._next_id = int(payload.get("next_id", 1))
        self._base_slow_feature_dim = int(payload.get("base_slow_feature_dim", self._base_slow_feature_dim))
        self._max_slow_feature_dim = int(payload.get("max_slow_feature_dim", self._max_slow_feature_dim))
        self._growth_event_count = int(payload.get("growth_event_count", 0))
        self._prune_event_count = int(payload.get("prune_event_count", 0))
        self._growth_events = [dict(item) for item in list(payload.get("growth_events", []) or []) if isinstance(item, dict)][-8:]
        self._last_growth_episode = int(payload.get("last_growth_episode", 0))
        self._last_prune_episode = int(payload.get("last_prune_episode", 0))
        self._slow_features = OnlineSlowFeatureMap()
        self._slow_features.load_state_dict(dict(payload.get("slow_features", {})))

        for item in payload.get("entries", []):
            concept_id = str(item.get("concept_id", f"c{self._next_id}"))
            if concept_id.startswith("c"):
                try:
                    self._next_id = max(self._next_id, int(concept_id[1:]) + 1)
                except ValueError:
                    pass

            def _tensor(values: Any) -> torch.Tensor | None:
                if isinstance(values, list) and values:
                    return _normalize_signature(torch.tensor(values, dtype=torch.float32))
                return None

            raw_centroid = _tensor(item.get("raw_centroid", item.get("centroid")))
            slow_centroid = _tensor(item.get("slow_centroid", item.get("centroid")))
            last_slow = _tensor(item.get("last_slow_signature", item.get("slow_centroid", item.get("centroid"))))
            restored_slow_centroid = (
                slow_centroid if slow_centroid is not None else raw_centroid
            )
            self._entries[concept_id] = {
                "concept_id": concept_id,
                "raw_centroid": raw_centroid,
                "slow_centroid": restored_slow_centroid,
                "last_slow_signature": (
                    last_slow
                    if last_slow is not None
                    else restored_slow_centroid
                ),
                "score_total": float(item.get("score_total", 0.0)),
                "observations": int(item.get("observations", 0)),
                "match_count_total": int(item.get("match_count_total", 0)),
                "memory_indices": [int(value) for value in item.get("memory_indices", [])],
                "example_windows": [str(value) for value in item.get("example_windows", [])],
                "term_weights": Counter({str(key): float(value) for key, value in dict(item.get("term_weights", {})).items()}),
                "uncertainty_ema": float(item.get("uncertainty_ema", 1.0)),
                "drift_ema": float(item.get("drift_ema", 0.0)),
                "temporal_coherence_ema": float(item.get("temporal_coherence_ema", 0.0)),
                "abstraction_gain_ema": float(item.get("abstraction_gain_ema", 0.5)),
                "growth_pressure_ema": float(item.get("growth_pressure_ema", 0.0)),
                "growth_streak": int(item.get("growth_streak", 0)),
                "split_bias": float(item.get("split_bias", 0.0)),
                "first_episode": int(item.get("first_episode", 0)),
                "last_episode": int(item.get("last_episode", item.get("first_episode", 0))),
            }
            self._normalized_centroid_cache[concept_id] = (
                raw_centroid,
                restored_slow_centroid,
            )
