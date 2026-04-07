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


def _normalize_signature(value: Any) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    vector = value.detach().clone().cpu().float().reshape(-1)
    if int(vector.numel()) <= 0 or float(vector.norm().item()) <= 1e-8:
        return None
    return F.normalize(vector, dim=0)


def _cosine_similarity(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(torch.dot(left, right).item())


def _blend_centroid(
    current: torch.Tensor | None,
    incoming: torch.Tensor | None,
    *,
    weight: float,
) -> torch.Tensor | None:
    if incoming is None:
        return None if current is None else current.clone()
    if current is None:
        return incoming.clone()
    return F.normalize(current * float(max(0.0, weight)) + incoming, dim=0)


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
        min_lexical_overlap_for_merge: float = 0.20,
        min_query_overlap_terms: int = 1,
    ) -> None:
        self._merge_similarity = float(merge_similarity)
        self._lexical_merge_threshold = float(lexical_merge_threshold)
        self._lexical_weight = float(lexical_weight)
        self._min_lexical_overlap_for_merge = float(max(0.0, min_lexical_overlap_for_merge))
        self._min_query_overlap_terms = int(max(0, min_query_overlap_terms))
        self._entries: dict[str, dict[str, Any]] = {}
        self._observations = 0
        self._episode_index = 0
        self._next_id = 1
        self._slow_features = OnlineSlowFeatureMap(output_dim=slow_feature_dim)

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
            return _normalize_signature(torch.stack(valid_signatures, dim=0).mean(dim=0))
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
            "first_episode": int(self._episode_index),
            "last_episode": int(self._episode_index),
        }

    def _top_terms(self, entry: dict[str, Any], limit: int = 4) -> list[str]:
        return [str(term) for term, _ in entry["term_weights"].most_common(limit)]

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

    def _assignment_score(
        self,
        entry: dict[str, Any],
        token_set: set[str],
        raw_signature: torch.Tensor | None,
        slow_signature: torch.Tensor | None,
    ) -> tuple[float, float, float, float]:
        concept_terms = set(self._top_terms(entry, limit=8))
        lexical_overlap = 0.0 if not token_set else float(len(token_set & concept_terms) / max(1, len(token_set)))
        slow_reference = _normalize_signature(entry.get("slow_centroid"))
        if slow_reference is None:
            slow_reference = _normalize_signature(entry.get("raw_centroid"))
        slow_query = slow_signature if slow_signature is not None else raw_signature
        slow_similarity = _cosine_similarity(slow_query, slow_reference)
        raw_similarity = _cosine_similarity(raw_signature, _normalize_signature(entry.get("raw_centroid")))
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
        best_id: str | None = None
        best_score = -1.0
        best_slow = 0.0
        best_raw = 0.0
        best_lexical = 0.0

        for concept_id, entry in self._entries.items():
            score, slow_similarity, raw_similarity, lexical_overlap = self._assignment_score(
                entry,
                token_set,
                raw_signature,
                slow_signature,
            )
            if score > best_score:
                best_id = concept_id
                best_score = score
                best_slow = slow_similarity
                best_raw = raw_similarity
                best_lexical = lexical_overlap

        if best_id is not None and (
            best_slow >= self._merge_similarity
            or best_raw >= (self._merge_similarity + 0.05)
            or best_lexical >= self._lexical_merge_threshold
        ) and best_lexical >= self._min_lexical_overlap_for_merge:
            return best_id

        entry = self._new_entry(
            tokens=tokens,
            raw_signature=raw_signature,
            slow_signature=slow_signature,
        )
        concept_id = str(entry["concept_id"])
        self._entries[concept_id] = entry
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
            "episode_span": int(max(1, last_episode - first_episode + 1)),
            "memory_indices": memory_indices[:4],
            "example_windows": example_windows[:2],
            "top_terms": self._top_terms(entry),
        }

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
            "concepts": concepts[: max(1, int(limit))],
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
                    "first_episode": int(entry.get("first_episode", 0)),
                    "last_episode": int(entry.get("last_episode", 0)),
                }
            )
        return {
            "concept_mode": "slow_feature_concept_memory",
            "observations": int(self._observations),
            "episode_index": int(self._episode_index),
            "next_id": int(self._next_id),
            "slow_features": self._slow_features.state_dict(),
            "entries": entries,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self._entries = {}
        self._observations = int(payload.get("observations", 0))
        self._episode_index = int(payload.get("episode_index", 0))
        self._next_id = int(payload.get("next_id", 1))
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
            self._entries[concept_id] = {
                "concept_id": concept_id,
                "raw_centroid": raw_centroid,
                "slow_centroid": slow_centroid if slow_centroid is not None else raw_centroid,
                "last_slow_signature": last_slow if last_slow is not None else slow_centroid if slow_centroid is not None else raw_centroid,
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
                "first_episode": int(item.get("first_episode", 0)),
                "last_episode": int(item.get("last_episode", item.get("first_episode", 0))),
            }
