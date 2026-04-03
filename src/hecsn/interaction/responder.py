from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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


@dataclass(frozen=True)
class _EvidenceCandidate:
    text: str
    memory_index: int
    similarity: float
    importance: float
    age_tokens: int
    matching_terms: tuple[str, ...]
    concept_labels: tuple[str, ...]
    concept_priority: float
    score: float


class EvidenceResponder:
    def __init__(
        self,
        *,
        min_similarity: float = 0.35,
        min_token_coverage: float = 0.25,
        min_native_decode_confidence: float = 0.55,
        min_native_decode_continuation_chars: int = 4,
    ) -> None:
        self.min_similarity = float(min_similarity)
        self.min_token_coverage = float(min_token_coverage)
        self.min_native_decode_confidence = float(min_native_decode_confidence)
        self.min_native_decode_continuation_chars = max(1, int(min_native_decode_continuation_chars))

    def build_response(
        self,
        query_text: str,
        query_summary: Mapping[str, Any],
        *,
        concept_summary: Mapping[str, Any] | None = None,
        max_evidence_items: int = 3,
    ) -> dict[str, Any]:
        evidence = self._select_evidence(
            query_text=query_text,
            memory_matches=query_summary.get("memory_matches", []),
            concept_summary=concept_summary,
            max_evidence_items=max_evidence_items,
        )
        selected_evidence = evidence["selected_evidence"]
        evidence_coverage = float(evidence["evidence_coverage"])
        unsupported_terms = list(evidence["unsupported_terms"])
        top_similarity = float(evidence["top_similarity"])
        support_score = float(min(1.0, 0.6 * top_similarity + 0.4 * evidence_coverage))
        concept_grounding = self._build_concept_grounding(
            concept_summary=concept_summary,
            selected_evidence=selected_evidence,
        )
        native_decode = dict(query_summary.get("native_decode") or {})
        native_decode["available"] = bool(native_decode.get("available", False))
        native_decode_ready = self._native_decode_ready(native_decode)

        if (not selected_evidence or evidence_coverage < self.min_token_coverage) and native_decode_ready:
            support_score = max(
                support_score,
                min(
                    1.0,
                    0.5 * float(native_decode.get("confidence", 0.0))
                    + 0.5 * float(native_decode.get("query_overlap_ratio", 0.0)),
                ),
            )
            response_text = self._native_decode_response_text(native_decode, selected_evidence)
            if unsupported_terms:
                response_text += f" I do not have grounded support for: {', '.join(unsupported_terms)}."
            return {
                "response_mode": "native_decode",
                "response_text": response_text,
                "support_score": support_score,
                "evidence_coverage": evidence_coverage,
                "unsupported_terms": unsupported_terms,
                "selected_evidence": selected_evidence,
                "concept_grounding": concept_grounding,
                "native_decode": native_decode,
            }

        if not selected_evidence or evidence_coverage < self.min_token_coverage:
            response_text = "I do not have enough grounded evidence to answer this directly."
            if selected_evidence:
                response_text += f' Closest remembered evidence: "{selected_evidence[0]["text"]}".'
            if unsupported_terms:
                response_text += f" Missing grounded support for: {', '.join(unsupported_terms)}."
            return {
                "response_mode": "insufficient_evidence",
                "response_text": response_text,
                "support_score": support_score,
                "evidence_coverage": evidence_coverage,
                "unsupported_terms": unsupported_terms,
                "selected_evidence": selected_evidence,
                "concept_grounding": concept_grounding,
                "native_decode": native_decode if native_decode["available"] else None,
            }

        if self._should_use_native_decode(native_decode, selected_evidence):
            response_mode = "native_decode"
            response_text = self._native_decode_response_text(native_decode, selected_evidence)
        elif len(selected_evidence) == 1:
            response_mode = "quote"
            response_text = f'Based on the closest remembered evidence, "{selected_evidence[0]["text"]}".'
        else:
            response_mode = "stitch"
            merged = self._merge_overlapping_windows([item["text"] for item in selected_evidence])
            response_text = f'Based on the closest remembered evidence, "{merged}".'
            remaining = [item["text"] for item in selected_evidence[1:]]
            if remaining:
                response_text += " Supporting fragments: " + "; ".join(f'"{text}"' for text in remaining) + "."

        if unsupported_terms:
            response_text += f" I do not have grounded support for: {', '.join(unsupported_terms)}."

        return {
            "response_mode": response_mode,
            "response_text": response_text,
            "support_score": support_score,
            "evidence_coverage": evidence_coverage,
            "unsupported_terms": unsupported_terms,
            "selected_evidence": selected_evidence,
            "concept_grounding": concept_grounding,
            "native_decode": native_decode if native_decode["available"] else None,
        }

    def _should_use_native_decode(
        self,
        native_decode: Mapping[str, Any],
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> bool:
        if not self._native_decode_ready(native_decode):
            return False
        decoded_text = self._normalize_text(native_decode.get("decoded_text"))
        if not selected_evidence:
            return True
        lead_evidence = self._normalize_text(selected_evidence[0].get("text"))
        if lead_evidence and decoded_text.lower() == lead_evidence.lower():
            return False
        return True

    def _native_decode_ready(self, native_decode: Mapping[str, Any]) -> bool:
        if not native_decode.get("available"):
            return False
        decoded_text = self._normalize_text(native_decode.get("decoded_text"))
        continuation_text = self._normalize_text(native_decode.get("continuation_text"))
        confidence = float(native_decode.get("confidence", 0.0))
        query_overlap_ratio = float(native_decode.get("query_overlap_ratio", 0.0))
        if not decoded_text or confidence < self.min_native_decode_confidence:
            return False
        if len(continuation_text) < self.min_native_decode_continuation_chars:
            return False
        return query_overlap_ratio >= 0.45

    def _native_decode_response_text(
        self,
        native_decode: Mapping[str, Any],
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> str:
        decoded_text = self._normalize_text(native_decode.get("decoded_text"))
        response_text = f'Native assembly decode: "{decoded_text}".'
        supporting_fragments = [
            self._normalize_text(item.get("text"))
            for item in selected_evidence
            if self._normalize_text(item.get("text"))
            and self._normalize_text(item.get("text")).lower() not in decoded_text.lower()
        ]
        if supporting_fragments:
            response_text += " Supporting fragments: " + "; ".join(f'"{text}"' for text in supporting_fragments) + "."
        return response_text

    def _select_evidence(
        self,
        *,
        query_text: str,
        memory_matches: Sequence[Mapping[str, Any]],
        concept_summary: Mapping[str, Any] | None,
        max_evidence_items: int,
    ) -> dict[str, Any]:
        query_terms = set(self._tokenize(query_text))
        concept_lookup = self._build_concept_lookup(concept_summary)
        ranked: list[_EvidenceCandidate] = []

        for match in memory_matches:
            text = self._normalize_text(match.get("raw_window"))
            if not text:
                continue
            evidence_terms = set(self._tokenize(text))
            matching_terms = tuple(sorted(query_terms & evidence_terms))
            similarity = float(match.get("similarity", 0.0))
            importance = float(match.get("importance", 0.0))
            overlap = (len(matching_terms) / len(query_terms)) if query_terms else 1.0
            if similarity < self.min_similarity and overlap <= 0.0:
                continue
            memory_index = int(match.get("memory_index", -1))
            concept_entries = concept_lookup.get(memory_index, ())
            concept_labels = tuple(str(entry["label"]) for entry in concept_entries if entry.get("label"))
            concept_priority = max((float(entry.get("priority", 0.0)) for entry in concept_entries), default=0.0)
            score = similarity + 0.35 * overlap + 0.02 * min(5.0, importance) + 0.05 * concept_priority
            ranked.append(
                _EvidenceCandidate(
                    text=text,
                    memory_index=memory_index,
                    similarity=similarity,
                    importance=importance,
                    age_tokens=int(match.get("age_tokens", 0)),
                    matching_terms=matching_terms,
                    concept_labels=concept_labels,
                    concept_priority=concept_priority,
                    score=score,
                )
            )

        ranked.sort(
            key=lambda item: (item.score, item.concept_priority, item.similarity, -item.age_tokens),
            reverse=True,
        )

        selected_candidates = self._select_diverse_candidates(ranked, max_evidence_items)

        selected: list[dict[str, Any]] = []
        supported_terms: set[str] = set()
        for item in selected_candidates:
            selected.append(
                {
                    "memory_index": item.memory_index,
                    "text": item.text,
                    "similarity": item.similarity,
                    "importance": item.importance,
                    "age_tokens": item.age_tokens,
                    "matching_terms": list(item.matching_terms),
                    "concept_labels": list(item.concept_labels),
                    "primary_concept": None if not item.concept_labels else item.concept_labels[0],
                    "score": item.score,
                }
            )
            supported_terms.update(item.matching_terms)

        coverage = (len(supported_terms) / len(query_terms)) if query_terms else 1.0
        return {
            "selected_evidence": selected,
            "evidence_coverage": float(coverage),
            "unsupported_terms": sorted(query_terms - supported_terms),
            "top_similarity": 0.0 if not selected else float(selected[0]["similarity"]),
        }

    def _select_diverse_candidates(
        self,
        ranked: Sequence[_EvidenceCandidate],
        max_evidence_items: int,
    ) -> list[_EvidenceCandidate]:
        limit = max(1, int(max_evidence_items))
        selected: list[_EvidenceCandidate] = []
        deferred: list[_EvidenceCandidate] = []
        seen_texts: set[str] = set()
        seen_concepts: set[str] = set()

        for item in ranked:
            key = item.text.lower()
            if key in seen_texts:
                continue

            primary_concept = item.concept_labels[0] if item.concept_labels else None
            if primary_concept and primary_concept in seen_concepts and len(seen_concepts) < limit:
                deferred.append(item)
                continue

            selected.append(item)
            seen_texts.add(key)
            if primary_concept:
                seen_concepts.add(primary_concept)
            if len(selected) >= limit:
                return selected

        for item in deferred:
            key = item.text.lower()
            if key in seen_texts:
                continue
            selected.append(item)
            seen_texts.add(key)
            if len(selected) >= limit:
                break

        return selected

    def _build_concept_lookup(
        self,
        concept_summary: Mapping[str, Any] | None,
    ) -> dict[int, list[dict[str, Any]]]:
        if not isinstance(concept_summary, Mapping):
            return {}

        concepts = concept_summary.get("concepts") or []
        total = max(1, len(concepts))
        lookup: dict[int, list[dict[str, Any]]] = {}

        for rank, concept in enumerate(concepts):
            label = self._normalize_text(concept.get("label"))
            if not label:
                continue

            descriptor = {
                "concept_id": self._normalize_text(concept.get("concept_id")),
                "label": label,
                "score": float(concept.get("score", 0.0)),
                "priority": float(total - rank) / float(total),
                "uncertainty": float(concept.get("uncertainty", 1.0)),
                "drift": float(concept.get("drift", 0.0)),
                "top_terms": [
                    self._normalize_text(term)
                    for term in (concept.get("top_terms") or [])
                    if self._normalize_text(term)
                ][:4],
            }

            for memory_index in concept.get("memory_indices") or []:
                try:
                    normalized_index = int(memory_index)
                except (TypeError, ValueError):
                    continue
                lookup.setdefault(normalized_index, []).append(descriptor)

        for items in lookup.values():
            items.sort(
                key=lambda item: (float(item.get("priority", 0.0)), float(item.get("score", 0.0))),
                reverse=True,
            )
        return lookup

    def _build_concept_grounding(
        self,
        *,
        concept_summary: Mapping[str, Any] | None,
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(concept_summary, Mapping):
            return {
                "focus_label": None,
                "query_concepts": [],
                "selected_concepts": [],
                "query_concept_coverage": None,
            }

        concept_index: dict[str, dict[str, Any]] = {}
        query_concepts: list[dict[str, Any]] = []
        for concept in concept_summary.get("concepts") or []:
            label = self._normalize_text(concept.get("label"))
            if not label:
                continue
            record = {
                "concept_id": self._normalize_text(concept.get("concept_id")),
                "label": label,
                "score": float(concept.get("score", 0.0)),
                "match_count": int(concept.get("match_count", 0)),
                "observations": int(concept.get("observations", 0)),
                "uncertainty": float(concept.get("uncertainty", 1.0)),
                "drift": float(concept.get("drift", 0.0)),
                "top_terms": [
                    self._normalize_text(term)
                    for term in (concept.get("top_terms") or [])
                    if self._normalize_text(term)
                ][:4],
            }
            concept_index[label] = record
            query_concepts.append(record)

        selected_stats: dict[str, dict[str, Any]] = {}
        for item in selected_evidence:
            concept_labels = item.get("concept_labels") or []
            if not concept_labels and item.get("primary_concept"):
                concept_labels = [item["primary_concept"]]

            for label_value in concept_labels:
                label = self._normalize_text(label_value)
                if not label:
                    continue
                entry = selected_stats.setdefault(
                    label,
                    {
                        "label": label,
                        "concept_id": self._normalize_text(concept_index.get(label, {}).get("concept_id")),
                        "score": float(concept_index.get(label, {}).get("score", 0.0)),
                        "observations": int(concept_index.get(label, {}).get("observations", 0)),
                        "uncertainty": float(concept_index.get(label, {}).get("uncertainty", 1.0)),
                        "drift": float(concept_index.get(label, {}).get("drift", 0.0)),
                        "top_terms": list(concept_index.get(label, {}).get("top_terms") or []),
                        "evidence_count": 0,
                        "memory_indices": [],
                    },
                )
                entry["evidence_count"] += 1
                memory_index = item.get("memory_index")
                if memory_index is None:
                    continue
                try:
                    normalized_index = int(memory_index)
                except (TypeError, ValueError):
                    continue
                if normalized_index not in entry["memory_indices"]:
                    entry["memory_indices"].append(normalized_index)

        ordered_labels = [
            concept["label"]
            for concept in query_concepts
            if concept["label"] in selected_stats
        ]
        ordered_labels.extend(label for label in selected_stats if label not in ordered_labels)

        selected_concepts: list[dict[str, Any]] = []
        for label in ordered_labels:
            concept_record = concept_index.get(label, {})
            selected_record = selected_stats[label]
            selected_concepts.append(
                {
                    "concept_id": self._normalize_text(concept_record.get("concept_id")),
                    "label": label,
                    "score": float(concept_record.get("score", 0.0)),
                    "observations": int(concept_record.get("observations", 0)),
                    "uncertainty": float(concept_record.get("uncertainty", 1.0)),
                    "drift": float(concept_record.get("drift", 0.0)),
                    "evidence_count": int(selected_record["evidence_count"]),
                    "memory_indices": selected_record["memory_indices"][:4],
                    "top_terms": list(concept_record.get("top_terms") or []),
                }
            )

        coverage = None
        if query_concepts:
            query_labels = {concept["label"] for concept in query_concepts}
            selected_labels = {concept["label"] for concept in selected_concepts}
            coverage = len(query_labels & selected_labels) / max(1, len(query_labels))

        return {
            "focus_label": None if not selected_concepts else selected_concepts[0]["label"],
            "query_concepts": query_concepts[:3],
            "selected_concepts": selected_concepts,
            "query_concept_coverage": coverage,
        }

    def _merge_overlapping_windows(self, windows: Sequence[str]) -> str:
        merged = [self._normalize_text(window) for window in windows if self._normalize_text(window)]
        if not merged:
            return ""
        output = merged[0]
        for window in merged[1:]:
            overlap = self._overlap_suffix_prefix(output, window)
            if overlap > 0:
                output = output + window[overlap:]
            elif window.lower() not in output.lower():
                output = output + " " + window
        return self._normalize_text(output)

    def _overlap_suffix_prefix(self, left: str, right: str) -> int:
        max_overlap = min(len(left), len(right))
        for size in range(max_overlap, 2, -1):
            if left[-size:].lower() == right[:size].lower():
                return size
        return 0

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = " ".join(str(value).split())
        return text.strip()

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for match in _TOKEN_RE.findall(text.lower()):
            if match in _STOPWORDS:
                continue
            if len(match) == 1 and not match.isdigit():
                continue
            tokens.append(match)
        return tokens