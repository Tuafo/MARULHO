from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from marulho.semantics.grounding_text import TOKEN_RE
from marulho.semantics.grounding_text import match_terms
from marulho.semantics.grounding_text import normalize_text
from marulho.semantics.grounding_text import salient_query_terms
from marulho.semantics.grounding_text import split_sentences


@dataclass(frozen=True)
class _EvidenceCandidate:
    text: str
    memory_index: int
    memory_indices: tuple[int, ...]
    similarity: float
    importance: float
    age_tokens: int
    matching_terms: tuple[str, ...]
    concept_labels: tuple[str, ...]
    concept_priority: float
    term_coverage: float
    fragmentary: bool
    complete_sentence: int
    clipped_overlap: int
    score: float
    source_name: str = ""
    source_type: str = ""
    provider: str = ""
    source_names: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()


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
            memory_episodes=query_summary.get("memory_episodes", []),
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

        fragmentary_support = self._selected_evidence_is_fragmentary(selected_evidence)
        synthesis_evidence = self._grounded_synthesis_candidates(selected_evidence)

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
            if unsupported_terms and (evidence_coverage < 0.85 or fragmentary_support):
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
                "subcortex_language": self._subcortex_language_surface(
                    native_decode=native_decode,
                    selected_evidence=selected_evidence,
                    concept_grounding=concept_grounding,
                    support_score=support_score,
                    evidence_coverage=evidence_coverage,
                ),
            }

        if not selected_evidence or (evidence_coverage < self.min_token_coverage and fragmentary_support):
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
        elif self._should_groundedly_synthesize(query_text, synthesis_evidence):
            response_mode = "grounded_synthesis"
            response_text = self._grounded_synthesis_response_text(query_text, synthesis_evidence)
        elif self._should_quote_top_evidence(selected_evidence):
            response_mode = "quote"
            response_text = f'Based on the closest remembered evidence, "{selected_evidence[0]["text"]}".'
        else:
            response_mode = "stitch"
            merged = self._merge_overlapping_windows([item["text"] for item in selected_evidence])
            response_text = f'Based on the closest remembered evidence, "{merged}".'
            remaining = [item["text"] for item in selected_evidence[1:]]
            if remaining:
                response_text += " Supporting fragments: " + "; ".join(f'"{text}"' for text in remaining) + "."

        if unsupported_terms and (evidence_coverage < 0.85 or fragmentary_support):
            response_text += f" I do not have grounded support for: {', '.join(unsupported_terms)}."

        response = {
            "response_mode": response_mode,
            "response_text": response_text,
            "support_score": support_score,
            "evidence_coverage": evidence_coverage,
            "unsupported_terms": unsupported_terms,
            "selected_evidence": selected_evidence,
            "concept_grounding": concept_grounding,
            "native_decode": native_decode if native_decode["available"] else None,
        }
        if response_mode == "native_decode":
            response["subcortex_language"] = self._subcortex_language_surface(
                native_decode=native_decode,
                selected_evidence=selected_evidence,
                concept_grounding=concept_grounding,
                support_score=support_score,
                evidence_coverage=evidence_coverage,
            )
        return response

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
        lead_item = selected_evidence[0]
        lead_evidence = self._normalize_text(lead_item.get("text"))
        if lead_evidence and decoded_text.lower() == lead_evidence.lower():
            return False
        if (
            not bool(lead_item.get("fragmentary"))
            and float(lead_item.get("term_coverage", 0.0)) >= max(self.min_token_coverage, 0.5)
        ):
            return False
        if len(selected_evidence) >= 2 and all(not bool(item.get("fragmentary")) for item in selected_evidence[:2]):
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

    def _subcortex_language_surface(
        self,
        *,
        native_decode: Mapping[str, Any],
        selected_evidence: Sequence[Mapping[str, Any]],
        concept_grounding: Mapping[str, Any],
        support_score: float,
        evidence_coverage: float,
    ) -> dict[str, Any]:
        decoded_text = self._normalize_text(native_decode.get("decoded_text"))
        continuation_text = self._normalize_text(native_decode.get("continuation_text"))
        confidence = float(native_decode.get("confidence", 0.0))
        overlap = float(native_decode.get("query_overlap_ratio", 0.0))
        source_indices: list[int] = []
        concept_focus = self._normalize_text(concept_grounding.get("focus_label"))
        if not concept_focus:
            selected_concepts = concept_grounding.get("selected_concepts") or []
            if isinstance(selected_concepts, Sequence) and not isinstance(selected_concepts, (str, bytes)):
                for concept in selected_concepts:
                    if isinstance(concept, Mapping):
                        concept_focus = self._normalize_text(concept.get("label"))
                        if concept_focus:
                            break

        for item in selected_evidence:
            memory_indices = item.get("memory_indices")
            if not isinstance(memory_indices, Sequence) or isinstance(memory_indices, (str, bytes)):
                memory_indices = [item.get("memory_index")]
            for raw_index in memory_indices:
                try:
                    memory_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if memory_index >= 0 and memory_index not in source_indices:
                    source_indices.append(memory_index)

        evidence_count = len(selected_evidence)
        state_text = (
            f"Native assembly decode is supported by {evidence_count} memory item"
            f"{'' if evidence_count == 1 else 's'} with {confidence:.2f} confidence, "
            f"{overlap:.2f} query overlap, and {float(evidence_coverage):.2f} term coverage."
        )
        if concept_focus:
            state_text += f" Focus: {concept_focus}."
        if not source_indices:
            state_text += " No memory index was attached to this decode."

        candidates = [text for text in (decoded_text, continuation_text) if text]
        return {
            "surface": "subcortical_language.v1",
            "available": True,
            "state_text": state_text,
            "source": "interaction.responder.native_decode",
            "grounded": True,
            "not_cognition_substrate": True,
            "candidate_phrases": candidates[:2],
            "grounding": {
                "support_score": float(support_score),
                "evidence_coverage": float(evidence_coverage),
                "native_decode_confidence": confidence,
                "query_overlap_ratio": overlap,
                "source_memory_indices": source_indices[:8],
                "concept_focus": concept_focus or None,
            },
            "limitations": [
                "Deterministic decode over native evidence, not an autonomous generator.",
            ],
        }

    def _select_evidence(
        self,
        *,
        query_text: str,
        memory_matches: Sequence[Mapping[str, Any]],
        memory_episodes: Sequence[Mapping[str, Any]],
        concept_summary: Mapping[str, Any] | None,
        max_evidence_items: int,
    ) -> dict[str, Any]:
        query_terms = salient_query_terms(query_text)
        concept_lookup = self._build_concept_lookup(concept_summary)
        ranked: list[_EvidenceCandidate] = []

        evidence_sources = memory_episodes if memory_episodes else memory_matches
        for match in evidence_sources:
            text = self._normalize_text(match.get("text") or match.get("raw_window"))
            if not text:
                continue
            raw_window = self._normalize_text(match.get("raw_window"))
            matching_terms = tuple(match_terms(query_terms, text))
            similarity = float(match.get("similarity", 0.0))
            importance = float(match.get("importance", 0.0))
            overlap = (len(matching_terms) / len(query_terms)) if query_terms else 1.0
            if query_terms and overlap <= 0.0:
                continue
            if similarity < self.min_similarity and overlap <= 0.0:
                continue
            memory_indices = self._memory_indices(match)
            memory_index = -1 if not memory_indices else int(memory_indices[0])
            concept_entries = self._concept_entries(concept_lookup, memory_indices)
            concept_labels = tuple(str(entry["label"]) for entry in concept_entries if entry.get("label"))
            concept_priority = max((float(entry.get("priority", 0.0)) for entry in concept_entries), default=0.0)
            complete_sentence = int(match.get("complete_sentence", int(text.endswith((".", "!", "?")))))
            expansion_chars = int(match.get("expansion_chars", max(0, len(text) - len(raw_window))))
            clipped_overlap = int(
                match.get(
                    "clipped_overlap",
                    int(bool(raw_window and not complete_sentence and text.lower() == raw_window.lower())),
                )
            )
            verified_action_evidence = bool(
                match.get("action_origin")
                and match.get("action_type")
                and complete_sentence
                and clipped_overlap <= 0
            )
            fragmentary = (
                False
                if verified_action_evidence
                else self._is_fragmentary_evidence(text, raw_window)
            )
            metadata_scaffold = text.strip().lower().startswith(("terms:", "topics:"))
            score = similarity + 0.35 * overlap + 0.02 * min(5.0, importance) + 0.05 * concept_priority
            score += 0.05 if text.endswith((".", "!", "?")) else -0.03
            score += 0.08 if not fragmentary else -0.05
            score += 0.04 * complete_sentence
            score += 0.01 * min(20.0, float(expansion_chars))
            score -= 0.08 * clipped_overlap
            score -= 0.60 if metadata_scaffold else 0.0
            ranked.append(
                _EvidenceCandidate(
                    text=text,
                    memory_index=memory_index,
                    memory_indices=memory_indices,
                    similarity=similarity,
                    importance=importance,
                    age_tokens=int(match.get("age_tokens", 0)),
                    matching_terms=matching_terms,
                    concept_labels=concept_labels,
                    concept_priority=concept_priority,
                    term_coverage=overlap,
                    fragmentary=fragmentary,
                    complete_sentence=complete_sentence,
                    clipped_overlap=clipped_overlap,
                    score=score,
                    source_name=" ".join(str(match.get("source_name", "")).split()).strip(),
                    source_type=" ".join(str(match.get("source_type", "")).split()).strip(),
                    provider=" ".join(str(match.get("provider", "")).split()).strip().lower(),
                    source_names=tuple(
                        str(item).strip()
                        for item in list(match.get("source_names") or [])
                        if str(item).strip()
                    ),
                    providers=tuple(
                        " ".join(str(item).split()).strip().lower()
                        for item in list(match.get("providers") or [])
                        if " ".join(str(item).split()).strip()
                    ),
                )
            )

        ranked.sort(
            key=lambda item: (item.score, item.concept_priority, item.similarity, -item.age_tokens),
            reverse=True,
        )

        selected_candidates = self._select_diverse_candidates(
            ranked,
            max_evidence_items,
            query_terms=query_terms,
        )

        selected: list[dict[str, Any]] = []
        supported_terms: set[str] = set()
        for item in selected_candidates:
            selected.append(
                {
                    "memory_index": item.memory_index,
                    "memory_indices": list(item.memory_indices),
                    "text": item.text,
                    "similarity": item.similarity,
                    "importance": item.importance,
                    "age_tokens": item.age_tokens,
                    "matching_terms": list(item.matching_terms),
                    "concept_labels": list(item.concept_labels),
                    "primary_concept": None if not item.concept_labels else item.concept_labels[0],
                    "term_coverage": float(item.term_coverage),
                    "fragmentary": bool(item.fragmentary),
                    "complete_sentence": int(item.complete_sentence),
                    "clipped_overlap": int(item.clipped_overlap),
                    "score": item.score,
                    "source_name": item.source_name,
                    "source_type": item.source_type,
                    "provider": item.provider,
                    "source_names": list(item.source_names),
                    "providers": list(item.providers),
                }
            )
            supported_terms.update(item.matching_terms)

        coverage = (len(supported_terms) / len(query_terms)) if query_terms else 1.0
        return {
            "selected_evidence": selected,
            "evidence_coverage": float(coverage),
            "unsupported_terms": [term for term in query_terms if term not in supported_terms],
            "top_similarity": 0.0 if not selected else float(selected[0]["similarity"]),
        }

    def _memory_indices(self, match: Mapping[str, Any]) -> tuple[int, ...]:
        values = match.get("memory_indices")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            values = [match.get("memory_index")]

        normalized: list[int] = []
        for value in values:
            try:
                memory_index = int(value)
            except (TypeError, ValueError):
                continue
            if memory_index < 0 or memory_index in normalized:
                continue
            normalized.append(memory_index)
        return tuple(normalized)

    def _concept_entries(
        self,
        concept_lookup: Mapping[int, Sequence[Mapping[str, Any]]],
        memory_indices: Sequence[int],
    ) -> tuple[dict[str, Any], ...]:
        concept_entries: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for memory_index in memory_indices:
            for entry in concept_lookup.get(int(memory_index), ()):
                label = self._normalize_text(entry.get("label"))
                if not label or label in seen_labels:
                    continue
                seen_labels.add(label)
                concept_entries.append(dict(entry))
        concept_entries.sort(
            key=lambda item: (float(item.get("priority", 0.0)), float(item.get("score", 0.0))),
            reverse=True,
        )
        return tuple(concept_entries)

    def _is_fragmentary_evidence(self, text: str, raw_window: str) -> bool:
        if not text:
            return True
        tokens = self._tokenize(text)
        if len(tokens) < 3:
            return True
        if len(tokens) < 4 and not text.endswith((".", "!", "?")):
            return True
        if raw_window and text.lower() == raw_window.lower() and len(text) < max(40, len(raw_window) + 12):
            return True
        return False

    def _selected_evidence_is_fragmentary(self, selected_evidence: Sequence[Mapping[str, Any]]) -> bool:
        if not selected_evidence:
            return True
        return all(bool(item.get("fragmentary")) for item in selected_evidence)

    def _should_quote_top_evidence(self, selected_evidence: Sequence[Mapping[str, Any]]) -> bool:
        if not selected_evidence:
            return False
        lead = selected_evidence[0]
        if bool(lead.get("fragmentary")):
            return len(selected_evidence) == 1
        if int(lead.get("clipped_overlap", 0)) > 0 and len(selected_evidence) > 1:
            return False
        if len(selected_evidence) == 1:
            return True
        lead_score = float(lead.get("score", 0.0))
        runner_up_score = float(selected_evidence[1].get("score", 0.0))
        if float(lead.get("term_coverage", 0.0)) >= 0.5:
            return True
        return lead_score >= runner_up_score + 0.08

    def _grounded_synthesis_candidates(
        self,
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        return [
            item
            for item in selected_evidence
            if not bool(item.get("fragmentary")) and int(item.get("clipped_overlap", 0)) <= 0
        ]

    def _should_groundedly_synthesize(
        self,
        query_text: str,
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> bool:
        if len(selected_evidence) < 2:
            return False

        query_terms = set(salient_query_terms(query_text))
        supported_terms = {
            str(term)
            for item in selected_evidence
            for term in (item.get("matching_terms") or [])
            if isinstance(term, str) and term
        }
        evidence_coverage = (len(supported_terms) / len(query_terms)) if query_terms else 1.0
        lead_coverage = float(selected_evidence[0].get("term_coverage", 0.0))
        if evidence_coverage < max(self.min_token_coverage, lead_coverage + 0.10):
            return False

        additional_support = False
        observed_terms: set[str] = set()
        for item in selected_evidence:
            matching_terms = {
                str(term)
                for term in (item.get("matching_terms") or [])
                if isinstance(term, str) and term
            }
            if matching_terms - observed_terms:
                if observed_terms:
                    additional_support = True
                observed_terms.update(matching_terms)
        if not additional_support:
            return False

        return bool(self._shared_anchor_phrase(selected_evidence, query_terms)) or evidence_coverage >= 0.80

    def _grounded_synthesis_response_text(
        self,
        query_text: str,
        selected_evidence: Sequence[Mapping[str, Any]],
    ) -> str:
        texts = self._best_grounded_clauses(selected_evidence)
        if not texts:
            return "I do not have enough grounded evidence to answer this directly."

        query_terms = set(self._tokenize(query_text))
        anchor_phrase = self._shared_anchor_phrase(selected_evidence, query_terms)
        if anchor_phrase:
            clauses = [
                self._strip_terminal_punctuation(self._strip_leading_phrase(text, anchor_phrase))
                for text in texts
            ]
            clauses = [clause for clause in clauses if clause]
            if clauses:
                joined_clauses = self._join_clauses(clauses)
                return f"Based on grounded evidence, {anchor_phrase} {joined_clauses}."

        statements = [self._ensure_terminal_sentence(text) for text in texts]
        lead = statements[0]
        remainder = statements[1:]
        if not remainder:
            return f"Based on grounded evidence, {lead}"
        return "Based on grounded evidence, " + " ".join(
            [lead] + [f"Also, {statement}" for statement in remainder]
        )

    def _select_diverse_candidates(
        self,
        ranked: Sequence[_EvidenceCandidate],
        max_evidence_items: int,
        query_terms: Sequence[str] | None = None,
    ) -> list[_EvidenceCandidate]:
        limit = max(1, int(max_evidence_items))
        selected: list[_EvidenceCandidate] = []
        deferred: list[_EvidenceCandidate] = []
        seen_texts: set[str] = set()
        seen_concepts: set[str] = set()
        covered_terms: set[str] = set()
        required_terms = {str(term) for term in (query_terms or []) if isinstance(term, str) and term}

        for item in ranked:
            key = item.text.lower()
            if key in seen_texts:
                continue

            primary_concept = item.concept_labels[0] if item.concept_labels else None
            contributes_new_terms = bool(set(item.matching_terms) - covered_terms)
            clean_additional_support = (
                contributes_new_terms
                and not item.fragmentary
                and not item.clipped_overlap
            )
            if (
                primary_concept
                and primary_concept in seen_concepts
                and len(seen_concepts) < limit
                and not clean_additional_support
            ):
                deferred.append(item)
                continue

            selected.append(item)
            seen_texts.add(key)
            if primary_concept:
                seen_concepts.add(primary_concept)
            covered_terms.update(item.matching_terms)
            if required_terms and covered_terms >= required_terms:
                clean_selected = [
                    candidate
                    for candidate in selected
                    if not candidate.fragmentary and candidate.clipped_overlap <= 0
                ]
                if clean_selected:
                    return selected
            if len(selected) >= limit:
                return selected

        for item in deferred:
            key = item.text.lower()
            if key in seen_texts:
                continue
            selected.append(item)
            seen_texts.add(key)
            covered_terms.update(item.matching_terms)
            if required_terms and covered_terms >= required_terms:
                clean_selected = [
                    candidate
                    for candidate in selected
                    if not candidate.fragmentary and candidate.clipped_overlap <= 0
                ]
                if clean_selected:
                    break
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
                memory_indices = item.get("memory_indices")
                if not isinstance(memory_indices, Sequence) or isinstance(memory_indices, (str, bytes)):
                    memory_indices = [item.get("memory_index")]
                for memory_index in memory_indices:
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
        return normalize_text(value)

    def _shared_anchor_phrase(
        self,
        selected_evidence: Sequence[Mapping[str, Any]],
        query_terms: set[str],
    ) -> str:
        texts = self._best_grounded_clauses(selected_evidence)
        if len(texts) < 2:
            return ""

        token_rows = [self._word_tokens(text) for text in texts]
        if any(not row for row in token_rows):
            return ""

        shared_prefix: list[str] = []
        for parts in zip(*token_rows):
            lowered = {part.lower() for part in parts}
            if len(lowered) != 1:
                break
            shared_prefix.append(parts[0])
        if shared_prefix:
            phrase = " ".join(shared_prefix).strip()
            if set(self._tokenize(phrase)):
                return phrase

        common_terms = set(self._tokenize(texts[0]))
        for text in texts[1:]:
            common_terms &= set(self._tokenize(text))
        if not common_terms:
            return ""

        preferred_terms = [term for term in self._tokenize(" ".join(texts)) if term in common_terms]
        if query_terms:
            preferred_terms = [term for term in preferred_terms if term in query_terms] or preferred_terms
        return preferred_terms[0] if preferred_terms else ""

    def _word_tokens(self, text: str) -> list[str]:
        return [match.group(0) for match in TOKEN_RE.finditer(text)]

    def _best_grounded_clauses(self, selected_evidence: Sequence[Mapping[str, Any]]) -> list[str]:
        clauses: list[str] = []
        for item in selected_evidence:
            text = self._normalize_text(item.get("text"))
            if not text:
                continue
            matching_terms = {
                str(term)
                for term in (item.get("matching_terms") or [])
                if isinstance(term, str) and term
            }
            clause = self._best_clause_for_terms(text, matching_terms)
            clause = self._normalize_text(clause or text)
            if not clause:
                continue
            normalized_clause = clause.lower()
            replaced = False
            for idx, existing in enumerate(list(clauses)):
                normalized_existing = existing.lower()
                if normalized_clause == normalized_existing:
                    replaced = True
                    break
                if normalized_clause in normalized_existing:
                    replaced = True
                    break
                if normalized_existing in normalized_clause:
                    clauses[idx] = clause
                    replaced = True
                    break
            if not replaced:
                clauses.append(clause)
        return clauses

    def _best_clause_for_terms(self, text: str, matching_terms: set[str]) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""
        segments = [self._normalize_text(segment) for segment in split_sentences(normalized) if self._normalize_text(segment)]
        if not segments:
            return normalized
        if len(segments) == 1:
            return segments[0]
        scored_segments: list[tuple[int, int, str]] = []
        for segment in segments:
            overlap = len(match_terms(list(matching_terms), segment)) if matching_terms else 0
            segment_terms = set(self._tokenize(segment))
            scored_segments.append((overlap, len(segment_terms), segment))
        scored_segments.sort(reverse=True)
        best_overlap, _, best_segment = scored_segments[0]
        if best_overlap > 0:
            return best_segment
        return segments[0]

    def _strip_leading_phrase(self, text: str, phrase: str) -> str:
        if not text or not phrase:
            return text
        pattern = re.compile(rf"^\s*{re.escape(phrase)}\b[\s,;:-]*", re.IGNORECASE)
        stripped = pattern.sub("", text, count=1)
        return self._normalize_text(stripped)

    def _strip_terminal_punctuation(self, text: str) -> str:
        return self._normalize_text(text).rstrip(" .!?;:")

    def _ensure_terminal_sentence(self, text: str) -> str:
        normalized = self._normalize_text(text)
        if not normalized:
            return ""
        if normalized.endswith((".", "!", "?")):
            return normalized
        return normalized + "."

    def _join_clauses(self, clauses: Sequence[str]) -> str:
        normalized = [self._strip_terminal_punctuation(clause) for clause in clauses if self._strip_terminal_punctuation(clause)]
        if not normalized:
            return ""
        if len(normalized) == 1:
            return normalized[0]
        if len(normalized) == 2:
            return f"{normalized[0]} and {normalized[1]}"
        return ", ".join(normalized[:-1]) + f", and {normalized[-1]}"

    def _tokenize(self, text: str) -> list[str]:
        return list(salient_query_terms(text) if text else [])
