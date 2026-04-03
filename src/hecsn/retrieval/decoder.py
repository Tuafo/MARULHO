from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class _DecoderCandidate:
    memory_index: int
    text: str
    similarity: float
    importance: float
    tag_strength: float
    bucket_id: int | None
    same_bucket: bool
    score: float


@dataclass(frozen=True)
class _MergeChoice:
    candidate: _DecoderCandidate
    merged_text: str
    overlap: int
    direction: str
    added_text: str


class NativeAssemblyDecoder:
    def __init__(
        self,
        *,
        min_overlap: int = 3,
        max_steps: int = 10,
        max_output_chars: int = 160,
        max_candidates: int = 32,
    ) -> None:
        self.min_overlap = max(2, int(min_overlap))
        self.max_steps = max(1, int(max_steps))
        self.max_output_chars = max(32, int(max_output_chars))
        self.max_candidates = max(4, int(max_candidates))

    def decode(
        self,
        *,
        query_window: str,
        winner_column: int | None,
        memory_matches: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        query_text = self._normalize_text(query_window)
        candidates = self._prepare_candidates(memory_matches, winner_column)
        if not candidates:
            return {
                "available": False,
                "reason": "no_memory_candidates",
                "query_text": query_text,
                "decoded_text": "",
                "continuation_text": "",
                "confidence": 0.0,
                "candidate_count": 0,
                "source_memory_indices": [],
                "steps": [],
            }

        seed_candidate, seed_overlap_ratio = self._choose_seed_candidate(query_text, candidates)
        if seed_candidate is not None:
            current_text = seed_candidate.text
            used_indices = {seed_candidate.memory_index}
            seed_text = seed_candidate.text
            seed_memory_index: int | None = seed_candidate.memory_index
        else:
            current_text = query_text or candidates[0].text
            used_indices = set()
            seed_text = current_text
            seed_memory_index = None

        seen_texts = {current_text.lower()}
        steps: list[dict[str, Any]] = []
        step_confidences: list[float] = []

        while len(current_text) < self.max_output_chars and len(steps) < self.max_steps:
            choice = self._best_merge(current_text, candidates, used_indices, seen_texts)
            if choice is None:
                break

            current_text = self._normalize_text(choice.merged_text[: self.max_output_chars])
            used_indices.add(choice.candidate.memory_index)
            seen_texts.add(current_text.lower())
            overlap_ratio = choice.overlap / max(1, min(len(current_text), len(choice.candidate.text)))
            step_confidences.append(min(1.0, 0.55 * choice.candidate.similarity + 0.45 * overlap_ratio))
            steps.append(
                {
                    "memory_index": int(choice.candidate.memory_index),
                    "bucket_id": choice.candidate.bucket_id,
                    "direction": choice.direction,
                    "overlap": int(choice.overlap),
                    "similarity": float(choice.candidate.similarity),
                    "added_text": choice.added_text,
                }
            )

        decoded_text = self._normalize_text(current_text)
        query_overlap_ratio = self._overlap_ratio(query_text, decoded_text)
        starts_with_query = bool(query_text) and decoded_text.lower().startswith(query_text.lower())
        continuation_text = self._continuation_text(query_text, decoded_text)
        confidence = self._confidence(
            query_text=query_text,
            decoded_text=decoded_text,
            seed_candidate=seed_candidate,
            seed_overlap_ratio=seed_overlap_ratio,
            step_confidences=step_confidences,
        )

        return {
            "available": bool(decoded_text),
            "query_text": query_text,
            "seed_text": seed_text,
            "seed_memory_index": seed_memory_index,
            "decoded_text": decoded_text,
            "continuation_text": continuation_text,
            "starts_with_query": starts_with_query,
            "query_overlap_ratio": float(query_overlap_ratio),
            "confidence": float(confidence),
            "candidate_count": int(len(candidates)),
            "source_memory_indices": sorted(int(index) for index in used_indices),
            "steps": steps,
        }

    def _prepare_candidates(
        self,
        memory_matches: Sequence[Mapping[str, Any]],
        winner_column: int | None,
    ) -> list[_DecoderCandidate]:
        candidates: list[_DecoderCandidate] = []
        seen_texts: set[str] = set()
        for match in memory_matches:
            text = self._normalize_text(match.get("raw_window"))
            if not text:
                continue
            key = text.lower()
            if key in seen_texts:
                continue
            seen_texts.add(key)
            similarity = float(match.get("similarity", 0.0))
            importance = float(match.get("importance", 0.0))
            tag_strength = float(match.get("tag_strength", 0.0))
            bucket_id = match.get("bucket_id")
            bucket_int = None if bucket_id is None else int(bucket_id)
            same_bucket = winner_column is not None and bucket_int == int(winner_column)
            score = similarity + 0.04 * min(5.0, importance) + 0.03 * min(5.0, tag_strength)
            if same_bucket:
                score += 0.12
            candidates.append(
                _DecoderCandidate(
                    memory_index=int(match.get("memory_index", -1)),
                    text=text,
                    similarity=similarity,
                    importance=importance,
                    tag_strength=tag_strength,
                    bucket_id=bucket_int,
                    same_bucket=same_bucket,
                    score=float(score),
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: self.max_candidates]

    def _choose_seed_candidate(
        self,
        query_text: str,
        candidates: Sequence[_DecoderCandidate],
    ) -> tuple[_DecoderCandidate | None, float]:
        if not query_text:
            top_candidate = candidates[0] if candidates else None
            return top_candidate, 1.0 if top_candidate is not None else 0.0

        best_candidate: _DecoderCandidate | None = None
        best_overlap = 0.0
        best_score = 0.0
        for candidate in candidates:
            overlap_ratio = self._overlap_ratio(query_text, candidate.text)
            contains_query = query_text.lower() in candidate.text.lower()
            if overlap_ratio < 0.50 and not contains_query:
                continue
            seed_score = 1.35 * overlap_ratio + 0.15 * candidate.similarity + (0.05 if candidate.same_bucket else 0.0)
            if contains_query:
                seed_score += 0.10
            if seed_score > best_score:
                best_score = seed_score
                best_overlap = overlap_ratio
                best_candidate = candidate

        if best_candidate is None:
            return None, 0.0
        return best_candidate, best_overlap

    def _best_merge(
        self,
        current_text: str,
        candidates: Sequence[_DecoderCandidate],
        used_indices: set[int],
        seen_texts: set[str],
    ) -> _MergeChoice | None:
        best_choice: _MergeChoice | None = None
        best_score = float("-inf")
        for candidate in candidates:
            if candidate.memory_index in used_indices:
                continue
            choice = self._merge_choice(current_text, candidate)
            if choice is None:
                continue
            if choice.merged_text.lower() in seen_texts:
                continue
            overlap_ratio = choice.overlap / max(1, min(len(current_text), len(candidate.text)))
            gain = len(choice.added_text.strip())
            score = candidate.score + 0.30 * overlap_ratio + 0.02 * gain
            if choice.direction == "cover":
                score += 0.05
            if score > best_score:
                best_score = score
                best_choice = choice
        return best_choice

    def _merge_choice(self, current_text: str, candidate: _DecoderCandidate) -> _MergeChoice | None:
        current_lower = current_text.lower()
        candidate_lower = candidate.text.lower()

        if current_lower == candidate_lower or candidate_lower in current_lower:
            return None
        if current_lower and current_lower in candidate_lower:
            return _MergeChoice(
                candidate=candidate,
                merged_text=candidate.text,
                overlap=len(current_text),
                direction="cover",
                added_text=candidate.text,
            )

        append_overlap = self._suffix_prefix_overlap(current_text, candidate.text)
        prepend_overlap = self._suffix_prefix_overlap(candidate.text, current_text)
        if append_overlap < self.min_overlap and prepend_overlap < self.min_overlap:
            return None

        if append_overlap >= prepend_overlap:
            return _MergeChoice(
                candidate=candidate,
                merged_text=current_text + candidate.text[append_overlap:],
                overlap=append_overlap,
                direction="append",
                added_text=candidate.text[append_overlap:],
            )
        return _MergeChoice(
            candidate=candidate,
            merged_text=candidate.text + current_text[prepend_overlap:],
            overlap=prepend_overlap,
            direction="prepend",
            added_text=candidate.text[:-prepend_overlap] if prepend_overlap < len(candidate.text) else "",
        )

    def _continuation_text(self, query_text: str, decoded_text: str) -> str:
        if not decoded_text:
            return ""
        if query_text and decoded_text.lower().startswith(query_text.lower()):
            return decoded_text[len(query_text):].lstrip()
        return decoded_text if decoded_text != query_text else ""

    def _confidence(
        self,
        *,
        query_text: str,
        decoded_text: str,
        seed_candidate: _DecoderCandidate | None,
        seed_overlap_ratio: float,
        step_confidences: Sequence[float],
    ) -> float:
        query_alignment = self._overlap_ratio(query_text, decoded_text) if query_text else 1.0
        seed_similarity = 0.0 if seed_candidate is None else float(seed_candidate.similarity)
        seed_confidence = max(seed_overlap_ratio, seed_similarity)
        step_confidence = float(mean(step_confidences)) if step_confidences else seed_confidence
        confidence = 0.45 * seed_confidence + 0.35 * step_confidence + 0.20 * query_alignment
        return max(0.0, min(1.0, float(confidence)))

    def _overlap_ratio(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        shared = self._longest_common_substring_length(left.lower(), right.lower())
        return float(shared / max(1, max(len(left), len(right))))

    def _longest_common_substring_length(self, left: str, right: str) -> int:
        if not left or not right:
            return 0
        previous = [0] * (len(right) + 1)
        best = 0
        for left_char in left:
            current = [0]
            for index, right_char in enumerate(right, start=1):
                if left_char == right_char:
                    value = previous[index - 1] + 1
                    current.append(value)
                    if value > best:
                        best = value
                else:
                    current.append(0)
            previous = current
        return best

    def _suffix_prefix_overlap(self, left: str, right: str) -> int:
        max_overlap = min(len(left), len(right))
        left_lower = left.lower()
        right_lower = right.lower()
        for size in range(max_overlap, self.min_overlap - 1, -1):
            if left_lower[-size:] == right_lower[:size]:
                return size
        return 0

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


__all__ = ["NativeAssemblyDecoder"]