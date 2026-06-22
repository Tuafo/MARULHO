from __future__ import annotations

import argparse
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from marulho.data.rtf_encoder import RTFEncoder
from marulho.retrieval import NativeAssemblyDecoder
from marulho.reporting.io import write_json_file
from marulho.semantics.grounding_text import TOKEN_RE
from marulho.semantics.grounding_text import query_focused_clauses
from marulho.semantics.grounding_text import salient_query_terms
from marulho.semantics.grounding_text import semantic_unit_similarity
from marulho.semantics.grounding_text import split_sentences
from marulho.semantics.grounding_text import stream_matching_units
from marulho.semantics.grounding_text import token_forms
from marulho.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from marulho.training.trainer import MarulhoTrainer


_NATIVE_DECODER = NativeAssemblyDecoder()
_TERM_MATCH_THRESHOLD = 0.70
_MAX_EVIDENCE_CACHE_ENTRIES = 8192
_MAX_FORM_CACHE_ENTRIES = 65536
_MAX_TERM_PAIR_CACHE_ENTRIES = 65536
_FEED_SOURCE_EPISODE_LIMIT = 32
_FEED_SOURCE_EPISODE_MAX_CHARS = 240
_FEED_SOURCE_EPISODE_IMPORTANCE = 0.20
_FEED_SOURCE_EPISODE_CAPTURE_TAG = 0.20


def _compact_match_unit(value: str) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _cache_put(cache: OrderedDict[Any, Any], key: Any, value: Any, max_entries: int) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(1, int(max_entries)):
        cache.popitem(last=False)
    return value


class _SemanticTermMatchCache:
    def __init__(
        self,
        *,
        max_evidence_entries: int = _MAX_EVIDENCE_CACHE_ENTRIES,
        max_form_entries: int = _MAX_FORM_CACHE_ENTRIES,
        max_pair_entries: int = _MAX_TERM_PAIR_CACHE_ENTRIES,
    ) -> None:
        self._max_evidence_entries = max(1, int(max_evidence_entries))
        self._max_form_entries = max(1, int(max_form_entries))
        self._max_pair_entries = max(1, int(max_pair_entries))
        self._evidence_cache: OrderedDict[str, tuple[str, ...]] = OrderedDict()
        self._evidence_form_cache: OrderedDict[str, frozenset[str]] = OrderedDict()
        self._form_cache: OrderedDict[str, frozenset[str]] = OrderedDict()
        self._pair_cache: OrderedDict[tuple[str, str], bool] = OrderedDict()

    def _forms(self, term: str) -> frozenset[str]:
        key = str(term)
        cached = self._form_cache.get(key)
        if cached is not None:
            self._form_cache.move_to_end(key)
            return cached
        return _cache_put(
            self._form_cache,
            key,
            frozenset(token_forms(key)),
            self._max_form_entries,
        )

    def _evidence_terms(self, text: str) -> tuple[str, ...]:
        key = str(text or "")
        cached = self._evidence_cache.get(key)
        if cached is not None:
            self._evidence_cache.move_to_end(key)
            return cached
        return _cache_put(
            self._evidence_cache,
            key,
            tuple(stream_matching_units(key)),
            self._max_evidence_entries,
        )

    def _evidence_forms(self, text: str, evidence_terms: Sequence[str]) -> frozenset[str]:
        key = str(text or "")
        cached = self._evidence_form_cache.get(key)
        if cached is not None:
            self._evidence_form_cache.move_to_end(key)
            return cached
        forms: set[str] = set()
        for evidence_term in evidence_terms:
            forms.update(self._forms(evidence_term))
        return _cache_put(
            self._evidence_form_cache,
            key,
            frozenset(forms),
            self._max_evidence_entries,
        )

    @staticmethod
    def _can_reach_threshold(left: str, right: str) -> bool:
        left_compact = _compact_match_unit(left)
        right_compact = _compact_match_unit(right)
        if not left_compact or not right_compact:
            return False
        if left_compact == right_compact:
            return True
        shorter, longer = (
            (left_compact, right_compact)
            if len(left_compact) <= len(right_compact)
            else (right_compact, left_compact)
        )
        length_ratio = float(len(shorter) / max(1, len(longer)))
        if length_ratio < _TERM_MATCH_THRESHOLD:
            return False
        if len(shorter) >= 4 and shorter in longer:
            return True
        return len(shorter) >= 4

    def _semantic_match(self, left: str, right: str) -> bool:
        pair = (str(left), str(right))
        cached = self._pair_cache.get(pair)
        if cached is not None:
            self._pair_cache.move_to_end(pair)
            return cached
        if not self._can_reach_threshold(pair[0], pair[1]):
            return _cache_put(self._pair_cache, pair, False, self._max_pair_entries)
        if _compact_match_unit(pair[0]) == _compact_match_unit(pair[1]):
            return _cache_put(self._pair_cache, pair, True, self._max_pair_entries)
        result = bool(semantic_unit_similarity(pair[0], pair[1]) >= _TERM_MATCH_THRESHOLD)
        return _cache_put(self._pair_cache, pair, result, self._max_pair_entries)

    def match_terms(self, query_terms: Sequence[str], text: str) -> list[str]:
        if not query_terms:
            return []
        evidence_terms = self._evidence_terms(text)
        if not evidence_terms:
            return []

        evidence_forms = self._evidence_forms(text, evidence_terms)
        matches: list[str] = []
        for raw_term in query_terms:
            term = str(raw_term)
            if term in matches:
                continue
            term_forms = self._forms(term)
            if term_forms and evidence_forms and term_forms & evidence_forms:
                matches.append(term)
                continue
            if any(self._semantic_match(term, evidence_term) for evidence_term in evidence_terms):
                matches.append(term)
        return matches


def episode_quality(text: str, raw_window: str | None = None) -> tuple[int, int]:
    normalized_text = str(text or "").strip()
    normalized_window = str(raw_window or "").strip()
    complete_sentence = int(normalized_text.endswith((".", "!", "?")))
    clipped_overlap = 0
    if normalized_text and not complete_sentence:
        trailing_tokens = TOKEN_RE.findall(normalized_text.lower())
        raw_tokens = TOKEN_RE.findall(normalized_window.lower()) if normalized_window else []
        if trailing_tokens and (
            len(normalized_text) < 48
            or (raw_tokens and trailing_tokens[-1] == raw_tokens[-1] and len(normalized_text) <= len(normalized_window) + 8)
        ):
            clipped_overlap = 1
    return complete_sentence, clipped_overlap


def _text_window_overlap(left: str, right: str, *, min_overlap: int = 2) -> int:
    left_text = str(left or "")
    right_text = str(right or "")
    max_overlap = min(len(left_text), len(right_text))
    for size in range(max_overlap, max(0, int(min_overlap) - 1), -1):
        if left_text[-size:].lower() == right_text[:size].lower():
            return int(size)
    return 0


def _merge_adjacent_text_windows(windows: Sequence[str], *, min_overlap: int = 2) -> str:
    merged = ""
    for raw_window in windows:
        window = str(raw_window or "")
        if not window:
            continue
        if not merged:
            merged = window
            continue
        overlap = _text_window_overlap(merged, window, min_overlap=min_overlap)
        if overlap > 0:
            merged += window[overlap:]
    return merged


def _memory_window_count(memory_store: Any | None) -> int:
    if memory_store is None:
        return 0
    summary = getattr(memory_store, "live_summary_stats", None)
    if callable(summary):
        try:
            return int(summary().get("size", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def _memory_store_size(memory_store: Any | None) -> int:
    return _memory_window_count(memory_store)


def _bounded_episode_source_text(
    match: Mapping[str, Any],
    *,
    memory_store: Any | None,
    neighbor_radius: int,
    window_cache: dict[int, str],
    loaded_indices: set[int],
) -> tuple[str, list[int]]:
    fallback = str(match.get("text") or match.get("raw_window") or "").strip()
    raw_window = str(match.get("raw_window") or "").strip()
    metadata = match.get("metadata") if isinstance(match.get("metadata"), Mapping) else {}
    complete_sentence, _clipped_overlap = episode_quality(fallback, raw_window)
    full_source_episode = (
        bool(complete_sentence)
        and str(metadata.get("source_type", "")).strip()
        == "explicit_feed_source_episode"
    )
    if full_source_episode:
        return fallback, []
    if memory_store is None or int(neighbor_radius) <= 0:
        return fallback, []
    try:
        center = int(match.get("memory_index", -1))
    except (TypeError, ValueError):
        return fallback, []
    window_count = _memory_window_count(memory_store)
    if center < 0 or center >= window_count:
        return fallback, []
    radius = max(0, int(neighbor_radius))
    start = max(0, center - radius)
    stop = min(window_count - 1, center + radius)
    source_row = getattr(memory_store, "query_neighbor_source_row", None)
    if not callable(source_row):
        return fallback, []
    source_windows: list[str] = []
    indices: list[int] = []
    for index in range(start, stop + 1):
        if index not in window_cache:
            try:
                row = source_row(
                    int(index),
                    skip_source_types=("explicit_feed_source_episode",),
                )
            except (TypeError, ValueError, IndexError, KeyError):
                row = {}
            window_cache[index] = str(row.get("text") or "")
            if bool(row.get("raw_text_payload_loaded")):
                loaded_indices.add(index)
        text = window_cache[index]
        if text:
            source_windows.append(text)
            indices.append(index)
    if center in indices:
        center_position = indices.index(center)
        anchored_windows = [source_windows[center_position]]
        anchored_indices = [indices[center_position]]
        for position in range(center_position - 1, -1, -1):
            if _text_window_overlap(source_windows[position], anchored_windows[0]) <= 0:
                break
            anchored_windows.insert(0, source_windows[position])
            anchored_indices.insert(0, indices[position])
        for position in range(center_position + 1, len(source_windows)):
            if _text_window_overlap(anchored_windows[-1], source_windows[position]) <= 0:
                break
            anchored_windows.append(source_windows[position])
            anchored_indices.append(indices[position])
    else:
        anchored_windows = source_windows
        anchored_indices = indices
    stitched = _merge_adjacent_text_windows(anchored_windows).strip()
    return (stitched or fallback), anchored_indices


def feature_label(index: int, representation: str, feature_dim: int) -> str:
    if representation == "hashed_ngram" or feature_dim != 128:
        return f"hash_{index}"
    if index == 0:
        return "<pad>"
    ch = chr(index)
    if ch == " ":
        return "<space>"
    if ch == "\t":
        return "<tab>"
    if ch == "\n":
        return "<newline>"
    if ch.isprintable():
        return ch
    return f"\\x{index:02x}"


def text_pattern_stream(text: str, encoder: RTFEncoder, window_size: int) -> Iterator[tuple[str, torch.Tensor]]:
    yield from encoder.iter_char_patterns(text, window_size, learn=False)


def _normalized_source_episode_key(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text or "").lower()))


def _clip_source_episode(text: str, max_chars: int) -> str:
    value = " ".join(str(text or "").split()).strip()
    limit = max(1, int(max_chars))
    if len(value) <= limit:
        return value
    clipped = value[:limit].strip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0].strip()
    return clipped or value[:limit].strip()


def _bounded_feed_source_episode_candidates(
    text: str,
    *,
    max_source_episodes: int,
    max_chars_per_episode: int,
) -> tuple[list[str], dict[str, Any]]:
    source_units = split_sentences(text)
    limit = max(0, int(max_source_episodes))
    candidates: list[str] = []
    seen: set[str] = set()
    considered = 0
    duplicate_count = 0
    low_signal_count = 0

    if limit <= 0:
        return [], {
            "source_unit_available_count": int(len(source_units)),
            "source_unit_considered_count": 0,
            "source_duplicate_count": 0,
            "source_low_signal_count": 0,
            "candidate_truncated": bool(source_units),
        }

    for source_unit in source_units:
        considered += 1
        candidate = _clip_source_episode(source_unit, max_chars_per_episode)
        tokens = TOKEN_RE.findall(candidate.lower())
        if len(tokens) < 2:
            low_signal_count += 1
            continue
        key = " ".join(tokens)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break

    return candidates, {
        "source_unit_available_count": int(len(source_units)),
        "source_unit_considered_count": int(considered),
        "source_duplicate_count": int(duplicate_count),
        "source_low_signal_count": int(low_signal_count),
        "candidate_truncated": bool(considered < len(source_units)),
    }


def _empty_feed_source_episode_admission_report(
    *,
    status: str,
    fallback_reason: str | None,
    memory_size: int = 0,
    token_count: int = 0,
    max_source_episodes: int = _FEED_SOURCE_EPISODE_LIMIT,
    max_chars_per_episode: int = _FEED_SOURCE_EPISODE_MAX_CHARS,
) -> dict[str, Any]:
    return {
        "surface": "bounded_feed_source_episode_admission.v1",
        "status": str(status),
        "scope": "query_runner_explicit_feed_slow_path",
        "memory_size_before": int(memory_size),
        "memory_size_after": int(memory_size),
        "token_count": int(token_count),
        "source_text_chars_scanned": 0,
        "source_unit_available_count": 0,
        "source_unit_considered_count": 0,
        "source_duplicate_count": 0,
        "source_low_signal_count": 0,
        "candidate_window_limit": int(max(0, int(max_source_episodes))),
        "candidate_window_policy": "explicit_feed_sentence_units_deduped",
        "candidate_scope": "explicit_feed_source_episode_window",
        "candidate_episode_count": 0,
        "candidate_truncated": False,
        "attempted_count": 0,
        "admitted_count": 0,
        "rejected_count": 0,
        "admitted_indices": [],
        "admitted_bucket_ids": [],
        "source_episode_max_chars": int(max(1, int(max_chars_per_episode))),
        "importance": float(_FEED_SOURCE_EPISODE_IMPORTANCE),
        "capture_tag": float(_FEED_SOURCE_EPISODE_CAPTURE_TAG),
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "plasticity_scope": "none",
        "language_reasoning": False,
        "hidden_language_reasoning": False,
        "archival_storage_device": "cpu",
        "active_computation_device": None,
        "selection_budget": {
            "candidate_episode_budget_entries": int(max(0, int(max_source_episodes))),
            "source_episode_max_chars": int(max(1, int(max_chars_per_episode))),
        },
        "fallback_reason": fallback_reason,
    }


def admit_bounded_feed_source_episodes(
    trainer: MarulhoTrainer,
    encoder: RTFEncoder,
    text: str,
    *,
    max_source_episodes: int = _FEED_SOURCE_EPISODE_LIMIT,
    max_chars_per_episode: int = _FEED_SOURCE_EPISODE_MAX_CHARS,
    importance: float = _FEED_SOURCE_EPISODE_IMPORTANCE,
    capture_tag: float = _FEED_SOURCE_EPISODE_CAPTURE_TAG,
) -> dict[str, Any]:
    started = time.perf_counter()
    store = trainer.model.memory_store
    memory_size_before = _memory_store_size(store)
    candidates, selection_report = _bounded_feed_source_episode_candidates(
        text,
        max_source_episodes=max_source_episodes,
        max_chars_per_episode=max_chars_per_episode,
    )
    if not candidates:
        return {
            **_empty_feed_source_episode_admission_report(
                status="empty",
                fallback_reason="no_source_episode_candidates",
                memory_size=memory_size_before,
                token_count=int(getattr(trainer, "token_count", 0)),
                max_source_episodes=max_source_episodes,
                max_chars_per_episode=max_chars_per_episode,
            ),
            "source_text_chars_scanned": int(len(str(text or ""))),
            **selection_report,
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
        }

    admitted_indices: list[int] = []
    admitted_bucket_ids: list[int] = []
    rejected_count = 0
    pattern_device: str | None = None
    routing_key_device: str | None = None
    assembly_compute_device: str | None = None

    for rank, source_episode in enumerate(candidates):
        examples = list(
            text_pattern_stream(source_episode, encoder, trainer.config.window_size)
        )
        if not examples:
            rejected_count += 1
            continue
        raw_window, pattern = examples[-1]
        pattern_device = str(pattern.device)
        winners, assembly, routing_key = trainer._offline_competition(
            pattern,
            return_routing_key=True,
        )
        routing_key_device = str(routing_key.device)
        assembly_compute_device = str(assembly.device)
        bucket_id = int(winners[0].item())
        memory_index = store.update(
            assembly.detach().cpu(),
            importance=max(1e-6, float(importance)),
            token_count=int(getattr(trainer, "token_count", 0)),
            bucket_id=int(bucket_id),
            input_pattern=pattern,
            routing_key=routing_key,
            raw_window=raw_window,
            text=source_episode,
            metadata={
                "source_type": "explicit_feed_source_episode",
                "source_name": "query_runner_explicit_feed",
                "provider": "query_runner",
                "admission_surface": "bounded_feed_source_episode_admission.v1",
                "source_episode_rank": int(rank),
                "source_episode_key": _normalized_source_episode_key(source_episode),
                "source_episode_chars": int(len(source_episode)),
                "source_episode_pattern_window": raw_window,
            },
            capture_tag=max(0.0, float(capture_tag)),
        )
        if memory_index is None:
            rejected_count += 1
            continue
        admitted_indices.append(int(memory_index))
        admitted_bucket_ids.append(int(bucket_id))

    memory_size_after = _memory_store_size(store)
    admitted_count = len(admitted_indices)
    return {
        "surface": "bounded_feed_source_episode_admission.v1",
        "status": "admitted" if admitted_count else "empty",
        "scope": "query_runner_explicit_feed_slow_path",
        "memory_size_before": int(memory_size_before),
        "memory_size_after": int(memory_size_after),
        "token_count": int(getattr(trainer, "token_count", 0)),
        "source_text_chars_scanned": int(len(str(text or ""))),
        **selection_report,
        "candidate_window_limit": int(max(0, int(max_source_episodes))),
        "candidate_window_policy": "explicit_feed_sentence_units_deduped",
        "candidate_scope": "explicit_feed_source_episode_window",
        "candidate_episode_count": int(len(candidates)),
        "attempted_count": int(len(candidates)),
        "admitted_count": int(admitted_count),
        "rejected_count": int(rejected_count),
        "admitted_indices": admitted_indices[:64],
        "admitted_index_sample_limit": 64,
        "admitted_index_truncated": bool(len(admitted_indices) > 64),
        "admitted_bucket_ids": admitted_bucket_ids[:64],
        "admitted_bucket_id_sample_limit": 64,
        "admitted_bucket_id_truncated": bool(len(admitted_bucket_ids) > 64),
        "source_episode_max_chars": int(max(1, int(max_chars_per_episode))),
        "importance": float(importance),
        "capture_tag": float(capture_tag),
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": bool(admitted_count),
        "applies_plasticity": bool(admitted_count),
        "plasticity_scope": (
            "slow_memory_admission_tag_only_no_column_weight_update"
            if admitted_count
            else "none"
        ),
        "language_reasoning": False,
        "hidden_language_reasoning": False,
        "archival_storage_device": "cpu",
        "active_computation_device": str(trainer.model.device),
        "input_pattern_device": pattern_device,
        "routing_key_device": routing_key_device,
        "assembly_compute_device": assembly_compute_device,
        "selection_budget": {
            "candidate_episode_budget_entries": int(max(0, int(max_source_episodes))),
            "source_episode_max_chars": int(max(1, int(max_chars_per_episode))),
            "source_payload_char_budget": int(
                max(0, int(max_source_episodes))
                * max(1, int(max_chars_per_episode))
            ),
        },
        "latency_ms": float((time.perf_counter() - started) * 1000.0),
        "assembly_policy": "bounded_offline_competition_winner_assembly",
        "dense_source_admission_assembly_retired": True,
        "fallback_reason": None if admitted_count else "source_episode_update_rejected",
    }


def top_feature_details(pattern: torch.Tensor, top_n: int, representation: str) -> list[dict[str, Any]]:
    flat = pattern.detach().cpu().float()
    count = min(max(1, int(top_n)), int(flat.numel()))
    values, indices = torch.topk(flat, k=count)
    features: list[dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        if float(value) <= 0.0:
            continue
        label = feature_label(int(index), representation, int(flat.numel()))
        item: dict[str, Any] = {
            "index": int(index),
            "char": label,
            "weight": float(value),
        }
        if representation != "hashed_ngram" and int(flat.numel()) == 128:
            item["ord"] = int(index)
        features.append(item)
    return features


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0), dim=1).item())


def feed_text(
    trainer: MarulhoTrainer,
    encoder: RTFEncoder,
    text: str,
    *,
    on_step: Callable[[str, dict[str, Any]], None] | None = None,
    admit_source_episodes: bool = True,
    source_episode_budget: int = _FEED_SOURCE_EPISODE_LIMIT,
    source_episode_max_chars: int = _FEED_SOURCE_EPISODE_MAX_CHARS,
) -> dict[str, Any]:
    trainer.encoder = encoder
    last_metrics: dict[str, Any] | None = None
    tokens = 0
    for raw_window, pattern in encoder.iter_char_patterns(text, trainer.config.window_size, learn=True):
        last_metrics = trainer.train_step(pattern, raw_window=raw_window)
        if on_step is not None:
            on_step(raw_window, last_metrics)
        tokens += 1

    if admit_source_episodes:
        source_admission_report = admit_bounded_feed_source_episodes(
            trainer,
            encoder,
            text,
            max_source_episodes=source_episode_budget,
            max_chars_per_episode=source_episode_max_chars,
        )
    else:
        source_admission_report = _empty_feed_source_episode_admission_report(
            status="disabled",
            fallback_reason="source_episode_admission_disabled",
            memory_size=_memory_store_size(trainer.model.memory_store),
            token_count=int(trainer.token_count),
            max_source_episodes=source_episode_budget,
            max_chars_per_episode=source_episode_max_chars,
        )

    return {
        "tokens_processed": int(tokens),
        "token_count": int(trainer.token_count),
        "last_winner": None if last_metrics is None else int(last_metrics["winner"]),
        "last_recon_error": None if last_metrics is None else float(last_metrics["recon_error"]),
        "memory_buffer_size": int(_memory_store_size(trainer.model.memory_store)),
        "source_memory_admission_report": source_admission_report,
        "source_memory_admission_count": int(source_admission_report.get("admitted_count", 0)),
        "source_memory_candidate_count": int(source_admission_report.get("candidate_episode_count", 0)),
    }


def prime_context(trainer: MarulhoTrainer, encoder: RTFEncoder, text: str) -> int:
    trainer.encoder = encoder
    patterns = [pattern for _, pattern in text_pattern_stream(text, encoder, trainer.config.window_size)]
    trainer.prime_context(patterns, update_weights=False)
    return len(patterns)


def candidate_details(trainer: MarulhoTrainer, routing_key: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    candidate_ids, _ = trainer.model.routing_index.search_tensors(
        routing_key.unsqueeze(0),
        k=max(1, int(top_k)),
    )
    row = candidate_ids[0].detach().cpu().tolist() if candidate_ids.numel() else []
    details: list[dict[str, Any]] = []
    for column_id in row:
        prototype = trainer.model.competitive.prototypes[int(column_id)].detach().cpu()
        details.append(
            {
                "column_id": int(column_id),
                "shard_id": int(column_id) % max(1, trainer.config.routing_shards),
                "similarity": cosine_similarity(routing_key.detach().cpu(), prototype),
            }
        )
    return details


def _dedupe_terms(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _memory_focus_priority(
    memory_priority: Mapping[object, object] | None,
    memory_indices: Sequence[int],
) -> float:
    if not memory_priority:
        return 0.0
    best = 0.0
    for memory_index in memory_indices:
        for key in (memory_index, str(memory_index)):
            value = memory_priority.get(key, 0.0)
            try:
                best = max(best, float(value))
            except (TypeError, ValueError):
                continue
    return float(best)


def _query_candidate_bucket_ids(
    trainer: MarulhoTrainer,
    routing_key: torch.Tensor,
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
    return [int(value) for value in candidate_ids[0].detach().cpu().flatten().tolist()]


def _empty_query_memory_match_report(
    *,
    memory_size: int,
    requested_count: int,
    fallback_reason: str,
) -> dict[str, Any]:
    return {
        "surface": "bounded_query_memory_match.v1",
        "candidate_surface": "bounded_query_memory_match_candidates.v1",
        "status": "empty",
        "scope": "query_memory_match_slow_path",
        "memory_size": int(memory_size),
        "requested_count": int(requested_count),
        "candidate_window_limit": int(requested_count),
        "candidate_window_policy": "query_memory_match_candidate_scope_missing",
        "candidate_scope": "query_memory_match_candidate_scope_missing",
        "candidate_bucket_ids": [],
        "candidate_bucket_count": 0,
        "candidate_index_available_count": 0,
        "candidate_index_count": 0,
        "match_indices": [],
        "similarity_score_count": 0,
        "replay_priority_score_count": 0,
        "result_count": 0,
        "returned_count": 0,
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "fallback_reason": fallback_reason,
    }


def memory_matches_with_report(
    trainer: MarulhoTrainer,
    pattern_vec: torch.Tensor,
    routing_key: torch.Tensor,
    top_k: int,
    top_chars: int,
    query_terms: Optional[list[str]] = None,
    *,
    focus_terms: Sequence[str] | None = None,
    memory_priority: Mapping[object, object] | None = None,
    memory_candidate_limit: int | None = None,
    candidate_bucket_ids: Sequence[int] | None = None,
    replay_entry_cache: dict[int, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    store = trainer.model.memory_store
    representation = getattr(trainer.config, "input_representation", "order_weighted_ascii")
    limit = max(1, int(top_k))
    candidate_limit = max(
        limit,
        int(memory_candidate_limit)
        if memory_candidate_limit is not None
        else max(32, limit * 8),
    )
    if candidate_bucket_ids is None:
        candidate_bucket_ids = _query_candidate_bucket_ids(
            trainer,
            routing_key,
            max(limit, int(getattr(trainer.config, "k_routing", limit))),
        )
    if not hasattr(store, "collect_query_memory_match_indices"):
        report = _empty_query_memory_match_report(
            memory_size=_memory_store_size(store),
            requested_count=candidate_limit,
            fallback_reason="memory_store_missing_bounded_query_match_collector",
        )
        return [], report
    query_row_reader = getattr(store, "query_match_row", None)
    if not callable(query_row_reader):
        report = _empty_query_memory_match_report(
            memory_size=_memory_store_size(store),
            requested_count=candidate_limit,
            fallback_reason="memory_store_missing_bounded_query_match_row_reader",
        )
        report.update(
            {
                "surface": "bounded_query_memory_match.v1",
                "candidate_surface": "bounded_query_memory_match_candidates.v1",
                "query_row_surface": "bounded_query_memory_match_row.v1",
                "query_row_reader_owned_by_store": False,
                "direct_slow_memory_array_reads_retired": True,
            }
        )
        return [], report

    candidate_report = store.collect_query_memory_match_indices(
        candidate_bucket_ids=candidate_bucket_ids,
        max_candidates=candidate_limit,
        scope="query_runner_memory_matches",
    )
    raw_candidate_indices: list[int] = []
    for raw_index in candidate_report.get("match_indices", []):
        try:
            raw_candidate_indices.append(int(raw_index))
        except (TypeError, ValueError):
            continue
    candidate_indices: list[int] = []
    initial_candidate_index_count = int(len(raw_candidate_indices))
    ordered_focus_terms = _dedupe_terms(focus_terms)
    term_match_cache = _SemanticTermMatchCache()
    text_ranking_required = bool(query_terms or ordered_focus_terms or memory_priority)
    matches: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    raw_text_payload_count = 0
    raw_text_payload_cache_hits = 0
    query_row_read_count = 0
    query_row_cache_hits = 0
    invalid_query_row_count = 0
    query_row_cache: dict[int, dict[str, Any]] = (
        replay_entry_cache if replay_entry_cache is not None else {}
    )
    query_input = pattern_vec.detach().cpu()
    query_key = routing_key.detach().cpu()

    def _read_query_row(idx: int, *, include_text_payload: bool) -> dict[str, Any] | None:
        nonlocal raw_text_payload_count
        nonlocal raw_text_payload_cache_hits
        nonlocal query_row_read_count
        nonlocal query_row_cache_hits
        nonlocal invalid_query_row_count
        cached = query_row_cache.get(int(idx))
        if cached is not None and (
            not include_text_payload or bool(cached.get("raw_text_payload_loaded"))
        ):
            query_row_cache_hits += 1
            if include_text_payload:
                raw_text_payload_cache_hits += 1
            return cached
        try:
            row = query_row_reader(
                int(idx),
                current_token=trainer.token_count,
                include_text_payload=include_text_payload,
            )
        except (TypeError, ValueError, IndexError, KeyError):
            invalid_query_row_count += 1
            return None
        query_row_read_count += 1
        row = dict(row)
        previous = query_row_cache.get(int(idx))
        if previous is not None:
            merged = dict(previous)
            merged.update(row)
            row = merged
        query_row_cache[int(idx)] = row
        if include_text_payload and bool(row.get("raw_text_payload_loaded")):
            raw_text_payload_count += 1
        return row

    replay_scores = store.replay_scores_for_indices(
        raw_candidate_indices,
        trainer.token_count,
    )

    def _finish(result_matches: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        report = dict(candidate_report)
        selection_budget = dict(report.get("selection_budget") or {})
        selection_budget.update(
            {
                "query_row_candidate_read_budget_entries": int(
                    len(raw_candidate_indices)
                ),
                "query_row_text_payload_budget_entries": int(
                    len(candidate_indices) if text_ranking_required else limit
                ),
                "query_row_total_read_budget_entries": int(
                    len(raw_candidate_indices)
                    + (len(candidate_indices) if text_ranking_required else limit)
                ),
            }
        )
        report.update(
            {
                "surface": "bounded_query_memory_match.v1",
                "candidate_surface": candidate_report.get("surface"),
                "status": "matched" if result_matches else "empty",
                "candidate_window_policy": candidate_report.get(
                    "candidate_window_policy"
                ),
                "candidate_scope": candidate_report.get("candidate_scope"),
                "initial_candidate_index_count": int(initial_candidate_index_count),
                "candidate_index_count": int(len(candidate_indices)),
                "match_indices": [int(index) for index in candidate_indices],
                "query_term_count": int(len(query_terms or [])),
                "focus_term_count": int(len(ordered_focus_terms)),
                "memory_priority_count": int(len(memory_priority or {})),
                "similarity_score_count": int(len(candidate_indices)),
                "replay_priority_score_count": int(len(replay_scores)),
                "result_count": int(
                    len(matches) if text_ranking_required else len(candidate_rows)
                ),
                "returned_count": int(len(result_matches)),
                "raw_text_payload_loaded": bool(raw_text_payload_count > 0),
                "raw_text_payload_count": int(raw_text_payload_count),
                "raw_text_payload_cache_hits": int(raw_text_payload_cache_hits),
                "raw_text_payload_policy": (
                    "candidate_window_text_ranking"
                    if text_ranking_required
                    else "returned_similarity_matches_only"
                ),
                "query_row_surface": "bounded_query_memory_match_row.v1",
                "query_row_access_policy": "explicit_bounded_query_candidate_indices_only",
                "query_row_reader_owned_by_store": True,
                "query_row_read_count": int(query_row_read_count),
                "query_row_cache_hits": int(query_row_cache_hits),
                "query_row_invalid_index_count": int(invalid_query_row_count),
                "direct_slow_memory_array_reads_retired": True,
                "language_reasoning": False,
                "runs_live_tick": False,
                "mutates_runtime_state": False,
                "applies_plasticity": False,
                "archival_storage_device": "cpu",
                "selection_budget": selection_budget,
            }
        )
        return result_matches, report

    def _build_match(
        idx: int,
        *,
        similarity: float,
        evidence_pattern: torch.Tensor,
        replay_priority: float,
    ) -> dict[str, Any]:
        replay_entry = _read_query_row(idx, include_text_payload=True)
        if replay_entry is None:
            return {}
        capture_tag = float(replay_entry.get("capture_tag", 0.0))
        prp_level = float(replay_entry.get("prp_level", 0.0))
        capture_strength = float(replay_entry.get("capture_strength", 0.0))
        consolidation_level = float(replay_entry.get("consolidation_level", 0.0))
        text = replay_entry.get("text") or replay_entry.get("raw_window") or ""
        raw_window = replay_entry.get("raw_window") or text
        replay_metadata = replay_entry.get("metadata") if isinstance(replay_entry.get("metadata"), Mapping) else {}
        source_name = " ".join(str(replay_metadata.get("source_name", "")).split()).strip()
        source_type = " ".join(str(replay_metadata.get("source_type", "")).split()).strip()
        provider = " ".join(str(replay_metadata.get("provider", "")).split()).strip().lower()
        complete_sentence, clipped_overlap = episode_quality(str(text or "").strip(), raw_window)
        matched_query_terms = term_match_cache.match_terms(query_terms, str(text or "")) if query_terms else []
        query_overlap = len(matched_query_terms)
        matched_focus_terms = (
            term_match_cache.match_terms(ordered_focus_terms, str(text or "")) if ordered_focus_terms else []
        )
        focus_overlap = len(matched_focus_terms)
        focus_priority = _memory_focus_priority(memory_priority, (idx,))
        return {
            "memory_index": int(idx),
            "similarity": float(similarity),
            "bucket_id": None if replay_entry.get("bucket_id") is None else int(replay_entry.get("bucket_id")),
            "raw_window": raw_window,
            "text": text,
            "metadata": dict(replay_metadata),
            "source_name": source_name,
            "source_type": source_type,
            "provider": provider,
            "age_tokens": int(replay_entry.get("age_tokens", 0) or 0),
            "importance": float(replay_entry.get("importance", 0.0) or 0.0),
            "tag_strength": float(capture_tag),
            "capture_tag": float(capture_tag),
            "prp_level": float(prp_level),
            "capture_strength": float(capture_strength),
            "consolidation_level": consolidation_level,
            "consolidation_gap": float(max(0.0, 1.0 - consolidation_level)),
            "replay_count": int(replay_entry.get("replay_count", 0) or 0),
            "replay_priority": float(replay_priority),
            "top_chars": top_feature_details(evidence_pattern, top_chars, representation),
            "query_overlap": int(query_overlap),
            "matched_query_terms": matched_query_terms,
            "focus_overlap": int(focus_overlap),
            "matched_focus_terms": matched_focus_terms,
            "memory_focus_priority": float(focus_priority),
            "complete_sentence": int(complete_sentence),
            "clipped_overlap": int(clipped_overlap),
        }

    def _score_index(idx: int) -> None:
        row = _read_query_row(idx, include_text_payload=False)
        if row is None:
            return
        ref_key = row.get("routing_key")
        ref_input = row.get("input_pattern")
        assembly = row.get("assembly")
        if isinstance(ref_key, torch.Tensor):
            similarity = cosine_similarity(query_key, ref_key.float())
        elif isinstance(ref_input, torch.Tensor):
            similarity = cosine_similarity(query_input, ref_input.float())
        elif isinstance(assembly, torch.Tensor):
            similarity = cosine_similarity(query_input, assembly.float())
        else:
            return

        evidence_pattern = ref_input.float() if isinstance(ref_input, torch.Tensor) else assembly.float()
        replay_priority = float(replay_scores.get(int(idx), 0.0))
        candidate_indices.append(int(idx))
        if text_ranking_required:
            match = _build_match(
                idx,
                similarity=float(similarity),
                evidence_pattern=evidence_pattern,
                replay_priority=replay_priority,
            )
            if match:
                matches.append(match)
        else:
            candidate_rows.append(
                {
                    "memory_index": int(idx),
                    "similarity": float(similarity),
                    "evidence_pattern": evidence_pattern,
                    "replay_priority": float(replay_priority),
                }
            )

    for idx in raw_candidate_indices:
        _score_index(idx)

    if not text_ranking_required:
        candidate_rows.sort(key=lambda item: float(item["similarity"]), reverse=True)
        selected_rows = candidate_rows[:limit]
        selected_matches = [
            _build_match(
                int(row["memory_index"]),
                similarity=float(row["similarity"]),
                evidence_pattern=row["evidence_pattern"],
                replay_priority=float(row["replay_priority"]),
            )
            for row in selected_rows
            if isinstance(row.get("evidence_pattern"), torch.Tensor)
        ]
        return _finish(selected_matches)

    if not query_terms:
        matches.sort(
            key=lambda item: (
                int(item.get("focus_overlap", 0)),
                float(item.get("memory_focus_priority", 0.0)),
                int(item.get("complete_sentence", 0)),
                -int(item.get("clipped_overlap", 0)),
                float(item["similarity"]),
                float(item["importance"]),
                -int(item["age_tokens"]),
            ),
            reverse=True,
        )
        return _finish(matches[:limit])

    similarity_ranked = sorted(
        matches,
        key=lambda item: (
            int(item.get("query_overlap", 0)),
            int(item.get("focus_overlap", 0)),
            float(item.get("memory_focus_priority", 0.0)),
            float(item["similarity"]),
            float(item["importance"]),
            -int(item["age_tokens"]),
        ),
        reverse=True,
    )
    support_ranked = sorted(
        matches,
        key=lambda item: (
            int(item.get("query_overlap", 0)),
            int(item.get("focus_overlap", 0)),
            float(item.get("memory_focus_priority", 0.0)),
            int(item.get("complete_sentence", 0)),
            -int(item.get("clipped_overlap", 0)),
            float(item["similarity"]),
            float(item["importance"]),
            -int(item["age_tokens"]),
        ),
        reverse=True,
    )

    merged: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    term_support: list[dict[str, Any]] = []
    for term in query_terms:
        supporting = [
            item
            for item in matches
            if term in {str(value) for value in item.get("matched_query_terms", [])}
        ]
        if not supporting:
            continue
        supporting.sort(
            key=lambda item: (
                int(item.get("complete_sentence", 0)),
                -int(item.get("clipped_overlap", 0)),
                int(item.get("query_overlap", 0)),
                float(item["similarity"]),
                float(item["importance"]),
                -int(item["age_tokens"]),
            ),
            reverse=True,
        )
        term_support.append(supporting[0])

    for item in term_support + support_ranked[: max(limit, 12)] + similarity_ranked[:limit]:
        memory_index = int(item.get("memory_index", -1))
        if memory_index in seen_indices:
            continue
        seen_indices.add(memory_index)
        merged.append(item)
        if len(merged) >= limit:
            break

    merged.sort(
        key=lambda item: (
            int(item.get("query_overlap", 0)),
            int(item.get("focus_overlap", 0)),
            float(item.get("memory_focus_priority", 0.0)),
            int(item.get("complete_sentence", 0)),
            -int(item.get("clipped_overlap", 0)),
            float(item["similarity"]),
            float(item["importance"]),
            -int(item["age_tokens"]),
        ),
        reverse=True,
    )
    return _finish(merged[:limit])


def build_memory_episodes_with_report(
    memory_matches: list[dict[str, Any]],
    *,
    top_k: int,
    query_terms: Optional[list[str]] = None,
    focus_terms: Sequence[str] | None = None,
    memory_priority: Mapping[object, object] | None = None,
    memory_store: Any | None = None,
    neighbor_radius: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_focus_terms = _dedupe_terms(focus_terms)
    clause_terms = _dedupe_terms([*(query_terms or []), *ordered_focus_terms])
    term_match_cache = _SemanticTermMatchCache()
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    source_text_match_count = 0
    neighbor_window_cache: dict[int, str] = {}
    loaded_neighbor_indices: set[int] = set()
    selected_neighbor_indices: set[int] = set()
    for match in memory_matches:
        source_text, neighbor_indices = _bounded_episode_source_text(
            match,
            memory_store=memory_store,
            neighbor_radius=max(0, int(neighbor_radius)),
            window_cache=neighbor_window_cache,
            loaded_indices=loaded_neighbor_indices,
        )
        source_text = source_text.strip()
        if not source_text:
            continue
        selected_neighbor_indices.update(neighbor_indices)
        source_text_match_count += 1
        match_metadata = (
            match.get("metadata") if isinstance(match.get("metadata"), Mapping) else {}
        )
        source_is_admitted_episode = (
            str(match_metadata.get("source_type", "")).strip()
            == "explicit_feed_source_episode"
            and bool(episode_quality(source_text, match.get("raw_window"))[0])
        )
        clause_candidates = (
            [source_text]
            if source_is_admitted_episode
            else query_focused_clauses(source_text, clause_terms)
        )
        for text in clause_candidates:
            if not text:
                continue
            key = text.lower()
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "text": text,
                    "raw_window": match.get("raw_window"),
                    "memory_index": int(match.get("memory_index", -1)),
                    "memory_indices": [],
                    "similarity": float(match.get("similarity", 0.0)),
                    "importance": float(match.get("importance", 0.0)),
                    "age_tokens": int(match.get("age_tokens", 0)),
                    "match_count": 0,
                    "query_overlap": 0,
                    "focus_overlap": 0,
                    "memory_focus_priority": 0.0,
                    "complete_sentence": 0,
                    "clipped_overlap": 1,
                    "expansion_chars": 0,
                    "metadata": dict(match.get("metadata") or {}),
                    "source_name": " ".join(str(match.get("source_name", "")).split()).strip(),
                    "source_type": " ".join(str(match.get("source_type", "")).split()).strip(),
                    "provider": " ".join(str(match.get("provider", "")).split()).strip().lower(),
                    "source_names": [],
                    "providers": [],
                }
                grouped[key] = entry
                order.append(key)
            entry["match_count"] += 1
            entry["similarity"] = max(float(entry["similarity"]), float(match.get("similarity", 0.0)))
            entry["importance"] = max(float(entry["importance"]), float(match.get("importance", 0.0)))
            entry["age_tokens"] = min(int(entry["age_tokens"]), int(match.get("age_tokens", 0)))
            source_name = " ".join(str(match.get("source_name", "")).split()).strip()
            if source_name and source_name not in entry["source_names"]:
                entry["source_names"].append(source_name)
            provider = " ".join(str(match.get("provider", "")).split()).strip().lower()
            if provider and provider not in entry["providers"]:
                entry["providers"].append(provider)
            complete_sentence, clipped_overlap = episode_quality(text, match.get("raw_window"))
            raw_window_text = str(match.get("raw_window") or "").strip()
            expansion_chars = max(0, len(text.strip()) - len(raw_window_text))
            previous_expansion = int(entry.get("expansion_chars", 0))
            entry["complete_sentence"] = max(int(entry["complete_sentence"]), int(complete_sentence))
            entry["clipped_overlap"] = min(int(entry["clipped_overlap"]), int(clipped_overlap))
            entry["expansion_chars"] = max(previous_expansion, int(expansion_chars))
            query_overlap = len(term_match_cache.match_terms(query_terms, text)) if query_terms else 0
            focus_overlap = (
                len(term_match_cache.match_terms(ordered_focus_terms, text)) if ordered_focus_terms else 0
            )
            entry["query_overlap"] = max(int(entry["query_overlap"]), query_overlap)
            entry["focus_overlap"] = max(int(entry["focus_overlap"]), focus_overlap)
            memory_index = int(match.get("memory_index", -1))
            if memory_index >= 0 and memory_index not in entry["memory_indices"]:
                entry["memory_indices"].append(memory_index)
            entry["memory_focus_priority"] = max(
                float(entry.get("memory_focus_priority", 0.0)),
                _memory_focus_priority(memory_priority, entry["memory_indices"]),
            )
            if (
                expansion_chars > previous_expansion
                or float(match.get("similarity", 0.0)) >= float(entry["similarity"])
            ):
                entry["memory_index"] = memory_index
                if match.get("raw_window"):
                    entry["raw_window"] = match.get("raw_window")
                entry["metadata"] = dict(match.get("metadata") or {})
                entry["source_name"] = source_name
                entry["source_type"] = " ".join(str(match.get("source_type", "")).split()).strip()
                entry["provider"] = provider

    episodes = [grouped[key] for key in order]
    episodes.sort(
        key=lambda item: (
            int(item.get("query_overlap", 0)),
            int(item.get("focus_overlap", 0)),
            float(item.get("memory_focus_priority", 0.0)),
            int(item.get("complete_sentence", 0)),
            int(item.get("expansion_chars", 0)),
            -int(item.get("clipped_overlap", 0)),
            float(item.get("similarity", 0.0)),
            int(item.get("match_count", 0)),
            float(item.get("importance", 0.0)),
        ),
        reverse=True,
    )
    support_filtered_episodes = [
        episode
        for episode in episodes
        if int(episode.get("query_overlap", 0)) > 0
        or int(episode.get("focus_overlap", 0)) > 0
        or float(episode.get("memory_focus_priority", 0.0)) > 0.0
    ]
    support_filter_applied = bool(
        (query_terms or ordered_focus_terms or memory_priority)
        and support_filtered_episodes
    )
    selected_source = support_filtered_episodes if support_filter_applied else episodes
    selected = selected_source[: max(1, int(top_k))]
    selected_memory_indices: list[int] = []
    for episode in selected:
        for index in list(episode.get("memory_indices") or []):
            try:
                idx = int(index)
            except (TypeError, ValueError):
                continue
            if idx >= 0 and idx not in selected_memory_indices:
                selected_memory_indices.append(idx)
    report = {
        "surface": "bounded_query_memory_episode_readout.v1",
        "status": "selected" if selected else "empty",
        "scope": "query_runner_memory_episode_readout",
        "requested_count": int(max(1, int(top_k))),
        "input_match_count": int(len(memory_matches)),
        "source_text_match_count": int(source_text_match_count),
        "candidate_episode_count": int(len(episodes)),
        "support_episode_count": int(len(support_filtered_episodes)),
        "support_filter_applied": bool(support_filter_applied),
        "support_filtered_out_count": int(
            max(0, len(episodes) - len(selected_source))
        ),
        "returned_count": int(len(selected)),
        "selected_memory_indices": selected_memory_indices,
        "selected_memory_index_count": int(len(selected_memory_indices)),
        "query_term_count": int(len(query_terms or [])),
        "focus_term_count": int(len(ordered_focus_terms)),
        "memory_priority_count": int(len(memory_priority or {})),
        "clause_term_count": int(len(clause_terms)),
        "neighbor_radius": int(max(0, int(neighbor_radius))),
        "neighbor_window_policy": (
            "selected_match_neighbor_windows_only"
            if loaded_neighbor_indices
            else "preselected_memory_match_text_only"
        ),
        "neighbor_row_surface": "bounded_query_neighbor_source_row.v1",
        "neighbor_row_reader_owned_by_store": bool(
            memory_store is not None
            and callable(getattr(memory_store, "query_neighbor_source_row", None))
        ),
        "direct_slow_memory_array_reads_retired": True,
        "neighbor_window_index_count": int(len(selected_neighbor_indices)),
        "neighbor_window_indices": sorted(selected_neighbor_indices)[:64],
        "neighbor_window_index_sample_limit": 64,
        "neighbor_window_index_truncated": bool(len(selected_neighbor_indices) > 64),
        "raw_text_payload_loaded": bool(loaded_neighbor_indices),
        "raw_text_payload_count": int(len(loaded_neighbor_indices)),
        "raw_text_payload_source": (
            "selected_match_neighbor_windows"
            if loaded_neighbor_indices
            else "preselected_bounded_memory_matches"
        ),
        "raw_text_payload_policy": (
            "selected_match_neighbor_windows_only"
            if loaded_neighbor_indices
            else "selected_memory_matches_only_no_store_access"
        ),
        "global_score_scan": False,
        "global_candidate_scan": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "language_reasoning": False,
        "hidden_language_reasoning": False,
        "archival_storage_device": "cpu",
        "quality_metric": "episode_clause_selection_over_bounded_matches",
        "selection_budget": {
            "input_match_budget_entries": int(len(memory_matches)),
            "return_budget_entries": int(max(1, int(top_k))),
            "neighbor_radius": int(max(0, int(neighbor_radius))),
            "neighbor_window_read_budget_entries": int(
                max(0, int(neighbor_radius)) * 2 + 1
            )
            * int(len(memory_matches)),
        },
    }
    return selected, report


def read_text_argument(text: Optional[str], file_path: Optional[Path]) -> Optional[str]:
    if text is not None:
        return text
    if file_path is not None:
        return file_path.read_text(encoding="utf-8")
    return None


def build_context_memory_match_report(
    context_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reports = [dict(report) for report in context_reports if isinstance(report, Mapping)]
    match_indices: list[int] = []
    context_labels: list[str] = []
    for report in reports:
        label = str(report.get("context_label") or "").strip()
        if label:
            context_labels.append(label)
        match_indices.extend(
            int(index)
            for index in list(report.get("match_indices", []))
            if isinstance(index, int) or str(index).lstrip("-").isdigit()
        )
    any_global_candidate_scan = any(
        bool(report.get("global_candidate_scan")) for report in reports
    )
    any_global_score_scan = any(bool(report.get("global_score_scan")) for report in reports)
    any_language_reasoning = any(
        bool(report.get("language_reasoning")) for report in reports
    )
    raw_payload_count = sum(
        int(report.get("raw_text_payload_count", 0) or 0) for report in reports
    )
    raw_payload_cache_hits = sum(
        int(report.get("raw_text_payload_cache_hits", 0) or 0) for report in reports
    )
    candidate_index_count = sum(
        int(report.get("candidate_index_count", 0) or 0) for report in reports
    )
    similarity_score_count = sum(
        int(report.get("similarity_score_count", 0) or 0) for report in reports
    )
    replay_priority_score_count = sum(
        int(report.get("replay_priority_score_count", 0) or 0) for report in reports
    )
    returned_count = sum(int(report.get("returned_count", 0) or 0) for report in reports)
    fallback_reasons = [
        str(report.get("fallback_reason"))
        for report in reports
        if report.get("fallback_reason") not in (None, "")
    ]
    return {
        "surface": "bounded_context_comparison_memory_match.v1",
        "status": "matched" if returned_count > 0 else "empty",
        "scope": "context_comparison_query_memory_match_slow_path",
        "context_count": int(len(reports)),
        "context_labels": context_labels,
        "candidate_surface": "bounded_query_memory_match.v1",
        "candidate_window_policy": "per_context_bounded_query_memory_match",
        "candidate_scope": "per_context_bucket_indexed_candidate_window",
        "candidate_index_count": int(candidate_index_count),
        "unique_candidate_index_count": int(len({int(index) for index in match_indices})),
        "match_indices": match_indices,
        "returned_count": int(returned_count),
        "raw_text_payload_loaded": bool(raw_payload_count > 0),
        "raw_text_payload_count": int(raw_payload_count),
        "raw_text_payload_cache_hits": int(raw_payload_cache_hits),
        "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        "similarity_score_count": int(similarity_score_count),
        "replay_priority_score_count": int(replay_priority_score_count),
        "global_candidate_scan": bool(any_global_candidate_scan),
        "global_score_scan": bool(any_global_score_scan),
        "runs_live_tick": False,
        "runs_every_token": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "language_reasoning": bool(any_language_reasoning),
        "score_device": "cpu",
        "archival_storage_device": "cpu",
        "quality_metric": "context_comparison_memory_match_parity",
        "fallback_reason": fallback_reasons[0] if fallback_reasons else None,
        "selection_budget": {
            "context_budget": int(len(reports)),
            "candidate_window_entries": int(candidate_index_count),
            "raw_text_payload_policy": "shared_returned_similarity_matches_only",
        },
        "context_reports": reports,
    }


def build_context_comparison(
    checkpoint: Path,
    query_text: str,
    feed_text_value: Optional[str],
    context_a: str,
    context_b: str,
    top_k_candidates: int,
    top_k_memories: int,
    top_chars: int,
) -> dict[str, Any]:
    trainer, _ = load_trainer_checkpoint(checkpoint)
    encoder = trainer.encoder

    if feed_text_value:
        feed_text(trainer, encoder, feed_text_value)

    query_examples = list(text_pattern_stream(query_text, encoder, trainer.config.window_size))
    if not query_examples:
        raise ValueError("Query text produced no patterns")
    _, pattern_vec = query_examples[-1]

    if trainer.model.context_layer is None:
        return {
            "supported": False,
            "reason": "Checkpoint does not include a context layer",
        }

    comparisons: list[dict[str, Any]] = []
    context_memory_reports: list[dict[str, Any]] = []
    replay_entry_cache: dict[int, dict[str, Any]] = {}
    for label, context_value in (("context_a", context_a), ("context_b", context_b)):
        primed = prime_context(trainer, encoder, context_value)
        routing_key = trainer.routing_key_for_pattern(pattern_vec)
        winner = trainer.contextual_winner_for_pattern(pattern_vec)
        matches, memory_match_report = memory_matches_with_report(
            trainer,
            pattern_vec,
            routing_key,
            top_k_memories,
            top_chars,
            replay_entry_cache=replay_entry_cache,
        )
        memory_match_report = {
            **dict(memory_match_report),
            "context_label": label,
        }
        context_memory_reports.append(memory_match_report)
        comparisons.append(
            {
                "label": label,
                "context_text": context_value,
                "primed_tokens": int(primed),
                "winner_column": int(winner),
                "winner_shard": int(winner % max(1, trainer.config.routing_shards)),
                "context_state_norm": float(torch.norm(trainer.context_state().float()).item()),
                "top_candidates": candidate_details(trainer, routing_key.detach().cpu(), top_k_candidates),
                "memory_match_report": memory_match_report,
                "memory_matches": matches,
            }
        )

    return {
        "supported": True,
        "query_text": query_text,
        "winner_switch": bool(comparisons[0]["winner_column"] != comparisons[1]["winner_column"]),
        "memory_match_report": build_context_memory_match_report(
            context_memory_reports
        ),
        "comparisons": comparisons,
    }


def _cross_modal_query_predictions(trainer: MarulhoTrainer, pattern_vec: torch.Tensor) -> dict[str, Any]:
    """Generate cross-modal predictions for a query pattern.

    Uses the trained cross-modal grounding layer to predict what visual/audio
    patterns should accompany the given text pattern, enabling inference-time
    cross-modal reasoning. pattern_vec must be input_dim-sized (the raw
    representation vector, not the routing key).
    """
    cross_modal = trainer.model.cross_modal
    if cross_modal is None:
        return {"available": False}
    text_assembly = F.normalize(pattern_vec.detach().unsqueeze(0), dim=1).squeeze(0).to(trainer.model.device)
    predicted_visual = cross_modal.predict_visual(text_assembly)
    predicted_audio = cross_modal.predict_audio(text_assembly)
    result: dict[str, Any] = {"available": True}
    if predicted_visual is not None:
        result["visual_prediction_norm"] = float(predicted_visual.norm().item())
        result["visual_prediction_top5"] = predicted_visual.abs().topk(min(5, predicted_visual.numel())).values.tolist()
    else:
        result["visual_prediction_norm"] = 0.0
    if predicted_audio is not None:
        result["audio_prediction_norm"] = float(predicted_audio.norm().item())
        result["audio_prediction_top5"] = predicted_audio.abs().topk(min(5, predicted_audio.numel())).values.tolist()
    else:
        result["audio_prediction_norm"] = 0.0
    result["word_grounding"] = {
        word: round(conf, 4)
        for word, conf in trainer.word_grounding_confidence.items()
        if conf > 0.01
    }
    return result


def build_query_result(
    trainer: MarulhoTrainer,
    checkpoint: Path,
    metadata: dict[str, Any],
    encoder: RTFEncoder,
    query_text_resolved: Optional[str],
    feed_text_resolved: Optional[str],
    context_text: Optional[str],
    top_k_candidates: int,
    top_k_memories: int,
    top_chars: int,
    compare_context_a: Optional[str],
    compare_context_b: Optional[str],
    retrieval_focus_terms: Sequence[str] | None = None,
    memory_priority: Mapping[object, object] | None = None,
) -> dict[str, Any]:
    trainer.reset_context_state()
    representation = getattr(trainer.config, "input_representation", "order_weighted_ascii")
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_metadata": metadata,
        "config": {
            "n_columns": int(trainer.config.n_columns),
            "column_latent_dim": int(trainer.config.column_latent_dim),
            "routing_shards": int(trainer.config.routing_shards),
            "k_routing": int(trainer.config.k_routing),
            "memory_capacity": int(trainer.config.memory_capacity),
            "input_representation": representation,
        },
        "feed_summary": None,
        "context_summary": None,
        "query_summary": None,
        "context_comparison": None,
    }

    if feed_text_resolved:
        result["feed_summary"] = feed_text(trainer, encoder, feed_text_resolved)

    if context_text is not None:
        primed = prime_context(trainer, encoder, context_text)
        result["context_summary"] = {
            "primed_tokens": int(primed),
            "context_state_norm": float(torch.norm(trainer.context_state().float()).item()),
            "context_supported": bool(trainer.model.context_layer is not None),
        }

    if query_text_resolved:
        query_examples = list(text_pattern_stream(query_text_resolved, encoder, trainer.config.window_size))
        if not query_examples:
            raise ValueError("Query text produced no patterns")
        query_window, pattern_vec = query_examples[-1]
        routing_key = trainer.routing_key_for_pattern(pattern_vec)
        winner = trainer.contextual_winner_for_pattern(pattern_vec) if trainer.model.context_layer is not None and context_text is not None else trainer.winner_for_pattern(pattern_vec)
        recon_error = trainer.reconstruction_error(pattern_vec)
        query_terms = salient_query_terms(query_text_resolved)
        ordered_focus_terms = _dedupe_terms(retrieval_focus_terms)
        decode_matches, memory_match_report = memory_matches_with_report(
            trainer,
            pattern_vec,
            routing_key,
            max(top_k_memories, 24),
            top_chars,
            query_terms=query_terms,
            focus_terms=ordered_focus_terms,
            memory_priority=memory_priority,
        )
        memory_episodes, memory_episode_report = build_memory_episodes_with_report(
            decode_matches,
            top_k=max(1, min(int(top_k_memories), 8)),
            query_terms=query_terms,
            focus_terms=ordered_focus_terms,
            memory_priority=memory_priority,
            memory_store=trainer.model.memory_store,
            neighbor_radius=3,
        )
        result["query_summary"] = {
            "query_text": query_text_resolved,
            "query_window": query_window,
            "reconstruction_error": float(recon_error),
            "winner_column": int(winner),
            "winner_shard": int(winner % max(1, trainer.config.routing_shards)),
            "top_query_chars": top_feature_details(pattern_vec, top_chars, representation),
            "top_candidates": candidate_details(trainer, routing_key.detach().cpu(), top_k_candidates),
            "memory_match_report": memory_match_report,
            "memory_matches": decode_matches[: max(1, int(top_k_memories))],
            "memory_episode_report": memory_episode_report,
            "memory_episodes": memory_episodes,
            "native_decode": _NATIVE_DECODER.decode(
                query_window=query_window,
                winner_column=int(winner),
                memory_matches=decode_matches,
            ),
            "cross_modal_predictions": _cross_modal_query_predictions(trainer, pattern_vec),
        }

    if compare_context_a is not None and compare_context_b is not None and query_text_resolved is not None:
        result["context_comparison"] = build_context_comparison(
            checkpoint=checkpoint,
            query_text=query_text_resolved,
            feed_text_value=feed_text_resolved,
            context_a=compare_context_a,
            context_b=compare_context_b,
            top_k_candidates=top_k_candidates,
            top_k_memories=top_k_memories,
            top_chars=top_chars,
        )

    return result


def run_query(
    checkpoint: Path,
    query_text: Optional[str],
    query_file: Optional[Path],
    feed_text_value: Optional[str],
    feed_file: Optional[Path],
    context_text: Optional[str],
    top_k_candidates: int,
    top_k_memories: int,
    top_chars: int,
    output_json: Optional[Path],
    save_checkpoint_path: Optional[Path],
    compare_context_a: Optional[str],
    compare_context_b: Optional[str],
) -> None:
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    encoder = trainer.encoder

    feed_text_resolved = read_text_argument(feed_text_value, feed_file)
    query_text_resolved = read_text_argument(query_text, query_file)

    result = build_query_result(
        trainer=trainer,
        checkpoint=checkpoint,
        metadata=metadata,
        encoder=encoder,
        query_text_resolved=query_text_resolved,
        feed_text_resolved=feed_text_resolved,
        context_text=context_text,
        top_k_candidates=top_k_candidates,
        top_k_memories=top_k_memories,
        top_chars=top_chars,
        compare_context_a=compare_context_a,
        compare_context_b=compare_context_b,
    )

    if save_checkpoint_path is not None:
        saved_path = save_trainer_checkpoint(save_checkpoint_path, trainer, metadata=metadata)
        result["saved_checkpoint"] = str(saved_path)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(output_json, result)

    print("MARULHO query summary")
    print(f"checkpoint={checkpoint}")
    if result["feed_summary"] is not None:
        print(f"feed_tokens_processed={result['feed_summary']['tokens_processed']}")
        print(f"feed_last_winner={result['feed_summary']['last_winner']}")
        print(f"feed_memory_buffer_size={result['feed_summary']['memory_buffer_size']}")
    if result["context_summary"] is not None:
        print(f"context_primed_tokens={result['context_summary']['primed_tokens']}")
        print(f"context_state_norm={result['context_summary']['context_state_norm']:.6f}")
    if result["query_summary"] is not None:
        query_summary = result["query_summary"]
        print(f"query_window={query_summary['query_window']}")
        print(f"winner_column={query_summary['winner_column']}")
        print(f"winner_shard={query_summary['winner_shard']}")
        print(f"reconstruction_error={query_summary['reconstruction_error']:.6f}")
        native_decode = query_summary.get("native_decode") or {}
        if native_decode.get("available"):
            print(f"native_decode_confidence={float(native_decode['confidence']):.6f}")
            print(f"native_decode_text={native_decode['decoded_text']}")
        if query_summary["top_candidates"]:
            top_candidate = query_summary["top_candidates"][0]
            print(f"top_candidate={top_candidate['column_id']} similarity={top_candidate['similarity']:.6f}")
        if query_summary["memory_matches"]:
            top_memory = query_summary["memory_matches"][0]
            print(f"top_memory_similarity={top_memory['similarity']:.6f}")
            print(f"top_memory_window={top_memory['raw_window']}")
    if result["context_comparison"] is not None:
        context_comparison = result["context_comparison"]
        print(f"context_comparison_supported={context_comparison['supported']}")
        if context_comparison["supported"]:
            print(f"context_winner_switch={context_comparison['winner_switch']}")
    if output_json is not None:
        print(f"output_json={output_json}")
    if save_checkpoint_path is not None:
        print(f"saved_checkpoint={save_checkpoint_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a MARULHO checkpoint, feed raw text, and retrieve matching memories")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query-text", type=str, default=None)
    parser.add_argument("--query-file", type=Path, default=None)
    parser.add_argument("--feed-text", type=str, default=None)
    parser.add_argument("--feed-file", type=Path, default=None)
    parser.add_argument("--context-text", type=str, default=None)
    parser.add_argument("--top-k-candidates", type=int, default=5)
    parser.add_argument("--top-k-memories", type=int, default=5)
    parser.add_argument("--top-chars", type=int, default=6)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-checkpoint", type=Path, default=None)
    parser.add_argument("--compare-context-a", type=str, default=None)
    parser.add_argument("--compare-context-b", type=str, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_query(
        checkpoint=args.checkpoint,
        query_text=args.query_text,
        query_file=args.query_file,
        feed_text_value=args.feed_text,
        feed_file=args.feed_file,
        context_text=args.context_text,
        top_k_candidates=args.top_k_candidates,
        top_k_memories=args.top_k_memories,
        top_chars=args.top_chars,
        output_json=args.output_json,
        save_checkpoint_path=args.save_checkpoint,
        compare_context_a=args.compare_context_a,
        compare_context_b=args.compare_context_b,
    )


if __name__ == "__main__":
    main()
