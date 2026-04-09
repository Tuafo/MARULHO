from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.retrieval import NativeAssemblyDecoder
from hecsn.reporting.io import write_json_file
from hecsn.semantics.grounding_text import TOKEN_RE
from hecsn.semantics.grounding_text import match_terms
from hecsn.semantics.grounding_text import query_focused_clauses
from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.trainer import HECSNTrainer


_NATIVE_DECODER = NativeAssemblyDecoder()


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
    window_codes: List[int] = []
    window_chars: List[str] = []
    for ch in text:
        code = ord(ch) if ord(ch) < 128 else 0
        display = ch if ord(ch) < 128 else "?"
        window_codes.append(code)
        window_chars.append(display)
        if len(window_codes) > window_size:
            window_codes.pop(0)
            window_chars.pop(0)
        yield "".join(window_chars), encoder.feature_vector(window_codes)


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
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    text: str,
    *,
    on_step: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    last_metrics: dict[str, Any] | None = None
    tokens = 0
    for raw_window, pattern in text_pattern_stream(text, encoder, trainer.config.window_size):
        last_metrics = trainer.train_step(pattern, raw_window=raw_window)
        if on_step is not None:
            on_step(raw_window, last_metrics)
        tokens += 1

    return {
        "tokens_processed": int(tokens),
        "token_count": int(trainer.token_count),
        "last_winner": None if last_metrics is None else int(last_metrics["winner"]),
        "last_recon_error": None if last_metrics is None else float(last_metrics["recon_error"]),
        "memory_buffer_size": int(len(trainer.model.memory_store.slow_buffer)),
    }


def prime_context(trainer: HECSNTrainer, encoder: RTFEncoder, text: str) -> int:
    patterns = [pattern for _, pattern in text_pattern_stream(text, encoder, trainer.config.window_size)]
    trainer.prime_context(patterns, update_weights=False)
    return len(patterns)


def candidate_details(trainer: HECSNTrainer, routing_key: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    candidate_ids, _ = trainer.model.hnsw_index.search(routing_key.unsqueeze(0), k=max(1, int(top_k)))
    row = candidate_ids[0] if candidate_ids else []
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


def memory_matches(
    trainer: HECSNTrainer,
    pattern_vec: torch.Tensor,
    routing_key: torch.Tensor,
    top_k: int,
    top_chars: int,
    query_terms: Optional[list[str]] = None,
    *,
    focus_terms: Sequence[str] | None = None,
    memory_priority: Mapping[object, object] | None = None,
) -> list[dict[str, Any]]:
    store = trainer.model.memory_store
    representation = getattr(trainer.config, "input_representation", "order_weighted_ascii")
    replay_scores = store.replay_scores(trainer.token_count)
    ordered_focus_terms = _dedupe_terms(focus_terms)
    matches: list[dict[str, Any]] = []
    query_input = pattern_vec.detach().cpu()
    query_key = routing_key.detach().cpu()
    for idx in range(len(store.slow_buffer)):
        ref_key = store.slow_routing_keys[idx]
        ref_input = store.slow_input_patterns[idx]
        if isinstance(ref_key, torch.Tensor):
            similarity = cosine_similarity(query_key, ref_key.float())
        elif isinstance(ref_input, torch.Tensor):
            similarity = cosine_similarity(query_input, ref_input.float())
        else:
            similarity = cosine_similarity(query_input, store.slow_buffer[idx].float())

        evidence_pattern = ref_input.float() if isinstance(ref_input, torch.Tensor) else store.slow_buffer[idx].float()
        replay_entry = store.replay_entry(idx, current_token=trainer.token_count)
        capture_tag = float(replay_entry.get("capture_tag", 0.0))
        prp_level = float(replay_entry.get("prp_level", 0.0))
        capture_strength = float(replay_entry.get("capture_strength", 0.0))
        consolidation_level = float(replay_entry.get("consolidation_level", 0.0))
        text = replay_entry.get("text") or store.slow_raw_windows[idx]
        raw_window = store.slow_raw_windows[idx]
        complete_sentence, clipped_overlap = episode_quality(str(text or "").strip(), raw_window)
        matched_query_terms = match_terms(query_terms or [], str(text or ""))
        query_overlap = len(matched_query_terms)
        matched_focus_terms = match_terms(ordered_focus_terms, str(text or "")) if ordered_focus_terms else []
        focus_overlap = len(matched_focus_terms)
        focus_priority = _memory_focus_priority(memory_priority, (idx,))
        matches.append(
            {
                "memory_index": int(idx),
                "similarity": float(similarity),
                "bucket_id": None if store.slow_bucket_ids[idx] is None else int(store.slow_bucket_ids[idx]),
                "raw_window": raw_window,
                "text": text,
                "age_tokens": int(max(0, trainer.token_count - int(store.slow_entry_timestamps[idx]))),
                "importance": float(store.slow_importance[idx]),
                "tag_strength": float(capture_tag),
                "capture_tag": float(capture_tag),
                "prp_level": float(prp_level),
                "capture_strength": float(capture_strength),
                "consolidation_level": consolidation_level,
                "consolidation_gap": float(max(0.0, 1.0 - consolidation_level)),
                "replay_count": int(store.slow_replay_count[idx]),
                "replay_priority": float(replay_scores[idx].item()) if idx < int(replay_scores.numel()) else 0.0,
                "top_chars": top_feature_details(evidence_pattern, top_chars, representation),
                "query_overlap": int(query_overlap),
                "matched_query_terms": matched_query_terms,
                "focus_overlap": int(focus_overlap),
                "matched_focus_terms": matched_focus_terms,
                "memory_focus_priority": float(focus_priority),
                "complete_sentence": int(complete_sentence),
                "clipped_overlap": int(clipped_overlap),
            }
        )

    limit = max(1, int(top_k))
    if not query_terms and not ordered_focus_terms and not memory_priority:
        matches.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return matches[:limit]

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
        return matches[:limit]

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
    return merged[:limit]


def build_memory_episodes(
    memory_matches: list[dict[str, Any]],
    *,
    top_k: int,
    query_terms: Optional[list[str]] = None,
    focus_terms: Sequence[str] | None = None,
    memory_priority: Mapping[object, object] | None = None,
) -> list[dict[str, Any]]:
    ordered_focus_terms = _dedupe_terms(focus_terms)
    clause_terms = _dedupe_terms([*(query_terms or []), *ordered_focus_terms])
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for match in memory_matches:
        source_text = str(match.get("text") or match.get("raw_window") or "").strip()
        if not source_text:
            continue
        clause_candidates = query_focused_clauses(source_text, clause_terms)
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
                }
                grouped[key] = entry
                order.append(key)
            entry["match_count"] += 1
            entry["similarity"] = max(float(entry["similarity"]), float(match.get("similarity", 0.0)))
            entry["importance"] = max(float(entry["importance"]), float(match.get("importance", 0.0)))
            entry["age_tokens"] = min(int(entry["age_tokens"]), int(match.get("age_tokens", 0)))
            complete_sentence, clipped_overlap = episode_quality(text, match.get("raw_window"))
            entry["complete_sentence"] = max(int(entry["complete_sentence"]), int(complete_sentence))
            entry["clipped_overlap"] = min(int(entry["clipped_overlap"]), int(clipped_overlap))
            entry["query_overlap"] = max(int(entry["query_overlap"]), len(match_terms(query_terms or [], text)))
            entry["focus_overlap"] = max(int(entry["focus_overlap"]), len(match_terms(ordered_focus_terms, text)))
            memory_index = int(match.get("memory_index", -1))
            if memory_index >= 0 and memory_index not in entry["memory_indices"]:
                entry["memory_indices"].append(memory_index)
            entry["memory_focus_priority"] = max(
                float(entry.get("memory_focus_priority", 0.0)),
                _memory_focus_priority(memory_priority, entry["memory_indices"]),
            )
            if float(match.get("similarity", 0.0)) >= float(entry["similarity"]):
                entry["memory_index"] = memory_index
                if match.get("raw_window"):
                    entry["raw_window"] = match.get("raw_window")

    episodes = [grouped[key] for key in order]
    episodes.sort(
        key=lambda item: (
            int(item.get("query_overlap", 0)),
            int(item.get("focus_overlap", 0)),
            float(item.get("memory_focus_priority", 0.0)),
            int(item.get("complete_sentence", 0)),
            -int(item.get("clipped_overlap", 0)),
            float(item.get("similarity", 0.0)),
            int(item.get("match_count", 0)),
            float(item.get("importance", 0.0)),
        ),
        reverse=True,
    )
    return episodes[: max(1, int(top_k))]


def read_text_argument(text: Optional[str], file_path: Optional[Path]) -> Optional[str]:
    if text is not None:
        return text
    if file_path is not None:
        return file_path.read_text(encoding="utf-8")
    return None


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
    encoder = RTFEncoder.from_config(trainer.config)

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
    for label, context_value in (("context_a", context_a), ("context_b", context_b)):
        primed = prime_context(trainer, encoder, context_value)
        routing_key = trainer.routing_key_for_pattern(pattern_vec)
        winner = trainer.contextual_winner_for_pattern(pattern_vec)
        comparisons.append(
            {
                "label": label,
                "context_text": context_value,
                "primed_tokens": int(primed),
                "winner_column": int(winner),
                "winner_shard": int(winner % max(1, trainer.config.routing_shards)),
                "context_state_norm": float(torch.norm(trainer.context_state().float()).item()),
                "top_candidates": candidate_details(trainer, routing_key.detach().cpu(), top_k_candidates),
                "memory_matches": memory_matches(trainer, pattern_vec, routing_key, top_k_memories, top_chars),
            }
        )

    return {
        "supported": True,
        "query_text": query_text,
        "winner_switch": bool(comparisons[0]["winner_column"] != comparisons[1]["winner_column"]),
        "comparisons": comparisons,
    }


def build_query_result(
    trainer: HECSNTrainer,
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
        decode_matches = memory_matches(
            trainer,
            pattern_vec,
            routing_key,
            max(top_k_memories, 24),
            top_chars,
            query_terms=query_terms,
            focus_terms=ordered_focus_terms,
            memory_priority=memory_priority,
        )
        result["query_summary"] = {
            "query_text": query_text_resolved,
            "query_window": query_window,
            "reconstruction_error": float(recon_error),
            "winner_column": int(winner),
            "winner_shard": int(winner % max(1, trainer.config.routing_shards)),
            "top_query_chars": top_feature_details(pattern_vec, top_chars, representation),
            "top_candidates": candidate_details(trainer, routing_key.detach().cpu(), top_k_candidates),
            "memory_matches": decode_matches[: max(1, int(top_k_memories))],
            "memory_episodes": build_memory_episodes(
                decode_matches,
                top_k=max(1, min(int(top_k_memories), 8)),
                query_terms=query_terms,
                focus_terms=ordered_focus_terms,
                memory_priority=memory_priority,
            ),
            "native_decode": _NATIVE_DECODER.decode(
                query_window=query_window,
                winner_column=int(winner),
                memory_matches=decode_matches,
            ),
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
    encoder = RTFEncoder.from_config(trainer.config)

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

    print("HECSN query summary")
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
    parser = argparse.ArgumentParser(description="Load a HECSN checkpoint, feed raw text, and retrieve matching memories")
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
