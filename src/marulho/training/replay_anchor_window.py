from __future__ import annotations

from typing import Any, Mapping

import torch


ANCHOR_BUCKET_SOURCE_WINDOW_LIMIT = 16
REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT = ANCHOR_BUCKET_SOURCE_WINDOW_LIMIT
REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_POLICY = "recent_anchor_capture_recency_window_v1"
SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT = ANCHOR_BUCKET_SOURCE_WINDOW_LIMIT
SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_POLICY = (
    "recent_sleep_anchor_capture_recency_window_v1"
)


def _anchor_metadata_int(anchor: Mapping[str, Any], key: str) -> int | None:
    value = anchor.get(key)
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _build_anchor_bucket_source_window(
    trainer: Any,
    *,
    surface: str,
    scope: str,
    window_policy: str,
    selection_criteria: list[str],
    max_buckets: int = ANCHOR_BUCKET_SOURCE_WINDOW_LIMIT,
    extra_fields: Mapping[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    anchors = getattr(trainer, "column_anchors", {}) or {}
    if not isinstance(anchors, Mapping):
        anchors = {}
    limit = max(0, int(max_buckets))
    total_count = int(len(anchors))
    selected_bucket_ids: list[int] = []
    selected_metadata: list[dict[str, Any]] = []
    source_read_count = 0
    materialized_count = 0
    source_full_scan = False

    if limit > 0:
        try:
            reverse_keys = reversed(anchors)
        except TypeError:
            materialized_keys = list(anchors)
            materialized_count = int(len(materialized_keys))
            source_full_scan = bool(materialized_count > limit)
            reverse_keys = reversed(materialized_keys)
        for raw_bucket_id in reverse_keys:
            source_read_count += 1
            try:
                bucket_id = int(raw_bucket_id)
            except (TypeError, ValueError, OverflowError):
                continue
            anchor = anchors.get(raw_bucket_id, {})
            anchor_mapping = anchor if isinstance(anchor, Mapping) else {}
            selected_bucket_ids.append(bucket_id)
            selected_metadata.append(
                {
                    "bucket_id": bucket_id,
                    "captured_at_token": _anchor_metadata_int(
                        anchor_mapping,
                        "captured_at_token",
                    ),
                    "captured_source_index": _anchor_metadata_int(
                        anchor_mapping,
                        "captured_source_index",
                    ),
                    "capture_sequence": _anchor_metadata_int(
                        anchor_mapping,
                        "capture_sequence",
                    ),
                }
            )
            if len(selected_bucket_ids) >= limit:
                break

    status = "selected" if selected_bucket_ids else "empty"
    fallback_reason = None if selected_bucket_ids else "empty_anchor_bucket_source"
    truncated_count = max(0, total_count - len(selected_bucket_ids))
    report = {
        "surface": str(surface),
        "status": status,
        "scope": str(scope),
        "window_policy": str(window_policy),
        "selection_criteria": list(selection_criteria),
        "source": "trainer.column_anchors",
        "anchor_bucket_source_total_count": total_count,
        "anchor_bucket_source_count_is_exact": True,
        "anchor_bucket_window_limit": limit,
        "anchor_bucket_window_count": int(len(selected_bucket_ids)),
        "anchor_bucket_ids": selected_bucket_ids,
        "selected_anchor_metadata": selected_metadata,
        "truncated_source_count": bool(truncated_count > 0),
        "anchor_bucket_source_truncated_count": int(truncated_count),
        "anchor_bucket_source_read_count": int(source_read_count),
        "anchor_bucket_source_materialized_count": int(materialized_count),
        "global_candidate_scan": False,
        "global_score_scan": False,
        "anchor_source_full_scan": bool(source_full_scan),
        "runs_live_tick": False,
        "runs_every_token": False,
        "raw_text_payload_loaded": False,
        "language_reasoning": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "source_window_selection_device": "cpu",
        "active_replay_compute_device": "cpu",
        "gpu_resident_archival_metadata": False,
        "fallback_reason": fallback_reason,
        "selection_budget": {
            "anchor_bucket_source_entries": total_count,
            "anchor_bucket_window_entries": int(len(selected_bucket_ids)),
            "anchor_bucket_window_limit": limit,
            "anchor_bucket_source_read_entries": int(source_read_count),
        },
        "memory_budget": {
            "anchor_bucket_window_entries": int(len(selected_bucket_ids)),
            "anchor_bucket_window_limit": limit,
            "archival_metadata_residency": "cpu",
            "gpu_resident_archival_metadata": False,
        },
    }
    if extra_fields:
        report.update(dict(extra_fields))
    return selected_bucket_ids, report


def replay_query_anchor_bucket_source_window(
    trainer: Any,
    *,
    max_buckets: int = REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_LIMIT,
    scope: str = "hf_task_a_anchor_query_collection",
) -> tuple[list[int], dict[str, Any]]:
    return _build_anchor_bucket_source_window(
        trainer,
        surface="bounded_replay_query_anchor_bucket_source_window.v1",
        scope=scope,
        window_policy=REPLAY_QUERY_ANCHOR_BUCKET_WINDOW_POLICY,
        selection_criteria=[
            "column_anchor_reverse_recency_order",
            "bounded_anchor_bucket_count",
            "bucket_indexed_replay_query_collection",
        ],
        max_buckets=max_buckets,
    )


def sleep_replay_anchor_bucket_source_window(
    trainer: Any,
    *,
    mode: str,
    max_buckets: int = SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_LIMIT,
) -> tuple[list[int] | None, dict[str, Any]]:
    if mode not in {"micro", "deep", "repair"}:
        return None, {
            "surface": "bounded_sleep_replay_anchor_bucket_source_window.v1",
            "status": "not_applicable",
            "mode": str(mode),
            "scope": f"{mode}_sleep_slow_path",
            "window_policy": SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_POLICY,
            "selection_criteria": [],
            "anchor_bucket_source_total_count": 0,
            "anchor_bucket_window_limit": max(0, int(max_buckets)),
            "anchor_bucket_window_count": 0,
            "anchor_bucket_ids": [],
            "global_candidate_scan": False,
            "global_score_scan": False,
            "anchor_source_full_scan": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "active_replay_compute_device": "cpu",
            "gpu_resident_archival_metadata": False,
            "fallback_reason": "mode_not_sleep_replay",
        }
    return _build_anchor_bucket_source_window(
        trainer,
        surface="bounded_sleep_replay_anchor_bucket_source_window.v1",
        scope=f"{mode}_sleep_slow_path",
        window_policy=SLEEP_REPLAY_ANCHOR_BUCKET_WINDOW_POLICY,
        selection_criteria=[
            "column_anchor_reverse_recency_order",
            "bounded_anchor_bucket_count",
            "bucket_indexed_sleep_replay_selection",
        ],
        max_buckets=max_buckets,
        extra_fields={"mode": str(mode)},
    )
