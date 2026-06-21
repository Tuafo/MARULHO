from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from itertools import islice
from typing import Any

SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT = 64
SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_SURFACE = (
    "bounded_snn_language_plasticity_runtime_transition_memory_source_window.v1"
)
SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_POLICY = (
    "recent_plasticity_runtime_transition_memory_source_window_v1"
)


def _mapping_length(source: Any) -> tuple[int, bool]:
    if not isinstance(source, Mapping):
        return 0, True
    try:
        return int(len(source)), True
    except TypeError:
        return 0, False


def _mapping_items(
    source: Mapping[str, Any],
    *,
    prefer_recent: bool,
) -> Any:
    if prefer_recent:
        try:
            return ((key, source[key]) for key in reversed(source))
        except TypeError:
            pass
    return source.items()


def bounded_transition_memory_mapping_items(
    source: Any,
    *,
    limit: int,
    prefer_recent: bool = False,
) -> tuple[list[tuple[str, Any]], int, bool, bool]:
    if not isinstance(source, Mapping):
        return [], 0, True, False

    source_limit = max(0, int(limit))
    retained_count, retained_count_known = _mapping_length(source)
    item_limit = source_limit if retained_count_known else source_limit + 1
    selected: list[tuple[str, Any]] = []
    sentinel_seen = False

    for key, value in islice(
        _mapping_items(source, prefer_recent=prefer_recent),
        item_limit,
    ):
        if len(selected) >= source_limit:
            sentinel_seen = True
            break
        selected.append((str(key), deepcopy(value)))

    if not retained_count_known:
        retained_count = len(selected)
    truncated = (
        retained_count > len(selected) if retained_count_known else sentinel_seen
    )
    return selected, int(retained_count), retained_count_known, bool(truncated)


def build_plasticity_runtime_transition_memory_source_window(
    state: Mapping[str, Any],
    *,
    limit: int = SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT,
) -> dict[str, Any]:
    source_limit = max(
        0,
        min(
            int(limit),
            int(SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_LIMIT),
        ),
    )
    raw_sparse = state.get("sparse_transition_weights")
    raw_provenance = state.get("synapse_provenance_by_key")
    sparse_items, retained_sparse, sparse_count_known, sparse_truncated = (
        bounded_transition_memory_mapping_items(
            raw_sparse,
            limit=source_limit,
            prefer_recent=True,
        )
    )
    provenance_items, retained_provenance, provenance_count_known, provenance_truncated = (
        bounded_transition_memory_mapping_items(
            raw_provenance,
            limit=source_limit,
            prefer_recent=True,
        )
    )
    sparse_source_count = len(sparse_items)
    provenance_source_count = len(provenance_items)
    source_window_complete = not bool(sparse_truncated or provenance_truncated)
    source_window = {
        "surface": SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_SURFACE,
        "policy": SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_POLICY,
        "window_policy": SNN_LANGUAGE_PLASTICITY_RUNTIME_TRANSITION_MEMORY_SOURCE_WINDOW_POLICY,
        "source": "snn_language_plasticity_runtime_state.transition_memory",
        "selection_criteria": [
            "newest_sparse_transition_weight_rows",
            "newest_synapse_provenance_rows",
            "bounded_runtime_state_source_window",
        ],
        "source_window_limit": int(source_limit),
        "source_window_count": max(sparse_source_count, provenance_source_count),
        "source_sparse_transition_weight_rows": int(sparse_source_count),
        "source_synapse_provenance_rows": int(provenance_source_count),
        "retained_sparse_transition_weight_rows": int(retained_sparse),
        "retained_synapse_provenance_rows": int(retained_provenance),
        "retained_sparse_weight_rows": int(retained_sparse),
        "source_record_count": max(int(retained_sparse), int(retained_provenance)),
        "source_record_count_known": bool(
            sparse_count_known and provenance_count_known
        ),
        "source_payload_truncated": not bool(source_window_complete),
        "source_truncated_counts": {
            "sparse_transition_weights": max(
                0,
                int(retained_sparse) - int(sparse_source_count),
            )
            if sparse_count_known
            else None,
            "synapse_provenance_by_key": max(
                0,
                int(retained_provenance) - int(provenance_source_count),
            )
            if provenance_count_known
            else None,
        },
        "source_counts": {
            "retained_sparse_transition_weights": int(retained_sparse),
            "retained_synapse_provenance_rows": int(retained_provenance),
            "source_sparse_transition_weights": int(sparse_source_count),
            "source_synapse_provenance_rows": int(provenance_source_count),
        },
        "source_limits": {
            "sparse_transition_weights": int(source_limit),
            "synapse_provenance_by_key": int(source_limit),
        },
        "source_window_complete": bool(source_window_complete),
        "integrity_scope": (
            "complete" if source_window_complete else "bounded_source_window"
        ),
        "global_candidate_scan": False,
        "global_score_scan": False,
        "raw_text_payload_loaded": False,
        "hidden_language_reasoning": False,
        "language_reasoning": False,
        "runs_live_tick": False,
        "runs_every_token": False,
        "runs_replay": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "archival_storage_device": "cpu",
        "source_window_selection_device": "cpu",
        "lookup_device": "cpu",
        "gpu_used": False,
        "gpu_resident_archival_metadata": False,
        "device_placement": {
            "archival_storage": "cpu",
            "source_window_selection": "cpu",
            "lookup": "cpu",
        },
        "memory_budget": {
            "max_sparse_transition_weight_rows": int(source_limit),
            "max_synapse_provenance_rows": int(source_limit),
            "max_records_total": int(source_limit * 2),
            "archival_storage_device": "cpu",
        },
    }
    return {
        "sparse_transition_weights": dict(sparse_items),
        "synapse_provenance_by_key": {
            key: value for key, value in provenance_items if isinstance(value, Mapping)
        },
        "sparse_transition_weight_items": sparse_items,
        "synapse_provenance_items": provenance_items,
        "source_window": source_window,
    }


def retained_transition_memory_counts(
    state: Mapping[str, Any],
    sparse_mapping: Mapping[str, Any],
    provenance_mapping: Mapping[str, Any],
) -> tuple[int, int]:
    source_window = (
        state.get("transition_memory_source_window")
        if isinstance(state.get("transition_memory_source_window"), Mapping)
        else {}
    )
    source_counts = (
        source_window.get("source_counts")
        if isinstance(source_window.get("source_counts"), Mapping)
        else {}
    )
    sparse_count = source_window.get(
        "retained_sparse_transition_weight_rows",
        source_counts.get("retained_sparse_transition_weights"),
    )
    provenance_count = source_window.get(
        "retained_synapse_provenance_rows",
        source_counts.get("retained_synapse_provenance_rows"),
    )
    if sparse_count is None:
        sparse_count, _known = _mapping_length(sparse_mapping)
    if provenance_count is None:
        provenance_count, _known = _mapping_length(provenance_mapping)
    return int(sparse_count or 0), int(provenance_count or 0)
