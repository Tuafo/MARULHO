from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Mapping


def read_history_record(
    history: deque[dict[str, Any]],
    *,
    record_id: str,
    id_field: str,
) -> dict[str, Any] | None:
    target_id = str(record_id)
    if not target_id:
        return None
    for item in list(history):
        if str(item.get(id_field, "")) == target_id:
            return deepcopy(item)
    return None


def replace_history_record(
    history: deque[dict[str, Any]],
    *,
    record_id: str,
    replacement: Mapping[str, Any],
    id_field: str,
) -> tuple[deque[dict[str, Any]], dict[str, Any] | None]:
    target_id = str(record_id)
    if not target_id:
        return history, None
    stored_replacement = deepcopy(dict(replacement))
    if str(stored_replacement.get(id_field, "")) != target_id:
        return history, None

    replaced = False
    updated: list[dict[str, Any]] = []
    for item in list(history):
        if not replaced and str(item.get(id_field, "")) == target_id:
            updated.append(deepcopy(stored_replacement))
            replaced = True
        else:
            updated.append(item)

    if not replaced:
        return history, None
    return deque(updated, maxlen=history.maxlen), deepcopy(stored_replacement)
