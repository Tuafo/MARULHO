from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class BrainPattern:
    raw_window: str
    pattern: torch.Tensor
    source: str


class BrainSourceBuffer:
    """Bounded feed/source buffer consumed by MarulhoBrain.tick."""

    def __init__(self, *, max_items: int = 8192) -> None:
        self._items: deque[BrainPattern] = deque(maxlen=max(1, int(max_items)))
        self._dropped_total = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def dropped_total(self) -> int:
        return int(self._dropped_total)

    def extend(self, items: Iterable[BrainPattern]) -> int:
        added = 0
        maxlen = self._items.maxlen or 0
        for item in items:
            if maxlen and len(self._items) >= maxlen:
                self._dropped_total += 1
            self._items.append(item)
            added += 1
        return added

    def pop_batch(self, limit: int) -> list[BrainPattern]:
        count = min(max(0, int(limit)), len(self._items))
        return [self._items.popleft() for _ in range(count)]

    def snapshot(self) -> dict[str, int]:
        return {
            "queued_tokens": len(self._items),
            "dropped_total": int(self._dropped_total),
            "max_items": int(self._items.maxlen or 0),
        }
