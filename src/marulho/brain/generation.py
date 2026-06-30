from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


TransitionKey = tuple[str, int]


def _surface_delta(left: str, right: str) -> str:
    left_text = str(left or "")
    right_text = str(right or "")
    if not right_text:
        return ""
    if left_text and right_text.startswith(left_text):
        delta = right_text[len(left_text) :]
        if delta:
            return delta[:1]
    return right_text[-1:]


class LocalTransitionReadout:
    """Small MARULHO-owned text readout over local sparse/SNN state."""

    surface = "marulho_local_transition_readout.v1"

    def __init__(self) -> None:
        self._transitions: dict[int, Counter[TransitionKey]] = {}
        self._observed_transition_count = 0

    @property
    def observed_transition_count(self) -> int:
        return int(self._observed_transition_count)

    @property
    def state_count(self) -> int:
        return len(self._transitions)

    def observe_sequence(
        self,
        state_keys: Iterable[int],
        raw_windows: Iterable[str],
    ) -> int:
        keys = [int(key) for key in state_keys]
        windows = [str(window) for window in raw_windows]
        if len(keys) < 2 or len(windows) < 2:
            return 0
        observed = 0
        for index in range(min(len(keys), len(windows)) - 1):
            char = _surface_delta(windows[index], windows[index + 1])
            if not char:
                continue
            source_key = int(keys[index])
            target_key = int(keys[index + 1])
            counter = self._transitions.setdefault(source_key, Counter())
            counter[(char, target_key)] += 1
            observed += 1
        self._observed_transition_count += observed
        return observed

    def generate(self, start_key: int | None, max_tokens: int) -> dict[str, Any]:
        limit = max(0, int(max_tokens))
        if start_key is None or limit <= 0:
            return self._empty_generation(start_key=start_key)

        requested_start_key = int(start_key)
        key = requested_start_key
        fallback_start_key_used = False
        if key not in self._transitions and self._transitions:
            key = min(
                self._transitions,
                key=lambda candidate: (
                    -sum(self._transitions[candidate].values()),
                    int(candidate),
                ),
            )
            fallback_start_key_used = True
        emitted: list[str] = []
        visited_states: list[int] = [key]
        used_transition_count = 0
        for _ in range(limit):
            counter = self._transitions.get(key)
            if not counter:
                break
            (char, next_key), _count = min(
                counter.items(),
                key=lambda item: (-int(item[1]), item[0][0], int(item[0][1])),
            )
            emitted.append(str(char))
            key = int(next_key)
            visited_states.append(key)
            used_transition_count += 1

        text = "".join(emitted)
        return {
            "surface": self.surface,
            "text": text,
            "available": bool(text),
            "start_key": int(key),
            "requested_start_key": requested_start_key,
            "fallback_start_key_used": bool(fallback_start_key_used),
            "end_key": int(key),
            "max_tokens": limit,
            "emitted_tokens": len(emitted),
            "visited_state_count": len(set(visited_states)),
            "used_transition_count": int(used_transition_count),
            "observed_transition_count": int(self._observed_transition_count),
            "transition_state_count": int(self.state_count),
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "thought_loop_used": False,
            "cortex_used": False,
        }

    def to_state(self) -> dict[str, Any]:
        transitions: list[dict[str, Any]] = []
        for source_key, counter in sorted(self._transitions.items()):
            for (char, target_key), count in sorted(
                counter.items(),
                key=lambda item: (int(item[0][1]), item[0][0]),
            ):
                transitions.append(
                    {
                        "from": int(source_key),
                        "char": str(char),
                        "to": int(target_key),
                        "count": int(count),
                    }
                )
        return {
            "surface": self.surface,
            "observed_transition_count": int(self._observed_transition_count),
            "transitions": transitions,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None) -> "LocalTransitionReadout":
        readout = cls()
        if not isinstance(state, Mapping):
            return readout
        for raw in list(state.get("transitions") or []):
            if not isinstance(raw, Mapping):
                continue
            try:
                source_key = int(raw.get("from"))
                target_key = int(raw.get("to"))
                char = str(raw.get("char") or "")[:1]
                count = max(1, int(raw.get("count", 1)))
            except (TypeError, ValueError):
                continue
            if not char:
                continue
            readout._transitions.setdefault(source_key, Counter())[
                (char, target_key)
            ] += count
        readout._observed_transition_count = max(
            int(state.get("observed_transition_count", 0) or 0),
            sum(sum(counter.values()) for counter in readout._transitions.values()),
        )
        return readout

    def _empty_generation(self, *, start_key: int | None) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "text": "",
            "available": False,
            "start_key": None if start_key is None else int(start_key),
            "end_key": None if start_key is None else int(start_key),
            "max_tokens": 0,
            "emitted_tokens": 0,
            "visited_state_count": 0,
            "used_transition_count": 0,
            "observed_transition_count": int(self._observed_transition_count),
            "transition_state_count": int(self.state_count),
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "thought_loop_used": False,
            "cortex_used": False,
        }
