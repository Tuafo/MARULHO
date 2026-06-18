from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest

from marulho.service.operator_interaction import (
    RUNTIME_CONCEPT_MEMORY_LOOKUP_LIMIT,
    OperatorInteractionRuntime,
)


class _ResolverOnlyMemoryStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("slow_"):
            raise AssertionError("service must not direct-read archival memory")
        raise AttributeError(name)

    def resolve_runtime_concept_memory_matches(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "matches": [
                {
                    "memory_index": 7,
                    "text": "bounded concept memory",
                    "raw_window": "bounded concept raw",
                    "similarity": 1.0,
                    "importance": 0.8,
                    "capture_tag": 0.2,
                    "consolidation_level": 0.4,
                }
            ],
            "source_pairs": [("bounded concept memory", "bounded concept raw")],
            "result_slots": [0, None],
            "report": {
                "surface": "bounded_runtime_concept_memory_lookup.v1",
                "match_indices": [7],
            },
        }


class _ConceptStoreSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def observe(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "observed": True,
            "memory_index": kwargs["memory_matches"][0]["memory_index"],
        }


class OperatorInteractionRuntimeTests(unittest.TestCase):
    def test_runtime_concept_observation_uses_memory_store_resolver(self) -> None:
        memory_store = _ResolverOnlyMemoryStore()
        concept_store = _ConceptStoreSpy()
        runtime = SimpleNamespace(
            _trainer=SimpleNamespace(
                model=SimpleNamespace(
                    memory_store=memory_store,
                    abstraction_layer=None,
                )
            ),
            _concept_store=concept_store,
            _geometric_curiosity=SimpleNamespace(update_lexicon=lambda *args: None),
        )

        results = OperatorInteractionRuntime._observe_runtime_concept_batch_locked(
            runtime,
            observations=[
                ("bounded concept raw", {"memory_index": 7}),
                ("ignored", None),
            ],
        )

        self.assertEqual(results[0], {"observed": True, "memory_index": 7})
        self.assertIsNone(results[1])
        self.assertEqual(len(memory_store.calls), 1)
        self.assertEqual(
            memory_store.calls[0]["max_observations"],
            RUNTIME_CONCEPT_MEMORY_LOOKUP_LIMIT,
        )
        self.assertEqual(len(concept_store.calls), 1)
        self.assertEqual(
            concept_store.calls[0]["memory_matches"][0]["raw_window"],
            "bounded concept raw",
        )


if __name__ == "__main__":
    unittest.main()
