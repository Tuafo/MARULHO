from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any, Mapping

import torch

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.reporting.readme_reports import write_json_report_with_readme


class _SyntheticTensorSequence:
    def __init__(self, size: int, dim: int) -> None:
        self.size = int(size)
        self.dim = int(dim)
        self.iteration_attempts = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        value = torch.zeros(self.dim, dtype=torch.float32)
        value[idx % self.dim] = 1.0
        return value

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_attempts += 1
        raise AssertionError("archive iteration is not allowed")


class _SyntheticTextSequence:
    def __init__(self, size: int, *, prefix: str, repeats: int) -> None:
        self.size = int(size)
        self.prefix = str(prefix)
        self.repeats = max(1, int(repeats))
        self.iteration_attempts = 0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> str:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        base = f"{self.prefix} {idx % 97} submarine ballast pressure concept evidence"
        return " ".join(base for _ in range(self.repeats))

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.iteration_attempts += 1
        raise AssertionError("archive iteration is not allowed")


class _SyntheticFloatSequence:
    def __init__(self, size: int, value: float) -> None:
        self.size = int(size)
        self.value = float(value)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> float:
        idx = int(index)
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        return self.value


class _SizedSequence:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size


def _build_store(*, capacity: int, dim: int, text_repeats: int) -> DualMemoryStore:
    store = DualMemoryStore(capacity=capacity)
    store.slow_buffer = _SizedSequence(capacity)  # type: ignore[assignment]
    store.slow_routing_keys = _SyntheticTensorSequence(capacity, dim)  # type: ignore[assignment]
    store.slow_texts = _SyntheticTextSequence(
        capacity,
        prefix="runtime text",
        repeats=text_repeats,
    )  # type: ignore[assignment]
    store.slow_raw_windows = _SyntheticTextSequence(
        capacity,
        prefix="runtime raw",
        repeats=max(1, text_repeats // 2),
    )  # type: ignore[assignment]
    store.slow_importance = _SyntheticFloatSequence(capacity, 0.8)  # type: ignore[assignment]
    store.slow_capture_tag = _SyntheticFloatSequence(capacity, 0.3)  # type: ignore[assignment]
    store.slow_consolidation_level = _SyntheticFloatSequence(capacity, 0.4)  # type: ignore[assignment]
    return store


def _build_observations(
    *,
    capacity: int,
    observation_count: int,
    unique_indices: int,
) -> list[tuple[str | None, dict[str, Any] | None]]:
    unique = max(1, min(int(unique_indices), int(capacity)))
    count = max(0, int(observation_count))
    stride = max(1, int(capacity) // unique)
    selected = [(index * stride) % int(capacity) for index in range(unique)]
    observations: list[tuple[str | None, dict[str, Any] | None]] = []
    for index in range(count):
        memory_index = selected[index % unique]
        observations.append((f"fallback runtime concept {memory_index}", {"memory_index": memory_index}))
    observations.extend(
        [
            ("invalid mapping", {}),
            ("missing metrics", None),
            ("out of bounds", {"memory_index": int(capacity) + 5}),
        ]
    )
    return observations


def _measure(fn: Any, iterations: int) -> tuple[list[float], list[Any]]:
    latencies: list[float] = []
    results: list[Any] = []
    for _ in range(max(1, int(iterations))):
        started = time.perf_counter()
        result = fn()
        latencies.append((time.perf_counter() - started) * 1000.0)
        results.append(result)
    return latencies, results


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    store = _build_store(
        capacity=args.capacity,
        dim=args.dim,
        text_repeats=args.text_repeats,
    )
    observations = _build_observations(
        capacity=args.capacity,
        observation_count=args.observation_count,
        unique_indices=args.unique_indices,
    )

    def bounded_call() -> dict[str, Any]:
        return store.resolve_runtime_concept_memory_matches(
            observations=observations,
            max_observations=args.max_observations,
        )

    bounded_latencies, bounded_results = _measure(bounded_call, args.iterations)
    bounded_result = bounded_results[-1]
    bounded_report = dict(bounded_result["report"])
    bounded_indices = [int(match["memory_index"]) for match in bounded_result["matches"]]
    bounded_mean = statistics.fmean(bounded_latencies)
    routing_sequence = store.slow_routing_keys
    text_sequence = store.slow_texts
    raw_sequence = store.slow_raw_windows
    expected_unique = []
    for _raw_window, metrics in observations[: int(args.max_observations)]:
        if not isinstance(metrics, Mapping) or "memory_index" not in metrics:
            continue
        try:
            index = int(metrics.get("memory_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < int(args.capacity) and index not in expected_unique:
            expected_unique.append(index)
    expected_unique = expected_unique[: int(args.unique_indices)]
    bounded_unique_indices = list(dict.fromkeys(bounded_indices))
    quality_min = 1.0 if bounded_unique_indices == expected_unique else 0.0

    gates = {
        "quality_gate_pass": bool(quality_min >= 1.0),
        "payload_cache_gate_pass": bool(
            int(bounded_report["raw_text_payload_count"]) <= int(args.unique_indices)
            and int(bounded_report["raw_text_payload_cache_hits"])
            >= max(0, int(args.observation_count) - int(args.unique_indices))
        ),
        "bounded_scope_gate_pass": bool(
            bounded_report["candidate_scope"] == "train_step_memory_index_evidence"
            and not bounded_report["global_candidate_scan"]
            and not bounded_report["global_score_scan"]
            and not bounded_report["language_reasoning"]
            and not bounded_report["runs_every_token"]
        ),
        "device_gate_pass": bool(
            bounded_report["archival_storage_device"] == "cpu"
            and bounded_report["score_device"] == "cpu"
        ),
        "latency_gate_pass": bool(bounded_mean <= float(args.max_bounded_mean_ms)),
        "no_archive_iteration_gate_pass": bool(
            getattr(routing_sequence, "iteration_attempts", 0) == 0
            and getattr(text_sequence, "iteration_attempts", 0) == 0
            and getattr(raw_sequence, "iteration_attempts", 0) == 0
        ),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "marulho_runtime_concept_memory_lookup_benchmark",
        "surface": "runtime_concept_memory_lookup_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(bool(value) for value in gates.values()),
        "capacity": int(args.capacity),
        "dim": int(args.dim),
        "iterations": int(args.iterations),
        "observation_count": int(args.observation_count),
        "unique_indices": int(args.unique_indices),
        "max_observations": int(args.max_observations),
        "text_repeats": int(args.text_repeats),
        "selection_criteria": "explicit train_step memory_index evidence",
        "memory_budget": {
            "archival_entries": int(args.capacity),
            "candidate_window_entries": int(args.max_observations),
            "unique_payload_budget_entries": int(args.unique_indices),
        },
        "retired_direct_runtime_concept_lookup_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "runtime_concept_direct_archive_lookup_comparator",
        },
        "quality": {
            "metric": "explicit_memory_index_evidence_recall",
            "min": float(quality_min),
            "selected_indices_match_expected": bool(
                bounded_unique_indices == expected_unique
            ),
            "expected_indices": expected_unique,
            "bounded_unique_indices": bounded_unique_indices,
            "bounded_selected_count": int(len(bounded_indices)),
        },
        "latency_ms": {
            "bounded_mean": float(bounded_mean),
            "bounded_min": float(min(bounded_latencies)),
            "max_bounded_mean_ms": float(args.max_bounded_mean_ms),
        },
        "bounded_report": bounded_report,
        "device_placement": {
            "archival_storage_device": bounded_report["archival_storage_device"],
            "score_device": bounded_report["score_device"],
            "active_replay_cuda_required": False,
        },
        "runtime_contract": {
            "global_candidate_scan": False,
            "global_score_scan": False,
            "runs_every_token": False,
            "cadenced_observation": True,
            "mutates_archival_memory": False,
            "applies_plasticity": False,
            "hidden_language_reasoning": False,
        },
        "gates": gates,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded runtime concept memory lookup"
    )
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--observation-count", type=int, default=512)
    parser.add_argument("--unique-indices", type=int, default=64)
    parser.add_argument("--max-observations", type=int, default=512)
    parser.add_argument("--text-repeats", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-bounded-mean-ms", type=float, default=500.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    report = run_benchmark(args)
    write_json_report_with_readme(
        args.output,
        report,
        title="Runtime Concept Memory Lookup Benchmark",
    )
    print(
        f"passed={report['passed']} "
        f"quality_min={report['quality']['min']:.6f} "
        f"bounded_mean_ms={report['latency_ms']['bounded_mean']:.3f}"
    )
    print(
        "payload bounded={bounded} cache_hits={hits}".format(
            bounded=report["bounded_report"]["raw_text_payload_count"],
            hits=report["bounded_report"]["raw_text_payload_cache_hits"],
        )
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
