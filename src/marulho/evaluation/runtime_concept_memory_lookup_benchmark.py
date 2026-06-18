from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

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


def _legacy_direct_runtime_concept_lookup(
    store: DualMemoryStore,
    observations: Sequence[tuple[str | None, Mapping[str, Any] | None]],
    *,
    max_observations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    routing_keys = getattr(store, "slow_routing_keys", []) or []
    stored_texts = getattr(store, "slow_texts", []) or []
    stored_windows = getattr(store, "slow_raw_windows", []) or []
    slow_importance = getattr(store, "slow_importance", []) or []
    slow_capture_tag = getattr(store, "slow_capture_tag", []) or []
    slow_consolidation = getattr(store, "slow_consolidation_level", []) or []
    matches: list[dict[str, Any]] = []
    invalid_observation_count = 0
    invalid_memory_index_count = 0
    out_of_bounds_index_count = 0
    missing_routing_key_count = 0
    empty_text_count = 0
    raw_text_payload_count = 0
    processed = min(len(observations), max(0, int(max_observations)))
    for raw_window, metrics in observations[:processed]:
        if not isinstance(metrics, Mapping):
            invalid_observation_count += 1
            continue
        try:
            idx = int(metrics.get("memory_index"))
        except (TypeError, ValueError):
            invalid_memory_index_count += 1
            continue
        if idx < 0 or idx >= len(routing_keys):
            out_of_bounds_index_count += 1
            continue
        if not isinstance(routing_keys[idx], torch.Tensor):
            missing_routing_key_count += 1
            continue

        source_text = ""
        if idx < len(stored_texts) and stored_texts[idx] is not None:
            source_text = str(stored_texts[idx])
        elif idx < len(stored_windows) and stored_windows[idx] is not None:
            source_text = str(stored_windows[idx])
        elif raw_window is not None:
            source_text = str(raw_window)
        source_text = " ".join(source_text.split()).strip()
        if not source_text or not any(char.isalnum() for char in source_text):
            empty_text_count += 1
            continue

        raw_match = (
            str(stored_windows[idx])
            if idx < len(stored_windows) and stored_windows[idx] is not None
            else source_text
        )
        raw_text_payload_count += 1
        matches.append(
            {
                "memory_index": idx,
                "text": source_text,
                "raw_window": raw_match,
                "similarity": 1.0,
                "importance": (
                    float(slow_importance[idx])
                    if idx < len(slow_importance)
                    else 1.0
                ),
                "capture_tag": (
                    float(slow_capture_tag[idx])
                    if idx < len(slow_capture_tag)
                    else 0.0
                ),
                "consolidation_level": (
                    float(slow_consolidation[idx])
                    if idx < len(slow_consolidation)
                    else 0.0
                ),
            }
        )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return matches, {
        "surface": "retired_service_direct_runtime_concept_lookup.v1",
        "latency_ms": float(latency_ms),
        "match_indices": [int(match["memory_index"]) for match in matches],
        "match_count": int(len(matches)),
        "raw_text_payload_count": int(raw_text_payload_count),
        "invalid_observation_count": int(invalid_observation_count),
        "invalid_memory_index_count": int(invalid_memory_index_count),
        "out_of_bounds_index_count": int(out_of_bounds_index_count),
        "missing_routing_key_count": int(missing_routing_key_count),
        "empty_text_count": int(empty_text_count),
        "global_candidate_scan": False,
        "global_score_scan": False,
    }


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

    def legacy_call() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return _legacy_direct_runtime_concept_lookup(
            store,
            observations,
            max_observations=args.max_observations,
        )

    def bounded_call() -> dict[str, Any]:
        return store.resolve_runtime_concept_memory_matches(
            observations=observations,
            max_observations=args.max_observations,
        )

    legacy_latencies, legacy_results = _measure(legacy_call, args.iterations)
    bounded_latencies, bounded_results = _measure(bounded_call, args.iterations)
    legacy_matches, legacy_report = legacy_results[-1]
    bounded_result = bounded_results[-1]
    bounded_report = dict(bounded_result["report"])
    legacy_indices = [int(match["memory_index"]) for match in legacy_matches]
    bounded_indices = [int(match["memory_index"]) for match in bounded_result["matches"]]
    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
    speedup = legacy_mean / max(1e-9, bounded_mean)
    routing_sequence = store.slow_routing_keys
    text_sequence = store.slow_texts
    raw_sequence = store.slow_raw_windows
    quality_min = 1.0 if legacy_indices == bounded_indices else 0.0

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
        "latency_gate_pass": bool(speedup >= float(args.min_speedup)),
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
        "quality": {
            "metric": "selected_memory_index_parity",
            "min": float(quality_min),
            "selected_indices_match": bool(legacy_indices == bounded_indices),
            "legacy_selected_count": int(len(legacy_indices)),
            "bounded_selected_count": int(len(bounded_indices)),
        },
        "latency_ms": {
            "legacy_mean": float(legacy_mean),
            "bounded_mean": float(bounded_mean),
            "speedup": float(speedup),
            "legacy_min": float(min(legacy_latencies)),
            "bounded_min": float(min(bounded_latencies)),
        },
        "legacy_report": legacy_report,
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
    parser.add_argument("--min-speedup", type=float, default=1.0)
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
        f"speedup={report['latency_ms']['speedup']:.3f}"
    )
    print(
        "payload legacy={legacy} bounded={bounded} cache_hits={hits}".format(
            legacy=report["legacy_report"]["raw_text_payload_count"],
            bounded=report["bounded_report"]["raw_text_payload_count"],
            hits=report["bounded_report"]["raw_text_payload_cache_hits"],
        )
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
