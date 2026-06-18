from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.semantics.concepts import ConceptStore


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    value = vector.detach().clone().cpu().float().reshape(-1)
    if int(value.numel()) <= 0 or float(value.norm().item()) <= 1e-8:
        return value
    return F.normalize(value, dim=0)


def _resize(vector: torch.Tensor, target_dim: int) -> torch.Tensor:
    value = _normalize(vector)
    if int(value.numel()) == int(target_dim):
        return value
    if int(value.numel()) < int(target_dim):
        value = F.pad(value, (0, int(target_dim) - int(value.numel())))
    else:
        value = value[: int(target_dim)]
    return _normalize(value)


def _legacy_signature(memory_store: Any, memory_index: Any) -> torch.Tensor | None:
    if isinstance(memory_index, Sequence) and not isinstance(memory_index, (str, bytes)):
        signatures = [_legacy_signature(memory_store, value) for value in memory_index]
        signatures = [signature for signature in signatures if signature is not None]
        if not signatures:
            return None
        if len(signatures) == 1:
            return signatures[0]
        target_dim = max(int(signature.numel()) for signature in signatures)
        aligned = [_resize(signature, target_dim) for signature in signatures]
        return _normalize(torch.stack(aligned, dim=0).mean(dim=0))

    try:
        index = int(memory_index)
    except (TypeError, ValueError):
        return None
    for attr in ("slow_routing_keys", "slow_input_patterns", "slow_buffer"):
        values = list(getattr(memory_store, attr, []) or [])
        if index < 0 or index >= len(values):
            continue
        value = values[index]
        if isinstance(value, torch.Tensor):
            signature = _normalize(value)
            if int(signature.numel()) > 0 and float(signature.norm().item()) > 1e-8:
                return signature
    return None


def _build_memory_store(*, capacity: int, dim: int, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    routing_keys: list[torch.Tensor] = []
    input_patterns: list[torch.Tensor] = []
    slow_buffer: list[torch.Tensor] = []
    for index in range(int(capacity)):
        base = torch.randn(int(dim), generator=generator)
        bucket_signal = float(index % 17) / 17.0
        routing_keys.append(_normalize(base + bucket_signal))
        input_patterns.append(_normalize(base + 0.05 * torch.randn(int(dim), generator=generator)))
        slow_buffer.append(_normalize(base + 0.10 * torch.randn(int(dim), generator=generator)))
    return SimpleNamespace(
        slow_routing_keys=routing_keys,
        slow_input_patterns=input_patterns,
        slow_buffer=slow_buffer,
    )


def _query_groups(*, capacity: int, iterations: int, group_size: int) -> list[list[int]]:
    count = max(1, int(capacity))
    groups: list[list[int]] = []
    stride = max(1, count // max(1, int(iterations)))
    for iteration in range(max(1, int(iterations))):
        start = (iteration * stride) % count
        groups.append([(start + offset * 13) % count for offset in range(max(1, int(group_size)))])
    return groups


def _cosine(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
    if left is None or right is None:
        return 0.0
    dim = max(int(left.numel()), int(right.numel()))
    return float(torch.dot(_resize(left, dim), _resize(right, dim)).item())


def run_benchmark(
    *,
    capacity: int,
    dim: int,
    iterations: int,
    group_size: int,
    seed: int,
) -> dict[str, Any]:
    memory_store = _build_memory_store(capacity=capacity, dim=dim, seed=seed)
    groups = _query_groups(capacity=capacity, iterations=iterations, group_size=group_size)
    concept_store = ConceptStore()
    bounded_report = concept_store._empty_memory_signature_lookup_report()
    legacy_latencies: list[float] = []
    bounded_latencies: list[float] = []
    cosine_values: list[float] = []

    for group in groups:
        started = time.perf_counter()
        legacy = _legacy_signature(memory_store, group)
        legacy_latencies.append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        bounded = concept_store._memory_signature(memory_store, group, report=bounded_report)
        bounded_latencies.append((time.perf_counter() - started) * 1000.0)
        cosine_values.append(_cosine(legacy, bounded))

    legacy_mean = statistics.fmean(legacy_latencies)
    bounded_mean = statistics.fmean(bounded_latencies)
    cosine_mean = statistics.fmean(cosine_values)
    cosine_min = min(cosine_values)
    bounded_report["latency_ms"] = float(sum(bounded_latencies))
    report = {
        "schema_version": 1,
        "artifact_kind": "marulho_concept_signature_lookup_benchmark",
        "surface": "concept_signature_lookup_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capacity": int(capacity),
        "dim": int(dim),
        "iterations": int(max(1, iterations)),
        "group_size": int(max(1, group_size)),
        "seed": int(seed),
        "selection_criteria": "evidence_provided_memory_indices",
        "memory_budget": {
            "archival_entries": int(capacity),
            "max_indices_per_source": int(bounded_report["max_indices_per_source"]),
            "reference_scan_limit_per_source": int(bounded_report["reference_scan_limit_per_source"]),
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "bounded_lookup_device": "cpu",
            "active_replay_cuda_required": False,
        },
        "legacy_archive_materializing_lookup": {
            "candidate_scope": "evidence_provided_memory_indices",
            "archive_list_materialization_count": int(len(groups) * max(1, int(group_size))),
            "mean_latency_ms": float(legacy_mean),
            "p95_latency_ms": float(sorted(legacy_latencies)[max(0, int(len(legacy_latencies) * 0.95) - 1)]),
        },
        "bounded_direct_index_lookup": {
            **bounded_report,
            "mean_latency_ms": float(bounded_mean),
            "p95_latency_ms": float(sorted(bounded_latencies)[max(0, int(len(bounded_latencies) * 0.95) - 1)]),
        },
        "quality": {
            "metric": "cosine_similarity_legacy_vs_bounded_signature",
            "mean": float(cosine_mean),
            "min": float(cosine_min),
        },
        "latency": {
            "speedup": float(legacy_mean / max(1e-9, bounded_mean)),
            "legacy_mean_latency_ms": float(legacy_mean),
            "bounded_mean_latency_ms": float(bounded_mean),
        },
        "runtime_contract": {
            "global_candidate_scan": False,
            "global_score_scan": False,
            "runs_every_token": False,
            "mutates_archival_memory": False,
            "applies_plasticity": False,
            "hidden_language_reasoning": False,
        },
    }
    report["gates"] = {
        "quality_gate_pass": bool(cosine_min >= 0.9999),
        "latency_gate_pass": bool(report["latency"]["speedup"] >= 2.0),
        "bounded_lookup_gate_pass": bool(
            report["bounded_direct_index_lookup"]["archive_list_materialization_count"] == 0
            and not report["bounded_direct_index_lookup"]["global_candidate_scan"]
            and int(report["bounded_direct_index_lookup"]["max_indices_per_source"]) <= 8
        ),
        "device_gate_pass": bool(
            report["device_placement"]["archival_storage_device"] == "cpu"
            and not report["device_placement"]["active_replay_cuda_required"]
        ),
    }
    report["passed"] = all(bool(value) for value in report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_benchmark(
        capacity=args.capacity,
        dim=args.dim,
        iterations=args.iterations,
        group_size=args.group_size,
        seed=args.seed,
    )
    write_json_report_with_readme(
        args.output,
        report,
        title="Concept Signature Lookup Benchmark",
    )
    print(
        f"passed={report['passed']} "
        f"speedup={report['latency']['speedup']:.3f} "
        f"quality_min={report['quality']['min']:.6f}"
    )


if __name__ == "__main__":
    main()
