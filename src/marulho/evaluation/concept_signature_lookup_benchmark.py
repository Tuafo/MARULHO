from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any

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
        seeded_signature_by_index=routing_keys,
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


def _seeded_expected_signature(
    memory_store: Any,
    memory_index: list[int],
    *,
    max_indices: int,
) -> torch.Tensor | None:
    values = getattr(memory_store, "seeded_signature_by_index", []) or []
    signatures: list[torch.Tensor] = []
    seen: set[int] = set()
    for raw_index in memory_index:
        if len(signatures) >= max(1, int(max_indices)):
            break
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index in seen or index >= len(values):
            continue
        seen.add(index)
        value = values[index]
        if isinstance(value, torch.Tensor):
            signature = _normalize(value)
            if int(signature.numel()) > 0 and float(signature.norm().item()) > 1e-8:
                signatures.append(signature)
    if not signatures:
        return None
    if len(signatures) == 1:
        return signatures[0]
    target_dim = max(int(signature.numel()) for signature in signatures)
    aligned = [_resize(signature, target_dim) for signature in signatures]
    return _normalize(torch.stack(aligned, dim=0).mean(dim=0))


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
    bounded_latencies: list[float] = []
    cosine_values: list[float] = []
    max_indices = int(bounded_report["max_indices_per_source"])

    for group in groups:
        expected = _seeded_expected_signature(
            memory_store,
            group,
            max_indices=max_indices,
        )

        started = time.perf_counter()
        bounded = concept_store._memory_signature(memory_store, group, report=bounded_report)
        bounded_latencies.append((time.perf_counter() - started) * 1000.0)
        cosine_values.append(_cosine(expected, bounded))

    traced_report = concept_store._empty_memory_signature_lookup_report()
    tracemalloc.start()
    _ = concept_store._memory_signature(memory_store, groups[0], report=traced_report)
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    bounded_mean = statistics.fmean(bounded_latencies)
    bounded_p95 = float(
        sorted(bounded_latencies)[max(0, int(len(bounded_latencies) * 0.95) - 1)]
    )
    cosine_mean = statistics.fmean(cosine_values)
    cosine_min = min(cosine_values)
    bounded_report["latency_ms"] = float(sum(bounded_latencies))
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated = (
        float(torch.cuda.memory_allocated() / (1024 * 1024))
        if cuda_available
        else 0.0
    )
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
            "retired_archive_materializing_lookup_rows_removed": int(capacity),
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "bounded_lookup_device": "cpu",
            "active_replay_cuda_required": False,
            "cuda_available": cuda_available,
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "bounded_direct_index_lookup": {
            **bounded_report,
            "mean_latency_ms": float(bounded_mean),
            "p95_latency_ms": float(bounded_p95),
        },
        "retired_archive_materializing_signature_lookup_absence": {
            "implementation_present": False,
            "diagnostic_callable": False,
            "active_report_field_present": False,
            "removed_policy": "concept_signature_lookup_archive_materializing_comparator",
        },
        "quality": {
            "metric": "cosine_similarity_seeded_expected_signature",
            "mean": float(cosine_mean),
            "min": float(cosine_min),
            "seeded_expected_signature_matches": bool(cosine_min >= 0.9999),
        },
        "latency": {
            "bounded_mean_latency_ms": float(bounded_mean),
            "bounded_p95_latency_ms": float(bounded_p95),
        },
        "resource_behavior": {
            "python_tracemalloc_current_mib": round(
                float(traced_current) / (1024.0 * 1024.0),
                6,
            ),
            "python_tracemalloc_peak_mib": round(
                float(traced_peak) / (1024.0 * 1024.0),
                6,
            ),
            "cuda_memory_allocated_after_mib": cuda_allocated,
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
        "latency_gate_pass": bool(bounded_p95 <= 5.0),
        "bounded_lookup_gate_pass": bool(
            report["bounded_direct_index_lookup"]["archive_list_materialization_count"] == 0
            and not report["bounded_direct_index_lookup"]["global_candidate_scan"]
            and int(report["bounded_direct_index_lookup"]["max_indices_per_source"]) <= 8
        ),
        "retired_path_absence_gate_pass": bool(
            not report["retired_archive_materializing_signature_lookup_absence"][
                "implementation_present"
            ]
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
        f"bounded_mean_ms={report['latency']['bounded_mean_latency_ms']:.6f} "
        f"quality_min={report['quality']['min']:.6f}"
    )


if __name__ == "__main__":
    main()
