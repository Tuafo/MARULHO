from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from marulho.retrieval.routing_index import (
    HierarchicalAssemblyIndex,
    ShardedHierarchicalAssemblyIndex,
)


def _index(
    *,
    entries: int,
    dim: int,
    shards: int,
    device: torch.device,
    seed: int,
) -> Any:
    torch.manual_seed(int(seed))
    vectors = F.normalize(
        torch.randn(int(entries), int(dim), dtype=torch.float32, device=device),
        dim=1,
    )
    ids = np.arange(int(entries), dtype=np.int64)
    if int(shards) > 1:
        index = ShardedHierarchicalAssemblyIndex(
            dim=int(dim),
            n_shards=int(shards),
            rebuild_threshold=max(1, int(entries) * 2),
            device=device,
        )
    else:
        index = HierarchicalAssemblyIndex(
            dim=int(dim),
            rebuild_threshold=max(1, int(entries) * 2),
            device=device,
        )
    index.add(vectors, ids)
    index.rebuild()
    index.routing_tensor_cache()
    return index


def _update_vectors(
    *,
    update_count: int,
    dim: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(int(seed))
    return F.normalize(
        torch.randn(int(update_count), int(dim), dtype=torch.float32, device=device),
        dim=1,
    )


def _measure(fn: Any, *, runs: int) -> tuple[list[float], list[dict[str, Any]]]:
    elapsed: list[float] = []
    rows: list[dict[str, Any]] = []
    for _ in range(int(runs)):
        started = time.perf_counter()
        row = dict(fn())
        elapsed.append(float((time.perf_counter() - started) * 1000.0))
        rows.append(row)
    return elapsed, rows


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values)) if values else 0.0,
        "median_ms": float(statistics.median(values)) if values else 0.0,
        "min_ms": float(min(values)) if values else 0.0,
        "max_ms": float(max(values)) if values else 0.0,
    }


def _top1_matches(index: Any, vectors: torch.Tensor, ids: np.ndarray) -> bool:
    found, _dist = index.search_tensors(vectors, k=1)
    if found.numel() != len(ids):
        return False
    expected = torch.tensor(ids, dtype=torch.long, device=found.device).reshape(-1, 1)
    return bool(torch.equal(found.long(), expected.long()))


def run_benchmark(
    *,
    entries: int,
    dim: int,
    update_count: int,
    missing_update_count: int,
    runs: int,
    shards: int,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    full_index = _index(
        entries=entries,
        dim=dim,
        shards=shards,
        device=device,
        seed=seed,
    )
    bounded_index = _index(
        entries=entries,
        dim=dim,
        shards=shards,
        device=device,
        seed=seed,
    )
    existing_update_ids = np.arange(int(update_count), dtype=np.int64)
    missing_update_ids = (
        np.arange(
            int(entries),
            int(entries) + int(missing_update_count),
            dtype=np.int64,
        )
        if int(missing_update_count) > 0
        else np.empty(0, dtype=np.int64)
    )
    update_ids = np.concatenate([existing_update_ids, missing_update_ids])
    total_update_count = int(len(update_ids))

    def _full() -> dict[str, Any]:
        update_vectors = _update_vectors(
            update_count=total_update_count,
            dim=dim,
            device=device,
            seed=seed + 10_000,
        )
        rebuild_count_before = int(full_index.stats()["rebuild_count"])
        full_index.add(update_vectors, update_ids)
        full_index.rebuild()
        return {
            "rebuild_count_delta": int(
                int(full_index.stats()["rebuild_count"]) - rebuild_count_before
            ),
            "cache_dirty_after": bool(full_index.stats().get("torch_cache_dirty", False))
            if shards <= 1
            else bool(full_index.stats().get("merged_torch_cache_dirty", False)),
            "top1_matches": _top1_matches(full_index, update_vectors, update_ids),
            "missing_update_count": int(missing_update_count),
        }

    def _bounded() -> dict[str, Any]:
        update_vectors = _update_vectors(
            update_count=total_update_count,
            dim=dim,
            device=device,
            seed=seed + 10_000,
        )
        rebuild_count_before = int(bounded_index.stats()["rebuild_count"])
        report = dict(bounded_index.update_existing(update_vectors, update_ids))
        bounded_size = int(bounded_index.ntotal)
        return {
            **report,
            "rebuild_count_delta": int(
                int(bounded_index.stats()["rebuild_count"]) - rebuild_count_before
            ),
            "top1_matches": _top1_matches(
                bounded_index,
                update_vectors[: int(update_count)],
                existing_update_ids,
            ),
            "missing_update_count": int(missing_update_count),
            "missing_ids_absent": bool(bounded_size == int(entries)),
        }

    full_elapsed, full_rows = _measure(_full, runs=runs)
    bounded_elapsed, bounded_rows = _measure(_bounded, runs=runs)
    full_stats = _latency(full_elapsed)
    bounded_stats = _latency(bounded_elapsed)
    bounded_last = bounded_rows[-1] if bounded_rows else {}
    full_last = full_rows[-1] if full_rows else {}
    quality = {
        "bounded_top1_matches": bool(bounded_last.get("top1_matches")),
        "full_top1_matches": bool(full_last.get("top1_matches")),
        "bounded_no_full_rebuild": bool(
            int(bounded_last.get("rebuild_count_delta", -1)) == 0
            and not bool(bounded_last.get("cache_dirty_after", True))
        ),
        "bounded_deferred_missing_recovery": bool(
            int(bounded_last.get("missing_id_count", -1)) == int(missing_update_count)
            and int(bounded_last.get("skipped_update_count", -1))
            == int(missing_update_count)
            and bool(bounded_last.get("recovery_required")) == (
                int(missing_update_count) > 0
            )
        ),
        "bounded_missing_ids_absent": bool(
            bounded_last.get("missing_ids_absent", False)
        ),
        "bounded_row_map_lookup": bool(
            bounded_last.get("row_lookup_mode") == "host_id_row_map"
        ),
        "full_path_rebuilt": bool(int(full_last.get("rebuild_count_delta", 0)) > 0),
        "latency_reduction_mean_ms": float(
            max(0.0, full_stats["mean_ms"] - bounded_stats["mean_ms"])
        ),
    }
    quality["pass"] = bool(
        quality["bounded_top1_matches"]
        and quality["full_top1_matches"]
        and quality["bounded_no_full_rebuild"]
        and quality["bounded_deferred_missing_recovery"]
        and quality["bounded_missing_ids_absent"]
        and quality["bounded_row_map_lookup"]
        and quality["full_path_rebuilt"]
        and bounded_stats["mean_ms"] < full_stats["mean_ms"]
    )
    cuda_available = bool(torch.cuda.is_available())
    cuda_allocated = (
        float(torch.cuda.memory_allocated(device) / (1024 * 1024))
        if device.type == "cuda"
        else 0.0
    )
    return {
        "artifact_kind": "sleep_replay_routing_index_refresh_benchmark",
        "surface": "routing_index_existing_row_refresh.v1",
        "entries": int(entries),
        "dim": int(dim),
        "update_count": int(update_count),
        "missing_update_count": int(missing_update_count),
        "requested_update_count": int(total_update_count),
        "runs": int(runs),
        "shards": int(shards),
        "selection_criteria": [
            "selected_sleep_replay_updated_prototype_ids",
            "routing_cache_ready",
            "existing_column_ids_update_in_place",
            "missing_or_unmapped_ids_defer_recovery",
        ],
        "memory_budget": {
            "selected_rows_updated": int(update_count),
            "selected_rows_deferred": int(missing_update_count),
            "full_rebuild_rows_retired": int(entries),
            "archival_storage_device": "cpu",
            "routing_row_lookup_metadata_device": "cpu",
            "routing_row_lookup_metadata_rows": int(entries),
        },
        "device_placement": {
            "routing_index_device": str(device),
            "archival_storage_device": "cpu",
            "routing_row_lookup_metadata_device": "cpu",
            "cuda_available": cuda_available,
            "cuda_memory_allocated_after_mib": cuda_allocated,
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "active_replay_window_only": True,
            "routing_index_full_rebuild": False,
            "routing_index_deferred_recovery": bool(
                int(missing_update_count) > 0
            ),
            "global_candidate_scan": False,
            "hidden_language_reasoning": False,
        },
        "retired_path": {
            "name": "full routing-index rebuild after selected sleep replay",
            "replacement": "routing_index_existing_row_refresh.v1",
        },
        "latency": {
            "retired_full_rebuild": full_stats,
            "bounded_existing_row_refresh": bounded_stats,
        },
        "last_full_rebuild": full_last,
        "last_bounded_refresh": bounded_last,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark selected-row routing-index refresh for sleep replay."
    )
    parser.add_argument("--entries", type=int, default=65_536)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--update-count", type=int, default=16)
    parser.add_argument("--missing-update-count", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        entries=args.entries,
        dim=args.dim,
        update_count=args.update_count,
        missing_update_count=args.missing_update_count,
        runs=args.runs,
        shards=args.shards,
        seed=args.seed,
        device_name=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={pass_gate} entries={entries} update_count={updates} "
        "missing_update_count={missing_updates} full_mean_ms={full_ms:.6f} "
        "bounded_mean_ms={bounded_ms:.6f} bounded_rebuild_delta={bounded_rebuild} "
        "deferred={deferred}".format(
            pass_gate=report["pass"],
            entries=report["entries"],
            updates=report["update_count"],
            missing_updates=report["missing_update_count"],
            full_ms=report["latency"]["retired_full_rebuild"]["mean_ms"],
            bounded_ms=report["latency"]["bounded_existing_row_refresh"]["mean_ms"],
            bounded_rebuild=report["last_bounded_refresh"].get(
                "rebuild_count_delta"
            ),
            deferred=report["last_bounded_refresh"].get("recovery_required"),
        )
    )


if __name__ == "__main__":
    main()
