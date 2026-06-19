from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch

from marulho.semantics import (
    SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT,
    SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
    build_spike_language_plasticity_shadow_delta,
    evaluate_spike_language_plasticity_replay,
    run_spike_language_plasticity_replay_experiment,
)


def _ready_trial() -> dict[str, Any]:
    return {
        "available": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "returns_trained_weights": False,
        "trial_summary": {
            "pre_pressure_score": 0.9,
            "post_pressure_score": 0.7,
            "expected_pressure_reduction": 0.2,
        },
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _ready_replay_evaluation() -> dict[str, Any]:
    return {
        "available": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "replay_evidence": {
            "pre_pressure_score": 0.9,
            "post_pressure_score": 0.7,
            "expected_pressure_reduction": 0.2,
        },
        "promotion_gate": {"status": "ready_for_operator_review"},
    }


def _ready_application_design() -> dict[str, Any]:
    return {
        "available": True,
        "owned_by_marulho": True,
        "device_evidence": {"device": "cpu", "source": "language_plasticity_replay_window_benchmark"},
        "application_design": {
            "learning_rate": 0.03,
            "max_weight_delta": 0.04,
            "locality_radius": 4,
            "grounded_replay_coverage": 0.8,
        },
    }


def _replay_window(count: int) -> list[dict[str, Any]]:
    return [
        {"case_id": f"language-plasticity-replay-{index}", "grounded": True}
        for index in range(int(count))
    ]


def _replay_sequences(count: int, *, index_count: int) -> list[dict[str, Any]]:
    return [
        {
            "sequence_id": f"language-plasticity-sequence-{index}",
            "pre_indices": list(range(int(index_count))),
            "post_indices": list(range(1, int(index_count) + 1)),
            "active_indices": list(range(int(index_count))),
            "grounded": True,
            "readout_evidence_hash": f"readout-{index}",
            "prediction_hash": f"prediction-{index}",
            "transition_memory_evaluation_hash": f"evaluation-{index}",
            "persistent_transition_weights_hash": f"weights-{index}",
        }
        for index in range(int(count))
    ]


def _case(*, payload_count: int, index_count: int) -> dict[str, Any]:
    replay_started = time.perf_counter()
    replay = evaluate_spike_language_plasticity_replay(
        _ready_trial(),
        replay_window=_replay_window(payload_count),
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "benchmark"},
    )
    replay_elapsed_ms = (time.perf_counter() - replay_started) * 1000.0

    experiment_started = time.perf_counter()
    experiment = run_spike_language_plasticity_replay_experiment(
        _ready_replay_evaluation(),
        replay_sequences=_replay_window(payload_count),
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "benchmark"},
    )
    experiment_elapsed_ms = (time.perf_counter() - experiment_started) * 1000.0

    shadow_started = time.perf_counter()
    shadow_delta = build_spike_language_plasticity_shadow_delta(
        _ready_application_design(),
        _replay_sequences(payload_count, index_count=index_count),
        device_evidence={"device": "cpu", "source": "language_plasticity_replay_window_benchmark"},
    )
    shadow_elapsed_ms = (time.perf_counter() - shadow_started) * 1000.0

    sparse_index_window = dict(shadow_delta.get("sparse_index_window") or {})
    first_index_report = (
        sparse_index_window.get("index_window_reports", [{}])[0]
        if sparse_index_window.get("index_window_reports")
        else {}
    )
    return {
        "replay_elapsed_ms": float(replay_elapsed_ms),
        "experiment_elapsed_ms": float(experiment_elapsed_ms),
        "shadow_elapsed_ms": float(shadow_elapsed_ms),
        "replay_window": replay["replay_window"],
        "experiment_sequence_window": experiment["replay_sequence_window"],
        "shadow_sequence_window": shadow_delta["replay_sequence_window"],
        "shadow_sparse_index_window": sparse_index_window,
        "first_sparse_index_window": first_index_report,
        "replay_window_count": int(replay["replay_evidence"]["replay_window_count"]),
        "experiment_sequence_count": int(experiment["replay_experiment"]["replay_sequence_count"]),
        "experiment_trace_count": len(experiment["ephemeral_replay"]["trace"]),
        "shadow_pair_check_count": int(sparse_index_window.get("pair_check_count", 0) or 0),
        "shadow_affected_synapse_count": int(shadow_delta.get("affected_synapse_count", 0) or 0),
        "shadow_available": bool(shadow_delta.get("available")),
        "runtime_mutation_absent": bool(
            not replay.get("mutates_runtime_state")
            and not experiment.get("mutates_runtime_state")
            and not shadow_delta.get("mutates_runtime_state")
            and not replay.get("applies_plasticity")
            and not experiment.get("applies_plasticity")
            and not shadow_delta.get("applies_plasticity")
        ),
    }


def _stats(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    elapsed = [float(row[f"{name}_elapsed_ms"]) for row in rows]
    return {
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
    }


def run_benchmark(*, payload_count: int, index_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    tracemalloc.start()
    rows = [
        _case(payload_count=int(payload_count), index_count=int(index_count))
        for _ in range(int(runs))
    ]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0

    last = rows[-1] if rows else {}
    replay_window = dict(last.get("replay_window") or {})
    experiment_window = dict(last.get("experiment_sequence_window") or {})
    shadow_window = dict(last.get("shadow_sequence_window") or {})
    sparse_window = dict(last.get("shadow_sparse_index_window") or {})
    first_index_window = dict(last.get("first_sparse_index_window") or {})
    pre_index_window = dict(first_index_window.get("pre_indices") or {})
    projected_full_pair_checks = int(payload_count) * int(index_count) * int(index_count)
    bounded_pair_checks = int(
        SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT
        * SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT
        * SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT
    )
    quality = {
        "replay_window_bounded": bool(
            replay_window.get("source_window_count") == SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT
            and replay_window.get("source_total_count") == int(payload_count)
            and replay_window.get("global_candidate_scan") is False
            and replay_window.get("global_score_scan") is False
            and replay_window.get("language_reasoning") is False
        ),
        "experiment_sequence_window_bounded": bool(
            experiment_window.get("source_window_count") == SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT
            and experiment_window.get("source_total_count") == int(payload_count)
            and experiment_window.get("global_candidate_scan") is False
            and experiment_window.get("global_score_scan") is False
            and experiment_window.get("language_reasoning") is False
        ),
        "shadow_sequence_window_bounded": bool(
            shadow_window.get("source_window_count") == SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT
            and shadow_window.get("source_total_count") == int(payload_count)
            and int(sparse_window.get("pair_check_count", 0) or 0) == bounded_pair_checks
        ),
        "sparse_index_window_bounded": bool(
            pre_index_window.get("source_window_count") == SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT
            and pre_index_window.get("source_total_count") == int(index_count)
            and int(pre_index_window.get("source_truncated_count", 0) or 0)
            == int(index_count) - SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT
        ),
        "runtime_mutation_absent": bool(last.get("runtime_mutation_absent")),
        "old_full_payload_path_executable": False,
        "old_full_payload_work_projected_from_source_count": True,
        "record_work_reduction": float(
            int(payload_count) / max(1, SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT)
        ),
        "pair_work_reduction": float(
            projected_full_pair_checks / max(1, bounded_pair_checks)
        ),
    }
    quality["pass"] = bool(
        quality["replay_window_bounded"]
        and quality["experiment_sequence_window_bounded"]
        and quality["shadow_sequence_window_bounded"]
        and quality["sparse_index_window_bounded"]
        and quality["runtime_mutation_absent"]
        and quality["record_work_reduction"] >= 2.0
        and quality["pair_work_reduction"] >= 2.0
    )
    return {
        "artifact_kind": "bounded_snn_language_plasticity_replay_window_benchmark",
        "surface": "bounded_snn_language_plasticity_replay_window_benchmark.v1",
        "payload_count": int(payload_count),
        "index_count": int(index_count),
        "window_limit": SNN_LANGUAGE_PLASTICITY_REPLAY_WINDOW_LIMIT,
        "index_limit": SNN_LANGUAGE_PLASTICITY_REPLAY_INDEX_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_replay_payload_order",
            "bounded_source_window_before_replay_materialization",
            "bounded_sparse_indices_before_pair_scoring",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "active_replay_computation_device": "cpu",
            "cuda_available": cuda_available,
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
            "cuda_memory_allocated_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_allocated_after_mib": float(cuda_after / (1024 * 1024)),
            "cuda_memory_reserved_before_mib": float(cuda_reserved_before / (1024 * 1024)),
            "cuda_memory_reserved_after_mib": float(cuda_reserved_after / (1024 * 1024)),
            "python_traced_peak_mib": float(peak_bytes / (1024 * 1024)),
        },
        "latency": {
            "replay_evaluation": _stats(rows, "replay"),
            "replay_experiment": _stats(rows, "experiment"),
            "shadow_delta": _stats(rows, "shadow"),
        },
        "last_run": last,
        "retired_full_payload_projection": {
            "executable_path_retired": True,
            "projected_record_count": int(payload_count),
            "projected_pair_check_count": projected_full_pair_checks,
            "bounded_pair_check_count": bounded_pair_checks,
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded SNN language plasticity replay windows."
    )
    parser.add_argument("--payload-count", type=int, default=2048)
    parser.add_argument("--index-count", type=int, default=256)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(
        payload_count=args.payload_count,
        index_count=args.index_count,
        runs=args.runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={passed} replay={replay:.0f}/{source} experiment={experiment:.0f}/{exp_source} "
        "pairs={pairs}/{projected_pairs} record_reduction={record_reduction:.6f} "
        "pair_reduction={pair_reduction:.6f}".format(
            passed=report["pass"],
            replay=report["last_run"]["replay_window_count"],
            source=report["last_run"]["replay_window"]["source_total_count"],
            experiment=report["last_run"]["experiment_sequence_count"],
            exp_source=report["last_run"]["experiment_sequence_window"]["source_total_count"],
            pairs=report["retired_full_payload_projection"]["bounded_pair_check_count"],
            projected_pairs=report["retired_full_payload_projection"]["projected_pair_check_count"],
            record_reduction=report["quality"]["record_work_reduction"],
            pair_reduction=report["quality"]["pair_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
