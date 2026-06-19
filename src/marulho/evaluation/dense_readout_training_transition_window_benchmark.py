from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT,
    SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT,
    SNNLanguagePlasticityApplicationExecutor,
)


def _preflight() -> dict[str, Any]:
    return {
        "surface": "snn_language_dense_readout_training_loop_preflight.v1",
        "ready": True,
        "owned_by_marulho": True,
        "executable": False,
        "mutates_runtime_state": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "checkpoint_path": "dense-training.pt",
        "preflight_hash": "sha256:dense-training-window-benchmark",
        "tensor_summary": {
            "shape": [128, 128],
            "device": "cpu",
            "dtype": "torch.float32",
            "nonzero_count": 0,
        },
        "training_design": {
            "training_transition_count": (
                SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            ),
            "validation_transition_count": 4,
            "learning_rate": 0.02,
            "max_delta_norm": 0.05,
            "transition_budget": (
                SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            ),
            "requires_cuda": False,
        },
        "promotion_gate": {
            "status": "ready_for_checkpoint_backed_dense_readout_training_executor",
            "required_evidence": {
                "expected_state_revision_current": True,
                "checkpoint_path_available": True,
                "bounded_delta_application_capability_available": True,
            },
        },
    }


def _transitions(count: int) -> list[dict[str, Any]]:
    return [
        {
            "transition_id": f"training-window-{index}",
            "pre_indices": [index % 64],
            "post_indices": [(index % 64) + 1],
        }
        for index in range(int(count))
    ]


def _executor(
    tmp_dir: Path,
) -> tuple[
    SNNLanguagePlasticityApplicationExecutor,
    RuntimeState,
    dict[str, Any],
    list[str | None],
]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state: dict[str, Any] = {
        "dense_readout_weights": torch.zeros((128, 128), dtype=torch.float32),
        "sparse_transition_weights": {},
    }
    checkpoint_calls: list[str | None] = []

    def save_checkpoint(path: str | None) -> dict[str, str]:
        checkpoint_calls.append(path)
        target = Path(path or tmp_dir / "checkpoint.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"checkpoint")
        return {"path": str(target)}

    executor = SNNLanguagePlasticityApplicationExecutor(
        lock=lock,
        runtime_state=runtime_state,
        language_plasticity_state=lambda: language_state,
        save_checkpoint=save_checkpoint,
        checkpoint_path=lambda: tmp_dir / "checkpoint.pt",
        verify_checkpoint=lambda path: path.exists(),
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )
    return executor, runtime_state, language_state, checkpoint_calls


def _case(*, payload_count: int, index_count: int, tmp_dir: Path, case_index: int) -> dict[str, Any]:
    case_dir = tmp_dir / f"case-{case_index}"
    case_dir.mkdir(parents=True, exist_ok=True)

    payload_executor, payload_state, payload_language_state, payload_checkpoint_calls = _executor(
        case_dir / "payload-block"
    )
    payload_started = time.perf_counter()
    payload_block = payload_executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_preflight(),
        training_transitions=_transitions(payload_count),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
        checkpoint_path=str(case_dir / "payload-block" / "dense-training.pt"),
    )
    payload_block_elapsed_ms = (time.perf_counter() - payload_started) * 1000.0

    index_executor, index_state, index_language_state, index_checkpoint_calls = _executor(
        case_dir / "index-block"
    )
    index_started = time.perf_counter()
    index_block = index_executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_preflight(),
        training_transitions=[
            {
                "transition_id": "oversized-index-window",
                "pre_indices": list(range(int(index_count))),
                "post_indices": [2],
            }
        ],
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
        checkpoint_path=str(case_dir / "index-block" / "dense-training.pt"),
    )
    index_block_elapsed_ms = (time.perf_counter() - index_started) * 1000.0

    exact_executor, exact_state, exact_language_state, exact_checkpoint_calls = _executor(
        case_dir / "exact-window"
    )
    exact_started = time.perf_counter()
    exact = exact_executor.apply_dense_readout_training_loop(
        dense_readout_training_loop_preflight=_preflight(),
        training_transitions=_transitions(
            SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
        ),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
        checkpoint_path=str(case_dir / "exact-window" / "dense-training.pt"),
    )
    exact_elapsed_ms = (time.perf_counter() - exact_started) * 1000.0
    exact_snapshot = exact_executor.snapshot()

    payload_evidence = dict(
        payload_block.get("promotion_gate", {}).get("required_evidence") or {}
    )
    index_evidence = dict(
        index_block.get("promotion_gate", {}).get("required_evidence") or {}
    )
    exact_event = dict(exact.get("dense_readout_training") or {})
    return {
        "payload_block_elapsed_ms": float(payload_block_elapsed_ms),
        "index_block_elapsed_ms": float(index_block_elapsed_ms),
        "exact_elapsed_ms": float(exact_elapsed_ms),
        "payload_block": {
            "accepted": bool(payload_block.get("accepted")),
            "reason": payload_block.get("reason"),
            "transition_source_window": dict(
                payload_evidence.get("training_transition_source_window") or {}
            ),
            "index_source_window": dict(
                payload_evidence.get("training_transition_index_source_window") or {}
            ),
            "payload_not_truncated": payload_evidence.get(
                "training_transition_payload_not_truncated"
            ),
            "checkpoint_call_count": len(payload_checkpoint_calls),
            "state_revision": int(payload_state.state_revision),
            "weight_count": len(
                payload_language_state.get("sparse_transition_weights") or {}
            ),
        },
        "index_block": {
            "accepted": bool(index_block.get("accepted")),
            "reason": index_block.get("reason"),
            "transition_source_window": dict(
                index_evidence.get("training_transition_source_window") or {}
            ),
            "index_source_window": dict(
                index_evidence.get("training_transition_index_source_window") or {}
            ),
            "index_payload_not_truncated": index_evidence.get(
                "training_transition_index_payload_not_truncated"
            ),
            "checkpoint_call_count": len(index_checkpoint_calls),
            "state_revision": int(index_state.state_revision),
            "weight_count": len(
                index_language_state.get("sparse_transition_weights") or {}
            ),
        },
        "exact_window": {
            "accepted": bool(exact.get("accepted")),
            "transition_source_window": dict(
                exact.get("training_transition_source_window") or {}
            ),
            "index_source_window": dict(
                exact.get("training_transition_index_source_window") or {}
            ),
            "checkpoint_call_count": len(exact_checkpoint_calls),
            "state_revision": int(exact_state.state_revision),
            "weight_count": len(
                exact_language_state.get("sparse_transition_weights") or {}
            ),
            "dense_nonzero_count": int(
                exact_snapshot.get("dense_readout_tensor", {}).get("nonzero_count", 0)
                or 0
            ),
            "training_transition_count": int(
                exact_event.get("training_transition_count", 0) or 0
            ),
            "updated_cell_count": int(exact_event.get("updated_cell_count", 0) or 0),
            "active_training_device": str(exact_event.get("active_training_device") or ""),
            "tensor_is_cuda": bool(exact_event.get("tensor_is_cuda")),
        },
    }


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    elapsed = [float(row[f"{key}_elapsed_ms"]) for row in rows]
    return {
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
    }


def run_benchmark(*, payload_count: int, index_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="marulho-training-window-") as raw_tmp:
        tmp_dir = Path(raw_tmp)
        rows = [
            _case(
                payload_count=int(payload_count),
                index_count=int(index_count),
                tmp_dir=tmp_dir,
                case_index=index,
            )
            for index in range(int(runs))
        ]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0

    last = rows[-1] if rows else {}
    payload_block = dict(last.get("payload_block") or {})
    payload_window = dict(payload_block.get("transition_source_window") or {})
    index_block = dict(last.get("index_block") or {})
    index_window = dict(index_block.get("index_source_window") or {})
    exact = dict(last.get("exact_window") or {})
    exact_transition_window = dict(exact.get("transition_source_window") or {})
    exact_index_window = dict(exact.get("index_source_window") or {})

    reduction = float(
        int(payload_count)
        / max(1, SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT)
    )
    quality = {
        "oversized_transition_payload_blocked": bool(
            payload_block.get("accepted") is False
            and payload_window.get("source_window_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            and payload_window.get("source_total_count") == int(payload_count)
            and payload_window.get("source_payload_truncated") is True
            and payload_block.get("payload_not_truncated") is False
            and payload_block.get("checkpoint_call_count") == 0
            and payload_block.get("state_revision") == 0
            and payload_block.get("weight_count") == 0
        ),
        "oversized_index_payload_blocked": bool(
            index_block.get("accepted") is False
            and index_window.get("source_payload_truncated") is True
            and index_block.get("index_payload_not_truncated") is False
            and index_block.get("checkpoint_call_count") == 0
            and index_block.get("state_revision") == 0
            and index_block.get("weight_count") == 0
        ),
        "exact_window_still_trains": bool(
            exact.get("accepted") is True
            and exact_transition_window.get("source_window_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            and exact_transition_window.get("source_payload_truncated") is False
            and exact_index_window.get("source_payload_truncated") is False
            and exact.get("checkpoint_call_count") == 2
            and exact.get("state_revision") == 1
            and exact.get("training_transition_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            and exact.get("updated_cell_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            and exact.get("weight_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            and exact.get("dense_nonzero_count")
            == SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
        ),
        "dense_sparse_transition_reconstruction_matches": bool(
            exact.get("weight_count") == exact.get("dense_nonzero_count")
            == exact.get("updated_cell_count")
        ),
        "no_global_scans": bool(
            payload_window.get("global_candidate_scan") is False
            and payload_window.get("global_score_scan") is False
            and index_window.get("global_candidate_scan") is False
            and index_window.get("global_score_scan") is False
        ),
        "record_work_reduction": reduction,
        "old_full_payload_path_executable": False,
        "old_full_payload_work_projected_from_source_count": True,
    }
    quality["pass"] = bool(
        quality["oversized_transition_payload_blocked"]
        and quality["oversized_index_payload_blocked"]
        and quality["exact_window_still_trains"]
        and quality["dense_sparse_transition_reconstruction_matches"]
        and quality["no_global_scans"]
        and quality["record_work_reduction"] >= 2.0
    )
    return {
        "artifact_kind": "bounded_snn_dense_readout_training_transition_window_benchmark",
        "surface": "bounded_snn_dense_readout_training_transition_window_benchmark.v1",
        "payload_count": int(payload_count),
        "index_count": int(index_count),
        "window_limit": SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT,
        "index_limit": SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_transition_order",
            "bounded_transition_window_before_index_canonicalization",
            "bounded_pre_post_index_windows_before_dense_cell_update",
            "untruncated_windows_required_before_checkpoint_mutation",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "oversized_payload_mutates_runtime_state": False,
            "oversized_payload_writes_checkpoint": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "active_training_device": "cpu",
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
            "oversized_transition_block": _stats(rows, "payload_block"),
            "oversized_index_block": _stats(rows, "index_block"),
            "exact_window_accept": _stats(rows, "exact"),
        },
        "last_run": last,
        "retired_full_payload_projection": {
            "executable_path_retired": True,
            "projected_transition_record_count": int(payload_count),
            "bounded_transition_record_count": (
                SNN_LANGUAGE_DENSE_READOUT_TRAINING_TRANSITION_WINDOW_LIMIT
            ),
            "projected_index_count": int(index_count),
            "bounded_index_count": (
                SNN_LANGUAGE_DENSE_READOUT_TRAINING_INDEX_WINDOW_LIMIT
            ),
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark bounded checkpointed dense-readout training transition windows."
        )
    )
    parser.add_argument("--payload-count", type=int, default=2048)
    parser.add_argument("--index-count", type=int, default=2048)
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
        "pass={passed} transition_block={selected}/{source} "
        "index_block_truncated={index_truncated} exact_updates={updates} "
        "reduction={reduction:.6f}".format(
            passed=report["pass"],
            selected=report["last_run"]["payload_block"][
                "transition_source_window"
            ]["source_window_count"],
            source=report["last_run"]["payload_block"][
                "transition_source_window"
            ]["source_total_count"],
            index_truncated=report["last_run"]["index_block"][
                "index_source_window"
            ]["source_payload_truncated"],
            updates=report["last_run"]["exact_window"]["updated_cell_count"],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
