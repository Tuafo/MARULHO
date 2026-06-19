from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc
from threading import RLock
from typing import Any

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
    SNNLanguagePlasticityApplicationExecutor,
)


def _synapses(count: int, *, initial_weight: float | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(count)):
        pre_index = index % 63
        post_index = pre_index + 1
        item: dict[str, Any] = {
            "pre_index": pre_index,
            "post_index": post_index,
            "locality_distance": 1,
        }
        if initial_weight is not None:
            item["initial_weight"] = float(initial_weight)
            item["source_synapse_id"] = f"benchmark-local:{pre_index}:{post_index}:{index}"
            item["source_trace_index"] = int(index)
            item["source_rollout_step_index"] = int(index)
            item["target_rollout_step_index"] = int(index + 1)
            item["source_active_indices_hash"] = f"source-active-{index}"
            item["target_active_indices_hash"] = f"target-active-{index}"
        rows.append(item)
    return rows


def _live_readiness() -> dict[str, Any]:
    return {
        "available": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
        "promotion_gate": {"status": "ready_for_operator_review"},
        "rollback_readiness": {
            "checkpoint_available": True,
            "restore_endpoint_available": True,
        },
        "operator_approval": {"approved": True},
    }


def _shadow_delta(synapses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "available": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
        "max_abs_weight_delta": 0.01,
        "pressure_before": 0.9,
        "pressure_after": 0.8,
        "bounded_synapses": synapses,
    }


def _regeneration_proposal(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "available": True,
        "owned_by_marulho": True,
        "generates_text": False,
        "loads_external_checkpoint": False,
        "replay_evidence": {
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "source": "benchmark.regeneration_permit",
            "permit_id": "permit-benchmark",
            "replay_window_id": "replay-window-benchmark",
            "replay_artifact_id": "artifact-benchmark",
            "replay_artifact_hash": "artifact-hash-benchmark",
            "replay_window_hash": "window-hash-benchmark",
            "readout_evidence_hashes": ["readout-hash-benchmark"],
            "source_metadata_hash": "source-metadata-hash-benchmark",
            "emission_lineage": {"emission_hash": "emission-hash-benchmark"},
            "evidence_hash": "sha256:replay-window-benchmark",
        },
        "promotion_gate": {"status": "ready_for_operator_review"},
        "regeneration_design": {
            "locality_radius": 2,
            "mismatch_score": 0.9,
            "candidate_synapses": candidates,
        },
    }


def _executor(tmp_dir: Path) -> tuple[
    SNNLanguagePlasticityApplicationExecutor,
    RuntimeState,
    dict[str, Any],
    list[str | None],
]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    language_state: dict[str, Any] = {"sparse_transition_weights": {}}
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
        verify_regeneration_permit=lambda proposal: True,
        publish_committed_checkpoint=lambda path, operation: {
            "checkpoint_path": str(path),
            "operation": operation,
        },
    )
    return executor, runtime_state, language_state, checkpoint_calls


def _case(*, payload_count: int, tmp_dir: Path, case_index: int) -> dict[str, Any]:
    case_dir = tmp_dir / f"case-{case_index}"
    case_dir.mkdir(parents=True, exist_ok=True)
    oversized_live_payload = _synapses(payload_count)
    oversized_regeneration_payload = _synapses(payload_count, initial_weight=0.01)
    exact_live_payload = _synapses(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)
    exact_regeneration_payload = _synapses(
        SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        initial_weight=0.01,
    )

    live_executor, live_state, live_language_state, live_checkpoint_calls = _executor(
        case_dir / "live-block"
    )
    live_started = time.perf_counter()
    live_block = live_executor.apply_live_application(
        live_application_readiness=_live_readiness(),
        shadow_delta=_shadow_delta(oversized_live_payload),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
    )
    live_block_elapsed_ms = (time.perf_counter() - live_started) * 1000.0

    regeneration_executor, regeneration_state, regeneration_language_state, regeneration_checkpoint_calls = _executor(
        case_dir / "regeneration-block"
    )
    regeneration_started = time.perf_counter()
    regeneration_block = regeneration_executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(oversized_regeneration_payload),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
    )
    regeneration_block_elapsed_ms = (time.perf_counter() - regeneration_started) * 1000.0

    live_accept_executor, live_accept_state, live_accept_language_state, live_accept_checkpoint_calls = _executor(
        case_dir / "live-accept"
    )
    live_accept_started = time.perf_counter()
    live_accept = live_accept_executor.apply_live_application(
        live_application_readiness=_live_readiness(),
        shadow_delta=_shadow_delta(exact_live_payload),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
    )
    live_accept_elapsed_ms = (time.perf_counter() - live_accept_started) * 1000.0

    regeneration_accept_executor, regeneration_accept_state, regeneration_accept_language_state, regeneration_accept_checkpoint_calls = _executor(
        case_dir / "regeneration-accept"
    )
    regeneration_accept_started = time.perf_counter()
    regeneration_accept = regeneration_accept_executor.regenerate_transition_memory(
        regeneration_proposal=_regeneration_proposal(exact_regeneration_payload),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
    )
    regeneration_accept_elapsed_ms = (
        time.perf_counter() - regeneration_accept_started
    ) * 1000.0

    live_evidence = dict(live_block.get("promotion_gate", {}).get("required_evidence") or {})
    regeneration_evidence = dict(
        regeneration_block.get("promotion_gate", {}).get("required_evidence") or {}
    )
    return {
        "live_block_elapsed_ms": float(live_block_elapsed_ms),
        "regeneration_block_elapsed_ms": float(regeneration_block_elapsed_ms),
        "live_accept_elapsed_ms": float(live_accept_elapsed_ms),
        "regeneration_accept_elapsed_ms": float(regeneration_accept_elapsed_ms),
        "live_block": {
            "accepted": bool(live_block.get("accepted")),
            "reason": live_block.get("reason"),
            "synapse_source_window": dict(live_evidence.get("synapse_source_window") or {}),
            "checkpoint_call_count": len(live_checkpoint_calls),
            "state_revision": int(live_state.state_revision),
            "weight_count": len(live_language_state.get("sparse_transition_weights") or {}),
            "payload_not_truncated": live_evidence.get("synapse_payload_not_truncated"),
        },
        "regeneration_block": {
            "accepted": bool(regeneration_block.get("accepted")),
            "reason": regeneration_block.get("reason"),
            "candidate_source_window": dict(
                regeneration_evidence.get("candidate_source_window") or {}
            ),
            "checkpoint_call_count": len(regeneration_checkpoint_calls),
            "state_revision": int(regeneration_state.state_revision),
            "weight_count": len(
                regeneration_language_state.get("sparse_transition_weights") or {}
            ),
            "payload_not_truncated": regeneration_evidence.get(
                "candidate_payload_not_truncated"
            ),
        },
        "live_accept": {
            "accepted": bool(live_accept.get("accepted")),
            "synapse_source_window": dict(live_accept.get("synapse_source_window") or {}),
            "checkpoint_call_count": len(live_accept_checkpoint_calls),
            "state_revision": int(live_accept_state.state_revision),
            "weight_count": len(
                live_accept_language_state.get("sparse_transition_weights") or {}
            ),
            "applied_synapse_count": int(
                live_accept.get("application_target", {}).get("applied_synapse_count", 0)
                or 0
            ),
        },
        "regeneration_accept": {
            "accepted": bool(regeneration_accept.get("accepted")),
            "candidate_source_window": dict(
                regeneration_accept.get("candidate_source_window") or {}
            ),
            "checkpoint_call_count": len(regeneration_accept_checkpoint_calls),
            "state_revision": int(regeneration_accept_state.state_revision),
            "weight_count": len(
                regeneration_accept_language_state.get("sparse_transition_weights")
                or {}
            ),
            "regenerated_synapse_count": int(
                regeneration_accept.get("regeneration", {}).get(
                    "regenerated_synapse_count",
                    0,
                )
                or 0
            ),
        },
    }


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    elapsed = [float(row[f"{key}_elapsed_ms"]) for row in rows]
    return {
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
    }


def run_benchmark(*, payload_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    tracemalloc.start()
    with tempfile.TemporaryDirectory(prefix="marulho-application-window-") as raw_tmp:
        tmp_dir = Path(raw_tmp)
        rows = [
            _case(payload_count=int(payload_count), tmp_dir=tmp_dir, case_index=index)
            for index in range(int(runs))
        ]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0

    last = rows[-1] if rows else {}
    live_block = dict(last.get("live_block") or {})
    regeneration_block = dict(last.get("regeneration_block") or {})
    live_window = dict(live_block.get("synapse_source_window") or {})
    regeneration_window = dict(regeneration_block.get("candidate_source_window") or {})
    live_accept = dict(last.get("live_accept") or {})
    regeneration_accept = dict(last.get("regeneration_accept") or {})
    live_accept_window = dict(live_accept.get("synapse_source_window") or {})
    regeneration_accept_window = dict(
        regeneration_accept.get("candidate_source_window") or {}
    )
    reduction = float(
        int(payload_count) / max(1, SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)
    )
    quality = {
        "live_oversized_blocked": bool(
            live_block.get("accepted") is False
            and live_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and live_window.get("source_total_count") == int(payload_count)
            and live_window.get("source_payload_truncated") is True
            and live_block.get("checkpoint_call_count") == 0
            and live_block.get("state_revision") == 0
            and live_block.get("weight_count") == 0
        ),
        "regeneration_oversized_blocked": bool(
            regeneration_block.get("accepted") is False
            and regeneration_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and regeneration_window.get("source_total_count") == int(payload_count)
            and regeneration_window.get("source_payload_truncated") is True
            and regeneration_block.get("checkpoint_call_count") == 0
            and regeneration_block.get("state_revision") == 0
            and regeneration_block.get("weight_count") == 0
        ),
        "live_exact_window_still_applies": bool(
            live_accept.get("accepted") is True
            and live_accept_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and live_accept_window.get("source_payload_truncated") is False
            and live_accept.get("checkpoint_call_count") == 2
            and live_accept.get("state_revision") == 1
            and live_accept.get("applied_synapse_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
        ),
        "regeneration_exact_window_still_applies": bool(
            regeneration_accept.get("accepted") is True
            and regeneration_accept_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and regeneration_accept_window.get("source_payload_truncated") is False
            and regeneration_accept.get("checkpoint_call_count") == 2
            and regeneration_accept.get("state_revision") == 1
            and regeneration_accept.get("regenerated_synapse_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
        ),
        "no_global_scans": bool(
            live_window.get("global_candidate_scan") is False
            and live_window.get("global_score_scan") is False
            and regeneration_window.get("global_candidate_scan") is False
            and regeneration_window.get("global_score_scan") is False
        ),
        "record_work_reduction": reduction,
        "old_full_payload_path_executable": False,
        "old_full_payload_work_projected_from_source_count": True,
    }
    quality["pass"] = bool(
        quality["live_oversized_blocked"]
        and quality["regeneration_oversized_blocked"]
        and quality["live_exact_window_still_applies"]
        and quality["regeneration_exact_window_still_applies"]
        and quality["no_global_scans"]
        and quality["record_work_reduction"] >= 2.0
    )
    return {
        "artifact_kind": "bounded_snn_language_application_synapse_window_benchmark",
        "surface": "bounded_snn_language_application_synapse_window_benchmark.v1",
        "payload_count": int(payload_count),
        "window_limit": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_synapse_order",
            "bounded_source_window_before_topology_validation",
            "untruncated_window_required_before_checkpoint_mutation",
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
            "active_application_device": "cpu",
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
            "live_oversized_block": _stats(rows, "live_block"),
            "regeneration_oversized_block": _stats(rows, "regeneration_block"),
            "live_exact_window_accept": _stats(rows, "live_accept"),
            "regeneration_exact_window_accept": _stats(rows, "regeneration_accept"),
        },
        "last_run": last,
        "retired_full_payload_projection": {
            "executable_path_retired": True,
            "projected_record_count": int(payload_count),
            "bounded_record_count": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded checkpointed SNN language application synapse windows."
    )
    parser.add_argument("--payload-count", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_benchmark(payload_count=args.payload_count, runs=args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "pass={passed} live_block={live_selected}/{live_source} "
        "regeneration_block={regeneration_selected}/{regeneration_source} "
        "reduction={reduction:.6f}".format(
            passed=report["pass"],
            live_selected=report["last_run"]["live_block"]["synapse_source_window"][
                "source_window_count"
            ],
            live_source=report["last_run"]["live_block"]["synapse_source_window"][
                "source_total_count"
            ],
            regeneration_selected=report["last_run"]["regeneration_block"][
                "candidate_source_window"
            ]["source_window_count"],
            regeneration_source=report["last_run"]["regeneration_block"][
                "candidate_source_window"
            ]["source_total_count"],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
