from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
    SNNLanguageReadoutEvidenceLedger,
)


def _ready_draft_for(
    prediction_hash: str,
    evaluation_hash: str,
    weights_hash: str,
    labels: list[str],
) -> dict[str, Any]:
    return {
        "surface": "snn_language_readout_draft.v1",
        "generation_scope": "bounded_grounded_readout_label_draft",
        "freeform_language_generation": False,
        "mutates_runtime_state": False,
        "draft": {"labels": labels, "text": " ".join(labels)},
        "sparse_decode_evidence": {
            "candidate_matches": [
                {"label": label, "grounded": True}
                for label in labels
            ]
        },
        "transition_memory_evaluation_evidence": {
            "provenance_match": True,
            "prediction_hash": prediction_hash,
            "transition_memory_evaluation_hash": evaluation_hash,
            "persistent_transition_weights_hash": weights_hash,
        },
        "promotion_gate": {
            "eligible_for_bounded_readout_generation": True,
            "eligible_for_cognition_substrate": False,
        },
    }


def _repeat_to_count(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if not items or count <= 0:
        return []
    return [dict(items[index % len(items)]) for index in range(int(count))]


def _case(*, payload_count: int) -> dict[str, Any]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, Any] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            "prediction-target-window",
            "evaluation-target-window",
            "weights-target-window",
            ["memory pressure", "prediction error", "transition support"],
        ),
        expected_state_revision=runtime_state.state_revision,
        operator_id="benchmark",
        confirmation=True,
    )
    evaluation = ledger.rehearsal_evaluation(
        ledger.replay_priority(limit=1),
        candidate_limit=1,
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    experiment = ledger.rehearsal_experiment(evaluation, replay_cycles=6)
    design = ledger.replay_design(
        experiment,
        replay_policy={
            "max_candidates": 1,
            "max_replay_cycles": 6,
            "min_pressure_gain": 0.01,
        },
        rollback_policy={"available": True, "snapshot_id": "snapshot-benchmark"},
    )
    oversized_design = dict(design)
    oversized_design["selected_replay_targets"] = _repeat_to_count(
        list(design.get("selected_replay_targets") or []),
        int(payload_count),
    )
    before_replay_snapshot = runtime_state.snapshot()
    before_replay_revision = int(runtime_state.state_revision)
    dry_started = time.perf_counter()
    dry_run = ledger.replay_dry_run(
        oversized_design,
        operator_approval=True,
        operator_id="benchmark",
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    dry_elapsed_ms = (time.perf_counter() - dry_started) * 1000.0
    preflight = ledger.plasticity_preflight(
        dry_run,
        plasticity_policy={"locality_radius": 8},
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-benchmark"},
    )
    oversized_preflight = dict(preflight)
    oversized_preflight["candidate_replay_sequences"] = _repeat_to_count(
        list(preflight.get("candidate_replay_sequences") or []),
        int(payload_count),
    )
    bridge_started = time.perf_counter()
    bridge = ledger.plasticity_replay_bridge(
        oversized_preflight,
        runtime_truth_delta={"improved_or_stable": True},
        rollback_policy={"available": True, "snapshot_id": "snapshot-benchmark"},
    )
    bridge_elapsed_ms = (time.perf_counter() - bridge_started) * 1000.0
    after_replay_snapshot = runtime_state.snapshot()
    return {
        "dry_run_elapsed_ms": float(dry_elapsed_ms),
        "bridge_elapsed_ms": float(bridge_elapsed_ms),
        "dry_run_target_count": int(dry_run["isolated_replay_summary"]["target_count"]),
        "dry_run_source_count": int(dry_run["replay_target_source_count"] or 0),
        "dry_run_truncated_count": int(
            dry_run["replay_target_window"]["source_truncated_count"] or 0
        ),
        "dry_run_window": dry_run["replay_target_window"],
        "dry_run_gate": dry_run["promotion_gate"]["required_evidence"],
        "bridge_sequence_count": int(
            bridge["replay_experiment"]["replay_sequence_count"]
        ),
        "bridge_source_count": int(bridge["replay_sequence_source_count"] or 0),
        "bridge_truncated_count": int(
            bridge["replay_sequence_window"]["source_truncated_count"] or 0
        ),
        "bridge_window": bridge["replay_sequence_window"],
        "bridge_gate": bridge["promotion_gate"]["required_evidence"],
        "replay_runtime_state_unchanged": bool(
            before_replay_snapshot == after_replay_snapshot
        ),
        "runtime_state_revision_before_replay": int(before_replay_revision),
        "runtime_state_revision": int(runtime_state.state_revision),
        "dirty_state": bool(runtime_state.dirty_state),
    }


def _stats(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected_name = "target_count" if prefix == "dry_run" else "sequence_count"
    elapsed = [float(row[f"{prefix}_elapsed_ms"]) for row in rows]
    selected = [int(row[f"{prefix}_{selected_name}"]) for row in rows]
    source = [int(row[f"{prefix}_source_count"]) for row in rows]
    truncated = [int(row[f"{prefix}_truncated_count"]) for row in rows]
    return {
        "elapsed_mean_ms": float(statistics.fmean(elapsed)) if elapsed else 0.0,
        "elapsed_median_ms": float(statistics.median(elapsed)) if elapsed else 0.0,
        "selected_count_mean": float(statistics.fmean(selected)) if selected else 0.0,
        "source_count_mean": float(statistics.fmean(source)) if source else 0.0,
        "truncated_count_mean": float(statistics.fmean(truncated)) if truncated else 0.0,
    }


def run_benchmark(*, payload_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    rows = [_case(payload_count=payload_count) for _ in range(int(runs))]
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0
    dry = _stats(rows, "dry_run")
    bridge = _stats(rows, "bridge")
    dry_reduction = dry["source_count_mean"] / max(1.0, dry["selected_count_mean"])
    bridge_reduction = bridge["source_count_mean"] / max(
        1.0,
        bridge["selected_count_mean"],
    )
    last = rows[-1] if rows else {}
    dry_window = dict(last.get("dry_run_window") or {})
    bridge_window = dict(last.get("bridge_window") or {})
    quality = {
        "dry_run_window_bounded": bool(
            dry_window.get("source_window_count")
            == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
            and dry_window.get("source_total_count") == int(payload_count)
            and dry_window.get("global_candidate_scan") is False
            and dry_window.get("global_score_scan") is False
            and dry_window.get("language_reasoning") is False
        ),
        "bridge_window_bounded": bool(
            bridge_window.get("source_window_count")
            == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
            and bridge_window.get("source_total_count") == int(payload_count)
            and bridge_window.get("global_candidate_scan") is False
            and bridge_window.get("global_score_scan") is False
            and bridge_window.get("language_reasoning") is False
        ),
        "dry_run_work_reduction": float(dry_reduction),
        "bridge_work_reduction": float(bridge_reduction),
        "runtime_mutation_absent": bool(
            last.get("replay_runtime_state_unchanged") is True
            and int(last.get("runtime_state_revision", -1))
            == int(last.get("runtime_state_revision_before_replay", -2))
        ),
        "old_full_payload_path_executable": False,
        "old_full_payload_work_projected_from_source_count": True,
    }
    quality["pass"] = bool(
        quality["dry_run_window_bounded"]
        and quality["bridge_window_bounded"]
        and quality["dry_run_work_reduction"] >= 2.0
        and quality["bridge_work_reduction"] >= 2.0
        and quality["runtime_mutation_absent"]
    )
    return {
        "artifact_kind": "bounded_snn_readout_replay_target_window_benchmark",
        "surface": "bounded_snn_readout_replay_target_window_benchmark.v1",
        "payload_count": int(payload_count),
        "window_limit": SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_replay_payload_order",
            "bounded_source_window_before_tensor_materialization",
            "provenance_hash_revalidation_after_windowing",
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
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "active_replay_computation_device": "cpu",
            "cuda_available": cuda_available,
            "gpu_used": False,
            "cuda_memory_allocated_before_mib": float(cuda_before / (1024 * 1024)),
            "cuda_memory_allocated_after_mib": float(cuda_after / (1024 * 1024)),
            "cuda_memory_reserved_before_mib": float(
                cuda_reserved_before / (1024 * 1024)
            ),
            "cuda_memory_reserved_after_mib": float(
                cuda_reserved_after / (1024 * 1024)
            ),
        },
        "dry_run": dry,
        "bridge": bridge,
        "last_run": last,
        "retired_full_payload_projection": {
            "executable_path_retired": True,
            "dry_run_projected_target_count": int(payload_count),
            "bridge_projected_sequence_count": int(payload_count),
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded SNN readout replay target windows."
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
        "pass={passed} dry_run={dry:.0f}/{source:.0f} bridge={bridge:.0f}/{bridge_source:.0f} "
        "dry_reduction={dry_reduction:.6f} bridge_reduction={bridge_reduction:.6f}".format(
            passed=report["pass"],
            dry=report["dry_run"]["selected_count_mean"],
            source=report["dry_run"]["source_count_mean"],
            bridge=report["bridge"]["selected_count_mean"],
            bridge_source=report["bridge"]["source_count_mean"],
            dry_reduction=report["quality"]["dry_run_work_reduction"],
            bridge_reduction=report["quality"]["bridge_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
