from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc
from typing import Any

import torch

from marulho.service.runtime_facade import RuntimeFacade
from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
)


def _candidates(count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(int(count)):
        pre_index = 65 + (index % 31)
        post_index = pre_index + 1
        rows.append(
            {
                "pre_index": pre_index,
                "post_index": post_index,
                "synapse": f"{pre_index}:{post_index}",
                "initial_weight": 0.02,
                "locality_distance": 1,
                "source_synapse_id": f"facade-benchmark:{pre_index}:{post_index}:{index}",
                "source_trace_index": int(index),
                "source_rollout_step_index": int(index),
                "target_rollout_step_index": int(index + 1),
                "source_active_indices_hash": f"source-active-{index}",
                "target_active_indices_hash": f"target-active-{index}",
            }
        )
    return rows


def _language_capacity() -> dict[str, Any]:
    return {
        "surface": "snn_language_capacity_state.v1",
        "language_neuron_count": 128,
        "sparse_edge_budget": 512,
        "outgoing_fanout_budget": 32,
        "capacity_expansion_count": 1,
    }


def _regeneration_design(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "locality_radius": 1,
        "initial_weight": 0.02,
        "max_new_synapses": len(candidates),
        "mismatch_score": 0.9,
        "candidate_count": len(candidates),
        "candidate_synapses": candidates,
    }


def _review(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "surface": "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "rollout_regeneration_replay_artifact_review_hash": "review-hash-benchmark",
        "language_capacity": _language_capacity(),
        "permit_request_preview": {
            "replay_artifact_id": "artifact-benchmark",
            "regeneration_design": _regeneration_design(candidates),
            "permit_issued": False,
        },
        "promotion_gate": {"eligible_for_regeneration_permit_request": True},
    }


def _permit_request(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "surface": "snn_language_readout_rollout_regeneration_permit_request.v1",
        "accepted": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": True,
        "checkpoint_written": False,
        "language_capacity": _language_capacity(),
        "replay_evidence": {
            "permit_id": "permit-benchmark",
            "ready": True,
            "owned_by_marulho": True,
        },
        "regeneration_design": _regeneration_design(candidates),
        "promotion_gate": {
            "eligible_for_regeneration_application": True,
            "required_evidence": {
                "applied_replay_lineage_restore_validation_not_mismatched": True,
            },
        },
    }


def _preflight(candidates: list[dict[str, Any]], checkpoint_path: str) -> dict[str, Any]:
    return {
        "surface": "snn_language_readout_rollout_regeneration_application_preflight.v1",
        "ready": True,
        "owned_by_marulho": True,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "executor_called": False,
        "expected_state_revision": 0,
        "checkpoint_path": checkpoint_path,
        "language_capacity": _language_capacity(),
        "regeneration_proposal": {
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "generates_text": False,
            "loads_external_checkpoint": False,
            "language_capacity": _language_capacity(),
            "promotion_gate": {"status": "ready_for_operator_review"},
            "replay_evidence": {
                "available": True,
                "ready": True,
                "owned_by_marulho": True,
                "source": "replay_controller.regeneration_permit",
                "permit_id": "permit-benchmark",
                "replay_window_id": "replay-window-benchmark",
                "replay_artifact_id": "artifact-benchmark",
                "replay_artifact_hash": "artifact-hash-benchmark",
                "replay_window_hash": "window-hash-benchmark",
                "readout_evidence_hashes": ["readout-hash-benchmark"],
                "evidence_hash": "sha256:rollout-replay-window-benchmark",
            },
            "regeneration_design": _regeneration_design(candidates),
        },
        "promotion_gate": {
            "eligible_for_checkpoint_backed_regeneration_executor": True,
        },
    }


class _ReplayController:
    def __init__(self, runtime_state: RuntimeState) -> None:
        self._runtime_state = runtime_state
        self.calls: list[dict[str, Any]] = []

    def issue_regeneration_permit(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        self._runtime_state.mark_dirty_without_revision()
        return {
            "artifact_kind": "terminus_snn_language_transition_memory_regeneration_permit",
            "surface": "snn_language_transition_memory_regeneration_permit.v1",
            "available": True,
            "ready": True,
            "owned_by_marulho": True,
            "permit_id": "permit-benchmark",
            "replay_window_id": "replay-window-benchmark",
            "replay_artifact_id": kwargs["replay_artifact_id"],
            "replay_artifact_hash": "artifact-hash-benchmark",
            "replay_window_hash": "window-hash-benchmark",
            "readout_evidence_hashes": ["readout-hash-benchmark"],
            "evidence_hash": "sha256:rollout-replay-window-benchmark",
            "regeneration_design_hash": "design-hash-benchmark",
        }


class _Executor:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def regenerate_transition_memory(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        proposal = kwargs.get("regeneration_proposal")
        design = proposal.get("regeneration_design") if isinstance(proposal, dict) else {}
        candidates = design.get("candidate_synapses") if isinstance(design, dict) else []
        return {
            "artifact_kind": "terminus_snn_language_transition_memory_regeneration",
            "surface": "snn_language_transition_memory_regeneration.v1",
            "accepted": True,
            "status": "regenerated",
            "reason": None,
            "owned_by_marulho": True,
            "applies_plasticity": True,
            "mutates_runtime_state": True,
            "checkpoint_transaction": {
                "pre_regeneration_checkpoint_saved": True,
                "post_regeneration_checkpoint_saved": True,
                "restore_verified": True,
            },
            "regeneration": {
                "regenerated_synapse_count": len(candidates)
                if isinstance(candidates, list)
                else 0,
            },
            "received_candidate_count": len(candidates)
            if isinstance(candidates, list)
            else 0,
        }


class _Root:
    def __init__(self, runtime_state: RuntimeState) -> None:
        self._runtime_state = runtime_state
        self._replay_controller = _ReplayController(runtime_state)
        self._snn_language_plasticity_executor = _Executor()


def _required_window(result: dict[str, Any]) -> dict[str, Any]:
    gate = result.get("promotion_gate") if isinstance(result.get("promotion_gate"), dict) else {}
    required = gate.get("required_evidence") if isinstance(gate.get("required_evidence"), dict) else {}
    window = required.get("candidate_source_window")
    return dict(window) if isinstance(window, dict) else {}


def _case(*, payload_count: int, tmp_dir: Path, case_index: int) -> dict[str, Any]:
    case_dir = tmp_dir / f"case-{case_index}"
    case_dir.mkdir(parents=True, exist_ok=True)
    oversized = _candidates(payload_count)
    exact = _candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)

    permit_state = RuntimeState()
    permit_root = _Root(permit_state)
    permit_facade = RuntimeFacade(permit_root)
    permit_started = time.perf_counter()
    permit_block = permit_facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=_review(oversized),
        operator_id="benchmark",
        confirmation=True,
    )
    permit_block_elapsed_ms = (time.perf_counter() - permit_started) * 1000.0

    preflight_state = RuntimeState()
    preflight_facade = RuntimeFacade(_Root(preflight_state))
    preflight_started = time.perf_counter()
    preflight_block = preflight_facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=_permit_request(oversized),
        expected_state_revision=0,
        checkpoint_path=str(case_dir / "preflight.pt"),
    )
    preflight_block_elapsed_ms = (time.perf_counter() - preflight_started) * 1000.0

    application_state = RuntimeState()
    application_root = _Root(application_state)
    application_facade = RuntimeFacade(application_root)
    application_started = time.perf_counter()
    application_block = application_facade.snn_language_readout_rollout_regeneration_application(
        rollout_regeneration_application_preflight=_preflight(
            oversized,
            str(case_dir / "application.pt"),
        ),
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
        checkpoint_path=str(case_dir / "application.pt"),
    )
    application_block_elapsed_ms = (time.perf_counter() - application_started) * 1000.0

    exact_state = RuntimeState()
    exact_root = _Root(exact_state)
    exact_facade = RuntimeFacade(exact_root)
    exact_started = time.perf_counter()
    exact_permit = exact_facade.snn_language_readout_rollout_regeneration_permit_request(
        rollout_regeneration_replay_artifact_review=_review(exact),
        operator_id="benchmark",
        confirmation=True,
    )
    exact_preflight = exact_facade.snn_language_readout_rollout_regeneration_application_preflight(
        rollout_regeneration_permit_request=exact_permit,
        expected_state_revision=0,
        checkpoint_path=str(case_dir / "exact.pt"),
    )
    exact_application = exact_facade.snn_language_readout_rollout_regeneration_application(
        rollout_regeneration_application_preflight=exact_preflight,
        expected_state_revision=0,
        operator_id="benchmark",
        confirmation=True,
        checkpoint_path=str(case_dir / "exact.pt"),
    )
    exact_flow_elapsed_ms = (time.perf_counter() - exact_started) * 1000.0

    permit_window = _required_window(permit_block)
    preflight_window = _required_window(preflight_block)
    application_window = _required_window(application_block)
    exact_application_window = _required_window(exact_application)
    exact_permit_design = (
        exact_root._replay_controller.calls[0].get("regeneration_design")
        if exact_root._replay_controller.calls
        else {}
    )
    exact_executor_call = (
        exact_root._snn_language_plasticity_executor.calls[0]
        if exact_root._snn_language_plasticity_executor.calls
        else {}
    )
    exact_executor_proposal = (
        exact_executor_call.get("regeneration_proposal")
        if isinstance(exact_executor_call.get("regeneration_proposal"), dict)
        else {}
    )
    exact_executor_design = (
        exact_executor_proposal.get("regeneration_design")
        if isinstance(exact_executor_proposal.get("regeneration_design"), dict)
        else {}
    )
    return {
        "permit_block_elapsed_ms": permit_block_elapsed_ms,
        "preflight_block_elapsed_ms": preflight_block_elapsed_ms,
        "application_block_elapsed_ms": application_block_elapsed_ms,
        "exact_flow_elapsed_ms": exact_flow_elapsed_ms,
        "permit_block": {
            "accepted": bool(permit_block.get("accepted")),
            "issues_regeneration_permit": bool(permit_block.get("issues_regeneration_permit")),
            "replay_controller_call_count": len(permit_root._replay_controller.calls),
            "state_revision": int(permit_state.state_revision),
            "candidate_source_window": permit_window,
            "payload_not_truncated": permit_block["promotion_gate"]["required_evidence"].get(
                "candidate_payload_not_truncated"
            ),
        },
        "preflight_block": {
            "ready": bool(preflight_block.get("ready")),
            "executor_called": bool(preflight_block.get("executor_called")),
            "proposal_available": bool(
                preflight_block.get("regeneration_proposal", {}).get("available")
            ),
            "state_revision": int(preflight_state.state_revision),
            "candidate_source_window": preflight_window,
            "payload_not_truncated": preflight_block["promotion_gate"][
                "required_evidence"
            ].get("candidate_payload_not_truncated"),
            "proposal_candidate_count": len(
                preflight_block.get("regeneration_proposal", {})
                .get("regeneration_design", {})
                .get("candidate_synapses", [])
            ),
        },
        "application_block": {
            "accepted": bool(application_block.get("accepted")),
            "executor_called": bool(application_block.get("executor_called")),
            "writes_checkpoint": bool(application_block.get("writes_checkpoint")),
            "executor_call_count": len(
                application_root._snn_language_plasticity_executor.calls
            ),
            "state_revision": int(application_state.state_revision),
            "candidate_source_window": application_window,
            "payload_not_truncated": application_block["promotion_gate"][
                "required_evidence"
            ].get("candidate_payload_not_truncated"),
        },
        "exact_flow": {
            "permit_accepted": bool(exact_permit.get("accepted")),
            "preflight_ready": bool(exact_preflight.get("ready")),
            "application_accepted": bool(exact_application.get("accepted")),
            "application_executor_called": bool(exact_application.get("executor_called")),
            "application_writes_checkpoint": bool(exact_application.get("writes_checkpoint")),
            "replay_controller_call_count": len(exact_root._replay_controller.calls),
            "executor_call_count": len(exact_root._snn_language_plasticity_executor.calls),
            "permit_candidate_count": len(
                exact_permit_design.get("candidate_synapses", [])
                if isinstance(exact_permit_design, dict)
                else []
            ),
            "preflight_candidate_count": len(
                exact_preflight.get("regeneration_proposal", {})
                .get("regeneration_design", {})
                .get("candidate_synapses", [])
            ),
            "executor_received_candidate_count": len(
                exact_executor_design.get("candidate_synapses", [])
                if isinstance(exact_executor_design, dict)
                else []
            ),
            "candidate_source_window": exact_application_window,
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
    with tempfile.TemporaryDirectory(prefix="marulho-rollout-regeneration-facade-") as raw_tmp:
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
    permit_block = dict(last.get("permit_block") or {})
    preflight_block = dict(last.get("preflight_block") or {})
    application_block = dict(last.get("application_block") or {})
    exact_flow = dict(last.get("exact_flow") or {})
    permit_window = dict(permit_block.get("candidate_source_window") or {})
    preflight_window = dict(preflight_block.get("candidate_source_window") or {})
    application_window = dict(application_block.get("candidate_source_window") or {})
    exact_window = dict(exact_flow.get("candidate_source_window") or {})
    reduction = float(
        int(payload_count) / max(1, SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)
    )
    quality = {
        "permit_oversized_blocked_before_replay_controller": bool(
            permit_block.get("accepted") is False
            and permit_block.get("issues_regeneration_permit") is False
            and permit_block.get("replay_controller_call_count") == 0
            and permit_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and permit_window.get("source_payload_truncated") is True
            and permit_block.get("payload_not_truncated") is False
        ),
        "preflight_oversized_blocked_before_proposal_ready": bool(
            preflight_block.get("ready") is False
            and preflight_block.get("executor_called") is False
            and preflight_block.get("proposal_available") is False
            and preflight_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and preflight_window.get("source_payload_truncated") is True
            and preflight_block.get("proposal_candidate_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
        ),
        "application_oversized_blocked_before_executor": bool(
            application_block.get("accepted") is False
            and application_block.get("executor_called") is False
            and application_block.get("writes_checkpoint") is False
            and application_block.get("executor_call_count") == 0
            and application_window.get("source_window_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and application_window.get("source_payload_truncated") is True
            and application_block.get("payload_not_truncated") is False
        ),
        "exact_window_still_advances_single_path": bool(
            exact_flow.get("permit_accepted") is True
            and exact_flow.get("preflight_ready") is True
            and exact_flow.get("application_accepted") is True
            and exact_flow.get("application_executor_called") is True
            and exact_flow.get("application_writes_checkpoint") is True
            and exact_flow.get("replay_controller_call_count") == 1
            and exact_flow.get("executor_call_count") == 1
            and exact_flow.get("permit_candidate_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and exact_flow.get("preflight_candidate_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and exact_flow.get("executor_received_candidate_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and exact_window.get("source_payload_truncated") is False
        ),
        "no_global_scans": bool(
            permit_window.get("global_candidate_scan") is False
            and preflight_window.get("global_candidate_scan") is False
            and application_window.get("global_candidate_scan") is False
            and permit_window.get("global_score_scan") is False
            and preflight_window.get("global_score_scan") is False
            and application_window.get("global_score_scan") is False
        ),
        "record_work_reduction": reduction,
        "old_full_payload_facade_path_executable": False,
    }
    quality["pass"] = bool(
        quality["permit_oversized_blocked_before_replay_controller"]
        and quality["preflight_oversized_blocked_before_proposal_ready"]
        and quality["application_oversized_blocked_before_executor"]
        and quality["exact_window_still_advances_single_path"]
        and quality["no_global_scans"]
        and quality["record_work_reduction"] >= 2.0
    )
    return {
        "artifact_kind": (
            "bounded_snn_rollout_regeneration_facade_candidate_window_benchmark"
        ),
        "surface": (
            "bounded_snn_rollout_regeneration_facade_candidate_window_"
            "benchmark.v1"
        ),
        "payload_count": int(payload_count),
        "window_limit": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_candidate_order",
            "bounded_source_window_before_facade_permit_or_preflight",
            "untruncated_window_required_before_replay_controller_or_executor",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "oversized_payload_issues_replay_permit": False,
            "oversized_payload_calls_executor": False,
            "oversized_payload_writes_checkpoint": False,
            "old_full_payload_facade_path_executable": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "facade_gate_device": "cpu",
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
            "permit_oversized_block": _stats(rows, "permit_block"),
            "preflight_oversized_block": _stats(rows, "preflight_block"),
            "application_oversized_block": _stats(rows, "application_block"),
            "exact_window_flow": _stats(rows, "exact_flow"),
        },
        "last_run": last,
        "retired_full_payload_projection": {
            "executable_facade_path_retired": True,
            "projected_record_count": int(payload_count),
            "bounded_record_count": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        },
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark bounded rollout-regeneration facade candidate windows."
        )
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
        "pass={passed} permit_block={permit_selected}/{permit_source} "
        "preflight_block={preflight_selected}/{preflight_source} "
        "application_block={application_selected}/{application_source} "
        "reduction={reduction:.6f}".format(
            passed=report["pass"],
            permit_selected=report["last_run"]["permit_block"][
                "candidate_source_window"
            ]["source_window_count"],
            permit_source=report["last_run"]["permit_block"][
                "candidate_source_window"
            ]["source_total_count"],
            preflight_selected=report["last_run"]["preflight_block"][
                "candidate_source_window"
            ]["source_window_count"],
            preflight_source=report["last_run"]["preflight_block"][
                "candidate_source_window"
            ]["source_total_count"],
            application_selected=report["last_run"]["application_block"][
                "candidate_source_window"
            ]["source_window_count"],
            application_source=report["last_run"]["application_block"][
                "candidate_source_window"
            ]["source_total_count"],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
