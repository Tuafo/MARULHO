from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
from threading import RLock
import time
import tracemalloc
from typing import Any

import torch

from marulho.service.replay_runtime import ReplayController
from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_plasticity_executor import (
    SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
)
from marulho.service.snn_language_readout_ledger import (
    SNNLanguageReadoutEvidenceLedger,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _language_capacity(
    *,
    language_neuron_count: int = 64,
    sparse_edge_budget: int = 512,
    outgoing_fanout_budget: int = 16,
    capacity_expansion_count: int = 1,
) -> dict[str, object]:
    return {
        "surface": "snn_language_capacity_state.v1",
        "language_neuron_count": language_neuron_count,
        "sparse_edge_budget": sparse_edge_budget,
        "outgoing_fanout_budget": outgoing_fanout_budget,
        "capacity_expansion_count": capacity_expansion_count,
    }


def _ready_rollout_replay_evaluation() -> dict[str, object]:
    return {
        "surface": "snn_language_readout_rollout_replay_evaluation.v1",
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "freeform_language_generation": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "recorded_in_ledger": False,
        "eligible_for_replay_priority": False,
        "replay_evaluation": {
            "trace_step_count": 2,
            "replay_targets": [
                {
                    "step_index": 0,
                    "selected_label": "memory pressure",
                    "grounded": True,
                    "selection_score": 0.4,
                    "transition_support": 0.6,
                    "predicted_sparse_indices": [1, 2, 3],
                    "active_indices_hash": _sha256_json([1, 2, 3]),
                    "active_indices_hash_valid": True,
                },
                {
                    "step_index": 1,
                    "selected_label": "prediction error",
                    "grounded": True,
                    "selection_score": 0.3,
                    "transition_support": 0.5,
                    "predicted_sparse_indices": [2, 3, 4],
                    "active_indices_hash": _sha256_json([2, 3, 4]),
                    "active_indices_hash_valid": True,
                },
            ],
        },
        "provenance_evidence": {
            "rollout_replay_evaluation_hash": "rollout-eval-hash-benchmark",
            "rollout_hash": "rollout-hash-benchmark",
            "rollout_id": "snn-readout-rollout:rollout-hash-benchmark",
            "prediction_hash": "prediction-hash-benchmark",
            "current_sparse_code_hash": "current-sparse-code-hash-benchmark",
            "transition_memory_evaluation_hash": "evaluation-hash-benchmark",
            "persistent_transition_weights_hash": "weights-hash-benchmark",
            "server_transition_memory_hash": "weights-hash-benchmark",
            "server_transition_memory_hash_match": True,
            "transition_memory_state_source": (
                "service.runtime_facade.snn_language_plasticity_runtime_state"
            ),
        },
        "device_evidence": {
            "requested_device": "cpu",
            "tensor_device": "cpu",
            "cuda_tensor": False,
            "device_source": "benchmark",
        },
        "promotion_gate": {
            "eligible_for_readout_rollout_ledger_recording_review": True,
            "eligible_for_replay_priority": False,
        },
    }


def _sparse_transition_candidate(index: int) -> dict[str, object]:
    source_index = index % 32
    target_index = source_index + 1
    return {
        "source_index": source_index,
        "target_index": target_index,
        "source_trace_index": index,
        "source_step_index": index,
        "target_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([source_index]),
        "target_active_indices_hash": _sha256_json([target_index]),
    }


def _sparse_transition_candidates(count: int) -> list[dict[str, object]]:
    return [_sparse_transition_candidate(index) for index in range(int(count))]


def _design_candidate(index: int) -> dict[str, object]:
    source_index = index % 32
    target_index = source_index + 1
    return {
        "synapse_id": f"snn-rollout-local:{source_index}:{target_index}:{index}",
        "source_step_index": source_index,
        "target_step_index": target_index,
        "source_neuron_index": source_index,
        "target_neuron_index": target_index,
        "source_trace_index": index,
        "source_rollout_step_index": index,
        "target_rollout_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([source_index]),
        "target_active_indices_hash": _sha256_json([target_index]),
        "local_only": True,
        "proposed_weight_delta": 0.02,
        "homeostatic_decay": 0.0,
        "normalization": True,
        "applied_to_runtime": False,
    }


def _design_candidates(count: int) -> list[dict[str, object]]:
    return [_design_candidate(index) for index in range(int(count))]


def _growth_candidate(index: int) -> dict[str, object]:
    pre_index = index % 32
    post_index = pre_index + 1
    return {
        "synapse": f"{pre_index}:{post_index}",
        "pre_index": pre_index,
        "post_index": post_index,
        "initial_weight": 0.02,
        "locality_distance": abs(post_index - pre_index),
        "source_synapse_id": f"snn-rollout-local:{pre_index}:{post_index}:{index}",
        "source_trace_index": index,
        "source_rollout_step_index": index,
        "target_rollout_step_index": index + 1,
        "source_active_indices_hash": _sha256_json([pre_index]),
        "target_active_indices_hash": _sha256_json([post_index]),
        "local_only": True,
        "normalization": True,
        "applied_to_runtime": False,
    }


def _growth_candidates(count: int) -> list[dict[str, object]]:
    return [_growth_candidate(index) for index in range(int(count))]


def _replay_artifact(runtime_state: RuntimeState) -> dict[str, object]:
    material = {
        "recorded_state_revision": runtime_state.state_revision,
        "operator_id": "benchmark",
        "confirmation": True,
        "mismatch_hash": "mismatch-hash-benchmark",
        "mismatch_score": 0.9,
        "pressure_hash": "pressure-hash-benchmark",
        "pressure_score": 0.9,
        "replay_window_hash": "window-hash-benchmark",
        "replay_window_size": 1,
        "internal_ledger_backed": True,
        "artifact_proposal_hash": "proposal-hash-benchmark",
        "replay_evaluation_context_id": "context-benchmark",
        "replay_evaluation_context_hash": "context-hash-benchmark",
        "review_ticket_id": "ticket-benchmark",
        "review_ticket_hash": "ticket-hash-benchmark",
        "readout_evidence_hashes": ["readout-hash-benchmark"],
    }
    return {
        "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
        "surface": "snn_transition_memory_replay_artifact.v1",
        "available": True,
        "ready": True,
        "owned_by_marulho": True,
        "source": "replay_controller.snn_transition_memory_replay_artifact",
        "replay_artifact_id": "artifact-benchmark",
        "replay_window_id": "replay-window-benchmark",
        "evidence_hash": _sha256_json(material),
        **material,
    }


def _make_ledger() -> tuple[RuntimeState, SNNLanguageReadoutEvidenceLedger]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    ledger.record_readout_rollout_replay_evaluation(
        readout_rollout_replay_evaluation=_ready_rollout_replay_evaluation(),
        expected_state_revision=runtime_state.state_revision,
        operator_id="benchmark",
        confirmation=True,
    )
    return runtime_state, ledger


def _candidate_window(result: dict[str, Any], key: str) -> dict[str, Any]:
    window = result.get(key)
    return dict(window) if isinstance(window, dict) else {}


def _required(result: dict[str, Any]) -> dict[str, Any]:
    gate = result.get("promotion_gate") if isinstance(result.get("promotion_gate"), dict) else {}
    required = gate.get("required_evidence") if isinstance(gate.get("required_evidence"), dict) else {}
    return dict(required)


def _run_case(*, payload_count: int) -> dict[str, Any]:
    runtime_state, ledger = _make_ledger()
    policy = ledger.rollout_rehearsal_promotion_policy(candidate_limit=4)
    rehearsal = ledger.rollout_rehearsal_evaluation(policy, candidate_limit=4)
    experiment = ledger.rollout_rehearsal_experiment(rehearsal, replay_cycles=4)
    before_snapshot = runtime_state.snapshot()

    exact_experiment = deepcopy(experiment)
    exact_experiment["ephemeral_experiment"]["sparse_transition_candidates"] = (
        _sparse_transition_candidates(SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)
    )
    started = time.perf_counter()
    exact_design = ledger.rollout_consolidation_design(
        exact_experiment,
        rollback_policy={"available": True, "snapshot_id": "benchmark-snapshot"},
    )
    exact_design_ms = (time.perf_counter() - started) * 1000.0

    oversized_experiment = deepcopy(experiment)
    oversized_experiment["ephemeral_experiment"]["sparse_transition_candidates"] = (
        _sparse_transition_candidates(payload_count)
    )
    started = time.perf_counter()
    design_block = ledger.rollout_consolidation_design(
        oversized_experiment,
        rollback_policy={"available": True, "snapshot_id": "benchmark-snapshot"},
    )
    design_block_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    exact_shadow = ledger.rollout_consolidation_shadow_delta(
        exact_design,
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    exact_shadow_ms = (time.perf_counter() - started) * 1000.0

    oversized_design = deepcopy(exact_design)
    oversized_design["rollout_consolidation_design"]["candidate_synapses"] = _design_candidates(
        payload_count
    )
    oversized_design["rollout_consolidation_design"]["candidate_synapse_count"] = payload_count
    started = time.perf_counter()
    shadow_block = ledger.rollout_consolidation_shadow_delta(
        oversized_design,
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    shadow_block_ms = (time.perf_counter() - started) * 1000.0

    growth_memory = {
        "surface": "snn_language_plasticity_runtime_state.v1",
        "owned_by_marulho": True,
        "sparse_transition_weights": {},
        "language_capacity": _language_capacity(),
    }
    growth_preflight = ledger.rollout_consolidation_shadow_application_preflight(
        exact_design,
        exact_shadow,
        transition_memory_state=growth_memory,
    )
    started = time.perf_counter()
    exact_developmental = ledger.rollout_developmental_plasticity_review(
        exact_design,
        growth_preflight,
        transition_memory_state=growth_memory,
    )
    exact_developmental_ms = (time.perf_counter() - started) * 1000.0

    oversized_developmental_design = deepcopy(exact_design)
    oversized_developmental_design["rollout_consolidation_design"]["candidate_synapses"] = (
        _design_candidates(payload_count)
    )
    started = time.perf_counter()
    developmental_block = ledger.rollout_developmental_plasticity_review(
        oversized_developmental_design,
        growth_preflight,
        transition_memory_state=growth_memory,
    )
    developmental_block_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    exact_adapter = ledger.rollout_regeneration_proposal_adapter(exact_developmental)
    exact_adapter_ms = (time.perf_counter() - started) * 1000.0

    oversized_review = deepcopy(exact_developmental)
    oversized_review["developmental_plasticity_review"]["growth_candidate_count"] = payload_count
    oversized_review["developmental_plasticity_review"]["growth_candidates"] = _growth_candidates(
        payload_count
    )
    started = time.perf_counter()
    adapter_block = ledger.rollout_regeneration_proposal_adapter(oversized_review)
    adapter_block_ms = (time.perf_counter() - started) * 1000.0

    replay = _replay_artifact(runtime_state)
    started = time.perf_counter()
    exact_replay_review = ledger.rollout_regeneration_replay_artifact_review(
        exact_adapter,
        replay,
    )
    exact_replay_review_ms = (time.perf_counter() - started) * 1000.0

    oversized_adapter = deepcopy(exact_adapter)
    oversized_adapter["regeneration_design"]["candidate_count"] = payload_count
    oversized_adapter["regeneration_design"]["max_new_synapses"] = payload_count
    oversized_adapter["regeneration_design"]["candidate_synapses"] = _growth_candidates(
        payload_count
    )
    started = time.perf_counter()
    replay_review_block = ledger.rollout_regeneration_replay_artifact_review(
        oversized_adapter,
        replay,
    )
    replay_review_block_ms = (time.perf_counter() - started) * 1000.0

    controller_design = {
        "locality_radius": 2,
        "initial_weight": 0.02,
        "max_new_synapses": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        "mismatch_score": 0.9,
        "candidate_synapses": [
            {
                "pre_index": index % 31,
                "post_index": (index % 31) + 1,
                "initial_weight": 0.02,
            }
            for index in range(int(payload_count))
        ],
    }
    started = time.perf_counter()
    controller_error = None
    try:
        ReplayController._normalize_regeneration_design(controller_design)
    except ValueError as exc:
        controller_error = str(exc)
    controller_block_ms = (time.perf_counter() - started) * 1000.0

    after_snapshot = runtime_state.snapshot()
    return {
        "elapsed_ms": {
            "exact_design": exact_design_ms,
            "design_block": design_block_ms,
            "exact_shadow": exact_shadow_ms,
            "shadow_block": shadow_block_ms,
            "exact_developmental": exact_developmental_ms,
            "developmental_block": developmental_block_ms,
            "exact_adapter": exact_adapter_ms,
            "adapter_block": adapter_block_ms,
            "exact_replay_review": exact_replay_review_ms,
            "replay_review_block": replay_review_block_ms,
            "controller_block": controller_block_ms,
        },
        "exact_path": {
            "design_ready": bool(
                exact_design["promotion_gate"].get(
                    "eligible_for_operator_rollout_consolidation_design_review"
                )
            ),
            "shadow_ready": bool(
                exact_shadow["promotion_gate"].get(
                    "eligible_for_operator_rollout_consolidation_shadow_review"
                )
            ),
            "developmental_ready": bool(
                exact_developmental["promotion_gate"].get(
                    "eligible_for_operator_rollout_developmental_plasticity_review"
                )
            ),
            "adapter_ready": bool(
                exact_adapter["promotion_gate"].get(
                    "eligible_for_operator_rollout_regeneration_adapter_review"
                )
            ),
            "replay_review_ready": bool(
                exact_replay_review["promotion_gate"].get(
                    "eligible_for_operator_rollout_regeneration_replay_artifact_review"
                )
            ),
            "permit_preview_ready": bool(
                exact_replay_review["promotion_gate"].get(
                    "eligible_for_regeneration_permit_request"
                )
            ),
            "candidate_count": len(
                exact_replay_review["regeneration_design"].get("candidate_synapses", [])
            ),
            "source_windows": {
                "design": _candidate_window(exact_design, "sparse_candidate_source_window"),
                "shadow": _candidate_window(exact_shadow, "candidate_source_window"),
                "developmental": _candidate_window(
                    exact_developmental,
                    "candidate_source_window",
                ),
                "adapter": _candidate_window(exact_adapter, "growth_candidate_source_window"),
                "replay_review": _candidate_window(
                    exact_replay_review,
                    "candidate_source_window",
                ),
            },
        },
        "oversized_blocks": {
            "design": {
                "ready": bool(
                    design_block["promotion_gate"].get(
                        "eligible_for_operator_rollout_consolidation_design_review"
                    )
                ),
                "required": _required(design_block),
                "source_window": _candidate_window(
                    design_block,
                    "sparse_candidate_source_window",
                ),
            },
            "shadow": {
                "ready": bool(
                    shadow_block["promotion_gate"].get(
                        "eligible_for_operator_rollout_consolidation_shadow_review"
                    )
                ),
                "required": _required(shadow_block),
                "source_window": _candidate_window(shadow_block, "candidate_source_window"),
                "affected_synapse_count": int(shadow_block.get("affected_synapse_count", 0) or 0),
            },
            "developmental": {
                "ready": bool(
                    developmental_block["promotion_gate"].get(
                        "eligible_for_operator_rollout_developmental_plasticity_review"
                    )
                ),
                "integrity": dict(developmental_block.get("integrity_evidence") or {}),
                "source_window": _candidate_window(
                    developmental_block,
                    "candidate_source_window",
                ),
            },
            "adapter": {
                "ready": bool(
                    adapter_block["promotion_gate"].get(
                        "eligible_for_operator_rollout_regeneration_adapter_review"
                    )
                ),
                "integrity": dict(adapter_block.get("integrity_evidence") or {}),
                "source_window": _candidate_window(
                    adapter_block,
                    "growth_candidate_source_window",
                ),
            },
            "replay_review": {
                "ready": bool(
                    replay_review_block["promotion_gate"].get(
                        "eligible_for_operator_rollout_regeneration_replay_artifact_review"
                    )
                ),
                "permit_request_ready": bool(
                    replay_review_block["promotion_gate"].get(
                        "eligible_for_regeneration_permit_request"
                    )
                ),
                "required": _required(replay_review_block),
                "source_window": _candidate_window(
                    replay_review_block,
                    "candidate_source_window",
                ),
            },
            "controller_normalize": {
                "blocked": bool(controller_error),
                "error": controller_error,
            },
        },
        "runtime_state_unchanged_after_reviews": before_snapshot == after_snapshot,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.fmean(values)) if values else 0.0


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.median(values)) if values else 0.0


def _window_ok(window: dict[str, Any], *, truncated: bool) -> bool:
    return bool(
        window.get("source_window_count") == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
        and window.get("source_payload_truncated") is truncated
        and window.get("global_candidate_scan") is False
        and window.get("global_score_scan") is False
        and window.get("runs_live_tick") is False
        and window.get("runs_every_token") is False
        and window.get("gpu_used") is False
    )


def run_benchmark(*, payload_count: int, runs: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_before = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_before = int(torch.cuda.memory_reserved()) if cuda_available else 0
    tracemalloc.start()
    rows = [_run_case(payload_count=int(payload_count)) for _ in range(int(runs))]
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = int(torch.cuda.memory_allocated()) if cuda_available else 0
    cuda_reserved_after = int(torch.cuda.memory_reserved()) if cuda_available else 0

    last = rows[-1] if rows else {}
    exact = dict(last.get("exact_path") or {})
    blocks = dict(last.get("oversized_blocks") or {})
    exact_windows = dict(exact.get("source_windows") or {})
    oversized_windows = {
        name: dict(dict(payload).get("source_window") or {})
        for name, payload in blocks.items()
        if isinstance(payload, dict)
    }
    exact_window_checks = {
        name: _window_ok(dict(window), truncated=False)
        for name, window in exact_windows.items()
    }
    oversized_window_checks = {
        name: _window_ok(window, truncated=True)
        for name, window in oversized_windows.items()
    }
    quality = {
        "exact_32_candidate_path_reaches_permit_preview": bool(
            exact.get("design_ready")
            and exact.get("shadow_ready")
            and exact.get("developmental_ready")
            and exact.get("adapter_ready")
            and exact.get("replay_review_ready")
            and exact.get("permit_preview_ready")
            and exact.get("candidate_count") == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
            and all(exact_window_checks.values())
        ),
        "oversized_design_blocked": bool(
            blocks.get("design", {}).get("ready") is False
            and oversized_window_checks.get("design")
            and blocks.get("design", {}).get("required", {}).get(
                "sparse_candidate_payload_not_truncated"
            )
            is False
        ),
        "oversized_shadow_blocked": bool(
            blocks.get("shadow", {}).get("ready") is False
            and oversized_window_checks.get("shadow")
            and blocks.get("shadow", {}).get("affected_synapse_count")
            == SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT
        ),
        "oversized_developmental_blocked": bool(
            blocks.get("developmental", {}).get("ready") is False
            and oversized_window_checks.get("developmental")
            and blocks.get("developmental", {}).get("integrity", {}).get(
                "candidate_payload_not_truncated"
            )
            is False
        ),
        "oversized_adapter_blocked": bool(
            blocks.get("adapter", {}).get("ready") is False
            and oversized_window_checks.get("adapter")
            and blocks.get("adapter", {}).get("integrity", {}).get(
                "growth_candidate_payload_not_truncated"
            )
            is False
        ),
        "oversized_replay_review_blocked": bool(
            blocks.get("replay_review", {}).get("ready") is False
            and blocks.get("replay_review", {}).get("permit_request_ready") is False
            and oversized_window_checks.get("replay_review")
            and blocks.get("replay_review", {}).get("required", {}).get(
                "candidate_payload_not_truncated"
            )
            is False
        ),
        "direct_replay_controller_normalize_blocks_oversized": bool(
            blocks.get("controller_normalize", {}).get("blocked")
            and "candidate count must be bounded"
            in str(blocks.get("controller_normalize", {}).get("error") or "")
        ),
        "runtime_state_unchanged_after_reviews": bool(
            last.get("runtime_state_unchanged_after_reviews")
        ),
        "record_work_reduction": float(
            int(payload_count) / max(1, SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT)
        ),
        "old_full_payload_rollout_candidate_path_executable": False,
    }
    quality["pass"] = bool(
        quality["exact_32_candidate_path_reaches_permit_preview"]
        and quality["oversized_design_blocked"]
        and quality["oversized_shadow_blocked"]
        and quality["oversized_developmental_blocked"]
        and quality["oversized_adapter_blocked"]
        and quality["oversized_replay_review_blocked"]
        and quality["direct_replay_controller_normalize_blocks_oversized"]
        and quality["runtime_state_unchanged_after_reviews"]
        and quality["record_work_reduction"] >= 2.0
    )
    latency = {
        key: {
            "elapsed_mean_ms": _mean(rows, key),
            "elapsed_median_ms": _median(rows, key),
        }
        for key in [
            "exact_design",
            "design_block",
            "exact_shadow",
            "shadow_block",
            "exact_developmental",
            "developmental_block",
            "exact_adapter",
            "adapter_block",
            "exact_replay_review",
            "replay_review_block",
            "controller_block",
        ]
    }
    return {
        "artifact_kind": "bounded_snn_readout_ledger_rollout_candidate_window_benchmark",
        "surface": "bounded_snn_readout_ledger_rollout_candidate_window_benchmark.v1",
        "payload_count": int(payload_count),
        "window_limit": SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_candidate_order",
            "bounded_source_window_before_rollout_consolidation_or_regeneration_review",
            "untruncated_window_required_before_permit_preview_or_replay_controller_normalization",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "oversized_payload_reaches_permit_preview": False,
            "old_full_payload_rollout_candidate_path_executable": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "readout_ledger_gate_device": "cpu",
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
        "latency": latency,
        "last_run": last,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded readout-ledger rollout candidate windows."
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
        "pass={passed} exact={exact}/{limit} design={design}/{payload} "
        "shadow={shadow}/{payload} developmental={developmental}/{payload} "
        "adapter={adapter}/{payload} replay_review={replay_review}/{payload} "
        "controller_block={controller} "
        "reduction={reduction:.6f}".format(
            passed=report["pass"],
            exact=report["last_run"]["exact_path"]["candidate_count"],
            limit=report["window_limit"],
            payload=report["payload_count"],
            design=report["last_run"]["oversized_blocks"]["design"]["source_window"][
                "source_window_count"
            ],
            shadow=report["last_run"]["oversized_blocks"]["shadow"]["source_window"][
                "source_window_count"
            ],
            developmental=report["last_run"]["oversized_blocks"]["developmental"][
                "source_window"
            ]["source_window_count"],
            adapter=report["last_run"]["oversized_blocks"]["adapter"]["source_window"][
                "source_window_count"
            ],
            replay_review=report["last_run"]["oversized_blocks"]["replay_review"][
                "source_window"
            ]["source_window_count"],
            controller=report["quality"][
                "direct_replay_controller_normalize_blocks_oversized"
            ],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
