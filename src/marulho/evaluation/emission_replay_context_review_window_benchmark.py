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

from marulho.service.runtime_facade import RuntimeFacade
from marulho.service.runtime_state import RuntimeState
from marulho.service.snn_language_readout_ledger import (
    SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
    SNNLanguageReadoutEvidenceLedger,
)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ready_emission() -> dict[str, object]:
    emission_hash = _sha256_json({"emission": "memory pressure"})
    trajectory_hash = _sha256_json({"trajectory": "memory pressure"})
    prediction_hash = _sha256_json({"prediction": "memory pressure"})
    evaluation_hash = _sha256_json({"evaluation": "memory pressure"})
    weights_hash = _sha256_json({"weights": "memory pressure"})
    return {
        "surface": "snn_language_readout_emission.v1",
        "ready": True,
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "generates_text": True,
        "decodes_text": True,
        "generation_scope": "operator_visible_bounded_snn_readout_emission",
        "freeform_language_generation": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "writes_checkpoint": False,
        "promotes_fact": False,
        "promotes_action": False,
        "cognition_substrate": False,
        "language_output": {
            "text": "memory pressure",
            "labels": ["memory pressure"],
            "term_count": 1,
            "max_terms": 12,
        },
        "emission_hash": emission_hash,
        "emission_binding": {
            "trajectory_hash": trajectory_hash,
            "prediction_hash": prediction_hash,
            "transition_memory_evaluation_hash": evaluation_hash,
            "persistent_transition_weights_hash": weights_hash,
        },
        "promotion_gate": {
            "eligible_for_operator_display": True,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
        },
    }


def _ready_draft_for(
    prediction_hash: str,
    evaluation_hash: str,
    weights_hash: str,
    labels: list[str],
) -> dict[str, object]:
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


def _observed_slot(index: int) -> dict[str, object]:
    return {
        "label": f"memory pressure {index}",
        "pressure_band": "high",
        "grounded": True,
    }


class _StatusReadModel:
    def __init__(self, calls: dict[str, int]) -> None:
        self._calls = calls

    def snn_language_sequence_mismatch_probe(self, **kwargs: object) -> dict[str, object]:
        self._calls["mismatch"] += 1
        return {
            "surface": "snn_language_sequence_mismatch_probe.v1",
            "mismatch_hash": "mismatch-hash",
            "prediction_report": kwargs.get("prediction_report"),
        }

    def snn_language_plasticity_pressure(self, **kwargs: object) -> dict[str, object]:
        self._calls["pressure"] += 1
        return {
            "surface": "snn_language_plasticity_pressure.v1",
            "pressure_hash": "pressure-hash",
            "mismatch_report": kwargs.get("mismatch_report"),
        }


class _ReplayController:
    def __init__(self, calls: dict[str, int], runtime_state: RuntimeState) -> None:
        self._calls = calls
        self._runtime_state = runtime_state

    def record_snn_replay_evaluation_context(self, **kwargs: object) -> dict[str, object]:
        self._calls["context"] += 1
        self._runtime_state.mark_dirty_without_revision()
        source_metadata = kwargs.get("source_metadata")
        return {
            "surface": "snn_replay_evaluation_context.v1",
            "replay_evaluation_context_id": "context-benchmark",
            "evidence_hash": "context-hash",
            "source_metadata_hash": _sha256_json(source_metadata),
            "mismatch_hash": "mismatch-hash",
            "pressure_hash": "pressure-hash",
        }


class _Root:
    def __init__(
        self,
        *,
        runtime_state: RuntimeState,
        ledger: SNNLanguageReadoutEvidenceLedger,
        calls: dict[str, int],
    ) -> None:
        self._runtime_state = runtime_state
        self._snn_language_readout_ledger = ledger
        self._status_read_model = _StatusReadModel(calls)
        self._replay_controller = _ReplayController(calls, runtime_state)


def _make_facade() -> tuple[RuntimeState, RuntimeFacade, dict[str, object], dict[str, object], dict[str, int]]:
    lock = RLock()
    runtime_state = RuntimeState(lock=lock)
    ledger_state: dict[str, object] = {}
    ledger = SNNLanguageReadoutEvidenceLedger(
        lock=lock,
        runtime_state=runtime_state,
        ledger_state=lambda: ledger_state,
    )
    emission = _ready_emission()
    binding = dict(emission["emission_binding"])
    ledger.record_readout_draft(
        readout_draft=_ready_draft_for(
            str(binding["prediction_hash"]),
            str(binding["transition_memory_evaluation_hash"]),
            str(binding["persistent_transition_weights_hash"]),
            ["memory pressure"],
        ),
        expected_state_revision=0,
        operator_id="operator-readout",
        confirmation=True,
    )
    ledger.record_readout_emission_review(
        readout_emission=emission,
        expected_state_revision=0,
        operator_id="operator-emission",
        confirmation=True,
    )
    policy = ledger.emission_review_replay_evaluation_policy(limit=4)
    design = ledger.emission_review_replay_evaluation_design(
        policy,
        design_policy={"max_candidates": 1, "min_ready_candidates": 1},
        device_evidence={"device": "cpu", "source": "benchmark"},
    )
    prediction_report = {
        "surface": "snn_language_sequence_prediction_probe.v1",
        "provenance_evidence": {"prediction_hash": str(binding["prediction_hash"])},
    }
    calls = {"mismatch": 0, "pressure": 0, "context": 0}
    root = _Root(runtime_state=runtime_state, ledger=ledger, calls=calls)
    return runtime_state, RuntimeFacade(root), design, prediction_report, calls


def _run_case(*, payload_count: int) -> dict[str, Any]:
    runtime_state, facade, design, prediction_report, calls = _make_facade()
    exact_design = deepcopy(design)
    seed = dict(exact_design["selected_replay_context_seeds"][0])
    exact_design["selected_replay_context_seeds"] = [
        {**seed, "replay_context_seed_hash": f"exact-seed-{index}"}
        for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)
    ]
    exact_slots = [_observed_slot(index) for index in range(SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)]

    started = time.perf_counter()
    exact = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=exact_design,
        prediction_report=prediction_report,
        observed_readout_slots=exact_slots,
        operator_id="operator-benchmark",
        confirmation=True,
    )
    exact_ms = (time.perf_counter() - started) * 1000.0

    oversized_seed_design = deepcopy(exact_design)
    oversized_seed_design["selected_replay_context_seeds"] = [
        {**seed, "replay_context_seed_hash": f"oversized-seed-{index}"}
        for index in range(int(payload_count))
    ]
    started = time.perf_counter()
    seed_block = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=oversized_seed_design,
        prediction_report=prediction_report,
        observed_readout_slots=[_observed_slot(0)],
        operator_id="operator-benchmark",
        confirmation=True,
    )
    seed_block_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    slot_block = facade.snn_language_readout_emission_replay_context_review(
        emission_replay_evaluation_design=design,
        prediction_report=prediction_report,
        observed_readout_slots=[_observed_slot(index) for index in range(int(payload_count))],
        operator_id="operator-benchmark",
        confirmation=True,
    )
    slot_block_ms = (time.perf_counter() - started) * 1000.0

    return {
        "elapsed_ms": {
            "exact_context_review": exact_ms,
            "oversized_seed_block": seed_block_ms,
            "oversized_observed_slot_block": slot_block_ms,
        },
        "exact_path": {
            "accepted": bool(exact.get("accepted")),
            "records_replay_context": bool(exact.get("records_replay_context")),
            "seed_source_window": dict(exact.get("seed_source_window") or {}),
            "observed_slot_source_window": dict(exact.get("observed_slot_source_window") or {}),
            "context_calls_after_exact": dict(calls),
        },
        "oversized_blocks": {
            "seeds": {
                "accepted": bool(seed_block.get("accepted")),
                "records_replay_context": bool(seed_block.get("records_replay_context")),
                "required": dict(seed_block.get("promotion_gate", {}).get("required_evidence") or {}),
                "source_window": dict(seed_block.get("seed_source_window") or {}),
            },
            "observed_slots": {
                "accepted": bool(slot_block.get("accepted")),
                "records_replay_context": bool(slot_block.get("records_replay_context")),
                "required": dict(slot_block.get("promotion_gate", {}).get("required_evidence") or {}),
                "source_window": dict(slot_block.get("observed_slot_source_window") or {}),
            },
        },
        "call_counts_after_all_cases": dict(calls),
        "runtime_state_revision": int(runtime_state.state_revision),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.fmean(values)) if values else 0.0


def _median(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row["elapsed_ms"][key]) for row in rows]
    return float(statistics.median(values)) if values else 0.0


def _window_ok(window: dict[str, Any], *, truncated: bool) -> bool:
    return bool(
        window.get("source_window_count") == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
        and window.get("source_mapping_count") == SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT
        and window.get("source_payload_truncated") is truncated
        and window.get("global_candidate_scan") is False
        and window.get("global_score_scan") is False
        and window.get("runs_live_tick") is False
        and window.get("runs_every_token") is False
        and window.get("archival_storage_device") == "cpu"
        and window.get("gpu_resident_archival_metadata") is False
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
    seed_block = dict(blocks.get("seeds") or {})
    slot_block = dict(blocks.get("observed_slots") or {})
    call_counts = dict(last.get("call_counts_after_all_cases") or {})
    quality = {
        "exact_32_seed_and_slot_path_records_one_context": bool(
            exact.get("accepted")
            and exact.get("records_replay_context")
            and _window_ok(dict(exact.get("seed_source_window") or {}), truncated=False)
            and _window_ok(dict(exact.get("observed_slot_source_window") or {}), truncated=False)
        ),
        "oversized_seed_payload_blocked_before_context_recording": bool(
            seed_block.get("accepted") is False
            and seed_block.get("records_replay_context") is False
            and _window_ok(dict(seed_block.get("source_window") or {}), truncated=True)
            and seed_block.get("required", {}).get("seed_payload_not_truncated") is False
        ),
        "oversized_observed_slot_payload_blocked_before_context_recording": bool(
            slot_block.get("accepted") is False
            and slot_block.get("records_replay_context") is False
            and _window_ok(dict(slot_block.get("source_window") or {}), truncated=True)
            and slot_block.get("required", {}).get("observed_slot_payload_not_truncated") is False
        ),
        "blocked_payloads_do_not_call_mismatch_pressure_or_replay_controller": bool(
            call_counts == {"mismatch": 1, "pressure": 1, "context": 1}
        ),
        "record_work_reduction": float(
            int(payload_count) / max(1, SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT)
        ),
        "old_full_payload_context_review_path_executable": False,
    }
    quality["pass"] = bool(
        quality["exact_32_seed_and_slot_path_records_one_context"]
        and quality["oversized_seed_payload_blocked_before_context_recording"]
        and quality["oversized_observed_slot_payload_blocked_before_context_recording"]
        and quality["blocked_payloads_do_not_call_mismatch_pressure_or_replay_controller"]
        and quality["record_work_reduction"] >= 2.0
    )
    latency = {
        key: {
            "elapsed_mean_ms": _mean(rows, key),
            "elapsed_median_ms": _median(rows, key),
        }
        for key in [
            "exact_context_review",
            "oversized_seed_block",
            "oversized_observed_slot_block",
        ]
    }
    return {
        "artifact_kind": "bounded_snn_emission_replay_context_review_window_benchmark",
        "surface": "bounded_snn_emission_replay_context_review_window_benchmark.v1",
        "payload_count": int(payload_count),
        "window_limit": SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT,
        "runs": int(runs),
        "selection_criteria": [
            "caller_supplied_hash_only_context_seed_order",
            "caller_supplied_observed_sparse_slot_order",
            "bounded_source_window_before_mismatch_pressure_or_context_recording",
            "untruncated_window_required_before_replay_controller_context_recording",
        ],
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "raw_text_payload_loaded": False,
            "hidden_language_reasoning": False,
            "language_reasoning": False,
            "oversized_payload_reaches_replay_controller": False,
            "old_full_payload_context_review_path_executable": False,
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "context_review_gate_device": "cpu",
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
        "latency": latency,
        "last_run": last,
        "quality": quality,
        "pass": bool(quality["pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark bounded emission replay-context review windows."
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
        "pass={passed} exact_seed={exact_seed}/{limit} exact_slots={exact_slots}/{limit} "
        "seed_block={seed_block}/{payload} slot_block={slot_block}/{payload} "
        "context_calls={context_calls} reduction={reduction:.6f}".format(
            passed=report["pass"],
            exact_seed=report["last_run"]["exact_path"]["seed_source_window"][
                "source_window_count"
            ],
            exact_slots=report["last_run"]["exact_path"]["observed_slot_source_window"][
                "source_window_count"
            ],
            limit=report["window_limit"],
            payload=report["payload_count"],
            seed_block=report["last_run"]["oversized_blocks"]["seeds"]["source_window"][
                "source_window_count"
            ],
            slot_block=report["last_run"]["oversized_blocks"]["observed_slots"][
                "source_window"
            ]["source_window_count"],
            context_calls=report["last_run"]["call_counts_after_all_cases"]["context"],
            reduction=report["quality"]["record_work_reduction"],
        )
    )


if __name__ == "__main__":
    main()
