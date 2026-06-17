from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Mapping, Sequence, TextIO

import torch

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.semantics.spike_language_neurons import (
    build_snn_language_transition_memory_prediction_evaluation,
)


ARTIFACT_KIND = "terminus_snn_language_readout_corpus_evaluation"
SURFACE = "snn_language_readout_corpus_evaluation.v1"
CHECKPOINT_REVIEW_ARTIFACT_KIND = "terminus_snn_language_readout_corpus_checkpoint_review"
CHECKPOINT_REVIEW_SURFACE = "snn_language_readout_corpus_checkpoint_review.v1"
SPARSE_READOUT_CHECKPOINT_ARTIFACT_KIND = "marulho_snn_language_sparse_readout_checkpoint"
SPARSE_READOUT_CHECKPOINT_SURFACE = "snn_language_sparse_readout_checkpoint.v1"
PASSING_STATUS = "promote_bounded_readout_review"
FAILING_STATUS = "reject_live_readout_collect_evidence"
CHECKPOINT_PASSING_STATUS = "promote_checkpointed_bounded_readout_review"
CHECKPOINT_FAILING_STATUS = "reject_checkpointed_readout_collect_evidence"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _batches(value: Any, *, max_batches: int, max_slots: int) -> list[list[dict[str, Any]]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[list[dict[str, Any]]] = []
    for batch in value[:max_batches]:
        if not isinstance(batch, Sequence) or isinstance(batch, (str, bytes)):
            continue
        slots = [dict(slot) for slot in batch[:max_slots] if isinstance(slot, Mapping)]
        if slots:
            output.append(slots)
    return output


def _grounding_summary(batches: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    total = 0
    grounded = 0
    labels: list[str] = []
    unsupported_labels: list[str] = []
    for batch in batches:
        for slot in batch:
            label = str(slot.get("label") or "").strip()
            if label and label not in labels:
                labels.append(label)
            total += 1
            if bool(slot.get("grounded")):
                grounded += 1
            elif label and label not in unsupported_labels:
                unsupported_labels.append(label)
    fraction = grounded / max(1, total)
    return {
        "slot_count": total,
        "grounded_slot_count": grounded,
        "unsupported_slot_count": max(0, total - grounded),
        "grounded_fraction": float(fraction),
        "support_terms": labels[:24],
        "unsupported_terms": unsupported_labels[:12],
        "supported": bool(total > 0 and fraction >= 0.5),
    }


def _resolve_device(device_evidence: Mapping[str, Any]) -> torch.device:
    requested = str(
        device_evidence.get("device")
        or device_evidence.get("tensor_device")
        or "cpu"
    )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        return torch.device(requested)
    except (RuntimeError, TypeError):
        return torch.device("cpu")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_sparse_transition_weights(
    corpus: Mapping[str, Any],
    *,
    max_transition_weights: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _mapping(corpus.get("transition_memory_state"))
    raw_weights = (
        dict(state.get("sparse_transition_weights"))
        if isinstance(state.get("sparse_transition_weights"), Mapping)
        else {}
    )
    bounded = {
        str(key): raw_weights[key]
        for key in sorted(raw_weights, key=lambda item: str(item))[:max_transition_weights]
        if str(key).strip()
    }
    metadata = {
        "transition_weight_count": len(raw_weights),
        "checkpointed_transition_weight_count": len(bounded),
        "max_transition_weights": int(max_transition_weights),
        "within_checkpoint_bound": len(raw_weights) <= int(max_transition_weights),
        "persistent_transition_weights_hash": _sha256_json(raw_weights),
        "checkpointed_transition_weights_hash": _sha256_json(bounded),
    }
    return bounded, metadata


def _cuda_memory_allocated(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.memory_allocated(device))
    except RuntimeError:
        return None


def _corpus_metadata(corpus: Mapping[str, Any], sample_size: int) -> dict[str, Any]:
    metadata = _mapping(corpus.get("corpus") or corpus.get("dataset") or corpus.get("metadata"))
    source_type = str(metadata.get("source_type") or metadata.get("kind") or "local_bounded_corpus")
    dataset_name = str(metadata.get("dataset_name") or metadata.get("name") or "unnamed_readout_corpus")
    cache_path = metadata.get("cache_path")
    split = str(metadata.get("split") or corpus.get("split") or "evaluation")
    license_name = str(metadata.get("license") or metadata.get("terms") or "unspecified")
    recorded_sample_size = metadata.get("sample_size", sample_size)
    return {
        "source_type": source_type,
        "dataset_name": dataset_name,
        "license": license_name,
        "terms": metadata.get("terms") or license_name,
        "split": split,
        "sample_size": int(recorded_sample_size or 0),
        "cache_path": str(cache_path) if cache_path is not None else None,
        "external_data_source": bool(metadata.get("external_data_source") or source_type in {"huggingface", "hf"}),
        "runtime_cognition_dependency": False,
        "loads_external_checkpoint": False,
        "corpus_hash": _sha256_json(
            {
                "metadata": metadata,
                "split": split,
                "sample_size": int(sample_size),
            }
        ),
    }


def evaluate_snn_language_readout_corpus(
    corpus: Mapping[str, Any],
    device_evidence: Mapping[str, Any] | None = None,
    *,
    learning_rate: float = 0.08,
    epochs: int = 2,
    top_k: int = 8,
    max_train_batches: int = 128,
    max_eval_batches: int = 128,
    max_slots_per_batch: int = 8,
) -> dict[str, Any]:
    """Evaluate sparse next-readout trajectories over a bounded corpus.

    The runner is intentionally isolated: it may build temporary local weights
    inside the existing prediction evaluator, but it never returns weights,
    applies plasticity, mutates RuntimeState, or loads an external checkpoint.
    """

    device_report = dict(device_evidence or corpus.get("device_evidence") or {})
    device = _resolve_device(device_report)
    train_batches = _batches(
        corpus.get("training_readout_slot_batches") or corpus.get("train"),
        max_batches=max_train_batches,
        max_slots=max_slots_per_batch,
    )
    eval_batches = _batches(
        corpus.get("evaluation_readout_slot_batches") or corpus.get("eval"),
        max_batches=max_eval_batches,
        max_slots=max_slots_per_batch,
    )
    transition_memory_state = _mapping(corpus.get("transition_memory_state"))
    all_batches = [*train_batches, *eval_batches]
    grounding = _grounding_summary(all_batches)
    metadata = _corpus_metadata(corpus, sample_size=len(eval_batches))

    cuda_before = _cuda_memory_allocated(device)
    tracemalloc.start()
    started = time.perf_counter()
    transition_evaluation = build_snn_language_transition_memory_prediction_evaluation(
        train_batches,
        eval_batches,
        transition_memory_state,
        {**device_report, "device": str(device)},
        learning_rate=learning_rate,
        epochs=epochs,
        top_k=top_k,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_after = _cuda_memory_allocated(device)

    summary = _mapping(transition_evaluation.get("evaluation_summary"))
    gate = _mapping(transition_evaluation.get("promotion_gate"))
    required = _mapping(gate.get("required_evidence"))
    memory_count = int(summary.get("persistent_transition_weight_count", 0) or 0)
    pair_count = int(summary.get("evaluation_pair_count", 0) or 0)
    worsened_count = int(summary.get("worsened_sequence_count", 0) or 0)
    readiness = {
        "evaluation_available": bool(pair_count > 0),
        "persistent_transition_memory_available": memory_count > 0,
        "persistent_memory_non_worsening": bool(required.get("persistent_memory_non_worsening")),
        "grounding_supported": bool(grounding["supported"]),
        "device_evidence_available": bool(str(device) != "unknown"),
        "external_checkpoint_absent": True,
        "runtime_mutation_absent": True,
        "freeform_generation_absent": True,
        "worsened_sequence_absent": worsened_count == 0,
    }
    promotable = all(readiness.values()) and bool(
        gate.get("eligible_for_bounded_readout_generation_review")
    )
    if promotable:
        decision = PASSING_STATUS
        next_gate = "operator_review_bounded_snn_readout_corpus_report"
        reason_codes = ["bounded_sparse_readout_evaluation_passed"]
    else:
        decision = FAILING_STATUS
        next_gate = "collect_grounded_transition_memory_evaluation_window"
        reason_codes = [
            name
            for name, passed in readiness.items()
            if not bool(passed)
        ]
        if not reason_codes:
            reason_codes.append("transition_memory_gate_not_ready")

    report_hash = _sha256_json(
        {
            "metadata": metadata,
            "training_batches": train_batches,
            "evaluation_batches": eval_batches,
            "transition_memory_hash": _mapping(
                transition_evaluation.get("provenance_evidence")
            ).get("persistent_transition_weights_hash"),
            "summary": summary,
            "decision": decision,
        }
    )
    return {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": decision,
        "available": bool(pair_count > 0),
        "passed": promotable,
        "source": "evaluation.snn_language_readout_corpus",
        "owned_by_marulho": True,
        "external_dependency": False,
        "external_runtime_cognition": False,
        "loads_external_checkpoint": False,
        "generates_text": False,
        "decodes_text": False,
        "freeform_language_generation": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": False,
        "evaluates_sequence_readout": True,
        "verification_pattern": "baseline_vs_persistent_sparse_next_readout_evaluation",
        "corpus_provenance": metadata,
        "corpus_bounds": {
            "training_batch_count": len(train_batches),
            "evaluation_batch_count": len(eval_batches),
            "evaluation_pair_count": pair_count,
            "max_train_batches": int(max_train_batches),
            "max_eval_batches": int(max_eval_batches),
            "max_slots_per_batch": int(max_slots_per_batch),
        },
        "grounding_evidence": grounding,
        "device_evidence": {
            "requested_device": str(device_report.get("device") or device_report.get("tensor_device") or "unknown"),
            "tensor_device": str(device),
            "cuda_tensor": device.type == "cuda",
            "device_source": device_report.get("source") or device_report.get("device_source"),
        },
        "metabolism_evidence": {
            "latency_ms": float(latency_ms),
            "python_peak_memory_bytes": int(peak_bytes),
            "cuda_memory_allocated_before_bytes": cuda_before,
            "cuda_memory_allocated_after_bytes": cuda_after,
            "cuda_memory_allocated_delta_bytes": (
                int(cuda_after - cuda_before)
                if cuda_before is not None and cuda_after is not None
                else None
            ),
        },
        "sequence_evaluation_summary": {
            "evaluation_pair_count": pair_count,
            "baseline_mean_mismatch_score": summary.get("baseline_mean_mismatch_score"),
            "memory_mean_mismatch_score": summary.get("memory_mean_mismatch_score"),
            "mean_mismatch_delta": summary.get("mean_mismatch_delta"),
            "influenced_prediction_count": int(summary.get("influenced_prediction_count", 0) or 0),
            "improved_sequence_count": int(summary.get("improved_sequence_count", 0) or 0),
            "worsened_sequence_count": worsened_count,
            "persistent_transition_weight_count": memory_count,
        },
        "transition_memory_evaluation": transition_evaluation,
        "runtime_truth_gate": {
            "available_status": "available" if pair_count > 0 else "missing_evaluation_pairs",
            "trained_status": "isolated_evaluation_only_not_runtime_trained",
            "grounded_status": "grounded" if grounding["supported"] else "insufficient_grounding",
            "device_status": "cuda" if device.type == "cuda" else str(device),
            "mutation_gate_status": "mutation_absent",
            "latency_ms": float(latency_ms),
            "memory_cost_bytes": int(peak_bytes),
            "vram_delta_bytes": (
                int(cuda_after - cuda_before)
                if cuda_before is not None and cuda_after is not None
                else None
            ),
            "promotion_decision": decision,
            "promotable": promotable,
            "next_gate": next_gate,
            "reason_codes": reason_codes,
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if promotable else "rejected_for_live_readout",
            "eligible_for_bounded_readout_generation_review": promotable,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "eligible_for_runtime_training": False,
            "eligible_for_plasticity_application": False,
            "requires_operator_approval": promotable,
            "next_gate": next_gate,
            "required_evidence": readiness,
        },
        "provenance_evidence": {
            "report_hash": report_hash,
            "report_id": f"snn-readout-corpus-eval:{report_hash[:16]}",
            "transition_memory_evaluation_hash": _mapping(
                transition_evaluation.get("provenance_evidence")
            ).get("evaluation_hash"),
            "corpus_hash": metadata["corpus_hash"],
            "hash_algorithm": "sha256_canonical_json",
        },
    }


def build_snn_language_readout_corpus_checkpoint_review(
    corpus: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    rollback_path: str | Path | None = None,
    max_transition_weights: int = 4096,
) -> dict[str, Any]:
    """Write an isolated sparse readout checkpoint for operator review.

    This is not a live runtime checkpoint. It persists the bounded sparse
    transition-memory material that was already evaluated, writes a rollback
    manifest for deleting the isolated artifact, and verifies that the checkpoint
    can be read back with the expected hash.
    """

    report = dict(evaluation_report or {})
    checkpoint = Path(checkpoint_path)
    rollback = Path(rollback_path) if rollback_path is not None else checkpoint.with_suffix(
        checkpoint.suffix + ".rollback.json"
    )
    transition_evaluation = _mapping(report.get("transition_memory_evaluation"))
    transition_provenance = _mapping(transition_evaluation.get("provenance_evidence"))
    report_provenance = _mapping(report.get("provenance_evidence"))
    corpus_provenance = _mapping(report.get("corpus_provenance"))
    grounding = _mapping(report.get("grounding_evidence"))
    device = _mapping(report.get("device_evidence"))
    runtime_gate = _mapping(report.get("runtime_truth_gate"))
    weights, weight_metadata = _checkpoint_sparse_transition_weights(
        corpus,
        max_transition_weights=max_transition_weights,
    )
    expected_transition_hash = str(
        transition_provenance.get("persistent_transition_weights_hash")
        or transition_provenance.get("transition_memory_hash")
        or ""
    )
    observed_transition_hash = str(weight_metadata["persistent_transition_weights_hash"])
    started = time.perf_counter()
    required = {
        "evaluation_report_available": bool(report),
        "evaluation_artifact_kind_valid": report.get("artifact_kind") == ARTIFACT_KIND,
        "evaluation_surface_valid": report.get("surface") == SURFACE,
        "evaluation_passed": bool(report.get("passed")),
        "runtime_gate_promotable": bool(runtime_gate.get("promotable")),
        "grounding_supported": bool(grounding.get("supported")),
        "transition_weights_available": bool(weights),
        "transition_weights_within_bound": bool(weight_metadata["within_checkpoint_bound"]),
        "transition_weight_hash_matches_report": bool(
            expected_transition_hash and expected_transition_hash == observed_transition_hash
        ),
        "checkpoint_path_available": bool(str(checkpoint)),
        "rollback_path_available": bool(str(rollback)),
        "external_checkpoint_absent": not bool(report.get("loads_external_checkpoint")),
        "runtime_mutation_absent": not bool(report.get("mutates_runtime_state")),
        "freeform_generation_absent": not bool(report.get("freeform_language_generation")),
    }
    ready_before_write = all(required.values())
    checkpoint_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": SPARSE_READOUT_CHECKPOINT_ARTIFACT_KIND,
        "surface": SPARSE_READOUT_CHECKPOINT_SURFACE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "evaluation.snn_language_readout_corpus.checkpoint_review",
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "runtime_cognition_dependency": False,
        "generates_text": False,
        "decodes_text": False,
        "trains_runtime_model": False,
        "mutates_runtime_state": False,
        "applies_plasticity": False,
        "checkpoint_scope": "isolated_slow_path_sparse_readout_review",
        "source_report_hash": report_provenance.get("report_hash"),
        "source_transition_evaluation_hash": transition_provenance.get("evaluation_hash"),
        "corpus_hash": report_provenance.get("corpus_hash"),
        "dataset_name": corpus_provenance.get("dataset_name"),
        "dataset_license": corpus_provenance.get("license"),
        "dataset_terms": corpus_provenance.get("terms"),
        "split": corpus_provenance.get("split"),
        "sample_size": corpus_provenance.get("sample_size"),
        "tensor_device": device.get("tensor_device"),
        "sparse_transition_weights": weights if ready_before_write else {},
        "weight_metadata": weight_metadata,
    }
    checkpoint_payload_hash = _sha256_json(checkpoint_payload)
    checkpoint_payload["checkpoint_payload_hash"] = checkpoint_payload_hash

    checkpoint_written = False
    checkpoint_sha256: str | None = None
    rollback_manifest_hash: str | None = None
    restore_verified = False
    write_error: str | None = None
    if ready_before_write:
        try:
            _write_json(checkpoint, checkpoint_payload)
            checkpoint_sha256 = _sha256_file(checkpoint)
            rollback_manifest = {
                "schema_version": 1,
                "artifact_kind": "terminus_snn_language_readout_corpus_checkpoint_rollback_manifest",
                "surface": "snn_language_readout_corpus_checkpoint_rollback.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_payload_hash": checkpoint_payload_hash,
                "source_report_hash": report_provenance.get("report_hash"),
                "source_transition_evaluation_hash": transition_provenance.get("evaluation_hash"),
                "production_runtime_changed": False,
                "rollback_action": "delete_isolated_sparse_readout_checkpoint_and_checkpoint_review_report",
            }
            rollback_manifest_hash = _sha256_json(rollback_manifest)
            rollback_manifest["rollback_manifest_hash"] = rollback_manifest_hash
            _write_json(rollback, rollback_manifest)
            restored = json.loads(checkpoint.read_text(encoding="utf-8"))
            restore_verified = (
                isinstance(restored, Mapping)
                and restored.get("checkpoint_payload_hash") == checkpoint_payload_hash
                and _sha256_file(checkpoint) == checkpoint_sha256
            )
            checkpoint_written = restore_verified
        except OSError as exc:
            write_error = str(exc)

    latency_ms = (time.perf_counter() - started) * 1000.0
    required["checkpoint_written"] = checkpoint_written
    required["checkpoint_restore_verified"] = restore_verified
    required["rollback_manifest_written"] = bool(rollback.exists()) if ready_before_write else False
    ready = all(required.values())
    reason_codes = [name for name, passed in required.items() if not bool(passed)]
    if not reason_codes and ready:
        reason_codes = ["checkpointed_bounded_sparse_readout_review_passed"]
    decision = CHECKPOINT_PASSING_STATUS if ready else CHECKPOINT_FAILING_STATUS
    return {
        "schema_version": 1,
        "artifact_kind": CHECKPOINT_REVIEW_ARTIFACT_KIND,
        "surface": CHECKPOINT_REVIEW_SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": decision,
        "available": bool(report),
        "ready": ready,
        "passed": ready,
        "source": "evaluation.snn_language_readout_corpus.checkpoint_review",
        "owned_by_marulho": True,
        "external_dependency": False,
        "loads_external_checkpoint": False,
        "external_runtime_cognition": False,
        "generates_text": False,
        "decodes_text": False,
        "freeform_language_generation": False,
        "trains_runtime_model": False,
        "returns_trained_weights": False,
        "applies_plasticity": False,
        "mutates_runtime_state": False,
        "writes_checkpoint": checkpoint_written,
        "writes_live_checkpoint": False,
        "checkpoint_scope": "isolated_slow_path_sparse_readout_review",
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_payload_hash": checkpoint_payload_hash,
        "checkpoint_file_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
        "rollback_evidence": {
            "available": bool(rollback.exists()) if ready_before_write else False,
            "rollback_manifest_path": str(rollback),
            "rollback_manifest_hash": rollback_manifest_hash,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_restore_verified": restore_verified,
            "production_runtime_changed": False,
            "rollback_action": "delete_isolated_sparse_readout_checkpoint_and_checkpoint_review_report",
            "bound_to_evaluation_hash": bool(
                report_provenance.get("report_hash")
                and checkpoint_payload.get("source_report_hash") == report_provenance.get("report_hash")
            ),
        },
        "checkpoint_weight_evidence": weight_metadata,
        "corpus_provenance": dict(corpus_provenance),
        "grounding_evidence": dict(grounding),
        "device_evidence": dict(device),
        "metabolism_evidence": {
            "latency_ms": float(latency_ms),
            "checkpoint_file_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
            "vram_delta_bytes": None,
        },
        "runtime_truth_gate": {
            "available_status": "available" if bool(report) else "missing_evaluation_report",
            "trained_status": "checkpointed_sparse_transition_memory_review_only",
            "grounded_status": "grounded" if grounding.get("supported") else "insufficient_grounding",
            "device_status": device.get("tensor_device") or "unknown",
            "mutation_gate_status": "runtime_mutation_absent",
            "checkpoint_status": "restore_verified" if restore_verified else "checkpoint_not_verified",
            "rollback_status": "rollback_manifest_available" if rollback.exists() else "rollback_missing",
            "latency_ms": float(latency_ms),
            "memory_cost_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
            "vram_delta_bytes": None,
            "promotion_decision": decision,
            "promotable": ready,
            "next_gate": "operator_review_checkpointed_bounded_snn_readout"
            if ready
            else "collect_checkpoint_rollback_and_grounded_readout_evidence",
            "reason_codes": reason_codes,
        },
        "promotion_gate": {
            "status": "ready_for_operator_review" if ready else "blocked_missing_checkpoint_review_evidence",
            "eligible_for_checkpointed_bounded_readout_review": ready,
            "eligible_for_freeform_language_generation": False,
            "eligible_for_cognition_substrate": False,
            "eligible_for_fact_promotion": False,
            "eligible_for_action": False,
            "eligible_for_runtime_training": False,
            "eligible_for_plasticity_application": False,
            "requires_operator_approval": ready,
            "requires_checkpoint_restore_verification": True,
            "requires_rollback_manifest": True,
            "next_gate": "operator_review_checkpointed_bounded_snn_readout"
            if ready
            else "collect_checkpoint_rollback_and_grounded_readout_evidence",
            "required_evidence": required,
        },
        "provenance_evidence": {
            "checkpoint_review_hash": _sha256_json(
                {
                    "checkpoint_payload_hash": checkpoint_payload_hash,
                    "checkpoint_sha256": checkpoint_sha256,
                    "rollback_manifest_hash": rollback_manifest_hash,
                    "required": required,
                    "decision": decision,
                }
            ),
            "source_report_hash": report_provenance.get("report_hash"),
            "source_transition_evaluation_hash": transition_provenance.get("evaluation_hash"),
            "persistent_transition_weights_hash": observed_transition_hash,
            "hash_algorithm": "sha256_canonical_json",
        },
        "write_error": write_error,
    }


def evaluate_snn_language_readout_corpus_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    checkpoint_output_path: str | Path | None = None,
    checkpoint_review_output_path: str | Path | None = None,
    learning_rate: float = 0.08,
    epochs: int = 2,
    top_k: int = 8,
    max_checkpoint_transition_weights: int = 4096,
) -> dict[str, Any]:
    path = Path(input_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("SNN language readout corpus input must be a JSON object.")
    report = evaluate_snn_language_readout_corpus(
        loaded,
        learning_rate=learning_rate,
        epochs=epochs,
        top_k=top_k,
    )
    report["input_path"] = str(path)
    if checkpoint_output_path is not None:
        review_output = (
            Path(checkpoint_review_output_path)
            if checkpoint_review_output_path is not None
            else Path(checkpoint_output_path).with_name("readout-corpus-checkpoint-review.json")
        )
        checkpoint_review = build_snn_language_readout_corpus_checkpoint_review(
            loaded,
            report,
            checkpoint_output_path,
            max_transition_weights=max_checkpoint_transition_weights,
        )
        checkpoint_review["output_path"] = str(review_output)
        written_review = write_json_report_with_readme(
            review_output,
            checkpoint_review,
            title="SNN Language Readout Corpus Checkpoint Review",
        )
        checkpoint_review["output_path"] = str(written_review)
        report["checkpoint_review"] = checkpoint_review
    if output_path is not None:
        report["output_path"] = str(output_path)
        output = write_json_report_with_readme(
            output_path,
            report,
            title="SNN Language Readout Corpus Evaluation",
        )
        report["output_path"] = str(output)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a bounded MARULHO-owned SNN language readout corpus."
    )
    parser.add_argument("--input", type=Path, required=True, help="Bounded readout corpus JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional report JSON path, for example reports/snn_language_readout_corpus/readout-corpus-evaluation.json.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=None,
        help="Optional isolated sparse readout checkpoint path for operator review.",
    )
    parser.add_argument(
        "--checkpoint-review-output",
        type=Path,
        default=None,
        help="Optional checkpoint review report path. Defaults next to --checkpoint-output.",
    )
    parser.add_argument("--max-checkpoint-transition-weights", type=int, default=4096)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    if args.max_checkpoint_transition_weights <= 0:
        parser.error("--max-checkpoint-transition-weights must be positive")
    report = evaluate_snn_language_readout_corpus_file(
        args.input,
        output_path=args.output,
        checkpoint_output_path=args.checkpoint_output,
        checkpoint_review_output_path=args.checkpoint_review_output,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        top_k=args.top_k,
        max_checkpoint_transition_weights=args.max_checkpoint_transition_weights,
    )
    encoded = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    checkpoint_review = _mapping(report.get("checkpoint_review"))
    checkpoint_ready = (
        bool(checkpoint_review.get("ready"))
        if args.checkpoint_output is not None
        else True
    )
    return 0 if bool(report.get("passed")) and checkpoint_ready else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
