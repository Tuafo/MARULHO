from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, MutableMapping

APPLIED_REPLAY_LINEAGE_INCREMENTAL_SURFACE = (
    "snn_applied_replay_lineage_incremental_summary.v1"
)
APPLIED_REPLAY_LINEAGE_CHECKPOINT_SUMMARY_SURFACE = (
    "snn_applied_replay_lineage_checkpoint_summary.v1"
)

_DIGEST_MODULUS = 1 << 256
_ZERO_DIGEST = "0" * 64


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest_int(value: Any) -> int:
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        return 0


def _digest_hex(value: int) -> str:
    return f"{int(value) % _DIGEST_MODULUS:064x}"


def _lineage_row_material(
    synapse_key: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    if str(provenance.get("provenance_type") or "") != "replay_regeneration":
        return None
    source_metadata_hash = str(provenance.get("source_metadata_hash") or "")
    emission_lineage = (
        dict(provenance.get("emission_lineage"))
        if isinstance(provenance.get("emission_lineage"), Mapping)
        else {}
    )
    if not source_metadata_hash and not emission_lineage:
        return None
    return {
        "synapse_key": str(synapse_key),
        "source_metadata_hash": source_metadata_hash,
        "emission_lineage": emission_lineage,
    }


def _row_complete(material: Mapping[str, Any]) -> bool:
    emission_lineage = (
        material.get("emission_lineage")
        if isinstance(material.get("emission_lineage"), Mapping)
        else {}
    )
    return bool(
        material.get("source_metadata_hash")
        and emission_lineage.get("emission_hash")
        and emission_lineage.get("readout_evidence_hash")
        and emission_lineage.get("prediction_hash")
    )


def _empty_incremental_summary() -> dict[str, Any]:
    return {
        "surface": APPLIED_REPLAY_LINEAGE_INCREMENTAL_SURFACE,
        "source": "snn_language_plasticity.synapse_provenance_mutation",
        "owned_by_marulho": True,
        "summary_policy": "mutation_time_o1_digest",
        "selection_criteria": [
            "only provenance rows written with provenance_type=replay_regeneration",
            "row must carry source_metadata_hash or emission_lineage",
            "digest is updated when a synapse provenance row is written or removed",
        ],
        "memory_budget": "one CPU row hash and completion bit per replay-regenerated synapse",
        "archival_metadata_device": "cpu",
        "gpu_used": False,
        "full_provenance_scan": False,
        "source_record_scan_count": 0,
        "lineage_row_hash_by_synapse_key": {},
        "lineage_complete_by_synapse_key": {},
        "lineage_digest_xor": _ZERO_DIGEST,
        "lineage_digest_sum_mod_2_256": _ZERO_DIGEST,
        "applied_replay_lineage_count": 0,
        "complete_applied_replay_lineage_count": 0,
        "incomplete_applied_replay_lineage_count": 0,
        "lineage_material_hash": None,
        "raw_text_absent": True,
        "operator_identity_absent": True,
        "language_reasoning": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "issues_regeneration_permit": False,
    }


def ensure_applied_replay_lineage_summary_state(
    plasticity_state: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    raw_summary = plasticity_state.get("applied_replay_lineage_incremental_summary")
    if isinstance(raw_summary, MutableMapping) and raw_summary.get(
        "surface"
    ) == APPLIED_REPLAY_LINEAGE_INCREMENTAL_SURFACE:
        return raw_summary
    summary = _empty_incremental_summary()
    plasticity_state["applied_replay_lineage_incremental_summary"] = summary
    return summary


def _recompute_public_digest(summary: MutableMapping[str, Any]) -> None:
    count = int(summary.get("applied_replay_lineage_count", 0) or 0)
    complete = int(summary.get("complete_applied_replay_lineage_count", 0) or 0)
    incomplete = max(0, count - complete)
    summary["incomplete_applied_replay_lineage_count"] = incomplete
    if count <= 0:
        summary["lineage_material_hash"] = None
        summary["lineage_digest_xor"] = _ZERO_DIGEST
        summary["lineage_digest_sum_mod_2_256"] = _ZERO_DIGEST
        return
    summary["lineage_material_hash"] = _sha256_json(
        {
            "applied_replay_lineage_count": count,
            "complete_applied_replay_lineage_count": complete,
            "incomplete_applied_replay_lineage_count": incomplete,
            "lineage_digest_xor": str(summary.get("lineage_digest_xor") or _ZERO_DIGEST),
            "lineage_digest_sum_mod_2_256": str(
                summary.get("lineage_digest_sum_mod_2_256") or _ZERO_DIGEST
            ),
        }
    )


def _remove_lineage_row(
    summary: MutableMapping[str, Any],
    synapse_key: str,
) -> None:
    row_hashes = summary.setdefault("lineage_row_hash_by_synapse_key", {})
    complete_by_key = summary.setdefault("lineage_complete_by_synapse_key", {})
    if not isinstance(row_hashes, MutableMapping):
        row_hashes = {}
        summary["lineage_row_hash_by_synapse_key"] = row_hashes
    if not isinstance(complete_by_key, MutableMapping):
        complete_by_key = {}
        summary["lineage_complete_by_synapse_key"] = complete_by_key
    old_hash = row_hashes.pop(str(synapse_key), None)
    old_complete = bool(complete_by_key.pop(str(synapse_key), False))
    if old_hash is None:
        _recompute_public_digest(summary)
        return
    old_int = _digest_int(old_hash)
    summary["lineage_digest_xor"] = _digest_hex(
        _digest_int(summary.get("lineage_digest_xor")) ^ old_int
    )
    summary["lineage_digest_sum_mod_2_256"] = _digest_hex(
        _digest_int(summary.get("lineage_digest_sum_mod_2_256")) - old_int
    )
    summary["applied_replay_lineage_count"] = max(
        0,
        int(summary.get("applied_replay_lineage_count", 0) or 0) - 1,
    )
    if old_complete:
        summary["complete_applied_replay_lineage_count"] = max(
            0,
            int(summary.get("complete_applied_replay_lineage_count", 0) or 0) - 1,
        )
    _recompute_public_digest(summary)


def clear_applied_replay_lineage_provenance(
    plasticity_state: MutableMapping[str, Any],
    synapse_key: str,
) -> MutableMapping[str, Any]:
    summary = ensure_applied_replay_lineage_summary_state(plasticity_state)
    _remove_lineage_row(summary, str(synapse_key))
    return summary


def record_applied_replay_lineage_provenance(
    plasticity_state: MutableMapping[str, Any],
    synapse_key: str,
    provenance: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    summary = clear_applied_replay_lineage_provenance(
        plasticity_state,
        str(synapse_key),
    )
    material = _lineage_row_material(str(synapse_key), provenance)
    if material is None:
        return summary
    row_hash = _sha256_json(material)
    row_int = _digest_int(row_hash)
    complete = _row_complete(material)
    row_hashes = summary.setdefault("lineage_row_hash_by_synapse_key", {})
    complete_by_key = summary.setdefault("lineage_complete_by_synapse_key", {})
    if not isinstance(row_hashes, MutableMapping):
        row_hashes = {}
        summary["lineage_row_hash_by_synapse_key"] = row_hashes
    if not isinstance(complete_by_key, MutableMapping):
        complete_by_key = {}
        summary["lineage_complete_by_synapse_key"] = complete_by_key
    row_hashes[str(synapse_key)] = row_hash
    complete_by_key[str(synapse_key)] = bool(complete)
    summary["lineage_digest_xor"] = _digest_hex(
        _digest_int(summary.get("lineage_digest_xor")) ^ row_int
    )
    summary["lineage_digest_sum_mod_2_256"] = _digest_hex(
        _digest_int(summary.get("lineage_digest_sum_mod_2_256")) + row_int
    )
    summary["applied_replay_lineage_count"] = (
        int(summary.get("applied_replay_lineage_count", 0) or 0) + 1
    )
    if complete:
        summary["complete_applied_replay_lineage_count"] = (
            int(summary.get("complete_applied_replay_lineage_count", 0) or 0) + 1
        )
    _recompute_public_digest(summary)
    return summary


def applied_replay_lineage_checkpoint_summary(
    plasticity_state: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    raw_summary = plasticity_state.get("applied_replay_lineage_incremental_summary")
    summary_available = bool(
        isinstance(raw_summary, Mapping)
        and raw_summary.get("surface") == APPLIED_REPLAY_LINEAGE_INCREMENTAL_SURFACE
    )
    raw = raw_summary if summary_available else {}
    count = int(raw.get("applied_replay_lineage_count", 0) or 0)
    complete = int(raw.get("complete_applied_replay_lineage_count", 0) or 0)
    incomplete = int(
        raw.get(
            "incomplete_applied_replay_lineage_count",
            max(0, count - complete),
        )
        or 0
    )
    return {
        "surface": APPLIED_REPLAY_LINEAGE_CHECKPOINT_SUMMARY_SURFACE,
        "source": source,
        "summary_source_surface": (
            APPLIED_REPLAY_LINEAGE_INCREMENTAL_SURFACE
            if summary_available
            else None
        ),
        "summary_source_available": summary_available,
        "summary_policy": "mutation_time_o1_digest",
        "owned_by_marulho": True,
        "selection_criteria": [
            "replay-regenerated synapse provenance rows are counted at mutation time",
            "checkpoint save reads the maintained digest only",
            "missing incremental state blocks exact lineage validation instead of scanning provenance",
        ],
        "memory_budget": "O(1) checkpoint summary read; one CPU row hash per replay-regenerated synapse",
        "archival_metadata_device": "cpu",
        "gpu_used": False,
        "full_provenance_scan": False,
        "source_record_scan_count": 0,
        "applied_replay_lineage_count": count,
        "complete_applied_replay_lineage_count": complete,
        "incomplete_applied_replay_lineage_count": incomplete,
        "lineage_digest_xor": raw.get("lineage_digest_xor") if summary_available else _ZERO_DIGEST,
        "lineage_digest_sum_mod_2_256": (
            raw.get("lineage_digest_sum_mod_2_256")
            if summary_available
            else _ZERO_DIGEST
        ),
        "lineage_material_hash": (
            raw.get("lineage_material_hash") if summary_available else None
        ),
        "summary_unavailable_reason": (
            None if summary_available else "missing_incremental_lineage_summary"
        ),
        "raw_text_absent": True,
        "operator_identity_absent": True,
        "language_reasoning": False,
        "runs_replay": False,
        "applies_plasticity": False,
        "issues_regeneration_permit": False,
    }
