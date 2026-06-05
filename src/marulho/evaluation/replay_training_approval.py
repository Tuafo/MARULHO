from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.reporting.readme_reports import write_json_report_with_readme


APPROVAL_SCHEMA_VERSION = 1
ALLOWED_APPROVAL_SCOPE = "dry_run_training_plan_only"
ISOLATED_ADAPTER_TRAINING_SCOPE = "isolated_adapter_training"
EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE = "experimental_adapter_promotion"
ALLOWED_APPROVAL_SCOPES = (
    ALLOWED_APPROVAL_SCOPE,
    ISOLATED_ADAPTER_TRAINING_SCOPE,
    EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
)
APPROVAL_INTENDED_TARGETS = {
    ALLOWED_APPROVAL_SCOPE: "dry_run_replay_training_plan",
    ISOLATED_ADAPTER_TRAINING_SCOPE: "isolated_replay_adapter_training_artifact",
    EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE: "experimental_replay_adapter_promotion_gate",
}
DEFAULT_APPROVAL_TTL_HOURS = 24
APPROVAL_ARTIFACT_KIND = "terminus_replay_training_operator_approval"
REQUIRED_SAFETY_ACKNOWLEDGEMENTS = (
    "dry_run_only",
    "no_training_started",
    "no_memory_mutation",
    "no_adapter_creation",
    "no_feedback_posting",
    "no_action_execution",
    "no_sleep_or_external_calls",
    "production_runtime_unchanged",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _parse_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 datetime.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return dict(loaded)


def hash_json_file(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    loaded = load_json_object(path, label=label)
    return loaded, _sha256_json(loaded)


def build_replay_training_approval(
    bundle: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    *,
    operator_id: str,
    scope: str = ALLOWED_APPROVAL_SCOPE,
    intended_target: str | None = None,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a read-only operator approval artifact for dry-run planning only."""

    normalized_operator_id = operator_id.strip()
    if not normalized_operator_id:
        raise ValueError("operator_id is required.")
    if scope not in ALLOWED_APPROVAL_SCOPES:
        raise ValueError(f"approval scope must be one of {ALLOWED_APPROVAL_SCOPES!r}.")
    normalized_intended_target = intended_target or APPROVAL_INTENDED_TARGETS[scope]
    if normalized_intended_target != APPROVAL_INTENDED_TARGETS[scope]:
        raise ValueError("approval intended_target does not match approval scope.")

    created = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = (expires_at or created + timedelta(hours=DEFAULT_APPROVAL_TTL_HOURS)).astimezone(timezone.utc)
    if expires <= created:
        raise ValueError("approval expiry must be after creation time.")

    bundle_hash = _sha256_json(bundle)
    gate_report_hash = _sha256_json(gate_report)
    source_bundle = gate_report.get("source_bundle") if isinstance(gate_report.get("source_bundle"), Mapping) else {}
    declared_bundle_hash = bundle.get("bundle_hash")
    reported_bundle_hash = source_bundle.get("bundle_hash")
    if declared_bundle_hash and reported_bundle_hash and declared_bundle_hash != reported_bundle_hash:
        raise ValueError("bundle and gate report refer to different bundle hashes.")

    safety_acknowledgements = {name: True for name in REQUIRED_SAFETY_ACKNOWLEDGEMENTS}
    approval_material = {
        "bundle_hash": bundle_hash,
        "created_at": created.isoformat(),
        "gate_report_hash": gate_report_hash,
        "intended_target": normalized_intended_target,
        "operator_id": normalized_operator_id,
        "scope": scope,
    }
    approval_id = hashlib.sha256(_canonical_json(approval_material).encode("utf-8")).hexdigest()

    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "artifact_kind": APPROVAL_ARTIFACT_KIND,
        "approval_id": approval_id,
        "created_at": created.isoformat(),
        "expires_at": expires.isoformat(),
        "operator_id": normalized_operator_id,
        "scope": scope,
        "intended_target": normalized_intended_target,
        "bundle_hash": bundle_hash,
        "gate_report_hash": gate_report_hash,
        "bundle_identity": {
            "bundle_id": bundle.get("bundle_id"),
            "bundle_version": bundle.get("bundle_version"),
            "declared_bundle_hash": declared_bundle_hash,
            "source_preview_hash": bundle.get("source_preview_hash"),
        },
        "gate_report_identity": {
            "gate_name": gate_report.get("gate_name"),
            "status": gate_report.get("status"),
            "passed": bool(gate_report.get("passed")),
            "eligible_for_training": bool(gate_report.get("eligible_for_training")),
            "source_bundle_hash": reported_bundle_hash,
        },
        "safety_acknowledgements": safety_acknowledgements,
        "rollback_note": "No production state is changed by this approval; delete the approval and generated dry-run plan to roll back.",
        "read_only": True,
        "training_started": False,
        "memory_mutated": False,
        "adapter_created": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
        "sleep_started": False,
    }


def validate_replay_training_approval(
    approval: Mapping[str, Any],
    bundle: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    *,
    now: datetime | None = None,
    expected_scope: str = ALLOWED_APPROVAL_SCOPE,
    expected_intended_target: str | None = None,
) -> dict[str, Any]:
    """Fail closed unless approval exactly matches bundle/report and expected scope."""

    if approval.get("artifact_kind") != APPROVAL_ARTIFACT_KIND:
        raise ValueError("approval artifact kind is invalid.")
    if int(approval.get("schema_version", 0) or 0) != APPROVAL_SCHEMA_VERSION:
        raise ValueError("approval schema version is invalid.")
    if str(approval.get("operator_id", "")).strip() == "":
        raise ValueError("approval operator_id is required.")
    if expected_scope not in ALLOWED_APPROVAL_SCOPES:
        raise ValueError(f"expected approval scope must be one of {ALLOWED_APPROVAL_SCOPES!r}.")
    if approval.get("scope") != expected_scope:
        raise ValueError(f"approval scope must be {expected_scope!r}.")
    required_target = expected_intended_target or APPROVAL_INTENDED_TARGETS[expected_scope]
    if approval.get("intended_target") != required_target:
        raise ValueError("approval intended_target is not valid for this replay training step.")

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires_at = _parse_datetime(str(approval.get("expires_at", "")), "expires_at")
    if expires_at <= current:
        raise ValueError("approval has expired.")

    bundle_hash = _sha256_json(bundle)
    gate_report_hash = _sha256_json(gate_report)
    if approval.get("bundle_hash") != bundle_hash:
        raise ValueError("approval bundle hash does not match bundle.")
    if approval.get("gate_report_hash") != gate_report_hash:
        raise ValueError("approval gate report hash does not match gate report.")

    acknowledgements = approval.get("safety_acknowledgements")
    if not isinstance(acknowledgements, Mapping) or any(
        acknowledgements.get(name) is not True for name in REQUIRED_SAFETY_ACKNOWLEDGEMENTS
    ):
        raise ValueError("approval safety acknowledgements are incomplete.")

    for field in (
        "training_started",
        "memory_mutated",
        "adapter_created",
        "feedback_posted",
        "digital_action_executed",
        "external_calls_made",
        "sleep_started",
    ):
        if approval.get(field) is not False:
            raise ValueError(f"approval side-effect flag {field} must be false.")

    return {
        "bundle_hash": bundle_hash,
        "gate_report_hash": gate_report_hash,
        "expires_at": expires_at.isoformat(),
        "operator_id": approval.get("operator_id"),
        "scope": approval.get("scope"),
    }


def create_replay_training_approval_file(
    bundle_path: str | Path,
    gate_report_path: str | Path,
    *,
    operator_id: str,
    scope: str,
    output_path: str | Path,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> dict[str, Any]:
    bundle = load_json_object(bundle_path, label="Replay bundle")
    gate_report = load_json_object(gate_report_path, label="Replay gate report")
    approval = build_replay_training_approval(
        bundle,
        gate_report,
        operator_id=operator_id,
        scope=scope,
        created_at=created_at,
        expires_at=expires_at,
    )
    write_json_report_with_readme(
        output_path,
        approval,
        title="Replay Training Operator Approval",
    )
    return approval


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a dry-run-only replay training approval artifact.")
    parser.add_argument("--bundle", type=Path, required=True, help="Replay dataset bundle JSON.")
    parser.add_argument("--gate-report", type=Path, required=True, help="Replay training gate report JSON.")
    parser.add_argument("--operator-id", required=True, help="Operator id approving this dry-run planning artifact.")
    parser.add_argument("--scope", required=True, choices=list(ALLOWED_APPROVAL_SCOPES))
    parser.add_argument("--output", type=Path, required=True, help="Approval artifact JSON path.")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    approval = create_replay_training_approval_file(
        args.bundle,
        args.gate_report,
        operator_id=args.operator_id,
        scope=args.scope,
        output_path=args.output,
    )
    encoded = json.dumps(approval, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
