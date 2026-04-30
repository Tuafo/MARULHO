from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO


REPLAY_TRAINING_GATE_SCHEMA_VERSION = 1
OFFLINE_REPLAY_TRAINING_GATE_NAME = "offline_replay_to_learning_gate"
PASSING_STATUS = "passed_pending_operator_training_approval"
FAILING_STATUS = "failed_offline_replay_training_eval_gate"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _check(name: str, passed: bool, summary: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": str(name),
        "passed": bool(passed),
        "summary": str(summary),
        "details": dict(details or {}),
    }


def _bundle_splits(bundle: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    splits = bundle.get("splits")
    if not isinstance(splits, Mapping):
        return {"train": [], "holdout": [], "eval": []}
    result: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "eval": []}
    for name in result:
        values = splits.get(name)
        if isinstance(values, list):
            result[name] = [dict(item) for item in values if isinstance(item, Mapping)]
    return result


def _packaged_items(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [item for values in _bundle_splits(bundle).values() for item in values]


def _excluded_items(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = bundle.get("excluded_items")
    return [dict(item) for item in values if isinstance(item, Mapping)] if isinstance(values, list) else []


def _blocked_terms(bundle: Mapping[str, Any]) -> list[str]:
    packaging_policy = bundle.get("packaging_policy")
    if not isinstance(packaging_policy, Mapping):
        return []
    decontamination = packaging_policy.get("decontamination")
    if not isinstance(decontamination, Mapping):
        return []
    terms = decontamination.get("blocked_terms")
    if not isinstance(terms, list):
        return []
    return [str(term).lower() for term in terms if str(term).strip()]


def _scan_decontamination_terms(items: Sequence[Mapping[str, Any]], terms: Sequence[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {}
    for item in items:
        item_id = str(item.get("package_item_id") or item.get("source_item_id") or item.get("target_id") or "unknown")
        canonical = _canonical_json(item).lower()
        item_matches = [term for term in terms if term and term in canonical]
        if item_matches:
            matches[item_id] = item_matches[:8]
    return matches


def evaluate_replay_training_gate(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a saved replay dataset bundle without training or mutating runtime state."""

    splits = _bundle_splits(bundle)
    packaged_items = _packaged_items(bundle)
    excluded_items = _excluded_items(bundle)
    manifest = bundle.get("manifest") if isinstance(bundle.get("manifest"), Mapping) else {}
    training_gate = bundle.get("training_gate") if isinstance(bundle.get("training_gate"), Mapping) else {}
    safety_flags = bundle.get("safety_flags") if isinstance(bundle.get("safety_flags"), Mapping) else {}
    split_counts = bundle.get("split_counts") if isinstance(bundle.get("split_counts"), Mapping) else {}

    item_hashes = [str(item.get("dedupe_fingerprint")) for item in packaged_items if item.get("dedupe_fingerprint")]
    excluded_hashes = [str(item.get("dedupe_fingerprint")) for item in excluded_items if item.get("dedupe_fingerprint")]
    manifest_item_hashes = [str(value) for value in manifest.get("item_hashes", [])] if isinstance(manifest.get("item_hashes"), list) else []
    manifest_excluded_hashes = (
        [str(value) for value in manifest.get("excluded_hashes", [])]
        if isinstance(manifest.get("excluded_hashes"), list)
        else []
    )
    actual_split_counts = {name: len(values) for name, values in splits.items()}
    blocked_terms = _blocked_terms(bundle)
    decontamination_matches = _scan_decontamination_terms(packaged_items, blocked_terms)

    no_side_effect_flags = {
        "training_started": False,
        "memory_mutated": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
        "sleep_started": False,
        "eligible_for_training": False,
    }
    unsafe_flags = {
        name: safety_flags.get(name)
        for name, expected in no_side_effect_flags.items()
        if safety_flags.get(name) is not expected
    }

    duplicate_hashes = sorted({value for value in item_hashes if item_hashes.count(value) > 1})
    split_mismatches = {
        name: {"declared": int(split_counts.get(name, -1) or 0), "actual": actual_split_counts[name]}
        for name in actual_split_counts
        if int(split_counts.get(name, -1) or 0) != actual_split_counts[name]
    }

    checks = [
        _check(
            "bundle_schema",
            int(bundle.get("schema_version", 0) or 0) >= 1
            and str(bundle.get("export_kind", "")) == "terminus_replay_dataset_bundle_preview",
            "Bundle is a replay dataset bundle preview.",
            {"schema_version": bundle.get("schema_version"), "export_kind": bundle.get("export_kind")},
        ),
        _check(
            "manifest_integrity",
            bool(manifest.get("bundle_hash"))
            and str(manifest.get("artifact_role", "")) == "preview_export_only_not_training"
            and manifest_item_hashes == item_hashes
            and manifest_excluded_hashes == excluded_hashes,
            "Manifest records the packaged and excluded item fingerprints.",
            {
                "bundle_hash_present": bool(manifest.get("bundle_hash")),
                "artifact_role": manifest.get("artifact_role"),
                "item_hash_count": len(item_hashes),
                "excluded_hash_count": len(excluded_hashes),
            },
        ),
        _check(
            "dedupe",
            not duplicate_hashes,
            "Packaged item fingerprints are unique.",
            {"duplicate_hashes": duplicate_hashes},
        ),
        _check(
            "split_counts",
            not split_mismatches,
            "Declared train/holdout/eval split counts match the packaged items.",
            {"declared": dict(split_counts), "actual": actual_split_counts, "mismatches": split_mismatches},
        ),
        _check(
            "decontamination",
            not decontamination_matches,
            "Packaged items do not contain configured blocked decontamination terms.",
            {"blocked_terms": blocked_terms, "matches": decontamination_matches},
        ),
        _check(
            "no_training_side_effects",
            not unsafe_flags,
            "Bundle safety flags keep the artifact preview-only and non-mutating.",
            {"unsafe_flags": unsafe_flags},
        ),
        _check(
            "training_gate_blocked",
            str(training_gate.get("status", "")) == "blocked_preview_only"
            and training_gate.get("eligible_for_training") is False,
            "Source bundle remains blocked from training before this offline evaluation report.",
            {
                "status": training_gate.get("status"),
                "eligible_for_training": training_gate.get("eligible_for_training"),
            },
        ),
    ]

    passed = sum(1 for item in checks if bool(item["passed"]))
    failed = len(checks) - passed
    offline_checks_passed = failed == 0
    status = PASSING_STATUS if offline_checks_passed else FAILING_STATUS
    unsatisfied_conditions = ["explicit_operator_training_approval"]
    if not offline_checks_passed:
        unsatisfied_conditions.insert(0, "offline_regression_benchmark")

    return {
        "schema_version": REPLAY_TRAINING_GATE_SCHEMA_VERSION,
        "gate_name": OFFLINE_REPLAY_TRAINING_GATE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "passed": offline_checks_passed,
        "eligible_for_training": False,
        "training_started": False,
        "memory_mutated": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
        "source_bundle": {
            "bundle_id": bundle.get("bundle_id"),
            "bundle_version": bundle.get("bundle_version"),
            "bundle_hash": bundle.get("bundle_hash"),
            "source_preview_hash": bundle.get("source_preview_hash"),
            "count": int(bundle.get("count", 0) or 0),
            "split_counts": actual_split_counts,
        },
        "checks": checks,
        "passed_checks": int(passed),
        "failed_checks": int(failed),
        "satisfied_conditions": {
            "versioned_dataset_manifest": bool(manifest.get("bundle_hash")),
            "dedupe_complete": not duplicate_hashes,
            "train_holdout_eval_split": not split_mismatches,
            "decontamination_check": not decontamination_matches,
            "offline_regression_benchmark": offline_checks_passed,
            "explicit_operator_training_approval": False,
        },
        "unsatisfied_conditions": unsatisfied_conditions,
        "next_action": "request_explicit_operator_training_approval" if offline_checks_passed else "fix_bundle_and_rerun_gate",
        "advisory": True,
        "executable": False,
    }


def evaluate_replay_training_gate_file(input_path: str | Path, *, output_path: str | Path | None = None) -> dict[str, Any]:
    bundle_path = Path(input_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(bundle, Mapping):
        raise ValueError("Replay training gate input must be a JSON object.")
    report = evaluate_replay_training_gate(bundle)
    report["input_path"] = str(bundle_path)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["output_path"] = str(output)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved replay dataset bundle before any training path can use it.")
    parser.add_argument("--input", type=Path, required=True, help="Saved replay dataset bundle JSON.")
    parser.add_argument("--output", type=Path, default=None, help="Optional gate report JSON path.")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    report = evaluate_replay_training_gate_file(args.input, output_path=args.output)
    encoded = json.dumps(report, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0 if bool(report.get("passed")) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
