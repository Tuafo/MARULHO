from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from .replay_training_approval import (
    ALLOWED_APPROVAL_SCOPE,
    load_json_object,
    validate_replay_training_approval,
)


TRAINING_PLAN_SCHEMA_VERSION = 1
TRAINING_PLAN_ARTIFACT_KIND = "terminus_replay_training_dry_run_plan"


def _splits(bundle: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    values = bundle.get("splits")
    result: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "eval": []}
    if not isinstance(values, Mapping):
        return result
    for name in result:
        split = values.get(name)
        if isinstance(split, list):
            result[name] = [dict(item) for item in split if isinstance(item, Mapping)]
    return result


def _contamination_result(gate_report: Mapping[str, Any]) -> dict[str, Any]:
    checks = gate_report.get("checks")
    if not isinstance(checks, list):
        return {"status": "unknown", "blocked_terms": [], "matches": {}}
    for check in checks:
        if isinstance(check, Mapping) and check.get("name") == "decontamination":
            details = check.get("details") if isinstance(check.get("details"), Mapping) else {}
            return {
                "status": "passed" if check.get("passed") is True else "failed",
                "blocked_terms": details.get("blocked_terms", []),
                "matches": details.get("matches", {}),
            }
    return {"status": "unknown", "blocked_terms": [], "matches": {}}


def build_replay_training_plan(
    bundle: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    validation = validate_replay_training_approval(approval, bundle, gate_report)
    if gate_report.get("passed") is not True:
        raise ValueError("replay training gate report must be passing before dry-run planning.")
    if gate_report.get("eligible_for_training") is not False:
        raise ValueError("replay training gate report must remain ineligible for training.")

    splits = _splits(bundle)
    split_counts = {name: len(items) for name, items in splits.items()}
    bundle_id = str(bundle.get("bundle_id") or "unknown_bundle")
    bundle_version = str(bundle.get("bundle_version") or "unknown_version")
    adapter_name = f"{bundle_id}-{bundle_version}-dry-run-adapter".replace(" ", "_")
    adapter_path = f"artifacts/replay_training/isolated/{adapter_name}"
    bundle_path = "<bundle.json>"
    holdout_path = "<holdout.json>"
    eval_path = "<eval.json>"

    return {
        "schema_version": TRAINING_PLAN_SCHEMA_VERSION,
        "artifact_kind": TRAINING_PLAN_ARTIFACT_KIND,
        "generated_at": approval.get("created_at"),
        "status": "dry_run_plan_only_not_executable",
        "approval": {
            "approval_id": approval.get("approval_id"),
            "operator_id": validation["operator_id"],
            "scope": ALLOWED_APPROVAL_SCOPE,
            "expires_at": validation["expires_at"],
        },
        "dataset_identity": {
            "bundle_id": bundle.get("bundle_id"),
            "bundle_version": bundle.get("bundle_version"),
            "bundle_hash": validation["bundle_hash"],
            "declared_bundle_hash": bundle.get("bundle_hash"),
            "source_preview_hash": bundle.get("source_preview_hash"),
            "gate_report_hash": validation["gate_report_hash"],
        },
        "split_counts": split_counts,
        "contamination_result": _contamination_result(gate_report),
        "target_adapter": {
            "name": adapter_name,
            "path": adapter_path,
            "production_runtime_target": False,
            "create_weights_in_this_phase": False,
        },
        "proposed_train_command": (
            "python -m hecsn.training.replay_adapter_experiment "
            f"--bundle {bundle_path} --output {adapter_path} --dry-run false"
        ),
        "proposed_eval_command": (
            "python -m pytest tests/test_replay_training_gate.py tests/test_service_benchmark.py "
            f"&& python -m hecsn.evaluation.service_benchmark --output {eval_path} "
            f"&& python -m hecsn.evaluation.replay_training_gate --input {bundle_path} --output {holdout_path}"
        ),
        "expected_cost_time": {
            "cost_usd": 0,
            "wall_time": "planner only; no training time incurred",
            "compute": "no model weights are created by this plan",
        },
        "benchmark_suite": {
            "before": [
                "replay holdout/eval split report",
                "contamination/decontamination report",
                "service benchmark",
                "long-test health",
                "runtime truth safety flags",
            ],
            "after": [
                "replay holdout/eval split report",
                "service benchmark",
                "long-test health",
                "runtime truth safety flags",
                "no-mutation safety assertions",
            ],
        },
        "rollback_path": "No production state is changed; remove this plan and any later isolated artifact directory.",
        "unresolved_risks": [
            "approval is limited to dry-run planning and cannot authorize training",
            "training implementation and isolated artifact directory are not created in this phase",
            "benchmark commands must be resolved against the operator environment before any later experiment",
        ],
        "side_effects": {
            "training_started": False,
            "memory_mutated": False,
            "adapter_created": False,
            "feedback_posted": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "sleep_started": False,
            "production_runtime_changed": False,
        },
        "executable": False,
    }


def create_replay_training_plan_file(
    bundle_path: str | Path,
    gate_report_path: str | Path,
    approval_path: str | Path,
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    bundle = load_json_object(bundle_path, label="Replay bundle")
    gate_report = load_json_object(gate_report_path, label="Replay gate report")
    approval = load_json_object(approval_path, label="Replay training approval")
    plan = build_replay_training_plan(bundle, gate_report, approval)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a deterministic dry-run replay training plan.")
    parser.add_argument("--bundle", type=Path, required=True, help="Replay dataset bundle JSON.")
    parser.add_argument("--gate-report", type=Path, required=True, help="Replay training gate report JSON.")
    parser.add_argument("--approval", type=Path, required=True, help="Dry-run replay training approval JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Dry-run training plan output JSON path.")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    plan = create_replay_training_plan_file(args.bundle, args.gate_report, args.approval, output_path=args.output)
    encoded = json.dumps(plan, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
