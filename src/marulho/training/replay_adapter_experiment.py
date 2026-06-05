from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping, Sequence, TextIO

from marulho.evaluation.replay_training_approval import (
    ISOLATED_ADAPTER_TRAINING_SCOPE,
    _canonical_json,
    _sha256_json,
    load_json_object,
    validate_replay_training_approval,
)


ISOLATED_EXPERIMENT_SCHEMA_VERSION = 1
ISOLATED_EXPERIMENT_ARTIFACT_KIND = "terminus_isolated_replay_adapter_training_experiment"
ADAPTER_MANIFEST_NAME = "adapter_manifest.json"
ADAPTER_DELTA_NAME = "adapter_delta.json"
COMPARISON_REPORT_NAME = "comparison_report.json"
UNSAFE_SIDE_EFFECT_FLAGS = (
    "memory_mutated",
    "feedback_posted",
    "digital_action_executed",
    "external_calls_made",
    "sleep_started",
)
RUNTIME_TRUTH_ORDER = {"failed": 0, "degraded": 1, "partial": 2, "alive": 3}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_commit(repo_root: Path | None = None) -> str:
    root = repo_root or _repo_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _ensure_isolated_output_dir(output_dir: str | Path, *, repo_root: Path | None = None) -> Path:
    root = (repo_root or _repo_root()).resolve()
    output = Path(output_dir).resolve()
    if "isolated" not in {part.lower() for part in output.parts}:
        raise ValueError("isolated adapter training output directory must include an isolated path segment.")
    forbidden_roots = [root / "src", root / "checkpoints", root / "reports" / "service", root / "MARULHO_UI"]
    for forbidden in forbidden_roots:
        try:
            output.relative_to(forbidden.resolve())
        except ValueError:
            continue
        raise ValueError(f"isolated adapter training output directory cannot be inside {forbidden}.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _hash_mapping(value: Mapping[str, Any]) -> str:
    return _sha256_json(value)


def _load_optional_report(path: str | Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    return load_json_object(path, label=label)


def _runtime_truth(report: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("status_runtime_truth_summary", "terminus_runtime_truth_summary", "final_runtime_truth", "runtime_truth"):
        value = report.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _runtime_truth_verdict(report: Mapping[str, Any]) -> str:
    truth = _runtime_truth(report)
    verdict = str(truth.get("verdict", "unknown"))
    return verdict if verdict in RUNTIME_TRUTH_ORDER else "unknown"


def _replay_safety_flags(report: Mapping[str, Any]) -> dict[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    truth = _runtime_truth(report)
    truth_safety = truth.get("safety_flags") if isinstance(truth.get("safety_flags"), Mapping) else {}
    if isinstance(truth_safety, Mapping) and isinstance(truth_safety.get("replay_safety"), Mapping):
        candidates.append(truth_safety["replay_safety"])
    for key in (
        "replay_dataset_summary",
        "replay_dataset_bundle_summary",
        "replay_executor_summary",
        "replay_sample_summary",
    ):
        value = report.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("safety_flags"), Mapping):
            candidates.append(value["safety_flags"])
    merged: dict[str, Any] = {}
    for candidate in candidates:
        merged.update(dict(candidate))
    return merged


def _unsafe_flags_increased(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    before_flags = _replay_safety_flags(before)
    after_flags = _replay_safety_flags(after)
    increased: dict[str, dict[str, Any]] = {}
    for name in UNSAFE_SIDE_EFFECT_FLAGS:
        if before_flags.get(name) is not True and after_flags.get(name) is True:
            increased[name] = {"before": before_flags.get(name), "after": after_flags.get(name)}
    if before_flags.get("eligible_for_training") is not True and after_flags.get("eligible_for_training") is True:
        increased["eligible_for_training"] = {
            "before": before_flags.get("eligible_for_training"),
            "after": after_flags.get("eligible_for_training"),
        }
    return increased


def _benchmark_success(report: Mapping[str, Any]) -> bool:
    if not report:
        return False
    if "success" in report:
        return bool(report.get("success"))
    timings = report.get("endpoint_timings")
    return isinstance(timings, list) and all(isinstance(item, Mapping) and item.get("success") for item in timings)


def _long_test_health(report: Mapping[str, Any]) -> str:
    verdict = str(report.get("health_verdict") or report.get("acceptance_verdict") or "unknown")
    if verdict in {"alive", "passed"}:
        return "alive"
    if verdict in {"degraded", "partial"}:
        return "degraded"
    if verdict in {"dead", "failed"}:
        return "failed"
    if int(report.get("total_errors", 0) or 0) > 0:
        return "failed"
    return "unknown"


def compare_before_after_reports(
    *,
    before_benchmark: Mapping[str, Any],
    after_benchmark: Mapping[str, Any],
    before_long_test: Mapping[str, Any],
    after_long_test: Mapping[str, Any],
    gate_report: Mapping[str, Any],
) -> dict[str, Any]:
    before_runtime_truth = _runtime_truth_verdict(before_benchmark)
    after_runtime_truth = _runtime_truth_verdict(after_benchmark)
    runtime_truth_regressed = (
        before_runtime_truth in RUNTIME_TRUTH_ORDER
        and after_runtime_truth in RUNTIME_TRUTH_ORDER
        and RUNTIME_TRUTH_ORDER[after_runtime_truth] < RUNTIME_TRUTH_ORDER[before_runtime_truth]
    )
    unsafe_increase = _unsafe_flags_increased(before_benchmark, after_benchmark)
    contamination_failed = False
    for check in gate_report.get("checks", []) if isinstance(gate_report.get("checks"), list) else []:
        if isinstance(check, Mapping) and check.get("name") == "decontamination":
            contamination_failed = check.get("passed") is not True

    checks = {
        "before_benchmark_success": _benchmark_success(before_benchmark),
        "after_benchmark_success": _benchmark_success(after_benchmark),
        "runtime_truth_no_regression": not runtime_truth_regressed,
        "long_test_no_regression": _long_test_health(after_long_test) != "failed"
        and _long_test_health(before_long_test) != "alive"
        or _long_test_health(after_long_test) == "alive",
        "decontamination_passed": not contamination_failed,
        "unsafe_replay_or_action_not_increased": not unsafe_increase,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "runtime_truth": {
            "before": before_runtime_truth,
            "after": after_runtime_truth,
            "regressed": runtime_truth_regressed,
        },
        "long_test_health": {
            "before": _long_test_health(before_long_test),
            "after": _long_test_health(after_long_test),
        },
        "unsafe_flag_increases": unsafe_increase,
        "claimed_improvement": {
            "claimed": False,
            "evidence": "This phase records an isolated artifact and comparison only; no improvement is claimed.",
        },
    }


def _adapter_delta(bundle: Mapping[str, Any], hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    splits = bundle.get("splits") if isinstance(bundle.get("splits"), Mapping) else {}
    item_ids: list[str] = []
    if isinstance(splits, Mapping):
        for values in splits.values():
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, Mapping):
                        item_ids.append(str(item.get("package_item_id") or item.get("target_id") or "unknown"))
    material = {"bundle_hash": _hash_mapping(bundle), "item_ids": sorted(item_ids), "hyperparameters": dict(hyperparameters)}
    return {
        "artifact_kind": "terminus_replay_adapter_delta_summary",
        "delta_hash": hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest(),
        "source_item_count": len(item_ids),
        "source_item_ids": sorted(item_ids),
        "weights_format": "metadata_only_deterministic_adapter_summary",
        "production_runtime_compatible": False,
    }


def run_isolated_replay_adapter_experiment(
    *,
    bundle: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    training_plan: Mapping[str, Any],
    approval: Mapping[str, Any],
    before_benchmark: Mapping[str, Any],
    after_benchmark: Mapping[str, Any],
    before_long_test: Mapping[str, Any],
    after_long_test: Mapping[str, Any],
    output_dir: str | Path,
    command: Sequence[str] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    validation = validate_replay_training_approval(
        approval,
        bundle,
        gate_report,
        expected_scope=ISOLATED_ADAPTER_TRAINING_SCOPE,
    )
    if training_plan.get("artifact_kind") != "terminus_replay_training_dry_run_plan":
        raise ValueError("training plan artifact kind is invalid.")
    plan_identity = training_plan.get("dataset_identity") if isinstance(training_plan.get("dataset_identity"), Mapping) else {}
    if plan_identity.get("bundle_hash") != validation["bundle_hash"]:
        raise ValueError("training plan bundle hash does not match approved bundle.")
    if plan_identity.get("gate_report_hash") != validation["gate_report_hash"]:
        raise ValueError("training plan gate report hash does not match approved gate report.")

    output = _ensure_isolated_output_dir(output_dir, repo_root=repo_root)
    params = dict(hyperparameters or {"epochs": 1, "learning_rate": 0.0, "seed": 0})
    adapter_delta = _adapter_delta(bundle, params)
    comparison = compare_before_after_reports(
        before_benchmark=before_benchmark,
        after_benchmark=after_benchmark,
        before_long_test=before_long_test,
        after_long_test=after_long_test,
        gate_report=gate_report,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    source_hashes = {
        "bundle": validation["bundle_hash"],
        "gate_report": validation["gate_report_hash"],
        "training_plan": _hash_mapping(training_plan),
        "approval": _hash_mapping(approval),
        "before_benchmark": _hash_mapping(before_benchmark),
        "after_benchmark": _hash_mapping(after_benchmark),
        "before_long_test": _hash_mapping(before_long_test),
        "after_long_test": _hash_mapping(after_long_test),
    }
    command_record = list(command or ["python", "-m", "marulho.training.replay_adapter_experiment"])
    manifest = {
        "schema_version": ISOLATED_EXPERIMENT_SCHEMA_VERSION,
        "artifact_kind": ISOLATED_EXPERIMENT_ARTIFACT_KIND,
        "created_at": created_at,
        "status": "completed_isolated_artifact_no_production_switch",
        "adapter": {
            "name": training_plan.get("target_adapter", {}).get("name")
            if isinstance(training_plan.get("target_adapter"), Mapping)
            else "isolated-replay-adapter",
            "path": str(output),
            "delta_file": ADAPTER_DELTA_NAME,
            "production_runtime_target": False,
            "production_runtime_switched": False,
        },
        "approval": {
            "approval_id": approval.get("approval_id"),
            "operator_id": validation["operator_id"],
            "scope": ISOLATED_ADAPTER_TRAINING_SCOPE,
            "expires_at": validation["expires_at"],
        },
        "command": command_record,
        "git_commit": _git_commit(repo_root),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "source_hashes": source_hashes,
        "hyperparameters": params,
        "wall_time_seconds": round(time.perf_counter() - start, 6),
        "comparison_report": COMPARISON_REPORT_NAME,
        "rollback": {
            "production_runtime_changed": False,
            "rollback_path": "Delete this isolated artifact directory; no runtime configuration was changed.",
        },
        "side_effects": {
            "training_started": True,
            "memory_mutated": False,
            "feedback_posted": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "sleep_started": False,
            "production_runtime_switched": False,
        },
    }

    (output / ADAPTER_DELTA_NAME).write_text(json.dumps(adapter_delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / COMPARISON_REPORT_NAME).write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / ADAPTER_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def run_isolated_replay_adapter_experiment_files(
    *,
    bundle_path: str | Path,
    gate_report_path: str | Path,
    training_plan_path: str | Path,
    approval_path: str | Path,
    before_benchmark_path: str | Path,
    after_benchmark_path: str | Path,
    before_long_test_path: str | Path,
    after_long_test_path: str | Path,
    output_dir: str | Path,
    hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return run_isolated_replay_adapter_experiment(
        bundle=load_json_object(bundle_path, label="Replay bundle"),
        gate_report=load_json_object(gate_report_path, label="Replay gate report"),
        training_plan=load_json_object(training_plan_path, label="Replay training plan"),
        approval=load_json_object(approval_path, label="Isolated adapter training approval"),
        before_benchmark=load_json_object(before_benchmark_path, label="Before benchmark report"),
        after_benchmark=load_json_object(after_benchmark_path, label="After benchmark report"),
        before_long_test=load_json_object(before_long_test_path, label="Before long-test report"),
        after_long_test=load_json_object(after_long_test_path, label="After long-test report"),
        output_dir=output_dir,
        hyperparameters=hyperparameters,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated replay adapter training experiment artifact.")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--gate-report", type=Path, required=True)
    parser.add_argument("--training-plan", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--before-benchmark", type=Path, required=True)
    parser.add_argument("--after-benchmark", type=Path, required=True)
    parser.add_argument("--before-long-test", type=Path, required=True)
    parser.add_argument("--after-long-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    manifest = run_isolated_replay_adapter_experiment_files(
        bundle_path=args.bundle,
        gate_report_path=args.gate_report,
        training_plan_path=args.training_plan,
        approval_path=args.approval,
        before_benchmark_path=args.before_benchmark,
        after_benchmark_path=args.after_benchmark,
        before_long_test_path=args.before_long_test,
        after_long_test_path=args.after_long_test,
        output_dir=args.output_dir,
        hyperparameters={"epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed},
    )
    encoded = json.dumps(manifest, indent=args.indent, sort_keys=True) + "\n"
    stream = stdout
    if stream is None:
        import sys

        stream = sys.stdout
    stream.write(encoded)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
