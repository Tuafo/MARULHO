from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from hecsn.service.manager import HECSNServiceManager
from hecsn.service.replay_dataset_bundle import (
    DEFAULT_REPLAY_DATASET_BUNDLE_RETENTION_DAYS,
    DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
    MAX_REPLAY_DATASET_EXPORT_LIMIT,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an operator-approved Terminus replay dataset bundle preview from a HECSN checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="HECSN checkpoint to load.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Omit or pass '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--operator-id",
        type=str,
        required=True,
        help="Operator identifier approving this package preview export.",
    )
    parser.add_argument(
        "--operator-note",
        type=str,
        default=None,
        help="Optional bounded operator note for the approval audit block.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that this is a preview/export package and not a training run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        help=f"Maximum preview items to package (1-{MAX_REPLAY_DATASET_EXPORT_LIMIT}).",
    )
    parser.add_argument(
        "--endpoint",
        "--type",
        dest="endpoint",
        type=str,
        default=None,
        help="Optional operation/endpoint filter, e.g. respond or /respond.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--retention-days",
        type=int,
        default=DEFAULT_REPLAY_DATASET_BUNDLE_RETENTION_DAYS,
    )
    parser.add_argument(
        "--decontamination-term",
        dest="decontamination_terms",
        action="append",
        default=None,
        help="Additional lowercase term that excludes matching examples from the bundle.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("reports") / "service" / "traces",
        help="Trace directory used while instantiating the service manager.",
    )
    parser.add_argument(
        "--env-root",
        type=Path,
        default=None,
        help="Optional runtime environment root. Defaults to checkpoint ancestry only.",
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation level.")
    return parser


def export_replay_dataset_bundle(
    checkpoint_path: str | Path,
    *,
    operator_id: str,
    confirmation: bool,
    operator_note: str | None = None,
    limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
    endpoint: str | None = None,
    holdout_fraction: float = 0.2,
    eval_fraction: float = 0.2,
    seed: int | None = None,
    retention_days: int = DEFAULT_REPLAY_DATASET_BUNDLE_RETENTION_DAYS,
    decontamination_terms: Sequence[str] | None = None,
    trace_dir: str | Path | None = None,
    env_root: str | Path | None = None,
) -> dict[str, Any]:
    manager = HECSNServiceManager(
        checkpoint_path,
        trace_history_limit=max(1, int(limit)),
        trace_dir=trace_dir,
        env_root=env_root,
    )
    try:
        bundle = manager.replay_dataset_bundle(
            operator_id=operator_id,
            operator_note=operator_note,
            confirmation=confirmation,
            limit=limit,
            endpoint=endpoint,
            holdout_fraction=holdout_fraction,
            eval_fraction=eval_fraction,
            seed=seed,
            retention_days=retention_days,
            decontamination_terms=decontamination_terms,
        )
    finally:
        manager.close()

    metadata: dict[str, Any] = {
        "source": "checkpoint_replay_dataset_preview_package_gate",
        "generated_by": "hecsn.service.replay_dataset_bundle_runner",
        "sanitization": "HECSNServiceManager.replay_dataset_bundle",
        "contains_items": bool(bundle.get("count", 0)),
        "operator_approved": bool(bundle.get("operator_approval", {}).get("approved", False))
        if isinstance(bundle.get("operator_approval"), Mapping)
        else False,
        "preview_only": True,
        "training_started": False,
        "memory_mutated": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
    }
    for key in ("bundle_id", "bundle_version", "bundle_hash", "split_counts", "training_gate", "safety_flags"):
        if key in bundle:
            metadata[key] = bundle[key]
    return {**bundle, "metadata": metadata}


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit < 1 or args.limit > MAX_REPLAY_DATASET_EXPORT_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_REPLAY_DATASET_EXPORT_LIMIT}")
    if args.indent < 0:
        parser.error("--indent must be non-negative")
    if not args.confirm:
        parser.error("--confirm is required for operator-approved package preview export")


def _write_json(payload: Mapping[str, Any], output: Path | None, *, indent: int, stdout: TextIO | None = None) -> None:
    encoded = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    if output is None or str(output) == "-":
        stream = stdout
        if stream is None:
            import sys

            stream = sys.stdout
        stream.write(encoded)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    bundle = export_replay_dataset_bundle(
        args.checkpoint,
        operator_id=args.operator_id,
        operator_note=args.operator_note,
        confirmation=args.confirm,
        limit=args.limit,
        endpoint=args.endpoint,
        holdout_fraction=args.holdout_fraction,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        retention_days=args.retention_days,
        decontamination_terms=args.decontamination_terms,
        trace_dir=args.trace_dir,
        env_root=args.env_root,
    )
    _write_json(bundle, args.output, indent=args.indent, stdout=stdout)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
