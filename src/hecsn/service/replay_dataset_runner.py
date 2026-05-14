from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from hecsn.reporting.readme_reports import write_json_report_with_readme
from hecsn.service.manager import (
    DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
    HECSNServiceManager,
    MAX_REPLAY_DATASET_EXPORT_LIMIT,
)


_EMPTY_EXPORT_REASON = "checkpoint_contains_no_eligible_sanitized_runtime_traces"
_EMPTY_EXPORT_BEHAVIOR = "valid_empty_dataset_when_checkpoint_has_no_eligible_sanitized_runtime_traces"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a sanitized Terminus replay dataset preview from a HECSN checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="HECSN checkpoint to load.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Omit or pass '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        help=f"Maximum dataset items to export (1-{MAX_REPLAY_DATASET_EXPORT_LIMIT}).",
    )
    parser.add_argument(
        "--endpoint",
        "--type",
        dest="endpoint",
        type=str,
        default=None,
        help="Optional operation/endpoint filter, e.g. respond or /respond.",
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


def export_replay_dataset_preview(
    checkpoint_path: str | Path,
    *,
    limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
    endpoint: str | None = None,
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
        dataset = manager.runtime_facade.replay_dataset_preview(limit=limit, endpoint=endpoint)
    finally:
        manager.close()

    metadata: dict[str, Any] = {
        "source": "checkpoint_runtime_episode_traces_with_replay_context",
        "generated_by": "hecsn.service.replay_dataset_runner",
        "sanitization": "RuntimeFacade.replay_dataset_preview",
        "empty_export_behavior": _EMPTY_EXPORT_BEHAVIOR,
        "contains_items": bool(dataset.get("count", 0)),
        "preview_only": True,
        "training_started": False,
        "memory_mutated": False,
        "feedback_posted": False,
        "digital_action_executed": False,
        "external_calls_made": False,
    }
    for key in ("policy_decision", "replay_plan_summary", "replay_sample_summary", "safety_flags"):
        if isinstance(dataset.get(key), dict):
            metadata[key] = dict(dataset[key])
    if not metadata["contains_items"]:
        metadata["empty_reason"] = _EMPTY_EXPORT_REASON

    return {**dataset, "metadata": metadata}


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit < 1 or args.limit > MAX_REPLAY_DATASET_EXPORT_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_REPLAY_DATASET_EXPORT_LIMIT}")
    if args.indent < 0:
        parser.error("--indent must be non-negative")


def _write_json(payload: Mapping[str, Any], output: Path | None, *, indent: int, stdout: TextIO | None = None) -> None:
    encoded = json.dumps(payload, indent=indent, sort_keys=True) + "\n"
    if output is None or str(output) == "-":
        stream = stdout
        if stream is None:
            import sys

            stream = sys.stdout
        stream.write(encoded)
        return

    write_json_report_with_readme(
        output,
        payload,
        title="Replay Dataset Preview",
        indent=indent,
    )


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    dataset = export_replay_dataset_preview(
        args.checkpoint,
        limit=args.limit,
        endpoint=args.endpoint,
        trace_dir=args.trace_dir,
        env_root=args.env_root,
    )
    _write_json(dataset, args.output, indent=args.indent, stdout=stdout)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
