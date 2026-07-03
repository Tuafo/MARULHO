from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from marulho.brain import MarulhoBrain
from marulho.reporting.readme_reports import write_json_report_with_readme


DEFAULT_BRAIN_TRACE_EXPORT_LIMIT = 20
MAX_BRAIN_TRACE_EXPORT_LIMIT = 50
BRAIN_TRACE_EXPORT_SCHEMA_VERSION = 1
_EMPTY_EXPORT_REASON = "checkpoint_contains_no_persisted_brain_traces"
_EMPTY_EXPORT_BEHAVIOR = "valid_empty_dataset_when_checkpoint_has_no_persisted_brain_traces"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export compact BrainTrace examples from a MARULHO checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="MARULHO checkpoint to load.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Omit or pass '-' to write JSON to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_BRAIN_TRACE_EXPORT_LIMIT,
        help=f"Maximum examples to export (1-{MAX_BRAIN_TRACE_EXPORT_LIMIT}).",
    )
    parser.add_argument(
        "--endpoint",
        "--type",
        dest="endpoint",
        type=str,
        default=None,
        help="Optional BrainTrace event filter, e.g. tick, generate, replay, save.",
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


def export_runtime_trace_dataset(
    checkpoint_path: str | Path,
    *,
    limit: int = DEFAULT_BRAIN_TRACE_EXPORT_LIMIT,
    endpoint: str | None = None,
    trace_dir: str | Path | None = None,
    env_root: str | Path | None = None,
) -> dict[str, Any]:
    del trace_dir, env_root
    count = min(MAX_BRAIN_TRACE_EXPORT_LIMIT, max(1, int(limit)))
    event_filter = _normalize_event_filter(endpoint)
    brain = MarulhoBrain.load(checkpoint_path, trace_limit=max(MAX_BRAIN_TRACE_EXPORT_LIMIT, count))
    traces = [
        trace
        for trace in brain.trace_history(limit=MAX_BRAIN_TRACE_EXPORT_LIMIT)
        if event_filter is None or str(trace.get("event", "")).lower() == event_filter
    ][:count]
    dataset: dict[str, Any] = {
        "export_kind": "marulho_brain_trace_dataset_preview",
        "schema_version": BRAIN_TRACE_EXPORT_SCHEMA_VERSION,
        "training_role": "brain_trace_dataset_preview_only_not_training",
        "description": (
            "Bounded compact BrainTrace events from MarulhoBrain. "
            "This export does not train a model or revive legacy service traces."
        ),
        "limit": count,
        "max_limit": MAX_BRAIN_TRACE_EXPORT_LIMIT,
        "event": event_filter,
        "count": len(traces),
        "examples": traces,
        "source_window": {
            "surface": "marulho_brain_trace_export_window.v1",
            "selection_policy": "newest_brain_traces_first_bounded_window",
            "source_trace_count_evaluated": len(traces),
            "returned_count": len(traces),
            "return_limit_reached": bool(len(traces) >= count),
            "filter": event_filter,
        },
    }

    metadata: dict[str, Any] = {
        "source": "checkpoint_brain_state_trace_history",
        "generated_by": "marulho.service.trace_export_runner",
        "sanitization": "BrainTrace.to_dict compact telemetry only",
        "empty_export_behavior": _EMPTY_EXPORT_BEHAVIOR,
        "contains_examples": bool(dataset.get("count", 0)),
    }
    if not metadata["contains_examples"]:
        metadata["empty_reason"] = _EMPTY_EXPORT_REASON

    return {**dataset, "metadata": metadata}


def _normalize_event_filter(endpoint: str | None) -> str | None:
    if endpoint is None:
        return None
    value = str(endpoint).strip().lower()
    if not value:
        return None
    value = value.lstrip("/")
    aliases = {
        "feed": "feed",
        "brain/feed": "feed",
        "tick": "tick",
        "brain/tick": "tick",
        "generate": "generate",
        "brain/generate": "generate",
        "replay": "replay",
        "brain/replay": "replay",
        "grow-prune": "grow_prune",
        "grow_prune": "grow_prune",
        "brain/grow-prune": "grow_prune",
        "checkpoint/save": "save",
        "brain/checkpoint/save": "save",
        "save": "save",
        "checkpoint/restore": "restore",
        "brain/checkpoint/restore": "restore",
        "restore": "restore",
        "start": "start",
        "brain/start": "start",
        "stop": "stop",
        "brain/stop": "stop",
    }
    return aliases.get(value, value)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.limit < 1 or args.limit > MAX_BRAIN_TRACE_EXPORT_LIMIT:
        parser.error(f"--limit must be between 1 and {MAX_BRAIN_TRACE_EXPORT_LIMIT}")
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
        title="Runtime Trace Export",
        indent=indent,
    )


def main(argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    dataset = export_runtime_trace_dataset(
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
