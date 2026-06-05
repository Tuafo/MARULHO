from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _format_scalar(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    text = str(value).replace("|", "/").replace("\n", " ")
    return text if len(text) <= 240 else f"{text[:237]}..."


def _human_title(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title() or "Report"


def render_json_report_markdown(payload: Mapping[str, Any], *, title: str | None = None) -> str:
    """Render a compact operator-facing Markdown view for a JSON report."""
    report_title = title or _human_title(str(payload.get("artifact_kind") or payload.get("benchmark") or "MARULHO Report"))
    lines = [
        f"# {report_title}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
    ]
    for key, value in payload.items():
        if isinstance(value, (Mapping, list, tuple)):
            continue
        lines.append(f"| {_human_title(str(key))} | {_format_scalar(value)} |")

    section_count = 0
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        scalar_items = {
            str(item_key): item_value
            for item_key, item_value in value.items()
            if not isinstance(item_value, (Mapping, list, tuple))
        }
        if not scalar_items:
            continue
        if section_count >= 8:
            break
        section_count += 1
        lines.extend(["", f"## {_human_title(str(key))}", "", "| Field | Value |", "|-------|-------|"])
        for item_key, item_value in scalar_items.items():
            lines.append(f"| {_human_title(item_key)} | {_format_scalar(item_value)} |")

    lines.extend(["", "## JSON Preview", "", "```json"])
    lines.append(json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False)[:4000])
    lines.extend(["```", ""])
    return "\n".join(lines)


def write_json_report_with_readme(
    output_path: str | Path,
    payload: Mapping[str, Any],
    *,
    title: str | None = None,
    indent: int = 2,
    sort_keys: bool = True,
) -> Path:
    """Write JSON report and a sibling README.md Markdown mirror."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=indent, sort_keys=sort_keys, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme_path = path.parent / "README.md"
    readme_path.write_text(render_json_report_markdown(payload, title=title), encoding="utf-8")
    return path
