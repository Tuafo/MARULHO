"""Validate the MARULHO docs/vault Markdown workflow."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT / "docs" / "vault"
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
WIKI_RE = re.compile(r"\[\[([^\]|#]+)")


def has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") and "\n---\n" in text[4:]


def strip_anchor(target: str) -> str:
    return target.split("#", 1)[0].strip()


def resolve_link(source: Path, raw: str) -> Path | None:
    target = strip_anchor(raw)
    if not target or "://" in target or target.startswith("mailto:"):
        return None
    target = target.replace("%20", " ")
    candidate = (source.parent / target).resolve()
    return candidate


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    if not VAULT.exists():
        errors.append("docs/vault does not exist")
        print("\n".join(errors))
        return 1

    notes = sorted(VAULT.rglob("*.md"))
    note_names = {p.stem: p for p in notes}
    required = [
        VAULT / "index.md",
        VAULT / "generated" / "README.md",
        VAULT / "concepts" / "index.md",
        VAULT / "modules" / "index.md",
        VAULT / "papers" / "index.md",
        VAULT / "adrs" / "index.md",
        VAULT / "benchmarks" / "index.md",
        VAULT / "capabilities" / "index.md",
        VAULT / "retired" / "index.md",
        VAULT / "questions" / "index.md",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing required file: {path.relative_to(ROOT)}")

    for note in notes:
        text = note.read_text(encoding="utf-8")
        if not has_frontmatter(text):
            errors.append(f"missing frontmatter: {note.relative_to(ROOT)}")
        if note.is_relative_to(VAULT / "generated") and "type: generated" not in text[:300]:
            errors.append(f"generated note missing generated type: {note.relative_to(ROOT)}")

        outgoing = 0
        for match in LINK_RE.finditer(text):
            raw = match.group(1)
            target = resolve_link(note, raw)
            if target is None:
                continue
            outgoing += 1
            if target.suffix == "":
                if not target.exists() and not target.with_suffix(".md").exists():
                    errors.append(f"broken link: {note.relative_to(ROOT)} -> {raw}")
            elif not target.exists():
                errors.append(f"broken link: {note.relative_to(ROOT)} -> {raw}")

        for match in WIKI_RE.finditer(text):
            outgoing += 1
            name = match.group(1).strip()
            if name not in note_names:
                errors.append(f"unresolved wikilink: {note.relative_to(ROOT)} -> {name}")

        if note.parent.name == "concepts" and note.name != "index.md" and outgoing == 0:
            warnings.append(f"concept note has no links: {note.relative_to(ROOT)}")

    if errors:
        print("Vault validation failed:")
        for error in errors:
            print(f"- {error}")
    else:
        print(f"Vault validation passed: {len(notes)} markdown notes checked.")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
