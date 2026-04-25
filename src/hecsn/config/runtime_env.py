from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def _normalize_anchor(anchor: str | Path | None) -> Path | None:
    if anchor is None:
        return None
    path = Path(anchor).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.exists() and path.is_file():
        return path.parent
    if path.suffix:
        return path.parent
    return path


def _candidate_env_paths(anchor_paths: Iterable[str | Path | None]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_anchor in anchor_paths:
        anchor = _normalize_anchor(raw_anchor)
        if anchor is None:
            continue
        for directory in (anchor, *anchor.parents):
            env_path = directory / ".env"
            if env_path in seen:
                continue
            seen.add(env_path)
            candidates.append(env_path)
    return candidates


def find_runtime_env(*, anchor_paths: Iterable[str | Path | None]) -> Path | None:
    for candidate in _candidate_env_paths(anchor_paths):
        if candidate.is_file():
            return candidate
    return None


def load_runtime_env(
    *,
    anchor_paths: Iterable[str | Path | None],
    override: bool = False,
) -> dict[str, Any]:
    """Load a project runtime `.env` file deterministically.

    We avoid `python-dotenv`'s implicit `find_dotenv()` behavior because it can
    behave inconsistently in interactive contexts. Instead we search explicit
    anchor paths and their parents for the nearest `.env` file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return {
            "dotenv_available": False,
            "dotenv_loaded": False,
            "dotenv_path": None,
            "reason": "python-dotenv-unavailable",
        }

    env_path = find_runtime_env(anchor_paths=anchor_paths)
    if env_path is None:
        return {
            "dotenv_available": True,
            "dotenv_loaded": False,
            "dotenv_path": None,
            "reason": "not-found",
        }

    loaded = bool(load_dotenv(dotenv_path=env_path, override=override))
    return {
        "dotenv_available": True,
        "dotenv_loaded": loaded,
        "dotenv_path": str(env_path),
        "reason": "loaded" if loaded else "already-present-or-empty",
    }
