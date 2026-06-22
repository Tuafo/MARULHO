from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return dict(loaded)


def hash_json_file(path: str | Path, *, label: str) -> tuple[dict[str, Any], str]:
    loaded = load_json_object(path, label=label)
    return loaded, sha256_json(loaded)


_canonical_json = canonical_json
_sha256_json = sha256_json
