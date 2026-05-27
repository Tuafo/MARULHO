from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

from hecsn.data.corpus_loader import extract_web_text
from hecsn.semantics.grounding_text import salient_query_terms


_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".html",
        ".css",
        ".scss",
        ".ini",
        ".cfg",
        ".log",
        ".rst",
    }
)
_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        "coverage",
        ".pytest_cache",
        "__pycache__",
    }
)
_IGNORED_FILE_NAMES: frozenset[str] = frozenset(
    {
    }
)
_IGNORED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".pt",
    }
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _action_id(action_type: str, payload: Mapping[str, Any], *, recorded_at: str) -> str:
    seed = f"{action_type}|{recorded_at}|{dict(payload)}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"act-{digest}"


def _workspace_terms(query_text: str) -> tuple[str, ...]:
    normalized = _normalize_text(query_text).lower()
    if not normalized:
        return ()
    terms = [term.lower() for term in salient_query_terms(normalized) if term]
    if not terms:
        terms = [
            token.lower()
            for token in re.findall(r"[a-zA-Z0-9_./:-]+", normalized)
            if len(token) >= 2
        ]
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        compact = _normalize_text(term).lower()
        if not compact or compact in seen:
            continue
        deduped.append(compact)
        seen.add(compact)
    return tuple(deduped[:6])


def _iter_workspace_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except Exception:
            continue
        for child in children:
            if child.name in _IGNORED_DIR_NAMES:
                continue
            if child.is_dir():
                stack.append(child)
                continue
            if not child.is_file():
                continue
            if child.name in _IGNORED_FILE_NAMES:
                continue
            suffix = child.suffix.lower()
            if suffix in _IGNORED_SUFFIXES:
                continue
            if suffix and suffix not in _TEXT_SUFFIXES:
                continue
            yield child


def _resolve_workspace_path(root: Path, raw_path: str) -> Path | None:
    normalized = _normalize_text(raw_path).replace("\\", "/")
    if not normalized:
        return None
    candidate = Path(normalized)
    resolved = candidate if candidate.is_absolute() else (root / candidate)
    try:
        resolved = resolved.resolve()
    except Exception:
        return None
    if resolved != root and root not in resolved.parents:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    if resolved.name in _IGNORED_FILE_NAMES:
        return None
    suffix = resolved.suffix.lower()
    if suffix in _IGNORED_SUFFIXES:
        return None
    if suffix and suffix not in _TEXT_SUFFIXES:
        return None
    return resolved


def _dedupe_topics(values: Sequence[str], *, limit: int = 8) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_text(value).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max(1, int(limit)):
            break
    return tuple(deduped)


def _normalize_url(value: str) -> str:
    cleaned = _normalize_text(value).strip("`'\".,;:!?()[]{} ")
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return cleaned


def _normalize_http_method(value: Any) -> str:
    method = _normalize_text(value).upper()
    if not method:
        return "GET"
    if method in {"GET", "POST"}:
        return method
    return ""


def _normalize_json_path(value: Any) -> str:
    path = _normalize_text(value).replace(" ", "")
    if not path:
        return ""
    if path == "$":
        return "$"
    if not path.startswith("$"):
        if path.startswith(".") or path.startswith("["):
            path = f"${path}"
        else:
            path = f"$.{path}"
    return path


def _json_path_has_wildcards(path: str) -> bool:
    normalized = _normalize_json_path(path)
    return "[*]" in normalized or ".*" in normalized


def _json_path_pattern_regex(path: str) -> re.Pattern[str]:
    normalized = _normalize_json_path(path)
    escaped = re.escape(normalized)
    escaped = escaped.replace(r"\.\*", r"\.[^.\[\]]+")
    escaped = escaped.replace(r"\[\*\]", r"\[\d+\]")
    return re.compile(f"^{escaped}$")


def _json_path_matches(nodes: Mapping[str, Any], raw_path: str) -> list[tuple[str, Any]]:
    path = _normalize_json_path(raw_path)
    if not path:
        return []
    if not _json_path_has_wildcards(path):
        if path in nodes:
            return [(path, nodes[path])]
        return []
    pattern = _json_path_pattern_regex(path)
    matches = [(concrete_path, value) for concrete_path, value in nodes.items() if pattern.fullmatch(concrete_path)]
    matches.sort(key=lambda item: item[0])
    return matches


def _normalize_expected_json_paths(value: Any, *, limit: int = 16) -> tuple[tuple[str, ...], str | None]:
    if value is None:
        return (), None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return (), "API request expected_json_paths must be a list of JSON path strings."
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in list(value)[: max(1, int(limit))]:
        path = _normalize_json_path(raw)
        if not path:
            continue
        if path in seen:
            continue
        seen.add(path)
        normalized.append(path)
    return tuple(normalized), None


def _normalize_expected_response_shape(value: Any) -> tuple[str, str | None]:
    shape = _normalize_text(value).lower()
    if not shape:
        return "", None
    if shape in {"object", "array", "scalar", "null"}:
        return shape, None
    return "", "API request expected_response_shape must be one of: object, array, scalar, null."


def _normalize_expected_json_values(value: Any, *, limit: int = 16) -> tuple[dict[str, Any], str | None]:
    if value is None:
        return {}, None
    if not isinstance(value, Mapping):
        return {}, "API request expected_json_values must be a mapping of JSON paths to JSON values."
    normalized: dict[str, Any] = {}
    for raw_path, raw_expected in list(value.items())[: max(1, int(limit))]:
        path = _normalize_json_path(raw_path)
        if not path:
            continue
        try:
            encoded = json.dumps(raw_expected, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            expected_value = json.loads(encoded.decode("utf-8"))
        except Exception:
            return {}, f"API request expected_json_values entry for '{path}' must be valid JSON-serializable content."
        normalized[path] = expected_value
    return normalized, None


def _normalize_expected_json_predicates(value: Any, *, limit: int = 16) -> tuple[tuple[dict[str, Any], ...], str | None]:
    if value is None:
        return (), None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return (), "API request expected_json_predicates must be a list of predicate objects."
    normalized: list[dict[str, Any]] = []
    allowed_ops = {
        "contains",
        "regex",
        "gt",
        "gte",
        "lt",
        "lte",
        "between",
        "startswith",
        "endswith",
        "any_contains",
        "any_regex",
        "all_contains",
        "all_regex",
        "none_contains",
        "none_regex",
    }
    for index, raw_item in enumerate(list(value)[: max(1, int(limit))], start=1):
        if not isinstance(raw_item, Mapping):
            return (), f"API request expected_json_predicates entry {index} must be an object."
        path = _normalize_json_path(raw_item.get("path"))
        if not path:
            return (), f"API request expected_json_predicates entry {index} must include a valid JSON path."
        op = _normalize_text(raw_item.get("op")).lower()
        if op not in allowed_ops:
            return (), (
                f"API request expected_json_predicates entry {index} for '{path}' must use one of: "
                "contains, regex, gt, gte, lt, lte, between, startswith, endswith, any_contains, any_regex, all_contains, all_regex, none_contains, none_regex."
            )
        if "value" not in raw_item:
            return (), f"API request expected_json_predicates entry {index} for '{path}' must include a value."
        raw_expected = raw_item.get("value")
        if op in {"gt", "gte", "lt", "lte"}:
            if isinstance(raw_expected, bool) or not isinstance(raw_expected, (int, float)):
                return (), f"API request expected_json_predicates entry {index} for '{path}' must use a numeric value for '{op}'."
            expected_value: Any = float(raw_expected)
        elif op == "between":
            if not isinstance(raw_expected, Mapping):
                return (), f"API request expected_json_predicates entry {index} for '{path}' must use an object with numeric min/max for 'between'."
            has_min = "min" in raw_expected
            has_max = "max" in raw_expected
            if not has_min and not has_max:
                return (), f"API request expected_json_predicates entry {index} for '{path}' must provide at least one of min/max for 'between'."
            range_value: dict[str, float] = {}
            if has_min:
                lower = raw_expected.get("min")
                if isinstance(lower, bool) or not isinstance(lower, (int, float)):
                    return (), f"API request expected_json_predicates entry {index} for '{path}' must use a numeric min for 'between'."
                range_value["min"] = float(lower)
            if has_max:
                upper = raw_expected.get("max")
                if isinstance(upper, bool) or not isinstance(upper, (int, float)):
                    return (), f"API request expected_json_predicates entry {index} for '{path}' must use a numeric max for 'between'."
                range_value["max"] = float(upper)
            expected_value = range_value
        else:
            if op in {"regex", "any_regex", "all_regex", "none_regex", "startswith", "endswith", "any_contains", "all_contains", "none_contains"} and not _normalize_text(raw_expected):
                return (), f"API request expected_json_predicates entry {index} for '{path}' must use a non-empty value for '{op}'."
            try:
                encoded = json.dumps(raw_expected, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
                expected_value = json.loads(encoded.decode("utf-8"))
            except Exception:
                return (), f"API request expected_json_predicates entry {index} for '{path}' must use a JSON-serializable value."
        normalized.append({"path": path, "op": op, "value": expected_value})
    return tuple(normalized), None


def _normalize_expected_json_predicate_groups(
    value: Any,
    *,
    limit: int = 8,
    predicates_per_group: int = 8,
    groups_per_group: int = 8,
    depth: int = 0,
    max_depth: int = 4,
) -> tuple[tuple[dict[str, Any], ...], str | None]:
    if value is None:
        return (), None
    if depth >= max_depth:
        return (), "API request expected_json_predicate_groups exceeded the supported nesting depth."
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return (), "API request expected_json_predicate_groups must be a list of predicate-group objects."
    normalized_groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(list(value)[: max(1, int(limit))], start=1):
        if not isinstance(raw_group, Mapping):
            return (), f"API request expected_json_predicate_groups entry {index} must be an object."
        logic = _normalize_text(raw_group.get("logic")).lower()
        if logic not in {"all", "any", "none"}:
            return (), f"API request expected_json_predicate_groups entry {index} must use logic 'all', 'any', or 'none'."
        predicates, predicate_error = _normalize_expected_json_predicates(
            raw_group.get("predicates"),
            limit=predicates_per_group,
        )
        if predicate_error is not None:
            return (), f"API request expected_json_predicate_groups entry {index} is invalid: {predicate_error}"
        groups, group_error = _normalize_expected_json_predicate_groups(
            raw_group.get("groups"),
            limit=groups_per_group,
            predicates_per_group=predicates_per_group,
            groups_per_group=groups_per_group,
            depth=depth + 1,
            max_depth=max_depth,
        )
        if group_error is not None:
            return (), f"API request expected_json_predicate_groups entry {index} is invalid: {group_error}"
        if not predicates and not groups:
            return (), f"API request expected_json_predicate_groups entry {index} must include at least one predicate or nested group."
        normalized_groups.append(
            {
                "logic": logic,
                "predicates": [dict(item) for item in predicates],
                "groups": [dict(item) for item in groups],
            }
        )
    return tuple(normalized_groups), None


def _normalize_request_params(value: Any, *, limit: int = 16) -> tuple[dict[str, str], str | None]:
    if value is None:
        return {}, None
    if not isinstance(value, Mapping):
        return {}, "API request params must be a mapping of scalar values."
    params: dict[str, str] = {}
    for raw_key, raw_value in list(value.items())[: max(1, int(limit))]:
        key = _normalize_text(raw_key)
        if not key:
            continue
        if raw_value is None:
            params[key] = ""
            continue
        if isinstance(raw_value, bool):
            params[key] = "true" if raw_value else "false"
            continue
        if isinstance(raw_value, (int, float)):
            params[key] = str(raw_value)
            continue
        if isinstance(raw_value, str):
            params[key] = _normalize_text(raw_value)
            continue
        return {}, f"API request param '{key}' must be a scalar JSON value."
    return params, None


def _serialize_json_request_body(value: Any, *, max_bytes: int = 64_000) -> tuple[Any | None, bytes | None, str | None]:
    if value is None:
        return None, None, None
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        normalized = json.loads(encoded.decode("utf-8"))
    except Exception:
        return None, None, "API request json_body must be valid JSON-serializable content."
    if len(encoded) > max(256, int(max_bytes)):
        return None, None, "API request json_body exceeded the bounded request-body budget."
    return normalized, encoded, None


def _url_with_query_params(url: str, params: Mapping[str, str]) -> str:
    if not params:
        return url
    parsed = urlparse(url)
    existing = parse_qsl(parsed.query, keep_blank_values=True)
    merged = existing + sorted((str(key), str(value)) for key, value in params.items())
    return urlunparse(parsed._replace(query=urlencode(merged, doseq=True)))


def _topical_evidence_from_lines(
    *,
    lines: Sequence[str],
    relative_path: str,
    query_text: str,
    max_hits: int,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query_text)
    query_lower = normalized_query.lower()
    terms = _workspace_terms(normalized_query)
    evidence: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        snippet = _normalize_text(raw_line)
        if not snippet:
            continue
        line_lower = snippet.lower()
        matched_terms = tuple(term for term in terms if term in line_lower)
        exact_query = bool(query_lower and query_lower in line_lower)
        score = (3 if exact_query else 0) + len(matched_terms)
        if score <= 0:
            continue
        evidence.append(
            {
                "path": relative_path,
                "line_number": int(line_number),
                "snippet": snippet[:240],
                "matched_terms": list(matched_terms),
                "exact_query": bool(exact_query),
                "score": int(score),
            }
        )
    evidence.sort(
        key=lambda item: (
            int(item.get("score", 0)),
            -int(item.get("line_number", 0)),
        ),
        reverse=True,
    )
    if evidence:
        return evidence[: max(1, int(max_hits))]

    fallback: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        snippet = _normalize_text(raw_line)
        if not snippet:
            continue
        fallback.append(
            {
                "path": relative_path,
                "line_number": int(line_number),
                "snippet": snippet[:240],
                "matched_terms": [],
                "exact_query": False,
                "score": 0,
            }
        )
        if len(fallback) >= max(1, int(max_hits)):
            break
    return fallback


def _json_leaf_pairs(payload: Any, *, path: str = "$", max_leaves: int = 256) -> list[tuple[str, str]]:
    leaves: list[tuple[str, str]] = []

    def visit(value: Any, current_path: str) -> None:
        if len(leaves) >= max(1, int(max_leaves)):
            return
        if isinstance(value, Mapping):
            for key, nested in list(value.items())[:64]:
                clean_key = _normalize_text(key)
                if not clean_key:
                    continue
                next_path = f"{current_path}.{clean_key}" if current_path != "$" else f"$.{clean_key}"
                visit(nested, next_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for idx, nested in enumerate(list(value)[:64]):
                visit(nested, f"{current_path}[{idx}]")
            return
        if value is None:
            return
        text = _normalize_text(value)
        if text:
            leaves.append((current_path, text))

    visit(payload, path)
    return leaves[: max(1, int(max_leaves))]


def _flatten_json_fields(payload: Any, *, max_fields: int = 256) -> list[tuple[str, str]]:
    return _json_leaf_pairs(payload, path="$", max_leaves=max_fields)


def _relative_json_subpath(root_path: str, leaf_path: str) -> str:
    root = _normalize_text(root_path)
    leaf = _normalize_text(leaf_path)
    if not leaf:
        return ""
    if leaf == root:
        return "$"
    if not root or root == "$":
        return leaf.lstrip("$").lstrip(".") or "$"
    if leaf.startswith(root):
        return leaf[len(root):].lstrip(".") or "$"
    return leaf.lstrip("$").lstrip(".") or "$"


def _json_structure_kind(value: Any) -> str:
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    if value is None:
        return "null"
    return "scalar"


def _json_node_snippet(path: str, value: Any, *, max_fields: int = 6) -> tuple[str, str, int]:
    structure_kind = _json_structure_kind(value)
    if structure_kind == "object" or structure_kind == "array":
        leaves = _json_leaf_pairs(value, path=path, max_leaves=max_fields)
        if leaves:
            fragments: list[str] = []
            for leaf_path, leaf_value in leaves:
                relative = _relative_json_subpath(path, leaf_path)
                if relative == "$":
                    fragments.append(leaf_value)
                else:
                    fragments.append(f"{relative} = {leaf_value}")
            snippet = _normalize_text(f"{path} => {'; '.join(fragments)}")[:240]
            return snippet, structure_kind, int(len(leaves))
        return f"{path} => empty {structure_kind}", structure_kind, 0
    if structure_kind == "null":
        return f"{path} = null", structure_kind, 0
    return f"{path} = {_normalize_text(value)}"[:240], structure_kind, 1


def _collect_json_nodes(payload: Any, *, path: str = "$", max_nodes: int = 512) -> dict[str, Any]:
    nodes: dict[str, Any] = {}

    def visit(value: Any, current_path: str) -> None:
        if len(nodes) >= max(1, int(max_nodes)):
            return
        nodes[current_path] = value
        if isinstance(value, Mapping):
            for key, nested in list(value.items())[:64]:
                clean_key = _normalize_text(key)
                if not clean_key:
                    continue
                next_path = f"{current_path}.{clean_key}" if current_path != "$" else f"$.{clean_key}"
                visit(nested, next_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for idx, nested in enumerate(list(value)[:64]):
                visit(nested, f"{current_path}[{idx}]")

    visit(payload, path)
    return nodes


def _structured_json_candidates(
    payload: Any,
    *,
    max_candidates: int = 128,
    max_fields_per_candidate: int = 6,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_candidate(value: Any, path: str, *, structure_kind: str) -> None:
        if len(candidates) >= max(1, int(max_candidates)):
            return
        snippet, actual_kind, field_count = _json_node_snippet(path, value, max_fields=max_fields_per_candidate)
        if actual_kind not in {"object", "array"}:
            return
        if field_count < 2:
            return
        key = (path.lower(), snippet.lower())
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "json_path": path,
                "snippet": snippet,
                "structure_kind": structure_kind,
                "field_count": int(field_count),
            }
        )

    def visit(value: Any, path: str) -> None:
        if len(candidates) >= max(1, int(max_candidates)):
            return
        if isinstance(value, Mapping):
            add_candidate(value, path, structure_kind="object")
            for key, nested in list(value.items())[:64]:
                clean_key = _normalize_text(key)
                if not clean_key:
                    continue
                next_path = f"{path}.{clean_key}" if path != "$" else f"$.{clean_key}"
                visit(nested, next_path)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            add_candidate(value, path, structure_kind="array")
            for idx, nested in enumerate(list(value)[:64]):
                visit(nested, f"{path}[{idx}]")

    visit(payload, "$")
    return candidates[: max(1, int(max_candidates))]


def _asserted_json_evidence(
    *,
    payload: Any,
    source_path: str,
    expected_json_paths: Sequence[str],
    expected_response_shape: str,
) -> tuple[list[dict[str, Any]], list[str], str | None, str | None]:
    nodes = _collect_json_nodes(payload)
    evidence: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    line_number = 0
    for raw_path in expected_json_paths:
        path = _normalize_json_path(raw_path)
        if not path:
            continue
        matches = _json_path_matches(nodes, path)
        if not matches:
            missing_paths.append(path)
            continue
        for concrete_path, value in matches:
            line_number += 1
            snippet, structure_kind, field_count = _json_node_snippet(concrete_path, value)
            evidence.append(
                {
                    "path": source_path,
                    "line_number": int(line_number),
                    "json_path": concrete_path,
                    "asserted_json_path": path,
                    "wildcard_match": bool(_json_path_has_wildcards(path)),
                    "snippet": snippet,
                    "matched_terms": [],
                    "exact_query": False,
                    "score": 0,
                    "evidence_kind": "json_assertion",
                    "assertion_kind": "expected_json_path",
                    "structure_kind": structure_kind,
                    "field_count": int(field_count),
                }
            )

    shape_evidence: str | None = None
    actual_shape = _json_structure_kind(payload)
    if expected_response_shape:
        if actual_shape != expected_response_shape:
            return evidence, missing_paths, actual_shape, shape_evidence
        line_number += 1
        snippet, structure_kind, field_count = _json_node_snippet("$", payload)
        shape_evidence = snippet
        evidence.append(
            {
                "path": source_path,
                "line_number": int(line_number),
                "json_path": "$",
                "snippet": snippet,
                "matched_terms": [],
                "exact_query": False,
                "score": 0,
                "evidence_kind": "json_assertion",
                "assertion_kind": "expected_response_shape",
                "expected_response_shape": expected_response_shape,
                "actual_response_shape": actual_shape,
                "structure_kind": structure_kind,
                "field_count": int(field_count),
            }
        )
    return evidence, missing_paths, actual_shape, shape_evidence


def _json_predicate_actual_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return _normalize_text(value)


def _predicate_quantifier_items(value: Any) -> list[Any] | None:
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return None


def _compile_regex_predicate(pattern: str) -> tuple[re.Pattern[str] | None, str | None]:
    text = _normalize_text(pattern)
    if not text:
        return None, "empty_regex"
    flags = 0
    body = text
    if text.startswith("/") and text.count("/") >= 2:
        last = text.rfind("/")
        body = text[1:last]
        flag_text = text[last + 1 :]
        for flag in flag_text:
            if flag == "i":
                flags |= re.IGNORECASE
            elif flag == "m":
                flags |= re.MULTILINE
            elif flag == "s":
                flags |= re.DOTALL
            else:
                return None, f"unsupported_regex_flag:{flag}"
    try:
        return re.compile(body, flags), None
    except re.error as exc:
        return None, f"invalid_regex:{exc}"


def _predicate_match_error(op: str, actual: Any, expected: Any) -> str | None:
    if op == "contains":
        if isinstance(actual, str):
            return None
        if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
            return None
        return "contains_requires_string_or_array"
    if op == "regex":
        pattern, compile_error = _compile_regex_predicate(str(expected))
        if compile_error is not None:
            return compile_error
        return None if pattern is not None else "invalid_regex"
    if op in {"gt", "gte", "lt", "lte"}:
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return "numeric_predicate_requires_numeric_actual"
        return None
    if op == "between":
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            return "between_requires_numeric_actual"
        if not isinstance(expected, Mapping):
            return "between_requires_range_object"
        return None
    if op in {"startswith", "endswith"}:
        if not isinstance(actual, str):
            return f"{op}_requires_string_actual"
        return None
    if op in {"any_contains", "any_regex", "all_contains", "all_regex", "none_contains", "none_regex"}:
        items = _predicate_quantifier_items(actual)
        if items is None:
            return f"{op}_requires_array_or_object"
        if op.endswith("regex"):
            pattern, compile_error = _compile_regex_predicate(str(expected))
            if compile_error is not None:
                return compile_error
            return None if pattern is not None else "invalid_regex"
        return None
    return "unsupported_predicate"


def _predicate_matches(op: str, actual: Any, expected: Any) -> bool:
    if op == "contains":
        if isinstance(actual, str):
            return str(expected) in actual
        if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray)):
            return any(item == expected for item in list(actual))
        return False
    if op == "regex":
        pattern, compile_error = _compile_regex_predicate(str(expected))
        if compile_error is not None or pattern is None:
            return False
        return bool(pattern.search(_json_predicate_actual_text(actual)))
    if op == "gt":
        return float(actual) > float(expected)
    if op == "gte":
        return float(actual) >= float(expected)
    if op == "lt":
        return float(actual) < float(expected)
    if op == "lte":
        return float(actual) <= float(expected)
    if op == "between":
        lower = expected.get("min") if isinstance(expected, Mapping) else None
        upper = expected.get("max") if isinstance(expected, Mapping) else None
        value = float(actual)
        if lower is not None and value < float(lower):
            return False
        if upper is not None and value > float(upper):
            return False
        return True
    if op == "startswith":
        return str(actual).startswith(str(expected))
    if op == "endswith":
        return str(actual).endswith(str(expected))
    items = _predicate_quantifier_items(actual)
    if op == "any_contains":
        return any(str(expected) in _json_predicate_actual_text(item) for item in (items or []))
    if op == "all_contains":
        return bool(items) and all(str(expected) in _json_predicate_actual_text(item) for item in items)
    if op == "none_contains":
        return all(str(expected) not in _json_predicate_actual_text(item) for item in (items or []))
    if op in {"any_regex", "all_regex", "none_regex"}:
        pattern, compile_error = _compile_regex_predicate(str(expected))
        if compile_error is not None or pattern is None:
            return False
        if op == "any_regex":
            return any(bool(pattern.search(_json_predicate_actual_text(item))) for item in (items or []))
        if op == "all_regex":
            return bool(items) and all(bool(pattern.search(_json_predicate_actual_text(item))) for item in items)
        return all(not pattern.search(_json_predicate_actual_text(item)) for item in (items or []))
    return False


def _asserted_json_value_evidence(
    *,
    payload: Any,
    source_path: str,
    expected_json_values: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    nodes = _collect_json_nodes(payload)
    evidence: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    mismatches: list[dict[str, Any]] = []
    line_number = 0
    for raw_path, expected_value in expected_json_values.items():
        path = _normalize_json_path(raw_path)
        if not path:
            continue
        matches = _json_path_matches(nodes, path)
        if not matches:
            missing_paths.append(path)
            continue
        wildcard = _json_path_has_wildcards(path)
        matched_any = False
        local_matches: list[dict[str, Any]] = []
        local_mismatches: list[dict[str, Any]] = []
        for concrete_path, actual_value in matches:
            snippet, structure_kind, field_count = _json_node_snippet(concrete_path, actual_value)
            line_number += 1
            candidate = {
                "path": source_path,
                "line_number": int(line_number),
                "json_path": concrete_path,
                "asserted_json_path": path,
                "wildcard_match": bool(wildcard),
                "snippet": snippet,
                "matched_terms": [],
                "exact_query": False,
                "score": 0,
                "evidence_kind": "json_assertion",
                "assertion_kind": "expected_json_value",
                "structure_kind": structure_kind,
                "field_count": int(field_count),
                "expected_value": expected_value,
                "actual_value": actual_value,
            }
            if actual_value == expected_value:
                matched_any = True
                local_matches.append(candidate)
            else:
                local_mismatches.append(candidate)
        if wildcard:
            if matched_any:
                evidence.extend(local_matches)
            else:
                mismatches.extend(local_mismatches)
            continue
        if local_matches:
            evidence.append(local_matches[0])
            continue
        mismatches.extend(local_mismatches[:1])
    return evidence, missing_paths, mismatches


def _asserted_json_predicate_evidence(
    *,
    payload: Any,
    source_path: str,
    expected_json_predicates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = _collect_json_nodes(payload)
    evidence: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    mismatches: list[dict[str, Any]] = []
    predicate_errors: list[dict[str, Any]] = []
    line_number = 0
    collection_ops = {"any_contains", "any_regex", "all_contains", "all_regex", "none_contains", "none_regex"}
    for raw_predicate in expected_json_predicates:
        path = _normalize_json_path(raw_predicate.get("path"))
        op = _normalize_text(raw_predicate.get("op")).lower()
        expected_value = raw_predicate.get("value")
        if not path:
            continue
        matches = _json_path_matches(nodes, path)
        if not matches:
            missing_paths.append(path)
            continue
        wildcard = _json_path_has_wildcards(path)
        if wildcard and op in collection_ops:
            aggregated_actual = [value for _, value in matches]
            line_number += 1
            snippet = _normalize_text(
                f"{path} => " + "; ".join(f"{concrete_path} = {_json_predicate_actual_text(value)}" for concrete_path, value in matches[:6])
            )[:240]
            candidate = {
                "path": source_path,
                "line_number": int(line_number),
                "json_path": path,
                "asserted_json_path": path,
                "wildcard_match": True,
                "matched_json_paths": [concrete_path for concrete_path, _ in matches],
                "snippet": snippet,
                "matched_terms": [],
                "exact_query": False,
                "score": 0,
                "evidence_kind": "json_assertion",
                "assertion_kind": "expected_json_predicate",
                "predicate_op": op,
                "structure_kind": "array",
                "field_count": int(len(matches)),
                "expected_value": expected_value,
                "actual_value": aggregated_actual,
            }
            predicate_error = _predicate_match_error(op, aggregated_actual, expected_value)
            if predicate_error is not None:
                predicate_errors.append({**candidate, "predicate_error": predicate_error})
                continue
            if _predicate_matches(op, aggregated_actual, expected_value):
                evidence.append(candidate)
            else:
                mismatches.append(dict(candidate))
            continue

        matched_any = False
        local_matches: list[dict[str, Any]] = []
        local_mismatches: list[dict[str, Any]] = []
        local_errors: list[dict[str, Any]] = []
        for concrete_path, actual_value in matches:
            snippet, structure_kind, field_count = _json_node_snippet(concrete_path, actual_value)
            line_number += 1
            candidate = {
                "path": source_path,
                "line_number": int(line_number),
                "json_path": concrete_path,
                "asserted_json_path": path,
                "wildcard_match": bool(wildcard),
                "snippet": snippet,
                "matched_terms": [],
                "exact_query": False,
                "score": 0,
                "evidence_kind": "json_assertion",
                "assertion_kind": "expected_json_predicate",
                "predicate_op": op,
                "structure_kind": structure_kind,
                "field_count": int(field_count),
                "expected_value": expected_value,
                "actual_value": actual_value,
            }
            predicate_error = _predicate_match_error(op, actual_value, expected_value)
            if predicate_error is not None:
                local_errors.append({**candidate, "predicate_error": predicate_error})
                continue
            if _predicate_matches(op, actual_value, expected_value):
                matched_any = True
                local_matches.append(candidate)
            else:
                local_mismatches.append(dict(candidate))
        if wildcard:
            if local_errors and not local_matches:
                predicate_errors.extend(local_errors)
                continue
            if matched_any:
                evidence.extend(local_matches)
            else:
                mismatches.extend(local_mismatches or local_errors[:1])
            continue
        if local_errors:
            predicate_errors.extend(local_errors[:1])
            continue
        if local_matches:
            evidence.append(local_matches[0])
        else:
            mismatches.extend(local_mismatches[:1])
    return evidence, missing_paths, mismatches, predicate_errors


def _predicate_group_snippet(
    logic: str,
    predicates: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]] = (),
) -> str:
    parts: list[str] = []
    for predicate in list(predicates)[:4]:
        path = _normalize_json_path(predicate.get("path")) or "$"
        op = _normalize_text(predicate.get("op")).lower() or "predicate"
        value_text = json.dumps(predicate.get("value"), ensure_ascii=False, sort_keys=True)
        parts.append(f"{path} {op} {value_text}")
    for group in list(groups)[:2]:
        group_logic = _normalize_text(group.get("logic")).lower() or "group"
        parts.append(f"group:{group_logic}")
    return f"{logic} of [{'; '.join(parts)}]"[:240]


def _asserted_json_predicate_group_evidence(
    *,
    payload: Any,
    source_path: str,
    expected_json_predicate_groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    line_number = 0
    for raw_group in expected_json_predicate_groups:
        logic = _normalize_text(raw_group.get("logic")).lower()
        predicates = [
            dict(item)
            for item in list(raw_group.get("predicates") or [])
            if isinstance(item, Mapping)
        ]
        child_groups = [
            dict(item)
            for item in list(raw_group.get("groups") or [])
            if isinstance(item, Mapping)
        ]
        matched, missing_paths, mismatches, predicate_errors = _asserted_json_predicate_evidence(
            payload=payload,
            source_path=source_path,
            expected_json_predicates=predicates,
        )
        child_evidence, child_failures = _asserted_json_predicate_group_evidence(
            payload=payload,
            source_path=source_path,
            expected_json_predicate_groups=child_groups,
        )
        matched_predicate_count = int(len(matched))
        matched_group_count = int(len(child_evidence))
        predicate_count = int(len(predicates))
        child_group_count = int(len(child_groups))
        total_clause_count = predicate_count + child_group_count
        matched_clause_count = matched_predicate_count + matched_group_count
        snippet = _predicate_group_snippet(logic, predicates, child_groups)
        line_number += 1
        child_evaluation_failures = int(
            sum(
                1
                for item in child_failures
                if int(item.get("missing_paths_count", 0) or 0) > 0
                or int(item.get("predicate_error_count", 0) or 0) > 0
                or int(item.get("child_group_failure_count", 0) or 0) > 0
            )
        )
        candidate = {
            "path": source_path,
            "line_number": int(line_number),
            "json_path": "$",
            "snippet": snippet,
            "matched_terms": [],
            "exact_query": False,
            "score": 0,
            "evidence_kind": "json_assertion",
            "assertion_kind": "expected_json_predicate_group",
            "group_logic": logic,
            "predicate_count": predicate_count,
            "matched_predicate_count": matched_predicate_count,
            "child_group_count": child_group_count,
            "matched_child_group_count": matched_group_count,
            "missing_paths": list(missing_paths),
            "missing_paths_count": int(len(missing_paths)),
            "mismatch_count": int(len(mismatches)),
            "predicate_error_count": int(len(predicate_errors)),
            "child_group_failure_count": int(len(child_failures)),
            "child_group_evaluation_failure_count": child_evaluation_failures,
            "predicates": [dict(item) for item in predicates],
            "groups": [dict(item) for item in child_groups],
        }
        if logic == "all":
            success = (
                total_clause_count > 0
                and not missing_paths
                and not mismatches
                and not predicate_errors
                and not child_failures
                and matched_clause_count == total_clause_count
            )
            failure_reason = "not all predicates or child groups were satisfied"
        elif logic == "any":
            success = matched_clause_count > 0
            failure_reason = "none of the predicates or child groups were satisfied"
        else:
            success = (
                matched_clause_count == 0
                and not missing_paths
                and not predicate_errors
                and child_evaluation_failures == 0
            )
            failure_reason = "one or more predicates or child groups matched when logic required none"
        if success:
            evidence.append(candidate)
            continue
        failures.append({**candidate, "group_failure_reason": failure_reason})
    return evidence, failures


def _topical_evidence_from_json_payload(
    *,
    payload: Any,
    source_path: str,
    query_text: str,
    max_hits: int,
) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(query_text)
    query_lower = normalized_query.lower()
    terms = _workspace_terms(normalized_query)
    evidence: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []

    fields = _flatten_json_fields(payload)
    for idx, (json_path, raw_value) in enumerate(list(fields), start=1):
        value = _normalize_text(raw_value)
        if not value:
            continue
        snippet = f"{json_path} = {value}"[:240]
        lower = snippet.lower()
        matched_terms = tuple(term for term in terms if term in lower)
        exact_query = bool(query_lower and query_lower in lower)
        score = (3 if exact_query else 0) + len(matched_terms)
        candidate = {
            "path": source_path,
            "line_number": int(idx),
            "json_path": json_path,
            "snippet": snippet,
            "matched_terms": list(matched_terms),
            "exact_query": bool(exact_query),
            "score": int(score),
            "evidence_kind": "json_field",
            "structure_kind": "scalar",
            "field_count": 1,
        }
        if score > 0:
            evidence.append(candidate)
        elif len(fallback) < max(1, int(max_hits)):
            fallback.append(candidate)

    base_line = len(fields)
    for offset, structure in enumerate(_structured_json_candidates(payload), start=1):
        json_path = _normalize_text(structure.get("json_path", ""))
        snippet = _normalize_text(structure.get("snippet", ""))[:240]
        if not json_path or not snippet:
            continue
        lower = f"{json_path} {snippet}".lower()
        matched_terms = tuple(term for term in terms if term in lower)
        exact_query = bool(query_lower and query_lower in lower)
        score = (3 if exact_query else 0) + len(matched_terms)
        candidate = {
            "path": source_path,
            "line_number": int(base_line + offset),
            "json_path": json_path,
            "snippet": snippet,
            "matched_terms": list(matched_terms),
            "exact_query": bool(exact_query),
            "score": int(score),
            "evidence_kind": "json_structure",
            "structure_kind": _normalize_text(structure.get("structure_kind", "object")) or "object",
            "field_count": max(2, int(structure.get("field_count", 2) or 2)),
        }
        if score > 0:
            evidence.append(candidate)
        elif len(fallback) < max(1, int(max_hits)):
            fallback.append(candidate)

    evidence.sort(
        key=lambda item: (
            int(item.get("score", 0)),
            int(bool(item.get("exact_query", False))),
            str(item.get("json_path", "")).count(".") + str(item.get("json_path", "")).count("["),
            1 if str(item.get("evidence_kind", "")) == "json_field" else 0,
            int(item.get("field_count", 1)),
            -int(item.get("line_number", 0)),
        ),
        reverse=True,
    )
    if evidence:
        return evidence[: max(1, int(max_hits))]
    return fallback[: max(1, int(max_hits))]


@dataclass(frozen=True)
class ActionVerification:
    status: str
    success: bool
    confidence: float
    contradiction: bool
    summary: str
    evidence: tuple[dict[str, Any], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": bool(self.success),
            "confidence": float(self.confidence),
            "contradiction": bool(self.contradiction),
            "summary": self.summary,
            "evidence": [dict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class DigitalActionResult:
    action_id: str
    action_type: str
    inputs: dict[str, Any]
    predicted_outcome: str
    actual_outcome: str
    verification: ActionVerification
    topics: tuple[str, ...]
    recorded_at: str
    episode_text: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "inputs": dict(self.inputs),
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "verification": self.verification.to_payload(),
            "topics": list(self.topics),
            "recorded_at": self.recorded_at,
            "episode_text": self.episode_text,
        }

    def memory_metadata(self) -> dict[str, Any]:
        return {
            "observation_kind": "action",
            "grounded": True,
            "grounding_signal": 0.92 if self.verification.success else 0.72,
            "evidence_unit_count": max(1, len(self.verification.evidence)),
            "source_name": "workspace",
            "source_type": "action",
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_inputs": dict(self.inputs),
            "predicted_outcome": self.predicted_outcome,
            "actual_outcome": self.actual_outcome,
            "verification_status": self.verification.status,
            "verification_confidence": float(self.verification.confidence),
            "contradiction": bool(self.verification.contradiction),
            "evidence": [dict(item) for item in self.verification.evidence],
        }


def execute_workspace_search(
    root: Path,
    *,
    query_text: str,
    predicted_outcome: str = "",
    max_hits: int = 6,
    max_files: int = 256,
    max_file_bytes: int = 200_000,
) -> DigitalActionResult:
    workspace_root = root.resolve()
    normalized_query = _normalize_text(query_text)
    recorded_at = datetime.now(timezone.utc).isoformat()
    terms = _workspace_terms(normalized_query)
    query_lower = normalized_query.lower()
    evidence: list[dict[str, Any]] = []
    files_scanned = 0

    if workspace_root.exists() and workspace_root.is_dir() and normalized_query:
        for path in _iter_workspace_files(workspace_root):
            if files_scanned >= max(1, int(max_files)) or len(evidence) >= max(1, int(max_hits)):
                break
            files_scanned += 1
            try:
                if path.stat().st_size > int(max_file_bytes):
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            best_match: dict[str, Any] | None = None
            best_score = -1
            for line_number, raw_line in enumerate(content.splitlines(), start=1):
                snippet = _normalize_text(raw_line)
                if not snippet:
                    continue
                line_lower = snippet.lower()
                matched_terms = tuple(term for term in terms if term in line_lower)
                exact_query = bool(query_lower and query_lower in line_lower)
                if not exact_query and not matched_terms:
                    continue
                score = (3 if exact_query else 0) + len(matched_terms)
                if score <= best_score:
                    continue
                best_score = score
                best_match = {
                    "path": str(path.relative_to(workspace_root)).replace("\\", "/"),
                    "line_number": int(line_number),
                    "snippet": snippet[:240],
                    "matched_terms": list(matched_terms),
                    "exact_query": bool(exact_query),
                    "score": int(score),
                }
            if best_match is not None:
                evidence.append(best_match)

    success = len(evidence) > 0
    contradiction = not success
    if success:
        snippets = "; ".join(
            f"{item['path']}:{item['line_number']} ({item['snippet']})"
            for item in evidence[:3]
        )
        actual_outcome = f"Workspace search found {len(evidence)} matching file hits for '{normalized_query}': {snippets}"
        confidence = min(0.97, 0.76 + 0.05 * min(len(evidence), 4))
        summary = f"Verified workspace search found {len(evidence)} matching hits."
        status = "verified"
    else:
        actual_outcome = (
            f"Workspace search found no matching file hits for '{normalized_query}' "
            f"after scanning {files_scanned} files."
        )
        confidence = 0.74 if files_scanned > 0 else 0.45
        summary = "Workspace search contradicted the expected finding and returned no matches."
        status = "contradicted"

    topic_terms = list(terms)
    for item in evidence[:3]:
        stem = Path(str(item.get("path", ""))).stem.strip().lower()
        if stem:
            topic_terms.append(stem)
    topics: list[str] = []
    seen_topics: set[str] = set()
    for item in topic_terms:
        normalized = _normalize_text(item).lower()
        if not normalized or normalized in seen_topics:
            continue
        topics.append(normalized)
        seen_topics.add(normalized)
    predicted = _normalize_text(predicted_outcome)
    episode_text = (
        f"Digital action workspace_search for '{normalized_query}'. "
        f"Predicted outcome: {predicted or 'no explicit prediction provided'}. "
        f"Actual outcome: {actual_outcome}. "
        f"Verification: {summary}"
    )
    payload = {
        "root": str(workspace_root),
        "query_text": normalized_query,
        "predicted_outcome": predicted,
        "files_scanned": int(files_scanned),
        "hit_count": int(len(evidence)),
    }
    return DigitalActionResult(
        action_id=_action_id("workspace_search", payload, recorded_at=recorded_at),
        action_type="workspace_search",
        inputs=payload,
        predicted_outcome=predicted,
        actual_outcome=actual_outcome,
        verification=ActionVerification(
            status=status,
            success=bool(success),
            confidence=float(confidence),
            contradiction=bool(contradiction),
            summary=summary,
            evidence=tuple(dict(item) for item in evidence),
        ),
        topics=tuple(topics[:8]),
        recorded_at=recorded_at,
        episode_text=episode_text[:640],
    )


def execute_workspace_read(
    root: Path,
    *,
    path: str,
    query_text: str = "",
    predicted_outcome: str = "",
    max_hits: int = 6,
    max_file_bytes: int = 200_000,
) -> DigitalActionResult:
    workspace_root = root.resolve()
    normalized_path = _normalize_text(path)
    normalized_query = _normalize_text(query_text)
    predicted = _normalize_text(predicted_outcome)
    recorded_at = datetime.now(timezone.utc).isoformat()
    target = _resolve_workspace_path(workspace_root, normalized_path)
    payload = {
        "root": str(workspace_root),
        "path": normalized_path,
        "query_text": normalized_query,
        "predicted_outcome": predicted,
    }
    if target is None:
        actual_outcome = f"Workspace read could not open '{normalized_path}' inside the allowed workspace."
        summary = "Workspace read was contradicted because the target file was missing, invalid, or outside the workspace."
        return DigitalActionResult(
            action_id=_action_id("workspace_read", payload, recorded_at=recorded_at),
            action_type="workspace_read",
            inputs=payload,
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=0.86,
                contradiction=True,
                summary=summary,
                evidence=(),
            ),
            topics=_dedupe_topics([Path(normalized_path).stem, *list(_workspace_terms(normalized_query))]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action workspace_read for '{normalized_path}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    try:
        if target.stat().st_size > int(max_file_bytes):
            raise ValueError("file_too_large")
        content = target.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        actual_outcome = f"Workspace read could not decode '{target.name}' as a bounded text file."
        summary = "Workspace read was contradicted because the file could not be read safely as text."
        relative = str(target.relative_to(workspace_root)).replace("\\", "/") if target.exists() else normalized_path
        return DigitalActionResult(
            action_id=_action_id("workspace_read", payload, recorded_at=recorded_at),
            action_type="workspace_read",
            inputs=payload,
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=0.82,
                contradiction=True,
                summary=summary,
                evidence=({"path": relative},),
            ),
            topics=_dedupe_topics([target.stem, *list(_workspace_terms(normalized_query))]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action workspace_read for '{relative}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    relative_path = str(target.relative_to(workspace_root)).replace("\\", "/")
    evidence = _topical_evidence_from_lines(
        lines=content.splitlines(),
        relative_path=relative_path,
        query_text=normalized_query,
        max_hits=max_hits,
    )
    snippets = "; ".join(
        f"{item['path']}:{item['line_number']} ({item['snippet']})"
        for item in evidence[:3]
    )
    if any(list(item.get("matched_terms") or []) for item in evidence):
        actual_outcome = f"Workspace read extracted {len(evidence)} relevant snippets from '{relative_path}': {snippets}"
        confidence = min(0.98, 0.80 + 0.04 * min(len(evidence), 4))
        summary = f"Verified workspace read extracted {len(evidence)} relevant snippets."
    else:
        actual_outcome = f"Workspace read opened '{relative_path}' and extracted {len(evidence)} leading snippets: {snippets}"
        confidence = 0.77
        summary = "Verified workspace read opened the file successfully but found no direct query-specific match."
    topics = _dedupe_topics([target.stem, *list(_workspace_terms(normalized_query))])
    return DigitalActionResult(
        action_id=_action_id("workspace_read", {**payload, "hit_count": len(evidence)}, recorded_at=recorded_at),
        action_type="workspace_read",
        inputs={**payload, "path": relative_path, "hit_count": int(len(evidence))},
        predicted_outcome=predicted,
        actual_outcome=actual_outcome,
        verification=ActionVerification(
            status="verified",
            success=True,
            confidence=float(confidence),
            contradiction=False,
            summary=summary,
            evidence=tuple(dict(item) for item in evidence),
        ),
        topics=topics,
        recorded_at=recorded_at,
        episode_text=(
            f"Digital action workspace_read for '{relative_path}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
            f"Actual outcome: {actual_outcome}. Verification: {summary}"
        )[:640],
    )


def execute_web_fetch(
    root: Path,
    *,
    url: str,
    query_text: str = "",
    predicted_outcome: str = "",
    max_hits: int = 6,
    max_response_bytes: int = 200_000,
    timeout_seconds: float = 10.0,
) -> DigitalActionResult:
    del root  # maintained action surface uses root for policy symmetry; web fetch itself is URL-scoped
    normalized_url = _normalize_url(url)
    normalized_query = _normalize_text(query_text)
    predicted = _normalize_text(predicted_outcome)
    recorded_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "url": normalized_url or _normalize_text(url),
        "query_text": normalized_query,
        "predicted_outcome": predicted,
    }
    if not normalized_url:
        actual_outcome = f"Web fetch could not use '{_normalize_text(url)}' as a valid http/https URL."
        summary = "Web fetch was contradicted because the target URL was invalid."
        return DigitalActionResult(
            action_id=_action_id("web_fetch", payload, recorded_at=recorded_at),
            action_type="web_fetch",
            inputs=payload,
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=0.9,
                contradiction=True,
                summary=summary,
                evidence=(),
            ),
            topics=_dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(_normalize_text(url)).netloc]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action web_fetch for '{_normalize_text(url)}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    headers = {
        "User-Agent": "HECSN/1.0 (+https://github.com/) action-loop-web-fetch",
        "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
    }
    request = Request(normalized_url, headers=headers)
    try:
        with urlopen(request, timeout=max(0.1, float(timeout_seconds))) as response:
            payload_bytes = response.read(max(1000, int(max_response_bytes)) + 1)
            content_type = response.headers.get("Content-Type")
            encoding = response.headers.get_content_charset() or "utf-8"
    except Exception as exc:
        actual_outcome = f"Web fetch could not retrieve '{normalized_url}': {type(exc).__name__}."
        summary = "Web fetch was contradicted because the target URL could not be retrieved."
        return DigitalActionResult(
            action_id=_action_id("web_fetch", payload, recorded_at=recorded_at),
            action_type="web_fetch",
            inputs=payload,
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=0.84,
                contradiction=True,
                summary=summary,
                evidence=(),
            ),
            topics=_dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(normalized_url).netloc]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action web_fetch for '{normalized_url}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    truncated = len(payload_bytes) > int(max_response_bytes)
    if truncated:
        payload_bytes = payload_bytes[: int(max_response_bytes)]
    try:
        decoded = payload_bytes.decode(encoding, errors="ignore")
    except LookupError:
        decoded = payload_bytes.decode("utf-8", errors="ignore")
    visible_text = extract_web_text(decoded, content_type=content_type, max_chars=int(max_response_bytes))
    if not visible_text:
        actual_outcome = f"Web fetch retrieved '{normalized_url}' but found no visible text content."
        summary = "Web fetch was contradicted because the response yielded no usable visible text."
        return DigitalActionResult(
            action_id=_action_id("web_fetch", payload, recorded_at=recorded_at),
            action_type="web_fetch",
            inputs=payload,
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=0.78,
                contradiction=True,
                summary=summary,
                evidence=(),
            ),
            topics=_dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(normalized_url).netloc]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action web_fetch for '{normalized_url}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    evidence = _topical_evidence_from_lines(
        lines=visible_text.splitlines(),
        relative_path=normalized_url,
        query_text=normalized_query,
        max_hits=max_hits,
    )
    snippets = "; ".join(
        f"{item['path']}:{item['line_number']} ({item['snippet']})"
        for item in evidence[:3]
    )
    matched = any(list(item.get("matched_terms") or []) for item in evidence)
    if matched:
        actual_outcome = f"Web fetch extracted {len(evidence)} relevant snippets from '{normalized_url}': {snippets}"
        confidence = min(0.98, 0.80 + 0.04 * min(len(evidence), 4))
        summary = f"Verified web fetch extracted {len(evidence)} relevant snippets."
    else:
        extra = " The response was truncated to fit the bounded payload budget." if truncated else ""
        actual_outcome = f"Web fetch opened '{normalized_url}' and extracted {len(evidence)} leading snippets: {snippets}{extra}"
        confidence = 0.76
        summary = "Verified web fetch opened the URL successfully but found no direct query-specific match."
    topics = _dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(normalized_url).netloc])
    return DigitalActionResult(
        action_id=_action_id("web_fetch", {**payload, "hit_count": len(evidence)}, recorded_at=recorded_at),
        action_type="web_fetch",
        inputs={**payload, "url": normalized_url, "hit_count": int(len(evidence))},
        predicted_outcome=predicted,
        actual_outcome=actual_outcome,
        verification=ActionVerification(
            status="verified",
            success=True,
            confidence=float(confidence),
            contradiction=False,
            summary=summary,
            evidence=tuple(dict(item) for item in evidence),
        ),
        topics=topics,
        recorded_at=recorded_at,
        episode_text=(
            f"Digital action web_fetch for '{normalized_url}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
            f"Actual outcome: {actual_outcome}. Verification: {summary}"
        )[:640],
    )


def execute_api_request(
    root: Path,
    *,
    url: str,
    query_text: str = "",
    predicted_outcome: str = "",
    method: str = "GET",
    params: Mapping[str, Any] | None = None,
    json_body: Any = None,
    expected_json_paths: Sequence[str] | None = None,
    expected_json_values: Mapping[str, Any] | None = None,
    expected_json_predicates: Sequence[Mapping[str, Any]] | None = None,
    expected_json_predicate_groups: Sequence[Mapping[str, Any]] | None = None,
    expected_response_shape: str | None = None,
    max_hits: int = 6,
    max_response_bytes: int = 200_000,
    timeout_seconds: float = 10.0,
) -> DigitalActionResult:
    del root
    normalized_url = _normalize_url(url)
    normalized_query = _normalize_text(query_text)
    predicted = _normalize_text(predicted_outcome)
    normalized_method = _normalize_http_method(method)
    normalized_params, params_error = _normalize_request_params(params)
    normalized_body, request_body, body_error = _serialize_json_request_body(json_body)
    normalized_expected_paths, expected_paths_error = _normalize_expected_json_paths(expected_json_paths)
    normalized_expected_values, expected_values_error = _normalize_expected_json_values(expected_json_values)
    normalized_expected_predicates, expected_predicates_error = _normalize_expected_json_predicates(expected_json_predicates)
    normalized_expected_predicate_groups, expected_predicate_groups_error = _normalize_expected_json_predicate_groups(expected_json_predicate_groups)
    normalized_expected_shape, expected_shape_error = _normalize_expected_response_shape(expected_response_shape)
    request_url = _url_with_query_params(normalized_url, normalized_params) if normalized_url else ""
    recorded_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "url": request_url or normalized_url or _normalize_text(url),
        "base_url": normalized_url or _normalize_text(url),
        "query_text": normalized_query,
        "predicted_outcome": predicted,
        "method": normalized_method or _normalize_text(method).upper() or "GET",
    }
    if normalized_params:
        payload["params"] = dict(normalized_params)
    if normalized_body is not None:
        payload["json_body"] = normalized_body
    if normalized_expected_paths:
        payload["expected_json_paths"] = list(normalized_expected_paths)
    if normalized_expected_values:
        payload["expected_json_values"] = dict(normalized_expected_values)
    if normalized_expected_predicates:
        payload["expected_json_predicates"] = [dict(item) for item in normalized_expected_predicates]
    if normalized_expected_predicate_groups:
        payload["expected_json_predicate_groups"] = [dict(item) for item in normalized_expected_predicate_groups]
    if normalized_expected_shape:
        payload["expected_response_shape"] = normalized_expected_shape

    def contradicted(
        *,
        actual_outcome: str,
        summary: str,
        confidence: float,
        evidence: Sequence[Mapping[str, Any]] = (),
        topic_url: str | None = None,
    ) -> DigitalActionResult:
        target = topic_url or request_url or normalized_url or _normalize_text(url)
        return DigitalActionResult(
            action_id=_action_id("api_request", payload, recorded_at=recorded_at),
            action_type="api_request",
            inputs=dict(payload),
            predicted_outcome=predicted,
            actual_outcome=actual_outcome,
            verification=ActionVerification(
                status="contradicted",
                success=False,
                confidence=float(confidence),
                contradiction=True,
                summary=summary,
                evidence=tuple(dict(item) for item in evidence),
            ),
            topics=_dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(target).netloc]),
            recorded_at=recorded_at,
            episode_text=(
                f"Digital action api_request for '{target}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
                f"Actual outcome: {actual_outcome}. Verification: {summary}"
            )[:640],
        )

    if not normalized_url:
        return contradicted(
            actual_outcome=f"API request could not use '{_normalize_text(url)}' as a valid http/https URL.",
            summary="API request was contradicted because the target URL was invalid.",
            confidence=0.9,
            topic_url=_normalize_text(url),
        )
    if not normalized_method:
        return contradicted(
            actual_outcome=f"API request could not use HTTP method '{_normalize_text(method).upper() or 'unknown'}'; only GET and POST are supported.",
            summary="API request was contradicted because the requested HTTP method is unsupported.",
            confidence=0.88,
        )
    if params_error:
        return contradicted(
            actual_outcome=params_error,
            summary="API request was contradicted because the request parameters were invalid.",
            confidence=0.86,
        )
    if body_error:
        return contradicted(
            actual_outcome=body_error,
            summary="API request was contradicted because the JSON request body was invalid.",
            confidence=0.86,
        )
    if expected_paths_error:
        return contradicted(
            actual_outcome=expected_paths_error,
            summary="API request was contradicted because the expected JSON path assertions were invalid.",
            confidence=0.86,
        )
    if expected_values_error:
        return contradicted(
            actual_outcome=expected_values_error,
            summary="API request was contradicted because the expected JSON value assertions were invalid.",
            confidence=0.86,
        )
    if expected_predicates_error:
        return contradicted(
            actual_outcome=expected_predicates_error,
            summary="API request was contradicted because the expected JSON predicate assertions were invalid.",
            confidence=0.86,
        )
    if expected_predicate_groups_error:
        return contradicted(
            actual_outcome=expected_predicate_groups_error,
            summary="API request was contradicted because the expected JSON predicate-group assertions were invalid.",
            confidence=0.86,
        )
    if expected_shape_error:
        return contradicted(
            actual_outcome=expected_shape_error,
            summary="API request was contradicted because the expected response-shape assertion was invalid.",
            confidence=0.86,
        )
    if normalized_method == "GET" and normalized_body is not None:
        return contradicted(
            actual_outcome="API request could not send a JSON body with GET; use query params or POST for structured request bodies.",
            summary="API request was contradicted because GET requests do not support the maintained JSON-body path.",
            confidence=0.87,
        )

    headers = {
        "User-Agent": "HECSN/1.0 (+https://github.com/) action-loop-api-request",
        "Accept": "application/json,text/plain;q=0.9,*/*;q=0.1",
    }
    if normalized_method == "POST" and request_body is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = Request(request_url, headers=headers, data=request_body, method=normalized_method)
    try:
        with urlopen(request, timeout=max(0.1, float(timeout_seconds))) as response:
            payload_bytes = response.read(max(1000, int(max_response_bytes)) + 1)
            content_type = response.headers.get("Content-Type") or ""
            encoding = response.headers.get_content_charset() or "utf-8"
    except Exception as exc:
        return contradicted(
            actual_outcome=f"API request {normalized_method} could not retrieve '{request_url}': {type(exc).__name__}.",
            summary="API request was contradicted because the target endpoint could not be retrieved.",
            confidence=0.84,
        )

    truncated = len(payload_bytes) > int(max_response_bytes)
    if truncated:
        payload_bytes = payload_bytes[: int(max_response_bytes)]
    try:
        decoded = payload_bytes.decode(encoding, errors="ignore")
    except LookupError:
        decoded = payload_bytes.decode("utf-8", errors="ignore")
    try:
        parsed = json.loads(decoded)
    except Exception:
        return contradicted(
            actual_outcome=f"API request {normalized_method} retrieved '{request_url}' but the response was not valid JSON.",
            summary="API request was contradicted because the endpoint did not return structured JSON.",
            confidence=0.82,
            evidence=({"path": request_url, "content_type": content_type, "method": normalized_method},),
        )

    fields = _flatten_json_fields(parsed)
    asserted_evidence, missing_paths, actual_shape, _ = _asserted_json_evidence(
        payload=parsed,
        source_path=request_url,
        expected_json_paths=normalized_expected_paths,
        expected_response_shape=normalized_expected_shape,
    )
    value_evidence, missing_value_paths, mismatched_values = _asserted_json_value_evidence(
        payload=parsed,
        source_path=request_url,
        expected_json_values=normalized_expected_values,
    )
    predicate_evidence, missing_predicate_paths, mismatched_predicates, predicate_errors = _asserted_json_predicate_evidence(
        payload=parsed,
        source_path=request_url,
        expected_json_predicates=normalized_expected_predicates,
    )
    predicate_group_evidence, mismatched_predicate_groups = _asserted_json_predicate_group_evidence(
        payload=parsed,
        source_path=request_url,
        expected_json_predicate_groups=normalized_expected_predicate_groups,
    )
    if normalized_expected_paths or normalized_expected_values or normalized_expected_predicates or normalized_expected_predicate_groups or normalized_expected_shape:
        if missing_paths:
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but did not satisfy expected JSON paths: "
                    + ", ".join(missing_paths)
                ),
                summary="API request was contradicted because one or more expected JSON paths were missing.",
                confidence=0.84,
                evidence=tuple(list(asserted_evidence)[: max(1, int(max_hits))]),
            )
        if missing_value_paths:
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but could not verify expected JSON values because these paths were missing: "
                    + ", ".join(missing_value_paths)
                ),
                summary="API request was contradicted because one or more expected JSON value paths were missing.",
                confidence=0.84,
                evidence=tuple(list(asserted_evidence)[: max(1, int(max_hits))]),
            )
        if missing_predicate_paths:
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but could not verify expected JSON predicates because these paths were missing: "
                    + ", ".join(missing_predicate_paths)
                ),
                summary="API request was contradicted because one or more expected JSON predicate paths were missing.",
                confidence=0.84,
                evidence=tuple(list(asserted_evidence)[: max(1, int(max_hits))]),
            )
        if mismatched_values:
            mismatch = dict(mismatched_values[0])
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but path '{mismatch.get('json_path', '$')}' had value "
                    f"{json.dumps(mismatch.get('actual_value'), ensure_ascii=False)} instead of expected {json.dumps(mismatch.get('expected_value'), ensure_ascii=False)}."
                ),
                summary="API request was contradicted because one or more expected JSON values did not match.",
                confidence=0.85,
                evidence=tuple(dict(item) for item in mismatched_values[: max(1, int(max_hits))]),
            )
        if predicate_errors:
            mismatch = dict(predicate_errors[0])
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but predicate '{mismatch.get('predicate_op', 'unknown')}' "
                    f"could not be evaluated at path '{mismatch.get('json_path', '$')}' ({mismatch.get('predicate_error', 'predicate_error')})."
                ),
                summary="API request was contradicted because one or more expected JSON predicates could not be evaluated.",
                confidence=0.85,
                evidence=tuple(dict(item) for item in predicate_errors[: max(1, int(max_hits))]),
            )
        if mismatched_predicates:
            mismatch = dict(mismatched_predicates[0])
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but path '{mismatch.get('json_path', '$')}' did not satisfy predicate "
                    f"{mismatch.get('predicate_op', 'unknown')} with expected {json.dumps(mismatch.get('expected_value'), ensure_ascii=False)}; "
                    f"actual value was {json.dumps(mismatch.get('actual_value'), ensure_ascii=False)}."
                ),
                summary="API request was contradicted because one or more expected JSON predicates did not match.",
                confidence=0.85,
                evidence=tuple(dict(item) for item in mismatched_predicates[: max(1, int(max_hits))]),
            )
        if mismatched_predicate_groups:
            mismatch = dict(mismatched_predicate_groups[0])
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' but predicate group with logic "
                    f"'{mismatch.get('group_logic', 'unknown')}' failed because {mismatch.get('group_failure_reason', 'predicate_group_failed')}."
                ),
                summary="API request was contradicted because one or more expected JSON predicate groups did not match.",
                confidence=0.85,
                evidence=tuple(dict(item) for item in mismatched_predicate_groups[: max(1, int(max_hits))]),
            )
        if normalized_expected_shape and actual_shape != normalized_expected_shape:
            return contradicted(
                actual_outcome=(
                    f"API request {normalized_method} parsed JSON from '{request_url}' with response shape '{actual_shape}', "
                    f"not expected '{normalized_expected_shape}'."
                ),
                summary="API request was contradicted because the response shape did not match the expected assertion.",
                confidence=0.84,
                evidence=(
                    {
                        "path": request_url,
                        "json_path": "$",
                        "expected_response_shape": normalized_expected_shape,
                        "actual_response_shape": actual_shape,
                        "structure_kind": actual_shape,
                    },
                ),
            )
        evidence = [
            dict(item)
            for item in [*list(asserted_evidence), *list(value_evidence), *list(predicate_evidence), *list(predicate_group_evidence)]
        ]
        for idx, item in enumerate(evidence, start=1):
            item["line_number"] = int(idx)
        evidence = evidence[: max(1, int(max_hits))]
        snippets = "; ".join(
            f"{item.get('json_path', '$')} ({item['snippet']})"
            for item in evidence[:3]
        )
        assertion_bits: list[str] = []
        if normalized_expected_paths:
            assertion_bits.append(f"{len(normalized_expected_paths)} JSON path assertion(s)")
        if normalized_expected_values:
            assertion_bits.append(f"{len(normalized_expected_values)} JSON value assertion(s)")
        if normalized_expected_predicates:
            assertion_bits.append(f"{len(normalized_expected_predicates)} JSON predicate assertion(s)")
        if normalized_expected_predicate_groups:
            assertion_bits.append(f"{len(normalized_expected_predicate_groups)} JSON predicate-group assertion(s)")
        if normalized_expected_shape:
            assertion_bits.append(f"response shape '{normalized_expected_shape}'")
        actual_outcome = (
            f"API request {normalized_method} satisfied {' and '.join(assertion_bits)} for '{request_url}': {snippets}"
        )
        confidence = min(
            0.99,
            0.86
            + 0.04 * min(len(evidence), 4)
            + (0.03 if normalized_expected_shape else 0.0)
            + (0.03 if normalized_expected_values else 0.0)
            + (0.03 if normalized_expected_predicates else 0.0)
            + (0.03 if normalized_expected_predicate_groups else 0.0),
        )
        summary = f"Verified API request {normalized_method} satisfied explicit structured JSON assertions."
    else:
        if not fields:
            return contradicted(
                actual_outcome=f"API request {normalized_method} retrieved '{request_url}' but the JSON payload contained no scalar fields to verify.",
                summary="API request was contradicted because the endpoint yielded no verifiable scalar JSON content.",
                confidence=0.78,
                evidence=({"path": request_url, "content_type": content_type, "method": normalized_method},),
            )

        evidence = _topical_evidence_from_json_payload(
            payload=parsed,
            source_path=request_url,
            query_text=normalized_query,
            max_hits=max_hits,
        )
        matched = any(list(item.get("matched_terms") or []) for item in evidence)
        structure_aware = bool(evidence and str(evidence[0].get("structure_kind", "scalar")) != "scalar")
        evidence_label = "JSON structures" if structure_aware else "JSON fields"
        if matched:
            snippets = "; ".join(
                f"{item.get('json_path', '$')} ({item['snippet']})"
                for item in evidence[:3]
            )
            actual_outcome = f"API request {normalized_method} extracted {len(evidence)} relevant {evidence_label} from '{request_url}': {snippets}"
            confidence = min(0.98, 0.82 + 0.04 * min(len(evidence), 4) + (0.04 if structure_aware else 0.0))
            summary = f"Verified API request {normalized_method} extracted {len(evidence)} relevant {evidence_label}."
        else:
            snippets = "; ".join(
                f"{item.get('json_path', '$')} ({item['snippet']})"
                for item in evidence[:3]
            )
            extra = " The response was truncated to fit the bounded payload budget." if truncated else ""
            actual_outcome = f"API request {normalized_method} parsed JSON from '{request_url}' and extracted {len(evidence)} leading {evidence_label.lower()}: {snippets}{extra}"
            confidence = 0.80 if structure_aware else 0.78
            summary = f"Verified API request {normalized_method} parsed JSON successfully but found no direct query-specific match."
    topics = _dedupe_topics([*list(_workspace_terms(normalized_query)), urlparse(request_url).netloc])
    return DigitalActionResult(
        action_id=_action_id("api_request", {**payload, "hit_count": len(evidence)}, recorded_at=recorded_at),
        action_type="api_request",
        inputs={**dict(payload), "url": request_url, "hit_count": int(len(evidence)), "content_type": content_type},
        predicted_outcome=predicted,
        actual_outcome=actual_outcome,
        verification=ActionVerification(
            status="verified",
            success=True,
            confidence=float(confidence),
            contradiction=False,
            summary=summary,
            evidence=tuple(dict(item) for item in evidence),
        ),
        topics=topics,
        recorded_at=recorded_at,
        episode_text=(
            f"Digital action api_request for '{request_url}'. Predicted outcome: {predicted or 'no explicit prediction provided'}. "
            f"Actual outcome: {actual_outcome}. Verification: {summary}"
        )[:640],
    )


def execute_digital_action(root: Path, action_spec: Mapping[str, Any]) -> DigitalActionResult:
    action_type = _normalize_text(action_spec.get("action_type", action_spec.get("type"))).lower()
    if action_type == "workspace_search":
        return execute_workspace_search(
            root,
            query_text=_normalize_text(action_spec.get("query_text", action_spec.get("query"))),
            predicted_outcome=_normalize_text(action_spec.get("predicted_outcome")),
            max_hits=max(1, int(action_spec.get("max_hits", 6))),
            max_files=max(1, int(action_spec.get("max_files", 256))),
            max_file_bytes=max(1000, int(action_spec.get("max_file_bytes", 200_000))),
        )
    if action_type == "workspace_read":
        return execute_workspace_read(
            root,
            path=_normalize_text(action_spec.get("path")),
            query_text=_normalize_text(action_spec.get("query_text", action_spec.get("query"))),
            predicted_outcome=_normalize_text(action_spec.get("predicted_outcome")),
            max_hits=max(1, int(action_spec.get("max_hits", 6))),
            max_file_bytes=max(1000, int(action_spec.get("max_file_bytes", 200_000))),
        )
    if action_type == "web_fetch":
        return execute_web_fetch(
            root,
            url=_normalize_text(action_spec.get("url")),
            query_text=_normalize_text(action_spec.get("query_text", action_spec.get("query"))),
            predicted_outcome=_normalize_text(action_spec.get("predicted_outcome")),
            max_hits=max(1, int(action_spec.get("max_hits", 6))),
            max_response_bytes=max(1000, int(action_spec.get("max_response_bytes", action_spec.get("max_file_bytes", 200_000)))),
            timeout_seconds=max(0.1, float(action_spec.get("timeout_seconds", 10.0))),
        )
    if action_type == "api_request":
        return execute_api_request(
            root,
            url=_normalize_text(action_spec.get("url")),
            query_text=_normalize_text(action_spec.get("query_text", action_spec.get("query"))),
            predicted_outcome=_normalize_text(action_spec.get("predicted_outcome")),
            method=_normalize_text(action_spec.get("method", "GET")),
            params=action_spec.get("params") if isinstance(action_spec.get("params"), Mapping) else action_spec.get("params"),
            json_body=action_spec.get("json_body"),
            expected_json_paths=action_spec.get("expected_json_paths"),
            expected_json_values=action_spec.get("expected_json_values") if isinstance(action_spec.get("expected_json_values"), Mapping) else action_spec.get("expected_json_values"),
            expected_json_predicates=action_spec.get("expected_json_predicates"),
            expected_json_predicate_groups=action_spec.get("expected_json_predicate_groups"),
            expected_response_shape=_normalize_text(action_spec.get("expected_response_shape")),
            max_hits=max(1, int(action_spec.get("max_hits", 6))),
            max_response_bytes=max(1000, int(action_spec.get("max_response_bytes", action_spec.get("max_file_bytes", 200_000)))),
            timeout_seconds=max(0.1, float(action_spec.get("timeout_seconds", 10.0))),
        )
    raise ValueError(f"Unsupported digital action type: {action_type or 'unknown'}")
