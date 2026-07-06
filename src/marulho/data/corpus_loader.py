from __future__ import annotations

import gzip
import json
import os
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from queue import Empty, Full, Queue
import re
from threading import Event, Thread
from typing import Any, Generic, Iterator, Literal, Mapping, Optional, Sequence, TypeVar, cast
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


SourceType = Literal["auto", "file", "hf", "web"]
_StreamItemT = TypeVar("_StreamItemT")
_HF_TOKEN_ENV_NAMES: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_API_KEY",
)

_NOISE_LINE_PREFIXES = (
    "jump to content",
    "donate",
    "create account",
    "log in",
    "view history",
    "privacy policy",
    "about wikipedia",
    "disclaimers",
    "contact wikipedia",
    "cookie statement",
    "mobile view",
)
_PRIMARY_SECTION_RE = re.compile(r"<(main|article)\b[^>]*>(.*?)</\1>", flags=re.IGNORECASE | re.DOTALL)
_HTML_STRIP_PATTERNS = (
    r"<!--.*?-->",
    r"<script\b.*?</script>",
    r"<style\b.*?</style>",
    r"<noscript\b.*?</noscript>",
    r"<svg\b.*?</svg>",
    r"<iframe\b.*?</iframe>",
    r"<footer\b.*?</footer>",
    r"<header\b.*?</header>",
    r"<nav\b.*?</nav>",
    r"<aside\b.*?</aside>",
    r"<form\b.*?</form>",
    r"<figure\b.*?</figure>",
    r"<sup\b.*?</sup>",
)
_HTML_BLOCK_TAG_RE = re.compile(r"</?(p|div|section|article|main|aside|li|ul|ol|h[1-6]|tr|td|th|br|hr)\b[^>]*>", flags=re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MEDIAWIKI_COMMENT_RE = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_MEDIAWIKI_REF_RE = re.compile(r"<ref\b[^>/]*/\s*>|<ref\b[^>]*>.*?</ref>", flags=re.IGNORECASE | re.DOTALL)
_MEDIAWIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", flags=re.DOTALL)
_MEDIAWIKI_MEDIA_LINE_RE = re.compile(r"^\s*\[\[(?:File|Image|Category):.*$", flags=re.IGNORECASE | re.MULTILINE)
_MEDIAWIKI_INTERNAL_LINK_RE = re.compile(r"\[\[([^\[\]]+)\]\]")
_MEDIAWIKI_EXTERNAL_LINK_RE = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
_MEDIAWIKI_HEADING_RE = re.compile(r"^\s*=+\s*(.*?)\s*=+\s*$", flags=re.MULTILINE)
_MEDIAWIKI_MAGIC_WORD_RE = re.compile(r"__[^_]+__")
_MEDIAWIKI_LIST_PREFIX_RE = re.compile(r"^\s*[:*#;]+\s*", flags=re.MULTILINE)
_MEDIAWIKI_TABLE_LINE_RE = re.compile(r"^\s*(?:\{\||\|\}|[|!].*)$", flags=re.MULTILINE)
_TEXT_FIELD_SEPARATOR_RE = re.compile(r"[,|+]")
_STRUCTURED_TEXT_PRIORITY_KEYS = (
    "role",
    "content",
    "text",
    "problem",
    "question",
    "generated_solution",
    "solution",
    "expected_answer",
    "response",
    "response1",
    "response2",
    "context",
    "reasoning",
)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _looks_like_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def huggingface_token_from_env() -> str | None:
    for env_name in _HF_TOKEN_ENV_NAMES:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


def project_dataset_columns(dataset: Any, columns: Sequence[str] | None) -> Any:
    requested = [str(column).strip() for column in list(columns or []) if str(column).strip()]
    if not requested:
        return dataset
    select_columns = getattr(dataset, "select_columns", None)
    if callable(select_columns):
        try:
            return select_columns(requested)
        except Exception:
            return dataset
    return dataset


def _text_field_names(text_field: str) -> list[str]:
    fields = [
        item.strip()
        for item in _TEXT_FIELD_SEPARATOR_RE.split(str(text_field or "text"))
        if item.strip()
    ]
    return fields or ["text"]


def _structured_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        role = _normalize_text(value.get("role"))
        content = _structured_text_value(value.get("content"))
        if content and role:
            return f"{role}: {content}"
        if content:
            return content
        parts: list[str] = []
        seen: set[str] = set()
        for key in _STRUCTURED_TEXT_PRIORITY_KEYS:
            if key in value and key not in {"role", "content"}:
                text = _structured_text_value(value.get(key))
                if text:
                    parts.append(text)
                    seen.add(key)
        for key, item in value.items():
            if key in seen or key in {"role", "content"}:
                continue
            text = _structured_text_value(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "\n".join(
            text for item in value if (text := _structured_text_value(item))
        )
    return str(value)


def extract_dataset_row_text(row: Mapping[str, Any], text_field: str = "text") -> str:
    fields = _text_field_names(text_field)
    parts = [
        text
        for field in fields
        if (text := _structured_text_value(row.get(field)))
    ]
    return "\n".join(parts)


def load_hf_first_rows(
    source: str,
    *,
    hf_config: str | None = None,
    split: str = "train",
    columns: Sequence[str] | None = None,
    max_rows: int = 10,
    timeout_seconds: float = 20.0,
) -> list[dict[str, Any]]:
    query = {
        "dataset": str(source),
        "split": str(split or "train"),
    }
    if hf_config not in (None, "", "None"):
        query["config"] = str(hf_config)
    url = "https://datasets-server.huggingface.co/first-rows?" + urlencode(query)
    headers = {
        "User-Agent": "MARULHO/1.0 (+https://github.com/) hf-first-rows-loader",
        "Accept": "application/json",
    }
    token = huggingface_token_from_env()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=float(timeout_seconds)) as response:
        payload = json.load(response)
    requested = [str(column).strip() for column in list(columns or []) if str(column).strip()]
    rows: list[dict[str, Any]] = []
    for item in list(payload.get("rows") or [])[: max(1, int(max_rows))]:
        row = dict(item.get("row") or {})
        if requested:
            row = {key: row.get(key) for key in requested if key in row}
        rows.append(row)
    return rows


def _normalize_plain_text(text: str, *, max_chars: int | None = None) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lower = line.lower()
        if any(lower.startswith(prefix) for prefix in _NOISE_LINE_PREFIXES):
            continue
        lines.append(line)
    normalized = "\n".join(lines).strip()
    return normalized[:max_chars] if max_chars is not None else normalized


def _looks_like_mediawiki_raw(payload: str, *, content_type: str | None = None) -> bool:
    if content_type and "wiki" in content_type.lower():
        return True
    markers = sum(token in payload for token in ("[[", "{{", "==", "'''"))
    return markers >= 2


def _looks_like_json_payload(payload: str, *, content_type: str | None = None) -> bool:
    if content_type and "json" in content_type.lower():
        return True
    stripped = payload.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


def _openalex_abstract_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    ordered_tokens: dict[int, str] = {}
    for raw_token, raw_positions in value.items():
        token = _normalize_text(raw_token)
        if not token:
            continue
        if not isinstance(raw_positions, Sequence) or isinstance(raw_positions, (str, bytes)):
            continue
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if position >= 0 and position not in ordered_tokens:
                ordered_tokens[position] = token
    if not ordered_tokens:
        return ""
    return _normalize_text(" ".join(token for _position, token in sorted(ordered_tokens.items())))


def _openalex_topic_terms(record: Mapping[str, Any]) -> list[str]:
    terms: list[str] = []

    def add_topic_mapping(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        display_name = _normalize_text(value.get("display_name"))
        if display_name:
            terms.append(display_name)
        for key in ("subfield", "field", "domain"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                nested_name = _normalize_text(nested.get("display_name"))
                if nested_name:
                    terms.append(nested_name)

    add_topic_mapping(record.get("primary_topic"))
    for value in list(record.get("topics") or [])[:4]:
        add_topic_mapping(value)
    for value in list(record.get("keywords") or [])[:8]:
        if isinstance(value, Mapping):
            display_name = _normalize_text(value.get("display_name"))
            if display_name:
                terms.append(display_name)
    for value in list(record.get("concepts") or [])[:8]:
        if isinstance(value, Mapping):
            display_name = _normalize_text(value.get("display_name"))
            if display_name:
                terms.append(display_name)
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        lowered = term.lower()
        if not term or lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(term)
    return deduped[:16]


def _openalex_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        identifier = _normalize_text(value.get("id"))
        if identifier.startswith("https://openalex.org/") or any(
            key in value
            for key in ("display_name", "title", "abstract_inverted_index", "primary_topic")
        ):
            return [value]
        results = value.get("results")
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes)):
            return [item for item in results if isinstance(item, Mapping)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _normalize_openalex_json_text(payload: str, *, max_chars: int | None = None) -> str:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    records = _openalex_records(decoded)
    if not records:
        return ""

    sections: list[str] = []
    for record in records[:3]:
        title = _normalize_text(record.get("display_name") or record.get("title"))
        abstract = _openalex_abstract_text(record.get("abstract_inverted_index"))
        source_name = _normalize_text((((record.get("primary_location") or {}).get("source") or {}).get("display_name")))
        topics = _openalex_topic_terms(record)
        authors = [
            _normalize_text(((authorship.get("author") or {}).get("display_name")))
            for authorship in list(record.get("authorships") or [])[:5]
            if isinstance(authorship, Mapping)
        ]
        authors = [author for author in authors if author]
        section_parts = [
            part
            for part in (
                title,
                abstract,
                f"Source {source_name}" if source_name else "",
                f"Topics: {', '.join(topics)}" if topics else "",
                f"Authors: {', '.join(authors)}" if authors else "",
            )
            if part
        ]
        if section_parts:
            sections.append("\n".join(section_parts))
    return _normalize_plain_text("\n\n".join(sections), max_chars=max_chars)


def _normalize_mediawiki_text(text: str, *, max_chars: int | None = None) -> str:
    normalized = _MEDIAWIKI_COMMENT_RE.sub(" ", text)
    normalized = _MEDIAWIKI_REF_RE.sub(" ", normalized)
    for _ in range(12):
        updated = _MEDIAWIKI_TEMPLATE_RE.sub(" ", normalized)
        if updated == normalized:
            break
        normalized = updated
    normalized = _MEDIAWIKI_MEDIA_LINE_RE.sub(" ", normalized)
    normalized = _MEDIAWIKI_HEADING_RE.sub(lambda match: f"\n{match.group(1)}\n", normalized)

    def replace_internal_link(match: re.Match[str]) -> str:
        target = match.group(1)
        parts = [part.strip() for part in target.split("|") if part.strip()]
        if not parts:
            return " "
        return parts[-1].replace("_", " ")

    normalized = _MEDIAWIKI_INTERNAL_LINK_RE.sub(replace_internal_link, normalized)
    normalized = _MEDIAWIKI_EXTERNAL_LINK_RE.sub(lambda match: f" {(match.group(2) or '').strip()} ", normalized)
    normalized = _MEDIAWIKI_MAGIC_WORD_RE.sub(" ", normalized)
    normalized = re.sub(r"'{2,5}", "", normalized)
    normalized = _MEDIAWIKI_TABLE_LINE_RE.sub(" ", normalized)
    normalized = _MEDIAWIKI_LIST_PREFIX_RE.sub("", normalized)
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    return _normalize_plain_text(unescape(normalized), max_chars=max_chars)


class _VisibleTextExtractor(HTMLParser):
    _SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "footer",
        "header",
        "nav",
        "form",
        "button",
        "select",
        "option",
        "textarea",
        "figure",
        "figcaption",
        "sup",
    }
    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "aside",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
        "th",
    }
    _PRIMARY_TAGS = {"main", "article"}
    _VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    _NOISE_MARKERS = (
        "footer",
        "sidebar",
        "nav",
        "cookie",
        "reference",
        "editsection",
        "infobox",
        "authority-control",
        "metadata",
        "toc",
        "vector-page-toolbar",
        "mw-jump-link",
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._skip_depth = 0
        self._primary_depth = 0
        self._stack: list[tuple[bool, bool]] = []
        self._fallback_chunks: list[str] = []
        self._primary_chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = {str(key).lower(): ("" if value is None else str(value)) for key, value in attrs}
        marker = " ".join(
            part
            for part in (
                attrs_map.get("class", ""),
                attrs_map.get("id", ""),
                attrs_map.get("role", ""),
            )
            if part
        ).lower()
        should_skip = tag in self._SKIP_TAGS or any(token in marker for token in self._NOISE_MARKERS)
        is_primary = tag in self._PRIMARY_TAGS
        if is_primary:
            self._primary_depth += 1
        if should_skip:
            self._skip_depth += 1
            if tag in self._VOID_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            else:
                self._stack.append((should_skip, is_primary))
            return
        if tag in self._BLOCK_TAGS:
            self._append_break()
        if tag not in self._VOID_TAGS:
            self._stack.append((should_skip, is_primary))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        should_skip, is_primary = self._stack.pop()
        if should_skip:
            self._skip_depth = max(0, self._skip_depth - 1)
        if is_primary:
            self._primary_depth = max(0, self._primary_depth - 1)
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = " ".join(unescape(data).split()).strip()
        if not text:
            return
        self._append_text(text)

    def extracted_text(self, *, max_chars: int | None = None) -> str:
        primary = _normalize_plain_text("".join(self._primary_chunks), max_chars=max_chars)
        fallback = _normalize_plain_text("".join(self._fallback_chunks), max_chars=max_chars)
        if len(primary) >= min(512, len(fallback)):
            return primary
        return fallback

    def _append_text(self, text: str) -> None:
        if self._fallback_chunks and not self._fallback_chunks[-1].endswith((" ", "\n")):
            self._fallback_chunks.append(" ")
        self._fallback_chunks.append(text)
        if self._primary_depth > 0:
            if self._primary_chunks and not self._primary_chunks[-1].endswith((" ", "\n")):
                self._primary_chunks.append(" ")
            self._primary_chunks.append(text)

    def _append_break(self) -> None:
        if not self._fallback_chunks or not self._fallback_chunks[-1].endswith("\n"):
            self._fallback_chunks.append("\n")
        if self._primary_depth > 0 and (not self._primary_chunks or not self._primary_chunks[-1].endswith("\n")):
            self._primary_chunks.append("\n")


def extract_web_text(payload: str, *, content_type: str | None = None, max_chars: int | None = None) -> str:
    looks_like_html = bool(content_type and "html" in content_type.lower()) or bool(re.search(r"<\s*(html|body|main|article)\b", payload[:4096], flags=re.IGNORECASE))
    if not looks_like_html:
        if _looks_like_json_payload(payload, content_type=content_type):
            openalex_text = _normalize_openalex_json_text(payload, max_chars=max_chars)
            if openalex_text:
                return openalex_text
        if _looks_like_mediawiki_raw(payload, content_type=content_type):
            return _normalize_mediawiki_text(payload, max_chars=max_chars)
        return _normalize_plain_text(payload, max_chars=max_chars)

    primary_matches = [match[1] for match in _PRIMARY_SECTION_RE.findall(payload)]
    html_fragment = max(primary_matches, key=len) if primary_matches else payload
    cleaned = html_fragment
    for pattern in _HTML_STRIP_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = _HTML_BLOCK_TAG_RE.sub("\n", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\[[0-9]+\]", " ", cleaned)
    return _normalize_plain_text(unescape(cleaned), max_chars=max_chars)


class _StreamFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


_STREAM_END = object()


class BackgroundPrefetchIterator(Generic[_StreamItemT]):
    """Prefetch iterator items on a daemon thread and expose timeout-aware reads.

    This lets active runtime execution check whether a slow remote source has
    produced an item yet without blocking the caller on a single upstream
    `next(...)` call.
    """

    def __init__(
        self,
        iterator: Iterator[_StreamItemT],
        *,
        max_buffer: int = 1,
        name: str = "stream",
    ) -> None:
        self._iterator = iter(iterator)
        self._queue: Queue[object] = Queue(maxsize=max(1, int(max_buffer)))
        self._stop_event = Event()
        self._closed = False
        self._thread = Thread(target=self._pump, name=f"marulho-prefetch-{name}", daemon=True)
        self._thread.start()

    def __iter__(self) -> "BackgroundPrefetchIterator[_StreamItemT]":
        return self

    def __next__(self) -> _StreamItemT:
        return self.next_ready(timeout=None)

    def next_ready(self, timeout: float | None = None) -> _StreamItemT:
        if self._closed and self._queue.empty():
            raise StopIteration
        try:
            payload = self._queue.get() if timeout is None else self._queue.get(timeout=max(0.0, float(timeout)))
        except Empty as exc:
            raise TimeoutError("Background-prefetched stream has not produced an item yet.") from exc
        if payload is _STREAM_END:
            self._closed = True
            raise StopIteration
        if isinstance(payload, _StreamFailure):
            self._closed = True
            raise payload.error
        return cast(_StreamItemT, payload)

    def close(self) -> None:
        self._closed = True
        self._stop_event.set()
        close = getattr(self._iterator, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _pump(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = next(self._iterator)
                if self._stop_event.is_set():
                    break
                if not self._put_payload(item):
                    return
        except StopIteration:
            self._put_payload(_STREAM_END)
        except BaseException as exc:  # pragma: no cover - background guard
            self._put_payload(_StreamFailure(exc))

    def _put_payload(self, payload: object) -> bool:
        while not self._stop_event.is_set():
            try:
                self._queue.put(payload, timeout=0.05)
                return True
            except Full:
                continue
        return False


class StreamingCorpusLoader:
    """Streaming character loader for local files, HF datasets, and web pages.

    For local sources (.txt/.gz), streams one printable character at a time.
    For HuggingFace sources, streams from split='train' and text field.
    For web sources, fetches one page and emits visible text content.
    """

    def __init__(
        self,
        source: str,
        source_type: SourceType = "auto",
        text_field: str = "text",
        hf_config: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_split: str = "train",
        web_max_chars: int = 200000,
        web_timeout_seconds: float = 20.0,
    ) -> None:
        self.source = source
        self.source_type = source_type
        self.text_field = text_field
        self.hf_config = hf_config
        self.hf_token = None if hf_token in (None, "") else str(hf_token)
        self.hf_split = str(hf_split or "train")
        self.web_max_chars = max(1000, int(web_max_chars))
        self.web_timeout_seconds = float(web_timeout_seconds)

        if self.source_type == "auto":
            if _looks_like_url(self.source):
                self.source_type = "web"
            else:
                path = Path(self.source)
                self.source_type = "file" if path.exists() else "hf"

    def _file_char_stream(self) -> Iterator[str]:
        path = Path(self.source)
        if not path.exists():
            raise FileNotFoundError(f"File source not found: {self.source}")

        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if ch.isprintable() or ch.isspace():
                    yield ch

    def _hf_char_stream(self) -> Iterator[str]:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "HuggingFace streaming requires the 'datasets' package. "
                "Install it with: pip install datasets"
            ) from exc

        split = self.hf_split or "train"
        token = self.hf_token or huggingface_token_from_env()
        load_kwargs: dict[str, Any] = {"split": split, "streaming": True}
        if token:
            load_kwargs["token"] = token
        try:
            if self.hf_config:
                ds = load_dataset(self.source, self.hf_config, **load_kwargs)
            else:
                ds = load_dataset(self.source, **load_kwargs)
        except TypeError:
            legacy_kwargs = dict(load_kwargs)
            if token:
                legacy_kwargs.pop("token", None)
                legacy_kwargs["use_auth_token"] = token
            if self.hf_config:
                ds = load_dataset(self.source, self.hf_config, **legacy_kwargs)
            else:
                ds = load_dataset(self.source, **legacy_kwargs)
        ds = project_dataset_columns(ds, _text_field_names(self.text_field))
        for row in ds:
            text = extract_dataset_row_text(row, self.text_field)
            for ch in text:
                if ch.isprintable() or ch.isspace():
                    yield ch

    def _web_text(self) -> str:
        request = Request(
            self.source,
            headers={
                "User-Agent": "MARULHO/1.0 (+https://github.com/) open-web loader",
                "Accept": "text/html,text/plain;q=0.9,*/*;q=0.1",
            },
        )
        with urlopen(request, timeout=self.web_timeout_seconds) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type")
            encoding = response.headers.get_content_charset() or "utf-8"

        try:
            decoded = payload.decode(encoding, errors="ignore")
        except LookupError:
            decoded = payload.decode("utf-8", errors="ignore")

        visible_text = extract_web_text(decoded, content_type=content_type, max_chars=self.web_max_chars)
        if not visible_text:
            raise RuntimeError(f"Web source did not yield visible text: {self.source}")
        return visible_text

    def _web_char_stream(self) -> Iterator[str]:
        for ch in self._web_text():
            if ch.isprintable() or ch.isspace():
                yield ch

    def char_stream(self) -> Iterator[str]:
        if self.source_type == "file":
            return self._file_char_stream()
        if self.source_type == "hf":
            return self._hf_char_stream()
        if self.source_type == "web":
            return self._web_char_stream()
        raise ValueError(f"Unsupported source_type: {self.source_type}")
