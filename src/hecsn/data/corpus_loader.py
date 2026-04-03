from __future__ import annotations

import gzip
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Iterator, Literal, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SourceType = Literal["auto", "file", "hf", "web"]

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


def _looks_like_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
        web_max_chars: int = 200000,
        web_timeout_seconds: float = 20.0,
    ) -> None:
        self.source = source
        self.source_type = source_type
        self.text_field = text_field
        self.hf_config = hf_config
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

        if self.hf_config:
            ds = load_dataset(self.source, self.hf_config, split="train", streaming=True)
        else:
            ds = load_dataset(self.source, split="train", streaming=True)
        for row in ds:
            text = str(row.get(self.text_field, ""))
            for ch in text:
                if ch.isprintable() or ch.isspace():
                    yield ch

    def _web_text(self) -> str:
        request = Request(
            self.source,
            headers={
                "User-Agent": "HECSN/1.0 (+https://github.com/) open-web loader",
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
