from __future__ import annotations

from collections import deque
from itertools import chain
from typing import Deque, Iterable, Iterator, Optional, Sequence, TypeAlias, TypeVar

import torch

from .corpus_loader import SourceType, StreamingCorpusLoader
from .rtf_encoder import RTFEncoder


PatternExample: TypeAlias = tuple[str, torch.Tensor]

T = TypeVar("T")


def _normalize_char(ch: str) -> tuple[int, str]:
    code = ord(ch)
    if code < 128:
        return code, ch
    return 0, "?"


def pattern_stream(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    *,
    learn_chunking: bool = False,
) -> Iterator[torch.Tensor]:
    for _, pattern in encoder.iter_char_patterns(chars, window_size, learn=learn_chunking):
        yield pattern


def raw_window_stream(
    chars: Iterable[str],
    window_size: int,
) -> Iterator[str]:
    maxlen = max(1, int(window_size))
    window_chars: Deque[str] = deque(maxlen=maxlen)
    for ch in chars:
        _, display = _normalize_char(ch)
        window_chars.append(display)
        yield "".join(window_chars)


def labeled_pattern_stream(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    *,
    learn_chunking: bool = False,
) -> Iterator[PatternExample]:
    yield from encoder.iter_char_patterns(chars, window_size, learn=learn_chunking)


def _split_stream(stream: Iterable[T], first_count: int, second_count: int) -> tuple[list[T], list[T]]:
    leading_count = max(0, int(first_count))
    trailing_count = max(0, int(second_count))
    leading: list[T] = []
    trailing: list[T] = []
    needed = leading_count + trailing_count
    for idx, item in enumerate(stream, start=1):
        if idx <= leading_count:
            leading.append(item)
        elif idx <= needed:
            trailing.append(item)
        else:
            break
    return leading, trailing


def _prefixed_char_stream(chars: Iterable[str], prefix_text: str | None) -> Iterable[str]:
    prefix = str(prefix_text or "")
    if not prefix:
        return chars
    if prefix and not prefix.endswith(" "):
        prefix = f"{prefix} "
    return chain(prefix, chars)


def _prefix_patterns(
    prefix_text: str | None,
    encoder: RTFEncoder,
    window_size: int,
) -> list[torch.Tensor]:
    prefix = str(prefix_text or "")
    if not prefix:
        return []
    return list(pattern_stream(prefix, encoder, window_size))


def _prefix_examples(
    prefix_text: str | None,
    encoder: RTFEncoder,
    window_size: int,
) -> list[tuple[str, torch.Tensor]]:
    prefix = str(prefix_text or "")
    if not prefix:
        return []
    return list(labeled_pattern_stream(prefix, encoder, window_size))


def interleave_tagged_blocks(
    first: Sequence[T],
    second: Sequence[T],
    block_tokens: int,
    first_label: str,
    second_label: str,
) -> list[tuple[str, T]]:
    block_size = max(1, int(block_tokens))
    merged: list[tuple[str, T]] = []
    first_pos = 0
    second_pos = 0
    turn = 0
    while first_pos < len(first) or second_pos < len(second):
        if (turn % 2 == 0 and first_pos < len(first)) or second_pos >= len(second):
            end = min(len(first), first_pos + block_size)
            merged.extend((first_label, item) for item in first[first_pos:end])
            first_pos = end
        else:
            end = min(len(second), second_pos + block_size)
            merged.extend((second_label, item) for item in second[second_pos:end])
            second_pos = end
        turn += 1
    return merged


def build_train_eval_patterns(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return _split_stream(
        pattern_stream(chars, encoder, window_size),
        train_tokens,
        eval_tokens,
    )


def build_train_eval_examples(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[PatternExample], list[PatternExample]]:
    return _split_stream(
        labeled_pattern_stream(chars, encoder, window_size),
        train_tokens,
        eval_tokens,
    )


def build_probe_train_patterns(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    probe_tokens: int,
    train_tokens: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return _split_stream(
        pattern_stream(chars, encoder, window_size),
        probe_tokens,
        train_tokens,
    )


def build_train_eval_windows(
    chars: Iterable[str],
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[str], list[str]]:
    return _split_stream(
        raw_window_stream(chars, window_size),
        train_tokens,
        eval_tokens,
    )


def build_probe_train_examples(
    chars: Iterable[str],
    encoder: RTFEncoder,
    window_size: int,
    probe_tokens: int,
    train_tokens: int,
) -> tuple[list[torch.Tensor], list[str], list[torch.Tensor], list[str]]:
    probe_examples, train_examples = _split_stream(
        labeled_pattern_stream(chars, encoder, window_size),
        probe_tokens,
        train_tokens,
    )
    probe_patterns = [pattern for _, pattern in probe_examples]
    probe_raw_windows = [raw_window for raw_window, _ in probe_examples]
    train_patterns = [pattern for _, pattern in train_examples]
    train_raw_windows = [raw_window for raw_window, _ in train_examples]
    return probe_patterns, probe_raw_windows, train_patterns, train_raw_windows


def load_train_eval_patterns(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    encoder: RTFEncoder,
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    loader = StreamingCorpusLoader(
        source=source,
        source_type=source_type,
        text_field=text_field,
        hf_config=hf_config,
    )
    return build_train_eval_patterns(
        chars=loader.char_stream(),
        encoder=encoder,
        window_size=window_size,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
    )


def load_train_eval_examples(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    encoder: RTFEncoder,
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[PatternExample], list[PatternExample]]:
    loader = StreamingCorpusLoader(
        source=source,
        source_type=source_type,
        text_field=text_field,
        hf_config=hf_config,
    )
    return build_train_eval_examples(
        chars=loader.char_stream(),
        encoder=encoder,
        window_size=window_size,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
    )


def load_train_eval_windows(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    window_size: int,
    train_tokens: int,
    eval_tokens: int,
) -> tuple[list[str], list[str]]:
    loader = StreamingCorpusLoader(
        source=source,
        source_type=source_type,
        text_field=text_field,
        hf_config=hf_config,
    )
    return build_train_eval_windows(
        chars=loader.char_stream(),
        window_size=window_size,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
    )


def load_probe_train_patterns(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    encoder: RTFEncoder,
    window_size: int,
    probe_tokens: int,
    train_tokens: int,
    *,
    prefix_text: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    loader = StreamingCorpusLoader(
        source=source,
        source_type=source_type,
        text_field=text_field,
        hf_config=hf_config,
    )
    probe_patterns, train_patterns = build_probe_train_patterns(
        chars=_prefixed_char_stream(loader.char_stream(), prefix_text),
        encoder=encoder,
        window_size=window_size,
        probe_tokens=probe_tokens,
        train_tokens=train_tokens,
    )
    prefix_patterns = _prefix_patterns(prefix_text, encoder, window_size)
    if prefix_patterns:
        train_patterns = [*prefix_patterns, *train_patterns][: max(0, int(train_tokens))]
    return probe_patterns, train_patterns


def load_probe_train_examples(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    encoder: RTFEncoder,
    window_size: int,
    probe_tokens: int,
    train_tokens: int,
    *,
    prefix_text: str | None = None,
) -> tuple[list[torch.Tensor], list[str], list[torch.Tensor], list[str]]:
    loader = StreamingCorpusLoader(
        source=source,
        source_type=source_type,
        text_field=text_field,
        hf_config=hf_config,
    )
    probe_patterns, probe_raw_windows, train_patterns, train_raw_windows = build_probe_train_examples(
        chars=_prefixed_char_stream(loader.char_stream(), prefix_text),
        encoder=encoder,
        window_size=window_size,
        probe_tokens=probe_tokens,
        train_tokens=train_tokens,
    )
    prefix_examples = _prefix_examples(prefix_text, encoder, window_size)
    if prefix_examples:
        prefix_patterns = [pattern for _, pattern in prefix_examples]
        prefix_raw_windows = [raw_window for raw_window, _ in prefix_examples]
        train_limit = max(0, int(train_tokens))
        train_patterns = [*prefix_patterns, *train_patterns][:train_limit]
        train_raw_windows = [*prefix_raw_windows, *train_raw_windows][:train_limit]
    return probe_patterns, probe_raw_windows, train_patterns, train_raw_windows


__all__ = [
    "PatternExample",
    "build_probe_train_examples",
    "build_probe_train_patterns",
    "build_train_eval_examples",
    "build_train_eval_patterns",
    "labeled_pattern_stream",
    "load_probe_train_examples",
    "load_probe_train_patterns",
    "load_train_eval_examples",
    "load_train_eval_patterns",
    "pattern_stream",
]
