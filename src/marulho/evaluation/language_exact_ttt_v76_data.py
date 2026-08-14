"""Frozen real-document contract shared by V76 Stage A1 runners."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[3]
TRAIN_SOURCES = (
    ROOT / "reports/language_curriculum/fineweb-edu-replay-75k-shard2-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-replay-75k-shard4-20260710.txt",
)
EVAL_SOURCES = (
    ROOT / "reports/language_curriculum/fineweb-edu-eval-10k-shard1-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-eval-10k-shard2-20260710.txt",
)
SOURCE_NAMES = ("fineweb_edu", "cosmopedia_v2")
DOCUMENT_MARKER = "<|MARULHO_DOCUMENT|>"
TRAIN_DOCUMENTS_PER_SOURCE = 4096
EVAL_DOCUMENTS_PER_SOURCE = 512
DOCUMENT_TOKENS = 961
SEGMENT_LENGTH = 320
SEGMENTS = 3
DATA_SEED = 72121


def _iter_documents(path: Path) -> Iterable[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == DOCUMENT_MARKER:
                document = "".join(lines).strip()
                if document:
                    yield document
                lines.clear()
            else:
                lines.append(line)
    document = "".join(lines).strip()
    if document:
        yield document


def _select_documents(
    path: Path,
    *,
    count: int,
    tokenizer: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected: list[list[int]] = []
    pending: list[str] = []
    parsed = 0
    eligible = 0

    def flush() -> None:
        nonlocal eligible
        if not pending or len(selected) >= count:
            pending.clear()
            return
        for token_ids in tokenizer.encode_batch(pending, add_bos=True, add_eos=True):
            if len(token_ids) < DOCUMENT_TOKENS:
                continue
            eligible += 1
            selected.append(token_ids[:DOCUMENT_TOKENS])
            if len(selected) >= count:
                break
        pending.clear()

    for document in _iter_documents(path):
        parsed += 1
        pending.append(document)
        if len(pending) >= 128:
            flush()
        if len(selected) >= count:
            break
    flush()
    if len(selected) != count:
        raise RuntimeError(
            f"{path.name} has only {len(selected)} eligible documents; expected {count}"
        )
    tensor = torch.tensor(selected, dtype=torch.int32)
    return tensor, {
        "path": str(path),
        "requested": count,
        "parsed_before_completion": parsed,
        "eligible_before_completion": eligible,
        "selected_token_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "document_tokens": DOCUMENT_TOKENS,
        "selection": "first_eligible_in_file_order",
    }


def prepare_v76_language_data(tokenizer: Any) -> dict[str, Any]:
    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    selections: dict[str, Any] = {"train": {}, "eval": {}}
    for name, path in zip(SOURCE_NAMES, TRAIN_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=TRAIN_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
        )
        train_parts.append(tensor)
        selections["train"][name] = report
    for name, path in zip(SOURCE_NAMES, EVAL_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=EVAL_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
        )
        eval_parts.append(tensor)
        selections["eval"][name] = report
    train_documents = torch.cat(train_parts, dim=0)
    eval_documents = torch.cat(eval_parts, dim=0)
    eval_sources = torch.cat(
        [
            torch.full((EVAL_DOCUMENTS_PER_SOURCE,), index, dtype=torch.long)
            for index in range(len(SOURCE_NAMES))
        ]
    )
    schedule = torch.randperm(
        int(train_documents.shape[0]),
        generator=torch.Generator().manual_seed(DATA_SEED),
    )
    digest = hashlib.sha256()
    digest.update(train_documents.numpy().tobytes())
    digest.update(eval_documents.numpy().tobytes())
    digest.update(schedule.numpy().tobytes())
    return {
        "train_documents": train_documents,
        "eval_documents": eval_documents,
        "eval_sources": eval_sources,
        "schedule": schedule,
        "contract_sha256": digest.hexdigest(),
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "selections": selections,
    }


def select_document_batch(
    documents: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    return documents.index_select(0, indices).to(
        device=device, dtype=torch.long, non_blocking=False
    )
