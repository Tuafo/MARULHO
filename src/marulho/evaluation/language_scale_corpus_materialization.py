"""Materialize V80's deduplicated all-source scale corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4

import pyarrow.parquet as pq
import torch

from marulho.evaluation.language_dclm_materialization import (
    MAXIMUM_CHARACTERS,
    MINIMUM_CHARACTERS,
    MINIMUM_EDU_INT_SCORE,
    MINIMUM_LANGUAGE_SCORE,
    REJECTED_TEMPLATE_PHRASES,
    _filter_reason as _dclm_filter_reason,
    _normalized_text_sha256,
    _token_tensor_sha256,
)
from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_MARKER,
    DOCUMENT_TOKENS,
    EVAL_DOCUMENTS_PER_SOURCE,
    ROOT,
    TOKENIZER_SHA256,
    _atomic_json,
    _iter_documents,
    _select_documents,
    file_sha256,
)
from marulho.training.language_model import (
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_language_scale_corpus_materialization.v80"
PARENT = (
    ROOT
    / "reports/language_scaling/"
    "v78-unique-document-qualified-100m-257m-20260814.pt"
)
PARENT_SHA256 = "b66753983316b5a0cf61b293d36e4fda9b15929168067a59ed95ef816da4313b"
PARENT_STATE_SHA256 = "4ebf6ae3a500a0a77a256be80bb652a3439e47310eb008c002d14312bb34b75e"
V79_DCLM_ARTIFACT = (
    ROOT
    / "reports/language_curriculum/"
    "v79-dclm-edu-selected-16896-20260814.pt"
)
V79_DCLM_ARTIFACT_SHA256 = "04812812d5f2a319a9e88132d1cd01867b98600fc45ba03f3fe78b86bf9eeea0"
V79_DCLM_HOLDOUT_SHA256 = "906f73b29c8496f098986153fe1c01a97a47db9c7cec8317d04155b401f3c9c6"
MAXIMUM_REPLACEMENT_CHARACTER_RATIO = 0.001
BATCH_SIZE = 256

FINEWEB_SOURCES = (
    (
        ROOT
        / "reports/language_curriculum/"
        "fineweb-edu-train-75k-shard0-20260710.txt",
        366_337_778,
        "75f07f85c15c971e1d6eeba623c3f8e20d794e81b9c356ad6fadff2366c99434",
    ),
    (
        ROOT
        / "reports/language_curriculum/"
        "fineweb-edu-replay-75k-shard2-20260710.txt",
        364_102_210,
        "034a3a00ea86ec097b913f6002485a6081c6adb2b66c14ddc82be7d57b13751c",
    ),
)
COSMOPEDIA_SOURCES = (
    (
        ROOT
        / "reports/language_curriculum/"
        "cosmopedia-v2-train-150k-shard1-20260710.txt",
        565_235_962,
        "c4c846e1d08965c2c3f0e615b67d5b23554965e9222eb72bbb9ecaa4d7199b65",
    ),
    (
        ROOT
        / "reports/language_curriculum/"
        "cosmopedia-v2-train-75k-shard3-20260710.txt",
        282_143_932,
        "3a135b5f9c8386ca2edd7c18deefec82cafc6e5922691324428d050158d6da51",
    ),
    (
        ROOT
        / "reports/language_curriculum/"
        "cosmopedia-v2-replay-75k-shard4-20260710.txt",
        282_834_502,
        "7b6f41e3b3d2c1871d0124dc19f212713e3c8136e9f66cb462c845354e267aa7",
    ),
)
EVAL_SOURCES = (
    (
        "fineweb_edu",
        ROOT
        / "reports/language_curriculum/"
        "fineweb-edu-eval-10k-shard1-20260710.txt",
        51_156_091,
        "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a",
    ),
    (
        "cosmopedia_v2",
        ROOT
        / "reports/language_curriculum/"
        "cosmopedia-v2-eval-10k-shard2-20260710.txt",
        37_553_596,
        "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491",
    ),
)
DCLM_SOURCES = (
    (
        ROOT / "reports/language_curriculum/.downloads/dclm-edu-0000.parquet",
        1_330_073_247,
        351_429,
        "187504cc626d1121cfa867ae4e0a9879dea306a6027a7f96b443e2b4b40d70fc",
        "https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/resolve/refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet",
    ),
    (
        ROOT / "reports/language_curriculum/.downloads/dclm-edu-0001.parquet",
        1_323_725_638,
        355_189,
        "746ed7551cdbd31242cc1b8a97913188f6e4794a3eb9b91cf82f52caeb15e6cf",
        "https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/resolve/refs%2Fconvert%2Fparquet/default/partial-train/0001.parquet",
    ),
    (
        ROOT / "reports/language_curriculum/.downloads/dclm-edu-0002.parquet",
        1_590_129_712,
        429_056,
        "b4e634ce8ad024ac2dd841360c447ba3c38efbf20b1cf6cab3ada2968b2de797",
        "https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/resolve/refs%2Fconvert%2Fparquet/default/partial-train/0002.parquet",
    ),
)


def _generic_filter_reason(text: Any) -> str | None:
    if not isinstance(text, str) or not text:
        return "text_missing"
    if DOCUMENT_MARKER in text:
        return "contains_marulho_document_marker"
    if text.count("\ufffd") / len(text) > MAXIMUM_REPLACEMENT_CHARACTER_RATIO:
        return "replacement_character_ratio_above_0_001"
    folded = text.casefold()
    if any(phrase in folded for phrase in REJECTED_TEMPLATE_PHRASES):
        return "contains_rejected_template_phrase"
    return None


def _hash_list_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _verify_files(
    rows: Iterable[tuple[Path, int, str]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path, size, expected_hash in rows:
        actual_size = path.stat().st_size
        actual_hash = file_sha256(path)
        if actual_size != size or actual_hash != expected_hash:
            raise RuntimeError(
                f"V80 source changed: {path} size={actual_size} hash={actual_hash}"
            )
        reports.append(
            {
                "path": str(path),
                "size_bytes": actual_size,
                "sha256": actual_hash,
            }
        )
    return reports


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_and_verify_artifact(
    *,
    path: Path,
    source_name: str,
    tokens: torch.Tensor,
    normalized_hashes: list[str],
    source_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if path.exists():
        raise ValueError(f"V80 corpus artifact already exists: {path}")
    token_hash = _token_tensor_sha256(tokens)
    hash_list_hash = _hash_list_sha256(normalized_hashes)
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_scale_source_tensor",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data_used": source_name == "dclm_edu",
        "source_name": source_name,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "document_tokens": DOCUMENT_TOKENS,
        "tokens": tokens,
        "token_sha256": token_hash,
        "normalized_text_sha256": normalized_hashes,
        "normalized_hash_list_sha256": hash_list_hash,
        "source_files": source_reports,
    }
    _atomic_torch_save(path, payload)
    artifact_hash = file_sha256(path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "surface_exact": restored.get("surface") == SURFACE,
        "source_exact": restored.get("source_name") == source_name,
        "tokenizer_exact": restored.get("tokenizer_sha256") == TOKENIZER_SHA256,
        "token_shape_exact": tuple(restored["tokens"].shape) == tuple(tokens.shape),
        "token_hash_exact": _token_tensor_sha256(restored["tokens"]) == token_hash,
        "normalized_hashes_exact": restored["normalized_text_sha256"]
        == normalized_hashes,
        "hash_list_exact": restored["normalized_hash_list_sha256"]
        == hash_list_hash,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"V80 artifact verification failed: {checks}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": artifact_hash,
        "documents": int(tokens.shape[0]),
        "token_sha256": token_hash,
        "normalized_hash_list_sha256": hash_list_hash,
        "verification": checks,
    }


def _local_text_rows(
    sources: Iterable[tuple[Path, int, str]],
) -> Iterator[tuple[str, dict[str, Any]]]:
    for path, _size, _hash in sources:
        for index, text in enumerate(_iter_documents(path)):
            yield text, {"file": path.name, "document_index": index}


def _first_eligible_hashes(path: Path, *, tokenizer: Any, count: int) -> list[str]:
    selected: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if not pending or len(selected) >= count:
            pending.clear()
            return
        for text, token_ids in zip(
            pending,
            tokenizer.encode_batch(pending, add_bos=True, add_eos=True),
            strict=True,
        ):
            if len(token_ids) >= DOCUMENT_TOKENS:
                selected.append(_normalized_text_sha256(text))
                if len(selected) >= count:
                    break
        pending.clear()

    for text in _iter_documents(path):
        pending.append(text)
        if len(pending) >= BATCH_SIZE:
            flush()
        if len(selected) >= count:
            break
    flush()
    if len(selected) != count:
        raise RuntimeError(f"V80 could not recover {count} eval hashes from {path}")
    return selected


def _dclm_rows() -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    columns = (
        "text",
        "id",
        "edu_int_score",
        "edu_score",
        "language",
        "language_score",
        "url",
    )
    for path, _size, _row_count, _hash, _url in DCLM_SOURCES:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=columns):
            for row_index, row in enumerate(batch.to_pylist()):
                yield row, {"file": path.name, "batch_row_index": row_index}


def _select_source(
    *,
    source_name: str,
    rows: Iterable[tuple[Any, dict[str, Any]]],
    tokenizer: Any,
    holdout_hashes: set[str],
    seen_train_hashes: set[str],
    dclm: bool,
    accepted_text_stream: Any | None = None,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    rejection_counts: Counter[str] = Counter()
    chunks: list[torch.Tensor] = []
    normalized_hashes: list[str] = []
    pending: list[tuple[str, str]] = []
    scanned = 0
    encoded = 0

    def reject(reason: str) -> None:
        rejection_counts[reason] += 1

    def flush() -> None:
        nonlocal encoded
        if not pending:
            return
        texts = [text for text, _hash in pending]
        accepted_tokens: list[list[int]] = []
        for (text, normalized_hash), token_ids in zip(
            pending,
            tokenizer.encode_batch(texts, add_bos=True, add_eos=True),
            strict=True,
        ):
            encoded += 1
            if len(token_ids) < DOCUMENT_TOKENS:
                reject("below_961_token_eligibility")
                continue
            if normalized_hash in holdout_hashes:
                reject("exact_holdout_overlap")
                continue
            if normalized_hash in seen_train_hashes:
                reject("exact_selected_duplicate")
                continue
            seen_train_hashes.add(normalized_hash)
            normalized_hashes.append(normalized_hash)
            accepted_tokens.append(token_ids[:DOCUMENT_TOKENS])
            if accepted_text_stream is not None:
                accepted_text_stream.write(f"{DOCUMENT_MARKER}\n")
                accepted_text_stream.write(text.strip())
                accepted_text_stream.write("\n")
        if accepted_tokens:
            chunks.append(torch.tensor(accepted_tokens, dtype=torch.int32))
        pending.clear()

    for row, _identity in rows:
        scanned += 1
        text = row.get("text") if dclm else row
        if dclm:
            reason = _dclm_filter_reason(row)
            if reason is not None:
                reject(reason)
                continue
        reason = _generic_filter_reason(text)
        if reason is not None:
            reject(reason)
            continue
        normalized_hash = _normalized_text_sha256(text)
        if normalized_hash in holdout_hashes:
            reject("exact_holdout_overlap")
            continue
        if normalized_hash in seen_train_hashes:
            reject("exact_selected_duplicate")
            continue
        pending.append((text, normalized_hash))
        if len(pending) >= BATCH_SIZE:
            flush()
        if scanned % 25_000 == 0:
            print(
                f"V80 materialize source={source_name} scanned={scanned} "
                f"selected={len(normalized_hashes)}",
                flush=True,
            )
    flush()
    if not chunks:
        raise RuntimeError(f"V80 selected no {source_name} documents")
    tokens = torch.cat(chunks, dim=0)
    if int(tokens.shape[0]) != len(normalized_hashes):
        raise RuntimeError(f"V80 {source_name} tensor/hash count mismatch")
    return tokens, normalized_hashes, {
        "source_name": source_name,
        "scanned_documents": scanned,
        "encoded_documents": encoded,
        "selected_documents": len(normalized_hashes),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "selection": "source_order_after_filters_and_cross_source_exact_dedup",
        "deduplication_priority": "fineweb_edu_then_dclm_edu_then_cosmopedia_v2",
    }


def _load_parent_and_holdout() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    checkpoint_hash = file_sha256(PARENT)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V80 parent changed: {checkpoint_hash}")
    model, tokenizer, metadata = load_language_model_checkpoint(PARENT, map_location="cpu")
    state_hash = language_model_state_sha256(model)
    if state_hash != PARENT_STATE_SHA256:
        raise RuntimeError(f"V80 parent state changed: {state_hash}")
    if tokenizer.vocabulary_hash() != TOKENIZER_SHA256:
        raise RuntimeError("V80 tokenizer changed")
    if metadata.get("decision") != "save_v78_unique_document_checkpoint_for_unseen_generation":
        raise RuntimeError("V80 parent qualification metadata changed")
    if file_sha256(V79_DCLM_ARTIFACT) != V79_DCLM_ARTIFACT_SHA256:
        raise RuntimeError("V80 V79 DCLM artifact changed")
    dclm = torch.load(V79_DCLM_ARTIFACT, map_location="cpu", weights_only=False)
    holdout = dclm["holdout_tokens"].to(dtype=torch.int32)
    if int(holdout.shape[0]) != EVAL_DOCUMENTS_PER_SOURCE:
        raise RuntimeError("V80 DCLM holdout count changed")
    if _token_tensor_sha256(holdout) != V79_DCLM_HOLDOUT_SHA256:
        raise RuntimeError("V80 DCLM holdout tensor changed")
    return tokenizer, {
        "path": str(PARENT),
        "sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "metadata_decision": metadata.get("decision"),
    }, dclm


def materialize(
    *,
    fineweb_output: Path,
    dclm_output: Path,
    cosmopedia_output: Path,
    eval_output: Path,
    dclm_text_output: Path,
    report_output: Path,
) -> dict[str, Any]:
    outputs = (
        fineweb_output,
        dclm_output,
        cosmopedia_output,
        eval_output,
        dclm_text_output,
        report_output,
    )
    if any(path.exists() for path in outputs):
        raise ValueError("V80 materialization output already exists")
    tokenizer, parent_audit, v79_dclm = _load_parent_and_holdout()
    fineweb_sources = _verify_files(FINEWEB_SOURCES)
    cosmopedia_sources = _verify_files(COSMOPEDIA_SOURCES)
    eval_sources = _verify_files(
        (path, size, expected_hash) for _name, path, size, expected_hash in EVAL_SOURCES
    )
    dclm_sources: list[dict[str, Any]] = []
    for path, size, rows, expected_hash, url in DCLM_SOURCES:
        report = _verify_files(((path, size, expected_hash),))[0]
        actual_rows = pq.ParquetFile(path).metadata.num_rows
        if actual_rows != rows:
            raise RuntimeError(f"V80 DCLM row count changed: {path}")
        report.update({"rows": actual_rows, "url": url, "recreatable": True})
        dclm_sources.append(report)

    holdout_hashes: set[str] = set()
    raw_eval_counts: dict[str, int] = {}
    eval_parts: list[torch.Tensor] = []
    eval_selected_hashes: list[str] = []
    eval_selection: dict[str, Any] = {}
    for name, path, _size, _hash in EVAL_SOURCES:
        raw_count = 0
        for text in _iter_documents(path):
            holdout_hashes.add(_normalized_text_sha256(text))
            raw_count += 1
        raw_eval_counts[name] = raw_count
        tensor, selection = _select_documents(
            path,
            count=EVAL_DOCUMENTS_PER_SOURCE,
            tokenizer=tokenizer,
        )
        eval_parts.append(tensor)
        eval_selected_hashes.extend(
            _first_eligible_hashes(
                path,
                tokenizer=tokenizer,
                count=EVAL_DOCUMENTS_PER_SOURCE,
            )
        )
        eval_selection[name] = selection
    dclm_holdout_hashes = [
        str(row["normalized_text_sha256"]) for row in v79_dclm["holdout_rows"]
    ]
    holdout_hashes.update(dclm_holdout_hashes)
    dclm_holdout = v79_dclm["holdout_tokens"].to(dtype=torch.int32)
    eval_parts.append(dclm_holdout)
    eval_selected_hashes.extend(dclm_holdout_hashes)
    eval_tokens = torch.cat(eval_parts, dim=0)
    eval_hashes = {
        "fineweb_eval_token_sha256": _token_tensor_sha256(eval_parts[0]),
        "cosmopedia_eval_token_sha256": _token_tensor_sha256(eval_parts[1]),
        "dclm_eval_token_sha256": _token_tensor_sha256(eval_parts[2]),
    }
    eval_normalized_hashes = sorted(holdout_hashes)
    eval_artifact = _write_and_verify_artifact(
        path=eval_output,
        source_name="three_source_holdout",
        tokens=eval_tokens,
        normalized_hashes=eval_selected_hashes,
        source_reports=[*eval_sources, {"path": str(V79_DCLM_ARTIFACT), "sha256": V79_DCLM_ARTIFACT_SHA256}],
    )
    del eval_tokens, eval_parts

    seen_train_hashes: set[str] = set()
    source_results: dict[str, Any] = {}

    fineweb_tokens, fineweb_hashes, fineweb_selection = _select_source(
        source_name="fineweb_edu",
        rows=_local_text_rows(FINEWEB_SOURCES),
        tokenizer=tokenizer,
        holdout_hashes=holdout_hashes,
        seen_train_hashes=seen_train_hashes,
        dclm=False,
    )
    fineweb_artifact = _write_and_verify_artifact(
        path=fineweb_output,
        source_name="fineweb_edu",
        tokens=fineweb_tokens,
        normalized_hashes=fineweb_hashes,
        source_reports=fineweb_sources,
    )
    source_results["fineweb_edu"] = {
        "selection": fineweb_selection,
        "artifact": fineweb_artifact,
    }
    del fineweb_tokens, fineweb_hashes

    dclm_text_output.parent.mkdir(parents=True, exist_ok=True)
    handle, dclm_text_temporary = tempfile.mkstemp(
        dir=dclm_text_output.parent, suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            dclm_tokens, dclm_hashes, dclm_selection = _select_source(
                source_name="dclm_edu",
                rows=_dclm_rows(),
                tokenizer=tokenizer,
                holdout_hashes=holdout_hashes,
                seen_train_hashes=seen_train_hashes,
                dclm=True,
                accepted_text_stream=stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(dclm_text_temporary, dclm_text_output)
    except BaseException:
        Path(dclm_text_temporary).unlink(missing_ok=True)
        raise
    dclm_artifact = _write_and_verify_artifact(
        path=dclm_output,
        source_name="dclm_edu",
        tokens=dclm_tokens,
        normalized_hashes=dclm_hashes,
        source_reports=dclm_sources,
    )
    dclm_text_hash = file_sha256(dclm_text_output)
    source_results["dclm_edu"] = {
        "selection": dclm_selection,
        "artifact": dclm_artifact,
        "selected_text": {
            "path": str(dclm_text_output),
            "size_bytes": dclm_text_output.stat().st_size,
            "sha256": dclm_text_hash,
        },
    }
    del dclm_tokens, dclm_hashes

    cosmopedia_tokens, cosmopedia_hashes, cosmopedia_selection = _select_source(
        source_name="cosmopedia_v2",
        rows=_local_text_rows(COSMOPEDIA_SOURCES),
        tokenizer=tokenizer,
        holdout_hashes=holdout_hashes,
        seen_train_hashes=seen_train_hashes,
        dclm=False,
    )
    cosmopedia_artifact = _write_and_verify_artifact(
        path=cosmopedia_output,
        source_name="cosmopedia_v2",
        tokens=cosmopedia_tokens,
        normalized_hashes=cosmopedia_hashes,
        source_reports=cosmopedia_sources,
    )
    source_results["cosmopedia_v2"] = {
        "selection": cosmopedia_selection,
        "artifact": cosmopedia_artifact,
    }
    del cosmopedia_tokens, cosmopedia_hashes

    report = {
        "surface": SURFACE,
        "artifact_kind": "marulho_scale_corpus_materialization",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data_used": True,
        "parent": parent_audit,
        "filter": {
            "minimum_encoded_tokens": DOCUMENT_TOKENS,
            "normalized_exact_deduplication": True,
            "deduplication_priority": ["fineweb_edu", "dclm_edu", "cosmopedia_v2"],
            "all_eval_documents_excluded_by_normalized_hash": True,
            "maximum_replacement_character_ratio": MAXIMUM_REPLACEMENT_CHARACTER_RATIO,
            "rejected_template_phrases": list(REJECTED_TEMPLATE_PHRASES),
            "dclm": {
                "language": "en",
                "minimum_language_score": MINIMUM_LANGUAGE_SCORE,
                "minimum_edu_int_score": MINIMUM_EDU_INT_SCORE,
                "minimum_characters": MINIMUM_CHARACTERS,
                "maximum_characters": MAXIMUM_CHARACTERS,
            },
        },
        "holdout": {
            "normalized_hash_count": len(holdout_hashes),
            "normalized_hash_list_sha256": _hash_list_sha256(eval_normalized_hashes),
            "raw_eval_document_counts": raw_eval_counts,
            "dclm_holdout_documents": len(dclm_holdout_hashes),
            "selection": eval_selection,
            "token_hashes": eval_hashes,
            "artifact": eval_artifact,
        },
        "sources": source_results,
        "selected_train_documents": len(seen_train_hashes),
        "all_artifacts_verified": all(
            row["artifact"]["verification"]["passed"]
            for row in source_results.values()
        )
        and eval_artifact["verification"]["passed"],
        "raw_dclm_delete_after_report_verification": True,
        "decision": "freeze_v80_scale_corpus",
    }
    _atomic_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fineweb-output", type=Path, required=True)
    parser.add_argument("--dclm-output", type=Path, required=True)
    parser.add_argument("--cosmopedia-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--dclm-text-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(
        fineweb_output=args.fineweb_output,
        dclm_output=args.dclm_output,
        cosmopedia_output=args.cosmopedia_output,
        eval_output=args.eval_output,
        dclm_text_output=args.dclm_text_output,
        report_output=args.report,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "selected_train_documents": report["selected_train_documents"],
                "source_documents": {
                    name: row["artifact"]["documents"]
                    for name, row in report["sources"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
