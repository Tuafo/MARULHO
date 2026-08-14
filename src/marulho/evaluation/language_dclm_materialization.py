"""Materialize V79's bounded DCLM-Edu train/holdout corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping
from uuid import uuid4

import pyarrow.parquet as pq
import torch

from marulho.evaluation.language_base_scale_continuation import (
    PARENT_SHA256 as V77_CHECKPOINT_SHA256,
    V78_CHECKPOINT_SHA256,
)
from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_MARKER,
    DOCUMENT_TOKENS,
    TOKENIZER_SHA256,
    _atomic_json,
    file_sha256,
)
from marulho.training.language_model import load_language_model_checkpoint


ROOT = Path(__file__).resolve().parents[3]
V78_CHECKPOINT = (
    ROOT
    / "reports/language_scaling/"
    "v78-unique-document-qualified-100m-257m-20260814.pt"
)
SURFACE = "marulho_dclm_edu_materialization.v79"
SOURCE_URL = (
    "https://huggingface.co/datasets/HuggingFaceTB/dclm-edu/resolve/"
    "refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet"
)
SOURCE_PATH = ROOT / "reports/language_curriculum/.downloads/dclm-edu-0000.parquet"
SOURCE_SHA256 = "187504cc626d1121cfa867ae4e0a9879dea306a6027a7f96b443e2b4b40d70fc"
SOURCE_SIZE_BYTES = 1_330_073_247
SOURCE_ROW_COUNT = 351_429
HOLDOUT_DOCUMENTS = 512
TRAIN_DOCUMENTS = 16_384
SELECTED_DOCUMENTS = HOLDOUT_DOCUMENTS + TRAIN_DOCUMENTS
MINIMUM_CHARACTERS = 2_000
MAXIMUM_CHARACTERS = 100_000
MINIMUM_LANGUAGE_SCORE = 0.90
MINIMUM_EDU_INT_SCORE = 3
REJECTED_TEMPLATE_PHRASES = (
    "this chapter will",
    "this course unit will",
    "will delve into",
)


def _normalized_text_sha256(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _token_tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _filter_reason(row: Mapping[str, Any]) -> str | None:
    text = row.get("text")
    if row.get("language") != "en":
        return "language_not_en"
    if float(row.get("language_score") or 0.0) < MINIMUM_LANGUAGE_SCORE:
        return "language_score_below_0_90"
    if int(row.get("edu_int_score") or 0) < MINIMUM_EDU_INT_SCORE:
        return "edu_int_score_below_3"
    if not isinstance(text, str):
        return "text_missing"
    if len(text) < MINIMUM_CHARACTERS:
        return "text_below_2000_characters"
    if len(text) > MAXIMUM_CHARACTERS:
        return "text_above_100000_characters"
    if DOCUMENT_MARKER in text:
        return "contains_marulho_document_marker"
    folded = text.casefold()
    if any(phrase in folded for phrase in REJECTED_TEMPLATE_PHRASES):
        return "contains_rejected_template_phrase"
    return None


def _rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    parquet = pq.ParquetFile(path)
    row_index = 0
    columns = (
        "text",
        "id",
        "edu_int_score",
        "edu_score",
        "language",
        "language_score",
        "url",
    )
    for batch in parquet.iter_batches(batch_size=256, columns=columns):
        rows = batch.to_pylist()
        for row in rows:
            yield row_index, row
            row_index += 1


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


def _atomic_selected_text(path: Path, texts: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for text in texts:
                stream.write(f"{DOCUMENT_MARKER}\n")
                stream.write(text.strip())
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def materialize_dclm(
    *,
    source_path: Path,
    tensor_output_path: Path,
    text_output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    for output in (tensor_output_path, text_output_path, report_output_path):
        if output.exists():
            raise ValueError(f"V79 materialization output already exists: {output}")
    if source_path.stat().st_size != SOURCE_SIZE_BYTES:
        raise RuntimeError("V79 DCLM source size changed")
    source_hash = file_sha256(source_path)
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(f"V79 DCLM source hash changed: {source_hash}")
    parquet = pq.ParquetFile(source_path)
    if parquet.metadata.num_rows != SOURCE_ROW_COUNT:
        raise RuntimeError("V79 DCLM row count changed")
    checkpoint_hash = file_sha256(V78_CHECKPOINT)
    if checkpoint_hash != V78_CHECKPOINT_SHA256:
        raise RuntimeError(f"V79 V78 checkpoint hash changed: {checkpoint_hash}")
    _model, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        V78_CHECKPOINT, map_location="cpu"
    )
    if tokenizer.vocabulary_hash() != TOKENIZER_SHA256:
        raise RuntimeError("V79 tokenizer hash changed")

    rejection_counts: dict[str, int] = {}
    seen_normalized_hashes: set[str] = set()
    selected_texts: list[str] = []
    selected_tokens: list[list[int]] = []
    selected_rows: list[dict[str, Any]] = []
    pending: list[tuple[int, dict[str, Any], str]] = []
    scanned_rows = 0

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def flush() -> None:
        if not pending or len(selected_tokens) >= SELECTED_DOCUMENTS:
            pending.clear()
            return
        texts = [row[1]["text"] for row in pending]
        encoded = tokenizer.encode_batch(texts, add_bos=True, add_eos=True)
        for (row_index, row, normalized_hash), token_ids in zip(
            pending, encoded, strict=True
        ):
            if len(token_ids) < DOCUMENT_TOKENS:
                reject("below_961_token_eligibility")
                continue
            clipped = token_ids[:DOCUMENT_TOKENS]
            selected_texts.append(row["text"])
            selected_tokens.append(clipped)
            selected_rows.append(
                {
                    "row_index": row_index,
                    "id": str(row.get("id")),
                    "url": str(row.get("url")),
                    "edu_int_score": int(row["edu_int_score"]),
                    "edu_score": float(row.get("edu_score") or 0.0),
                    "language_score": float(row["language_score"]),
                    "character_count": len(row["text"]),
                    "encoded_token_count": len(token_ids),
                    "normalized_text_sha256": normalized_hash,
                }
            )
            if len(selected_tokens) >= SELECTED_DOCUMENTS:
                break
        pending.clear()

    for row_index, row in _rows(source_path):
        scanned_rows = row_index + 1
        reason = _filter_reason(row)
        if reason is not None:
            reject(reason)
            continue
        normalized_hash = _normalized_text_sha256(row["text"])
        if normalized_hash in seen_normalized_hashes:
            reject("duplicate_normalized_text")
            continue
        seen_normalized_hashes.add(normalized_hash)
        pending.append((row_index, row, normalized_hash))
        if len(pending) >= 256:
            flush()
        if len(selected_tokens) >= SELECTED_DOCUMENTS:
            break
    flush()
    if len(selected_tokens) != SELECTED_DOCUMENTS:
        raise RuntimeError(
            f"V79 found only {len(selected_tokens)} eligible DCLM documents"
        )

    all_tokens = torch.tensor(selected_tokens, dtype=torch.int32)
    holdout_tokens = all_tokens[:HOLDOUT_DOCUMENTS].clone()
    train_tokens = all_tokens[HOLDOUT_DOCUMENTS:].clone()
    holdout_rows = selected_rows[:HOLDOUT_DOCUMENTS]
    train_rows = selected_rows[HOLDOUT_DOCUMENTS:]
    selected_ids_digest = hashlib.sha256()
    for row in selected_rows:
        selected_ids_digest.update(str(row["row_index"]).encode("ascii"))
        selected_ids_digest.update(b"\0")
        selected_ids_digest.update(row["id"].encode("utf-8"))
        selected_ids_digest.update(b"\0")

    _atomic_selected_text(text_output_path, selected_texts)
    text_hash = file_sha256(text_output_path)
    payload = {
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data_used": True,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "document_tokens": DOCUMENT_TOKENS,
        "holdout_tokens": holdout_tokens,
        "train_tokens": train_tokens,
        "holdout_rows": holdout_rows,
        "train_rows": train_rows,
        "selected_text_path": str(text_output_path),
        "selected_text_sha256": text_hash,
    }
    _atomic_torch_save(tensor_output_path, payload)
    tensor_hash = file_sha256(tensor_output_path)
    restored = torch.load(tensor_output_path, map_location="cpu")
    verification = {
        "surface_exact": restored.get("surface") == SURFACE,
        "tokenizer_exact": restored.get("tokenizer_sha256") == TOKENIZER_SHA256,
        "holdout_tensor_exact": torch.equal(restored["holdout_tokens"], holdout_tokens),
        "train_tensor_exact": torch.equal(restored["train_tokens"], train_tokens),
        "selected_text_hash_exact": file_sha256(text_output_path)
        == restored["selected_text_sha256"],
    }
    verification["passed"] = all(verification.values())
    if not verification["passed"]:
        raise RuntimeError(f"V79 materialization verification failed: {verification}")
    report = {
        "surface": SURFACE,
        "artifact_kind": "marulho_external_text_materialization",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data_used": True,
        "source": {
            "dataset": "HuggingFaceTB/dclm-edu",
            "config": "default",
            "split": "train",
            "filename": "0000.parquet",
            "url": SOURCE_URL,
            "license": "CC-BY-4.0",
            "path": str(source_path),
            "size_bytes": SOURCE_SIZE_BYTES,
            "sha256": source_hash,
            "row_count": SOURCE_ROW_COUNT,
            "recreatable_download": True,
            "delete_after_verified_materialization": True,
        },
        "filter": {
            "language": "en",
            "minimum_language_score": MINIMUM_LANGUAGE_SCORE,
            "minimum_edu_int_score": MINIMUM_EDU_INT_SCORE,
            "minimum_characters": MINIMUM_CHARACTERS,
            "maximum_characters": MAXIMUM_CHARACTERS,
            "rejected_template_phrases": list(REJECTED_TEMPLATE_PHRASES),
            "normalized_exact_deduplication": True,
            "minimum_encoded_tokens": DOCUMENT_TOKENS,
        },
        "selection": {
            "order": "first_eligible_in_parquet_row_order",
            "scanned_rows": scanned_rows,
            "holdout_documents": HOLDOUT_DOCUMENTS,
            "train_documents": TRAIN_DOCUMENTS,
            "selected_documents": SELECTED_DOCUMENTS,
            "selected_ids_sha256": selected_ids_digest.hexdigest(),
            "rejection_counts": rejection_counts,
            "holdout_token_sha256": _token_tensor_sha256(holdout_tokens),
            "train_token_sha256": _token_tensor_sha256(train_tokens),
            "first_selected_row_index": selected_rows[0]["row_index"],
            "last_selected_row_index": selected_rows[-1]["row_index"],
        },
        "tokenizer": {
            "sha256": tokenizer.vocabulary_hash(),
            "checkpoint_path": str(V78_CHECKPOINT),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_parent_sha256": V77_CHECKPOINT_SHA256,
            "checkpoint_metadata": checkpoint_metadata,
        },
        "outputs": {
            "tensor_path": str(tensor_output_path),
            "tensor_sha256": tensor_hash,
            "tensor_size_bytes": tensor_output_path.stat().st_size,
            "selected_text_path": str(text_output_path),
            "selected_text_sha256": text_hash,
            "selected_text_size_bytes": text_output_path.stat().st_size,
        },
        "verification": verification,
        "decision": "freeze_v79_dclm_materialization",
    }
    _atomic_json(report_output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_dclm(
        source_path=args.source,
        tensor_output_path=args.tensor_output,
        text_output_path=args.text_output,
        report_output_path=args.report_output,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "scanned_rows": report["selection"]["scanned_rows"],
                "selected_documents": report["selection"]["selected_documents"],
                "tensor_sha256": report["outputs"]["tensor_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
