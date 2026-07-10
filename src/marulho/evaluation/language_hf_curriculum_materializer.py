from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from marulho.data.corpus_loader import (
    extract_dataset_row_text,
    huggingface_token_from_env,
)


ARTIFACT_KIND = "marulho_language_hf_curriculum_materialization"
SURFACE = "marulho_language_hf_curriculum_materialization.v1"
DATASET_VIEWER_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_VIEWER_FIRST_ROWS_URL = "https://datasets-server.huggingface.co/first-rows"


@dataclass(frozen=True)
class HFCurriculumSource:
    dataset: str
    config: str
    split: str
    text_field: str
    role: str
    license: str
    access: str = "open"
    notes: str = ""


NVIDIA_OPEN_REPAIR_V1_SOURCES: tuple[HFCurriculumSource, ...] = (
    HFCurriculumSource(
        dataset="nvidia/Nemotron-Post-Training-Dataset-v1",
        config="default",
        split="chat",
        text_field="messages",
        role="general_sft_chat",
        license="cc-by-4.0",
        notes="general instruction-following SFT rows",
    ),
    HFCurriculumSource(
        dataset="nvidia/Nemotron-Post-Training-Dataset-v1",
        config="default",
        split="math",
        text_field="messages",
        role="general_sft_math",
        license="cc-by-4.0",
        notes="Nemotron SFT math rows",
    ),
    HFCurriculumSource(
        dataset="nvidia/Nemotron-Post-Training-Dataset-v1",
        config="default",
        split="code",
        text_field="messages",
        role="general_sft_code",
        license="cc-by-4.0",
        notes="Nemotron SFT code rows",
    ),
    HFCurriculumSource(
        dataset="nvidia/OpenMathInstruct-2",
        config="default",
        split="train_1M",
        text_field="problem,generated_solution,expected_answer",
        role="math_reasoning",
        license="cc-by-4.0",
        notes="problem-solution-answer math curriculum",
    ),
    HFCurriculumSource(
        dataset="nvidia/HelpSteer3",
        config="preference",
        split="train",
        text_field="context,response1,response2,individual_preference",
        role="preference_review",
        license="cc-by-4.0",
        notes="preference rows for later reward/ranking conversion",
    ),
    HFCurriculumSource(
        dataset="nvidia/Nemotron-Competitive-Programming-v1",
        config="default",
        split="competitive_coding_python_part00",
        text_field="messages",
        role="code_reasoning",
        license="cc-by-4.0",
        notes="competitive-programming Python reasoning rows",
    ),
    HFCurriculumSource(
        dataset="nvidia/Nemotron-Personas-USA",
        config="default",
        split="train",
        text_field=(
            "persona,professional_persona,skills_and_expertise,"
            "hobbies_and_interests,career_goals_and_ambitions"
        ),
        role="persona_diversity",
        license="cc-by-4.0",
        notes="style/diversity conditioning, not factual evaluation",
    ),
)

PRESETS: dict[str, tuple[HFCurriculumSource, ...]] = {
    "nvidia-open-repair-v1": NVIDIA_OPEN_REPAIR_V1_SOURCES,
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows_url(source: HFCurriculumSource, *, offset: int, length: int) -> str:
    return DATASET_VIEWER_ROWS_URL + "?" + urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": max(0, int(offset)),
            "length": max(1, min(100, int(length))),
        }
    )


def _first_rows_url(source: HFCurriculumSource) -> str:
    return DATASET_VIEWER_FIRST_ROWS_URL + "?" + urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
        }
    )


def _dataset_viewer_get_json(url: str, *, timeout_seconds: float) -> Mapping[str, Any]:
    headers = {
        "User-Agent": "MARULHO/1.0 language-curriculum-materializer",
        "Accept": "application/json",
    }
    token = huggingface_token_from_env()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    for attempt in range(7):
        try:
            with urlopen(request, timeout=float(timeout_seconds)) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retryable = int(exc.code) == 429 or 500 <= int(exc.code) < 600
            if not retryable or attempt >= 6:
                raise
            retry_after = None if exc.headers is None else exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 0.0
            except ValueError:
                delay = 0.0
            time.sleep(max(delay, min(30.0, float(2**attempt))))
    raise RuntimeError("Dataset Viewer retry loop exited unexpectedly")


def _fetch_dataset_viewer_rows(
    source: HFCurriculumSource,
    *,
    offset: int,
    length: int,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    rows_url = _rows_url(source, offset=offset, length=length)
    try:
        return _dataset_viewer_get_json(rows_url, timeout_seconds=timeout_seconds)
    except Exception as rows_exc:
        if int(offset) != 0:
            raise
        try:
            payload = dict(
                _dataset_viewer_get_json(
                    _first_rows_url(source),
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as first_rows_exc:
            raise RuntimeError(
                f"rows endpoint failed: {rows_exc}; first-rows fallback failed: "
                f"{first_rows_exc}"
            ) from first_rows_exc
        rows = list(payload.get("rows") or [])[: max(1, min(100, int(length)))]
        payload["rows"] = rows
        payload["fallback_endpoint"] = "first-rows"
        payload["rows_endpoint_failure"] = str(rows_exc)
        return payload


def _source_report(
    source: HFCurriculumSource,
    *,
    status: str,
    requested_rows: int,
    offset: int,
    page_size: int,
    page_count: int = 0,
    fetched_rows: int = 0,
    materialized_rows: int = 0,
    character_count: int = 0,
    row_hashes: Sequence[str] = (),
    failure_reason: str | None = None,
    fallback_endpoint: str | None = None,
    rows_endpoint_failure: str | None = None,
    rows_endpoint_failures: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "surface": "marulho_language_hf_curriculum_source.v1",
        **asdict(source),
        "status": status,
        "requested_rows": int(requested_rows),
        "offset": int(offset),
        "page_size": int(page_size),
        "page_count": int(page_count),
        "fetched_rows": int(fetched_rows),
        "materialized_rows": int(materialized_rows),
        "character_count": int(character_count),
        "row_hashes": list(row_hashes),
        "failure_reason": failure_reason,
        "fallback_endpoint": fallback_endpoint,
        "rows_endpoint_failure": rows_endpoint_failure,
        "rows_endpoint_failures": list(rows_endpoint_failures),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_partial_report(
    output_path: Path,
    *,
    corpus_output_path: Path,
    source_reports: Sequence[Mapping[str, Any]],
    requested_source_count: int,
) -> None:
    _write_json(
        output_path,
        {
            "artifact_kind": ARTIFACT_KIND,
            "surface": SURFACE,
            "report_status": "partial",
            "status": "materializing",
            "corpus_output_path": str(corpus_output_path),
            "requested_source_count": int(requested_source_count),
            "completed_source_count": len(source_reports),
            "source_reports": [dict(item) for item in source_reports],
            "reports_not_run_by_service": True,
            "mutates_runtime_state": False,
            "external_llm_used": False,
            "service_owned_cognition": False,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def materialize_hf_curriculum(
    *,
    output_path: str | Path,
    corpus_output_path: str | Path | None = None,
    sources: Sequence[HFCurriculumSource] = NVIDIA_OPEN_REPAIR_V1_SOURCES,
    rows_per_source: int = 4,
    offset: int = 0,
    row_page_size: int = 100,
    request_interval_seconds: float = 0.0,
    timeout_seconds: float = 60.0,
    write_partial_after_each_source: bool = True,
) -> dict[str, Any]:
    output = Path(output_path)
    corpus_output = (
        Path(corpus_output_path)
        if corpus_output_path is not None
        else output.with_suffix(".txt")
    )
    requested_rows = max(1, int(rows_per_source))
    requested_offset = max(0, int(offset))
    page_size = max(1, min(100, int(row_page_size)))
    source_reports: list[dict[str, Any]] = []
    corpus_sections: list[str] = []

    for source in sources:
        try:
            row_items: list[Any] = []
            fallback_endpoint: str | None = None
            rows_endpoint_failure: str | None = None
            rows_endpoint_failures: list[str] = []
            page_count = 0
            next_offset = requested_offset
            page_failure: str | None = None
            while len(row_items) < requested_rows:
                page_length = min(page_size, requested_rows - len(row_items))
                try:
                    payload = _fetch_dataset_viewer_rows(
                        source,
                        offset=next_offset,
                        length=page_length,
                        timeout_seconds=float(timeout_seconds),
                    )
                except Exception as exc:
                    page_failure = str(exc) or exc.__class__.__name__
                    rows_endpoint_failure = rows_endpoint_failure or page_failure
                    rows_endpoint_failures.append(page_failure)
                    break
                page_count += 1
                page_rows = list(payload.get("rows") or [])
                row_items.extend(page_rows)
                if payload.get("fallback_endpoint"):
                    fallback_endpoint = str(payload.get("fallback_endpoint"))
                    if payload.get("rows_endpoint_failure"):
                        rows_endpoint_failure = str(payload["rows_endpoint_failure"])
                        rows_endpoint_failures.append(rows_endpoint_failure)
                    break
                if payload.get("rows_endpoint_failure"):
                    failure = str(payload["rows_endpoint_failure"])
                    rows_endpoint_failure = rows_endpoint_failure or failure
                    rows_endpoint_failures.append(failure)
                if not page_rows or len(page_rows) < page_length:
                    break
                next_offset += len(page_rows)
                if (
                    len(row_items) < requested_rows
                    and float(request_interval_seconds) > 0.0
                ):
                    time.sleep(float(request_interval_seconds))
            row_texts: list[str] = []
            row_hashes: list[str] = []
            for item in row_items:
                row = item.get("row") if isinstance(item, Mapping) else None
                if not isinstance(row, Mapping):
                    continue
                text = extract_dataset_row_text(row, source.text_field).strip()
                if not text:
                    continue
                row_texts.append(text)
                row_hashes.append(_sha256_text(text))
            if row_texts:
                header = (
                    f"### source={source.dataset} config={source.config} "
                    f"split={source.split} role={source.role}"
                )
                corpus_sections.append(header + "\n" + "\n\n".join(row_texts))
            source_reports.append(
                _source_report(
                    source,
                    status=(
                        "materialized"
                        if page_failure is None and row_texts
                        else "partial"
                        if row_texts
                        else "failed"
                    ),
                    requested_rows=requested_rows,
                    offset=requested_offset,
                    page_size=page_size,
                    page_count=page_count,
                    fetched_rows=len(row_items),
                    materialized_rows=len(row_texts),
                    character_count=sum(len(text) for text in row_texts),
                    row_hashes=row_hashes,
                    failure_reason=page_failure,
                    fallback_endpoint=fallback_endpoint,
                    rows_endpoint_failure=rows_endpoint_failure,
                    rows_endpoint_failures=rows_endpoint_failures,
                )
            )
        except Exception as exc:
            source_reports.append(
                _source_report(
                    source,
                    status="failed",
                    requested_rows=requested_rows,
                    offset=requested_offset,
                    page_size=page_size,
                    failure_reason=str(exc) or exc.__class__.__name__,
                )
            )

        if write_partial_after_each_source:
            _write_partial_report(
                output,
                corpus_output_path=corpus_output,
                source_reports=source_reports,
                requested_source_count=len(sources),
            )

    corpus_text = "\n\n".join(corpus_sections).strip()
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.write_text(corpus_text + ("\n" if corpus_text else ""), encoding="utf-8")

    failed_sources = [item for item in source_reports if item["status"] == "failed"]
    materialized_sources = [
        item for item in source_reports if int(item["materialized_rows"]) > 0
    ]
    incomplete_sources = [
        item for item in source_reports if item["status"] != "materialized"
    ]
    status = (
        "final"
        if corpus_text and not incomplete_sources and materialized_sources
        else "partial"
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "report_status": status,
        "status": (
            "materialized_curriculum"
            if status == "final"
            else "partial_curriculum_materialization"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_output_path": str(corpus_output),
        "corpus": {
            "surface": "marulho_language_hf_curriculum_corpus.v1",
            "path": str(corpus_output),
            "character_count": len(corpus_text),
            "byte_count": len(corpus_text.encode("utf-8")),
            "sha256": _sha256_text(corpus_text),
            "section_count": len(corpus_sections),
            "source_count": len(materialized_sources),
            "row_count": sum(int(item["materialized_rows"]) for item in source_reports),
        },
        "requested_source_count": len(sources),
        "materialized_source_count": len(materialized_sources),
        "failed_source_count": len(failed_sources),
        "source_reports": source_reports,
        "raw_row_payloads_retained": False,
        "reports_not_run_by_service": True,
        "mutates_runtime_state": False,
        "external_llm_used": False,
        "service_owned_cognition": False,
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
        "claim_boundary": (
            "bounded_source_materialization_for_future_training; not a "
            "quality, runtime, or promotion claim"
        ),
    }
    _write_json(output, report)
    return report


def _source_from_cli(value: str) -> HFCurriculumSource:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) < 6:
        raise ValueError(
            "--source must be dataset|config|split|text_field|role|license[|access]"
        )
    return HFCurriculumSource(
        dataset=parts[0],
        config=parts[1],
        split=parts[2],
        text_field=parts[3],
        role=parts[4],
        license=parts[5],
        access=parts[6] if len(parts) > 6 and parts[6] else "open",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize bounded Hugging Face curriculum rows for MARULHO."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=tuple(sorted(PRESETS)),
        default="nvidia-open-repair-v1",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="dataset|config|split|text_field|role|license[|access]",
    )
    parser.add_argument("--rows-per-source", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--row-page-size", type=int, default=100)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--no-partial-after-each-source",
        action="store_true",
    )
    args = parser.parse_args(argv)
    sources = (
        tuple(_source_from_cli(item) for item in args.source)
        if args.source
        else PRESETS[args.preset]
    )
    materialize_hf_curriculum(
        output_path=args.output,
        corpus_output_path=args.corpus_output,
        sources=sources,
        rows_per_source=args.rows_per_source,
        offset=args.offset,
        row_page_size=args.row_page_size,
        request_interval_seconds=max(0.0, float(args.request_interval_seconds)),
        timeout_seconds=args.timeout_seconds,
        write_partial_after_each_source=not args.no_partial_after_each_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
