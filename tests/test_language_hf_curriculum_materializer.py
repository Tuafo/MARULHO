from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

from marulho.evaluation.language_hf_curriculum_materializer import (
    HFCurriculumSource,
    SURFACE,
    materialize_hf_curriculum,
)


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_materialize_hf_curriculum_writes_flattened_corpus_and_report(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "curriculum.json"
    corpus = tmp_path / "curriculum.txt"
    sources = (
        HFCurriculumSource(
            dataset="nvidia/Nemotron-Post-Training-Dataset-v1",
            config="default",
            split="chat",
            text_field="messages",
            role="chat_sft",
            license="cc-by-4.0",
        ),
        HFCurriculumSource(
            dataset="nvidia/OpenMathInstruct-2",
            config="default",
            split="train_1M",
            text_field="problem,generated_solution,expected_answer",
            role="math_reasoning",
            license="cc-by-4.0",
        ),
    )

    def _fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        split = query["split"][0]
        if split == "chat":
            return _Response(
                {
                    "rows": [
                        {
                            "row": {
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain recurrence.",
                                    },
                                    {
                                        "role": "assistant",
                                        "content": "Track state over time.",
                                    },
                                ]
                            }
                        }
                    ]
                }
            )
        return _Response(
            {
                "rows": [
                    {
                        "row": {
                            "problem": "What is 2 + 2?",
                            "generated_solution": "Add the values.",
                            "expected_answer": "4",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "marulho.evaluation.language_hf_curriculum_materializer.urlopen",
        _fake_urlopen,
    )

    report = materialize_hf_curriculum(
        output_path=output,
        corpus_output_path=corpus,
        sources=sources,
        rows_per_source=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    corpus_text = corpus.read_text(encoding="utf-8")

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["external_llm_used"] is False
    assert report["service_owned_cognition"] is False
    assert report["mutates_runtime_state"] is False
    assert report["promotes_runtime_claim"] is False
    assert report["raw_row_payloads_retained"] is False
    assert report["corpus"]["row_count"] == 2
    assert report["corpus"]["sha256"]
    assert len(report["source_reports"]) == 2
    assert all(item["row_hashes"] for item in report["source_reports"])
    assert "user: Explain recurrence." in corpus_text
    assert "assistant: Track state over time." in corpus_text
    assert "What is 2 + 2?\nAdd the values.\n4" in corpus_text
    assert "{'role'" not in corpus_text


def test_materialize_hf_curriculum_records_partial_source_failures(
    tmp_path,
    monkeypatch,
) -> None:
    source = HFCurriculumSource(
        dataset="nvidia/HelpSteer3",
        config="preference",
        split="train",
        text_field="context,response1,response2",
        role="preference_review",
        license="cc-by-4.0",
    )

    def _fake_urlopen(request, timeout):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(
        "marulho.evaluation.language_hf_curriculum_materializer.urlopen",
        _fake_urlopen,
    )

    report = materialize_hf_curriculum(
        output_path=tmp_path / "curriculum.json",
        corpus_output_path=tmp_path / "curriculum.txt",
        sources=(source,),
        rows_per_source=1,
    )

    assert report["report_status"] == "partial"
    assert report["failed_source_count"] == 1
    assert report["source_reports"][0]["status"] == "failed"
    assert "dataset unavailable" in report["source_reports"][0]["failure_reason"]


def test_materialize_hf_curriculum_falls_back_to_first_rows_for_preview(
    tmp_path,
    monkeypatch,
) -> None:
    source = HFCurriculumSource(
        dataset="nvidia/Nemotron-Competitive-Programming-v1",
        config="default",
        split="competitive_coding_python_part00",
        text_field="messages",
        role="code_reasoning",
        license="cc-by-4.0",
    )
    requested_urls: list[str] = []

    def _fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        if "/rows?" in request.full_url:
            raise RuntimeError("rows failed")
        return _Response(
            {
                "rows": [
                    {
                        "row": {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Write a dynamic program.",
                                },
                                {
                                    "role": "assistant",
                                    "content": "Define the recurrence.",
                                },
                            ]
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(
        "marulho.evaluation.language_hf_curriculum_materializer.urlopen",
        _fake_urlopen,
    )

    report = materialize_hf_curriculum(
        output_path=tmp_path / "curriculum.json",
        corpus_output_path=tmp_path / "curriculum.txt",
        sources=(source,),
        rows_per_source=1,
    )

    assert report["report_status"] == "final"
    assert report["source_reports"][0]["fallback_endpoint"] == "first-rows"
    assert report["source_reports"][0]["rows_endpoint_failure"] == "rows failed"
    assert any("/rows?" in url for url in requested_urls)
    assert any("/first-rows?" in url for url in requested_urls)


def test_materialize_hf_curriculum_paginates_requested_rows(
    tmp_path,
    monkeypatch,
) -> None:
    source = HFCurriculumSource(
        dataset="nvidia/OpenMathInstruct-2",
        config="default",
        split="train_1M",
        text_field="problem,generated_solution,expected_answer",
        role="math_reasoning",
        license="cc-by-4.0",
    )
    requests: list[tuple[int, int]] = []

    def _fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        requests.append((offset, length))
        return _Response(
            {
                "rows": [
                    {
                        "row": {
                            "problem": f"Problem {idx}",
                            "generated_solution": f"Solution {idx}",
                            "expected_answer": f"Answer {idx}",
                        }
                    }
                    for idx in range(offset, offset + length)
                ]
            }
        )

    monkeypatch.setattr(
        "marulho.evaluation.language_hf_curriculum_materializer.urlopen",
        _fake_urlopen,
    )

    report = materialize_hf_curriculum(
        output_path=tmp_path / "curriculum.json",
        corpus_output_path=tmp_path / "curriculum.txt",
        sources=(source,),
        rows_per_source=105,
        offset=7,
    )

    assert requests == [(7, 100), (107, 5)]
    assert report["report_status"] == "final"
    assert report["corpus"]["row_count"] == 105
    assert report["source_reports"][0]["requested_rows"] == 105
    assert report["source_reports"][0]["offset"] == 7
    assert report["source_reports"][0]["page_size"] == 100
    assert report["source_reports"][0]["page_count"] == 2
    assert report["source_reports"][0]["materialized_rows"] == 105


def test_materialize_hf_curriculum_preserves_rows_before_later_page_failure(
    tmp_path,
    monkeypatch,
) -> None:
    source = HFCurriculumSource(
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        role="base_pretraining",
        license="odc-by",
    )

    def _fake_urlopen(request, timeout):
        query = parse_qs(urlparse(request.full_url).query)
        offset = int(query["offset"][0])
        if offset >= 100:
            raise RuntimeError("rate limited after first page")
        return _Response(
            {
                "rows": [
                    {"row": {"text": f"Document {idx}"}}
                    for idx in range(100)
                ]
            }
        )

    monkeypatch.setattr(
        "marulho.evaluation.language_hf_curriculum_materializer.urlopen",
        _fake_urlopen,
    )

    report = materialize_hf_curriculum(
        output_path=tmp_path / "fineweb.json",
        corpus_output_path=tmp_path / "fineweb.txt",
        sources=(source,),
        rows_per_source=105,
    )

    assert report["report_status"] == "partial"
    assert report["corpus"]["row_count"] == 100
    assert report["source_reports"][0]["status"] == "partial"
    assert report["source_reports"][0]["fetched_rows"] == 100
    assert "rate limited" in report["source_reports"][0]["failure_reason"]
