from __future__ import annotations

import json

from marulho.reporting.evidence_inventory import (
    SURFACE,
    build_evidence_report_inventory,
)


def test_evidence_report_inventory_summarizes_saved_reports_without_promotion(
    tmp_path,
) -> None:
    reports = tmp_path / "reports"
    suite_dir = reports / "language_benchmark_suite"
    suite_dir.mkdir(parents=True)
    report_path = suite_dir / "language-suite.json"
    report_path.write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_runtime_benchmark_suite",
                "surface": "marulho_language_runtime_benchmark_suite.v1",
                "success": True,
                "external_llm_used": False,
                "promotion_gate": {
                    "status": "blocked_missing_required_evidence",
                    "promotes_runtime_claim": False,
                    "missing_required_category_names": [
                        "grounding_support",
                        "gpu_kernel_correctness",
                    ],
                    "failed_category_names": [],
                },
            }
        ),
        encoding="utf-8",
    )
    invalid_path = reports / "invalid.json"
    invalid_path.write_text("{not json", encoding="utf-8")

    inventory = build_evidence_report_inventory(reports, limit=10)
    records = {item["relative_path"]: item for item in inventory["reports"]}

    assert inventory["surface"] == SURFACE
    assert inventory["reports_not_run_by_service"] is True
    assert inventory["mutates_runtime_state"] is False
    assert inventory["report_count"] == 2
    assert records["language_benchmark_suite/language-suite.json"]["readable"] is True
    assert records["language_benchmark_suite/language-suite.json"]["artifact_kind"] == (
        "marulho_language_runtime_benchmark_suite"
    )
    assert records["language_benchmark_suite/language-suite.json"]["promotion_status"] == (
        "blocked_missing_required_evidence"
    )
    assert records["language_benchmark_suite/language-suite.json"]["promotes_runtime_claim"] is False
    assert records["language_benchmark_suite/language-suite.json"][
        "missing_required_category_names"
    ] == ["grounding_support", "gpu_kernel_correctness"]
    assert records["invalid.json"]["readable"] is False
    assert "JSONDecodeError" in records["invalid.json"]["parse_error"]
