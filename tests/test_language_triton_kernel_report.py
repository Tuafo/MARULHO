from __future__ import annotations

import json

import torch

from marulho.core.language_rmsnorm_triton import language_rmsnorm_triton_stats
from marulho.evaluation.language_triton_kernel_report import (
    ARTIFACT_KIND,
    KERNEL_NAME,
    SURFACE,
    run_language_triton_kernel_report,
)


def test_language_triton_kernel_report_writes_parity_evidence(tmp_path) -> None:
    output = tmp_path / "rmsnorm-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        shapes=((8, 16),),
        dtypes=("float32",),
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (tmp_path / "README.md").exists()

    if torch.cuda.is_available() and bool(language_rmsnorm_triton_stats()["triton_available"]):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0
