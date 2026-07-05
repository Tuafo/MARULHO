from __future__ import annotations

import json

import torch

from marulho.core.language_expert_dispatch_triton import (
    language_expert_dispatch_triton_stats,
)
from marulho.core.language_eligibility_trace_triton import (
    language_eligibility_trace_triton_stats,
)
from marulho.core.language_memory_slots_triton import (
    language_memory_slots_triton_stats,
)
from marulho.core.language_plif_triton import language_plif_triton_stats
from marulho.core.language_rmsnorm_triton import language_rmsnorm_triton_stats
from marulho.core.language_route_topk_triton import language_route_topk_triton_stats
from marulho.core.language_sampled_vocab_ce_triton import (
    language_sampled_vocab_ce_triton_stats,
)
from marulho.core.language_selective_scan_triton import (
    language_selective_scan_triton_stats,
)
from marulho.evaluation.language_triton_kernel_report import (
    ARTIFACT_KIND,
    EXPERT_DISPATCH_KERNEL_NAME,
    ELIGIBILITY_TRACE_KERNEL_NAME,
    KERNEL_NAME,
    MEMORY_SLOTS_KERNEL_NAME,
    PLIF_FORWARD_KERNEL_NAME,
    PLIF_SURROGATE_KERNEL_NAME,
    ROUTE_TOPK_KERNEL_NAME,
    SAMPLED_VOCAB_CE_KERNEL_NAME,
    SELECTIVE_SCAN_KERNEL_NAME,
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


def test_language_triton_kernel_report_writes_plif_surrogate_evidence(tmp_path) -> None:
    output = tmp_path / "plif-surrogate-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="plif-surrogate",
        shapes=((16, 32),),
        dtypes=("float32",),
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == PLIF_SURROGATE_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "plif_triton_backward_surrogate_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(language_plif_triton_stats()["triton_available"]):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
        assert report["shape_results"][0]["stats_delta"]["triton_backward_calls"] >= 1
        assert report["shape_results"][0]["max_grad_abs_error"] <= report[
            "shape_results"
        ][0]["tolerance"]
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_plif_forward_evidence(tmp_path) -> None:
    output = tmp_path / "plif-forward-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="plif-forward",
        shapes=((16, 32),),
        dtypes=("float32",),
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == PLIF_FORWARD_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "plif_triton_backward_surrogate_parity"
        in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(language_plif_triton_stats()["triton_available"]):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_selective_scan_evidence(tmp_path) -> None:
    output = tmp_path / "selective-scan-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="selective-scan",
        shapes=((8, 32),),
        dtypes=("float32",),
        scan_time_steps=8,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == SELECTIVE_SCAN_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "selective_scan_triton_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_selective_scan_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["time_steps"] == 8
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_expert_dispatch_evidence(tmp_path) -> None:
    output = tmp_path / "expert-dispatch-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="expert-dispatch",
        shapes=((64, 32),),
        dtypes=("float32",),
        expert_count=16,
        active_experts=2,
        expert_hidden_dim=64,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == EXPERT_DISPATCH_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "block_sparse_expert_dispatch_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_expert_dispatch_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["active_experts"] == 2
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_route_topk_evidence(tmp_path) -> None:
    output = tmp_path / "route-topk-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="route-topk",
        shapes=((64, 32),),
        dtypes=("float32",),
        expert_count=16,
        route_candidate_count=4,
        active_experts=2,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == ROUTE_TOPK_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "route_vote_topk_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_route_topk_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["route_candidate_count"] == 4
        assert report["shape_results"][0]["active_experts"] == 2
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_sampled_vocab_ce_evidence(tmp_path) -> None:
    output = tmp_path / "sampled-vocab-ce-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="sampled-vocab-ce",
        shapes=((64, 32),),
        dtypes=("float32",),
        vocab_size=512,
        sampled_vocab_size=128,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == SAMPLED_VOCAB_CE_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "sampled_vocab_cross_entropy_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_sampled_vocab_ce_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["sampled_vocab_size"] == 128
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_memory_slots_evidence(tmp_path) -> None:
    output = tmp_path / "memory-slots-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="memory-slots",
        shapes=((64, 32),),
        dtypes=("float32",),
        memory_slot_count=32,
        memory_slot_candidate_count=4,
        active_memory_slots=2,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == MEMORY_SLOTS_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "bounded_memory_slot_retrieval_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_memory_slots_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["memory_slot_candidate_count"] == 4
        assert report["shape_results"][0]["active_memory_slots"] == 2
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0


def test_language_triton_kernel_report_writes_eligibility_trace_evidence(tmp_path) -> None:
    output = tmp_path / "eligibility-trace-triton.json"

    report = run_language_triton_kernel_report(
        output_path=output,
        kernel="eligibility-trace",
        shapes=((8, 32),),
        dtypes=("float32",),
        scan_time_steps=8,
        warmup=1,
        repeats=2,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["artifact_kind"] == ARTIFACT_KIND
    assert written["surface"] == SURFACE
    assert report["kernel_name"] == ELIGIBILITY_TRACE_KERNEL_NAME
    assert report["external_llm_used"] is False
    assert report["promotion_gate"]["promotes_hot_path"] is False
    assert (
        "local_eligibility_trace_update_parity"
        not in report["promotion_gate"]["remaining_kernel_backlog"]
    )

    if torch.cuda.is_available() and bool(
        language_eligibility_trace_triton_stats()["triton_available"]
    ):
        assert report["parity_passed"] is True
        assert report["valid_shape_result_count"] >= 1
        assert report["promotion_gate"]["kernel_parity_available"] is True
        assert report["shape_results"][0]["time_steps"] == 8
        assert report["shape_results"][0]["stats_delta"]["triton_kernel_used"] is True
    else:
        assert report["promotion_gate"]["status"] == "unavailable"
        assert report["valid_shape_result_count"] == 0
