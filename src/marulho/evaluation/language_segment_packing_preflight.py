"""Validate V80's packed independent-segment training layout."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Callable, Mapping

import torch
from torch.nn import functional as F

from marulho.evaluation.language_dclm_materialization import (
    SURFACE as DCLM_SURFACE,
    _token_tensor_sha256,
)
from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_TOKENS,
    EFFECTIVE_BATCH,
    ROOT,
    SEGMENT_LENGTH,
    SEGMENTS,
    TOKENIZER_SHA256,
    _atomic_json,
    _episode,
    _gradient_audit,
    file_sha256,
)
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    load_language_model_state,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_language_segment_packing_preflight.v80"
PARENT = (
    ROOT
    / "reports/language_scaling/"
    "v78-unique-document-qualified-100m-257m-20260814.pt"
)
PARENT_SHA256 = "b66753983316b5a0cf61b293d36e4fda9b15929168067a59ed95ef816da4313b"
PARENT_STATE_SHA256 = "4ebf6ae3a500a0a77a256be80bb652a3439e47310eb008c002d14312bb34b75e"
DCLM_ARTIFACT = (
    ROOT
    / "reports/language_curriculum/"
    "v79-dclm-edu-selected-16896-20260814.pt"
)
DCLM_ARTIFACT_SHA256 = "04812812d5f2a319a9e88132d1cd01867b98600fc45ba03f3fe78b86bf9eeea0"
DCLM_TRAIN_SHA256 = "fa4dc5151406c23e19c8fe28dd12872b2a5b179e078c43609ec65ab48abd530a"
PHYSICAL_BATCH = 8
ACCUMULATION_STEPS = EFFECTIVE_BATCH // PHYSICAL_BATCH
TIMED_STEPS = 8
MAXIMUM_PEAK_BYTES = 8 * 1024**3


Episode = Callable[[MarulhoLanguageModel, torch.Tensor], Mapping[str, torch.Tensor]]


def packed_episode(
    model: MarulhoLanguageModel,
    documents: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate three independent segments in one larger batch call."""
    if documents.ndim != 2 or int(documents.shape[1]) != DOCUMENT_TOKENS:
        raise ValueError("packed episode expects [batch, 961] documents")
    windows = documents.unfold(1, SEGMENT_LENGTH + 1, SEGMENT_LENGTH)
    if tuple(windows.shape[1:]) != (SEGMENTS, SEGMENT_LENGTH + 1):
        raise RuntimeError(f"packed segment view changed: {tuple(windows.shape)}")
    batch = int(documents.shape[0])
    inputs = windows[:, :, :-1].contiguous().reshape(
        batch * SEGMENTS, SEGMENT_LENGTH
    )
    targets = windows[:, :, 1:].contiguous().reshape(
        batch * SEGMENTS, SEGMENT_LENGTH
    )
    logits = model(inputs, collect_telemetry=False)["logits"]
    per_token = F.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).reshape(batch, SEGMENTS, SEGMENT_LENGTH)
    per_document_segment = per_token.mean(2)
    segment_losses = per_document_segment.mean(0)
    return {
        "loss": segment_losses.mean(),
        "segment_losses": segment_losses.detach(),
        "per_document_segment_losses": per_document_segment.detach(),
    }


def _load_inputs() -> tuple[torch.Tensor, dict[str, Any]]:
    artifact_hash = file_sha256(DCLM_ARTIFACT)
    if artifact_hash != DCLM_ARTIFACT_SHA256:
        raise RuntimeError(f"V80 DCLM artifact changed: {artifact_hash}")
    artifact = torch.load(DCLM_ARTIFACT, map_location="cpu", weights_only=False)
    train: torch.Tensor = artifact["train_tokens"].to(dtype=torch.int32)
    checks = {
        "surface_exact": artifact.get("surface") == DCLM_SURFACE,
        "external_llm_absent": artifact.get("external_llm_used") is False,
        "tokenizer_exact": artifact.get("tokenizer_sha256") == TOKENIZER_SHA256,
        "shape_exact": tuple(train.shape) == (16_384, DOCUMENT_TOKENS),
        "tensor_hash_exact": _token_tensor_sha256(train) == DCLM_TRAIN_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V80 DCLM validation failed: {checks}")
    required = (TIMED_STEPS + 2) * EFFECTIVE_BATCH
    return train[:required].clone(), {
        "path": str(DCLM_ARTIFACT),
        "sha256": artifact_hash,
        "train_tensor_sha256": DCLM_TRAIN_SHA256,
        "selected_documents": required,
        "checks": checks,
    }


def _load_parent() -> tuple[MarulhoLanguageModel, dict[str, Any]]:
    checkpoint_hash = file_sha256(PARENT)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V80 parent checkpoint changed: {checkpoint_hash}")
    model, tokenizer, metadata = load_language_model_checkpoint(PARENT, map_location="cpu")
    state_hash = language_model_state_sha256(model)
    checks = {
        "state_exact": state_hash == PARENT_STATE_SHA256,
        "tokenizer_exact": tokenizer.vocabulary_hash() == TOKENIZER_SHA256,
        "context_exact": int(model.config.transformer_context_length)
        == SEGMENT_LENGTH,
        "dropout_disabled": float(model.config.transformer_dropout) == 0.0,
        "parameter_count_exact": sum(p.numel() for p in model.parameters())
        == 100_679_424,
        "metadata_exact": metadata.get("decision")
        == "save_v78_unique_document_checkpoint_for_unseen_generation",
    }
    if not all(checks.values()):
        raise RuntimeError(f"V80 parent validation failed: {checks}")
    return model, {
        "path": str(PARENT),
        "sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "checks": checks,
    }


def _clone_to_cuda(
    parent: MarulhoLanguageModel,
    *,
    device: torch.device,
) -> MarulhoLanguageModel:
    model = MarulhoLanguageModel(parent.config)
    load_language_model_state(model, parent.state_dict())
    return model.to(device=device, dtype=torch.bfloat16)


def _optimizer(model: MarulhoLanguageModel):
    return build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )


def _backward_complete_step(
    model: MarulhoLanguageModel,
    optimizer: Any,
    *,
    documents: torch.Tensor,
    episode: Episode,
    device: torch.device,
    audit_gradients: bool,
    collect_details: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    rows: list[torch.Tensor] = []
    for micro in range(ACCUMULATION_STEPS):
        start = micro * PHYSICAL_BATCH
        batch = documents[start : start + PHYSICAL_BATCH].to(
            device=device, dtype=torch.long
        )
        result = episode(model, batch)
        (result["loss"] / ACCUMULATION_STEPS).backward()
        if collect_details:
            losses.append(result["loss"].detach().cpu().float())
            rows.append(result["per_document_segment_losses"].cpu().float())
        del batch, result
    return {
        "loss": float(torch.stack(losses).mean().item()) if losses else None,
        "per_document_segment_losses": torch.cat(rows, dim=0) if rows else None,
        "gradient_audit": _gradient_audit(model) if audit_gradients else None,
    }


@torch.no_grad()
def _post_update_loss(
    model: MarulhoLanguageModel,
    documents: torch.Tensor,
    *,
    episode: Episode,
    device: torch.device,
) -> float:
    model.eval()
    batch = documents[:PHYSICAL_BATCH].to(device=device, dtype=torch.long)
    value = float(episode(model, batch)["loss"].item())
    model.train()
    return value


def _global_tensor_comparison(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, float | int | bool]:
    if tuple(left) != tuple(right):
        raise RuntimeError("V80 tensor comparison keys changed")
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    difference_square = 0.0
    maximum_absolute_delta = 0.0
    equal = 0
    count = 0
    for name in left:
        a = left[name].float()
        b = right[name].float()
        difference = a - b
        dot += float((a.double() * b.double()).sum().item())
        left_square += float(a.double().square().sum().item())
        right_square += float(b.double().square().sum().item())
        difference_square += float(difference.double().square().sum().item())
        maximum_absolute_delta = max(
            maximum_absolute_delta, float(difference.abs().max().item())
        )
        equal += int(torch.count_nonzero(a == b).item())
        count += int(a.numel())
    return {
        "keys_exact": True,
        "element_count": count,
        "bit_equal_element_count": equal,
        "bit_equal_fraction": equal / count,
        "maximum_absolute_delta": maximum_absolute_delta,
        "cosine": dot / max(1.0e-30, (left_square * right_square) ** 0.5),
        "relative_l2_error": difference_square**0.5
        / max(1.0e-30, left_square**0.5),
    }


def _parity_arm(
    parent: MarulhoLanguageModel,
    documents: torch.Tensor,
    *,
    episode: Episode,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model = _clone_to_cuda(parent, device=device)
    initial_state_hash = language_model_state_sha256(model)
    optimizer, optimizer_report = _optimizer(model)
    model.train()
    step = _backward_complete_step(
        model,
        optimizer,
        documents=documents[:EFFECTIVE_BATCH],
        episode=episode,
        device=device,
        audit_gradients=True,
        collect_details=True,
    )
    assert step["gradient_audit"] is not None
    assert step["per_document_segment_losses"] is not None
    if not step["gradient_audit"]["passed"]:
        raise RuntimeError(f"V80 parity gradients failed: {step['gradient_audit']}")
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    post_loss = _post_update_loss(
        model,
        documents[EFFECTIVE_BATCH : EFFECTIVE_BATCH + PHYSICAL_BATCH],
        episode=episode,
        device=device,
    )
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    report = {
        "initial_bf16_state_sha256": initial_state_hash,
        "step_loss": step["loss"],
        "per_document_segment_losses": step["per_document_segment_losses"],
        "gradient_audit": step["gradient_audit"],
        "post_update_loss": post_loss,
        "optimizer": optimizer_report,
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return report, gradients, state


def _timed_layout(
    parent: MarulhoLanguageModel,
    documents: torch.Tensor,
    *,
    name: str,
    episode: Episode,
    device: torch.device,
) -> dict[str, Any]:
    model = _clone_to_cuda(parent, device=device)
    optimizer, optimizer_report = _optimizer(model)
    model.train()
    warmup = _backward_complete_step(
        model,
        optimizer,
        documents=documents[:EFFECTIVE_BATCH],
        episode=episode,
        device=device,
        audit_gradients=True,
        collect_details=False,
    )
    assert warmup["gradient_audit"] is not None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.reset_peak_memory_stats(device)
    step_seconds: list[float] = []
    for step in range(TIMED_STEPS):
        start = (step + 1) * EFFECTIVE_BATCH
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = _backward_complete_step(
            model,
            optimizer,
            documents=documents[start : start + EFFECTIVE_BATCH],
            episode=episode,
            device=device,
            audit_gradients=False,
            collect_details=False,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        del result
    median_seconds = statistics.median(step_seconds)
    report = {
        "layout": name,
        "warmup_gradient_audit": warmup["gradient_audit"],
        "timed_steps": TIMED_STEPS,
        "step_seconds": step_seconds,
        "median_complete_step_seconds": median_seconds,
        "median_positions_per_second": (
            EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH / median_seconds
        ),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "model_state_finite": all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in model.parameters()
        ),
        "optimizer": optimizer_report,
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def admission_checks(
    *,
    forward_loss_delta: float,
    per_document_max_delta: float,
    gradient: Mapping[str, Any],
    state: Mapping[str, Any],
    post_update_loss_delta: float,
    baseline: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> dict[str, bool]:
    throughput_ratio = float(packed["median_positions_per_second"]) / float(
        baseline["median_positions_per_second"]
    )
    return {
        "forward_loss_delta_at_most_0_002": forward_loss_delta <= 0.002,
        "per_document_segment_delta_at_most_0_02": per_document_max_delta <= 0.02,
        "gradient_cosine_at_least_0_9999": float(gradient["cosine"]) >= 0.9999,
        "gradient_relative_l2_at_most_0_02": float(gradient["relative_l2_error"])
        <= 0.02,
        "state_relative_l2_at_most_0_002": float(state["relative_l2_error"])
        <= 0.002,
        "post_update_loss_delta_at_most_0_002": post_update_loss_delta <= 0.002,
        "baseline_gradients_complete": bool(
            baseline["warmup_gradient_audit"]["passed"]
        ),
        "packed_gradients_complete": bool(packed["warmup_gradient_audit"]["passed"]),
        "baseline_state_finite": bool(baseline["model_state_finite"]),
        "packed_state_finite": bool(packed["model_state_finite"]),
        "baseline_peak_within_8_gib": int(baseline["peak_cuda_allocated_bytes"])
        <= MAXIMUM_PEAK_BYTES,
        "packed_peak_within_8_gib": int(packed["peak_cuda_allocated_bytes"])
        <= MAXIMUM_PEAK_BYTES,
        "packed_throughput_gain_at_least_1_10": throughput_ratio >= 1.10,
    }


def run_preflight(*, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"V80 preflight output already exists: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("V80 segment packing preflight requires CUDA")
    parent, parent_audit = _load_parent()
    documents, data_audit = _load_inputs()
    device = torch.device("cuda")
    torch.manual_seed(80_131)
    torch.cuda.manual_seed_all(80_131)

    baseline_parity, baseline_gradients, baseline_state = _parity_arm(
        parent, documents, episode=_episode, device=device
    )
    packed_parity, packed_gradients, packed_state = _parity_arm(
        parent, documents, episode=packed_episode, device=device
    )
    gradient_comparison = _global_tensor_comparison(
        baseline_gradients, packed_gradients
    )
    state_comparison = _global_tensor_comparison(baseline_state, packed_state)
    forward_loss_delta = abs(
        float(baseline_parity["step_loss"]) - float(packed_parity["step_loss"])
    )
    per_document_max_delta = float(
        (
            baseline_parity["per_document_segment_losses"]
            - packed_parity["per_document_segment_losses"]
        )
        .abs()
        .max()
        .item()
    )
    post_update_loss_delta = abs(
        float(baseline_parity["post_update_loss"])
        - float(packed_parity["post_update_loss"])
    )
    initial_state_exact = (
        baseline_parity["initial_bf16_state_sha256"]
        == packed_parity["initial_bf16_state_sha256"]
    )
    for report in (baseline_parity, packed_parity):
        report.pop("per_document_segment_losses")
    del baseline_gradients, packed_gradients, baseline_state, packed_state
    gc.collect()

    baseline_timing = _timed_layout(
        parent, documents, name="three_sequential_calls", episode=_episode, device=device
    )
    packed_timing = _timed_layout(
        parent, documents, name="one_packed_call", episode=packed_episode, device=device
    )
    checks = admission_checks(
        forward_loss_delta=forward_loss_delta,
        per_document_max_delta=per_document_max_delta,
        gradient=gradient_comparison,
        state=state_comparison,
        post_update_loss_delta=post_update_loss_delta,
        baseline=baseline_timing,
        packed=packed_timing,
    )
    checks["initial_bf16_state_exact"] = initial_state_exact
    passed = all(checks.values())
    throughput_ratio = float(packed_timing["median_positions_per_second"]) / float(
        baseline_timing["median_positions_per_second"]
    )
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_training_layout_preflight",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": (
            "admit_v80_packed_segment_training"
            if passed
            else "retain_v80_sequential_segment_training"
        ),
        "parent": parent_audit,
        "data": data_audit,
        "configuration": {
            "physical_document_batch": PHYSICAL_BATCH,
            "effective_document_batch": EFFECTIVE_BATCH,
            "segments_per_document": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "packed_sequence_batch": PHYSICAL_BATCH * SEGMENTS,
            "timed_steps": TIMED_STEPS,
            "dtype": "torch.bfloat16",
            "compiled": False,
            "batch32_forbidden": True,
        },
        "parity": {
            "baseline": baseline_parity,
            "packed": packed_parity,
            "forward_loss_absolute_delta": forward_loss_delta,
            "maximum_per_document_segment_loss_delta": per_document_max_delta,
            "post_update_loss_absolute_delta": post_update_loss_delta,
            "gradient_comparison": gradient_comparison,
            "post_update_state_comparison": state_comparison,
        },
        "timing": {
            "baseline": baseline_timing,
            "packed": packed_timing,
            "packed_baseline_throughput_ratio": throughput_ratio,
        },
        "admission_checks": checks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    _atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(output_path=args.output)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "passed": result["passed"],
                "throughput_ratio": result["timing"][
                    "packed_baseline_throughput_ratio"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
