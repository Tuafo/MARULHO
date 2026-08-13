"""Bounded V70 100M full-step admission evaluator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import torch

from marulho.training.language_macro_cortex import (
    MarulhoMacroCortexLanguageModel,
    transfer_transformer_common_state,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_muon import build_language_muon


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=320,
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_macro_cortex_v70",
    )


def _is_common(name: str) -> bool:
    return not name.endswith(
        (
            ".summary_queries",
            ".start_macro",
            ".query_macro_scale",
            ".output_macro_scale",
        )
    )


def _common_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not _is_common(name):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _build(arm: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(7001)
    control = MarulhoLanguageModel(_config())
    control_hash = _common_hash(control)
    if arm == "control":
        return control, {"common_parameter_hash": control_hash, "transfer": None}
    torch.manual_seed(7002)
    candidate = MarulhoMacroCortexLanguageModel(_config())
    transfer = transfer_transformer_common_state(control, candidate)
    candidate_hash = _common_hash(candidate)
    if candidate_hash != control_hash:
        raise RuntimeError("V70 common initialization hash mismatch")
    del control
    gc.collect()
    return candidate, {
        "common_parameter_hash": candidate_hash,
        "transfer": transfer,
    }


def _run(arm: str) -> dict[str, Any]:
    model, truth = _build(arm)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )
    torch.manual_seed(7003)
    torch.cuda.manual_seed_all(7003)
    inputs = torch.randint(0, 8192, (32, 320), device=device)
    targets = torch.randint(0, 8192, (32, 320), device=device)

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(
            inputs, targets, collect_telemetry=False, return_evidence=False
        )["loss"]
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss, gradient_norm

    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    durations: list[float] = []
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(3):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        loss, gradient_norm = step()
        end.record()
        end.synchronize()
        durations.append(begin.elapsed_time(end) / 1_000.0)
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(gradient_norm.detach().item()))
    peak = torch.cuda.max_memory_allocated()
    median = statistics.median(durations)
    gradients = [parameter.grad for parameter in model.parameters()]
    return {
        "arm": arm,
        "parameter_count": parameter_count,
        **truth,
        "batch_size": 32,
        "context_length": 320,
        "positions_per_step": 10240,
        "precision": "bfloat16",
        "warmup_steps": 1,
        "timing_steps": 3,
        "timing_step_seconds": durations,
        "median_step_seconds": median,
        "positions_per_second": 10240 / median,
        "losses": losses,
        "gradient_norms_before_clip": gradient_norms,
        "all_gradients_present": all(gradient is not None for gradient in gradients),
        "all_gradients_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all())
            for gradient in gradients
        ),
        "all_gradients_nonzero": all(
            gradient is not None and bool(torch.count_nonzero(gradient))
            for gradient in gradients
        ),
        "peak_cuda_bytes": peak,
        "peak_below_11_5_gib": peak < int(11.5 * 1024**3),
        "optimizer": optimizer_report,
        "torch_compile_used": False,
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("candidate", "control"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V70 preflight requires CUDA")
    payload = _run(arguments.arm)
    _write(arguments.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
