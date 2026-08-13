"""Bounded V71 full-step admission evaluator."""

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

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_muon import build_language_muon
from marulho.training.language_periodic_hierarchy import (
    MarulhoPeriodicHierarchyLanguageModel,
    transfer_periodic_common_state,
)


EXPECTED_COMMON_HASH = "700f403ac0405b11cc25262f87434b9a00174d4ed10bc46198e778b7ad84127a"


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
        active_language_path="marulho_periodic_hierarchy_v71",
    )


def _common_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.endswith(
            (".summary_queries", ".start_macro", ".query_macro_scale", ".output_macro_scale")
        ):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def run(arm: str) -> dict[str, Any]:
    torch.manual_seed(70131)
    control = MarulhoLanguageModel(_config())
    if _common_hash(control) != EXPECTED_COMMON_HASH:
        raise RuntimeError("V71 does not reproduce immutable V70 common hash")
    torch.manual_seed(71132)
    model = MarulhoPeriodicHierarchyLanguageModel(
        _config(), macro_enabled=arm == "periodic_macro"
    )
    transfer = transfer_periodic_common_state(control, model)
    del control
    gc.collect()
    common_hash = _common_hash(model)
    if common_hash != EXPECTED_COMMON_HASH:
        raise RuntimeError("V71 transferred common hash mismatch")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16)
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
    )
    torch.manual_seed(71133)
    torch.cuda.manual_seed_all(71133)
    inputs = torch.randint(0, 8192, (32, 320), device=device)
    targets = torch.randint(0, 8192, (32, 320), device=device)

    def step() -> tuple[torch.Tensor, torch.Tensor]:
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(
            inputs, targets, collect_telemetry=False, return_evidence=False
        )["loss"]
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss, norm

    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    times: list[float] = []
    losses: list[float] = []
    norms: list[float] = []
    for _ in range(3):
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        begin.record()
        loss, norm = step()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) / 1_000.0)
        losses.append(float(loss.detach().item()))
        norms.append(float(norm.detach().item()))
    gradients = [parameter.grad for parameter in model.parameters()]
    median = statistics.median(times)
    peak = torch.cuda.max_memory_allocated()
    return {
        "artifact_kind": "marulho_periodic_hierarchy_v71_preflight_arm",
        "surface": "marulho_periodic_hierarchy_v71_preflight_arm.v1",
        "arm": arm,
        "parameter_count": parameter_count,
        "common_parameter_hash": common_hash,
        "transfer": transfer,
        "timing_step_seconds": times,
        "median_step_seconds": median,
        "positions_per_second": 10240 / median,
        "losses": losses,
        "gradient_norms_before_clip": norms,
        "all_gradients_present_finite_nonzero": all(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and bool(torch.count_nonzero(gradient))
            for gradient in gradients
        ),
        "peak_cuda_bytes": peak,
        "below_11_5_gib": peak < int(11.5 * 1024**3),
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
    parser.add_argument("--arm", choices=("periodic_macro", "periodic_local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V71 preflight requires CUDA")
    payload = run(args.arm)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
