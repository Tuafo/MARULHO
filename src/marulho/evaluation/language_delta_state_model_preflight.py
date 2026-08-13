"""Bounded full-model parity preflight for V64's direct Triton backend."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
from uuid import uuid4

import torch
import torch.nn.functional as F

from marulho.training.language_delta_state import (
    DeltaStateLanguageModelConfig,
    MarulhoDeltaStateLanguageModel,
    delta_state_parameter_report,
)


SURFACE = "marulho_delta_state_direct_model_preflight.v1"


@dataclass(frozen=True)
class DirectModelPreflightConfig:
    batch_size: int = 2
    context_length: int = 320
    timing_steps: int = 1
    seed: int = 64801
    answer_weight: float = 4.0
    loss_tolerance: float = 1.0e-3
    gradient_cosine_tolerance: float = 0.999
    gradient_maximum_delta_tolerance: float = 0.01


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(backend: str) -> DeltaStateLanguageModelConfig:
    return DeltaStateLanguageModelConfig(delta_execution_backend=backend)


def _batch(
    config: DirectModelPreflightConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(config.seed + 1)
    shape = (config.batch_size, config.context_length)
    inputs = torch.randint(0, 8192, shape, generator=generator, device="cuda")
    targets = torch.randint(0, 8192, shape, generator=generator, device="cuda")
    weights = torch.ones(shape, device="cuda", dtype=torch.float32)
    weights[:, -80:] = config.answer_weight
    return inputs, targets, weights


def _loss(
    model: MarulhoDeltaStateLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = model(inputs, collect_telemetry=False)["logits"]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    normalized = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * normalized).sum() / normalized.sum()


def _gradient_inventory(model: MarulhoDeltaStateLanguageModel) -> dict[str, Any]:
    missing = []
    nonfinite = []
    zero = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif not bool(torch.count_nonzero(parameter.grad)):
            zero.append(name)
    return {
        "parameter_tensor_count": sum(1 for _ in model.parameters()),
        "missing": missing,
        "nonfinite": nonfinite,
        "zero": zero,
        "all_present_finite_nonzero": not (missing or nonfinite or zero),
    }


def _copy_gradients(
    model: MarulhoDeltaStateLanguageModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _compare_gradients(
    expected: Mapping[str, torch.Tensor], model: MarulhoDeltaStateLanguageModel
) -> dict[str, Any]:
    dot = 0.0
    expected_norm = 0.0
    actual_norm = 0.0
    maximum = 0.0
    names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in expected:
            continue
        actual = parameter.grad.detach().float().cpu()
        reference = expected[name]
        local = float(torch.max(torch.abs(actual - reference)).item())
        if local > maximum:
            maximum = local
            names = [name]
        elif local == maximum:
            names.append(name)
        dot += float(torch.sum(actual * reference).item())
        actual_norm += float(torch.sum(actual.square()).item())
        expected_norm += float(torch.sum(reference.square()).item())
    return {
        "names_equal": set(expected)
        == {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        "global_cosine": dot
        / max(1.0e-30, math.sqrt(actual_norm * expected_norm)),
        "maximum_absolute_element_delta": maximum,
        "maximum_delta_names": names,
    }


def _execute(
    *,
    backend: str,
    initial_state: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: DirectModelPreflightConfig,
    expected_gradients: Mapping[str, torch.Tensor] | None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    model = MarulhoDeltaStateLanguageModel(_model_config(backend)).cuda()
    model.load_state_dict(initial_state, strict=True)
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = _loss(model, inputs, targets, weights)
    loss.backward()
    torch.cuda.synchronize()
    first_seconds = time.perf_counter() - started
    loss_value = float(loss.detach().float().cpu())
    inventory = _gradient_inventory(model)
    comparison = (
        _compare_gradients(expected_gradients, model)
        if expected_gradients is not None
        else None
    )
    copied = _copy_gradients(model) if expected_gradients is None else None
    model.zero_grad(set_to_none=True)
    durations = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(config.timing_steps):
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            timed_loss = _loss(model, inputs, targets, weights)
        timed_loss.backward()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - step_started)
        model.zero_grad(set_to_none=True)
    elapsed = sum(durations)
    positions = config.batch_size * config.context_length * config.timing_steps
    result = {
        "backend": backend,
        "loss": loss_value,
        "first_forward_backward_seconds_including_compile_if_uncached": first_seconds,
        "gradient_inventory": inventory,
        "gradient_comparison": comparison,
        "timing_step_seconds": durations,
        "timing_positions_per_second": positions / elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del model, loss
    torch.cuda.empty_cache()
    return result, copied


def run_preflight(
    *, config: DirectModelPreflightConfig, output: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V64 direct-model preflight requires CUDA")
    torch.manual_seed(config.seed)
    template = MarulhoDeltaStateLanguageModel(_model_config("eager"))
    initial_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in template.state_dict().items()
    }
    parameter_report = delta_state_parameter_report(template)
    del template
    inputs, targets, weights = _batch(config)
    eager, expected_gradients = _execute(
        backend="eager",
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        weights=weights,
        config=config,
        expected_gradients=None,
    )
    if expected_gradients is None:  # pragma: no cover - fixed control contract
        raise RuntimeError("eager preflight did not retain gradients")
    direct, _ = _execute(
        backend="triton_replay",
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        weights=weights,
        config=config,
        expected_gradients=expected_gradients,
    )
    comparison = direct["gradient_comparison"]
    gates = {
        "loss": abs(float(direct["loss"]) - float(eager["loss"]))
        <= config.loss_tolerance,
        "gradient_names": bool(comparison["names_equal"]),
        "gradient_cosine": float(comparison["global_cosine"])
        >= config.gradient_cosine_tolerance,
        "gradient_maximum_delta": float(
            comparison["maximum_absolute_element_delta"]
        )
        <= config.gradient_maximum_delta_tolerance,
        "eager_complete_gradients": bool(
            eager["gradient_inventory"]["all_present_finite_nonzero"]
        ),
        "direct_complete_gradients": bool(
            direct["gradient_inventory"]["all_present_finite_nonzero"]
        ),
    }
    payload = {
        "artifact_kind": "marulho_delta_state_direct_model_preflight",
        "surface": SURFACE,
        "decision": (
            "select_physical_batch16_advance_cuda_graph_optimizer_preflight"
            if all(gates.values()) and config.batch_size == 16
            else (
                "continue_direct_kernel_physical_batch_search"
                if all(gates.values()) and config.batch_size < 16
                else (
                    "advance_direct_kernel_to_optimizer_preflight"
                    if all(gates.values())
                    else "stop_direct_kernel_model_parity_failure"
                )
            )
        ),
        "config": asdict(config),
        "parameter_report": parameter_report,
        "eager": eager,
        "direct": direct,
        "direct_to_eager_throughput_ratio": float(
            direct["timing_positions_per_second"]
        )
        / float(eager["timing_positions_per_second"]),
        "gates": gates,
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "torch_compile_used": False,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    _atomic_json(output, payload)
    payload["report_path"] = str(output)
    payload["report_sha256"] = _sha256(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timing-steps", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    config = DirectModelPreflightConfig(
        batch_size=arguments.batch_size, timing_steps=arguments.timing_steps
    )
    output = arguments.output or Path(
        "reports/language_scaling/"
        f"delta-state-v64-direct-model-b{config.batch_size}-20260813.json"
    )
    print(json.dumps(run_preflight(config=config, output=output), indent=2))


if __name__ == "__main__":
    main()
