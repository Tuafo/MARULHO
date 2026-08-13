"""Optimizer-inclusive CUDA Graph preflight for the V64 Triton model."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
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
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


SURFACE = "marulho_delta_state_cuda_graph_preflight.v1"
DEFAULT_OUTPUT = Path(
    "reports/language_scaling/delta-state-v64-cuda-graph-preflight-20260813.json"
)


@dataclass(frozen=True)
class CudaGraphPreflightConfig:
    microbatch_size: int = 16
    gradient_accumulation_steps: int = 2
    context_length: int = 320
    timing_steps: int = 3
    learning_rate: float = 4.0e-4
    weight_decay: float = 0.1
    maximum_gradient_norm: float = 1.0
    answer_weight: float = 4.0
    graph_eager_loss_tolerance: float = 1.0e-5
    graph_eager_gradient_norm_tolerance: float = 1.0e-4
    graph_eager_update_cosine_tolerance: float = 0.999
    graph_eager_parameter_rms_tolerance: float = 1.0e-6
    graph_eager_parameter_maximum_tolerance: float = 5.0e-4
    preflight_minimum_throughput_ratio: float = 0.50
    promotion_minimum_throughput_ratio: float = 0.70
    maximum_peak_cuda_bytes: int = 12_348_027_699
    model_seed: int = 16411
    control_seed: int = 16412
    data_seed: int = 16413


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


def _candidate_config() -> DeltaStateLanguageModelConfig:
    return DeltaStateLanguageModelConfig(delta_execution_backend="triton_replay")


def _control_config(config: CudaGraphPreflightConfig) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=config.context_length,
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        active_language_path="marulho_v64_fresh_transformer_control",
    )


def _batch(
    config: CudaGraphPreflightConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_batch = config.microbatch_size * config.gradient_accumulation_steps
    generator = torch.Generator(device="cuda").manual_seed(config.data_seed)
    shape = (total_batch, config.context_length)
    inputs = torch.randint(0, 8192, shape, generator=generator, device="cuda")
    targets = torch.randint(0, 8192, shape, generator=generator, device="cuda")
    weights = torch.ones(shape, device="cuda", dtype=torch.float32)
    weights[:, -80:] = config.answer_weight
    return inputs, targets, weights


def _loss(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    logits = model(
        inputs, collect_telemetry=False, decode_vocab_only=False
    )["logits"]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    normalized = weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * normalized).sum() / normalized.sum()


def _optimizer(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
    config: CudaGraphPreflightConfig,
    *,
    capturable: bool,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=True,
        capturable=capturable,
    )


def _accumulated_step(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: CudaGraphPreflightConfig,
    *,
    clear_gradients: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clear_gradients:
        optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device="cuda", dtype=torch.float32)
    scale = 1.0 / config.gradient_accumulation_steps
    for index in range(config.gradient_accumulation_steps):
        start = index * config.microbatch_size
        stop = start + config.microbatch_size
        with torch.autocast("cuda", dtype=torch.bfloat16):
            microbatch_loss = _loss(
                model,
                inputs[start:stop],
                targets[start:stop],
                weights[start:stop],
            )
        (microbatch_loss * scale).backward()
        total_loss = total_loss + microbatch_loss.detach().float() * scale
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.maximum_gradient_norm
    )
    optimizer.step()
    return total_loss, gradient_norm


def _full_batch_step(
    model: MarulhoLanguageModel,
    optimizer: torch.optim.AdamW,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: CudaGraphPreflightConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = _loss(model, inputs, targets, weights)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.maximum_gradient_norm
    )
    optimizer.step()
    return loss, gradient_norm


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    return value


def _parameter_comparison(
    actual: MarulhoDeltaStateLanguageModel,
    expected: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    maximum = 0.0
    maximum_names: list[str] = []
    squared_delta = 0.0
    update_dot = 0.0
    actual_update_norm = 0.0
    expected_update_norm = 0.0
    elements = 0
    above_1e6 = 0
    above_1e5 = 0
    above_1e4 = 0
    names_equal = True
    actual_names = {name for name, _ in actual.named_parameters()}
    names_equal = actual_names == set(expected)
    for actual_name, actual_parameter in actual.named_parameters():
        if actual_name not in expected:
            continue
        expected_parameter = expected[actual_name].to(
            device=actual_parameter.device, dtype=torch.float32
        )
        baseline_parameter = baseline[actual_name].to(
            device=actual_parameter.device, dtype=torch.float32
        )
        actual_value = actual_parameter.detach().float()
        delta = actual_value - expected_parameter
        actual_update = actual_value - baseline_parameter
        expected_update = expected_parameter - baseline_parameter
        local = float(torch.max(torch.abs(delta)).item())
        if local > maximum:
            maximum = local
            maximum_names = [actual_name]
        elif local == maximum and local > 0.0:
            maximum_names.append(actual_name)
        squared_delta += float(torch.sum(delta.square()).item())
        update_dot += float(torch.sum(actual_update * expected_update).item())
        actual_update_norm += float(torch.sum(actual_update.square()).item())
        expected_update_norm += float(torch.sum(expected_update.square()).item())
        absolute = torch.abs(delta)
        above_1e6 += int(torch.count_nonzero(absolute > 1.0e-6).item())
        above_1e5 += int(torch.count_nonzero(absolute > 1.0e-5).item())
        above_1e4 += int(torch.count_nonzero(absolute > 1.0e-4).item())
        elements += delta.numel()
    return {
        "names_equal": names_equal,
        "maximum_absolute_element_delta": maximum,
        "maximum_delta_names": (
            maximum_names if maximum > 0.0 else ["all_parameters_exact"]
        ),
        "root_mean_square_element_delta": math.sqrt(
            squared_delta / max(1, elements)
        ),
        "update_global_cosine": update_dot
        / max(1.0e-30, math.sqrt(actual_update_norm * expected_update_norm)),
        "elements_above_1e_6": above_1e6,
        "elements_above_1e_5": above_1e5,
        "elements_above_1e_4": above_1e4,
        "compared_elements": elements,
    }


def _candidate_graph_arm(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: CudaGraphPreflightConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.model_seed)
    model = MarulhoDeltaStateLanguageModel(_candidate_config()).cuda().train()
    parameter_report = delta_state_parameter_report(model)
    optimizer = _optimizer(model, config, capturable=True)

    print("[v64-graph] warming candidate kernels and optimizer", flush=True)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        warmup_loss, warmup_norm = _accumulated_step(
            model,
            optimizer,
            inputs,
            targets,
            weights,
            config,
            clear_gradients=True,
        )
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    del warmup_loss, warmup_norm

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    graph = torch.cuda.CUDAGraph()
    print("[v64-graph] capturing two-microbatch optimizer step", flush=True)
    capture_started = time.perf_counter()
    with torch.cuda.graph(graph):
        captured_loss, captured_norm = _accumulated_step(
            model,
            optimizer,
            inputs,
            targets,
            weights,
            config,
            clear_gradients=False,
        )
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter() - capture_started
    capture_peak = int(torch.cuda.max_memory_allocated())

    print("[v64-graph] snapshotting one-model parity state", flush=True)
    baseline_model_state = _clone_to_cpu(model.state_dict())
    baseline_optimizer_state = _clone_to_cpu(optimizer.state_dict())
    graph.replay()
    torch.cuda.synchronize()
    graph_loss_value = float(captured_loss.detach().cpu())
    graph_norm_value = float(captured_norm.detach().cpu())
    graph_parameter_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }

    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    print("[v64-graph] timing graph replays", flush=True)
    for _ in range(config.timing_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    positions = (
        config.microbatch_size
        * config.context_length
        * config.gradient_accumulation_steps
        * config.timing_steps
    )
    timing_peak = int(torch.cuda.max_memory_allocated())
    del graph, captured_loss, captured_norm
    gc.collect()
    torch.cuda.empty_cache()

    print("[v64-graph] restoring state for eager parity step", flush=True)
    model.load_state_dict(baseline_model_state, strict=True)
    optimizer.load_state_dict(baseline_optimizer_state)
    eager_loss, eager_norm = _accumulated_step(
        model,
        optimizer,
        inputs,
        targets,
        weights,
        config,
        clear_gradients=True,
    )
    torch.cuda.synchronize()
    eager_loss_value = float(eager_loss.detach().cpu())
    eager_norm_value = float(eager_norm.detach().cpu())
    parameter_comparison = _parameter_comparison(
        model, graph_parameter_state, baseline_model_state
    )
    parity = {
        "graph_loss": graph_loss_value,
        "eager_loss": eager_loss_value,
        "loss_absolute_delta": abs(graph_loss_value - eager_loss_value),
        "graph_gradient_norm": graph_norm_value,
        "eager_gradient_norm": eager_norm_value,
        "gradient_norm_absolute_delta": abs(graph_norm_value - eager_norm_value),
        "updated_parameters": parameter_comparison,
    }
    result = {
        "arm": "delta_state_triton_cuda_graph",
        "parameter_report": parameter_report,
        "effective_batch_size": config.microbatch_size
        * config.gradient_accumulation_steps,
        "physical_microbatch_size": config.microbatch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "capture_seconds": capture_seconds,
        "capture_peak_cuda_bytes": capture_peak,
        "timing_step_seconds": durations,
        "timing_positions_per_second": positions / sum(durations),
        "timing_peak_cuda_bytes": timing_peak,
        "peak_cuda_bytes": max(capture_peak, timing_peak),
        "parity": parity,
        "optimizer": "fused_AdamW_capturable",
        "maximum_gradient_norm": config.maximum_gradient_norm,
        "torch_compile_used": False,
    }
    del (
        model,
        optimizer,
        baseline_model_state,
        baseline_optimizer_state,
        graph_parameter_state,
        eager_loss,
        eager_norm,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _control_arm(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: CudaGraphPreflightConfig,
) -> dict[str, Any]:
    torch.manual_seed(config.control_seed)
    model = MarulhoLanguageModel(_control_config(config)).cuda().train()
    optimizer = _optimizer(model, config, capturable=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("[v64-graph] warming Transformer control", flush=True)
    warmup_loss, warmup_norm = _full_batch_step(
        model,
        optimizer,
        inputs,
        targets,
        weights,
        config,
    )
    torch.cuda.synchronize()
    del warmup_loss, warmup_norm
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []
    print("[v64-graph] timing Transformer optimizer steps", flush=True)
    for _ in range(config.timing_steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _full_batch_step(
            model,
            optimizer,
            inputs,
            targets,
            weights,
            config,
        )
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    positions = inputs.numel() * config.timing_steps
    result = {
        "arm": "fresh_transformer_eager",
        "parameter_count": int(parameter_count),
        "effective_batch_size": inputs.shape[0],
        "physical_microbatch_size": int(inputs.shape[0]),
        "gradient_accumulation_steps": 1,
        "timing_step_seconds": durations,
        "timing_positions_per_second": positions / sum(durations),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "optimizer": "fused_AdamW",
        "maximum_gradient_norm": config.maximum_gradient_norm,
        "torch_compile_used": False,
    }
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_preflight(
    *, config: CudaGraphPreflightConfig, output: Path
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V64 CUDA Graph preflight requires CUDA")
    if config.gradient_accumulation_steps != 2:
        raise ValueError("V64 preflight is frozen at two accumulation steps")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    inputs, targets, weights = _batch(config)
    candidate = _candidate_graph_arm(inputs, targets, weights, config)
    partial = {
        "artifact_kind": "marulho_delta_state_cuda_graph_preflight",
        "surface": SURFACE,
        "decision": "incomplete_transformer_control_pending",
        "config": asdict(config),
        "candidate": candidate,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    _atomic_json(output, partial)
    control = _control_arm(inputs, targets, weights, config)

    parity = candidate["parity"]
    parameter_parity = parity["updated_parameters"]
    throughput_ratio = float(candidate["timing_positions_per_second"]) / float(
        control["timing_positions_per_second"]
    )
    parameter_ratio = float(candidate["parameter_report"]["total_parameters"]) / float(
        control["parameter_count"]
    )
    gates = {
        "parameter_ratio": 0.99 <= parameter_ratio <= 1.01,
        "graph_eager_loss": float(parity["loss_absolute_delta"])
        <= config.graph_eager_loss_tolerance,
        "graph_eager_gradient_norm": float(parity["gradient_norm_absolute_delta"])
        <= config.graph_eager_gradient_norm_tolerance,
        "graph_eager_parameter_names": bool(parameter_parity["names_equal"]),
        "graph_eager_update_cosine": float(
            parameter_parity["update_global_cosine"]
        )
        >= config.graph_eager_update_cosine_tolerance,
        "graph_eager_parameter_rms": float(
            parameter_parity["root_mean_square_element_delta"]
        )
        <= config.graph_eager_parameter_rms_tolerance,
        "graph_eager_parameter_maximum": float(
            parameter_parity["maximum_absolute_element_delta"]
        )
        <= config.graph_eager_parameter_maximum_tolerance,
        "throughput_floor": throughput_ratio
        >= config.preflight_minimum_throughput_ratio,
        "candidate_peak_memory": int(candidate["peak_cuda_bytes"])
        <= config.maximum_peak_cuda_bytes,
        "cuda_observed": int(candidate["peak_cuda_bytes"]) > 0
        and int(control["peak_cuda_bytes"]) > 0,
    }
    decision = (
        "advance_v64_to_terminal_training"
        if all(gates.values())
        else "stop_v64_for_kernel_redesign_no_quality_verdict"
    )
    payload = {
        **partial,
        "decision": decision,
        "control": control,
        "candidate_to_control_throughput_ratio": throughput_ratio,
        "candidate_to_control_parameter_ratio": parameter_ratio,
        "gates": gates,
        "promotion_throughput_gate_passed": throughput_ratio
        >= config.promotion_minimum_throughput_ratio,
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }
    _atomic_json(output, payload)
    payload["report_path"] = str(output)
    payload["report_sha256"] = _sha256(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    config = CudaGraphPreflightConfig(timing_steps=arguments.timing_steps)
    print(json.dumps(run_preflight(config=config, output=arguments.output), indent=2))


if __name__ == "__main__":
    main()
