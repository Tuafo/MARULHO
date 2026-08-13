"""V64 matched delta-state cortex falsification and CUDA preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

import torch
import torch.nn.functional as F

from marulho.evaluation.language_training_experiment import (
    _prepare_triton_compiler_compatibility,
)
from marulho.training.language_delta_state import (
    DeltaStateLanguageModelConfig,
    MarulhoDeltaStateLanguageModel,
    delta_state_parameter_report,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
)


SURFACE = "marulho_delta_state_falsification.v1"
PREFLIGHT_SURFACE = "marulho_delta_state_cuda_preflight.v1"
DEFAULT_PREFLIGHT_OUTPUT = Path(
    "reports/language_scaling/delta-state-v64-cuda-preflight-20260812.json"
)
DEFAULT_CANDIDATE_PREFLIGHT_OUTPUT = Path(
    "reports/language_scaling/delta-state-v64-candidate-preflight-20260812.json"
)
DEFAULT_CONTROL_PREFLIGHT_OUTPUT = Path(
    "reports/language_scaling/delta-state-v64-control-preflight-20260812.json"
)


@dataclass(frozen=True)
class V64Config:
    batch_size: int = 32
    context_length: int = 320
    optimizer_steps: int = 8192
    general_steps: int = 6144
    qa_steps: int = 2048
    answer_weight: float = 4.0
    learning_rate: float = 4.0e-4
    weight_decay: float = 0.1
    maximum_gradient_norm: float = 1.0
    warmup_fraction: float = 0.05
    minimum_learning_rate_fraction: float = 0.1
    compile_loss_tolerance: float = 1.0e-3
    gradient_cosine_tolerance: float = 0.999
    gradient_maximum_delta_tolerance: float = 1.0e-2
    preflight_minimum_throughput_ratio: float = 0.50
    promotion_minimum_throughput_ratio: float = 0.70
    maximum_peak_cuda_bytes: int = 12_348_027_699
    model_seed: int = 16411
    control_seed: int = 16412
    data_seed: int = 16413
    timing_steps: int = 3


def _atomic_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _control_config(config: V64Config) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=int(config.context_length),
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        active_language_path="marulho_v64_fresh_transformer_control",
    )


def weighted_causal_loss(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    logits = model(
        input_ids,
        collect_telemetry=False,
        decode_vocab_only=False,
    )["logits"]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    weights = target_weights.to(device=losses.device, dtype=losses.dtype)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def build_compiled_weighted_loss(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def logits(input_ids: torch.Tensor) -> torch.Tensor:
        return model(
            input_ids,
            collect_telemetry=False,
            decode_vocab_only=False,
        )["logits"]

    compiled_logits = torch.compile(
        logits,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
    )

    def loss(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_weights: torch.Tensor,
    ) -> torch.Tensor:
        observed_logits = compiled_logits(input_ids)
        losses = F.cross_entropy(
            observed_logits.reshape(-1, observed_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(target_ids.shape)
        weights = target_weights.to(device=losses.device, dtype=losses.dtype)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)

    return loss


def _gradient_inventory(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
) -> dict[str, Any]:
    rows = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        rows.append(
            {
                "name": name,
                "present": gradient is not None,
                "finite": gradient is not None
                and bool(torch.isfinite(gradient).all().item()),
                "nonzero": gradient is not None
                and bool(torch.count_nonzero(gradient).item()),
            }
        )
    return {
        "parameter_tensor_count": len(rows),
        "gradient_tensor_count": sum(row["present"] for row in rows),
        "finite_gradient_tensor_count": sum(row["finite"] for row in rows),
        "nonzero_gradient_tensor_count": sum(row["nonzero"] for row in rows),
        "all_present_finite_nonzero": bool(rows)
        and all(row["present"] and row["finite"] and row["nonzero"] for row in rows),
        "missing_or_invalid_names": [
            row["name"]
            for row in rows
            if not (row["present"] and row["finite"] and row["nonzero"])
        ],
    }


def _copy_gradients(
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def compare_gradients(
    reference: Mapping[str, torch.Tensor],
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
) -> dict[str, Any]:
    dot = 0.0
    reference_norm = 0.0
    observed_norm = 0.0
    maximum_delta = 0.0
    compared = 0
    names_equal = set(reference) == {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None or name not in reference:
            continue
        expected = reference[name]
        actual = parameter.grad.detach().float().cpu()
        dot += float(torch.sum(expected * actual).item())
        reference_norm += float(torch.sum(expected.square()).item())
        observed_norm += float(torch.sum(actual.square()).item())
        maximum_delta = max(
            maximum_delta,
            float(torch.max(torch.abs(expected - actual)).item()),
        )
        compared += 1
    cosine = dot / max(1.0e-30, math.sqrt(reference_norm * observed_norm))
    return {
        "names_equal": bool(names_equal),
        "compared_tensor_count": compared,
        "global_cosine": cosine,
        "maximum_absolute_element_delta": maximum_delta,
    }


def preflight_decision(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    config: V64Config,
) -> tuple[str, dict[str, bool]]:
    throughput_ratio = float(candidate["positions_per_second"]) / float(
        control["positions_per_second"]
    )
    parity = candidate["compiled_eager_parity"]
    gradients = candidate["compiled_gradients"]
    gates = {
        "parameter_ratio": 0.99
        <= float(candidate["parameter_count"]) / float(control["parameter_count"])
        <= 1.01,
        "compiled_loss_parity": float(parity["loss_absolute_delta"])
        <= float(config.compile_loss_tolerance),
        "compiled_gradient_names": bool(parity["gradients"]["names_equal"]),
        "compiled_gradient_cosine": float(parity["gradients"]["global_cosine"])
        >= float(config.gradient_cosine_tolerance),
        "compiled_gradient_maximum_delta": float(
            parity["gradients"]["maximum_absolute_element_delta"]
        )
        <= float(config.gradient_maximum_delta_tolerance),
        "complete_finite_nonzero_gradients": bool(
            gradients["all_present_finite_nonzero"]
        ),
        "throughput_floor": throughput_ratio
        >= float(config.preflight_minimum_throughput_ratio),
        "candidate_peak_memory": int(candidate["peak_cuda_bytes"])
        <= int(config.maximum_peak_cuda_bytes),
        "cuda_observed": int(candidate["peak_cuda_bytes"]) > 0
        and int(control["peak_cuda_bytes"]) > 0,
    }
    decision = (
        "advance_v64_to_terminal_training"
        if all(gates.values())
        else "stop_v64_for_kernel_redesign_no_quality_verdict"
    )
    return decision, gates


def _preflight_arm(
    *,
    name: str,
    model: MarulhoLanguageModel | MarulhoDeltaStateLanguageModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    config: V64Config,
    compare_compiled_gradients: bool,
) -> dict[str, Any]:
    print(f"[v64-preflight] {name} eager gradient oracle", flush=True)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(model.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_loss = weighted_causal_loss(model, inputs, targets, weights)
    eager_loss.backward()
    eager_inventory = _gradient_inventory(model)
    eager_gradients = _copy_gradients(model) if compare_compiled_gradients else {}
    eager_loss_value = float(eager_loss.detach().float().cpu())
    model.zero_grad(set_to_none=True)

    compiled_loss = build_compiled_weighted_loss(model)
    print(f"[v64-preflight] {name} compiling model graph", flush=True)
    torch.cuda.synchronize(model.device)
    compile_started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        observed = compiled_loss(inputs, targets, weights)
    observed.backward()
    torch.cuda.synchronize(model.device)
    compile_seconds = time.perf_counter() - compile_started
    compiled_loss_value = float(observed.detach().float().cpu())
    compiled_inventory = _gradient_inventory(model)
    gradient_parity = (
        compare_gradients(eager_gradients, model)
        if compare_compiled_gradients
        else {
            "names_equal": True,
            "compared_tensor_count": 0,
            "global_cosine": 1.0,
            "maximum_absolute_element_delta": 0.0,
            "not_requested_for_control": True,
        }
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        fused=True,
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
    optimizer.step()
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats(model.device)
    print(f"[v64-preflight] {name} timing optimizer-inclusive steps", flush=True)
    for _ in range(int(config.timing_steps)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compiled_loss(inputs, targets, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.maximum_gradient_norm)
        optimizer.step()
        torch.cuda.synchronize(model.device)
        durations.append(time.perf_counter() - started)
    elapsed = sum(durations)
    positions = int(inputs.numel()) * len(durations)
    return {
        "arm": name,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "eager_gradients": eager_inventory,
        "compiled_gradients": compiled_inventory,
        "compiled_eager_parity": {
            "eager_loss": eager_loss_value,
            "compiled_loss": compiled_loss_value,
            "loss_absolute_delta": abs(compiled_loss_value - eager_loss_value),
            "gradients": gradient_parity,
        },
        "compile_seconds": compile_seconds,
        "timing_step_seconds": durations,
        "timing_seconds": elapsed,
        "timing_steps": len(durations),
        "positions": positions,
        "positions_per_second": positions / max(elapsed, 1.0e-9),
        "amortized_positions_per_second_including_compile": positions
        / max(elapsed + compile_seconds, 1.0e-9),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "optimizer": "fused_AdamW",
        "compiled_scope": "model_forward_backward_fullgraph",
        "explicit_uncompiled_scope": "weighted_cross_entropy_and_optimizer",
        "fp32_master_parameters": all(
            parameter.dtype == torch.float32 for parameter in model.parameters()
        ),
        "dense_autocast": "bfloat16",
        "external_llm_used": False,
    }


def _synthetic_preflight_batch(
    runtime: torch.device, config: V64Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=runtime).manual_seed(int(config.data_seed))
    inputs = torch.randint(
        0,
        8192,
        (int(config.batch_size), int(config.context_length)),
        generator=generator,
        device=runtime,
    )
    targets = torch.randint(
        0,
        8192,
        inputs.shape,
        generator=generator,
        device=runtime,
    )
    weights = torch.ones(inputs.shape, device=runtime, dtype=torch.float32)
    weights[:, -80:] = float(config.answer_weight)
    return inputs, targets, weights


def _prepare_cuda_preflight(device: str) -> tuple[torch.device, dict[str, Any]]:
    runtime = torch.device(device)
    if runtime.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V64 preflight is frozen as a CUDA experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    return runtime, _prepare_triton_compiler_compatibility()


def run_v64_candidate_preflight(
    *,
    output_path: str | Path = DEFAULT_CANDIDATE_PREFLIGHT_OUTPUT,
    device: str = "cuda",
    config: V64Config = V64Config(),
) -> dict[str, Any]:
    runtime, compatibility = _prepare_cuda_preflight(device)
    inputs, targets, weights = _synthetic_preflight_batch(runtime, config)
    torch.manual_seed(int(config.model_seed))
    model = MarulhoDeltaStateLanguageModel(DeltaStateLanguageModelConfig()).to(runtime)
    parameter_report = delta_state_parameter_report(model)
    arm = _preflight_arm(
        name="delta_state_cortex",
        model=model,
        inputs=inputs,
        targets=targets,
        weights=weights,
        config=config,
        compare_compiled_gradients=True,
    )
    payload = {
        "artifact_kind": "marulho_delta_state_candidate_cuda_preflight",
        "surface": PREFLIGHT_SURFACE,
        "config": asdict(config),
        "shape": asdict(DeltaStateLanguageModelConfig()),
        "parameter_report": parameter_report,
        "arm": arm,
        "compiler_compatibility": compatibility,
        "hardware": {
            "device": str(runtime),
            "name": torch.cuda.get_device_name(runtime),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "synthetic_values_have_no_quality_meaning": True,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    output = _atomic_json(output_path, payload)
    payload["report_path"] = str(output)
    payload["report_sha256"] = _sha256_file(output)
    print(f"[v64-preflight] saved candidate {output}", flush=True)
    return payload


def run_v64_control_preflight(
    *,
    output_path: str | Path = DEFAULT_CONTROL_PREFLIGHT_OUTPUT,
    device: str = "cuda",
    config: V64Config = V64Config(),
) -> dict[str, Any]:
    runtime, compatibility = _prepare_cuda_preflight(device)
    inputs, targets, weights = _synthetic_preflight_batch(runtime, config)
    torch.manual_seed(int(config.control_seed))
    model = MarulhoLanguageModel(_control_config(config)).to(runtime)
    arm = _preflight_arm(
        name="transformer_control",
        model=model,
        inputs=inputs,
        targets=targets,
        weights=weights,
        config=config,
        compare_compiled_gradients=False,
    )
    payload = {
        "artifact_kind": "marulho_delta_state_control_cuda_preflight",
        "surface": PREFLIGHT_SURFACE,
        "config": asdict(config),
        "shape": asdict(_control_config(config)),
        "arm": arm,
        "compiler_compatibility": compatibility,
        "hardware": {
            "device": str(runtime),
            "name": torch.cuda.get_device_name(runtime),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "synthetic_values_have_no_quality_meaning": True,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    output = _atomic_json(output_path, payload)
    payload["report_path"] = str(output)
    payload["report_sha256"] = _sha256_file(output)
    print(f"[v64-preflight] saved control {output}", flush=True)
    return payload


def _load_preflight_arm(
    path: str | Path, *, expected_kind: str, config: V64Config
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != expected_kind:
        raise ValueError(f"Unexpected V64 preflight artifact at {path}")
    if payload.get("config") != asdict(config):
        raise ValueError(f"V64 preflight config differs at {path}")
    return dict(payload)


def run_v64_preflight(
    *,
    output_path: str | Path = DEFAULT_PREFLIGHT_OUTPUT,
    device: str = "cuda",
    config: V64Config = V64Config(),
) -> dict[str, Any]:
    del device
    candidate_payload = _load_preflight_arm(
        DEFAULT_CANDIDATE_PREFLIGHT_OUTPUT,
        expected_kind="marulho_delta_state_candidate_cuda_preflight",
        config=config,
    )
    control_payload = _load_preflight_arm(
        DEFAULT_CONTROL_PREFLIGHT_OUTPUT,
        expected_kind="marulho_delta_state_control_cuda_preflight",
        config=config,
    )
    candidate = dict(candidate_payload["arm"])
    control = dict(control_payload["arm"])

    decision, gates = preflight_decision(candidate, control, config)
    throughput_ratio = float(candidate["positions_per_second"]) / float(
        control["positions_per_second"]
    )
    report = {
        "artifact_kind": "marulho_delta_state_cuda_preflight",
        "surface": PREFLIGHT_SURFACE,
        "decision": decision,
        "config": asdict(config),
        "candidate_shape": asdict(DeltaStateLanguageModelConfig()),
        "control_shape": asdict(_control_config(config)),
        "candidate_parameter_report": candidate_payload["parameter_report"],
        "candidate": candidate,
        "control": control,
        "throughput_ratio": throughput_ratio,
        "promotion_throughput_gate_passed_diagnostic_only": throughput_ratio
        >= float(config.promotion_minimum_throughput_ratio),
        "gates": gates,
        "candidate_artifact_path": str(DEFAULT_CANDIDATE_PREFLIGHT_OUTPUT),
        "candidate_artifact_sha256": _sha256_file(DEFAULT_CANDIDATE_PREFLIGHT_OUTPUT),
        "control_artifact_path": str(DEFAULT_CONTROL_PREFLIGHT_OUTPUT),
        "control_artifact_sha256": _sha256_file(DEFAULT_CONTROL_PREFLIGHT_OUTPUT),
        "compiler_compatibility": candidate_payload["compiler_compatibility"],
        "hardware": candidate_payload["hardware"],
        "synthetic_values_have_no_quality_meaning": True,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    output = _atomic_json(output_path, report)
    report["report_path"] = str(output)
    report["report_sha256"] = _sha256_file(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=("candidate", "control", "assemble"), default="assemble"
    )
    arguments = parser.parse_args()
    if arguments.arm == "candidate":
        report = run_v64_candidate_preflight()
    elif arguments.arm == "control":
        report = run_v64_control_preflight()
    else:
        report = run_v64_preflight()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
