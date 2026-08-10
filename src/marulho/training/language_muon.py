"""Matrix-geometry optimizer candidate for MARULHO language training."""

from __future__ import annotations

from collections import defaultdict
import math
import time
from typing import Any, Iterable, Mapping

import torch
from torch import nn


MUON_PAPER_URL = "https://arxiv.org/abs/2502.16982"
MUON_REFERENCE_URL = (
    "https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py"
)
NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz_zeroth_power(
    update: torch.Tensor,
    *,
    steps: int = 5,
) -> torch.Tensor:
    """Approximate the polar factor of one matrix or a batch of matrices."""

    if update.ndim < 2:
        raise ValueError("Muon orthogonalization requires matrix-shaped updates")
    if int(steps) < 1:
        raise ValueError("Muon Newton-Schulz steps must be positive")
    work = update.to(dtype=torch.bfloat16)
    transposed = int(work.shape[-2]) > int(work.shape[-1])
    if transposed:
        work = work.transpose(-2, -1)
    norm = work.float().square().sum(dim=(-2, -1), keepdim=True).sqrt()
    work = work / norm.add(1.0e-7).to(dtype=work.dtype)
    coefficient_a, coefficient_b, coefficient_c = NEWTON_SCHULZ_COEFFICIENTS
    for _ in range(int(steps)):
        gram = work @ work.transpose(-2, -1)
        polynomial = coefficient_b * gram + coefficient_c * (gram @ gram)
        work = coefficient_a * work + polynomial @ work
    if transposed:
        work = work.transpose(-2, -1)
    return work


@torch.compile(fullgraph=True, dynamic=True)
def _compiled_newton_schulz5(update: torch.Tensor) -> torch.Tensor:
    return newton_schulz_zeroth_power(update, steps=5)


class MarulhoMuon(torch.optim.Optimizer):
    """Muon for hidden matrices with an AdamW fallback for other parameters."""

    def __init__(
        self,
        *,
        muon_parameters: Iterable[nn.Parameter],
        adamw_parameters: Iterable[nn.Parameter],
        learning_rate: float,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        newton_schulz_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_epsilon: float = 1.0e-8,
        update_rms_target: float = 0.2,
        compile_orthogonalizer: bool = True,
        matrix_row_partitions: Mapping[nn.Parameter, int] | None = None,
    ) -> None:
        muon = list(muon_parameters)
        adamw = list(adamw_parameters)
        if not muon:
            raise ValueError("Muon requires at least one hidden matrix")
        if any(parameter.ndim != 2 for parameter in muon):
            raise ValueError("Muon hidden parameters must all be matrices")
        if len({id(parameter) for parameter in [*muon, *adamw]}) != len(muon) + len(
            adamw
        ):
            raise ValueError("Muon and AdamW parameter groups must be disjoint")
        partition_by_id: dict[int, int] = {}
        for parameter, partition_count in (matrix_row_partitions or {}).items():
            if id(parameter) not in {id(candidate) for candidate in muon}:
                raise ValueError("Muon row partitions require a Muon parameter")
            count = int(partition_count)
            if count < 2:
                raise ValueError("Muon row partition count must be at least two")
            if int(parameter.shape[0]) % count != 0:
                raise ValueError("Muon matrix rows must divide evenly into partitions")
            partition_by_id[id(parameter)] = count
        if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError("Muon learning_rate must be finite and positive")
        if not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0:
            raise ValueError("Muon weight_decay must be finite and non-negative")
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("Muon momentum must be in [0, 1)")
        if int(newton_schulz_steps) < 1:
            raise ValueError("Muon Newton-Schulz steps must be positive")
        beta1, beta2 = (float(value) for value in adamw_betas)
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("Muon AdamW fallback betas must be in [0, 1)")
        if not math.isfinite(float(adamw_epsilon)) or float(adamw_epsilon) <= 0.0:
            raise ValueError("Muon AdamW epsilon must be finite and positive")
        if not math.isfinite(float(update_rms_target)) or float(
            update_rms_target
        ) <= 0.0:
            raise ValueError("Muon update RMS target must be finite and positive")
        defaults = {
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "newton_schulz_steps": int(newton_schulz_steps),
            "adamw_betas": (beta1, beta2),
            "adamw_epsilon": float(adamw_epsilon),
            "update_rms_target": float(update_rms_target),
        }
        super().__init__([*muon, *adamw], defaults)
        self._muon_ids = {id(parameter) for parameter in muon}
        self._matrix_row_partitions = partition_by_id
        self.compile_orthogonalizer = bool(compile_orthogonalizer)

    def _orthogonalize(self, stacked: torch.Tensor, *, steps: int) -> torch.Tensor:
        if self.compile_orthogonalizer and stacked.is_cuda and int(steps) == 5:
            return _compiled_newton_schulz5(stacked)
        return newton_schulz_zeroth_power(stacked, steps=int(steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            nesterov = bool(group["nesterov"])
            steps = int(group["newton_schulz_steps"])
            update_rms_target = float(group["update_rms_target"])
            grouped: dict[tuple[int, int], list[tuple[int, int, torch.Tensor]]] = (
                defaultdict(list)
            )
            prepared: list[tuple[nn.Parameter, list[torch.Tensor | None]]] = []
            for parameter in group["params"]:
                if id(parameter) not in self._muon_ids or parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")
                state = self.state[parameter]
                momentum_buffer = state.get("momentum_buffer")
                if not isinstance(momentum_buffer, torch.Tensor):
                    momentum_buffer = torch.zeros_like(gradient)
                    state["momentum_buffer"] = momentum_buffer
                momentum_buffer.mul_(momentum).add_(gradient)
                update = (
                    gradient.add(momentum_buffer, alpha=momentum)
                    if nesterov
                    else momentum_buffer
                )
                partition_count = int(self._matrix_row_partitions.get(id(parameter), 1))
                if partition_count == 1:
                    parts = (update,)
                else:
                    parts = update.reshape(
                        partition_count,
                        int(update.shape[0]) // partition_count,
                        int(update.shape[1]),
                    ).unbind(0)
                prepared_index = len(prepared)
                prepared.append((parameter, [None] * len(parts)))
                for part_index, part in enumerate(parts):
                    grouped[tuple(int(value) for value in part.shape)].append(
                        (prepared_index, part_index, part)
                    )
            for entries in grouped.values():
                stacked = torch.stack([update for _, _, update in entries], dim=0)
                orthogonal = self._orthogonalize(stacked, steps=steps)
                for (prepared_index, part_index, _), update in zip(
                    entries, orthogonal, strict=True
                ):
                    prepared[prepared_index][1][part_index] = update
            for parameter, parts in prepared:
                if any(part is None for part in parts):
                    raise RuntimeError("Muon orthogonalization omitted a matrix partition")
                complete_parts = [part for part in parts if isinstance(part, torch.Tensor)]
                update = (
                    complete_parts[0]
                    if len(complete_parts) == 1
                    else torch.cat(complete_parts, dim=0)
                )
                rows, columns = (int(value) for value in parameter.shape)
                adjusted_rate = (
                    learning_rate
                    * update_rms_target
                    * math.sqrt(float(max(rows, columns)))
                )
                parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(
                    update.to(dtype=parameter.dtype),
                    alpha=-adjusted_rate,
                )

            beta1, beta2 = group["adamw_betas"]
            epsilon = float(group["adamw_epsilon"])
            for parameter in group["params"]:
                if id(parameter) in self._muon_ids or parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError(
                        "Muon's AdamW fallback does not support sparse gradients"
                    )
                state = self.state[parameter]
                step = int(state.get("step", 0)) + 1
                state["step"] = step
                first_moment = state.get("first_moment")
                second_moment = state.get("second_moment")
                if not isinstance(first_moment, torch.Tensor):
                    first_moment = torch.zeros_like(gradient)
                    second_moment = torch.zeros_like(gradient)
                    state["first_moment"] = first_moment
                    state["second_moment"] = second_moment
                assert isinstance(second_moment, torch.Tensor)
                first_moment.lerp_(gradient, 1.0 - float(beta1))
                second_moment.lerp_(gradient.square(), 1.0 - float(beta2))
                bias_correction1 = 1.0 - float(beta1) ** step
                bias_correction2 = 1.0 - float(beta2) ** step
                correction = bias_correction1 / math.sqrt(bias_correction2)
                normalized = first_moment / (second_moment.sqrt() + epsilon)
                parameter.mul_(1.0 - learning_rate * weight_decay)
                parameter.add_(normalized, alpha=-learning_rate / correction)
        return loss


def build_language_muon(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    compile_orthogonalizer: bool = True,
    per_head_attention_qkv: bool = False,
) -> tuple[MarulhoMuon, Mapping[str, Any]]:
    """Assign hidden matrices to Muon and embeddings/norms to AdamW."""

    muon_named: list[tuple[str, nn.Parameter]] = []
    adamw_named: list[tuple[str, nn.Parameter]] = []
    for name, parameter in model.named_parameters():
        is_embedding = name.startswith("token_embedding.") or name.startswith(
            "lm_head."
        )
        if parameter.ndim == 2 and not is_embedding:
            muon_named.append((name, parameter))
        else:
            adamw_named.append((name, parameter))
    row_partitions: dict[nn.Parameter, int] = {}
    partitioned_names: list[str] = []
    if bool(per_head_attention_qkv):
        config = getattr(model, "config", None)
        attention_heads = int(getattr(config, "attention_heads", 0))
        if attention_heads < 1:
            raise ValueError("Per-head Muon requires model.config.attention_heads")
        partition_count = 3 * attention_heads
        for name, parameter in muon_named:
            if not name.endswith(".attention.qkv.weight"):
                continue
            if int(parameter.shape[0]) % partition_count != 0:
                raise ValueError("Combined QKV rows must divide into Q/K/V attention heads")
            row_partitions[parameter] = partition_count
            partitioned_names.append(name)
        if not partitioned_names:
            raise ValueError("Per-head Muon found no combined attention QKV matrices")
    optimizer = MarulhoMuon(
        muon_parameters=[parameter for _, parameter in muon_named],
        adamw_parameters=[parameter for _, parameter in adamw_named],
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        adamw_betas=adamw_betas,
        compile_orthogonalizer=bool(compile_orthogonalizer),
        matrix_row_partitions=row_partitions,
    )
    return optimizer, {
        "kind": "marulho_muon_with_adamw_fallback",
        "fused": False,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "momentum": 0.95,
        "nesterov": True,
        "newton_schulz_steps": 5,
        "newton_schulz_coefficients": list(NEWTON_SCHULZ_COEFFICIENTS),
        "update_rms_target": 0.2,
        "adamw_fallback_betas": list(adamw_betas),
        "muon_parameter_names": [name for name, _ in muon_named],
        "muon_parameter_count": sum(
            int(parameter.numel()) for _, parameter in muon_named
        ),
        "adamw_fallback_parameter_names": [name for name, _ in adamw_named],
        "adamw_fallback_parameter_count": sum(
            int(parameter.numel()) for _, parameter in adamw_named
        ),
        "orthogonalization_grouped_by_matrix_shape": True,
        "orthogonalizer_compile_requested": bool(compile_orthogonalizer),
        "orthogonalizer_dynamic_matrix_shapes": bool(compile_orthogonalizer),
        "per_head_attention_qkv": bool(per_head_attention_qkv),
        "row_partitioned_parameter_names": partitioned_names,
        "row_partition_count_by_parameter": {
            name: int(row_partitions[parameter])
            for name, parameter in muon_named
            if parameter in row_partitions
        },
        "paper": MUON_PAPER_URL,
        "reference_implementation": MUON_REFERENCE_URL,
        "external_weights_loaded": False,
    }


def warm_language_muon_orthogonalizer_shapes(
    shapes: Iterable[tuple[int, ...]],
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    """Compile Muon's dynamic orthogonalizer outside measured training steps."""

    normalized: set[tuple[int, int, int]] = set()
    for shape in shapes:
        if len(shape) == 2:
            rows, columns = shape
            batch = 1
        elif len(shape) == 3:
            batch, rows, columns = shape
        else:
            raise ValueError("Muon warmup shapes must be [rows, columns] or [batch, rows, columns]")
        normalized.add((int(batch), int(rows), int(columns)))
    unique_shapes = tuple(sorted(normalized))
    if device.type != "cuda":
        raise ValueError("Muon orthogonalizer warmup requires CUDA")
    if not unique_shapes:
        raise ValueError("Muon orthogonalizer warmup requires matrix shapes")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch, rows, columns in unique_shapes:
            probe = torch.zeros(
                batch, rows, columns, device=device, dtype=torch.bfloat16
            )
            result = _compiled_newton_schulz5(probe)
            if not bool(torch.isfinite(result).all().item()):
                raise RuntimeError("Muon orthogonalizer warmup produced non-finite output")
    torch.cuda.synchronize(device)
    return {
        "surface": "marulho_muon_orthogonalizer_warmup.v1",
        "dynamic_shapes": True,
        "stacked_matrix_shapes": [list(shape) for shape in unique_shapes],
        "shape_count": len(unique_shapes),
        "elapsed_seconds": time.perf_counter() - started,
        "included_in_arm_training_seconds": False,
    }
