"""Matrix-geometry optimizer candidate for MARULHO language training."""

from __future__ import annotations

from collections import defaultdict
import math
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


@torch.compile(fullgraph=True, dynamic=False)
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
            grouped: dict[
                tuple[int, int],
                list[tuple[nn.Parameter, torch.Tensor]],
            ] = defaultdict(list)
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
                grouped[tuple(int(value) for value in update.shape)].append(
                    (parameter, update)
                )
            for entries in grouped.values():
                stacked = torch.stack([update for _, update in entries], dim=0)
                orthogonal = self._orthogonalize(stacked, steps=steps)
                for (parameter, _), update in zip(entries, orthogonal, strict=True):
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
    optimizer = MarulhoMuon(
        muon_parameters=[parameter for _, parameter in muon_named],
        adamw_parameters=[parameter for _, parameter in adamw_named],
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        adamw_betas=adamw_betas,
        compile_orthogonalizer=bool(compile_orthogonalizer),
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
        "paper": MUON_PAPER_URL,
        "reference_implementation": MUON_REFERENCE_URL,
        "external_weights_loaded": False,
    }
