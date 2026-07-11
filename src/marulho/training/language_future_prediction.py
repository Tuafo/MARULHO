"""Training-only multi-horizon prediction heads for the V11 language trunk."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_model import LanguageBatch
from marulho.training.language_transformer import TransformerRMSNorm


@dataclass(frozen=True)
class FuturePredictionConfig:
    horizons: tuple[int, ...] = (2, 4, 8)
    auxiliary_weight: float = 0.25
    active_language_path: str = (
        "marulho_hashed_micro_experts_future_prediction_v13"
    )


def _validate_future_config(
    config: FuturePredictionConfig,
    *,
    context_length: int,
) -> None:
    horizons = tuple(int(value) for value in config.horizons)
    if not horizons:
        raise ValueError("future prediction requires at least one horizon")
    if horizons != tuple(sorted(set(horizons))):
        raise ValueError("future prediction horizons must be unique and sorted")
    if horizons[0] < 2:
        raise ValueError("future prediction horizons must be at least two")
    if horizons[-1] > int(context_length):
        raise ValueError("future prediction horizon exceeds model context")
    weight = float(config.auxiliary_weight)
    if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
        raise ValueError("future prediction auxiliary_weight must be in (0, 1]")
    if not str(config.active_language_path).strip():
        raise ValueError("future prediction active_language_path is required")


class MarulhoFuturePredictionLanguageModel(
    MarulhoHashedMicroExpertLanguageModel
):
    """V11 inference graph plus temporary independent future-token heads."""

    training_surface = "marulho_future_prediction_training.v1"

    def __init__(
        self,
        hashed_config: HashedMicroExpertConfig,
        future_config: FuturePredictionConfig = FuturePredictionConfig(),
    ) -> None:
        _validate_future_config(
            future_config,
            context_length=int(hashed_config.context_length),
        )
        super().__init__(hashed_config)
        self.future_config = future_config
        width = int(hashed_config.width)
        self.future_norms = nn.ModuleList(
            TransformerRMSNorm(width) for _ in future_config.horizons
        )
        self.future_projections = nn.ModuleList(
            nn.Linear(width, width, bias=False) for _ in future_config.horizons
        )
        for projection in self.future_projections:
            nn.init.eye_(projection.weight)

    def future_loss_components(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = False,
    ) -> dict[str, Any]:
        result = self._forward_hidden(
            input_ids,
            collect_telemetry=collect_telemetry,
        )
        hidden = result["hidden"]
        targets = target_ids.to(device=self.device, dtype=torch.long)
        if hidden.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        base_logits = self.lm_head(hidden)
        base_loss = F.cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            targets.reshape(-1),
        )
        future_losses: list[torch.Tensor] = []
        for horizon, norm, projection in zip(
            self.future_config.horizons,
            self.future_norms,
            self.future_projections,
            strict=True,
        ):
            shift = int(horizon) - 1
            future_hidden = projection(norm(hidden[:, :-shift]))
            future_logits = F.linear(future_hidden, self.lm_head.weight)
            future_targets = targets[:, shift:]
            future_losses.append(
                F.cross_entropy(
                    future_logits.reshape(-1, future_logits.shape[-1]),
                    future_targets.reshape(-1),
                )
            )
        auxiliary_loss = torch.stack(future_losses).mean()
        total_loss = base_loss + float(
            self.future_config.auxiliary_weight
        ) * auxiliary_loss
        return {
            "loss": total_loss,
            "base_loss": base_loss,
            "auxiliary_loss": auxiliary_loss,
            "future_losses": tuple(future_losses),
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
    ) -> dict[str, Any]:
        components = self.future_loss_components(
            input_ids,
            target_ids,
            collect_telemetry=collect_telemetry,
        )
        evidence = {
            "surface": self.training_surface,
            "horizons": [int(value) for value in self.future_config.horizons],
            "auxiliary_weight": float(self.future_config.auxiliary_weight),
            "full_vocab_logits_materialized_per_head": True,
            "future_heads_inference_persistent": False,
            "external_llm_used": False,
        }
        return {
            "loss": components["loss"],
            "loss_kind": "next_token_plus_multi_horizon_cross_entropy",
            "loss_evidence": evidence if return_evidence else {},
            "state": components["state"],
            "telemetry": components["telemetry"],
        }

    def training_parameter_report(self) -> dict[str, Any]:
        total = sum(int(value.numel()) for value in self.parameters())
        temporary = sum(
            int(value.numel())
            for name, value in self.named_parameters()
            if name.startswith("future_")
        )
        return {
            "surface": "marulho_future_prediction_parameter_report.v1",
            "training_parameters": total,
            "temporary_future_head_parameters": temporary,
            "inference_parameters": total - temporary,
            "future_head_fraction": temporary / total,
            "horizons": [int(value) for value in self.future_config.horizons],
            "future_heads_inference_persistent": False,
            "external_llm_used": False,
        }


def build_future_prediction_model(
    base_model: MarulhoHashedMicroExpertLanguageModel,
    future_config: FuturePredictionConfig = FuturePredictionConfig(),
) -> MarulhoFuturePredictionLanguageModel:
    """Attach initialized training heads while preserving base logits exactly."""

    if base_model.hashed_config.mode != "token_hash":
        raise ValueError("future prediction requires the token_hash base mode")
    hashed_config = replace(
        base_model.hashed_config,
        active_language_path=str(future_config.active_language_path),
    )
    model = MarulhoFuturePredictionLanguageModel(hashed_config, future_config)
    base_state = base_model.state_dict()
    incompatible = model.load_state_dict(base_state, strict=False)
    expected_missing = {
        name for name in model.state_dict() if name.startswith("future_")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "future prediction base load has unexpected missing tensors: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "future prediction base load has unexpected tensors: "
            f"{incompatible.unexpected_keys}"
        )
    model.train(base_model.training)
    return model


def strip_future_prediction_heads(
    model: MarulhoFuturePredictionLanguageModel,
) -> MarulhoHashedMicroExpertLanguageModel:
    """Return the trained MARULHO inference graph without auxiliary heads."""

    inference_model = MarulhoHashedMicroExpertLanguageModel(
        model.hashed_config
    ).to(model.device)
    source_state = model.state_dict()
    inference_state = {
        name: source_state[name] for name in inference_model.state_dict()
    }
    inference_model.load_state_dict(inference_state, strict=True)
    inference_model.train(model.training)
    return inference_model


@torch.no_grad()
def future_prediction_objective_report(
    model: MarulhoFuturePredictionLanguageModel,
    batches: Sequence[LanguageBatch],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("future prediction objective report requires batches")
    was_training = model.training
    model.eval()
    base_weighted = 0.0
    base_tokens = 0
    future_weighted = {
        int(horizon): 0.0 for horizon in model.future_config.horizons
    }
    future_tokens = {
        int(horizon): 0 for horizon in model.future_config.horizons
    }
    for batch in batches:
        runtime = batch.to(model.device)
        components = model.future_loss_components(
            runtime.input_ids,
            runtime.target_ids,
            collect_telemetry=False,
        )
        count = int(runtime.target_ids.numel())
        base_weighted += float(components["base_loss"].float().cpu()) * count
        base_tokens += count
        for horizon, loss in zip(
            model.future_config.horizons,
            components["future_losses"],
            strict=True,
        ):
            horizon_count = int(runtime.target_ids.shape[0]) * (
                int(runtime.target_ids.shape[1]) - int(horizon) + 1
            )
            future_weighted[int(horizon)] += float(loss.float().cpu()) * horizon_count
            future_tokens[int(horizon)] += horizon_count
    model.train(was_training)
    future_losses = {
        str(horizon): future_weighted[horizon] / future_tokens[horizon]
        for horizon in future_weighted
    }
    auxiliary = sum(future_losses.values()) / len(future_losses)
    base = base_weighted / base_tokens
    return {
        "surface": "marulho_future_prediction_objective_report.v1",
        "base_next_token_loss": base,
        "future_losses": future_losses,
        "mean_future_loss": auxiliary,
        "weighted_training_objective": base
        + float(model.future_config.auxiliary_weight) * auxiliary,
        "base_target_tokens": base_tokens,
        "future_target_tokens": {
            str(horizon): count for horizon, count in future_tokens.items()
        },
        "configuration": asdict(model.future_config),
        "external_llm_used": False,
    }
