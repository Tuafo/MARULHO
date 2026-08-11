"""MARULHO causal answer-emphasis objective for continual domain learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def answer_target_mask(
    input_ids: torch.Tensor,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    """Select targets after the latest marker and before the next document EOS."""

    if input_ids.ndim != 2 or marker_ids.ndim != 1:
        raise ValueError("answer mask expects [batch,time] ids and a flat marker")
    marker_size = int(marker_ids.numel())
    if marker_size < 1 or int(input_ids.shape[1]) < marker_size:
        raise ValueError("answer marker must fit inside the training sequence")
    matches = input_ids.unfold(1, marker_size, 1).eq(marker_ids).all(dim=-1)
    marker_ends = F.pad(matches, (marker_size - 1, 0))
    positions = torch.arange(
        1,
        int(input_ids.shape[1]) + 1,
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    last_marker = torch.cummax(
        torch.where(marker_ends, positions, torch.zeros_like(positions)), dim=1
    ).values
    last_eos = torch.cummax(
        torch.where(input_ids.eq(int(eos_id)), positions, torch.zeros_like(positions)),
        dim=1,
    ).values
    return last_marker > last_eos


def answer_weighted_next_token_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
    answer_weight: float,
    pad_id: int | None = None,
) -> torch.Tensor:
    """Renormalized next-token loss with extra credit on marked answer spans."""

    weight = float(answer_weight)
    if not weight >= 1.0:
        raise ValueError("answer_weight must be at least one")
    logits = model.forward(input_ids, collect_telemetry=False)["logits"]
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    mask = answer_target_mask(input_ids, marker_ids=marker_ids, eos_id=int(eos_id))
    valid = (
        torch.ones_like(target_ids, dtype=torch.bool)
        if pad_id is None
        else target_ids.ne(int(pad_id))
    )
    weights = valid.to(token_losses.dtype) * (
        1.0 + mask.to(token_losses.dtype) * (weight - 1.0)
    )
    return (token_losses * weights).sum() / weights.sum()
