"""Byte-trie contrastive objective for marked causal-language answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.training.language_answer_objective import answer_target_mask


@dataclass(frozen=True)
class RoleContrastiveBranch:
    """One wrong next-byte branch while spelling a known role filler."""

    group: str
    value: str
    pattern_ids: tuple[int, ...]
    target_offset: int
    negative_ids: tuple[int, ...]


PreparedRoleBranch = tuple[torch.Tensor, int, torch.Tensor]


def build_role_contrastive_branches(
    tokenizer,
    groups: Mapping[str, Sequence[str]],
) -> tuple[RoleContrastiveBranch, ...]:
    """Build trie-divergence negatives for a byte-level tokenizer.

    Patterns include a leading space to establish a word boundary. At each
    byte of the true value, only alternative bytes reachable after the same
    true prefix are negatives. This avoids treating a multi-byte word as one
    vocabulary token.
    """

    branches: list[RoleContrastiveBranch] = []
    for group, raw_values in groups.items():
        values = tuple(dict.fromkeys(str(value) for value in raw_values))
        if len(values) < 2:
            raise ValueError(f"role group {group!r} needs at least two values")
        encoded = {
            value: tuple(
                int(token_id)
                for token_id in tokenizer.encode(
                    value, add_bos=False, add_eos=False
                )
            )
            for value in values
        }
        for value, value_ids in encoded.items():
            pattern = tuple(
                int(token_id)
                for token_id in tokenizer.encode(
                    f" {value}", add_bos=False, add_eos=False
                )
            )
            leading_size = len(pattern) - len(value_ids)
            if leading_size < 1:
                raise ValueError("role patterns must own a leading boundary")
            for position, correct_id in enumerate(value_ids):
                prefix = value_ids[:position]
                negatives = sorted(
                    {
                        other_ids[position]
                        for other, other_ids in encoded.items()
                        if other != value
                        and len(other_ids) > position
                        and other_ids[:position] == prefix
                        and other_ids[position] != correct_id
                    }
                )
                if negatives:
                    branches.append(
                        RoleContrastiveBranch(
                            group=str(group),
                            value=value,
                            pattern_ids=pattern,
                            target_offset=leading_size + position,
                            negative_ids=tuple(int(token_id) for token_id in negatives),
                        )
                    )
    if not branches:
        raise ValueError("role groups produced no contrastive branches")
    return tuple(branches)


def prepare_role_contrastive_branches(
    branches: Sequence[RoleContrastiveBranch],
    *,
    device: torch.device | str,
) -> tuple[PreparedRoleBranch, ...]:
    """Move immutable branch constants to the execution device once."""

    return tuple(
        (
            torch.tensor(branch.pattern_ids, dtype=torch.long, device=device),
            int(branch.target_offset),
            torch.tensor(branch.negative_ids, dtype=torch.long, device=device),
        )
        for branch in branches
    )


def role_contrastive_active_count(
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    branches: Sequence[PreparedRoleBranch],
) -> torch.Tensor:
    """Count recognized trie-divergence decisions in a batch."""

    if target_ids.ndim != 2 or answer_mask.shape != target_ids.shape:
        raise ValueError("role contrast count expects aligned [batch,time] tensors")
    count = torch.zeros((), dtype=torch.long, device=target_ids.device)
    for pattern_ids, _, _ in branches:
        pattern_size = int(pattern_ids.numel())
        if pattern_size > int(target_ids.shape[1]):
            continue
        occurrences = target_ids.unfold(1, pattern_size, 1).eq(pattern_ids).all(dim=-1)
        occurrences = occurrences & answer_mask.unfold(1, pattern_size, 1).all(dim=-1)
        count = count + occurrences.sum()
    return count


def role_contrastive_unlikelihood(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    branches: Sequence[PreparedRoleBranch],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize probability mass on wrong byte-trie branches.

    Returns the mean active-branch penalty and the active branch count. A batch
    without a recognized answer filler returns differentiable zero.
    """

    if logits.ndim != 3 or target_ids.ndim != 2 or answer_mask.ndim != 2:
        raise ValueError("role contrast expects [batch,time,vocab] logits and ids")
    if logits.shape[:2] != target_ids.shape or target_ids.shape != answer_mask.shape:
        raise ValueError("role contrast logits, targets, and mask must align")
    total = logits.float().sum() * 0.0
    count = torch.zeros((), dtype=torch.long, device=logits.device)
    for pattern_ids, target_offset, negative_ids in branches:
        pattern_size = int(pattern_ids.numel())
        if pattern_size > int(target_ids.shape[1]):
            continue
        occurrences = target_ids.unfold(1, pattern_size, 1).eq(pattern_ids).all(dim=-1)
        occurrences = occurrences & answer_mask.unfold(1, pattern_size, 1).all(dim=-1)
        occurrence_count = int(occurrences.shape[1])
        selected_logits = logits[
            :, int(target_offset) : int(target_offset) + occurrence_count, :
        ].float()
        log_denominator = torch.logsumexp(selected_logits, dim=-1)
        log_negative = torch.logsumexp(
            selected_logits.index_select(-1, negative_ids), dim=-1
        )
        negative_probability = torch.exp(log_negative - log_denominator).clamp(
            max=1.0 - 1.0e-6
        )
        penalties = -torch.log1p(-negative_probability)
        active = occurrences.to(penalties.dtype)
        total = total + (penalties * active).sum()
        count = count + occurrences.sum()
    return total / count.clamp_min(1).to(total.dtype), count


def role_contrastive_answer_loss(
    model,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    contrastive_weight: torch.Tensor,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
    answer_weight: float,
    branches: Sequence[PreparedRoleBranch],
) -> torch.Tensor:
    """V39 normalized answer loss plus byte-trie role unlikelihood."""

    if contrastive_weight.ndim != 0:
        raise ValueError("contrastive_weight must be a scalar tensor")
    logits = model.forward(input_ids, collect_telemetry=False)["logits"]
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape(target_ids.shape)
    mask = answer_target_mask(input_ids, marker_ids=marker_ids, eos_id=int(eos_id))
    weights = 1.0 + mask.to(token_losses.dtype) * (float(answer_weight) - 1.0)
    causal_loss = (token_losses * weights).sum() / weights.sum()
    contrastive_loss, _ = role_contrastive_unlikelihood(
        logits, target_ids, mask, branches
    )
    return causal_loss + contrastive_weight.to(causal_loss.dtype) * contrastive_loss
