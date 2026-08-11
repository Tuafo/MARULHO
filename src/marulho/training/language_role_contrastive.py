"""Tokenizer-trie contrastive objective for marked causal-language answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.training.language_answer_objective import answer_target_mask


@dataclass(frozen=True)
class RoleContrastiveBranch:
    """One wrong next-token branch while spelling a known role filler."""

    group: str
    value: str
    pattern_ids: tuple[int, ...]
    target_offset: int
    negative_ids: tuple[int, ...]


PreparedRoleBranch = tuple[torch.Tensor, int, torch.Tensor]


@dataclass(frozen=True)
class PreparedRoleContrastiveLookup:
    """Vectorized negative-token table for first-subtoken role branches."""

    patterns: tuple[torch.Tensor, ...]
    negative_ids: torch.Tensor
    negative_counts: torch.Tensor


def build_role_contrastive_branches(
    tokenizer,
    groups: Mapping[str, Sequence[str]],
) -> tuple[RoleContrastiveBranch, ...]:
    """Build trie-divergence negatives from the checkpoint tokenizer.

    Each pattern encodes the leading space and value together, establishing a
    word boundary even when BPE merges that space with the first subtoken. At
    each true subtoken, only alternatives reachable after the same true prefix
    are negatives. This avoids assuming either byte or whole-word tokenization.
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
                    f" {value}", add_bos=False, add_eos=False
                )
            )
            for value in values
        }
        for value, pattern in encoded.items():
            for position, correct_id in enumerate(pattern):
                prefix = pattern[:position]
                negatives = sorted(
                    {
                        other_pattern[position]
                        for other, other_pattern in encoded.items()
                        if other != value
                        and len(other_pattern) > position
                        and other_pattern[:position] == prefix
                        and other_pattern[position] != correct_id
                    }
                )
                if negatives:
                    branches.append(
                        RoleContrastiveBranch(
                            group=str(group),
                            value=value,
                            pattern_ids=pattern,
                            target_offset=position,
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


def prepare_role_contrastive_lookup(
    branches: Sequence[RoleContrastiveBranch],
    *,
    vocab_size: int,
    device: torch.device | str,
) -> PreparedRoleContrastiveLookup:
    """Fuse first-subtoken branches into one vocabulary-indexed lookup."""

    rows: dict[int, tuple[int, ...]] = {}
    patterns: list[torch.Tensor] = []
    for branch in branches:
        if int(branch.target_offset) != 0:
            raise ValueError("fast role lookup requires first-subtoken divergence")
        true_id = int(branch.pattern_ids[0])
        negatives = tuple(int(token_id) for token_id in branch.negative_ids)
        if true_id in rows and rows[true_id] != negatives:
            raise ValueError("role lookup true token has conflicting negative sets")
        rows[true_id] = negatives
        patterns.append(
            torch.tensor(branch.pattern_ids, dtype=torch.long, device=device)
        )
    if not rows:
        raise ValueError("role lookup requires at least one branch")
    maximum = max(len(negatives) for negatives in rows.values())
    negative_ids = torch.zeros(
        (int(vocab_size), maximum), dtype=torch.long, device=device
    )
    negative_counts = torch.zeros(
        (int(vocab_size),), dtype=torch.long, device=device
    )
    for true_id, negatives in rows.items():
        if not 0 <= true_id < int(vocab_size):
            raise ValueError("role lookup token is outside the vocabulary")
        negative_counts[true_id] = len(negatives)
        negative_ids[true_id, : len(negatives)] = torch.tensor(
            negatives, dtype=torch.long, device=device
        )
    return PreparedRoleContrastiveLookup(
        patterns=tuple(patterns),
        negative_ids=negative_ids,
        negative_counts=negative_counts,
    )


def role_contrastive_lookup_active_mask(
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    lookup: PreparedRoleContrastiveLookup,
) -> torch.Tensor:
    """Find complete role patterns while marking only their first subtoken."""

    if target_ids.ndim != 2 or answer_mask.shape != target_ids.shape:
        raise ValueError("role lookup mask expects aligned [batch,time] tensors")
    active = torch.zeros_like(target_ids, dtype=torch.bool)
    for pattern_ids in lookup.patterns:
        pattern_size = int(pattern_ids.numel())
        if pattern_size > int(target_ids.shape[1]):
            continue
        occurrences = target_ids.unfold(1, pattern_size, 1).eq(pattern_ids).all(dim=-1)
        occurrences = occurrences & answer_mask.unfold(1, pattern_size, 1).all(dim=-1)
        active = active | F.pad(occurrences, (0, pattern_size - 1))
    return active & lookup.negative_counts[target_ids].gt(0)


def role_contrastive_lookup_unlikelihood(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    answer_mask: torch.Tensor,
    lookup: PreparedRoleContrastiveLookup,
    *,
    log_denominators: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply all role negatives through one gather and one gradient scatter."""

    if logits.ndim != 3 or logits.shape[:2] != target_ids.shape:
        raise ValueError("role lookup loss expects aligned logits and targets")
    if log_denominators.shape != target_ids.shape:
        raise ValueError("role lookup denominators must align with targets")
    active = role_contrastive_lookup_active_mask(target_ids, answer_mask, lookup)
    counts = lookup.negative_counts[target_ids]
    negative_ids = lookup.negative_ids[target_ids]
    selected = logits.gather(-1, negative_ids).float()
    slots = torch.arange(
        int(lookup.negative_ids.shape[1]), device=logits.device
    ).view(1, 1, -1)
    selected = selected.masked_fill(slots >= counts.unsqueeze(-1), float("-inf"))
    log_negative = torch.logsumexp(selected, dim=-1)
    negative_probability = torch.exp(log_negative - log_denominators.float()).clamp(
        max=1.0 - 1.0e-6
    )
    penalties = -torch.log1p(-negative_probability)
    active_float = active.to(penalties.dtype)
    count = active.sum()
    return (penalties * active_float).sum() / count.clamp_min(1).to(
        penalties.dtype
    ), count


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
    *,
    log_denominators: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize probability mass on wrong tokenizer-trie branches.

    Returns the mean active-branch penalty and the active branch count. A batch
    without a recognized answer filler returns differentiable zero.
    """

    if logits.ndim != 3 or target_ids.ndim != 2 or answer_mask.ndim != 2:
        raise ValueError("role contrast expects [batch,time,vocab] logits and ids")
    if logits.shape[:2] != target_ids.shape or target_ids.shape != answer_mask.shape:
        raise ValueError("role contrast logits, targets, and mask must align")
    total = logits.sum() * 0.0
    count = torch.zeros((), dtype=torch.long, device=logits.device)
    if log_denominators is None:
        log_denominators = torch.logsumexp(logits, dim=-1).float()
    elif log_denominators.shape != target_ids.shape:
        raise ValueError("precomputed log denominators must align with targets")
    else:
        log_denominators = log_denominators.float()
    for pattern_ids, target_offset, negative_ids in branches:
        pattern_size = int(pattern_ids.numel())
        if pattern_size > int(target_ids.shape[1]):
            continue
        occurrences = target_ids.unfold(1, pattern_size, 1).eq(pattern_ids).all(dim=-1)
        occurrences = occurrences & answer_mask.unfold(1, pattern_size, 1).all(dim=-1)
        occurrence_count = int(occurrences.shape[1])
        selected_negative_logits = logits[
            :, int(target_offset) : int(target_offset) + occurrence_count, :
        ].index_select(-1, negative_ids).float()
        log_denominator = log_denominators[
            :, int(target_offset) : int(target_offset) + occurrence_count
        ]
        log_negative = torch.logsumexp(
            selected_negative_logits, dim=-1
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
    lookup: PreparedRoleContrastiveLookup,
) -> torch.Tensor:
    """V39 normalized answer loss plus tokenizer-trie role unlikelihood."""

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
    target_logits = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).float()
    log_denominators = token_losses.float() + target_logits
    contrastive_loss, _ = role_contrastive_lookup_unlikelihood(
        logits,
        target_ids,
        mask,
        lookup,
        log_denominators=log_denominators,
    )
    return causal_loss + contrastive_weight.to(causal_loss.dtype) * contrastive_loss
