from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


_SUPPORTED_DTYPES = {torch.float32}
_MAX_STATE_DIM = 512
_MAX_SAMPLED_VOCAB = 4096
_DEFAULT_TOKEN_BLOCK = 16
_DEFAULT_VOCAB_BLOCK = 32
_DEFAULT_MIN_TOKENS = 128
_DEFAULT_MIN_SAMPLED_VOCAB = 128


@dataclass
class _SampledVocabCETritonStats:
    triton_forward_calls: int = 0
    triton_logits_calls: int = 0
    triton_loss_calls: int = 0
    triton_forward_tokens: int = 0
    triton_forward_elements: int = 0
    torch_fallback_calls: int = 0
    torch_fallback_elements: int = 0
    triton_failure_count: int = 0
    last_failure: str | None = None
    last_device: str | None = None
    last_dtype: str | None = None


_STATS = _SampledVocabCETritonStats()


def ensure_language_sampled_vocab_ce_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


def reset_language_sampled_vocab_ce_triton_stats() -> None:
    global _STATS
    _STATS = _SampledVocabCETritonStats()


def language_sampled_vocab_ce_triton_stats() -> dict[str, Any]:
    return {
        "surface": "marulho_language_sampled_vocab_ce_triton_stats.v1",
        "triton_available": bool(triton is not None),
        "default_min_tokens": _min_tokens(),
        "default_min_sampled_vocab": _min_sampled_vocab(),
        "triton_forward_calls": int(_STATS.triton_forward_calls),
        "triton_logits_calls": int(_STATS.triton_logits_calls),
        "triton_loss_calls": int(_STATS.triton_loss_calls),
        "triton_forward_tokens": int(_STATS.triton_forward_tokens),
        "triton_forward_elements": int(_STATS.triton_forward_elements),
        "torch_fallback_calls": int(_STATS.torch_fallback_calls),
        "torch_fallback_elements": int(_STATS.torch_fallback_elements),
        "triton_failure_count": int(_STATS.triton_failure_count),
        "last_failure": _STATS.last_failure,
        "last_device": _STATS.last_device,
        "last_dtype": _STATS.last_dtype,
    }


def language_sampled_vocab_ce_triton_stats_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    int_fields = (
        "triton_forward_calls",
        "triton_logits_calls",
        "triton_loss_calls",
        "triton_forward_tokens",
        "triton_forward_elements",
        "torch_fallback_calls",
        "torch_fallback_elements",
        "triton_failure_count",
    )
    delta = {
        key: int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in int_fields
    }
    return {
        "surface": "marulho_language_sampled_vocab_ce_triton_stats_delta.v1",
        "triton_available": bool(after.get("triton_available")),
        **delta,
        "last_failure": after.get("last_failure"),
        "last_device": after.get("last_device"),
        "last_dtype": after.get("last_dtype"),
        "triton_kernel_used": int(delta["triton_forward_calls"]) > 0,
    }


def _validate_sampled_vocab_contract(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
) -> None:
    if hidden.ndim != 2:
        raise ValueError("sampled-vocab CE expects hidden shaped [tokens, state_dim]")
    if target_ids.reshape(-1).numel() != int(hidden.shape[0]):
        raise ValueError("target_ids must contain one token id per hidden row")
    if sampled_vocab_ids.ndim != 1:
        raise ValueError("sampled_vocab_ids must be a 1-D tensor")
    if int(sampled_vocab_ids.numel()) < 2:
        raise ValueError("sampled_vocab_ids must contain at least two vocabulary rows")
    if lm_head_weight.ndim != 2:
        raise ValueError("lm_head_weight must be shaped [vocab_size, state_dim]")
    if lm_head_bias.ndim != 1:
        raise ValueError("lm_head_bias must be shaped [vocab_size]")
    if int(lm_head_weight.shape[1]) != int(hidden.shape[1]):
        raise ValueError("lm_head_weight state dimension does not match hidden")
    if int(lm_head_bias.shape[0]) != int(lm_head_weight.shape[0]):
        raise ValueError("lm_head_bias vocab size does not match lm_head_weight")


def _sampled_target_positions(
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    *,
    validate_targets: bool = True,
) -> torch.Tensor:
    flat_targets = target_ids.reshape(-1).to(
        device=sampled_vocab_ids.device,
        dtype=torch.long,
    )
    sampled = sampled_vocab_ids.reshape(-1).to(dtype=torch.long)
    matches = flat_targets[:, None].eq(sampled[None, :])
    found = matches.any(dim=1)
    if bool(validate_targets) and not bool(found.all().detach().cpu().item()):
        missing = flat_targets[~found].detach().cpu().unique(sorted=True).tolist()
        preview = ", ".join(str(int(value)) for value in missing[:8])
        raise ValueError(
            "sampled_vocab_ids must include every target id; "
            f"missing target ids: {preview}"
        )
    return matches.to(torch.int64).argmax(dim=1)


def build_sampled_vocab_ids(
    target_ids: torch.Tensor,
    *,
    vocab_size: int,
    sample_count: int,
    device: torch.device | str | None = None,
    validate_ids: bool = True,
) -> torch.Tensor:
    vocab = int(vocab_size)
    if vocab <= 1:
        raise ValueError("vocab_size must be greater than one")
    target_device = torch.device(device) if device is not None else target_ids.device
    flat_targets = target_ids.detach().reshape(-1).to(
        device=target_device,
        dtype=torch.long,
    )
    if flat_targets.numel() == 0:
        raise ValueError("target_ids must not be empty")
    if bool(validate_ids) and bool(
        (
            (flat_targets < 0).any() | (flat_targets >= vocab).any()
        ).detach().cpu().item()
    ):
        raise ValueError("target_ids must be inside [0, vocab_size)")
    unique_targets = torch.unique(flat_targets, sorted=True)
    target_count = int(unique_targets.numel())
    requested = max(2, int(sample_count), target_count)
    requested = min(vocab, requested)
    if target_count > requested:
        raise ValueError("sample_count cannot cover the unique target ids")

    if target_count >= requested:
        return unique_targets[:requested].to(device=target_device, dtype=torch.long)

    needed = int(requested - target_count)
    stride = max(1, vocab // requested)
    candidate_count = min(vocab, max(requested * 4, requested + target_count * 2))
    cursor = torch.arange(candidate_count, device=target_device, dtype=torch.long)
    candidates = (
        cursor * int(stride)
        + torch.div(cursor, max(1, requested), rounding_mode="floor")
    ) % int(vocab)
    candidates = torch.unique(candidates, sorted=True)
    extras = candidates[~torch.isin(candidates, unique_targets)]
    if int(extras.numel()) < needed:
        full_candidates = torch.arange(vocab, device=target_device, dtype=torch.long)
        full_extras = full_candidates[~torch.isin(full_candidates, unique_targets)]
        extras = torch.cat((extras, full_extras), dim=0)
        extras = torch.unique(extras, sorted=True)
    selected = torch.cat((unique_targets, extras[:needed]), dim=0)
    return selected[:requested].to(device=target_device, dtype=torch.long)


def language_sampled_vocab_cross_entropy_torch_reference(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
    *,
    validate_targets: bool = True,
    sparse_weight_gradient: bool = False,
) -> torch.Tensor:
    _validate_sampled_vocab_contract(
        hidden,
        target_ids,
        sampled_vocab_ids,
        lm_head_weight,
        lm_head_bias,
    )
    runtime_sampled_ids = sampled_vocab_ids.to(
        device=hidden.device,
        dtype=torch.long,
    )
    runtime_targets = target_ids.to(device=hidden.device, dtype=torch.long).reshape(-1)
    runtime_weight = lm_head_weight.to(device=hidden.device, dtype=hidden.dtype)
    if bool(sparse_weight_gradient):
        sampled_weight = F.embedding(runtime_sampled_ids, runtime_weight, sparse=True)
    else:
        sampled_weight = runtime_weight.index_select(0, runtime_sampled_ids)
    sampled_bias = lm_head_bias.to(device=hidden.device, dtype=hidden.dtype).index_select(
        0,
        runtime_sampled_ids,
    )
    logits = hidden.matmul(sampled_weight.transpose(0, 1)) + sampled_bias
    sampled_targets = _sampled_target_positions(
        runtime_targets,
        runtime_sampled_ids,
        validate_targets=validate_targets,
    )
    return F.cross_entropy(logits.float(), sampled_targets, reduction="mean")


def can_use_language_sampled_vocab_ce_triton(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
) -> bool:
    if triton is None or tl is None:
        return False
    tensors = (hidden, lm_head_weight, lm_head_bias)
    id_tensors = (target_ids, sampled_vocab_ids)
    if not all(isinstance(value, torch.Tensor) for value in (*tensors, *id_tensors)):
        return False
    if any(value.device.type != "cuda" for value in (*tensors, *id_tensors)):
        return False
    if hidden.dtype not in _SUPPORTED_DTYPES:
        return False
    if any(value.dtype != hidden.dtype for value in tensors[1:]):
        return False
    if not all(value.is_floating_point() for value in tensors):
        return False
    if target_ids.dtype != torch.long or sampled_vocab_ids.dtype != torch.long:
        return False
    if hidden.ndim != 2 or target_ids.reshape(-1).numel() != int(hidden.shape[0]):
        return False
    if sampled_vocab_ids.ndim != 1 or int(sampled_vocab_ids.numel()) < 2:
        return False
    token_count, state_dim = int(hidden.shape[0]), int(hidden.shape[1])
    sampled_count = int(sampled_vocab_ids.numel())
    if token_count <= 0 or state_dim <= 0:
        return False
    if state_dim > _MAX_STATE_DIM or sampled_count > _MAX_SAMPLED_VOCAB:
        return False
    return (
        lm_head_weight.ndim == 2
        and lm_head_bias.ndim == 1
        and int(lm_head_weight.shape[1]) == state_dim
        and int(lm_head_bias.shape[0]) == int(lm_head_weight.shape[0])
    )


def _min_tokens() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_MIN_TOKENS")
    if raw is None:
        return _DEFAULT_MIN_TOKENS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MIN_TOKENS


def _min_sampled_vocab() -> int:
    raw = os.environ.get("MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_MIN_VOCAB")
    if raw is None:
        return _DEFAULT_MIN_SAMPLED_VOCAB
    try:
        return max(2, int(raw))
    except ValueError:
        return _DEFAULT_MIN_SAMPLED_VOCAB


def should_use_language_sampled_vocab_ce_triton(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
) -> bool:
    if not can_use_language_sampled_vocab_ce_triton(
        hidden,
        target_ids,
        sampled_vocab_ids,
        lm_head_weight,
        lm_head_bias,
    ):
        return False
    return int(hidden.shape[0]) >= _min_tokens() and int(
        sampled_vocab_ids.numel()
    ) >= _min_sampled_vocab()


def _next_power_of_2(value: int) -> int:
    return 1 << (max(1, int(value)) - 1).bit_length()


if triton is not None:

    @triton.jit
    def _sampled_vocab_logits_kernel(
        hidden,
        lm_head_weight,
        lm_head_bias,
        sampled_vocab_ids,
        sampled_logits,
        token_count: tl.constexpr,
        state_dim: tl.constexpr,
        sampled_vocab_size: tl.constexpr,
        token_block: tl.constexpr,
        vocab_block: tl.constexpr,
        state_block: tl.constexpr,
    ):
        token_block_id = tl.program_id(0)
        vocab_block_id = tl.program_id(1)
        token_offsets = token_block_id * token_block + tl.arange(0, token_block)
        vocab_offsets = vocab_block_id * vocab_block + tl.arange(0, vocab_block)
        state_offsets = tl.arange(0, state_block)
        token_mask = token_offsets < token_count
        vocab_mask = vocab_offsets < sampled_vocab_size
        state_mask = state_offsets < state_dim
        sampled_ids = tl.load(
            sampled_vocab_ids + vocab_offsets,
            mask=vocab_mask,
            other=0,
        )
        hidden_values = tl.load(
            hidden + token_offsets[:, None] * state_dim + state_offsets[None, :],
            mask=token_mask[:, None] & state_mask[None, :],
            other=0.0,
        )
        weight_values = tl.load(
            lm_head_weight + state_offsets[:, None] + sampled_ids[None, :] * state_dim,
            mask=state_mask[:, None] & vocab_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(hidden_values, weight_values, input_precision="ieee")
        bias = tl.load(lm_head_bias + sampled_ids, mask=vocab_mask, other=0.0).to(
            tl.float32
        )
        logits += bias[None, :]
        tl.store(
            sampled_logits
            + token_offsets[:, None] * sampled_vocab_size
            + vocab_offsets[None, :],
            logits,
            mask=token_mask[:, None] & vocab_mask[None, :],
        )

    @triton.jit
    def _sampled_vocab_loss_kernel(
        sampled_logits,
        target_positions,
        token_losses,
        sampled_vocab_size: tl.constexpr,
        vocab_block: tl.constexpr,
    ):
        token = tl.program_id(0)
        offsets = tl.arange(0, vocab_block)
        mask = offsets < sampled_vocab_size
        logits = tl.load(
            sampled_logits + token * sampled_vocab_size + offsets,
            mask=mask,
            other=-3.4028234663852886e38,
        ).to(tl.float32)
        row_max = tl.max(logits, axis=0)
        exp_values = tl.exp(logits - row_max)
        denom = tl.sum(tl.where(mask, exp_values, 0.0), axis=0)
        target_position = tl.load(target_positions + token)
        target_logit = tl.load(
            sampled_logits + token * sampled_vocab_size + target_position
        ).to(tl.float32)
        loss = tl.log(denom) + row_max - target_logit
        tl.store(token_losses + token, loss)


def _language_sampled_vocab_ce_triton_forward(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    ensure_language_sampled_vocab_ce_triton_compiler()
    runtime_hidden = hidden.contiguous()
    runtime_targets = target_ids.to(device=hidden.device, dtype=torch.long).reshape(-1)
    runtime_sampled_ids = sampled_vocab_ids.to(
        device=hidden.device,
        dtype=torch.long,
    ).contiguous()
    runtime_weight = lm_head_weight.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    runtime_bias = lm_head_bias.to(device=hidden.device, dtype=hidden.dtype).contiguous()
    target_positions = _sampled_target_positions(
        runtime_targets,
        runtime_sampled_ids,
    ).contiguous()
    token_count = int(runtime_hidden.shape[0])
    state_dim = int(runtime_hidden.shape[1])
    sampled_vocab_size = int(runtime_sampled_ids.numel())
    sampled_logits = torch.empty(
        (token_count, sampled_vocab_size),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    token_block = min(_DEFAULT_TOKEN_BLOCK, _next_power_of_2(token_count))
    vocab_block = min(_DEFAULT_VOCAB_BLOCK, _next_power_of_2(sampled_vocab_size))
    state_block = _next_power_of_2(state_dim)
    _sampled_vocab_logits_kernel[
        (
            triton.cdiv(token_count, token_block),
            triton.cdiv(sampled_vocab_size, vocab_block),
        )
    ](
        runtime_hidden,
        runtime_weight,
        runtime_bias,
        runtime_sampled_ids,
        sampled_logits,
        token_count,
        state_dim,
        sampled_vocab_size,
        token_block,
        vocab_block,
        state_block,
    )
    loss_block = _next_power_of_2(sampled_vocab_size)
    token_losses = torch.empty(token_count, device=hidden.device, dtype=torch.float32)
    _sampled_vocab_loss_kernel[(token_count,)](
        sampled_logits,
        target_positions,
        token_losses,
        sampled_vocab_size,
        loss_block,
    )
    _STATS.triton_forward_calls += 1
    _STATS.triton_logits_calls += 1
    _STATS.triton_loss_calls += 1
    _STATS.triton_forward_tokens += int(token_count)
    _STATS.triton_forward_elements += int(token_count * sampled_vocab_size)
    _STATS.last_device = str(hidden.device)
    _STATS.last_dtype = str(hidden.dtype)
    return token_losses.mean()


def language_sampled_vocab_cross_entropy(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    sampled_vocab_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    lm_head_bias: torch.Tensor,
    *,
    prefer_triton: bool = True,
    force_triton: bool = False,
) -> torch.Tensor:
    _validate_sampled_vocab_contract(
        hidden,
        target_ids,
        sampled_vocab_ids,
        lm_head_weight,
        lm_head_bias,
    )
    runtime_targets = target_ids.to(device=hidden.device, dtype=torch.long)
    runtime_sampled_ids = sampled_vocab_ids.to(device=hidden.device, dtype=torch.long)
    if bool(prefer_triton) and hidden.device.type == "cuda":
        use_triton = (
            can_use_language_sampled_vocab_ce_triton(
                hidden,
                runtime_targets,
                runtime_sampled_ids,
                lm_head_weight,
                lm_head_bias,
            )
            if bool(force_triton)
            else should_use_language_sampled_vocab_ce_triton(
                hidden,
                runtime_targets,
                runtime_sampled_ids,
                lm_head_weight,
                lm_head_bias,
            )
        )
        if use_triton:
            try:
                return _language_sampled_vocab_ce_triton_forward(
                    hidden,
                    runtime_targets,
                    runtime_sampled_ids,
                    lm_head_weight,
                    lm_head_bias,
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent
                _STATS.triton_failure_count += 1
                _STATS.last_failure = f"{type(exc).__name__}: {exc}"
                _STATS.last_device = str(hidden.device)
                _STATS.last_dtype = str(hidden.dtype)
        _STATS.torch_fallback_calls += 1
        _STATS.torch_fallback_elements += int(hidden.shape[0]) * int(
            runtime_sampled_ids.numel()
        )
        _STATS.last_device = str(hidden.device)
        _STATS.last_dtype = str(hidden.dtype)
    return language_sampled_vocab_cross_entropy_torch_reference(
        hidden,
        runtime_targets,
        runtime_sampled_ids,
        lm_head_weight,
        lm_head_bias,
    )
