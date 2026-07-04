from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from marulho.core.language_sampled_vocab_ce_triton import (
    build_sampled_vocab_ids,
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
    language_sampled_vocab_cross_entropy,
    language_sampled_vocab_cross_entropy_torch_reference,
)


def _inputs(
    *,
    device: str = "cpu",
    token_count: int = 12,
    state_dim: int = 16,
    vocab_size: int = 128,
    sampled_vocab_size: int = 32,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260704)
    dtype = torch.float32
    hidden = torch.randn(token_count, state_dim, device=device, dtype=dtype)
    lm_head_weight = torch.randn(vocab_size, state_dim, device=device, dtype=dtype)
    lm_head_bias = torch.randn(vocab_size, device=device, dtype=dtype)
    seed_targets = torch.arange(
        min(token_count, sampled_vocab_size),
        device=device,
        dtype=torch.long,
    )
    sampled_vocab_ids = build_sampled_vocab_ids(
        seed_targets,
        vocab_size=vocab_size,
        sample_count=sampled_vocab_size,
        device=device,
    )
    target_positions = torch.randint(
        0,
        sampled_vocab_size,
        (token_count,),
        device=device,
        dtype=torch.long,
    )
    target_ids = sampled_vocab_ids.index_select(0, target_positions)
    return {
        "hidden": hidden,
        "target_ids": target_ids,
        "sampled_vocab_ids": sampled_vocab_ids,
        "lm_head_weight": lm_head_weight,
        "lm_head_bias": lm_head_bias,
    }


def test_build_sampled_vocab_ids_includes_targets() -> None:
    target_ids = torch.tensor([9, 7, 9, 31, 2], dtype=torch.long)

    sampled = build_sampled_vocab_ids(
        target_ids,
        vocab_size=64,
        sample_count=12,
        device="cpu",
    )

    assert sampled.numel() == 12
    assert set(target_ids.tolist()).issubset(set(sampled.tolist()))
    assert len(set(sampled.tolist())) == int(sampled.numel())


def test_language_sampled_vocab_ce_cpu_matches_dense_subset_reference() -> None:
    inputs = _inputs()

    output = language_sampled_vocab_cross_entropy(**inputs)
    reference = language_sampled_vocab_cross_entropy_torch_reference(**inputs)
    sampled_weight = inputs["lm_head_weight"].index_select(0, inputs["sampled_vocab_ids"])
    sampled_bias = inputs["lm_head_bias"].index_select(0, inputs["sampled_vocab_ids"])
    sampled_logits = inputs["hidden"].matmul(sampled_weight.t()) + sampled_bias
    target_positions = inputs["target_ids"][:, None].eq(
        inputs["sampled_vocab_ids"][None, :]
    ).to(torch.long).argmax(dim=1)
    dense_subset_reference = F.cross_entropy(sampled_logits, target_positions)

    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(output, dense_subset_reference)


def test_language_sampled_vocab_ce_rejects_missing_targets() -> None:
    inputs = _inputs()
    inputs["sampled_vocab_ids"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["target_ids"] = torch.tensor([4] * int(inputs["hidden"].shape[0]))

    with pytest.raises(ValueError, match="must include every target id"):
        language_sampled_vocab_cross_entropy(**inputs)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_sampled_vocab_ce_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for sampled-vocab CE kernel parity",
)
def test_language_sampled_vocab_ce_triton_matches_reference() -> None:
    inputs = _inputs(
        device="cuda",
        token_count=128,
        state_dim=32,
        vocab_size=1024,
        sampled_vocab_size=256,
    )

    before = language_sampled_vocab_ce_triton_stats()
    output = language_sampled_vocab_cross_entropy(**inputs, force_triton=True)
    reference = language_sampled_vocab_cross_entropy_torch_reference(**inputs)
    torch.cuda.synchronize()
    delta = language_sampled_vocab_ce_triton_stats_delta(
        before,
        language_sampled_vocab_ce_triton_stats(),
    )

    torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] >= 1
    assert delta["triton_logits_calls"] >= 1
    assert delta["triton_loss_calls"] >= 1
