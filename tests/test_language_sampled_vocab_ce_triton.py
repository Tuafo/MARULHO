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


def test_language_sampled_vocab_ce_reference_can_skip_builder_contract_validation() -> None:
    inputs = _inputs(token_count=18, vocab_size=256, sampled_vocab_size=48)

    validated = language_sampled_vocab_cross_entropy_torch_reference(**inputs)
    hot_contract = language_sampled_vocab_cross_entropy_torch_reference(
        **inputs,
        validate_targets=False,
    )

    torch.testing.assert_close(hot_contract, validated)


def test_language_sampled_vocab_ce_reference_can_emit_sparse_weight_gradients() -> None:
    inputs = _inputs(token_count=18, vocab_size=256, sampled_vocab_size=48)
    inputs["hidden"].requires_grad_(True)
    inputs["lm_head_weight"] = torch.nn.Parameter(inputs["lm_head_weight"])
    inputs["lm_head_bias"] = torch.nn.Parameter(inputs["lm_head_bias"])

    loss = language_sampled_vocab_cross_entropy_torch_reference(
        **inputs,
        sparse_weight_gradient=True,
    )
    loss.backward()

    assert inputs["lm_head_weight"].grad is not None
    assert inputs["lm_head_weight"].grad.is_sparse
    rows = inputs["lm_head_weight"].grad.coalesce().indices()[0]
    assert int(rows.unique().numel()) <= int(inputs["sampled_vocab_ids"].numel())
    assert inputs["lm_head_bias"].grad is not None
    assert not inputs["lm_head_bias"].grad.is_sparse


def test_language_sampled_vocab_ce_rejects_missing_targets() -> None:
    inputs = _inputs()
    inputs["sampled_vocab_ids"] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    inputs["target_ids"] = torch.tensor([4] * int(inputs["hidden"].shape[0]))

    with pytest.raises(ValueError, match="must include every target id"):
        language_sampled_vocab_cross_entropy(**inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_build_sampled_vocab_ids_can_stay_on_cuda() -> None:
    target_ids = torch.tensor([9, 7, 9, 31, 2], dtype=torch.long, device="cuda")

    sampled = build_sampled_vocab_ids(
        target_ids,
        vocab_size=4096,
        sample_count=128,
        device="cuda",
        validate_ids=False,
    )

    assert sampled.device.type == "cuda"
    assert sampled.numel() == 128
    assert set(target_ids.detach().cpu().tolist()).issubset(
        set(sampled.detach().cpu().tolist())
    )


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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_sampled_vocab_ce_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for sampled-vocab CE autograd parity",
)
def test_language_sampled_vocab_ce_triton_autograd_matches_reference() -> None:
    inputs = _inputs(
        device="cuda",
        token_count=128,
        state_dim=32,
        vocab_size=1024,
        sampled_vocab_size=256,
    )
    triton_hidden = inputs["hidden"].detach().clone().requires_grad_(True)
    reference_hidden = inputs["hidden"].detach().clone().requires_grad_(True)
    triton_weight = torch.nn.Parameter(inputs["lm_head_weight"].detach().clone())
    reference_weight = torch.nn.Parameter(inputs["lm_head_weight"].detach().clone())
    triton_bias = torch.nn.Parameter(inputs["lm_head_bias"].detach().clone())
    reference_bias = torch.nn.Parameter(inputs["lm_head_bias"].detach().clone())

    before = language_sampled_vocab_ce_triton_stats()
    triton_loss = language_sampled_vocab_cross_entropy(
        triton_hidden,
        inputs["target_ids"],
        inputs["sampled_vocab_ids"],
        triton_weight,
        triton_bias,
        force_triton=True,
        sparse_weight_gradient=True,
    )
    reference_loss = language_sampled_vocab_cross_entropy_torch_reference(
        reference_hidden,
        inputs["target_ids"],
        inputs["sampled_vocab_ids"],
        reference_weight,
        reference_bias,
        sparse_weight_gradient=True,
    )
    triton_loss.backward()
    reference_loss.backward()
    torch.cuda.synchronize()
    delta = language_sampled_vocab_ce_triton_stats_delta(
        before,
        language_sampled_vocab_ce_triton_stats(),
    )

    torch.testing.assert_close(triton_loss, reference_loss, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        triton_hidden.grad,
        reference_hidden.grad,
        rtol=2e-3,
        atol=2e-3,
    )
    assert triton_weight.grad is not None
    assert triton_weight.grad.is_sparse
    assert reference_weight.grad is not None
    assert reference_weight.grad.is_sparse
    torch.testing.assert_close(
        triton_weight.grad.coalesce().to_dense(),
        reference_weight.grad.coalesce().to_dense(),
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(triton_bias.grad, reference_bias.grad, rtol=2e-3, atol=2e-3)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_autograd_forward_calls"] >= 1
    assert delta["torch_autograd_backward_calls"] >= 1
    assert delta["sparse_weight_backward_calls"] >= 1


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bool(language_sampled_vocab_ce_triton_stats()["triton_available"]),
    reason="CUDA and Triton are required for sampled-vocab CE training policy",
)
def test_language_sampled_vocab_ce_training_defaults_to_measured_faster_torch_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING", raising=False)
    inputs = _inputs(
        device="cuda",
        token_count=128,
        state_dim=32,
        vocab_size=1024,
        sampled_vocab_size=256,
    )
    hidden = inputs["hidden"].detach().clone().requires_grad_(True)
    weight = torch.nn.Parameter(inputs["lm_head_weight"].detach().clone())
    bias = torch.nn.Parameter(inputs["lm_head_bias"].detach().clone())

    before = language_sampled_vocab_ce_triton_stats()
    loss = language_sampled_vocab_cross_entropy(
        hidden,
        inputs["target_ids"],
        inputs["sampled_vocab_ids"],
        weight,
        bias,
        sparse_weight_gradient=True,
    )
    loss.backward()
    torch.cuda.synchronize()
    delta = language_sampled_vocab_ce_triton_stats_delta(
        before,
        language_sampled_vocab_ce_triton_stats(),
    )

    assert loss.detach().item() > 0.0
    assert weight.grad is not None
    assert weight.grad.is_sparse
    assert delta["triton_kernel_used"] is False
    assert delta["torch_fallback_calls"] >= 1
    assert delta["triton_autograd_forward_calls"] == 0
