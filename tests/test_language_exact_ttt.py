from __future__ import annotations

import torch

from marulho.training.language_exact_ttt import V76Config, V76ExactTTT, make_v76_batch


def _model() -> V76ExactTTT:
    return V76ExactTTT(
        V76Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    )


def _batch(model: V76ExactTTT, seed: int = 11):
    return make_v76_batch(
        model.config,
        batch_size=4,
        generator=torch.Generator().manual_seed(seed),
        device="cpu",
    )


def test_v76_disabled_fast_path_and_query_targets_are_exact() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    batch = _batch(model)
    fast_a, fast_b = model.initial_fast_weights(4)
    with torch.no_grad():
        enabled = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
        disabled = model.forward_segment(
            batch.tokens[:, 0, :-1], fast_a, fast_b, fast_enabled=False
        )
    actual = batch.tokens[:, 2].index_select(1, batch.query_positions + 1)
    assert torch.equal(enabled, disabled)
    assert torch.equal(actual, batch.query_values)


def test_v76_exact_and_first_order_have_identical_numerical_episode() -> None:
    torch.manual_seed(12)
    model = _model().train()
    batch = _batch(model)
    exact = model.episode(batch, meta_gradient="exact")
    first_order = model.episode(batch, meta_gradient="first_order")
    for name in ("loss", "query_logits", "update_norms", "final_fast_a", "final_fast_b"):
        assert torch.equal(exact[name].detach(), first_order[name].detach())


def test_v76_future_perturbation_cannot_change_earlier_update() -> None:
    torch.manual_seed(13)
    model = _model().train()
    batch = _batch(model, seed=21)
    changed = _batch(model, seed=22)
    changed.tokens[:, :2] = batch.tokens[:, :2]
    first = model.episode(batch, meta_gradient="exact")
    second = model.episode(changed, meta_gradient="exact")
    assert torch.equal(first["segment_losses"][:2], second["segment_losses"][:2])
    assert torch.equal(first["update_norms"], second["update_norms"])


def test_v76_episode_resets_fast_state_exactly() -> None:
    torch.manual_seed(17)
    model = _model().train()
    batch = _batch(model)
    first = model.episode(batch, meta_gradient="exact")
    second = model.episode(batch, meta_gradient="exact")
    for name in ("query_logits", "final_fast_a", "final_fast_b"):
        assert torch.equal(first[name].detach(), second[name].detach())


def test_v76_exact_outer_gradient_differs_from_first_order() -> None:
    torch.manual_seed(18)
    model = _model().train()
    batch = _batch(model)
    exact = model.episode(batch, meta_gradient="exact")
    exact["loss"].backward()
    exact_gradient = model.fast_a0.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    first_order = model.episode(batch, meta_gradient="first_order")
    first_order["loss"].backward()
    first_gradient = model.fast_a0.grad.detach().clone()
    assert torch.isfinite(exact_gradient).all()
    assert torch.isfinite(first_gradient).all()
    assert not torch.equal(exact_gradient, first_gradient)


def test_v76_two_exact_updates_reach_all_meta_parameters() -> None:
    torch.manual_seed(19)
    model = _model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    seen: set[str] = set()
    for step in range(2):
        batch = _batch(model, seed=100 + step)
        optimizer.zero_grad(set_to_none=True)
        result = model.episode(batch, meta_gradient="exact")
        result["loss"].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and torch.count_nonzero(parameter.grad):
                assert torch.isfinite(parameter.grad).all()
                seen.add(name)
        optimizer.step()
    assert seen == set(dict(model.named_parameters()))
