from __future__ import annotations

import torch

from marulho.training.language_end_to_end_ttt import (
    V74Config,
    V74EndToEndTTT,
    make_v74_batch,
)


def _batch(model: V74EndToEndTTT, seed: int = 11):
    return make_v74_batch(
        model.config,
        batch_size=4,
        generator=torch.Generator().manual_seed(seed),
        device="cpu",
    )


def test_v74_disabled_fast_path_is_exact() -> None:
    torch.manual_seed(7)
    model = V74EndToEndTTT(
        V74Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    ).eval()
    batch = _batch(model)
    fast_a, fast_b = model.initial_fast_weights(4)
    with torch.no_grad():
        enabled = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
        disabled = model.forward_segment(
            batch.tokens[:, 0, :-1], fast_a, fast_b, fast_enabled=False
        )
    assert torch.equal(enabled, disabled)


def test_v74_query_targets_are_exact_next_vocabulary_tokens() -> None:
    model = V74EndToEndTTT(
        V74Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    )
    batch = _batch(model)
    actual = batch.tokens[:, 2].index_select(1, batch.query_positions + 1)
    assert torch.equal(actual, batch.query_values)
    assert bool((batch.query_values >= model.config.value_start).all())


def test_v74_future_perturbation_cannot_change_earlier_logits_or_update() -> None:
    torch.manual_seed(13)
    model = V74EndToEndTTT(
        V74Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    ).train()
    batch = _batch(model)
    changed = _batch(model)
    changed.tokens[:, :2] = batch.tokens[:, :2]
    first = model.episode(batch, mode="persistent_update")
    second = model.episode(changed, mode="persistent_update")
    assert torch.equal(first["segment_losses"][:2], second["segment_losses"][:2])
    assert torch.equal(first["update_norms"][:2], second["update_norms"][:2])


def test_v74_persistent_update_is_document_local() -> None:
    torch.manual_seed(17)
    model = V74EndToEndTTT(
        V74Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    ).train()
    batch = _batch(model)
    fast_a, fast_b = model.initial_fast_weights(4)
    logits = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
    targets = batch.tokens[:, 0, 1:]
    per_document = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), reduction="none"
    ).reshape(4, 64).mean(1)
    grad_a, grad_b = torch.autograd.grad(per_document.sum(), (fast_a, fast_b))
    own_a, own_b = model._advance(
        fast_a, fast_b, grad_a, grad_b, mode="persistent_update"
    )
    wrong_a, wrong_b = model._advance(
        fast_a, fast_b, grad_a, grad_b, mode="shuffled_update"
    )
    assert torch.equal(own_a[0] - fast_a[0], wrong_a[1] - fast_a[1])
    assert torch.equal(own_b[0] - fast_b[0], wrong_b[1] - fast_b[1])


def test_v74_two_updates_reach_all_meta_parameters() -> None:
    torch.manual_seed(19)
    model = V74EndToEndTTT(
        V74Config(width=32, attention_heads=4, mlp_width=64, rank=4)
    ).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    seen: set[str] = set()
    for step in range(2):
        batch = _batch(model, seed=100 + step)
        optimizer.zero_grad(set_to_none=True)
        result = model.episode(batch, mode="persistent_update")
        result["loss"].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and torch.count_nonzero(parameter.grad):
                assert torch.isfinite(parameter.grad).all()
                seen.add(name)
        optimizer.step()
    assert seen == set(dict(model.named_parameters()))
