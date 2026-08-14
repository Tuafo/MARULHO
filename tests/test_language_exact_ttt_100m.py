from __future__ import annotations

import torch

from marulho.training.language_exact_ttt_100m import V76ExactTTTLanguage
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _model() -> V76ExactTTTLanguage:
    base = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=80,
            embedding_dim=32,
            state_dim=32,
            state_layers=4,
            attention_heads=4,
            transformer_context_length=64,
            transformer_mlp_ratio=2.0,
            transformer_dropout=0.0,
            tie_embeddings=True,
        )
    )
    return V76ExactTTTLanguage(base, rank=4, fast_layer_indices=(3,))


def _documents(seed: int = 11) -> torch.Tensor:
    return torch.randint(0, 80, (3, 193), generator=torch.Generator().manual_seed(seed))


def test_v76_100m_disabled_fast_path_is_exact() -> None:
    torch.manual_seed(7)
    model = _model().eval()
    documents = _documents()
    fast_a, fast_b = model.initial_fast_weights(3)
    with torch.no_grad():
        enabled = model.forward_segment(documents[:, :64], fast_a, fast_b)
        disabled = model.forward_segment(
            documents[:, :64], fast_a, fast_b, fast_enabled=False
        )
    assert torch.equal(enabled, disabled)


def test_v76_100m_exact_and_first_order_are_numerically_identical() -> None:
    torch.manual_seed(12)
    model = _model().train()
    documents = _documents()
    exact = model.episode_documents(
        documents, meta_gradient="exact", segment_length=64
    )
    first = model.episode_documents(
        documents, meta_gradient="first_order", segment_length=64
    )
    for name in ("loss", "segment_losses", "update_norms"):
        assert torch.equal(exact[name].detach(), first[name].detach())


def test_v76_100m_future_tokens_cannot_change_earlier_updates() -> None:
    torch.manual_seed(13)
    model = _model().train()
    documents = _documents(seed=21)
    changed = _documents(seed=22)
    changed[:, :129] = documents[:, :129]
    first = model.episode_documents(
        documents, meta_gradient="exact", segment_length=64
    )
    second = model.episode_documents(
        changed, meta_gradient="exact", segment_length=64
    )
    assert torch.equal(first["segment_losses"][:2], second["segment_losses"][:2])
    assert torch.equal(first["update_norms"], second["update_norms"])


def test_v76_100m_exact_outer_gradient_differs_from_first_order() -> None:
    torch.manual_seed(17)
    model = _model().train()
    documents = _documents()
    exact = model.episode_documents(
        documents, meta_gradient="exact", segment_length=64
    )
    exact["loss"].backward()
    exact_gradient = model.fast_a0[0].grad.detach().clone()
    model.zero_grad(set_to_none=True)
    first = model.episode_documents(
        documents, meta_gradient="first_order", segment_length=64
    )
    first["loss"].backward()
    first_gradient = model.fast_a0[0].grad.detach().clone()
    assert torch.isfinite(exact_gradient).all()
    assert torch.isfinite(first_gradient).all()
    assert not torch.equal(exact_gradient, first_gradient)


def test_v76_100m_static_path_reuses_exact_base_logits() -> None:
    torch.manual_seed(19)
    model = _model().eval()
    documents = _documents()
    static = model.static_documents(documents, segment_length=64)
    fast_a, fast_b = model.initial_fast_weights(3)
    with torch.no_grad():
        logits = model.forward_segment(
            documents[:, :64], fast_a, fast_b, fast_enabled=False
        )
        expected = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), documents[:, 1:65].flatten()
        )
    assert torch.equal(static["segment_losses"][0], expected)
