from __future__ import annotations

import torch

from marulho.training.language_depth_assembly import install_depth_assembly
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _model() -> MarulhoLanguageModel:
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=96,
            embedding_dim=32,
            state_dim=32,
            state_layers=4,
            attention_heads=4,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
            transformer_dropout=0.0,
        )
    )


def test_depth_assembly_is_exactly_identity_initialized() -> None:
    torch.manual_seed(37)
    baseline = _model().eval()
    candidate = _model().eval()
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    block = install_depth_assembly(candidate)
    token_ids = torch.tensor([[1, 3, 5, 7, 9, 11]], dtype=torch.long)

    with torch.no_grad():
        expected = baseline(token_ids, collect_telemetry=False)
        actual = candidate(token_ids, collect_telemetry=False)

    assert torch.equal(actual["logits"], expected["logits"])
    assert tuple(actual["state"]) == tuple(expected["state"])
    for name in actual["state"]:
        assert torch.equal(actual["state"][name], expected["state"][name])
    assert block.depth_routes.shape == (6,)
    assert actual["telemetry"]["state_core"] == "depth_assembly_transformer"


def test_depth_assembly_cached_steps_match_full_forward_at_identity() -> None:
    torch.manual_seed(41)
    model = _model().eval()
    install_depth_assembly(model)
    token_ids = torch.tensor([[1, 2, 4, 8, 16, 32, 7]], dtype=torch.long)

    with torch.no_grad():
        full = model(token_ids, collect_telemetry=False)["logits"]
        prompt = model(token_ids[:, :2], collect_telemetry=False)
        state = prompt["state"]
        pieces = [prompt["logits"][:, -1]]
        for index in range(2, int(token_ids.shape[1])):
            step = model.forward_step(
                token_ids[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            pieces.append(step["logits"][:, -1])

    assert torch.allclose(torch.stack(pieces, dim=1), full[:, 1:], atol=2e-5)


def test_every_depth_route_receives_gradient_and_changes_behavior() -> None:
    torch.manual_seed(43)
    model = _model().train()
    block = install_depth_assembly(model)
    token_ids = torch.randint(0, 96, (3, 12))
    targets = torch.randint(0, 96, (3, 12))
    loss = model.next_token_loss(token_ids, targets)["loss"]
    loss.backward()

    assert block.depth_routes.grad is not None
    assert torch.isfinite(block.depth_routes.grad).all()
    assert torch.count_nonzero(block.depth_routes.grad) == block.depth_routes.numel()
    with torch.no_grad():
        before = model(token_ids, collect_telemetry=False)["logits"].clone()
        block.depth_routes.fill_(0.25)
        after = model(token_ids, collect_telemetry=False)["logits"]
    assert not torch.equal(before, after)
    report = block.route_report()
    assert report["nonzero_parameter_count"] == 6
    assert report["external_llm_used"] is False
