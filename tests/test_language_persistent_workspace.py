from __future__ import annotations

import torch

from marulho.training.language_persistent_workspace import (
    V72PersistentWorkspaceRecall,
    V72PersistentWorkspaceLanguageModel,
    V72RecallConfig,
    make_v72_recall_batch,
    transfer_v72_transformer_common_state,
    v72_recall_loss,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _batch(model: V72PersistentWorkspaceRecall, seed: int = 5):
    generator = torch.Generator().manual_seed(seed)
    return make_v72_recall_batch(
        model.config,
        batch_size=4,
        generator=generator,
        device="cpu",
    )


def test_v72_batch_hides_answer_after_first_segment() -> None:
    config = V72RecallConfig()
    model = V72PersistentWorkspaceRecall(config)
    batch = _batch(model)
    later = batch.segments[:, 1:]
    assert not torch.isin(
        later,
        torch.arange(config.value_start, config.vocab_size),
    ).any()
    assert torch.equal(
        batch.segments[:, 0, 0],
        torch.full((4,), config.write_token),
    )


def test_v72_future_perturbation_cannot_change_earlier_outputs() -> None:
    torch.manual_seed(7)
    model = V72PersistentWorkspaceRecall().eval()
    batch = _batch(model)
    changed = batch.segments.clone()
    changed[:, -1, 10:40] = changed[:, -1, 10:40].roll(1, dims=0)
    with torch.no_grad():
        original = model(batch.segments, mode="persistent")
        perturbed = model(changed, mode="persistent")
    assert torch.equal(original["segment_logits"][:, :2], perturbed["segment_logits"][:, :2])
    assert torch.equal(original["states"][:, :2], perturbed["states"][:, :2])


def test_v72_controls_only_change_boundary_state_identity() -> None:
    torch.manual_seed(11)
    model = V72PersistentWorkspaceRecall()
    state = torch.randn(4, model.config.workspace_tokens, model.config.width)
    assert torch.equal(model.boundary_state(state, "persistent"), state)
    assert torch.equal(
        model.boundary_state(state, "reset_each_segment"),
        model.initial_state(4),
    )
    assert torch.equal(
        model.boundary_state(state, "shuffled_document_state"),
        state.roll(1, 0),
    )
    expected_mean = state.mean(0, keepdim=True).expand_as(state)
    assert torch.equal(
        model.boundary_state(state, "nonpersistent_same_compute"),
        expected_mean,
    )


def test_v72_loss_is_finite_and_reaches_every_parameter() -> None:
    torch.manual_seed(13)
    model = V72PersistentWorkspaceRecall().train()
    batch = _batch(model)
    result = model(batch.segments, mode="persistent")
    loss, components = v72_recall_loss(result, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []
    assert nonfinite == []


def _language_config(*, workspace: bool) -> LanguageModelConfig:
    width = 32
    hidden = 114 if workspace else 128
    return LanguageModelConfig(
        vocab_size=96,
        embedding_dim=width,
        state_dim=width,
        state_layers=10,
        attention_heads=4,
        transformer_context_length=16,
        transformer_mlp_ratio=hidden / width,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="v72_test",
    )


def test_v72_language_candidate_is_causal_and_parameter_matched() -> None:
    torch.manual_seed(19)
    control = MarulhoLanguageModel(_language_config(workspace=False))
    torch.manual_seed(19)
    candidate = V72PersistentWorkspaceLanguageModel(
        _language_config(workspace=True), workspace_tokens=4
    ).eval()
    transfer = transfer_v72_transformer_common_state(control, candidate)
    ratio = sum(p.numel() for p in candidate.parameters()) / sum(
        p.numel() for p in control.parameters()
    )
    assert 0.99 <= ratio <= 1.01
    assert transfer["copied_parameter_count"] > 0
    first = torch.randint(0, 96, (3, 16))
    second = first.clone()
    second[:, 9:] = torch.randint(0, 96, (3, 7))
    workspace = candidate.initial_workspace(3)
    with torch.no_grad():
        first_result = candidate.forward_segment(first, workspace)
        second_result = candidate.forward_segment(second, workspace)
    assert torch.equal(first_result["logits"][:, :9], second_result["logits"][:, :9])
    assert not torch.equal(first_result["next_workspace"], workspace)


def test_v72_language_local_reconstruction_reaches_workspace_writer() -> None:
    torch.manual_seed(23)
    candidate = V72PersistentWorkspaceLanguageModel(
        _language_config(workspace=True), workspace_tokens=4
    ).train()
    inputs = torch.randint(0, 96, (2, 16))
    targets = torch.randint(0, 96, (2, 16))
    result = candidate.forward_segment(inputs, candidate.initial_workspace(2))
    landmark_targets = inputs[:, torch.tensor([3, 7, 11, 15])]
    loss = torch.nn.functional.cross_entropy(
        result["logits"].reshape(-1, 96), targets.reshape(-1)
    ) + 0.1 * torch.nn.functional.cross_entropy(
        result["workspace_logits"].reshape(-1, 96), landmark_targets.reshape(-1)
    )
    loss.backward()
    assert candidate.workspace_write.in_proj_weight.grad is not None
    assert torch.count_nonzero(candidate.workspace_write.in_proj_weight.grad)
    assert candidate.workspace_gate.weight.grad is not None
    assert torch.count_nonzero(candidate.workspace_gate.weight.grad)
