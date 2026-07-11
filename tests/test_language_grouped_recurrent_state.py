from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_grouped_recurrent_state import (
    GroupedRecurrentConfig,
    MarulhoGroupedRecurrentLanguageModel,
    build_grouped_recurrent_model,
)


def _base(*, mode: str = "token_hash"):
    return MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=96,
            width=32,
            layers=3,
            attention_heads=4,
            context_length=32,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=64,
            routing_heads=2,
            experts_per_head=2,
            mode=mode,
        )
    )


def _config(architecture: str, mode: str = "recurrent", **overrides):
    values = {
        "architecture": architecture,
        "mode": mode,
        "memory_layer_index": 0,
        "group_count": 4,
        "group_width": 4,
    }
    values.update(overrides)
    return GroupedRecurrentConfig(**values)


def _model(architecture: str, mode: str = "recurrent"):
    return build_grouped_recurrent_model(_base(), _config(architecture, mode))


@pytest.mark.parametrize("architecture", ("grouped", "dense"))
@pytest.mark.parametrize("mode", ("off", "local", "recurrent"))
def test_recurrent_attachment_is_exact_parent(
    architecture: str,
    mode: str,
) -> None:
    torch.manual_seed(181)
    base = _base().eval()
    input_ids = torch.randint(0, 96, (2, 16))
    expected = base(input_ids, collect_telemetry=False)["logits"]
    model = build_grouped_recurrent_model(
        base, _config(architecture, mode)
    ).eval()
    actual = model(input_ids, collect_telemetry=False)["logits"]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("architecture", ("grouped", "dense"))
@pytest.mark.parametrize("mode", ("off", "local", "recurrent"))
def test_recurrent_state_is_causal_and_streaming_equivalent(
    architecture: str,
    mode: str,
) -> None:
    torch.manual_seed(191)
    model = _model(architecture, mode).eval()
    with torch.no_grad():
        model.state_block.recurrent.output.weight.normal_(0.0, 0.02)
    first = torch.randint(0, 96, (2, 16))
    second = first.clone()
    second[:, 9:] = torch.randint(0, 96, second[:, 9:].shape)
    with torch.no_grad():
        first_logits = model(first, collect_telemetry=False)["logits"]
        second_logits = model(second, collect_telemetry=False)["logits"]
        prompt = model(first[:, :3], collect_telemetry=False)
        state = prompt["state"]
        incremental = [prompt["logits"][:, -1]]
        for index in range(3, int(first.shape[1])):
            step = model.forward_step(
                first[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            incremental.append(step["logits"][:, -1])
    torch.testing.assert_close(first_logits[:, :9], second_logits[:, :9])
    torch.testing.assert_close(
        torch.stack(incremental, dim=1),
        first_logits[:, 2:],
        atol=3e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    ("architecture", "mode"),
    (("grouped", "local"), ("grouped", "recurrent"), ("dense", "recurrent")),
)
def test_recurrent_training_reaches_every_organ_parameter(
    architecture: str,
    mode: str,
) -> None:
    torch.manual_seed(193)
    model = _model(architecture, mode).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    input_ids = torch.randint(0, 96, (3, 16))
    targets = torch.randint(0, 96, (3, 16))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, collect_telemetry=False)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, 96), targets.reshape(-1))
        loss.backward()
        optimizer.step()
    report = model.final_recurrent_gradient_report()
    assert report["all_parameters_received_gradient"] is True
    for row in report["parameters"]:
        if mode == "local" and "weight_hh" in row["name"]:
            assert row["nonzero_gradient_elements"] == 0
        else:
            assert row["nonzero_gradient_elements"] > 0


def test_grouped_uses_fewer_parameters_than_dense_at_same_state_width() -> None:
    grouped = _model("grouped")
    dense = _model("dense")
    grouped_report = grouped.recurrent_parameter_report()
    dense_report = dense.recurrent_parameter_report()
    assert grouped_report["total_state_width"] == dense_report["total_state_width"]
    assert grouped_report["recurrent_organ_parameters"] < dense_report[
        "recurrent_organ_parameters"
    ]
    assert grouped.state_block.recurrent.state_bytes(7) == (
        dense.state_block.recurrent.state_bytes(7)
    )


def test_recurrent_diagnostic_is_label_safe() -> None:
    model = _model("grouped").eval()
    report = model.recurrent_diagnostic_report(torch.randint(0, 96, (2, 16)))
    assert report["state"]["write_policy_uses_labels"] is False
    assert report["state"]["state_geometry"]["ambient_dimension"] == 16
    assert len(report["state"]["group_residual_root_mean_squares"]) == 4


@pytest.mark.parametrize(
    ("configuration", "match"),
    [
        (_config("slots"), "architecture"),
        (_config("grouped", mode="chaos"), "mode"),
        (_config("grouped", memory_layer_index=2), "precede"),
        (_config("grouped", group_count=1), "group_count"),
        (_config("grouped", group_width=0), "group_width"),
    ],
)
def test_recurrent_rejects_invalid_config(
    configuration: GroupedRecurrentConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_grouped_recurrent_model(_base(), configuration)


def test_recurrent_requires_token_hash_base() -> None:
    with pytest.raises(ValueError, match="token_hash"):
        build_grouped_recurrent_model(_base(mode="shared_only"), _config("grouped"))
