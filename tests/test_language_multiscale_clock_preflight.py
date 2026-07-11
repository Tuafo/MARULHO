from __future__ import annotations

import torch

from marulho.evaluation.language_multiscale_clock_preflight import (
    ADMIT_DECISION,
    ARM_NAMES,
    BANK_MODES,
    BankedRecallModel,
    FlatGatedRecallModel,
    RecallTaskConfig,
    _masked_loss,
    generate_recall_batch,
    multiscale_clock_decision,
    oracle_recall_accuracy,
)


def _task() -> RecallTaskConfig:
    return RecallTaskConfig(
        key_count=8,
        value_count=8,
        query_count=4,
        train_sequence_length=128,
        embedding_dim=8,
        bank_count=7,
        bank_width=4,
    )


def test_recall_profiles_are_oracle_valid_and_label_safe() -> None:
    task = _task()
    for profile in (
        "train_128",
        "heldout_128",
        "long_256",
        "long_512",
        "overwrite_256",
    ):
        batch = generate_recall_batch(
            task,
            profile=profile,
            batch_size=8,
            seed=151,
        )
        selected = batch.targets >= 0
        assert oracle_recall_accuracy(batch, task) == 1.0
        assert torch.all(batch.roles[selected] == 2)
        assert torch.all(batch.values[selected] == task.value_count)
        assert torch.all(batch.delays[selected] > 0)


def test_banked_modes_are_causal_and_parameter_matched() -> None:
    torch.manual_seed(157)
    task = _task()
    parent = BankedRecallModel(task, mode="dyadic_lowpass").eval()
    initial = {name: value.clone() for name, value in parent.state_dict().items()}
    batch = generate_recall_batch(
        task,
        profile="heldout_128",
        batch_size=2,
        seed=163,
    )
    changed_roles = batch.roles.clone()
    changed_keys = batch.keys.clone()
    changed_values = batch.values.clone()
    changed_roles[:, 80:] = 3
    changed_keys[:, 80:] = 0
    changed_values[:, 80:] = 0
    counts = []
    for mode in BANK_MODES:
        model = BankedRecallModel(task, mode=mode).eval()
        model.load_state_dict(initial, strict=True)
        model.set_mode(mode)
        first = model(batch.roles, batch.keys, batch.values)
        second = model(changed_roles, changed_keys, changed_values)
        torch.testing.assert_close(first["logits"][:, :80], second["logits"][:, :80])
        torch.testing.assert_close(first["state"][:, :80], second["state"][:, :80])
        counts.append(sum(value.numel() for value in model.parameters()))
    assert len(set(counts)) == 1


def test_controls_match_state_and_declared_update_counts() -> None:
    task = _task()
    flat = FlatGatedRecallModel(task)
    expected = {
        "token_banks": 896,
        "uniform_lowpass": 126,
        "dyadic_last_token": 127,
        "dyadic_lowpass": 127,
    }
    for mode, updates in expected.items():
        model = BankedRecallModel(task, mode=mode)
        assert model.state_bytes(7) == flat.state_bytes(7)
        assert model.recurrent_updates_per_sequence(128) == updates
        assert sum(value.numel() for value in model.parameters()) < sum(
            value.numel() for value in flat.parameters()
        )


def test_every_banked_parameter_receives_query_gradient() -> None:
    torch.manual_seed(167)
    task = _task()
    batch = generate_recall_batch(
        task,
        profile="train_128",
        batch_size=4,
        seed=173,
    )
    for mode in BANK_MODES:
        model = BankedRecallModel(task, mode=mode)
        output = model(batch.roles, batch.keys, batch.values)
        _masked_loss(output["logits"], batch.targets).backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def _replicate(accuracies: dict[str, dict[str, float]]) -> dict:
    return {
        "arms": {
            arm: {
                "profiles": {
                    profile: {"accuracy": value}
                    for profile, value in profiles.items()
                }
            }
            for arm, profiles in accuracies.items()
        }
    }


def test_decision_requires_isolated_dyadic_lowpass_gain() -> None:
    profiles = {
        arm: {
            "heldout_128": 0.70,
            "long_256": 0.60,
            "long_512": 0.50,
            "overwrite_256": 0.60,
        }
        for arm in ARM_NAMES
    }
    profiles["dyadic_lowpass"] = {
        "heldout_128": 0.70,
        "long_256": 0.65,
        "long_512": 0.55,
        "overwrite_256": 0.60,
    }
    assert multiscale_clock_decision(
        [_replicate(profiles) for _ in range(3)],
        requested_replicates=3,
    ) == ADMIT_DECISION
    profiles["uniform_lowpass"]["long_256"] = 0.64
    profiles["uniform_lowpass"]["long_512"] = 0.54
    assert multiscale_clock_decision(
        [_replicate(profiles) for _ in range(3)],
        requested_replicates=3,
    ) == "redesign_v16_multiscale_signal_below_isolation_margin"


def test_decision_retains_strongest_eligible_mechanism() -> None:
    profiles = {
        arm: {
            "heldout_128": 0.20,
            "long_256": 0.10,
            "long_512": 0.10,
            "overwrite_256": 0.15,
        }
        for arm in ARM_NAMES
    }
    profiles["uniform_lowpass"]["long_256"] = 0.18
    profiles["uniform_lowpass"]["long_512"] = 0.17
    profiles["token_banks"]["long_256"] = 0.24
    profiles["token_banks"]["long_512"] = 0.23
    assert multiscale_clock_decision(
        [_replicate(profiles) for _ in range(3)],
        requested_replicates=3,
    ) == "redesign_v16_retain_small_banks_reject_clock_claim"


def test_decision_rejects_incomplete_evidence() -> None:
    assert multiscale_clock_decision([], requested_replicates=3).startswith(
        "incomplete"
    )
