from __future__ import annotations

import torch

from marulho.evaluation.language_dyadic_memory_preflight import (
    ADMIT_DECISION,
    ARM_NAMES,
    DYADIC_MODES,
    DyadicRecallModel,
    FlatGatedRecallModel,
    RecallTaskConfig,
    _masked_loss,
    dyadic_preflight_decision,
    generate_recall_batch,
    oracle_recall_accuracy,
)


def _task() -> RecallTaskConfig:
    return RecallTaskConfig(
        key_count=8,
        value_count=8,
        query_count=4,
        train_sequence_length=128,
        embedding_dim=8,
        dyadic_levels=4,
        bank_width=4,
    )


def test_recall_profiles_are_oracle_valid_and_query_label_safe() -> None:
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
            seed=101,
        )
        selected = batch.targets >= 0
        assert oracle_recall_accuracy(batch, task) == 1.0
        assert torch.all(batch.roles[selected] == 2)
        assert torch.all(batch.values[selected] == task.value_count)
        assert torch.all(batch.delays[selected] > 0)
        assert int(selected.sum()) == 8 * task.query_count


def test_dyadic_modes_are_causal_and_parameter_matched() -> None:
    torch.manual_seed(103)
    task = _task()
    model = DyadicRecallModel(task).eval()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    batch = generate_recall_batch(
        task,
        profile="heldout_128",
        batch_size=2,
        seed=107,
    )
    changed_roles = batch.roles.clone()
    changed_keys = batch.keys.clone()
    changed_values = batch.values.clone()
    changed_roles[:, 80:] = 3
    changed_keys[:, 80:] = 0
    changed_values[:, 80:] = 0
    parameter_counts = []
    for mode in DYADIC_MODES:
        active = DyadicRecallModel(task, mode=mode).eval()
        active.load_state_dict(initial, strict=True)
        active.set_mode(mode)
        first = active(batch.roles, batch.keys, batch.values)
        second = active(changed_roles, changed_keys, changed_values)
        torch.testing.assert_close(first["logits"][:, :80], second["logits"][:, :80])
        torch.testing.assert_close(first["state"][:, :80], second["state"][:, :80])
        parameter_counts.append(sum(value.numel() for value in active.parameters()))
    assert len(set(parameter_counts)) == 1


def test_flat_control_has_same_state_bytes_and_more_parameters() -> None:
    task = _task()
    flat = FlatGatedRecallModel(task)
    dyadic = DyadicRecallModel(task)
    assert flat.state_bytes(7) == dyadic.state_bytes(7)
    assert flat.recurrent_updates_per_sequence(128) == 128
    assert dyadic.recurrent_updates_per_sequence(128) == 120
    assert sum(value.numel() for value in flat.parameters()) > sum(
        value.numel() for value in dyadic.parameters()
    )


def test_every_dyadic_parameter_receives_query_loss_gradient() -> None:
    torch.manual_seed(109)
    task = _task()
    batch = generate_recall_batch(
        task,
        profile="train_128",
        batch_size=4,
        seed=113,
    )
    for mode in DYADIC_MODES:
        model = DyadicRecallModel(task, mode=mode)
        output = model(batch.roles, batch.keys, batch.values)
        _masked_loss(output["logits"], batch.targets).backward()
        assert all(parameter.grad is not None for parameter in model.parameters())


def _replicate(
    accuracies: dict[str, dict[str, float]],
) -> dict:
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


def test_dyadic_decision_requires_long_profile_gain_and_guards() -> None:
    profiles = {
        arm: {
            "heldout_128": 0.80,
            "long_256": 0.70,
            "long_512": 0.60,
            "overwrite_256": 0.70,
        }
        for arm in ARM_NAMES
    }
    profiles["haar"] = {
        "heldout_128": 0.80,
        "long_256": 0.75,
        "long_512": 0.65,
        "overwrite_256": 0.70,
    }
    replicates = [_replicate(profiles) for _ in range(3)]
    assert dyadic_preflight_decision(
        replicates,
        requested_replicates=3,
    ) == ADMIT_DECISION
    profiles["haar"]["long_512"] = 0.61
    assert dyadic_preflight_decision(
        [_replicate(profiles) for _ in range(3)],
        requested_replicates=3,
    ) == "retire_v15_dyadic_preflight_no_gated_recurrence_gain"


def test_dyadic_decision_rejects_incomplete_evidence() -> None:
    assert dyadic_preflight_decision([], requested_replicates=3).startswith(
        "incomplete"
    )
