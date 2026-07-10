from __future__ import annotations

from pathlib import Path

import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_pmrm import (
    PMRM_FUSION_KINDS,
    PMRMLanguageConfig,
    MarulhoPMRMLanguageModel,
    load_pmrm_language_checkpoint,
    save_pmrm_language_checkpoint,
)
from marulho.training.language_model import load_language_model_checkpoint
from marulho.training.language_protocol import CausalLanguageModel


def _config(**overrides) -> PMRMLanguageConfig:
    values = {
        "vocab_size": 64,
        "embedding_dim": 16,
        "state_dim": 16,
        "column_count": 4,
        "active_columns": 2,
        "associative_dim": 4,
        "episodic_slots": 4,
        "episodic_reads": 2,
        "workspace_registers": 2,
        "workspace_layers": 2,
        "workspace_iterations": 2,
        "workspace_mlp_dim": 32,
        "context_length": 8,
    }
    values.update(overrides)
    return PMRMLanguageConfig(**values)


def _assert_state_equal(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)


def test_pmrm_step_scan_and_resume_are_exact() -> None:
    torch.manual_seed(7)
    model = MarulhoPMRMLanguageModel(_config()).eval()
    token_ids = torch.randint(0, 64, (2, 6))

    scan = model.scan(token_ids)
    state = None
    logits = []
    for index in range(token_ids.shape[1]):
        step = model.step(token_ids[:, index], state, collect_telemetry=False)
        logits.append(step["logits"])
        state = step["state"]
    assert state is not None
    torch.testing.assert_close(
        scan["logits"], torch.stack(logits, dim=1), rtol=0.0, atol=0.0
    )
    _assert_state_equal(scan["state"], state)

    serialized = model.serialize_state(state)
    restored = model.load_state(serialized)
    next_ids = torch.tensor([3, 5])
    expected = model.step(next_ids, state, collect_telemetry=False)
    actual = model.step(next_ids, restored, collect_telemetry=False)
    torch.testing.assert_close(expected["logits"], actual["logits"])
    _assert_state_equal(expected["state"], actual["state"])


def test_pmrm_sparse_columns_leave_inactive_state_unchanged() -> None:
    model = MarulhoPMRMLanguageModel(_config()).eval()
    initial = model.init_state(3)
    result = model.step(torch.tensor([1, 2, 3]), initial)
    state = result["state"]
    active = state["column_usage"].to(dtype=torch.bool)

    assert active.sum(dim=1).tolist() == [2, 2, 2]
    assert bool((state["temporal"][~active] == 0).all().item()) is True
    assert bool((state["associative"][~active] == 0).all().item()) is True
    assert result["telemetry"]["dense_router_scoring_reported"] is True
    assert result["telemetry"]["router_score_count"] == 12
    assert result["telemetry"]["active_column_update_count"] == 6


def test_pmrm_episode_is_written_only_after_next_observation() -> None:
    model = MarulhoPMRMLanguageModel(_config(episodic_policy="surprise")).eval()
    first = model.step(torch.tensor([7]), collect_telemetry=False)
    assert int(first["state"]["episodes_valid"].sum().item()) == 0
    assert int(first["state"]["episodic_writes"].item()) == 0

    second = model.step(
        torch.tensor([8]), first["state"], collect_telemetry=False
    )
    assert int(second["state"]["episodes_valid"].sum().item()) == 1
    assert int(second["state"]["episodic_considered"].item()) == 1
    assert int(second["state"]["episodic_writes"].item()) == 1


def test_pmrm_budget_and_compute_ledger_are_explicit() -> None:
    model = MarulhoPMRMLanguageModel(_config()).eval()
    result = model(torch.randint(0, 64, (2, 8)))
    telemetry = result["telemetry"]

    assert int(result["state"]["episodes_valid"].sum().item()) <= 8
    assert telemetry["router_score_count"] == 2 * 8 * 4
    assert telemetry["active_column_update_count"] == 2 * 8 * 2
    assert telemetry["relation_message_count"] == 2 * 8 * 2
    assert telemetry["workspace_update_count"] == 2 * 8 * 2 * 2 * 2
    assert telemetry["column_state_bytes_per_batch"] > 0
    assert telemetry["episodic_state_bytes_per_batch"] > 0
    assert telemetry["total_runtime_state_bytes_per_batch"] > 0
    assert 0.0 <= telemetry["episodic_write_fraction"] <= 1.0


def test_pmrm_backward_is_finite_and_future_tokens_are_causal() -> None:
    torch.manual_seed(11)
    model = MarulhoPMRMLanguageModel(_config())
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 9, 10, 11]])
    first = model(prefix, collect_telemetry=False)["logits"]
    second = model(changed, collect_telemetry=False)["logits"]
    torch.testing.assert_close(first[:, :3], second[:, :3], rtol=0.0, atol=0.0)

    targets = torch.tensor([[2, 3, 4, 5, 6, 7]])
    loss = model.next_token_loss(
        prefix, targets, collect_telemetry=False
    )["loss"]
    loss.backward()
    assert torch.isfinite(loss).item()
    for parameter in (
        model.token_embedding.weight,
        model.temporal_candidate.weight,
        model.associative_key.weight,
        model.dual_fusion.weight,
        model.workspace_layers[0].mlp_in.weight,
    ):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all().item()) is True


def test_pmrm_reset_mask_only_resets_selected_streams() -> None:
    model = MarulhoPMRMLanguageModel(_config()).eval()
    state = model(torch.randint(0, 64, (3, 4)), collect_telemetry=False)["state"]
    reset = model.reset_state(state, torch.tensor([False, True, False]))
    fresh = model.init_state(3)
    for key in state:
        torch.testing.assert_close(reset[key][0], state[key][0])
        torch.testing.assert_close(reset[key][2], state[key][2])
        torch.testing.assert_close(reset[key][1], fresh[key][1])


@pytest.mark.parametrize("fusion_kind", PMRM_FUSION_KINDS)
def test_pmrm_fusion_variants_share_one_model_interface(fusion_kind: str) -> None:
    model = MarulhoPMRMLanguageModel(_config(fusion_kind=fusion_kind))
    assert isinstance(model, CausalLanguageModel)
    result = model.next_token_loss(
        torch.randint(0, 64, (2, 4)),
        torch.randint(0, 64, (2, 4)),
        collect_telemetry=False,
    )
    assert bool(torch.isfinite(result["loss"]).item()) is True


def test_pmrm_none_policy_has_no_hidden_episode_activity() -> None:
    model = MarulhoPMRMLanguageModel(_config(episodic_policy="none")).eval()
    result = model(torch.randint(0, 64, (2, 6)))
    assert result["telemetry"]["episodic_write_count"] == 0
    assert result["telemetry"]["episodic_read_count"] == 0
    assert int(result["state"]["episodes_valid"].sum().item()) == 0


def test_pmrm_checkpoint_preserves_tokenizer_weights_and_runtime_state(
    tmp_path: Path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    config = _config(vocab_size=tokenizer.vocab_size)
    model = MarulhoPMRMLanguageModel(config).eval()
    state = model(
        torch.tensor([[tokenizer.bos_id, 10, 11, 12]]),
        collect_telemetry=False,
    )["state"]
    path = save_pmrm_language_checkpoint(
        tmp_path / "pmrm.pt",
        model,
        tokenizer,
        metadata={"test": True},
        runtime_state=state,
    )
    with pytest.raises(ValueError, match="legacy language checkpoint"):
        load_language_model_checkpoint(path)

    restored, restored_tokenizer, metadata, restored_state = (
        load_pmrm_language_checkpoint(path)
    )
    assert metadata == {"test": True}
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert restored_state is not None
    expected = model.step(torch.tensor([13]), state, collect_telemetry=False)
    actual = restored.step(
        torch.tensor([13]), restored_state, collect_telemetry=False
    )
    torch.testing.assert_close(expected["logits"], actual["logits"])
    _assert_state_equal(expected["state"], actual["state"])


def test_pmrm_matched_arm_parameter_count_is_within_half_percent() -> None:
    model = MarulhoPMRMLanguageModel(
        PMRMLanguageConfig(
            vocab_size=8192,
            embedding_dim=512,
            state_dim=512,
            column_count=8,
            active_columns=2,
            associative_dim=64,
            episodic_slots=16,
            episodic_reads=2,
            workspace_registers=2,
            workspace_layers=3,
            workspace_iterations=2,
            workspace_mlp_dim=1712,
            context_length=256,
        )
    )
    transformer_parameters = 20_976_128
    pmrm_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert pmrm_parameters == 20_977_792
    assert abs(pmrm_parameters - transformer_parameters) / transformer_parameters < 0.005
