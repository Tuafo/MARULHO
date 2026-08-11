from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_role_contrastive_falsification import (
    RETIRE_DECISION,
    RoleContrastivePilotConfig,
    select_role_contrastive_candidate,
)
from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_role_contrastive import (
    build_role_contrastive_branches,
    prepare_role_contrastive_branches,
    role_contrastive_unlikelihood,
)


def _inputs_and_targets(tokenizer, text: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    return torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])


def test_byte_trie_branches_follow_shared_prefixes() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    branches = build_role_contrastive_branches(
        tokenizer, {"container": ("cup", "case", "box", "basket")}
    )
    cup = [branch for branch in branches if branch.value == "cup"]
    assert len(cup) == 2
    assert cup[0].target_offset == 1
    assert set(cup[0].negative_ids) == {
        tokenizer.byte_offset + ord("b"),
    }
    assert cup[1].target_offset == 2
    assert set(cup[1].negative_ids) == {
        tokenizer.byte_offset + ord("a"),
    }


def test_role_unlikelihood_only_counts_marked_answer_occurrences() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    inputs, targets = _inputs_and_targets(
        tokenizer, "Ava saw a cup. Question: Where? Answer: The coin is in the cup."
    )
    marker = torch.tensor(
        tokenizer.encode(" Answer:", add_bos=False, add_eos=False)
    )
    mask = answer_target_mask(inputs, marker_ids=marker, eos_id=tokenizer.eos_id)
    branches = prepare_role_contrastive_branches(
        build_role_contrastive_branches(
            tokenizer, {"container": ("cup", "case", "box", "basket")}
        ),
        device="cpu",
    )
    logits = torch.zeros((1, targets.shape[1], tokenizer.vocab_size), requires_grad=True)
    loss, count = role_contrastive_unlikelihood(logits, targets, mask, branches)
    assert int(count) == 2
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert bool(torch.isfinite(logits.grad).all())


def test_role_unlikelihood_is_zero_without_role_answer() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    inputs, targets = _inputs_and_targets(tokenizer, "Question: Why? Answer: Unknown.")
    marker = torch.tensor(
        tokenizer.encode(" Answer:", add_bos=False, add_eos=False)
    )
    mask = answer_target_mask(inputs, marker_ids=marker, eos_id=tokenizer.eos_id)
    branches = prepare_role_contrastive_branches(
        build_role_contrastive_branches(tokenizer, {"polarity": ("Some", "No")}),
        device="cpu",
    )
    logits = torch.randn(
        (1, targets.shape[1], tokenizer.vocab_size), requires_grad=True
    )
    loss, count = role_contrastive_unlikelihood(logits, targets, mask, branches)
    assert int(count) == 0
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None


def test_pilot_selector_requires_joint_free_and_weak_kind_gain() -> None:
    def row(free: float, ownership: float, container: float, loss: float) -> dict:
        return {
            "all_parameters_received_final_gradient": True,
            "heldout": {"heldout_loss": loss},
            "relation": {
                "generation_exact_accuracy": free,
                "generation_kind_accuracy": {
                    "ownership": ownership,
                    "container": container,
                },
            },
        }

    arms = {
        "answer4_control": row(0.50, 0.10, 0.20, 3.10),
        "role_contrastive_025": row(0.56, 0.16, 0.20, 3.12),
        "role_contrastive_1": row(0.60, 0.10, 0.20, 3.09),
    }
    selected, decision, _ = select_role_contrastive_candidate(
        arms, config=RoleContrastivePilotConfig()
    )
    assert selected == "role_contrastive_025"
    assert decision != RETIRE_DECISION

    arms["role_contrastive_025"] = row(0.54, 0.30, 0.20, 3.10)
    selected, decision, _ = select_role_contrastive_candidate(
        arms, config=RoleContrastivePilotConfig()
    )
    assert selected is None
    assert decision == RETIRE_DECISION
