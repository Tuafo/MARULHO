from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_role_contrastive_falsification import (
    RETIRE_DECISION,
    RoleContrastivePilotConfig,
    select_role_contrastive_candidate,
)
from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_role_contrastive import (
    RoleContrastiveBranch,
    build_role_contrastive_branches,
    prepare_role_contrastive_branches,
    prepare_role_contrastive_lookup,
    role_contrastive_lookup_unlikelihood,
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


def test_cross_entropy_denominator_reuse_is_exact() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    inputs, targets = _inputs_and_targets(
        tokenizer, "Question: Where? Answer: The coin is in the cup."
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
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(
        (1, targets.shape[1], tokenizer.vocab_size), generator=generator
    )
    direct, direct_count = role_contrastive_unlikelihood(
        logits, targets, mask, branches
    )
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    reused, reused_count = role_contrastive_unlikelihood(
        logits,
        targets,
        mask,
        branches,
        log_denominators=token_losses + target_logits,
    )
    assert int(direct_count) == int(reused_count)
    torch.testing.assert_close(direct, reused, atol=1.0e-6, rtol=1.0e-6)


def test_vectorized_lookup_matches_reference_branch_loss() -> None:
    branches = (
        RoleContrastiveBranch("entity", "Ava", (10, 11), 0, (20, 30)),
        RoleContrastiveBranch("entity", "Ben", (20,), 0, (10, 30)),
        RoleContrastiveBranch("entity", "Cora", (30, 31), 0, (10, 20)),
    )
    prepared = prepare_role_contrastive_branches(branches, device="cpu")
    lookup = prepare_role_contrastive_lookup(
        branches, vocab_size=40, device="cpu"
    )
    targets = torch.tensor([[10, 11, 5, 20, 5, 30, 31]])
    answer_mask = torch.ones_like(targets, dtype=torch.bool)
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((1, targets.shape[1], 40), generator=generator)
    denominators = torch.logsumexp(logits, dim=-1)
    reference, reference_count = role_contrastive_unlikelihood(
        logits,
        targets,
        answer_mask,
        prepared,
        log_denominators=denominators,
    )
    fused, fused_count = role_contrastive_lookup_unlikelihood(
        logits,
        targets,
        answer_mask,
        lookup,
        log_denominators=denominators,
    )
    assert int(reference_count) == int(fused_count) == 3
    torch.testing.assert_close(reference, fused, atol=1.0e-6, rtol=1.0e-6)


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
