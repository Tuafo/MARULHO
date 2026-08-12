from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_native_context_falsification import (
    NativeContextFalsificationConfig,
    _decision,
    build_native_context_schedule,
    build_native_grounding_batches,
    native_context_gate,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _manifest() -> dict:
    cases = []
    for index, answer in enumerate(("cobalt", "amber")):
        question = f"Which material belongs to item {index}?"
        cases.append(
            {
                "case_id": f"case-{index}",
                "answers": [answer],
                "causal_prompt": (
                    f"Context: The item uses {answer}. Extra evidence follows.\n"
                    f"Question: {question}\nAnswer: "
                ),
                "oracle_causal_prompt": (
                    f"Context: The item uses {answer}.\n"
                    f"Question: {question}\nAnswer: "
                ),
            }
        )
    return {"cases": cases}


def test_grounding_batches_keep_prefix_answer_boundary_and_eos_weighted() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    manifest = _manifest()

    batches, report = build_native_grounding_batches(
        manifest,
        tokenizer,
        arm="native_full",
        sequence_length=160,
        batch_size=2,
    )

    assert len(batches) == 1
    batch = batches[0]
    for row, case in enumerate(manifest["cases"]):
        prefix = tokenizer.encode(
            case["causal_prompt"], add_bos=True, add_eos=False
        )
        answer = tokenizer.encode(
            case["answers"][0], add_bos=False, add_eos=True
        )
        assert batch.target_ids[row, len(prefix) - 1].item() == answer[0]
        assert batch.target_ids[row, len(prefix) + len(answer) - 2].item() == tokenizer.eos_id
        assert batch.answer_mask[row, : len(prefix) - 1].sum().item() == 0
        assert batch.answer_mask[
            row, len(prefix) - 1 : len(prefix) + len(answer) - 1
        ].all()
        assert batch.answer_mask[row].sum().item() == len(answer)
        assert batch.target_ids[row, len(prefix) + len(answer) - 1 :].eq(
            tokenizer.pad_id
        ).all()
    assert report["answer_mask_includes_eos"] is True
    assert report["right_padding_only_after_eos"] is True


def test_schedule_is_balanced_deterministic_and_epoch_complete() -> None:
    kwargs = {
        "grounding_batch_count": 4,
        "relation_batch_count": 4,
        "general_batch_counts": (2, 2),
        "grounding_epochs": 2,
        "seed": 57,
    }
    schedule, report = build_native_context_schedule(**kwargs)
    repeated, repeated_report = build_native_context_schedule(**kwargs)

    assert schedule == repeated
    assert report["sha256"] == repeated_report["sha256"]
    assert report["step_count"] == 16
    assert report["kind_counts"] == {
        "general_0": 2,
        "general_1": 2,
        "grounding": 8,
        "relation": 4,
    }
    grounding_indices = [index for kind, index in schedule if kind == "grounding"]
    assert {index: grounding_indices.count(index) for index in range(4)} == {
        0: 2,
        1: 2,
        2: 2,
        3: 2,
    }
    replay_entries = [(kind, index) for kind, index in schedule if kind != "grounding"]
    assert len(replay_entries) == len(set(replay_entries))


def test_rope_context_extension_preserves_parameters_and_short_prefix_logits() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    config = LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=32,
        state_dim=32,
        state_core="transformer",
        state_layers=2,
        attention_heads=4,
        transformer_context_length=16,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
    )
    torch.manual_seed(57)
    parent = MarulhoLanguageModel(config).eval()
    extended = MarulhoLanguageModel(
        replace(config, transformer_context_length=64)
    ).eval()
    extended.load_state_dict(parent.state_dict(), strict=True)
    inputs = torch.tensor([[1, 5, 8, 13, 21, 34]], dtype=torch.long)

    with torch.no_grad():
        parent_logits = parent(inputs, collect_telemetry=False)["logits"]
        extended_logits = extended(inputs, collect_telemetry=False)["logits"]

    assert sum(p.numel() for p in parent.parameters()) == sum(
        p.numel() for p in extended.parameters()
    )
    assert torch.equal(parent_logits, extended_logits)


def _arm(answer_count: int, *, mismatch_count: int = 0) -> dict:
    def condition(count: int) -> dict:
        return {"exact_answer_count": count, "exact_answer_accuracy": count / 256}

    return {
        "parameter_count": 100,
        "training": {
            "optimizer_steps": 2_048,
            "processed_padded_positions": 20_971_520,
            "all_parameters_received_final_gradient": True,
            "all_parameters_received_final_nonzero_gradient": True,
            "training_seconds": 1_000.0,
        },
        "general": {"heldout_loss": 3.05},
        "relation": {"generation_exact_accuracy": 0.48},
        "source_grounding": {
            "valid": True,
            "primary_gain_over_stronger_control": answer_count / 256,
            "conditions": {
                "primary": condition(answer_count),
                "mismatched_source": condition(mismatch_count),
            },
        },
        "checkpoint_fidelity": {"passed": True},
    }


def test_gate_and_terminal_branch_use_capability_before_retention() -> None:
    config = NativeContextFalsificationConfig()
    arms = {"oracle_short": _arm(130), "native_full": _arm(128)}
    parent = {
        "initial_short_prefix_exact": True,
        "checkpoint_file_exact": True,
        "tokenizer_exact": True,
    }
    baseline = {
        "general": {"heldout_loss": 3.0},
        "relation": {"generation_exact_accuracy": 0.50},
    }

    gate = native_context_gate(
        arms,
        baseline=baseline,
        parent=parent,
        config=config,
        parameter_count=100,
    )
    assert gate["passed"] is True
    assert _decision(gate) == "advance_v57_native_context_continual_checkpoint"

    failed = deepcopy(arms)
    failed["oracle_short"] = _arm(0)
    failed_gate = native_context_gate(
        failed,
        baseline=baseline,
        parent=parent,
        config=config,
        parameter_count=100,
    )
    assert _decision(failed_gate) == (
        "retire_v57_context_exonerated_base_or_objective_failure"
    )
