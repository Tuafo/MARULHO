from __future__ import annotations

import json

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_scale_ladder import (
    SURFACE,
    build_language_scale_ladder_report,
    build_smoke_fixture_report,
    default_language_scale_ladder,
    estimate_language_model_parameters,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
)


def test_language_scale_estimate_matches_instantiated_small_model() -> None:
    config = LanguageModelConfig(
        vocab_size=256,
        embedding_dim=12,
        state_dim=20,
        expert_count=3,
        active_expert_count=1,
        route_candidate_count=2,
        expert_hidden_dim=32,
    )
    model = MarulhoLanguageModel(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    estimate = estimate_language_model_parameters(config)

    assert estimate["total_parameters"] == actual
    assert estimate["parameter_breakdown"]["routed_experts"] > 0
    assert estimate["active_parameters_per_token_estimate"] < actual
    assert estimate["dense_vocab_head_active"] is True
    assert estimate["sampled_or_adaptive_vocab_xent_present"] is False


def test_language_scale_estimate_counts_bounded_memory_slots() -> None:
    config = LanguageModelConfig(
        vocab_size=256,
        embedding_dim=12,
        state_dim=20,
        expert_count=3,
        active_expert_count=1,
        route_candidate_count=2,
        expert_hidden_dim=32,
        memory_slot_count=4,
        memory_slot_candidate_count=2,
        active_memory_slot_count=1,
    )
    model = MarulhoLanguageModel(config)
    actual = sum(parameter.numel() for parameter in model.parameters())
    estimate = estimate_language_model_parameters(config)

    assert estimate["total_parameters"] == actual
    assert estimate["parameter_breakdown"]["memory_slots"] == 80
    assert estimate["parameter_breakdown"]["memory_slot_gate"] == 1
    assert estimate["memory_slot_count"] == 4
    assert estimate["memory_slot_candidate_count"] == 2
    assert estimate["active_memory_slot_count_per_token"] == 1
    assert estimate["active_parameters_per_token_estimate"] < actual


def test_default_language_scale_ladder_defines_target_classes() -> None:
    entries = default_language_scale_ladder()
    by_name = {entry.name: entry for entry in entries}

    assert set(by_name) == {
        "small_fixture",
        "nord_140m_class",
        "growth_500m_class",
        "neuronspark_0_9b_class",
        "research_2b_plus_class",
    }
    for entry in entries:
        estimate = estimate_language_model_parameters(entry.config)
        assert entry.min_total_parameters <= estimate["total_parameters"] <= entry.max_total_parameters
        assert 0.0 < estimate["active_parameter_fraction_estimate"] <= 1.0
        if entry.name != "small_fixture":
            assert entry.instantiate_by_default is False


def test_language_scale_ladder_report_writes_defined_not_promoted_json(tmp_path) -> None:
    output = tmp_path / "scale-ladder.json"

    report = build_language_scale_ladder_report(output_path=output)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["entry_count"] == 5
    assert report["promotion_gate"]["scale_ladder_defined"] is True
    assert report["promotion_gate"]["large_ladders_instantiated"] is False
    assert report["promotion_gate"]["frontier_competitiveness_claimed"] is False
    assert report["entries"][1]["gate"]["claim"] == "configuration_defined_not_trained"
    assert (tmp_path / "README.md").exists()


def test_language_scale_ladder_smoke_fixture_runs_loss_and_generation(tmp_path) -> None:
    output = tmp_path / "scale-ladder-smoke.json"

    report = build_smoke_fixture_report(output)
    smoke = report["smoke_fixture"]

    assert smoke["ran"] is True
    assert smoke["heldout_loss"] > 0.0
    assert smoke["heldout_perplexity"] > 0.0
    assert smoke["generated_token_count"] >= 1
    assert smoke["external_llm_used"] is False
    assert smoke["promotes_scale_claim"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["smoke_fixture"]["ran"] is True


def test_language_scale_ladder_accepts_explicit_smoke_model() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["explicit smoke model keeps scale report executable. " * 4],
        tokenizer,
        sequence_length=10,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )

    report = build_language_scale_ladder_report(
        smoke_model=model,
        smoke_tokenizer=tokenizer,
        smoke_eval_batches=split.eval,
    )

    assert report["smoke_fixture"]["ran"] is True
    assert report["smoke_fixture"]["eval_token_count"] > 0
    assert report["promotion_gate"]["large_ladders_trained"] is False
