from __future__ import annotations

from pathlib import Path

from marulho.evaluation.language_organism_falsification import (
    build_counterfactual_probe_schedule,
    build_matched_schedule,
    organism_falsification_decision,
    sample_corpus_ranges,
)


def test_bounded_range_sampler_is_deterministic_and_spans_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corpus.txt"
    source.write_text(
        "".join(f"document-{index:05d} value {index}\n" for index in range(5000)),
        encoding="utf-8",
    )
    first_text, first = sample_corpus_ranges(
        source, byte_budget=4096, range_count=4
    )
    second_text, second = sample_corpus_ranges(
        source, byte_budget=4096, range_count=4
    )
    assert first_text == second_text
    assert first == second
    assert len(first["ranges"]) == 4
    assert first["ranges"][0]["start"] == 0
    assert first["ranges"][-1]["end"] > source.stat().st_size * 0.9
    assert first["selected_size_bytes"] < source.stat().st_size


def test_organism_schedule_is_reproducible_and_source_balanced() -> None:
    kwargs = {
        "step_count": 20,
        "relation_fraction": 0.2,
        "relation_batch_count": 5,
        "general_batch_counts": (9, 9),
        "seed": 17,
    }
    first = build_matched_schedule(**kwargs)
    second = build_matched_schedule(**kwargs)
    assert first == second
    assert sum(kind == "relation" for kind, _index in first) == 4
    assert sum(kind == "general_0" for kind, _index in first) == 8
    assert sum(kind == "general_1" for kind, _index in first) == 8


def test_counterfactual_probe_schedule_is_exact_and_reproducible() -> None:
    first = build_counterfactual_probe_schedule(
        step_count=405, rate=0.125, seed=19
    )
    second = build_counterfactual_probe_schedule(
        step_count=405, rate=0.125, seed=19
    )
    changed = build_counterfactual_probe_schedule(
        step_count=405, rate=0.125, seed=20
    )
    assert first == second
    assert first != changed
    assert len(first) == 405
    assert sum(first) == round(405 * 0.125)


def _decision_row(
    name: str,
    *,
    parameters: int,
    loss: float,
    free: float,
    tokens_per_second: float,
    tokens: int = 4_199_040,
) -> dict:
    return {
        "name": name,
        "status": "completed",
        "parameters": {"total_parameters": parameters},
        "training": {
            "processed_tokens": tokens,
            "tokens_per_second": tokens_per_second,
        },
        "general_holdout": {"after": {"heldout_loss": loss}},
        "relation": {"generation_exact_accuracy": free},
    }


def test_organism_decision_requires_quality_memory_and_systems() -> None:
    baseline = _decision_row(
        "transformer",
        parameters=20_976_128,
        loss=5.0,
        free=0.10,
        tokens_per_second=80_000.0,
    )
    survivor = _decision_row(
        "organism",
        parameters=20_971_120,
        loss=5.04,
        free=0.09,
        tokens_per_second=30_000.0,
    )
    assert organism_falsification_decision((baseline, survivor)) == (
        "continue_organism_to_durable_budget_and_unseen_generation"
    )
    slow = {
        **survivor,
        "training": {**survivor["training"], "tokens_per_second": 10_000.0},
    }
    assert organism_falsification_decision((baseline, slow)) == (
        "redesign_organism_execution_before_scaling"
    )
