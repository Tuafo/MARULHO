from marulho.evaluation.language_episodic_memory_experiment import (
    _prompt_episodes_and_question,
    select_episode_indices,
)


def test_prompt_segments_observed_episodes_from_question() -> None:
    episodes, question = _prompt_episodes_and_question(
        "Nora put a coin in a cup. Nora moved the cup. Where is the coin? Answer:"
    )
    assert episodes == (
        "Nora put a coin in a cup.",
        "Nora moved the cup.",
    )
    assert question == "Where is the coin? Answer:"


def test_episodic_write_policies_are_budgeted_and_deterministic() -> None:
    surprise = (0.2, 3.0, 1.0, 2.0)
    oracle = (0, 2, 1, 0)
    assert select_episode_indices(
        policy="surprise",
        surprise_scores=surprise,
        oracle_overlap_scores=oracle,
        slot_budget=2,
        seed=7,
    ) == (1, 3)
    assert select_episode_indices(
        policy="recency",
        surprise_scores=surprise,
        oracle_overlap_scores=oracle,
        slot_budget=2,
        seed=7,
    ) == (2, 3)
    first_random = select_episode_indices(
        policy="random",
        surprise_scores=surprise,
        oracle_overlap_scores=oracle,
        slot_budget=2,
        seed=7,
    )
    second_random = select_episode_indices(
        policy="random",
        surprise_scores=surprise,
        oracle_overlap_scores=oracle,
        slot_budget=2,
        seed=7,
    )
    assert first_random == second_random
    assert len(first_random) == 2
    assert select_episode_indices(
        policy="oracle",
        surprise_scores=surprise,
        oracle_overlap_scores=oracle,
        slot_budget=1,
        seed=7,
    ) == (1,)
