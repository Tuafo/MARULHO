import pytest

from marulho.evaluation.language_macro_cortex_quality import (
    GENERAL_EVAL,
    GENERAL_TRAIN,
    RELATION_CASES,
    RELATION_CORPUS,
    TOKENIZER_CHECKPOINT,
    _learning_rate,
)


def test_v70_learning_rate_is_frozen_warmup_cosine() -> None:
    assert 3.0e-5 < _learning_rate(0) < 3.0e-4
    assert _learning_rate(25) == pytest.approx(3.0e-4)
    assert _learning_rate(511) == pytest.approx(3.0e-5)
    assert all(
        _learning_rate(step + 1) <= _learning_rate(step)
        for step in range(25, 511)
    )


def test_v70_frozen_local_inputs_exist() -> None:
    for path in (
        TOKENIZER_CHECKPOINT,
        RELATION_CORPUS,
        RELATION_CASES,
        *GENERAL_TRAIN,
        *GENERAL_EVAL,
    ):
        assert path.is_file()
