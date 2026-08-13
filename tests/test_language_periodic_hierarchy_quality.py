import pytest

from marulho.evaluation.language_periodic_hierarchy_quality import _learning_rate


def test_v71_learning_rate_contract() -> None:
    assert 3.0e-5 < _learning_rate(0) < 3.0e-4
    assert _learning_rate(25) == pytest.approx(3.0e-4)
    assert _learning_rate(511) == pytest.approx(3.0e-5)
    assert all(
        _learning_rate(step + 1) <= _learning_rate(step)
        for step in range(25, 511)
    )
