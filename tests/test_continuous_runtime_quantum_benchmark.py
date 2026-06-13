from pathlib import Path

from marulho.evaluation.continuous_runtime_quantum_benchmark import (
    run_continuous_runtime_quantum_ab,
)


def test_continuous_runtime_quantum_ab_reports_reversed_schedule_speedup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[tuple[int, float]] = []

    def _fake_run_arm(**kwargs):
        quantum_tokens = int(kwargs["quantum_tokens"])
        yield_seconds = float(kwargs["yield_seconds"])
        observed.append((quantum_tokens, yield_seconds))
        tokens_per_second = 800.0 if quantum_tokens == 8 else 180.0
        return {
            "success": True,
            "token_delta": 256,
            "tokens_per_second": tokens_per_second,
            "execution_schedule": {
                "quantum_tokens": quantum_tokens,
                "yield_seconds": yield_seconds,
            },
        }

    monkeypatch.setattr(
        "marulho.evaluation.continuous_runtime_quantum_benchmark._run_arm",
        _fake_run_arm,
    )
    output_path = tmp_path / "quantum-ab.json"

    report = run_continuous_runtime_quantum_ab(
        tmp_path / "runtime.pt",
        output_path=output_path,
        target_tokens=256,
    )

    assert observed == [
        (1, 0.005),
        (8, 0.0),
        (8, 0.0),
        (1, 0.005),
    ]
    assert report["success"] is True
    assert report["legacy_mean_tokens_per_second"] == 180.0
    assert report["quantum_mean_tokens_per_second"] == 800.0
    assert report["speedup"] > 4.0
    assert output_path.exists()
