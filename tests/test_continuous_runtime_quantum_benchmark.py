from pathlib import Path

from marulho.evaluation.continuous_runtime_quantum_benchmark import (
    run_continuous_runtime_quantum_ab,
)


def test_continuous_runtime_quantum_ab_reports_maintained_schedule_speedup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[tuple[int, float]] = []

    def _fake_run_arm(**kwargs):
        quantum_tokens = int(kwargs["quantum_tokens"])
        yield_seconds = float(kwargs["yield_seconds"])
        observed.append((quantum_tokens, yield_seconds))
        tokens_per_second = 900.0 if quantum_tokens == 16 else 800.0
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
        (8, 0.0),
        (16, 0.0),
        (16, 0.0),
        (8, 0.0),
    ]
    assert report["success"] is True
    assert report["surface"] == "continuous_runtime_quantum_ab.v2"
    assert report["baseline_quantum_mean_tokens_per_second"] == 800.0
    assert report["candidate_quantum_mean_tokens_per_second"] == 900.0
    assert report["candidate_over_baseline_quantum_speedup"] == 1.125
    assert "legacy_mean_tokens_per_second" not in report
    assert "legacy_to_candidate_speedup" not in report
    assert "quantum_mean_tokens_per_second" not in report
    assert "speedup" not in report
    assert output_path.exists()
