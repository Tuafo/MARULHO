from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from hecsn.core import AdExNeuron
from hecsn.reporting.io import write_json_file


def _run_case(
    *,
    name: str,
    neuron: AdExNeuron,
    current_value: float,
    steps: int,
    expect_spike: bool,
) -> dict[str, Any]:
    current = torch.full((neuron.n,), float(current_value), dtype=torch.float32, device=neuron.device)
    total_spikes = 0
    first_spike_time: float | None = None
    max_voltage = float(neuron.V.max().item())
    min_voltage = float(neuron.V.min().item())
    max_abs_adaptation = float(neuron.w.abs().max().item())

    for step in range(int(steps)):
        t = float(step) * float(neuron.dt)
        spikes = neuron.step(current, t=t)
        spike_count = int(spikes.sum().item())
        total_spikes += spike_count
        if spike_count > 0 and first_spike_time is None:
            first_spike_time = t
        max_voltage = max(max_voltage, float(neuron.V.max().item()))
        min_voltage = min(min_voltage, float(neuron.V.min().item()))
        max_abs_adaptation = max(max_abs_adaptation, float(neuron.w.abs().max().item()))

    finite_voltage = bool(torch.isfinite(neuron.V).all().item())
    finite_adaptation = bool(torch.isfinite(neuron.w).all().item())
    spiked = total_spikes > 0
    resting_ok = (not expect_spike) and (not spiked) and abs(float(neuron.V.mean().item()) - float(neuron.E_L)) < 1e-3
    passed = bool(finite_voltage and finite_adaptation and ((expect_spike and spiked) or resting_ok))

    return {
        "name": name,
        "n_neurons": int(neuron.n),
        "dt": float(neuron.dt),
        "current_value": float(current_value),
        "steps": int(steps),
        "expect_spike": bool(expect_spike),
        "spike_count_total": int(total_spikes),
        "spiked": bool(spiked),
        "first_spike_time": first_spike_time,
        "finite_voltage": finite_voltage,
        "finite_adaptation": finite_adaptation,
        "final_mean_voltage": float(neuron.V.mean().item()),
        "final_mean_adaptation": float(neuron.w.mean().item()),
        "max_voltage_observed": float(max_voltage),
        "min_voltage_observed": float(min_voltage),
        "max_abs_adaptation": float(max_abs_adaptation),
        "pass": bool(passed),
    }


def run_adex_stability_benchmark(*, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        _run_case(
            name="rest_exc",
            neuron=AdExNeuron(n_neurons=32, dt=0.5, device="cpu", burst_mode=True),
            current_value=0.0,
            steps=256,
            expect_spike=False,
        ),
        _run_case(
            name="burst_exc",
            neuron=AdExNeuron(n_neurons=32, dt=0.5, device="cpu", burst_mode=True),
            current_value=25.0,
            steps=512,
            expect_spike=True,
        ),
        _run_case(
            name="stress_exc",
            neuron=AdExNeuron(n_neurons=32, dt=0.5, device="cpu", burst_mode=True),
            current_value=40.0,
            steps=1024,
            expect_spike=True,
        ),
        _run_case(
            name="coarse_dt_exc",
            neuron=AdExNeuron(n_neurons=32, dt=1.0, device="cpu", burst_mode=True),
            current_value=20.0,
            steps=384,
            expect_spike=True,
        ),
        _run_case(
            name="fast_inhibitory",
            neuron=AdExNeuron.inhibitory(n_neurons=32, dt=0.5, device="cpu"),
            current_value=20.0,
            steps=512,
            expect_spike=True,
        ),
    ]

    case_count = len(cases)
    finite_case_fraction = float(
        sum(1 for case in cases if case["finite_voltage"] and case["finite_adaptation"]) / max(1, case_count)
    )
    expected_spiking_cases = [case for case in cases if case["expect_spike"]]
    spiking_case_fraction = float(
        sum(1 for case in expected_spiking_cases if case["spiked"]) / max(1, len(expected_spiking_cases))
    )
    max_abs_adaptation = float(max(case["max_abs_adaptation"] for case in cases))
    gate_pass = bool(all(bool(case["pass"]) for case in cases))

    summary = {
        "benchmark": "adex_stability",
        "runtime_scope": {
            "mode": "standalone_adex_surface",
            "note": "This benchmark validates the standalone AdExNeuron surface alongside the optional local-plasticity AdEx spike backend before full recurrent runtime integration.",
        },
        "cases": cases,
        "metrics": {
            "case_count": int(case_count),
            "finite_case_fraction": float(finite_case_fraction),
            "spiking_case_fraction": float(spiking_case_fraction),
            "max_abs_adaptation": float(max_abs_adaptation),
        },
        "adex_stability_gate": {
            "pass": bool(gate_pass),
            "thresholds": {
                "finite_case_fraction": 1.0,
                "spiking_case_fraction": 1.0,
            },
        },
    }

    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone AdEx stability benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_adex_stability_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    args = parser.parse_args()

    summary = run_adex_stability_benchmark(output_dir=args.output_dir)
    print(f"[adex_stability] pass={summary['adex_stability_gate']['pass']}")


if __name__ == "__main__":
    main()
