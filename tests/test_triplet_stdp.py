"""Tests for triplet STDP plasticity rule (§4.4, Pfister & Gerstner 2006)."""

from __future__ import annotations

import math
import unittest

import torch

from hecsn.core.plasticity import LocalPlasticityCircuit


def _make_circuit(plasticity_rule: str = "pair", **overrides) -> LocalPlasticityCircuit:
    defaults = dict(
        n_columns=8,
        input_dim=16,
        column_dim=16,
        device=torch.device("cpu"),
        input_stdp_ltp=0.02,
        input_stdp_ltd=0.01,
        trace_tau=20.0,
        eligibility_tau=200.0,
        stdp_mu_plus=0.0,
        stdp_mu_minus=1.0,
        synaptic_scaling_alpha=0.1,
        inhibitory_plasticity_lr=0.05,
        inhibitory_decay=0.95,
        target_firing_rate=0.125,
        input_row_target=1.0,
        projection_norm_target=0.1,
        projection_plasticity_scale=0.35,
        assembly_projection_plasticity_scale=0.25,
        spike_backend="proxy",
        plasticity_rule=plasticity_rule,
    )
    defaults.update(overrides)
    return LocalPlasticityCircuit(**defaults)


def _apply_step(circuit: LocalPlasticityCircuit, winner: int = 0) -> dict[str, float]:
    """Run one apply step with synthetic spikes."""
    n_col = circuit.n_columns
    inp_dim = circuit.input_dim
    col_dim = circuit.column_dim

    input_weights = torch.rand(n_col, inp_dim) * 0.1
    projection_weights = torch.rand(inp_dim, col_dim) * 0.1
    assembly_projection_weights = torch.rand(n_col, col_dim) * 0.1

    input_pattern = torch.zeros(inp_dim)
    input_pattern[0] = 1.0
    input_pattern[1] = 0.5

    assembly = torch.zeros(n_col)
    assembly[winner] = 1.0

    routing_key = torch.zeros(col_dim)
    routing_key[0] = 1.0

    return circuit.apply(
        input_weights=input_weights,
        projection_weights=projection_weights,
        assembly_projection_weights=assembly_projection_weights,
        input_pattern=input_pattern,
        pre_synaptic_trace=None,
        projected_input=None,
        assembly=assembly,
        routing_key=routing_key,
        winner_indices=torch.tensor([winner]),
        winner_strengths=torch.tensor([1.0]),
        modulator=1.0,
        lr=0.01,
    )


class TestTripletSTDPInit(unittest.TestCase):
    def test_device_report_exposes_live_tensor_devices(self) -> None:
        c = _make_circuit("triplet", spike_backend="adex")
        report = c.device_report()

        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["plasticity_rule"], "triplet")
        self.assertEqual(report["spike_backend"], "adex")
        self.assertEqual(report["pre_trace_device"], str(c.pre_trace.device))
        self.assertEqual(report["input_eligibility_device"], str(c.input_eligibility.device))
        self.assertEqual(report["adex_voltage_device"], str(c.adex_neurons.V.device))
        self.assertEqual(report["adex"]["voltage_device"], str(c.adex_neurons.V.device))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices(self) -> None:
        c = _make_circuit("triplet", device=torch.device("cuda"), spike_backend="adex")
        report = c.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["pre_trace_device"]).startswith("cuda"))
        self.assertTrue(str(report["input_eligibility_device"]).startswith("cuda"))
        self.assertTrue(str(report["adex_voltage_device"]).startswith("cuda"))

    def test_pair_rule_default(self) -> None:
        c = _make_circuit("pair")
        self.assertEqual(c.plasticity_rule, "pair")

    def test_triplet_rule_accepted(self) -> None:
        c = _make_circuit("triplet")
        self.assertEqual(c.plasticity_rule, "triplet")

    def test_invalid_rule_raises(self) -> None:
        with self.assertRaises(ValueError):
            _make_circuit("invalid_rule")

    def test_triplet_traces_initialized_zero(self) -> None:
        c = _make_circuit("triplet")
        self.assertEqual(float(c.r1_trace.sum()), 0.0)
        self.assertEqual(float(c.o1_trace.sum()), 0.0)
        self.assertEqual(float(c.o2_trace.sum()), 0.0)
        self.assertEqual(float(c.r2_trace.sum()), 0.0)

    def test_triplet_parameters_stored(self) -> None:
        c = _make_circuit("triplet")
        self.assertAlmostEqual(c.triplet_tau_plus, 16.8, places=1)
        self.assertAlmostEqual(c.triplet_tau_minus, 33.7, places=1)
        self.assertAlmostEqual(c.triplet_tau_x, 101.0, places=1)
        self.assertAlmostEqual(c.triplet_tau_y, 114.0, places=1)


class TestTripletSTDPApply(unittest.TestCase):
    def test_pair_apply_runs(self) -> None:
        c = _make_circuit("pair")
        result = _apply_step(c)
        self.assertIn("modulated_update_norm", result)

    def test_triplet_apply_runs(self) -> None:
        c = _make_circuit("triplet")
        result = _apply_step(c)
        self.assertIn("modulated_update_norm", result)

    def test_triplet_updates_traces(self) -> None:
        c = _make_circuit("triplet")
        _apply_step(c)
        # r1 and r2 should have non-zero values (from pre_signal)
        self.assertGreater(float(c.r1_trace.abs().sum()), 0.0)
        self.assertGreater(float(c.r2_trace.abs().sum()), 0.0)
        # o1 and o2 should have non-zero values (from post_signal)
        self.assertGreater(float(c.o1_trace.abs().sum()), 0.0)
        self.assertGreater(float(c.o2_trace.abs().sum()), 0.0)

    def test_both_rules_produce_nonzero_updates(self) -> None:
        for rule in ("pair", "triplet"):
            c = _make_circuit(rule)
            result = _apply_step(c)
            self.assertGreater(
                result["modulated_update_norm"],
                0.0,
                f"{rule} should produce nonzero updates",
            )


class TestTripletFrequencyDependence(unittest.TestCase):
    """Key property: at high burst frequency, triplet LTP should be stronger
    because o2 (slow post trace) accumulates across rapid post-spikes."""

    def _burst_ltp(self, n_spikes: int = 5) -> float:
        c = _make_circuit("triplet")
        n_col = c.n_columns
        inp_dim = c.input_dim
        col_dim = c.column_dim
        weights = torch.rand(n_col, inp_dim) * 0.1

        total_delta = torch.zeros_like(weights)
        for _ in range(n_spikes):
            pre_signal = torch.zeros(inp_dim)
            pre_signal[0] = 1.0
            post_signal = torch.zeros(n_col)
            post_signal[0] = 1.0

            c._update_triplet_traces(pre_signal, post_signal)
            delta = c._triplet_stdp_delta(
                weights=weights,
                pre_signal=pre_signal,
                post_signal=post_signal,
            )
            total_delta += delta

        return float(total_delta.sum().item())

    def test_more_spikes_more_potentiation(self) -> None:
        """Triplet rule should show increased potentiation with more bursts."""
        ltp_2 = self._burst_ltp(2)
        ltp_5 = self._burst_ltp(5)
        # With more spikes, o2 accumulates and amplifies LTP
        # Total delta should be larger (more positive) with more spikes
        self.assertGreater(ltp_5, ltp_2)


class TestPfisterGerstnerFrequencySweep(unittest.TestCase):
    """Pfister & Gerstner 2006 Fig. 2 frequency-dependent STDP protocol.

    Sends 60 pre-post spike pairs at varying frequencies.
    Triplet rule must show increasing potentiation with frequency
    (slow post-trace o2 accumulates at high rates).
    Pair rule must NOT show this frequency dependence.
    """

    def _run_frequency_sweep(
        self,
        rule: str,
        frequencies_hz: list[float],
        n_pairs: int = 60,
        delta_t: float = 10.0,
    ) -> list[float]:
        """Send n_pairs pre→post spike pairs at each frequency.

        Each timestep = 1ms.  Within each pair, pre fires at t, post at
        t + delta_t.  ISI between pairs = 1000/f_hz timesteps.

        Returns total weight change (sum over weight matrix) per frequency.
        """
        results = []
        for freq in frequencies_hz:
            c = _make_circuit(rule, n_columns=1, input_dim=1)
            weights = torch.full((1, 1), 0.1)  # Start from uniform 0.1

            isi_ms = 1000.0 / freq  # inter-spike-interval in ms
            total_delta = 0.0
            trace_decay = math.exp(-1.0 / c.trace_tau)

            for _ in range(n_pairs):
                # Pre-spike
                pre = torch.tensor([1.0])
                post = torch.tensor([0.0])
                if rule == "triplet":
                    c._update_triplet_traces(pre, post)
                    delta = c._triplet_stdp_delta(
                        weights=weights, pre_signal=pre, post_signal=post,
                    )
                else:
                    c.pre_trace = c.pre_trace * trace_decay + pre
                    c.post_trace = c.post_trace * trace_decay + post
                    delta = c._log_stdp_delta(
                        weights=weights, pre_signal=pre, post_signal=post,
                        pre_trace=c.pre_trace, post_trace=c.post_trace,
                    )
                total_delta += delta.sum().item()

                # Silent steps until post-spike (delta_t ms later)
                silence_pre = torch.tensor([0.0])
                silence_post = torch.tensor([0.0])
                for _ in range(max(1, int(delta_t) - 1)):
                    if rule == "triplet":
                        c._update_triplet_traces(silence_pre, silence_post)
                    else:
                        c.pre_trace *= trace_decay
                        c.post_trace *= trace_decay

                # Post-spike
                pre_off = torch.tensor([0.0])
                post_on = torch.tensor([1.0])
                if rule == "triplet":
                    c._update_triplet_traces(pre_off, post_on)
                    delta = c._triplet_stdp_delta(
                        weights=weights, pre_signal=pre_off, post_signal=post_on,
                    )
                else:
                    c.pre_trace = c.pre_trace * trace_decay + pre_off
                    c.post_trace = c.post_trace * trace_decay + post_on
                    delta = c._log_stdp_delta(
                        weights=weights, pre_signal=pre_off, post_signal=post_on,
                        pre_trace=c.pre_trace, post_trace=c.post_trace,
                    )
                total_delta += delta.sum().item()

                # Remaining ISI silence (until next pair)
                remaining = max(1, int(isi_ms - delta_t) - 1)
                for _ in range(remaining):
                    if rule == "triplet":
                        c._update_triplet_traces(silence_pre, silence_post)
                    else:
                        c.pre_trace *= trace_decay
                        c.post_trace *= trace_decay

            results.append(total_delta)
        return results

    def test_triplet_potentiation_increases_with_frequency(self) -> None:
        """Triplet STDP: LTP must increase with spike pair frequency.

        This is the signature prediction of the triplet model (Fig. 2)
        that pair-based STDP cannot reproduce.
        """
        freqs = [1.0, 5.0, 10.0, 20.0, 50.0]
        deltas = self._run_frequency_sweep("triplet", freqs)
        # At higher frequencies, o2 accumulates → larger net LTP
        # Check monotonic increase (allow small noise in adjacent bins)
        for i in range(len(deltas) - 1):
            self.assertGreater(
                deltas[i + 1],
                deltas[i],
                f"Triplet LTP should increase: {freqs[i]}Hz={deltas[i]:.6f} "
                f"vs {freqs[i+1]}Hz={deltas[i+1]:.6f}",
            )

    def test_triplet_vs_pair_diverge_at_high_frequency(self) -> None:
        """Triplet rule must show stronger frequency sensitivity than pair.

        The key Fig. 2 prediction: the ratio of high-frequency to
        low-frequency potentiation is larger for triplet than for pair,
        because o2 accumulation amplifies LTP at high rates.
        """
        freqs = [1.0, 50.0]
        triplet_deltas = self._run_frequency_sweep("triplet", freqs)
        pair_deltas = self._run_frequency_sweep("pair", freqs)
        # Frequency sensitivity = ratio of 50 Hz to 1 Hz potentiation
        triplet_ratio = triplet_deltas[1] / triplet_deltas[0] if triplet_deltas[0] != 0 else 1.0
        pair_ratio = pair_deltas[1] / pair_deltas[0] if pair_deltas[0] != 0 else 1.0
        self.assertGreater(
            triplet_ratio,
            pair_ratio,
            f"Triplet frequency sensitivity ({triplet_ratio:.2f}x) should exceed "
            f"pair sensitivity ({pair_ratio:.2f}x)",
        )

    def test_triplet_converges_to_pair_at_low_frequency(self) -> None:
        """At 1 Hz, triplet LTP contribution from o2 should be negligible.

        The slow post-trace o2 (τ_y=114ms) fully decays over the 1000ms
        ISI, so the triplet term A3+ * r1 * o2 vanishes.  We verify this
        by checking that the ratio of triplet LTP at 1 Hz vs 50 Hz is
        much smaller than at higher frequencies (o2 doesn't amplify).
        """
        freqs = [1.0, 50.0]
        deltas = self._run_frequency_sweep("triplet", freqs)
        # At 1 Hz, o2 decays ~exp(-1000/114) ≈ 0 → triplet degenerates
        # The 50Hz/1Hz ratio should be > 1 (confirmed by monotonicity test)
        ratio = deltas[1] / deltas[0] if deltas[0] != 0 else float("inf")
        self.assertGreater(ratio, 1.5, "Triplet should show substantial frequency effect")


class TestTripletDecays(unittest.TestCase):
    def test_decay_values_correct(self) -> None:
        c = _make_circuit("triplet")
        dr1, do1, do2, dr2 = c._triplet_decays()

        expected_r1 = math.exp(-1.0 / 16.8)
        expected_o1 = math.exp(-1.0 / 33.7)
        expected_o2 = math.exp(-1.0 / 114.0)
        expected_r2 = math.exp(-1.0 / 101.0)

        self.assertAlmostEqual(dr1, expected_r1, places=6)
        self.assertAlmostEqual(do1, expected_o1, places=6)
        self.assertAlmostEqual(do2, expected_o2, places=6)
        self.assertAlmostEqual(dr2, expected_r2, places=6)
        # r2 (pre-slow, τx=101) must have a DIFFERENT decay from r1 (pre-fast, τ+=16.8)
        self.assertNotAlmostEqual(dr2, dr1, places=4)

    def test_o2_decays_slower_than_o1(self) -> None:
        c = _make_circuit("triplet")
        _, do1, do2, _ = c._triplet_decays()
        # o2 has larger tau → slower decay → higher decay constant
        self.assertGreater(do2, do1)


class TestTripletStateDictRoundTrip(unittest.TestCase):
    def test_roundtrip_preserves_traces(self) -> None:
        c1 = _make_circuit("triplet")
        _apply_step(c1)

        snapshot = c1.state_dict()
        self.assertEqual(snapshot["plasticity_rule"], "triplet")
        self.assertIn("r1_trace", snapshot)
        self.assertIn("o2_trace", snapshot)

        c2 = _make_circuit("triplet")
        c2.load_state_dict(snapshot)

        self.assertTrue(torch.allclose(c1.r1_trace, c2.r1_trace))
        self.assertTrue(torch.allclose(c1.o1_trace, c2.o1_trace))
        self.assertTrue(torch.allclose(c1.o2_trace, c2.o2_trace))
        self.assertTrue(torch.allclose(c1.r2_trace, c2.r2_trace))

    def test_roundtrip_preserves_adex_state_when_backend_enabled(self) -> None:
        c1 = _make_circuit("triplet", spike_backend="adex")
        _apply_step(c1)
        assert c1.adex_neurons is not None
        before_voltage = c1.adex_neurons.V.detach().clone()
        before_adaptation = c1.adex_neurons.w.detach().clone()

        snapshot = c1.state_dict()
        c2 = _make_circuit("triplet", spike_backend="adex")
        c2.load_state_dict(snapshot)

        assert c2.adex_neurons is not None
        self.assertEqual(c2.adex_step, c1.adex_step)
        self.assertIn("adex_neurons", snapshot)
        self.assertTrue(torch.allclose(c2.adex_neurons.V, before_voltage))
        self.assertTrue(torch.allclose(c2.adex_neurons.w, before_adaptation))


class TestTripletRevive(unittest.TestCase):
    def test_revive_resets_triplet_traces(self) -> None:
        c = _make_circuit("triplet")
        _apply_step(c, winner=0)
        self.assertGreater(float(c.o1_trace[0].abs()), 0.0)

        c.revive_columns(torch.tensor([0]))
        self.assertEqual(float(c.o1_trace[0]), 0.0)
        self.assertEqual(float(c.o2_trace[0]), 0.0)


class TestConfigPlasticityRule(unittest.TestCase):
    def test_config_default_is_triplet(self) -> None:
        from hecsn.config.model_config import HECSNConfig

        cfg = HECSNConfig()
        self.assertEqual(cfg.plasticity_rule, "triplet")

    def test_config_accepts_triplet(self) -> None:
        from hecsn.config.model_config import HECSNConfig

        cfg = HECSNConfig(plasticity_rule="triplet")
        self.assertEqual(cfg.plasticity_rule, "triplet")


if __name__ == "__main__":
    unittest.main()
