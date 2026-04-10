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


class TestTripletDecays(unittest.TestCase):
    def test_decay_values_correct(self) -> None:
        c = _make_circuit("triplet")
        dr1, do1, do2, dr2 = c._triplet_decays()

        expected_r1 = math.exp(-1.0 / 16.8)
        expected_o1 = math.exp(-1.0 / 33.7)
        expected_o2 = math.exp(-1.0 / 114.0)

        self.assertAlmostEqual(dr1, expected_r1, places=6)
        self.assertAlmostEqual(do1, expected_o1, places=6)
        self.assertAlmostEqual(do2, expected_o2, places=6)
        self.assertEqual(dr2, dr1)  # r2 same tau as r1

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
