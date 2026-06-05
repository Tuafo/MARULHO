from __future__ import annotations

import unittest

import torch

from marulho.core import AdExNeuron


class AdExNeuronTests(unittest.TestCase):
    def test_step_stays_finite_under_strong_drive(self) -> None:
        neuron = AdExNeuron(n_neurons=16, dt=0.5, device="cpu")
        current = torch.full((16,), 20.0)

        saw_spike = False
        for step in range(400):
            spikes = neuron.step(current, t=float(step) * neuron.dt)
            saw_spike = saw_spike or bool(spikes.any().item())

        self.assertTrue(saw_spike)
        self.assertTrue(torch.isfinite(neuron.V).all().item())
        self.assertTrue(torch.isfinite(neuron.w).all().item())
        self.assertTrue(torch.all(neuron.V <= neuron.V_peak).item())
        self.assertTrue(torch.all(neuron.V >= neuron.E_L - 20.0).item())

    def test_device_report_exposes_live_tensor_devices(self) -> None:
        neuron = AdExNeuron(n_neurons=4, dt=0.5, device="cpu")

        report = neuron.device_report()

        self.assertEqual(report["module"], "adex_neuron")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["voltage_device"], str(neuron.V.device))
        self.assertEqual(report["adaptation_device"], str(neuron.w.device))
        self.assertEqual(report["spike_times_device"], str(neuron.spike_times.device))
        self.assertFalse(report["cuda"])

    def test_state_dict_roundtrip_preserves_dynamics_state_on_device(self) -> None:
        neuron = AdExNeuron(n_neurons=4, dt=0.5, device="cpu")
        current = torch.full((4,), 25.0)
        for step in range(16):
            neuron.step(current, t=float(step) * neuron.dt)

        snapshot = neuron.state_dict()
        restored = AdExNeuron(n_neurons=4, dt=0.5, device="cpu")
        restored.load_state_dict(snapshot)

        self.assertEqual(snapshot["V"].device.type, "cpu")
        self.assertTrue(torch.allclose(restored.V, neuron.V))
        self.assertTrue(torch.allclose(restored.w, neuron.w))
        self.assertTrue(torch.allclose(restored.spike_times, neuron.spike_times))

    def test_step_resets_voltage_and_records_spike_time(self) -> None:
        neuron = AdExNeuron(n_neurons=4, dt=0.5, device="cpu")
        current = torch.full((4,), 25.0)

        last_t = 0.0
        spikes = torch.zeros(4, dtype=torch.bool)
        for step in range(400):
            last_t = float(step) * neuron.dt
            spikes = neuron.step(current, t=last_t)
            if bool(spikes.any().item()):
                break

        self.assertTrue(bool(spikes.any().item()))
        self.assertTrue(torch.allclose(neuron.V[spikes], torch.full_like(neuron.V[spikes], neuron.V_reset)))
        self.assertTrue(torch.allclose(neuron.spike_times[spikes], torch.full_like(neuron.spike_times[spikes], last_t)))
        self.assertGreater(float(neuron.w[spikes].min().item()), 0.0)

    def test_inhibitory_constructor_sets_fast_spiking_parameters(self) -> None:
        neuron = AdExNeuron.inhibitory(n_neurons=8, dt=0.5, device="cpu")

        self.assertEqual(neuron.C_m, 100e-3)
        self.assertEqual(neuron.V_T, -45.0)
        self.assertEqual(neuron.tau_w, 20.0)
        self.assertEqual(neuron.a, 0.0)
        self.assertEqual(neuron.b, 0.0)
        self.assertEqual(neuron.V_reset, -65.0)
        self.assertTrue(torch.allclose(neuron.V, torch.full_like(neuron.V, neuron.E_L)))

    def test_reset_state_restores_resting_values(self) -> None:
        neuron = AdExNeuron(n_neurons=3, dt=0.5, device="cpu")
        current = torch.full((3,), 25.0)
        for step in range(8):
            neuron.step(current, t=float(step) * neuron.dt)

        neuron.reset_state()

        self.assertTrue(torch.allclose(neuron.V, torch.full_like(neuron.V, neuron.E_L)))
        self.assertTrue(torch.allclose(neuron.w, torch.zeros_like(neuron.w)))
        self.assertTrue(torch.allclose(neuron.spike_times, torch.full_like(neuron.spike_times, -1.0)))

    def test_invalid_current_shape_raises(self) -> None:
        neuron = AdExNeuron(n_neurons=4, device="cpu")

        with self.assertRaises(ValueError):
            neuron.step(torch.ones(3), t=0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_cuda_device_report_and_load_state_restore_live_tensors_to_cuda(self) -> None:
        source = AdExNeuron(n_neurons=4, dt=0.5, device="cpu")
        source.step(torch.full((4,), 25.0), t=0.0)
        restored = AdExNeuron(n_neurons=4, dt=0.5, device="cuda")

        restored.load_state_dict(source.state_dict())
        report = restored.device_report()

        self.assertTrue(str(report["device"]).startswith("cuda"))
        self.assertTrue(str(report["voltage_device"]).startswith("cuda"))
        self.assertEqual(restored.V.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
