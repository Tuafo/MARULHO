from __future__ import annotations

import unittest

import torch

from marulho.core.surprise import SurpriseMonitor


class SurpriseMonitorTests(unittest.TestCase):
    def test_update_neuromodulators_separates_positive_and_negative_valence(self) -> None:
        positive = SurpriseMonitor(layer_names=["competitive"])
        positive.predicted_error = 0.6
        positive.update_neuromodulators(current_error=0.1, novelty=0.2)

        negative = SurpriseMonitor(layer_names=["competitive"])
        negative.predicted_error = 0.2
        negative.update_neuromodulators(current_error=0.8, novelty=0.2)

        self.assertGreater(positive.dopamine, positive.serotonin)
        self.assertGreater(negative.serotonin, negative.dopamine)

    def test_update_neuromodulators_separates_expected_and_unexpected_uncertainty(self) -> None:
        expected = SurpriseMonitor(layer_names=["competitive"])
        expected.predicted_error = 0.4
        expected.update_neuromodulators(current_error=0.4, novelty=1.0)

        unexpected = SurpriseMonitor(layer_names=["competitive"])
        unexpected.predicted_error = 0.1
        unexpected.update_neuromodulators(current_error=0.9, novelty=0.0)

        self.assertGreater(expected.acetylcholine, expected.norepinephrine)
        self.assertGreater(unexpected.norepinephrine, unexpected.acetylcholine)

    def test_get_modulator_uses_serotonin_to_flip_learning_bias(self) -> None:
        monitor = SurpriseMonitor(layer_names=["competitive"])
        layer = monitor.layers["competitive"]
        layer["errors"].extend([0.1] * 9 + [0.9])
        layer["precision"] = 20.0

        monitor.dopamine = 0.8
        monitor.serotonin = 0.2
        monitor.acetylcholine = 0.8
        monitor.norepinephrine = 0.8
        positive_modulator = monitor.get_modulator("competitive")

        monitor.dopamine = 0.2
        monitor.serotonin = 0.8
        negative_modulator = monitor.get_modulator("competitive")

        self.assertGreater(positive_modulator, 0.0)
        self.assertLess(negative_modulator, 0.0)

    def test_record_error_matches_tensor_update_statistics(self) -> None:
        direct = SurpriseMonitor(layer_names=["competitive"])
        tensor = SurpriseMonitor(layer_names=["competitive"])
        for index in range(12):
            error = torch.tensor([0.1 + index * 0.01])
            direct.record_error("competitive", error.item())
            tensor.update(
                "competitive",
                error,
                torch.tensor([0.0]),
            )

        self.assertEqual(
            list(direct.layers["competitive"]["errors"]),
            list(tensor.layers["competitive"]["errors"]),
        )
        self.assertEqual(
            direct.layers["competitive"]["precision"],
            tensor.layers["competitive"]["precision"],
        )

    def test_modulator_revision_tracks_error_and_neuromodulator_changes(self) -> None:
        monitor = SurpriseMonitor(layer_names=["competitive"])
        initial = monitor.modulator_revision

        monitor.record_error("competitive", 0.2)
        self.assertEqual(monitor.modulator_revision, initial + 1)

        monitor.update_neuromodulators(current_error=0.3, novelty=0.1)
        self.assertEqual(monitor.modulator_revision, initial + 2)

        monitor.mark_modulator_state_changed()
        self.assertEqual(monitor.modulator_revision, initial + 3)


if __name__ == "__main__":
    unittest.main()
