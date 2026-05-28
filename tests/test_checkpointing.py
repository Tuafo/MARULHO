from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from hecsn.training.checkpointing import _checkpoint_load_device, load_trainer_checkpoint


class CheckpointDevicePlacementTests(unittest.TestCase):
    def test_checkpoint_load_device_prefers_runtime_env(self) -> None:
        with patch.dict("os.environ", {"HECSN_DEVICE": "cpu"}, clear=False):
            self.assertEqual(_checkpoint_load_device(), torch.device("cpu"))

    def test_checkpoint_restore_uses_runtime_device_not_hardcoded_cpu(self) -> None:
        captured: dict[str, object] = {}

        def fake_load(path: Path, *, map_location: object) -> dict[str, object]:
            captured["path"] = path
            captured["map_location"] = map_location
            raise RuntimeError("stop before trainer construction")

        with patch.dict("os.environ", {"HECSN_DEVICE": "cpu"}, clear=False):
            with patch("hecsn.training.checkpointing.torch.load", side_effect=fake_load):
                with self.assertRaisesRegex(RuntimeError, "stop before trainer construction"):
                    load_trainer_checkpoint(Path("checkpoint.pt"))

        self.assertEqual(captured["path"], Path("checkpoint.pt"))
        self.assertEqual(captured["map_location"], torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_checkpoint_restore_selects_cuda_when_available(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_checkpoint_load_device().type, "cuda")


if __name__ == "__main__":
    unittest.main()
