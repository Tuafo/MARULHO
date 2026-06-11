from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from marulho.training.checkpointing import _checkpoint_load_device, load_trainer_checkpoint, save_trainer_checkpoint


class CheckpointDevicePlacementTests(unittest.TestCase):
    def test_checkpoint_load_device_prefers_runtime_env(self) -> None:
        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            self.assertEqual(_checkpoint_load_device(), torch.device("cpu"))

    def test_checkpoint_restore_uses_runtime_device_not_hardcoded_cpu(self) -> None:
        captured: dict[str, object] = {}

        def fake_load(path: Path, *, map_location: object) -> dict[str, object]:
            captured["path"] = path
            captured["map_location"] = map_location
            raise RuntimeError("stop before trainer construction")

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
            with patch("marulho.training.checkpointing.torch.load", side_effect=fake_load):
                with self.assertRaisesRegex(RuntimeError, "stop before trainer construction"):
                    load_trainer_checkpoint(Path("checkpoint.pt"))

        self.assertEqual(captured["path"], Path("checkpoint.pt"))
        self.assertEqual(captured["map_location"], torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_checkpoint_restore_selects_cuda_when_available(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_checkpoint_load_device().type, "cuda")

    def test_checkpoint_save_failure_preserves_existing_file(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "checkpoint.pt"
            target.write_bytes(b"previous")
            trainer = SimpleNamespace(
                config=object(),
                encoder=SimpleNamespace(state_dict=lambda: {}),
                token_count=0,
                is_bootstrap=True,
                sleep_events=0,
                micro_sleep_events=0,
                deep_sleep_events=0,
                last_micro_sleep_token=0,
                last_deep_sleep_token=0,
                current_window_min_drift=0.0,
                previous_window_min_drift=None,
                recent_drifts=[],
                current_rolling_drift_floor=None,
                previous_rolling_drift_floor=None,
                last_floor_check_token=0,
                memory_warm_started=False,
                last_winner=None,
                pending_emergency_deep_sleep=False,
                last_network_reset_token=0,
                developmental_stage=0,
                _stage2_bootstrap_budget=0,
                _stage2_bootstrap_used_visual=0,
                _stage2_bootstrap_used_audio=0,
                column_anchors={},
            )

            with patch("marulho.training.checkpointing.asdict", return_value={}):
                with patch("marulho.training.checkpointing._model_snapshot", return_value={}):
                    with patch("marulho.training.checkpointing.torch.save", side_effect=RuntimeError("interrupted")):
                        with self.assertRaisesRegex(RuntimeError, "interrupted"):
                            save_trainer_checkpoint(target, trainer)

            self.assertEqual(target.read_bytes(), b"previous")
            self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_checkpoint_roundtrip_preserves_predictive_failure_streak(self) -> None:
        from tempfile import TemporaryDirectory

        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        with TemporaryDirectory() as tmpdir:
            cfg = MarulhoConfig(
                n_columns=8,
                column_latent_dim=4,
                bootstrap_tokens=0,
                memory_capacity=16,
            )
            trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
            trainer.model.predictive.prediction_failure_streak[:] = torch.arange(
                8,
                dtype=torch.int32,
                device=trainer.model.device,
            )
            checkpoint = save_trainer_checkpoint(Path(tmpdir) / "predictive.pt", trainer)

            with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}, clear=False):
                restored, _metadata = load_trainer_checkpoint(checkpoint)

            self.assertTrue(
                torch.equal(
                    restored.model.predictive.prediction_failure_streak.cpu(),
                    torch.arange(8, dtype=torch.int32),
                )
            )


if __name__ == "__main__":
    unittest.main()
