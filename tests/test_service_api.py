from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from hecsn.config.model_config import HECSNConfig
from hecsn.service.api import create_app
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _build_checkpoint(root: Path, *, test_case: str) -> Path:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
    )
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    return save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )


class ServiceApiAcquisitionSurfaceTests(unittest.TestCase):
    def test_acquisition_presets_only_list_maintained_hf_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_presets"), trace_dir=root / "traces")
            with TestClient(app) as client:
                response = client.get("/acquisition/presets")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"presets": ["autonomy_acquisition_hf_allocation"]})

    def test_acquisition_run_rejects_exploratory_preset_in_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_preset_schema_guard"), trace_dir=root / "traces")
            with TestClient(app) as client:
                response = client.post(
                    "/acquisition/run",
                    json={"preset": "autonomy_acquisition_open_web_scout", "policy": "active"},
                )

            self.assertEqual(response.status_code, 422)
            self.assertIn("preset", response.text)

    def test_acquisition_run_rejects_scout_policy_in_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_policy_schema_guard"), trace_dir=root / "traces")
            with TestClient(app) as client:
                response = client.post(
                    "/acquisition/run",
                    json={"preset": "autonomy_acquisition_hf_allocation", "policy": "scout_commit"},
                )

            self.assertEqual(response.status_code, 422)
            self.assertIn("policy", response.text)


if __name__ == "__main__":
    unittest.main()
