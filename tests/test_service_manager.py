from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.config.model_config import HECSNConfig
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _build_manager(root: Path, *, test_case: str) -> HECSNServiceManager:
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
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return HECSNServiceManager(
        checkpoint_path,
        trace_dir=root / "traces",
    )


class ServiceManagerCheckpointTests(unittest.TestCase):
    def test_save_restore_round_trips_concept_store_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_checkpoint_roundtrip")
            manager.feed(text="river bank water current\nmoney bank credit loan\nriver reeds current bank\n")
            river_query = manager.query(query_text="river bank current", top_k_memories=6)
            manager.query(query_text="money bank loan", top_k_memories=6)

            self.assertIn("gap_plan", river_query)
            self.assertEqual(river_query["gap_plan"]["planner_mode"], "semantic_gap_planner")

            before = manager.status()["concept_store"]
            self.assertGreater(int(before["concept_count"]), 0)
            self.assertGreater(int(before["observations"]), 0)

            saved = manager.save_checkpoint(str(root / "service.pt"))
            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
            )

            after_status = restored.status()
            after = after_status["concept_store"]
            metadata = after_status["checkpoint_metadata"]

            self.assertEqual(int(after["concept_count"]), int(before["concept_count"]))
            self.assertEqual(int(after["observations"]), int(before["observations"]))
            self.assertEqual(
                sorted(entry["concept_id"] for entry in after.get("top_concepts", [])),
                sorted(entry["concept_id"] for entry in before.get("top_concepts", [])),
            )
            self.assertEqual(
                metadata["service_state"]["concept_store"]["concept_mode"],
                "slow_feature_concept_memory",
            )


class ServiceManagerAcquisitionSurfaceTests(unittest.TestCase):
    def test_acquire_rejects_exploratory_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _build_manager(Path(tmpdir), test_case="service_manager_acquisition_preset_guard")

            with self.assertRaisesRegex(ValueError, "Supported presets: autonomy_acquisition_hf_allocation"):
                manager.acquire(preset="autonomy_acquisition_open_web_scout")

    def test_acquire_rejects_scout_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = _build_manager(Path(tmpdir), test_case="service_manager_acquisition_policy_guard")

            with self.assertRaisesRegex(ValueError, "Supported policies: active, round_robin"):
                manager.acquire(preset="autonomy_acquisition_hf_allocation", policy="scout_commit")


if __name__ == "__main__":
    unittest.main()
