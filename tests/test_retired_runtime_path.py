from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.config.model_config import HECSNConfig
from hecsn.service.retired_runtime_path import RETIRED_RUNTIME_PATH_STATE_FIELDS, RetiredRuntimePathState
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_manager(root: Path, *, test_case: str) -> HECSNServiceManager:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")


class RetiredRuntimePathStateTests(unittest.TestCase):
    def test_manager_routes_retired_runtime_path_state_to_holder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="retired_runtime_path_state_ownership")
            try:
                for field_name in RETIRED_RUNTIME_PATH_STATE_FIELDS:
                    with self.subTest(field_name=field_name):
                        self.assertNotIn(field_name, manager.__dict__)
                        self.assertIn(field_name, manager._retired_runtime_path_state.__dict__)
            finally:
                manager.close()

    def test_retired_runtime_path_snapshot_is_unavailable_and_retired(self) -> None:
        controller = RetiredRuntimePathState()

        snapshot = controller._retired_runtime_path_unavailable_snapshot()

        self.assertFalse(snapshot["enabled"])
        self.assertTrue(snapshot["retired"])
        self.assertEqual(snapshot["reason"], "retired_llm_path")
        self.assertEqual(snapshot["replacement"], "subcortex_living_loop")
        self.assertTrue(snapshot["initialization"]["finished"])
        self.assertIn("retired", snapshot["initialization"]["error"])
        self.assertFalse(hasattr(controller, "runtime_snapshot"))
        self.assertFalse(hasattr(controller, "ask"))
        self.assertFalse(hasattr(controller, "sleep"))
        self.assertFalse(hasattr(controller, "thoughts"))

    def test_no_thought_loop_slot_exists_to_revive_retired_path(self) -> None:
        controller = RetiredRuntimePathState()

        self.assertFalse(hasattr(controller, "_thought_loop"))
        self.assertFalse(hasattr(controller, "_thought_loop_actual"))
        self.assertFalse(controller._retired_runtime_path_available)
        snapshot = controller._retired_runtime_path_unavailable_snapshot()
        self.assertFalse(snapshot["initialization"]["started"])
        self.assertTrue(snapshot["initialization"]["finished"])


if __name__ == "__main__":
    unittest.main()
