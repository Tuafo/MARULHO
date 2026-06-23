from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from marulho.config.model_config import MarulhoConfig
from marulho.service.trace_export_runner import build_arg_parser, export_runtime_trace_dataset, main
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _build_checkpoint(root: Path) -> Path:
    cfg = MarulhoConfig(
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
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
    return save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": "trace_export_runner"},
    )


class TraceExportRunnerTests(unittest.TestCase):
    def test_arg_parser_accepts_checkpoint_output_limit_and_endpoint(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--output",
                "trace.json",
                "--limit",
                "7",
                "--endpoint",
                "respond",
            ]
        )

        self.assertEqual(args.checkpoint, Path("checkpoint.pt"))
        self.assertEqual(args.output, Path("trace.json"))
        self.assertEqual(args.limit, 7)
        self.assertEqual(args.endpoint, "respond")

    def test_export_runtime_trace_dataset_is_trace_only_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(root)

            dataset = export_runtime_trace_dataset(
                checkpoint,
                limit=2,
                trace_dir=root / "traces",
                env_root=root / "env",
            )

        self.assertEqual(dataset["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertEqual(dataset["count"], 0)
        self.assertEqual(dataset["metadata"]["generated_by"], "marulho.service.trace_export_runner")
        self.assertNotIn("replay_plan_summary", dataset)
        self.assertNotIn("replay_sample_summary", dataset)
        self.assertNotIn("replay_dataset_summary", dataset)
        self.assertNotIn("replay_plan_summary", dataset["metadata"])
        self.assertNotIn("replay_sample_summary", dataset["metadata"])
        self.assertNotIn("replay_dataset_summary", dataset["metadata"])

    def test_main_can_write_json_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(root)
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    "-",
                    "--limit",
                    "1",
                    "--trace-dir",
                    str(root / "traces"),
                    "--env-root",
                    str(root / "env"),
                ],
                stdout=stdout,
            )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertNotIn("replay_dataset_summary", payload)


if __name__ == "__main__":
    unittest.main()
