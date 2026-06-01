from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import socket
import tempfile
from threading import Event
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.data.corpus_loader import BackgroundPrefetchIterator
from hecsn.service import brain_runtime as brain_runtime_module
from hecsn.service import delayed_consequence as delayed_consequence_module
from hecsn.service import runtime_prewarm as runtime_prewarm_module
from hecsn.service import sensory_runtime as sensory_runtime_module
from hecsn.service import terminus_sensory as sensory_module
from hecsn.service.operator_interaction import (
    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    REQUEST_FEED_ENCODING_MODE,
)
from hecsn.service.manager import HECSNServiceManager
from hecsn.service.runtime_sources import RuntimeSources, _BrainSourceRuntime, _SensorySourceRuntime
from hecsn.service.terminus_sensory import SensoryEpisode
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.model import HECSNModel
from hecsn.training.trainer import HECSNTrainer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SilentSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - suppress test noise
        return None


class _EchoJsonApiHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - suppress test noise
        return None

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        payload = self.rfile.read(content_length) if content_length > 0 else b""
        parsed_body = json.loads(payload.decode("utf-8") or "null")
        parsed_url = urlparse(self.path)
        response = {
            "method": "POST",
            "path": parsed_url.path,
            "query": {key: values[0] if len(values) == 1 else values for key, values in parse_qs(parsed_url.query).items()},
            "payload": parsed_body,
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _png_bytes() -> bytes:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[8:24, 8:24] = 255
    buffer = io.BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _build_manager(root: Path, *, test_case: str, env_root: Path | None = None) -> HECSNServiceManager:
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
    return HECSNServiceManager(
        checkpoint_path,
        trace_dir=root / "traces",
        env_root=env_root,
    )


@contextmanager
def _serve_directory(root: Path) -> Iterator[str]:
    port = _free_port()
    handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _write_query_gap_notes(root: Path) -> None:
    (root / "notes.md").write_text(
        "Cats rest indoors during the day.\nCats chase mice at night.\n",
        encoding="utf-8",
    )


def _write_query_gap_page(root: Path) -> None:
    (root / "page.html").write_text(
        "<html><body><main><p>Cats chase mice at night.</p><p>Cats rest indoors during the day.</p></main></body></html>",
        encoding="utf-8",
    )


def _write_query_gap_data(root: Path) -> None:
    (root / "data.json").write_text(
        '{"facts": {"chase": "mice at night", "rest": "indoors during the day"}}',
        encoding="utf-8",
    )


@dataclass(frozen=True)
class _RespondActionAssistCase:
    label: str
    test_case: str
    query_text: str
    reason: str
    expected_response_fragment: str
    setup: Callable[[Path], None]
    expected_action_type: str | None = None
    expected_input_path: str | None = None
    served_path: str | None = None


class ServiceManagerBootstrapTests(unittest.TestCase):
    def test_manager_loads_runtime_env_from_checkpoint_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            env_path = root / ".env"
            env_path.write_text("HF_TOKEN=dotenv-checkpoint-token\n", encoding="utf-8")

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
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                checkpoint_dir / "initial.pt",
                trainer,
                metadata={"test_case": "bootstrap_checkpoint_ancestry"},
            )

            old_token = os.environ.pop("HF_TOKEN", None)
            try:
                manager = HECSNServiceManager(checkpoint_path, trace_dir=root / "traces")
                try:
                    env_info = manager.runtime_facade.status()["terminus_runtime"]["environment"]
                    self.assertTrue(env_info["dotenv_available"])
                    self.assertTrue(env_info["dotenv_loaded"])
                    self.assertEqual(Path(str(env_info["dotenv_path"])).resolve(), env_path.resolve())
                    self.assertEqual(os.environ.get("HF_TOKEN"), "dotenv-checkpoint-token")
                    self.assertTrue(env_info["hf_token_present"])
                    self.assertFalse(hasattr(manager, "_retired_runtime_path_available"))
                    self.assertFalse(hasattr(manager, "_thought_loop_actual"))
                finally:
                    manager.close()
            finally:
                if old_token is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = old_token

    def test_manager_loads_runtime_env_from_explicit_env_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_root = root / "project_root"
            env_root.mkdir(parents=True, exist_ok=True)
            env_path = env_root / ".env"
            env_path.write_text("HF_TOKEN=dotenv-explicit-root\n", encoding="utf-8")
            checkpoint_dir = root / "external_checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                checkpoint_dir / "initial.pt",
                trainer,
                metadata={"test_case": "bootstrap_env_root"},
            )

            old_token = os.environ.pop("HF_TOKEN", None)
            try:
                manager = HECSNServiceManager(
                    checkpoint_path,
                    trace_dir=root / "traces",
                    env_root=env_root,
                )
                try:
                    env_info = manager.runtime_facade.status()["terminus_runtime"]["environment"]
                    self.assertTrue(env_info["dotenv_available"])
                    self.assertTrue(env_info["dotenv_loaded"])
                    self.assertEqual(Path(str(env_info["dotenv_path"])).resolve(), env_path.resolve())
                    self.assertEqual(Path(str(env_info["env_root"])).resolve(), env_root.resolve())
                    self.assertEqual(os.environ.get("HF_TOKEN"), "dotenv-explicit-root")
                    self.assertTrue(env_info["hf_token_present"])
                    self.assertFalse(hasattr(manager, "_retired_runtime_path_available"))
                    self.assertFalse(hasattr(manager, "_thought_loop_actual"))
                finally:
                    manager.close()
            finally:
                if old_token is None:
                    os.environ.pop("HF_TOKEN", None)
                else:
                    os.environ["HF_TOKEN"] = old_token


class ServiceManagerCheckpointTests(unittest.TestCase):
    def test_feed_defers_due_deep_sleep_maintenance_for_request_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_feed_defers_sleep")
            try:
                trainer = manager._trainer
                trainer.token_count = trainer.config.deep_sleep_interval_tokens
                with patch.object(
                    trainer,
                    "_sleep_replay",
                    side_effect=AssertionError("feed must not synchronously run sleep maintenance"),
                ) as sleep_replay:
                    result = manager.runtime_facade.feed(text="x")

                self.assertEqual(sleep_replay.call_count, 0)
                self.assertEqual(result["feed_summary"]["tokens_processed"], 1)
                self.assertFalse(result["feed_summary"]["sleep_maintenance_allowed"])
                self.assertEqual(result["feed_summary"]["sleep_maintenance_deferred"], 1)
            finally:
                manager.close()

    def test_feed_trace_state_snapshot_skips_replay_dataset_preview_for_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_feed_light_state_snapshot")
            try:
                with patch.object(
                    manager,
                    "_replay_dataset_preview_summary_locked",
                    side_effect=AssertionError("feed traces must not build replay dataset previews"),
                ) as preview:
                    result = manager.runtime_facade.feed(text="Cats chase mice. Cats rest indoors.")

                self.assertEqual(preview.call_count, 0)
                self.assertEqual(result["runtime_episode"]["operation"], "feed")

                status = manager.runtime_facade.living_loop_status()
                self.assertIn("replay_dataset_summary", status["living_loop"])
            finally:
                manager.close()

    def test_feed_samples_runtime_concept_observation_for_request_latency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_feed_concept_sampling")
            try:
                text = ("submarine ballast buoyancy pressure control " * 16).strip()
                with patch.object(
                    manager._interaction_pipeline,
                    "_observe_runtime_concepts_fn",
                    wraps=manager._interaction_pipeline._observe_runtime_concepts_fn,
                ) as observe:
                    result = manager.runtime_facade.feed(text=text)

                summary = result["feed_summary"]
                tokens_processed = int(summary["tokens_processed"])
                concept_observations = int(summary["concept_observations"])
                concept_store = manager.runtime_facade.status()["concept_store"]

                self.assertGreater(tokens_processed, DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL)
                self.assertEqual(summary["feed_encoding_mode"], REQUEST_FEED_ENCODING_MODE)
                self.assertEqual(summary["concept_observation_mode"], "sampled")
                self.assertEqual(
                    int(summary["concept_observation_interval"]),
                    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
                )
                self.assertEqual(observe.call_count, concept_observations)
                self.assertGreaterEqual(concept_observations, 2)
                self.assertLess(concept_observations, tokens_processed)
                self.assertGreater(int(concept_store["concept_count"]), 0)
                self.assertGreater(int(concept_store["observations"]), 0)
            finally:
                manager.close()

    def test_save_restore_round_trips_concept_store_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_checkpoint_roundtrip")
            try:
                manager.runtime_facade.feed(text="river bank water current\nmoney bank credit loan\nriver reeds current bank\n")
                fed = manager.runtime_facade.status()["concept_store"]
                self.assertGreater(int(fed["concept_count"]), 0)
                self.assertGreater(int(fed["observations"]), 0)
                river_query = manager.runtime_facade.query(query_text="river bank current", top_k_memories=6)
                manager.runtime_facade.query(query_text="money bank loan", top_k_memories=6)

                self.assertIn("gap_plan", river_query)
                self.assertEqual(river_query["gap_plan"]["planner_mode"], "semantic_gap_planner")

                before_status = manager.runtime_facade.status()
                before = before_status["concept_store"]
                before_serotonin = float(before_status["serotonin"])
                self.assertGreater(int(before["concept_count"]), 0)
                self.assertGreater(int(before["observations"]), 0)

                saved = manager.runtime_facade.save_checkpoint(str(root / "service.pt"))
                restored = HECSNServiceManager(
                    saved["path"],
                    trace_dir=root / "restored_traces",
                )
                try:
                    after_status = restored.runtime_facade.status()
                    after = after_status["concept_store"]
                    metadata = after_status["checkpoint_metadata"]

                    self.assertEqual(int(after["concept_count"]), int(before["concept_count"]))
                    self.assertEqual(int(after["observations"]), int(before["observations"]))
                    self.assertEqual(
                        sorted(entry["concept_id"] for entry in after.get("top_concepts", [])),
                        sorted(entry["concept_id"] for entry in before.get("top_concepts", [])),
                    )
                    self.assertIn("serotonin", after_status)
                    self.assertAlmostEqual(float(after_status["serotonin"]), before_serotonin, places=6)
                    self.assertEqual(
                        metadata["service_state"]["concept_store"]["concept_mode"],
                        "slow_feature_concept_memory",
                    )
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_restore_checkpoint_marks_clean_and_increments_revision_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_restore_checkpoint_revision")
            try:
                manager.runtime_facade.feed(text="river bank water current\nmoney bank credit loan\n")
                manager.runtime_facade.query(query_text="river bank current", top_k_memories=6)
                revision_before_restore = manager.runtime_facade.status()["state_revision"]
                revision_after_restore = revision_before_restore + 1

                saved = manager.runtime_facade.save_checkpoint(str(root / "service.pt"))
                self.assertFalse(saved["dirty_state"])
                self.assertEqual(saved["state_revision"], revision_before_restore)

                restored = manager.runtime_facade.restore_checkpoint(saved["path"])
                after_restore = manager.runtime_facade.status()

                self.assertFalse(restored["dirty_state"])
                self.assertEqual(restored["state_revision"], revision_after_restore)
                self.assertFalse(after_restore["dirty_state"])
                self.assertEqual(after_restore["state_revision"], revision_after_restore)
            finally:
                manager.close()

    def test_restore_checkpoint_rebinds_concrete_runtime_collaborators(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_restore_rebind")
            try:
                external_root = root / "external"
                external_root.mkdir()
                saved = manager.runtime_facade.save_checkpoint(str(external_root / "service.pt"))
                saved_token_count = int(saved["token_count"])
                previous_trainer = manager._trainer
                manager.runtime_facade.status()
                self.assertIsNotNone(manager._status_read_model._cached_status)
                manager.runtime_facade.feed(text="restored runtime collaborator evidence")
                self.assertGreater(int(manager._trainer.token_count), saved_token_count)

                with (
                    patch.object(manager._brain_runtime, "restore_runtime_state", wraps=manager._brain_runtime.restore_runtime_state) as restore_brain,
                    patch.object(manager._delayed_consequence, "restore_state", wraps=manager._delayed_consequence.restore_state) as restore_delayed,
                ):
                    restored = manager.runtime_facade.restore_checkpoint(saved["path"])

                self.assertIsNot(manager._trainer, previous_trainer)
                self.assertGreaterEqual(restore_brain.call_count, 1)
                self.assertGreaterEqual(restore_delayed.call_count, 1)
                self.assertIs(manager._brain_runtime._trainer, manager._trainer)
                self.assertIs(manager._brain_runtime._encoder, manager._encoder)
                self.assertIs(manager._interaction_pipeline._trainer, manager._trainer)
                self.assertIs(manager._interaction_pipeline._encoder, manager._encoder)
                self.assertIs(manager._status_read_model._trainer, manager._trainer)
                self.assertIsNone(manager._status_read_model._cached_status)
                self.assertEqual(manager.runtime_facade.status()["token_count"], saved_token_count)
                self.assertEqual(manager.runtime_facade.status()["checkpoint_path"], restored["path"])
                self.assertIn("objects", Path(restored["path"]).parts)
                self.assertEqual(manager._action_executor.action_history()["root_path"], str(external_root.resolve()))

                manager.runtime_facade.feed(text="post restore trainer evidence")
                self.assertGreater(int(manager._trainer.token_count), saved_token_count)
                self.assertEqual(int(manager._interaction_pipeline._trainer.token_count), int(manager._trainer.token_count))
            finally:
                manager.close()

    def test_restart_after_operator_restore_resolves_new_published_brain_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bootstrap_path = root / "initial.pt"
            manager = _build_manager(root, test_case="service_manager_restore_restart")
            try:
                saved = manager.runtime_facade.save_checkpoint(str(root / "saved.pt"))
                saved_token_count = int(saved["token_count"])
                manager.runtime_facade.feed(text="discarded after operator restore")
                restored = manager.runtime_facade.restore_checkpoint(saved["path"])
                restored_revision = int(restored["state_revision"])
            finally:
                manager.close()

            restarted = HECSNServiceManager(bootstrap_path, trace_dir=root / "restart_traces")
            try:
                status = restarted.runtime_facade.status()
                self.assertEqual(status["checkpoint_path"], restored["path"])
                self.assertEqual(int(status["token_count"]), saved_token_count)
                self.assertEqual(int(status["state_revision"]), restored_revision)
                saved_after_restart = restarted.runtime_facade.save_checkpoint()
                saved_path = Path(saved_after_restart["current_checkpoint_manifest"]["checkpoint_path"])
                saved_parts = saved_path.parts
                self.assertIn("objects", saved_parts)
                self.assertNotIn("objects\\objects", str(saved_path))
            finally:
                restarted.close()

    def test_operator_restore_publication_failure_recovers_previous_live_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_restore_publish_failure")
            try:
                saved = manager.runtime_facade.save_checkpoint(str(root / "saved.pt"))
                manager.runtime_facade.feed(text="runtime state retained when restore publication fails")
                before = manager.runtime_facade.status()

                with patch.object(
                    manager._runtime_persistence,
                    "_write_atomic_json",
                    side_effect=RuntimeError("interrupted operator restore publication"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "interrupted operator restore publication"):
                        manager.runtime_facade.restore_checkpoint(saved["path"])

                after = manager.runtime_facade.status()
                self.assertEqual(after["checkpoint_path"], before["checkpoint_path"])
                self.assertEqual(after["token_count"], before["token_count"])
                self.assertEqual(after["state_revision"], before["state_revision"])
                self.assertEqual(after["dirty_state"], before["dirty_state"])
                self.assertIs(manager._brain_runtime._trainer, manager._trainer)
                self.assertIs(manager._interaction_pipeline._trainer, manager._trainer)
                self.assertIs(manager._status_read_model._trainer, manager._trainer)
            finally:
                manager.close()

    def test_published_save_refreshes_status_checkpoint_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_publish_refresh")
            try:
                manager.runtime_facade.status()
                saved = manager.runtime_facade.save_checkpoint(str(root / "service.pt"))
                published_path = saved["current_checkpoint_manifest"]["checkpoint_path"]
                status = manager.runtime_facade.status()

                self.assertEqual(status["checkpoint_path"], published_path)
                self.assertEqual(status["checkpoint_metadata"]["saved_by"], "hecsn.service")
            finally:
                manager.close()

    def test_quick_start_rebuild_refreshes_runtime_collaborators(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_quick_start_refresh")
            try:
                rebuilt_encoder = object()

                class _RebuiltTrainer:
                    def __init__(self, _model: object, config: object) -> None:
                        self.config = config
                        self.encoder = rebuilt_encoder
                        self.token_count = 0

                with (
                    patch("hecsn.service.runtime_control.HECSNModel", return_value=object()),
                    patch("hecsn.service.runtime_control.HECSNTrainer", _RebuiltTrainer),
                    patch.object(manager._runtime_control, "configure_terminus", return_value={}),
                    patch.object(manager._runtime_control, "start_terminus", return_value={}),
                    patch.object(manager, "_refresh_root_captures_locked", wraps=manager._refresh_root_captures_locked) as refresh,
                ):
                    result = manager._runtime_control.quick_start_terminus(preset="curriculum")

                self.assertEqual(refresh.call_count, 1)
                self.assertEqual(result["preset_applied"], "curriculum")
                self.assertIs(manager._brain_runtime._trainer, manager._trainer)
                self.assertIs(manager._brain_runtime._encoder, rebuilt_encoder)
                self.assertIs(manager._interaction_pipeline._trainer, manager._trainer)
                self.assertIs(manager._interaction_pipeline._encoder, rebuilt_encoder)
                self.assertIs(manager._status_read_model._trainer, manager._trainer)
            finally:
                manager.close()

    def test_save_restore_round_trips_replay_history_and_trace_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_replay_history_roundtrip")
            try:
                manager.runtime_facade.feed(text="Cats chase mice at night. Cats rest indoors during the day.")
                replay_record = manager.runtime_facade.replay_sample(
                    mode="sample",
                    operator_id="operator-1",
                    confirmation=True,
                )
                self.assertEqual(replay_record["status"], "recorded")
                self.assertEqual(manager.runtime_facade.replay_sample_history(limit=10)["count"], 1)
                self.assertGreaterEqual(len(manager.runtime_facade.recent_traces(limit=10)), 1)

                saved = manager.runtime_facade.save_checkpoint(str(root / "service.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "traces")
                try:
                    replay_history = restored.runtime_facade.replay_sample_history(limit=10)
                    traces = restored.runtime_facade.recent_traces(limit=10)

                    self.assertEqual(replay_history["count"], 1)
                    self.assertEqual(replay_history["history"][0]["replay_sample_id"], replay_record["replay_sample_id"])
                    self.assertGreaterEqual(len(traces), 1)
                    self.assertTrue(any(trace.get("operation") == "feed" for trace in traces))
                finally:
                    restored.close()
            finally:
                manager.close()


class ServiceManagerInteractionPipelineDelegationTests(unittest.TestCase):
    def test_query_feed_and_respond_delegate_to_interaction_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_interaction_pipeline_delegation")
            try:
                calls: list[tuple[str, dict[str, object]]] = []

                class _FakeInteractionPipeline:
                    def query(self, **kwargs: object) -> dict[str, object]:
                        calls.append(("query", dict(kwargs)))
                        return {"operation": "query"}

                    def feed(self, **kwargs: object) -> dict[str, object]:
                        calls.append(("feed", dict(kwargs)))
                        return {"operation": "feed"}

                    def respond(self, **kwargs: object) -> dict[str, object]:
                        calls.append(("respond", dict(kwargs)))
                        return {"operation": "respond"}

                manager._interaction_pipeline = _FakeInteractionPipeline()

                self.assertEqual(manager.runtime_facade.query(query_text="cats"), {"operation": "query"})
                self.assertEqual(manager.runtime_facade.feed(text="cats chase mice"), {"operation": "feed"})
                self.assertEqual(manager.runtime_facade.respond(query_text="cats"), {"operation": "respond"})
                self.assertEqual([call[0] for call in calls], ["query", "feed", "respond"])
                self.assertEqual(calls[0][1]["query_text"], "cats")
                self.assertEqual(calls[1][1]["text"], "cats chase mice")
                self.assertEqual(calls[2][1]["query_text"], "cats")
            finally:
                manager.close()


class ServiceManagerTerminusRuntimeTests(unittest.TestCase):
    def test_record_runtime_feedback_updates_and_persists_runtime_episode_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_runtime_feedback_episode")
            try:
                feed_result = manager.runtime_facade.feed(text="Cats chase mice at night. Cats rest indoors during the day.")
                episode_id = feed_result["runtime_episode"]["episode_id"]
                feedback = manager.runtime_facade.record_runtime_feedback(
                    {
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "contradicted",
                        "confidence": 0.82,
                        "summary": "Operator corrected the feed trace outcome.",
                        "corrected_output": {"summary": "The feed text mentioned cats and mice."},
                        "evidence": [{"note": "manual review", "api_key": "must be stripped"}],
                        "tags": ["Manual", "manual", "Runtime"],
                        "evaluator_id": " operator-1 ",
                    }
                )

                self.assertTrue(feedback["accepted"])
                self.assertTrue(feedback["dirty_state"])
                self.assertEqual(feedback["target"]["verification"]["status"], "contradicted")
                self.assertEqual(feedback["target"]["verification"]["provenance"], "contradicted")
                self.assertEqual(feedback["target"]["provenance"], "contradicted")
                self.assertEqual(feedback["target"]["feedback"][0]["tags"], ["manual", "runtime"])
                self.assertEqual(feedback["target"]["feedback"][0]["evaluator_id"], "operator-1")
                self.assertNotIn("api_key", feedback["target"]["feedback"][0]["evidence"][0])

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()

            restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
            try:
                restored_episode = list(restored._runtime_episode_traces)[0]
                self.assertEqual(restored_episode["episode_id"], episode_id)
                self.assertEqual(restored_episode["verification"]["status"], "contradicted")
                self.assertEqual(restored_episode["feedback"][0]["summary"], "Operator corrected the feed trace outcome.")
                self.assertEqual(restored_episode["corrected_output"]["summary"], "The feed text mentioned cats and mice.")
            finally:
                restored.close()

    def test_record_runtime_feedback_uses_runtime_state_for_mutation_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_runtime_feedback_runtime_state")
            try:
                feed_result = manager.runtime_facade.feed(text="Cats chase mice at night. Cats rest indoors during the day.")
                episode_id = feed_result["runtime_episode"]["episode_id"]
                expected_state_revision = int(manager._runtime_state.state_revision) + 1

                feedback = manager.runtime_facade.record_runtime_feedback(
                    {
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "contradicted",
                        "confidence": 0.82,
                        "summary": "Operator corrected the feed trace outcome.",
                    }
                )

                self.assertTrue(feedback["accepted"])
                self.assertTrue(feedback["dirty_state"])
                self.assertEqual(feedback["state_revision"], expected_state_revision)
                self.assertEqual(feedback["terminus_runtime"]["last_event"]["type"], "runtime_feedback_recorded")
                self.assertEqual(feedback["terminus_runtime"]["recent_events"][0]["type"], "runtime_feedback_recorded")
            finally:
                manager.close()

    def test_startup_hydrates_saved_revision_without_extra_increment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_startup_revision_hydration")
            try:
                manager.runtime_facade.feed(text="river bank water current\n")
                saved = manager.runtime_facade.save_checkpoint(str(root / "service.pt"))
                expected_revision = saved["state_revision"]
            finally:
                manager.close()

            restored = HECSNServiceManager(root / "initial.pt", trace_dir=root / "restored_traces")
            try:
                self.assertFalse(restored.runtime_facade.status()["dirty_state"])
                self.assertEqual(restored.runtime_facade.status()["state_revision"], expected_revision)
            finally:
                restored.close()

    def test_runtime_facade_acquire_uses_operator_interaction_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_facade_acquire_operator_runtime")
            try:
                captured: dict[str, object] = {}

                def _fake_run_live_acquisition(**kwargs: object) -> dict[str, object]:
                    captured.update(kwargs)
                    callback = kwargs.get("on_train_step")
                    if callable(callback):
                        callback("facade acquire concept", {"memory_index": 0})
                    return {
                        "tokens_trained_total": 3,
                        "candidate_results": [],
                        "policy_name": kwargs.get("policy_name"),
                    }

                checkpoint_path = root / "facade_acquire.pt"
                with patch("hecsn.service.operator_interaction.run_live_acquisition", side_effect=_fake_run_live_acquisition):
                    result = manager.runtime_facade.acquire(
                        acquisition_slots=1,
                        acquisition_tokens=3,
                        save_checkpoint_path=str(checkpoint_path),
                    )

                self.assertEqual(result["preset"], "autonomy_acquisition_hf_allocation")
                self.assertEqual(result["policy"], "active")
                self.assertEqual(result["acquisition_result"]["tokens_trained_total"], 3)
                self.assertFalse(result["dirty_state"])
                self.assertEqual(result["checkpoint_save"]["path"], str(checkpoint_path))
                self.assertTrue(Path(result["trace_path"]).exists())
                self.assertIs(captured["trainer"], manager._trainer)
                self.assertIs(captured["encoder"], manager._encoder)
                self.assertEqual(captured["policy_name"], "active")
                self.assertEqual(captured["acquisition_tokens"], 3)
                self.assertEqual(captured["acquisition_slots"], 1)
                self.assertTrue(callable(captured["on_train_step"]))
            finally:
                manager.close()

    def test_terminus_tick_trains_from_configured_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_tick")
            source_path = root / "terminus_source.txt"
            source_path.write_text("adaptive memory plasticity signal " * 32, encoding="utf-8")
            try:
                configured = manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "local_terminus_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=24,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                before_tokens = configured["token_count"]
                ticked = manager.runtime_facade.terminus_tick(steps=2)
                runtime = ticked["terminus_runtime"]

                self.assertTrue(runtime["configured"])
                self.assertFalse(runtime["running"])
                self.assertGreater(ticked["token_count"], before_tokens)
                self.assertGreater(runtime["background_tokens_processed"], 0)
                self.assertEqual(runtime["source_count"], 1)
                self.assertEqual(runtime["exhausted_source_count"], 0)
                self.assertEqual(runtime["next_source_name"], "local_terminus_source")
                self.assertIsNotNone(runtime["last_tick_completed_at"])
                self.assertGreater(float(runtime["last_tick_duration_ms"]), 0.0)
                self.assertGreater(int(runtime["last_tick_token_delta"]), 0)
                self.assertTrue(any(event.get("type") == "tick" for event in runtime["recent_events"]))
                self.assertEqual(runtime["last_event"]["type"], "tick")
                self.assertEqual(runtime["last_event"]["source"]["source_name"], "local_terminus_source")
                self.assertEqual(runtime["source_progress"][0]["name"], "local_terminus_source")
                self.assertGreater(runtime["source_progress"][0]["tokens_processed"], 0)
                self.assertGreater(runtime["source_progress"][0]["tick_visits"], 0)
                self.assertGreater(runtime["source_progress"][0]["last_tokens_trained"], 0)
                self.assertIsNotNone(runtime["source_progress"][0]["last_activity_at"])
                self.assertAlmostEqual(runtime["source_progress"][0]["share_of_background_tokens"], 1.0, places=6)
                concept_store = manager.runtime_facade.status()["concept_store"]
                self.assertGreater(int(concept_store["concept_count"]), 0)
                self.assertGreater(int(concept_store["observations"]), 0)
                top_terms = {
                    term
                    for concept in concept_store["top_concepts"]
                    for term in concept.get("top_terms", [])
                }
                self.assertIn("plasticity", top_terms)
            finally:
                manager.close()

    def test_terminus_tick_trains_from_hf_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_hf_source")
            try:
                fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
                fake_stream = iter(("adaptive memory plasticity", fake_pattern) for _ in range(64))
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=fake_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "fineweb_edu",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                                "hf_config": "sample-10BT",
                            }
                        ],
                        tick_tokens=24,
                        sleep_interval_seconds=0.01,
                        repeat_sources=True,
                    )
                before_tokens = configured["token_count"]
                ticked = manager.runtime_facade.terminus_tick(steps=2)
                runtime = ticked["terminus_runtime"]

                self.assertGreater(ticked["token_count"], before_tokens)
                self.assertEqual(runtime["source_count"], 1)
                self.assertEqual(runtime["source_progress"][0]["name"], "fineweb_edu")
                self.assertGreater(runtime["source_progress"][0]["tokens_processed"], 0)
                self.assertEqual(runtime["last_event"]["source"]["source_name"], "fineweb_edu")
                self.assertEqual(runtime["huggingface"]["source_count"], 1)
            finally:
                manager.close()

    def test_focus_aware_background_routing_prefers_aligned_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_focus_aware_background_routing")
            garden_path = root / "garden.txt"
            tectonics_path = root / "tectonics.txt"
            garden_path.write_text("tomatoes need watering and healthy soil " * 24, encoding="utf-8")
            tectonics_path.write_text("crust plates move over the mantle and form mountains " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "garden_source",
                            "source": str(garden_path),
                            "source_type": "file",
                            "metadata": {"label": "garden soil sunlight watering"},
                        },
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        },
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                manager.runtime_facade.query(query_text="How do crust plates move over the mantle?", top_k_memories=6)
                runtime = manager.runtime_facade.terminus_tick()["terminus_runtime"]
                garden_progress = next(item for item in runtime["source_progress"] if item["name"] == "garden_source")
                tectonics_progress = next(item for item in runtime["source_progress"] if item["name"] == "tectonics_source")

                self.assertEqual(runtime["last_event"]["source"]["source_name"], "tectonics_source")
                self.assertEqual(runtime["background_source_routing"]["mode"], "focus_aware_allocation")
                self.assertEqual(runtime["background_source_routing"]["selection_order"][0], "tectonics_source")
                self.assertGreater(float(tectonics_progress["last_semantic_match"]), float(garden_progress["last_semantic_match"]))
                self.assertGreater(float(tectonics_progress["last_selection_score"]), float(garden_progress["last_selection_score"]))
                self.assertEqual(int(tectonics_progress["tick_visits"]), 1)
                self.assertEqual(int(garden_progress["tick_visits"]), 0)
            finally:
                manager.close()

    def test_background_source_utility_biases_repeated_focused_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_source_utility")
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text("crust plates move over the mantle and form mountains " * 24, encoding="utf-8")
            second_path.write_text("subduction faults reshape crust and mantle boundaries " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "first_source",
                            "source": str(first_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        },
                        {
                            "name": "second_source",
                            "source": str(second_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        },
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                manager.runtime_facade.query(query_text="How do crust plates move over the mantle?", top_k_memories=6)
                first_tick = manager.runtime_facade.terminus_tick()["terminus_runtime"]
                second_tick = manager.runtime_facade.terminus_tick()["terminus_runtime"]
                first_progress = next(item for item in second_tick["source_progress"] if item["name"] == "first_source")
                second_progress = next(item for item in second_tick["source_progress"] if item["name"] == "second_source")

                self.assertEqual(first_tick["last_event"]["source"]["source_name"], "first_source")
                self.assertGreater(float(first_progress["utility_ema"]), 0.0)
                self.assertEqual(second_tick["last_event"]["source"]["source_name"], "first_source")
                self.assertGreater(float(first_progress["utility_ema"]), float(second_progress["utility_ema"]))
                self.assertGreater(float(first_progress["last_utility_score"]), float(second_progress["last_utility_score"]))
            finally:
                manager.close()

    def test_respond_grounded_outcome_reinforces_background_source_utility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_outcome_calibration")
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text("crust plates move over the mantle and form mountains " * 24, encoding="utf-8")
            second_path.write_text("garden tomatoes need soil sunlight and watering " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(first_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        },
                        {
                            "name": "garden_source",
                            "source": str(second_path),
                            "source_type": "file",
                            "metadata": {"label": "garden soil sunlight watering tomatoes"},
                        },
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                manager.runtime_facade.query(query_text="How do crust plates move over the mantle?", top_k_memories=6)
                manager.runtime_facade.terminus_tick()
                before_runtime = manager.runtime_facade.status()["terminus_runtime"]
                before_tectonics = next(item for item in before_runtime["source_progress"] if item["name"] == "tectonics_source")
                before_garden = next(item for item in before_runtime["source_progress"] if item["name"] == "garden_source")

                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates move over the mantle through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [{"text": "Crust plates move over the mantle and form mountains."}],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ):
                    manager.runtime_facade.respond(
                        query_text="How do crust plates move over the mantle?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                after_runtime = manager.runtime_facade.status()["terminus_runtime"]
                after_tectonics = next(item for item in after_runtime["source_progress"] if item["name"] == "tectonics_source")
                after_garden = next(item for item in after_runtime["source_progress"] if item["name"] == "garden_source")

                self.assertGreater(float(after_tectonics["grounded_outcome_ema"]), 0.0)
                self.assertGreaterEqual(float(after_tectonics["utility_ema"]), float(before_tectonics["utility_ema"]))
                self.assertEqual(float(after_garden["grounded_outcome_ema"]), float(before_garden["grounded_outcome_ema"]))
                self.assertGreater(float(after_tectonics["utility_ema"]), float(after_garden["utility_ema"]))
            finally:
                manager.close()

    def test_query_result_memory_episodes_expose_background_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_evidence_provenance")
            tectonics_path = root / "tectonics.txt"
            garden_path = root / "garden.txt"
            tectonics_path.write_text("crust plates move over the mantle and form mountains " * 24, encoding="utf-8")
            garden_path.write_text("garden tomatoes need soil sunlight and watering " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        },
                        {
                            "name": "garden_source",
                            "source": str(garden_path),
                            "source_type": "file",
                            "metadata": {"label": "garden soil sunlight watering tomatoes"},
                        },
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                manager.runtime_facade.query(query_text="How do crust plates move over the mantle?", top_k_memories=6)
                manager.runtime_facade.terminus_tick()
                query_result = manager.runtime_facade.query(
                    query_text="How do crust plates move over the mantle?",
                    top_k_memories=6,
                )
                episodes = query_result["query_summary"]["memory_episodes"]

                self.assertTrue(episodes)
                self.assertTrue(any((episode.get("source_name") == "tectonics_source") for episode in episodes))
                tectonics_episode = next(episode for episode in episodes if episode.get("source_name") == "tectonics_source")
                self.assertIn("tectonics_source", tectonics_episode.get("source_names") or [tectonics_episode.get("source_name")])
            finally:
                manager.close()

    def test_follow_up_query_improvement_reinforces_background_source_delayed_consequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_delayed_consequence")
            tectonics_path = root / "tectonics.txt"
            tectonics_path.write_text(
                (
                    "Crust plates drift over the mantle and slowly move continents. " * 3
                    + "Convergent plate collisions build mountain ranges and lift rock upward. " * 5
                ),
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountains mantle"},
                        }
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                query_text = "How do crust plates build mountain ranges?"
                manager.runtime_facade.terminus_tick()
                with patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    initial = manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                before_runtime = manager.runtime_facade.status()["terminus_runtime"]
                before_source = next(item for item in before_runtime["source_progress"] if item["name"] == "tectonics_source")

                self.assertIn(
                    "tectonics_source",
                    initial["response"]["delayed_consequence_candidate"]["source_names"],
                )
                self.assertEqual(float(before_source["delayed_consequence_ema"]), 0.0)

                for _ in range(3):
                    manager.runtime_facade.terminus_tick()
                follow_up = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                after_runtime = manager.runtime_facade.status()["terminus_runtime"]
                after_source = next(item for item in after_runtime["source_progress"] if item["name"] == "tectonics_source")

                self.assertGreater(
                    float(follow_up["gap_plan"]["grounded_fraction"]),
                    float(initial["query_result"]["gap_plan"]["grounded_fraction"]),
                )
                self.assertGreater(int(follow_up["delayed_consequence"]["credited_records"]), 0)
                self.assertIn("tectonics_source", follow_up["delayed_consequence"]["credited_source_names"])
                self.assertGreater(float(after_source["delayed_consequence_ema"]), float(before_source["delayed_consequence_ema"]))
                self.assertGreater(
                    int(after_runtime["background_source_routing"]["delayed_consequence_tracking"]["credited_record_count"]),
                    0,
                )
            finally:
                manager.close()

    def test_follow_up_query_regression_penalizes_background_source_long_horizon_utility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_long_horizon_penalty")
            source_path = root / "tectonics.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(source_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                query_text = "How do crust plates move over the mantle?"
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates move over the mantle through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates move over the mantle.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                runtime = manager._brain_source_runtimes[0]
                before_score, *_ = manager._brain_source_selection_score_locked(
                    runtime,
                    focus_terms=["tectonics", "crust", "plates", "mantle"],
                    focus_pressure=1.0,
                    tick_tokens=8,
                )

                follow_up = manager.runtime_facade.query(query_text=query_text, top_k_memories=6)
                after_score, *_ = manager._brain_source_selection_score_locked(
                    runtime,
                    focus_terms=["tectonics", "crust", "plates", "mantle"],
                    focus_pressure=1.0,
                    tick_tokens=8,
                )
                source_progress = manager.runtime_facade.status()["terminus_runtime"]["source_progress"][0]

                self.assertGreater(int(follow_up["delayed_consequence"]["penalized_records"]), 0)
                self.assertIn("tectonics_source", follow_up["delayed_consequence"]["penalized_source_names"])
                self.assertGreater(float(source_progress["contradiction_decay_ema"]), 0.0)
                self.assertLess(float(after_score), float(before_score))
            finally:
                manager.close()

    def test_later_grounded_improvement_forgives_background_source_long_horizon_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_long_horizon_forgiveness")
            tectonics_path = root / "tectonics.txt"
            tectonics_path.write_text(
                (
                    "Crust plates drift over the mantle and slowly move continents. " * 3
                    + "Convergent plate collisions build mountain ranges and lift rock upward. " * 5
                ),
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountains mantle"},
                        }
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                query_text = "How do crust plates build mountain ranges?"
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates build mountain ranges through convergent plate collisions.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates build mountain ranges.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                penalized = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                penalty_runtime = manager.runtime_facade.status()["terminus_runtime"]
                penalty_source = next(item for item in penalty_runtime["source_progress"] if item["name"] == "tectonics_source")
                runtime = manager._brain_source_runtimes[0]
                penalty_score, *_ = manager._brain_source_selection_score_locked(
                    runtime,
                    focus_terms=["tectonics", "crust", "plates", "mountain", "ranges"],
                    focus_pressure=1.0,
                    tick_tokens=120,
                )
                penalty_record = penalty_runtime["background_source_routing"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)
                self.assertIn("tectonics_source", penalized["delayed_consequence"]["penalized_source_names"])
                self.assertGreater(float(penalty_source["contradiction_decay_ema"]), 0.0)

                for _ in range(3):
                    manager.runtime_facade.terminus_tick()
                forgiven = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                forgiven_runtime = manager.runtime_facade.status()["terminus_runtime"]
                forgiven_source = next(item for item in forgiven_runtime["source_progress"] if item["name"] == "tectonics_source")
                forgiven_score, *_ = manager._brain_source_selection_score_locked(
                    runtime,
                    focus_terms=["tectonics", "crust", "plates", "mountain", "ranges"],
                    focus_pressure=1.0,
                    tick_tokens=120,
                )
                forgiven_record = forgiven_runtime["background_source_routing"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(
                    float(forgiven["gap_plan"]["grounded_fraction"]),
                    float(penalized["gap_plan"]["grounded_fraction"]),
                )
                self.assertGreater(int(forgiven["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(forgiven["delayed_consequence"]["forgiven_records"]), 0)
                self.assertIn("tectonics_source", forgiven["delayed_consequence"]["forgiven_source_names"])
                self.assertLess(
                    float(forgiven_source["contradiction_decay_ema"]),
                    float(penalty_source["contradiction_decay_ema"]),
                )
                self.assertLess(
                    float(forgiven_record["unresolved_penalty_balance"]),
                    float(penalty_record["unresolved_penalty_balance"]),
                )
                self.assertGreater(int(forgiven_record["forgiveness_events"]), 0)
                self.assertGreater(float(forgiven_record["last_forgiveness_score"]), 0.0)
                self.assertGreaterEqual(float(forgiven_score), 0.0)
                self.assertGreaterEqual(float(penalty_score), 0.0)
                self.assertGreater(
                    int(forgiven_runtime["background_source_routing"]["delayed_consequence_tracking"]["forgiven_record_count"]),
                    0,
                )
            finally:
                manager.close()

    def test_stale_background_consequence_state_cools_and_retires(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_consequence_cooling")
            source_path = root / "tectonics.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(source_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle subduction"},
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                query_text = "How do crust plates move over the mantle?"
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates move over the mantle through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates move over the mantle.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                manager.runtime_facade.query(query_text=query_text, top_k_memories=6)
                penalty_runtime = manager.runtime_facade.status()["terminus_runtime"]
                penalty_tracking = penalty_runtime["background_source_routing"]["delayed_consequence_tracking"]
                penalty_record = penalty_tracking["recent_records"][0]

                self.assertGreater(int(penalty_tracking["penalized_record_count"]), 0)
                self.assertGreater(float(penalty_record["unresolved_penalty_balance"]), 0.0)

                manager._trainer.token_count += (
                    delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS
                    + delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS
                )
                cooled_runtime = manager.runtime_facade.status()["terminus_runtime"]
                cooled_tracking = cooled_runtime["background_source_routing"]["delayed_consequence_tracking"]
                cooled_record = cooled_tracking["recent_records"][0]

                self.assertLess(
                    float(cooled_record["unresolved_penalty_balance"]),
                    float(penalty_record["unresolved_penalty_balance"]),
                )
                self.assertGreater(int(cooled_record["cooling_events"]), 0)
                self.assertGreater(int(cooled_tracking["cooled_record_count_total"]), 0)

                manager._trainer.token_count += delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS * 2
                retired_runtime = manager.runtime_facade.status()["terminus_runtime"]
                retired_tracking = retired_runtime["background_source_routing"]["delayed_consequence_tracking"]

                self.assertEqual(int(retired_tracking["record_count"]), 0)
                self.assertGreater(int(retired_tracking["retired_record_count_total"]), 0)
            finally:
                manager.close()

    def test_repeated_background_consequence_records_compact_query_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_consequence_compaction")
            source_path = root / "tectonics.txt"
            source_path.write_text("tectonic plates mountain ranges crust mantle " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(source_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountain ranges mantle"},
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates build mountain ranges through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates build mountain ranges through tectonic motion.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    first = manager.runtime_facade.respond(
                        query_text="How do crust plates build mountain ranges?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    second = manager.runtime_facade.respond(
                        query_text="How do crust plates form mountain ranges?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                tracking = manager.runtime_facade.status()["terminus_runtime"]["background_source_routing"]["delayed_consequence_tracking"]
                record = tracking["recent_records"][0]

                self.assertEqual(int(second["response"]["delayed_consequence_candidate"]["aggregate_count"]), 2)
                self.assertEqual(int(tracking["record_count"]), 1)
                self.assertGreater(int(tracking["aggregated_record_count"]), 0)
                self.assertGreaterEqual(int(tracking["aggregate_occurrence_count"]), 2)
                self.assertGreater(int(tracking["compacted_record_count_total"]), 0)
                self.assertEqual(int(record["aggregate_count"]), 2)
                self.assertGreater(float(record["aggregate_support_multiplier"]), 1.0)
                self.assertIn("How do crust plates build mountain ranges?", record["query_examples"])
                self.assertIn("How do crust plates form mountain ranges?", record["query_examples"])
            finally:
                manager.close()

    def test_background_consequence_family_trajectory_summary_tracks_penalty_then_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_consequence_trajectory")
            tectonics_path = root / "tectonics.txt"
            tectonics_path.write_text(
                (
                    "Crust plates drift over the mantle and slowly move continents. " * 3
                    + "Convergent plate collisions build mountain ranges and lift rock upward. " * 5
                ),
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountains mantle"},
                        }
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates build mountain ranges through convergent plate collisions.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates build mountain ranges through convergent plate collisions.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How do crust plates build mountain ranges?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    manager.runtime_facade.respond(
                        query_text="How do crust plates form mountain ranges?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                query_text = "How do crust plates build mountain ranges?"
                penalized = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                penalty_runtime = manager.runtime_facade.status()["terminus_runtime"]
                penalty_record = penalty_runtime["background_source_routing"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)
                self.assertEqual(int(penalty_record["aggregate_count"]), 2)
                self.assertEqual(str(penalty_record["trajectory_state"]), "negative")
                self.assertGreater(float(penalty_record["trajectory_penalty_total"]), 0.0)
                self.assertLess(float(penalty_record["trajectory_support_multiplier"]), 1.0)
                self.assertGreater(float(penalty_record["trajectory_penalty_multiplier"]), 1.0)

                for _ in range(3):
                    manager.runtime_facade.terminus_tick()
                recovered = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                recovered_runtime = manager.runtime_facade.status()["terminus_runtime"]
                recovered_record = recovered_runtime["background_source_routing"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(int(recovered["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(recovered["delayed_consequence"]["forgiven_records"]), 0)
                self.assertGreater(float(recovered_record["trajectory_credit_total"]), 0.0)
                self.assertGreater(float(recovered_record["trajectory_forgiveness_total"]), 0.0)
                self.assertGreater(
                    float(recovered_record["trajectory_net_score"]),
                    float(penalty_record["trajectory_net_score"]),
                )
                self.assertGreater(
                    float(recovered_record["trajectory_recent_delta_ema"]),
                    float(penalty_record["trajectory_recent_delta_ema"]),
                )
                self.assertGreater(
                    float(recovered_record["trajectory_support_multiplier"]),
                    float(penalty_record["trajectory_support_multiplier"]),
                )
                self.assertLess(
                    float(recovered_record["trajectory_penalty_multiplier"]),
                    float(penalty_record["trajectory_penalty_multiplier"]),
                )
                self.assertEqual(str(recovered_record["trajectory_state"]), "recovering")
            finally:
                manager.close()

    def test_background_consequence_family_divergence_split_separates_mixed_query_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_consequence_split")
            tectonics_path = root / "tectonics.txt"
            tectonics_path.write_text(
                (
                    "Crust plates drift over the mantle and slowly move continents. " * 4
                    + "Convergent plate collisions build mountain ranges and lift rock upward. " * 4
                ),
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountains continents mantle"},
                        }
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates move continents and build mountain ranges through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates move continents and build mountain ranges through tectonic motion.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How do crust plates build mountain ranges over the mantle?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    second = manager.runtime_facade.respond(
                        query_text="How do crust plates move continents over the mantle?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                self.assertEqual(int(second["response"]["delayed_consequence_candidate"]["aggregate_count"]), 2)

                penalized = manager.runtime_facade.query(
                    query_text="How do crust plates build mountain ranges over the mantle?",
                    top_k_memories=8,
                )
                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)

                for _ in range(3):
                    manager.runtime_facade.terminus_tick()
                recovered = manager.runtime_facade.query(
                    query_text="How do crust plates move continents over the mantle?",
                    top_k_memories=8,
                )
                tracking = manager.runtime_facade.status()["terminus_runtime"]["background_source_routing"]["delayed_consequence_tracking"]
                records = {
                    str(record.get("split_branch", "")): record
                    for record in tracking["recent_records"]
                    if str(record.get("split_branch", ""))
                }

                self.assertGreater(int(recovered["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(recovered["delayed_consequence"]["forgiven_records"]), 0)
                self.assertEqual(int(tracking["record_count"]), 2)
                self.assertGreater(int(tracking["split_record_count_total"]), 0)
                self.assertEqual(int(tracking["aggregate_occurrence_count"]), 2)
                self.assertIn("supportive", records)
                self.assertIn("adverse", records)
                supportive = records["supportive"]
                adverse = records["adverse"]
                self.assertIn("How do crust plates move continents over the mantle?", supportive["query_examples"])
                self.assertIn("How do crust plates build mountain ranges over the mantle?", adverse["query_examples"])
                self.assertEqual(str(adverse["trajectory_state"]), "negative")
                self.assertGreater(
                    float(supportive["trajectory_net_score"]),
                    float(adverse["trajectory_net_score"]),
                )
                self.assertEqual(str(supportive["split_group_id"]), str(adverse["split_group_id"]))
            finally:
                manager.close()

    def test_background_split_lineage_remerges_after_aligned_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_consequence_remerge")
            tectonics_path = root / "tectonics.txt"
            tectonics_path.write_text(
                (
                    "Crust plates drift over the mantle and slowly move continents. " * 4
                    + "Convergent plate collisions build mountain ranges and lift rock upward. " * 4
                ),
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "tectonics_source",
                            "source": str(tectonics_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mountains continents mantle"},
                        }
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Crust plates move continents and build mountain ranges through tectonic motion.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Crust plates move continents and build mountain ranges through tectonic motion.",
                                "source_name": "tectonics_source",
                                "source_names": ["tectonics_source"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How do crust plates build mountain ranges over the mantle?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    manager.runtime_facade.respond(
                        query_text="How do crust plates move continents over the mantle?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                manager.runtime_facade.query(
                    query_text="How do crust plates build mountain ranges over the mantle?",
                    top_k_memories=8,
                )
                for _ in range(3):
                    manager.runtime_facade.terminus_tick()
                split_result = manager.runtime_facade.query(
                    query_text="How do crust plates move continents over the mantle?",
                    top_k_memories=8,
                )
                split_tracking = manager.runtime_facade.status()["terminus_runtime"]["background_source_routing"]["delayed_consequence_tracking"]

                self.assertGreater(int(split_result["delayed_consequence"]["split_records"]), 0)
                self.assertEqual(int(split_tracking["record_count"]), 2)

                remerged = manager.runtime_facade.query(
                    query_text="How do crust plates move continents over the mantle?",
                    top_k_memories=8,
                )
                tracking = manager.runtime_facade.status()["terminus_runtime"]["background_source_routing"]["delayed_consequence_tracking"]
                record = tracking["recent_records"][0]

                self.assertGreater(int(remerged["delayed_consequence"]["remerged_records"]), 0)
                self.assertEqual(int(tracking["record_count"]), 1)
                self.assertGreater(int(tracking["remerged_record_count_total"]), 0)
                self.assertEqual(str(record["split_branch"]), "")
                self.assertGreater(int(record["remerge_events"]), 0)
                self.assertTrue(str(record["last_remerged_at"]))
                self.assertIn("How do crust plates move continents over the mantle?", record["query_examples"])
                self.assertIn("How do crust plates build mountain ranges over the mantle?", record["query_examples"])
            finally:
                manager.close()

    def test_grounded_family_summary_biases_equal_focus_background_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_family_summary")
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text(
                "crust plates drift over the mantle and build mountain ranges " * 24,
                encoding="utf-8",
            )
            second_path.write_text(
                "crust plates drift over the mantle and build mountain ranges " * 24,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "first_source",
                            "source": str(first_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle mountains"},
                        },
                        {
                            "name": "second_source",
                            "source": str(second_path),
                            "source_type": "file",
                            "metadata": {"label": "tectonics crust plates mantle mountains"},
                        },
                    ],
                    tick_tokens=120,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                record = manager._normalize_delayed_consequence_record(
                    {
                        "query_text": "How do crust plates build mountain ranges?",
                        "query_examples": ["How do crust plates build mountain ranges?"],
                        "baseline_query_score": 0.42,
                        "best_query_score": 0.96,
                        "baseline_grounded_fraction": 0.30,
                        "best_grounded_fraction": 1.0,
                        "outcome_score": 0.92,
                        "source_weights": {"first_source": 1.0},
                        "provider_weights": {},
                        "credit_events": 1,
                        "forgiveness_events": 1,
                        "trajectory_credit_total": 0.72,
                        "trajectory_forgiveness_total": 0.20,
                        "trajectory_penalty_total": 0.0,
                        "trajectory_event_count": 2,
                        "trajectory_net_score": 0.92,
                        "trajectory_recent_delta_ema": 0.48,
                        "resolved_improvement": 0.54,
                    }
                )
                self.assertIsNotNone(record)
                family_summary_score = manager._grounded_family_summary_score(record)
                self.assertGreater(float(family_summary_score), 0.0)
                manager._apply_background_source_family_summary_locked(
                    source_weights=record["source_weights"],
                    family_summary_score=family_summary_score,
                )
                runtime = manager.runtime_facade.status()["terminus_runtime"]
                first_progress = next(item for item in runtime["source_progress"] if item["name"] == "first_source")
                second_progress = next(item for item in runtime["source_progress"] if item["name"] == "second_source")

                self.assertGreater(float(first_progress["grounded_family_summary_ema"]), 0.0)
                self.assertEqual(float(second_progress["grounded_family_summary_ema"]), 0.0)

                first_runtime = next(runtime for runtime in manager._brain_source_runtimes if runtime.name == "first_source")
                second_runtime = next(runtime for runtime in manager._brain_source_runtimes if runtime.name == "second_source")
                first_runtime.tick_visits = 0
                second_runtime.tick_visits = 0
                first_runtime.last_activity_at = None
                second_runtime.last_activity_at = None
                first_runtime.buffered_patterns.clear()
                second_runtime.buffered_patterns.clear()
                first_entry = manager._background_source_utility_entry_locked(first_runtime)
                second_entry = manager._background_source_utility_entry_locked(second_runtime)
                for entry in (first_entry, second_entry):
                    entry["utility_ema"] = 0.10
                    entry["semantic_alignment_ema"] = 0.30
                    entry["grounding_signal_ema"] = 0.20
                    entry["focus_overlap_ema"] = 0.30
                    entry["grounded_outcome_ema"] = 0.10
                    entry["delayed_consequence_ema"] = 0.10
                    entry["contradiction_decay_ema"] = 0.0
                first_entry["grounded_family_summary_ema"] = max(0.55, float(first_progress["grounded_family_summary_ema"]))
                second_entry["grounded_family_summary_ema"] = 0.0

                first_score, *_ = manager._brain_source_selection_score_locked(
                    first_runtime,
                    focus_terms=["tectonics", "crust", "plates", "mantle", "mountains"],
                    focus_pressure=1.0,
                    tick_tokens=120,
                )
                second_score, *_ = manager._brain_source_selection_score_locked(
                    second_runtime,
                    focus_terms=["tectonics", "crust", "plates", "mantle", "mountains"],
                    focus_pressure=1.0,
                    tick_tokens=120,
                )

                self.assertGreater(float(first_score), float(second_score))
            finally:
                manager.close()

    def test_focus_aware_background_routing_preserves_rotation_when_focus_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_rotation_fairness")
            first_path = root / "first.txt"
            second_path = root / "second.txt"
            first_path.write_text("general background signal alpha " * 24, encoding="utf-8")
            second_path.write_text("general background signal beta " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "first_source",
                            "source": str(first_path),
                            "source_type": "file",
                            "metadata": {"label": "general background alpha"},
                        },
                        {
                            "name": "second_source",
                            "source": str(second_path),
                            "source_type": "file",
                            "metadata": {"label": "general background beta"},
                        },
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )
                manager.runtime_facade.terminus_tick()
                runtime = manager.runtime_facade.terminus_tick()["terminus_runtime"]
                first_progress = next(item for item in runtime["source_progress"] if item["name"] == "first_source")
                second_progress = next(item for item in runtime["source_progress"] if item["name"] == "second_source")

                self.assertEqual(int(first_progress["tick_visits"]), 1)
                self.assertEqual(int(second_progress["tick_visits"]), 1)
                self.assertEqual(runtime["last_event"]["source"]["source_name"], "second_source")
            finally:
                manager.close()

    def test_terminus_tick_prefetches_into_ingestion_queue_and_reports_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_ingestion_queue")
            source_path = root / "terminus_source.txt"
            source_path.write_text("adaptive memory plasticity signal " * 64, encoding="utf-8")
            try:
                ticked = manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "queued_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    ingestion={"queue_target_tokens": 16, "prewarm_on_startup": False},
                )
                self.assertEqual(ticked["terminus_runtime"]["ingestion"]["queue_target_tokens"], 16)

                runtime = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]
                source_progress = runtime["source_progress"][0]

                self.assertEqual(runtime["ingestion"]["queue_target_tokens"], 16)
                self.assertGreater(runtime["ingestion"]["total_buffered_tokens"], 0)
                self.assertGreater(runtime["ingestion"]["prefetch_events"], 0)
                self.assertGreater(source_progress["buffered_tokens"], 0)
                self.assertGreater(source_progress["prefetched_tokens"], source_progress["last_tokens_trained"])
                self.assertEqual(source_progress["last_prefetch_token_count"], 16)
                self.assertIsNotNone(source_progress["last_prefetch_at"])
                self.assertGreater(float(source_progress["last_prefetch_duration_ms"]), 0.0)
            finally:
                manager.close()

    def test_terminus_warm_queue_serves_second_tick_without_fetching_source_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_warm_queue")

            class _DelayedPatternStream:
                def __init__(self, items, *, delay_after: int, delay_seconds: float) -> None:
                    self._items = list(items)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            delayed_stream = _DelayedPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                delay_after=8,
                delay_seconds=0.12,
            )
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=delayed_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "buffered_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )

                manager.runtime_facade.terminus_tick(steps=1)
                calls_after_first = delayed_stream.next_calls
                second = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]

                self.assertEqual(calls_after_first, 8)
                self.assertEqual(delayed_stream.next_calls, calls_after_first)
                self.assertGreaterEqual(second["source_progress"][0]["queue_hits"], 1)
                self.assertEqual(second["source_progress"][0]["last_buffer_tokens_served"], 4)
                self.assertEqual(second["ingestion"]["queue_hits"], 1)
                self.assertEqual(second["huggingface"]["prefetch_events"], 1)
            finally:
                manager.close()

    def test_terminus_background_prewarm_warms_queue_before_first_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_background_prewarm")

            class _DelayedPatternStream:
                def __init__(self, items, *, delay_after: int, delay_seconds: float) -> None:
                    self._items = list(items)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            active_stream = _DelayedPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                delay_after=8,
                delay_seconds=0.12,
            )
            prewarm_stream = _DelayedPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                delay_after=8,
                delay_seconds=0.12,
            )
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_source_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "prewarmed_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": True},
                    )
                    self.assertIn(
                        configured["terminus_runtime"]["ingestion"]["startup_state"],
                        {"warming", "warm"},
                    )

                    deadline = time.time() + 1.5
                    warm_runtime = configured["terminus_runtime"]
                    while time.time() < deadline:
                        warm_runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if warm_runtime["ingestion"]["startup_state"] == "warm":
                            break
                        time.sleep(0.02)

                    self.assertEqual(warm_runtime["ingestion"]["startup_state"], "warm")
                    self.assertTrue(warm_runtime["ingestion"]["warm_ready"])
                    self.assertFalse(warm_runtime["ingestion"]["prewarm_running"])
                    self.assertIsNotNone(warm_runtime["ingestion"]["prewarm_started_at"])
                    self.assertIsNotNone(warm_runtime["ingestion"]["prewarm_completed_at"])
                    self.assertGreater(float(warm_runtime["ingestion"]["startup_warm_latency_ms"]), 0.0)
                    self.assertEqual(prewarm_stream.next_calls, 8)
                    self.assertEqual(active_stream.next_calls, 0)

                    first_tick = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]
                    self.assertEqual(prewarm_stream.next_calls, 8)
                    self.assertEqual(active_stream.next_calls, 0)
                    self.assertGreaterEqual(first_tick["source_progress"][0]["queue_hits"], 1)
            finally:
                manager.close()

    def test_terminus_startup_state_reports_cold_until_first_fill_without_background_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_cold_start_instrumentation")

            class _DelayedPatternStream:
                def __init__(self, items) -> None:
                    self._items = list(items)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            delayed_stream = _DelayedPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)])
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=delayed_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "cold_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )

                self.assertEqual(configured["terminus_runtime"]["ingestion"]["startup_state"], "cold")
                self.assertFalse(configured["terminus_runtime"]["ingestion"]["warm_ready"])
                self.assertFalse(configured["terminus_runtime"]["ingestion"]["prewarm_running"])
                self.assertIsNone(configured["terminus_runtime"]["ingestion"]["startup_warm_latency_ms"])
                self.assertEqual(delayed_stream.next_calls, 0)

                warmed = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]
                self.assertEqual(warmed["ingestion"]["startup_state"], "warm")
                self.assertTrue(warmed["ingestion"]["warm_ready"])
                self.assertGreater(float(warmed["ingestion"]["startup_warm_latency_ms"]), 0.0)
                self.assertEqual(delayed_stream.next_calls, 8)
            finally:
                manager.close()

    def test_remote_active_tick_returns_quickly_while_source_warms_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_active_tick_budget")

            class _SlowPatternStream:
                def __init__(self, items, *, first_delay_seconds: float) -> None:
                    self._items = list(items)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)], first_delay_seconds=0.3),
                max_buffer=2,
                name="slow-remote-active-text",
            )
            try:
                with patch.object(brain_runtime_module, "DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "slow_remote_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )

                    started = time.perf_counter()
                    first = manager.runtime_facade.terminus_tick(steps=1)
                    first_ms = (time.perf_counter() - started) * 1000.0
                    self.assertLess(first_ms, 200.0)
                    self.assertEqual(first["tick_summaries"][0]["source"]["reason"], "warming_remote_source")
                    self.assertFalse(first["tick_summaries"][0]["did_work"])

                    time.sleep(0.35)
                    second = manager.runtime_facade.terminus_tick(steps=1)
                    self.assertTrue(second["tick_summaries"][0]["source"]["did_work"])
                    self.assertEqual(second["tick_summaries"][0]["source"]["source_name"], "slow_remote_hf_source")
            finally:
                manager.close()

    def test_remote_source_first_rows_bootstrap_warms_runtime_quickly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_text_first_rows_bootstrap")

            class _SlowPatternStream:
                def __init__(self, items, *, first_delay_seconds: float) -> None:
                    self._items = list(items)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)], first_delay_seconds=1.0),
                max_buffer=4,
                name="slow-remote-first-rows-bootstrap",
            )
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ), patch.object(
                    runtime_prewarm_module,
                    "load_hf_first_rows",
                    return_value=[{"text": "Bootstrap cats rest indoors and chase mice at night."}],
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "bootstrap_remote_hf_source",
                                "source": "wikimedia/wikipedia",
                                "source_type": "hf",
                                "hf_config": "20231101.en",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )["terminus_runtime"]
                    self.assertEqual(configured["ingestion"]["startup_state"], "cold")

                    deadline = time.time() + 1.0
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["ingestion"]["warm_ready"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["ingestion"]["warm_ready"])
                    event_types = [event.get("type") for event in runtime["recent_events"]]
                    self.assertIn("remote_text_bootstrap_applied", event_types)
                    self.assertIn("remote_warm_promotion_started", event_types)

                    ticked = manager.runtime_facade.terminus_tick(steps=1)
                    self.assertTrue(ticked["tick_summaries"][0]["source"]["did_work"])
                    self.assertEqual(ticked["tick_summaries"][0]["source"]["source_name"], "bootstrap_remote_hf_source")
            finally:
                manager.close()

    def test_remote_text_bootstrap_timeout_does_not_block_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_text_bootstrap_timeout_close")

            class _SlowPatternStream:
                def __init__(self, items, *, first_delay_seconds: float) -> None:
                    self._items = list(items)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowPatternStream([(f"window-{idx}", fake_pattern) for idx in range(8)], first_delay_seconds=1.0),
                max_buffer=4,
                name="slow-remote-text-bootstrap-timeout",
            )
            bootstrap_started = Event()
            closed = False

            def _slow_first_rows(*_args, **_kwargs):
                bootstrap_started.set()
                time.sleep(0.3)
                return [{"text": "slow bootstrap row"}]

            try:
                with patch.object(runtime_prewarm_module, "DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ), patch.object(
                    runtime_prewarm_module,
                    "load_hf_first_rows",
                    side_effect=_slow_first_rows,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "timeout_remote_hf_source",
                                "source": "wikimedia/wikipedia",
                                "source_type": "hf",
                                "hf_config": "20231101.en",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )
                    self.assertTrue(bootstrap_started.wait(0.2))

                    deadline = time.time() + 0.4
                    event_types: list[str | None] = []
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        event_types = [event.get("type") for event in runtime["recent_events"]]
                        if "remote_text_bootstrap_timed_out" in event_types:
                            break
                        time.sleep(0.02)
                    self.assertIn("remote_text_bootstrap_timed_out", event_types)

                    started = time.perf_counter()
                    manager.close()
                    closed = True
                    close_ms = (time.perf_counter() - started) * 1000.0
                    self.assertLess(close_ms, 200.0)
            finally:
                if not closed:
                    manager.close()

    def test_remote_source_promotion_warms_runtime_without_repeated_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_text_promotion")

            class _SlowPatternStream:
                def __init__(self, items, *, first_delay_seconds: float) -> None:
                    self._items = list(items)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)], first_delay_seconds=0.3),
                max_buffer=4,
                name="slow-remote-promotion-text",
            )
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "promoted_remote_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )["terminus_runtime"]
                    self.assertEqual(configured["ingestion"]["startup_state"], "cold")

                    deadline = time.time() + 1.5
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["ingestion"]["warm_ready"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["ingestion"]["warm_ready"])
                    self.assertGreaterEqual(runtime["ingestion"]["total_buffered_tokens"], 1)
                    event_types = [event.get("type") for event in runtime["recent_events"]]
                    self.assertIn("remote_warm_promotion_started", event_types)
                    self.assertIn("remote_warm_promotion_completed", event_types)

                    ticked = manager.runtime_facade.terminus_tick(steps=1)
                    self.assertTrue(ticked["tick_summaries"][0]["source"]["did_work"])
                    self.assertEqual(ticked["tick_summaries"][0]["source"]["source_name"], "promoted_remote_hf_source")
            finally:
                manager.close()

    def test_remote_source_cache_restore_makes_runtime_warm_and_usable_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            manager = _build_manager(root, test_case="service_manager_remote_text_cache_seed")

            class _FastPatternStream:
                def __init__(self, items) -> None:
                    self._items = list(items)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            seed_stream = _FastPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)])
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=seed_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "cached_remote_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )
                    manager.runtime_facade.terminus_tick(steps=1)
            finally:
                manager.close()

            manager = _build_manager(root, test_case="service_manager_remote_text_cache_restore")

            class _UnusedPatternStream(_FastPatternStream):
                def __next__(self):
                    self.next_calls += 1
                    raise AssertionError("remote stream should not be touched before cached work is consumed")

            restored_stream = _UnusedPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)])
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=restored_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "cached_remote_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={"queue_target_tokens": 8, "prewarm_on_startup": False},
                    )["terminus_runtime"]
                    self.assertEqual(configured["ingestion"]["startup_state"], "warm")
                    self.assertTrue(configured["ingestion"]["warm_ready"])
                    self.assertGreaterEqual(configured["ingestion"]["total_buffered_tokens"], 4)
                    event_types = [event.get("type") for event in configured["recent_events"]]
                    self.assertIn("ingestion_cache_restored", event_types)

                    ticked = manager.runtime_facade.terminus_tick(steps=1)
                    self.assertTrue(ticked["tick_summaries"][0]["source"]["did_work"])
                    self.assertEqual(restored_stream.next_calls, 0)
            finally:
                manager.close()

    def test_terminus_prewarm_budget_exhaustion_surfaces_partial_warm_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_prewarm_budget")

            class _DelayedPatternStream:
                def __init__(self, items, *, delay_after: int, delay_seconds: float) -> None:
                    self._items = list(items)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            active_stream = _DelayedPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                delay_after=4,
                delay_seconds=0.12,
            )
            prewarm_stream = _DelayedPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                delay_after=4,
                delay_seconds=0.12,
            )
            try:
                with patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_source_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "budgeted_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={
                            "queue_target_tokens": 8,
                            "prewarm_on_startup": True,
                            "prewarm_max_seconds": 0.05,
                        },
                    )["terminus_runtime"]

                    self.assertIn(configured["ingestion"]["startup_state"], {"warming", "warm"})
                    deadline = time.time() + 1.5
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if not runtime["ingestion"]["prewarm_running"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["ingestion"]["prewarm_budget_exhausted"])
                    self.assertTrue(runtime["ingestion"]["warm_ready"])
                    self.assertFalse(runtime["ingestion"]["full_warm_ready"])
                    self.assertGreater(runtime["ingestion"]["ready_source_count"], 0)
                    self.assertEqual(runtime["ingestion"]["full_queue_source_count"], 0)
                    self.assertAlmostEqual(float(runtime["ingestion"]["prewarm_max_seconds"]), 0.05, places=6)
                    self.assertGreater(prewarm_stream.next_calls, 0)
                    self.assertLess(prewarm_stream.next_calls, 8)
                    self.assertEqual(active_stream.next_calls, 0)

                    calls_before_tick = prewarm_stream.next_calls
                    first_tick = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]
                    self.assertEqual(prewarm_stream.next_calls, calls_before_tick)
                    self.assertEqual(active_stream.next_calls, 0)
                    self.assertGreaterEqual(first_tick["source_progress"][0]["queue_hits"], 1)
            finally:
                manager.close()

    def test_immediate_tick_does_not_wait_behind_blocking_startup_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_isolated_prewarm_tick")

            class _FastPatternStream:
                def __init__(self, items) -> None:
                    self._items = list(items)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= len(self._items):
                        raise StopIteration
                    item = self._items[self._index]
                    self._index += 1
                    return item

            class _BlockingPatternStream(_FastPatternStream):
                def __init__(self, items, started: Event, delay_seconds: float) -> None:
                    super().__init__(items)
                    self._started = started
                    self._delay_seconds = float(delay_seconds)

                def __next__(self):
                    self._started.set()
                    time.sleep(self._delay_seconds)
                    return super().__next__()

            fake_pattern = manager._encoder.blended_feature_vector([97] * manager._trainer.config.window_size)
            active_stream = _FastPatternStream([(f"window-{idx}", fake_pattern) for idx in range(12)])
            prewarm_started = Event()
            prewarm_stream = _BlockingPatternStream(
                [(f"window-{idx}", fake_pattern) for idx in range(12)],
                started=prewarm_started,
                delay_seconds=0.45,
            )
            try:
                with patch.object(runtime_prewarm_module, "DEFAULT_REMOTE_PREWARM_GRACE_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_brain_source_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_source_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[
                            {
                                "name": "isolated_hf_source",
                                "source": "HuggingFaceFW/fineweb-edu",
                                "source_type": "hf",
                            }
                        ],
                        tick_tokens=4,
                        sleep_interval_seconds=0.01,
                        repeat_sources=False,
                        ingestion={
                            "queue_target_tokens": 8,
                            "prewarm_on_startup": True,
                            "prewarm_max_seconds": 5.0,
                        },
                    )["terminus_runtime"]

                    self.assertIn(configured["ingestion"]["startup_state"], {"warming", "warm"})
                    self.assertFalse(prewarm_started.is_set())

                    started = time.perf_counter()
                    ticked = manager.runtime_facade.terminus_tick(steps=1)["terminus_runtime"]
                    tick_ms = (time.perf_counter() - started) * 1000.0

                    self.assertLess(tick_ms, 500.0)
                    self.assertEqual(active_stream.next_calls, 8)
                    self.assertEqual(ticked["source_progress"][0]["last_buffer_tokens_served"], 4)
                    self.assertFalse(prewarm_started.wait(0.2))
                    event_types = [event.get("type") for event in manager.runtime_facade.terminus_status()["terminus_runtime"]["recent_events"]]
                    self.assertIn("ingestion_prewarm_skipped_after_active_execution", event_types)
            finally:
                manager.close()

    def test_terminus_tick_records_grounded_source_evidence_without_retired_loop_mirroring(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_grounded_source_observation")
            source_path = root / "terminus_grounded_source.txt"
            source_path.write_text(
                "Cats rest indoors and chase mice at night. " * 80,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "cats_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=48,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                ticked = manager.runtime_facade.terminus_tick(steps=2)
                self.assertIn("grounded_observation", ticked["tick_summaries"][0]["source"])
                grounded = ticked["tick_summaries"][0]["source"]["grounded_observation"]
                self.assertIn("cats", grounded["content"].lower())
                self.assertIn("indoors", grounded["content"].lower())
                self.assertIn("mice", grounded["content"].lower())
                self.assertNotIn("snn processed", grounded["content"].lower())
                self.assertNotIn("recent concepts", grounded["content"].lower())
                self.assertGreater(len(grounded["topics"]), 0)
                self.assertEqual(grounded["observation_sink"], "subcortex_grounded_source_observation")
                self.assertNotIn("retired_loop_mirrored", grounded)
                self.assertTrue(grounded["metadata"]["grounded"])
                self.assertEqual(grounded["metadata"]["observation_kind"], "source")
                self.assertEqual(grounded["metadata"]["source_name"], "cats_source")
                self.assertEqual(grounded["metadata"]["source_type"], "file")
                self.assertEqual(grounded["metadata"]["modality"], "text")
            finally:
                manager.close()

    def test_autonomy_semantic_registry_acquires_local_real_source_without_curriculum_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "background.txt").write_text("neutral background signal " * 40, encoding="utf-8")
            (root / "tectonics.html").write_text(
                "<html><body><main><p>Plate tectonics describes rigid crust plates moving over the mantle.</p>"
                "<p>Subduction, faults, and mountain building are key processes.</p></main></body></html>",
                encoding="utf-8",
            )
            (root / "gardening.html").write_text(
                "<html><body><main><p>Tomatoes need sunlight and watering.</p></main></body></html>",
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            manager = _build_manager(root, test_case="service_manager_real_source_autonomy_registry")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "background",
                            "source": str(root / "background.txt"),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "runtime_registry",
                                "catalog_mode": "semantic_registry",
                                "catalog_limit": 2,
                                "catalog_entries": [
                                    {
                                        "name": "tectonics_source",
                                        "source": f"http://127.0.0.1:{port}/tectonics.html",
                                        "source_type": "web",
                                        "summary": "plate tectonics crust mantle subduction faults mountains",
                                    },
                                    {
                                        "name": "garden_source",
                                        "source": f"http://127.0.0.1:{port}/gardening.html",
                                        "source_type": "web",
                                        "summary": "garden tomato soil sunlight watering",
                                    },
                                ],
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 96,
                        "probe_tokens": 48,
                        "acquisition_tokens": 96,
                        "acquisition_slots": 1,
                    },
                )
                manager.runtime_facade.query(query_text="How do crust plates move over the mantle?", top_k_memories=6)
                tick = manager.runtime_facade.terminus_tick()
                runtime = tick["terminus_runtime"]
                autonomy = runtime["autonomy"]

                self.assertIsNotNone(autonomy)
                self.assertIsNotNone(autonomy["last_acquisition_summary"])
                self.assertEqual(autonomy["last_acquisition_summary"]["acquired_sources"], ["tectonics_source"])
                self.assertGreaterEqual(int(autonomy["last_acquisition_summary"]["tokens_trained_total"]), 1)
                self.assertGreaterEqual(
                    int(runtime["text_learning_balance"]["autonomy_tokens_processed"]),
                    int(autonomy["last_acquisition_summary"]["tokens_trained_total"]),
                )
                self.assertGreater(float(runtime["text_learning_balance"]["autonomy_share_of_text_learning"]), 0.0)
                self.assertNotIn("curriculum", runtime)
                self.assertEqual(runtime["multimodal"]["mode"], "disabled")
                self.assertEqual(runtime["multimodal"]["episodes_completed"], 0)
                self.assertGreaterEqual(int(runtime["last_tick_token_delta"]), int(autonomy["last_acquisition_summary"]["tokens_trained_total"]))
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

    def test_real_sensory_episode_uses_hf_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_real_sensory")
            calls: list[dict[str, object]] = []
            episode = SensoryEpisode(
                text="A scientific figure shows two sharply separated regions in a lattice.",
                visual_spikes=torch.ones(64),
                audio_spikes=None,
                metadata={
                    "adapter": "s1_mmalign",
                    "device": "cpu",
                    "encoder": {"encoder": "event_camera", "device": "cpu"},
                    "spike_device": "cpu",
                    "spike_is_cuda": False,
                },
                visual_preview={
                    "mime_type": "image/png",
                    "bytes": b"fakepng",
                    "width": 16,
                    "height": 16,
                },
            )

            def _fake_train_step(pattern, **kwargs):
                calls.append(kwargs)
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": True, "cross_modal_audio_accepted": False}

            try:
                manager._trainer.config.enable_cross_modal = True
                manager._brain_config["sensory"] = {
                    "enabled": True,
                    "source_bank": [
                        {
                            "name": "science_figures",
                            "adapter": "s1_mmalign",
                            "source": "ScienceOne-AI/S1-MMAlign",
                            "split": "train",
                            "year_prefixes": ["07"],
                        }
                    ],
                    "episode_interval_tokens": 1,
                    "items_per_episode": 1,
                    "base_windows_per_item": 2,
                    "max_windows_per_item": 4,
                    "confidence_window_gain": 4.0,
                    "modality_target_confidence": 0.70,
                    "observation_salience": 0.80,
                    "cooldown_seconds": 1.0,
                    "repeat_sources": True,
                }
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=iter([episode]),
                ):
                    manager._rebuild_brain_sources_locked()
                manager._trainer.token_count = 64
                manager._last_real_sensory_episode_token_count = 0
                manager._last_real_sensory_episode_time = 0.0
                original_train_step = manager._trainer.train_step
                manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                try:
                    summary = manager._run_real_sensory_episode_locked()
                finally:
                    manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                self.assertIsNotNone(summary)
                self.assertTrue(calls)
                self.assertEqual(len(calls), 4)
                self.assertEqual(summary["sources"][0]["window_budget"], 4)
                self.assertIsNotNone(calls[0].get("visual_spikes"))
                self.assertIsNone(calls[0].get("audio_spikes"))
                grounded = summary["sources"][0]["grounded_observation"]
                self.assertEqual(grounded["observation_kind"], "sensory")
                self.assertEqual(grounded["modality"], "image")
                self.assertEqual(grounded["device"], "cpu")
                self.assertEqual(grounded["encoder"]["device"], "cpu")
                self.assertEqual(grounded["observation_sink"], "subcortex_grounded_sensory_observation")
                self.assertNotIn("retired_loop_mirrored", grounded)
                self.assertEqual(grounded["metadata"]["observation_sink"], "subcortex_grounded_sensory_observation")
                self.assertNotIn("retired_loop_mirrored", grounded["metadata"])
                self.assertIn("lattice", grounded["content"].lower())
                self.assertEqual(runtime["sensory"]["source_progress"][0]["episodes_processed"], 1)
                self.assertEqual(runtime["multimodal"]["real_episodes_completed"], 1)
                self.assertEqual(runtime["multimodal"]["recent_preview_count"], 1)
                self.assertIsNotNone(runtime["multimodal"]["latest_preview_id"])
                self.assertGreater(runtime["multimodal"]["real_cross_modal_visual_accepted"], 0)
                previews = manager.runtime_facade.sensory_previews(limit=1)
                self.assertEqual(previews["count"], 1)
                self.assertEqual(len(previews["previews"]), 1)
                self.assertTrue(previews["previews"][0]["visual"]["data_url"].startswith("data:image/png;base64,"))
            finally:
                manager.close()

    def test_sensory_background_prewarm_warms_queue_before_first_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_sensory_prewarm")

            class _DelayedSensoryStream:
                def __init__(self, episodes, *, delay_after: int = 999, delay_seconds: float = 0.0) -> None:
                    self._episodes = list(episodes)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episodes = [
                SensoryEpisode(
                    text=(
                        "Water splashes while wind and footsteps move through an outdoor path. "
                        "Water splashes while wind and footsteps move through an outdoor path. "
                        "Water splashes while wind and footsteps move through an outdoor path. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "water wind footsteps"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"waterwindsteps-1",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.4] * 8,
                    },
                ),
                SensoryEpisode(
                    text=(
                        "Rain taps against leaves while distant birds call in a damp garden. "
                        "Rain taps against leaves while distant birds call in a damp garden. "
                        "Rain taps against leaves while distant birds call in a damp garden. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "rain leaves birds"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"rainbirds-2",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.3] * 8,
                    },
                ),
            ]
            active_stream = _DelayedSensoryStream(episodes)
            prewarm_stream = _DelayedSensoryStream(list(episodes))

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 1,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 2,
                            "prewarm_on_startup": True,
                        },
                    )["terminus_runtime"]

                    self.assertIn(configured["sensory"]["startup_state"], {"warming", "warm"})
                    deadline = time.time() + 1.5
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["sensory"]["startup_state"] == "warm":
                            break
                        time.sleep(0.02)

                    self.assertEqual(runtime["sensory"]["startup_state"], "warm")
                    self.assertTrue(runtime["sensory"]["warm_ready"])
                    self.assertEqual(runtime["sensory"]["total_buffered_items"], 2)
                    self.assertEqual(runtime["sensory"]["ready_source_count"], 1)
                    self.assertGreater(float(runtime["sensory"]["startup_warm_latency_ms"]), 0.0)
                    self.assertEqual(prewarm_stream.next_calls, 2)
                    self.assertEqual(active_stream.next_calls, 0)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(prewarm_stream.next_calls, 2)
                    self.assertEqual(active_stream.next_calls, 0)
                    sensory_runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]["sensory"]
                    self.assertGreaterEqual(sensory_runtime["source_progress"][0]["queue_hits"], 1)
                    self.assertEqual(sensory_runtime["source_progress"][0]["buffered_items"], 1)
            finally:
                manager.close()

    def test_remote_active_sensory_episode_returns_quickly_while_source_warms_in_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_active_sensory_budget")

            class _SlowSensoryStream:
                def __init__(self, episodes, *, first_delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episodes = [
                SensoryEpisode(
                    text=(
                        "Environmental sound sample with water and footsteps moving across a path. "
                        "Environmental sound sample with water and footsteps moving across a path. "
                        "Environmental sound sample with water and footsteps moving across a path. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "water footsteps sample"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"slow-remote-sensory",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.25] * 8,
                    },
                )
            ]
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowSensoryStream(episodes, first_delay_seconds=0.3),
                max_buffer=2,
                name="slow-remote-active-sensory",
            )

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(sensory_runtime_module, "DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        started = time.perf_counter()
                        first = manager._run_real_sensory_episode_locked()
                        first_ms = (time.perf_counter() - started) * 1000.0
                        self.assertIsNone(first)
                        self.assertLess(first_ms, 200.0)

                        time.sleep(0.35)
                        second = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(second)
                    self.assertEqual(second["sources"][0]["name"], "environmental_audio")
            finally:
                manager.close()

    def test_remote_sensory_promotion_warms_runtime_without_repeated_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_sensory_promotion")

            class _SlowSensoryStream:
                def __init__(self, episodes, *, first_delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episode = SensoryEpisode(
                text=(
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                ),
                visual_spikes=None,
                audio_spikes=torch.ones(64),
                metadata={"caption": "water footsteps sample"},
                audio_preview={
                    "mime_type": "audio/wav",
                    "bytes": b"promoted-remote-sensory",
                    "sample_rate": 16000,
                    "duration_s": 1.0,
                    "waveform": [0.25] * 8,
                },
            )
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowSensoryStream([episode], first_delay_seconds=0.3),
                max_buffer=2,
                name="slow-remote-promotion-sensory",
            )

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )["terminus_runtime"]
                    self.assertEqual(configured["sensory"]["startup_state"], "cold")

                    deadline = time.time() + 1.5
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["sensory"]["warm_ready"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["sensory"]["warm_ready"])
                    self.assertGreaterEqual(runtime["sensory"]["total_buffered_items"], 1)
                    event_types = [event.get("type") for event in runtime["recent_events"]]
                    self.assertIn("remote_warm_promotion_started", event_types)
                    self.assertIn("remote_warm_promotion_completed", event_types)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(summary["sources"][0]["name"], "environmental_audio")
            finally:
                manager.close()

    def test_remote_sensory_first_rows_bootstrap_warms_runtime_quickly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_sensory_first_rows_bootstrap")

            class _SlowSensoryStream:
                def __init__(self, episodes, *, first_delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episode = SensoryEpisode(
                text=(
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                ),
                visual_spikes=None,
                audio_spikes=torch.ones(64),
                metadata={"caption": "water footsteps sample"},
                audio_preview={
                    "mime_type": "audio/wav",
                    "bytes": b"bootstrap-remote-sensory",
                    "sample_rate": 16000,
                    "duration_s": 1.0,
                    "waveform": [0.25] * 8,
                },
            )
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowSensoryStream([episode], first_delay_seconds=1.0),
                max_buffer=2,
                name="slow-remote-first-rows-bootstrap-sensory",
            )

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ), patch.object(
                    runtime_prewarm_module,
                    "load_hf_first_rows",
                    return_value=[{"caption": "Water pours while a woman talks nearby", "audio": [{"src": "https://example.com/audio.wav"}], "youtube_id": "abc123xyz99", "audiocap_id": 7, "start_time": 130}],
                ), patch.object(
                    runtime_prewarm_module,
                    "bootstrap_sensory_episode_from_row",
                    return_value=episode,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )["terminus_runtime"]
                    self.assertEqual(configured["sensory"]["startup_state"], "cold")

                    deadline = time.time() + 1.0
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["sensory"]["warm_ready"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["sensory"]["warm_ready"])
                    event_types = [event.get("type") for event in runtime["recent_events"]]
                    self.assertIn("remote_sensory_bootstrap_applied", event_types)
                    self.assertIn("remote_warm_promotion_started", event_types)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(summary["sources"][0]["name"], "environmental_audio")
            finally:
                manager.close()

    def test_remote_sensory_bootstrap_timeout_does_not_block_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_remote_sensory_bootstrap_timeout_close")

            class _SlowSensoryStream:
                def __init__(self, episodes, *, first_delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episode = SensoryEpisode(
                text=(
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                ),
                visual_spikes=None,
                audio_spikes=torch.ones(64),
                metadata={"caption": "water footsteps sample"},
                audio_preview={
                    "mime_type": "audio/wav",
                    "bytes": b"slow-bootstrap-close",
                    "sample_rate": 16000,
                    "duration_s": 1.0,
                    "waveform": [0.25] * 8,
                },
            )
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowSensoryStream([episode], first_delay_seconds=1.0),
                max_buffer=2,
                name="slow-remote-sensory-bootstrap-timeout",
            )
            bootstrap_started = Event()
            closed = False

            def _slow_first_rows(*_args, **_kwargs):
                bootstrap_started.set()
                time.sleep(0.3)
                return [{"caption": "slow sensory bootstrap row", "audio": [{"src": "https://example.com/audio.wav"}]}]

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(runtime_prewarm_module, "DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ), patch.object(
                    runtime_prewarm_module,
                    "load_hf_first_rows",
                    side_effect=_slow_first_rows,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )
                    self.assertTrue(bootstrap_started.wait(0.2))

                    deadline = time.time() + 0.4
                    event_types: list[str | None] = []
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        event_types = [event.get("type") for event in runtime["recent_events"]]
                        if "remote_sensory_bootstrap_timed_out" in event_types:
                            break
                        time.sleep(0.02)
                    self.assertIn("remote_sensory_bootstrap_timed_out", event_types)

                    started = time.perf_counter()
                    manager.close()
                    closed = True
                    close_ms = (time.perf_counter() - started) * 1000.0
                    self.assertLess(close_ms, 200.0)
            finally:
                if not closed:
                    manager.close()

    def test_remote_s1_bootstrap_does_not_wait_for_slow_recaption_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            sensory_module._reset_s1_recaption_index_runtime()
            manager = _build_manager(root, test_case="service_manager_remote_s1_bootstrap_fast_fallback")

            class _SlowSensoryStream:
                def __init__(self, episodes, *, first_delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._first_delay_seconds = float(first_delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index == 0:
                        time.sleep(self._first_delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            delayed_episode = SensoryEpisode(
                text="Scientific figure fig004 from paper 0501163 in archive bucket 0705.",
                visual_spikes=torch.ones(64),
                audio_spikes=None,
                metadata={"figure_id": "fig004", "text_source": "image_path_fallback"},
                visual_preview={
                    "mime_type": "image/png",
                    "bytes": b"slow-visual-preview",
                    "width": 8,
                    "height": 8,
                },
            )
            wrapped_stream = BackgroundPrefetchIterator(
                _SlowSensoryStream([delayed_episode], first_delay_seconds=1.0),
                max_buffer=2,
                name="slow-remote-s1-bootstrap-sensory",
            )
            loaded = Event()
            recaption_index = {
                "images/0705/0501163.tar.gz/fig004.png": {
                    "title": "Why do we live in 3+1 dimensions?",
                    "recaption": "A contour plot of a potential function with curved contours.",
                    "categories": "hep-th astro-ph gr-qc hep-ph",
                }
            }

            def _slow_index_loader(*_args, **_kwargs):
                time.sleep(0.75)
                loaded.set()
                return recaption_index

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": True, "cross_modal_audio_accepted": False}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=wrapped_stream,
                ), patch.object(
                    runtime_prewarm_module,
                    "load_hf_first_rows",
                    return_value=[{"png": {"src": "https://example.com/image.jpg"}, "__key__": "0705/0501163.tar.gz/fig004"}],
                ), patch(
                    "hecsn.service.terminus_sensory._download_binary_asset",
                    return_value=_png_bytes(),
                ), patch(
                    "hecsn.service.terminus_sensory._load_s1_recaption_index",
                    side_effect=_slow_index_loader,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "science_figures",
                                    "adapter": "s1_mmalign",
                                    "source": "ScienceOne-AI/S1-MMAlign",
                                    "split": "train",
                                    "year_prefixes": ["07"],
                                    "topic_terms": ["scientific figure diagram plot graph chart"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )["terminus_runtime"]
                    self.assertEqual(configured["sensory"]["startup_state"], "cold")

                    start = time.perf_counter()
                    deadline = start + 0.6
                    runtime = configured
                    while time.perf_counter() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if runtime["sensory"]["warm_ready"]:
                            break
                        time.sleep(0.02)
                    warm_elapsed = time.perf_counter() - start

                    self.assertTrue(runtime["sensory"]["warm_ready"])
                    self.assertLess(warm_elapsed, 0.7)
                    self.assertGreaterEqual(runtime["sensory"]["total_buffered_items"], 1)
                    event_types = [event.get("type") for event in runtime["recent_events"]]
                    self.assertIn("remote_sensory_bootstrap_applied", event_types)
                    self.assertIn("remote_warm_promotion_started", event_types)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(summary["sources"][0]["name"], "science_figures")
                    self.assertTrue(loaded.wait(2.0))
            finally:
                manager.close()
                sensory_module._reset_s1_recaption_index_runtime()

    def test_remote_sensory_cache_restore_makes_runtime_warm_and_usable_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            manager = _build_manager(root, test_case="service_manager_remote_sensory_cache_seed")

            class _FastSensoryStream:
                def __init__(self, episodes) -> None:
                    self._episodes = list(episodes)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episode = SensoryEpisode(
                text=(
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                    "Environmental sound sample with water and footsteps moving across a path. "
                ),
                visual_spikes=None,
                audio_spikes=torch.ones(64),
                metadata={"caption": "water footsteps sample"},
                audio_preview={
                    "mime_type": "audio/wav",
                    "bytes": b"cached-remote-sensory",
                    "sample_rate": 16000,
                    "duration_s": 1.0,
                    "waveform": [0.25] * 8,
                },
            )
            seed_stream = _FastSensoryStream([episode])

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=seed_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )
                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]
                    self.assertIsNotNone(summary)
            finally:
                manager.close()

            manager = _build_manager(root, test_case="service_manager_remote_sensory_cache_restore")

            class _UnusedSensoryStream(_FastSensoryStream):
                def __next__(self):
                    self.next_calls += 1
                    raise AssertionError("remote sensory stream should not be touched before cached work is consumed")

            restored_stream = _UnusedSensoryStream([episode])

            def _restored_fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=restored_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": False,
                        },
                    )["terminus_runtime"]
                    self.assertEqual(configured["sensory"]["startup_state"], "warm")
                    self.assertTrue(configured["sensory"]["warm_ready"])
                    self.assertGreaterEqual(configured["sensory"]["total_buffered_items"], 1)
                    event_types = [event.get("type") for event in configured["recent_events"]]
                    self.assertIn("sensory_cache_restored", event_types)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _restored_fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(restored_stream.next_calls, 0)
            finally:
                manager.close()

    def test_sensory_warm_queue_tolerates_follow_up_stall_until_buffer_depletes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_sensory_stall_tolerance")

            class _DelayedSensoryStream:
                def __init__(self, episodes, *, delay_after: int, delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episodes = [
                SensoryEpisode(
                    text=(
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": f"water footsteps sample {idx}"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": f"episode-{idx}".encode("utf-8"),
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.25] * 8,
                    },
                )
                for idx in range(4)
            ]
            delayed_stream = _DelayedSensoryStream(episodes, delay_after=2, delay_seconds=0.12)

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=delayed_stream,
                ):
                    manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 1,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 2,
                            "prewarm_on_startup": False,
                        },
                    )

                original_train_step = manager._trainer.train_step
                manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                try:
                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    first = manager._run_real_sensory_episode_locked()
                    calls_after_first = delayed_stream.next_calls

                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    second = manager._run_real_sensory_episode_locked()
                    calls_after_second = delayed_stream.next_calls

                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    third = manager._run_real_sensory_episode_locked()
                    calls_after_third = delayed_stream.next_calls
                finally:
                    manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertIsNotNone(third)
                self.assertEqual(calls_after_first, 2)
                self.assertEqual(calls_after_second, calls_after_first)
                self.assertGreater(calls_after_third, calls_after_second)

                sensory_runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]["sensory"]
                self.assertGreaterEqual(sensory_runtime["queue_hits"], 1)
                self.assertEqual(sensory_runtime["source_progress"][0]["last_buffer_items_served"], 1)
                self.assertEqual(sensory_runtime["source_progress"][0]["prefetch_events"], 2)
            finally:
                manager.close()

    def test_sensory_prewarm_budget_exhaustion_surfaces_partial_warm_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_sensory_prewarm_budget")

            class _DelayedSensoryStream:
                def __init__(self, episodes, *, delay_after: int, delay_seconds: float) -> None:
                    self._episodes = list(episodes)
                    self._delay_after = int(delay_after)
                    self._delay_seconds = float(delay_seconds)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= self._delay_after:
                        time.sleep(self._delay_seconds)
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            episodes = [
                SensoryEpisode(
                    text=(
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                        f"Environmental sound sample {idx} with water and footsteps moving across a path. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": f"water footsteps sample {idx}"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": f"budget-episode-{idx}".encode("utf-8"),
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.25] * 8,
                    },
                )
                for idx in range(4)
            ]
            active_stream = _DelayedSensoryStream(episodes, delay_after=1, delay_seconds=0.12)
            prewarm_stream = _DelayedSensoryStream(list(episodes), delay_after=1, delay_seconds=0.12)

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 3,
                            "prewarm_on_startup": True,
                            "prewarm_max_seconds": 0.05,
                        },
                    )["terminus_runtime"]

                    self.assertIn(configured["sensory"]["startup_state"], {"warming", "warm"})
                    deadline = time.time() + 1.5
                    runtime = configured
                    while time.time() < deadline:
                        runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                        if not runtime["sensory"]["prewarm_running"]:
                            break
                        time.sleep(0.02)

                    self.assertTrue(runtime["sensory"]["prewarm_budget_exhausted"])
                    self.assertTrue(runtime["sensory"]["warm_ready"])
                    self.assertFalse(runtime["sensory"]["full_warm_ready"])
                    self.assertGreater(runtime["sensory"]["ready_source_count"], 0)
                    self.assertEqual(runtime["sensory"]["full_queue_source_count"], 0)
                    self.assertAlmostEqual(float(runtime["sensory"]["prewarm_max_seconds"]), 0.05, places=6)
                    self.assertGreater(prewarm_stream.next_calls, 0)
                    self.assertLess(prewarm_stream.next_calls, 3)
                    self.assertEqual(active_stream.next_calls, 0)

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        summary = manager._run_real_sensory_episode_locked()
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertEqual(prewarm_stream.next_calls, 2)
                    self.assertEqual(active_stream.next_calls, 0)
                    sensory_runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]["sensory"]
                    self.assertGreaterEqual(sensory_runtime["source_progress"][0]["queue_hits"], 1)
            finally:
                manager.close()

    def test_immediate_sensory_episode_does_not_wait_behind_blocking_startup_prewarm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_isolated_sensory_prewarm")

            class _FastSensoryStream:
                def __init__(self, episodes) -> None:
                    self._episodes = list(episodes)
                    self._index = 0
                    self.next_calls = 0

                def __iter__(self):
                    return self

                def __next__(self):
                    self.next_calls += 1
                    if self._index >= len(self._episodes):
                        raise StopIteration
                    item = self._episodes[self._index]
                    self._index += 1
                    return item

            class _BlockingSensoryStream(_FastSensoryStream):
                def __init__(self, episodes, started: Event, delay_seconds: float) -> None:
                    super().__init__(episodes)
                    self._started = started
                    self._delay_seconds = float(delay_seconds)

                def __next__(self):
                    self._started.set()
                    time.sleep(self._delay_seconds)
                    return super().__next__()

            active_episodes = [
                SensoryEpisode(
                    text=(
                        "Fast sensory sample with water and footsteps near a path. "
                        "Fast sensory sample with water and footsteps near a path. "
                        "Fast sensory sample with water and footsteps near a path. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "fast water footsteps"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"fast-episode",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.3] * 8,
                    },
                )
            ]
            prewarm_episodes = [
                SensoryEpisode(
                    text=(
                        "Slow prewarm sensory sample with rain and distant birds. "
                        "Slow prewarm sensory sample with rain and distant birds. "
                        "Slow prewarm sensory sample with rain and distant birds. "
                    ),
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "slow rain birds"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"slow-episode",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.2] * 8,
                    },
                )
            ]
            active_stream = _FastSensoryStream(active_episodes)
            prewarm_started = Event()
            prewarm_stream = _BlockingSensoryStream(prewarm_episodes, started=prewarm_started, delay_seconds=0.45)

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._trainer.config.enable_cross_modal = True
                with patch.object(runtime_prewarm_module, "DEFAULT_REMOTE_PREWARM_GRACE_SECONDS", 0.05), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=active_stream,
                ), patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_from_spec",
                    autospec=True,
                    return_value=prewarm_stream,
                ):
                    configured = manager.runtime_facade.configure_terminus(
                        source_bank=[],
                        sensory={
                            "enabled": True,
                            "source_bank": [
                                {
                                    "name": "environmental_audio",
                                    "adapter": "audiocaps",
                                    "source": "OpenSound/AudioCaps",
                                    "split": "train",
                                    "topic_terms": ["audio sound water wind footsteps environment"],
                                }
                            ],
                            "episode_interval_tokens": 256,
                            "items_per_episode": 1,
                            "base_windows_per_item": 1,
                            "max_windows_per_item": 2,
                            "confidence_window_gain": 0.0,
                            "semantic_window_gain": 0.0,
                            "item_retrieval_lookahead": 1,
                            "item_retrieval_semantic_weight": 1.0,
                            "modality_target_confidence": 0.70,
                            "observation_salience": 0.82,
                            "cooldown_seconds": 1.0,
                            "repeat_sources": False,
                            "queue_target_items": 1,
                            "prewarm_on_startup": True,
                            "prewarm_max_seconds": 5.0,
                        },
                    )["terminus_runtime"]

                    self.assertIn(configured["sensory"]["startup_state"], {"warming", "warm"})
                    self.assertFalse(prewarm_started.is_set())

                    manager._trainer.token_count = 512
                    manager._last_real_sensory_episode_token_count = 0
                    manager._last_real_sensory_episode_time = 0.0
                    original_train_step = manager._trainer.train_step
                    manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                    try:
                        started = time.perf_counter()
                        summary = manager._run_real_sensory_episode_locked()
                        run_ms = (time.perf_counter() - started) * 1000.0
                    finally:
                        manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                    self.assertIsNotNone(summary)
                    self.assertLess(run_ms, 300.0)
                    self.assertEqual(active_stream.next_calls, 1)
                    self.assertEqual(summary["sources"][0]["name"], "environmental_audio")
                    self.assertFalse(prewarm_started.wait(0.2))
                    event_types = [event.get("type") for event in manager.runtime_facade.terminus_status()["terminus_runtime"]["recent_events"]]
                    self.assertIn("sensory_prewarm_skipped_after_active_execution", event_types)
            finally:
                manager.close()

    def test_sensory_selection_tracks_exploration_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_sensory_semantics")

            figure_episode = SensoryEpisode(
                text="A scientific figure showing a lattice phase transition.",
                visual_spikes=torch.ones(64),
                audio_spikes=None,
                metadata={"title": "phase diagram", "categories": "physics"},
            )
            audio_episode = SensoryEpisode(
                text="Water pours while footsteps and wind are audible.",
                visual_spikes=None,
                audio_spikes=torch.ones(64),
                metadata={"caption": "water and footsteps"},
            )

            def _stream_for_spec(_self, spec):
                if spec["adapter"] == "s1_mmalign":
                    return iter([figure_episode])
                return iter([audio_episode])

            try:
                manager._interaction_pipeline.record_recent_query_gap(
                    query_text="scientific diagram of lattice phases",
                    gap_plan={
                        "unsupported_terms": ["scientific", "diagram", "lattice", "phases"],
                        "gap_terms": [
                            {"term": "scientific", "weight": 2.0},
                            {"term": "diagram", "weight": 2.0},
                            {"term": "lattice", "weight": 2.0},
                            {"term": "phases", "weight": 2.0},
                        ],
                        "grounded_fraction": 0.0,
                    },
                    source="test",
                )
                manager._brain_config["sensory"] = {
                    "enabled": True,
                    "source_bank": [
                        {
                            "name": "science_figures",
                            "adapter": "s1_mmalign",
                            "topic_terms": ["scientific figure", "diagram graph lattice phase"],
                        },
                        {
                            "name": "environmental_audio",
                            "adapter": "audiocaps",
                            "topic_terms": ["audio sound water wind footsteps environment"],
                        },
                    ],
                    "episode_interval_tokens": 1,
                    "items_per_episode": 2,
                    "base_windows_per_item": 2,
                    "max_windows_per_item": 5,
                    "confidence_window_gain": 0.0,
                    "semantic_window_gain": 3.0,
                    "modality_target_confidence": 0.70,
                    "observation_salience": 0.80,
                    "cooldown_seconds": 1.0,
                    "repeat_sources": True,
                }
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    side_effect=_stream_for_spec,
                ):
                    manager._rebuild_brain_sources_locked()

                selection = manager._select_sensory_runtime_locked(set())
                self.assertIsNotNone(selection)
                idx, runtime, semantic_match, modality_need, selection_score = selection  # type: ignore[misc]
                self.assertEqual(runtime.name, "science_figures")
                self.assertGreater(semantic_match, 0.5)
                self.assertGreater(selection_score, 0.3)
                figure_budget = manager._sensory_window_budget_locked(
                    runtime,
                    semantic_match=semantic_match,
                    modality_need=modality_need,
                )
                self.assertEqual(figure_budget, 5)
                runtime_snapshot = manager.runtime_facade.terminus_status()["terminus_runtime"]
                self.assertIn("scientific", runtime_snapshot["multimodal"]["focus_terms"])

                manager._interaction_pipeline.record_recent_query_gap(
                    query_text="environmental sound of water and footsteps",
                    gap_plan={
                        "unsupported_terms": ["environmental", "sound", "water", "footsteps"],
                        "gap_terms": [
                            {"term": "environmental", "weight": 2.0},
                            {"term": "sound", "weight": 2.0},
                            {"term": "water", "weight": 2.0},
                            {"term": "footsteps", "weight": 2.0},
                        ],
                        "grounded_fraction": 0.0,
                    },
                    source="test",
                )
                selection2 = manager._select_sensory_runtime_locked(set())
                self.assertIsNotNone(selection2)
                idx2, runtime2, semantic_match2, modality_need2, selection_score2 = selection2  # type: ignore[misc]
                self.assertEqual(runtime2.name, "environmental_audio")
                self.assertGreater(semantic_match2, 0.5)
                self.assertGreater(selection_score2, 0.3)
                audio_budget = manager._sensory_window_budget_locked(
                    runtime2,
                    semantic_match=semantic_match2,
                    modality_need=modality_need2,
                )
                self.assertEqual(audio_budget, 5)
                runtime_snapshot = manager.runtime_facade.terminus_status()["terminus_runtime"]
                self.assertIn("environmental", runtime_snapshot["multimodal"]["focus_terms"])
            finally:
                manager.close()

    def test_real_sensory_item_retrieval_shortlists_within_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_sensory_item_retrieval")

            episodes = [
                SensoryEpisode(
                    text="Industrial machinery hums inside a factory hall.",
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "factory machinery"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"factory",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.1] * 8,
                    },
                ),
                SensoryEpisode(
                    text="Water splashes while wind and footsteps move through an outdoor path.",
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "water wind footsteps"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"waterwindsteps",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.4] * 8,
                    },
                ),
                SensoryEpisode(
                    text="Soft birdsong in a quiet forest.",
                    visual_spikes=None,
                    audio_spikes=torch.ones(64),
                    metadata={"caption": "forest birdsong"},
                    audio_preview={
                        "mime_type": "audio/wav",
                        "bytes": b"birdsong",
                        "sample_rate": 16000,
                        "duration_s": 1.0,
                        "waveform": [0.2] * 8,
                    },
                ),
            ]

            def _fake_train_step(pattern, **kwargs):
                manager._trainer.token_count += 1
                return {"cross_modal_visual_accepted": False, "cross_modal_audio_accepted": True}

            try:
                manager._interaction_pipeline.record_recent_query_gap(
                    query_text="environmental sound of water wind and footsteps",
                    gap_plan={
                        "unsupported_terms": ["environmental", "sound", "water", "wind", "footsteps"],
                        "gap_terms": [
                            {"term": "environmental", "weight": 2.0},
                            {"term": "sound", "weight": 2.0},
                            {"term": "water", "weight": 2.0},
                            {"term": "wind", "weight": 2.0},
                            {"term": "footsteps", "weight": 2.0},
                        ],
                        "grounded_fraction": 0.0,
                    },
                    source="test",
                )
                manager._trainer.config.enable_cross_modal = True
                manager._brain_config["sensory"] = {
                    "enabled": True,
                    "source_bank": [
                        {
                            "name": "environmental_audio",
                            "adapter": "audiocaps",
                            "source": "OpenSound/AudioCaps",
                            "split": "train",
                            "topic_terms": ["audio sound water wind footsteps environment"],
                        }
                    ],
                    "episode_interval_tokens": 1,
                    "items_per_episode": 1,
                    "base_windows_per_item": 1,
                    "max_windows_per_item": 2,
                    "confidence_window_gain": 0.0,
                    "semantic_window_gain": 0.0,
                    "item_retrieval_lookahead": 3,
                    "item_retrieval_semantic_weight": 0.9,
                    "modality_target_confidence": 0.70,
                    "observation_salience": 0.80,
                    "cooldown_seconds": 1.0,
                    "repeat_sources": True,
                }
                with patch.object(
                    RuntimeSources,
                    "_build_sensory_stream_locked",
                    autospec=True,
                    return_value=iter(episodes),
                ):
                    manager._rebuild_brain_sources_locked()
                manager._trainer.token_count = 64
                manager._last_real_sensory_episode_token_count = 0
                manager._last_real_sensory_episode_time = 0.0
                original_train_step = manager._trainer.train_step
                manager._trainer.train_step = _fake_train_step  # type: ignore[assignment]
                try:
                    summary = manager._run_real_sensory_episode_locked()
                finally:
                    manager._trainer.train_step = original_train_step  # type: ignore[assignment]

                self.assertIsNotNone(summary)
                self.assertEqual(summary["sources"][0]["name"], "environmental_audio")
                self.assertGreater(summary["sources"][0]["item_semantic_match"], 0.5)
                self.assertEqual(summary["sources"][0]["item_candidates_considered"], 3)
                grounded = summary["sources"][0]["grounded_observation"]
                self.assertEqual(grounded["observation_sink"], "subcortex_grounded_sensory_observation")
                self.assertNotIn("retired_loop_mirrored", grounded)
                grounded_terms = " ".join(list(grounded["topics"]) + list(grounded["metadata"]["focus_terms"]))
                self.assertIn("water", grounded_terms)
                self.assertIn("wind", grounded_terms)
                self.assertIn("footsteps", grounded["content"].lower())

                previews = manager.runtime_facade.sensory_previews(limit=1)
                preview = previews["previews"][0]
                self.assertIn("Water splashes", preview["text"])
                self.assertGreater(preview["item_semantic_match"], 0.5)
                self.assertEqual(preview["item_candidates_considered"], 3)

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                source_progress = runtime["sensory"]["source_progress"][0]
                self.assertGreater(source_progress["last_item_semantic_match"], 0.5)
                self.assertEqual(source_progress["last_item_candidates_considered"], 3)
                self.assertEqual(source_progress["last_item_retrieval_lookahead"], 3)
            finally:
                manager.close()

    def test_terminus_runtime_reports_autonomy_trigger_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_observability")
            source_path = root / "terminus_source.txt"
            source_path.write_text("active source seeking telemetry " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 50,
                    },
                )
                manager.runtime_facade.terminus_tick(steps=2)
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["candidate_count"], 1)
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["name"], "candidate_source")
                self.assertEqual(runtime["autonomy"]["candidate_names"], ["candidate_source"])
                self.assertFalse(runtime["autonomy"]["trigger_ready"])
                self.assertLess(runtime["autonomy"]["tokens_until_trigger"], 50)
                self.assertIsNone(runtime["autonomy"]["last_acquisition_summary"])
            finally:
                manager.close()

    def test_terminus_runtime_persists_recent_query_gap_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_recent_query_gap_focus")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 50,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["recent_query_gaps"][0]["query_text"], "submarine buoyancy ballast")
                self.assertIn("submarine", runtime["autonomy"]["focus_plan"]["unsupported_terms"])

                saved = manager.runtime_facade.save_checkpoint(str(root / "terminus_focus.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    restored_runtime = restored.runtime_facade.terminus_status()["terminus_runtime"]

                    self.assertEqual(
                        restored_runtime["autonomy"]["recent_query_gaps"][0]["query_text"],
                        "submarine buoyancy ballast",
                    )
                    self.assertIn("submarine", restored_runtime["autonomy"]["focus_plan"]["unsupported_terms"])
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_terminus_focus_plan_preserves_recent_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_recent_weak_concepts")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 50,
                    },
                )
                with manager._lock:
                    manager._interaction_pipeline.record_recent_query_gap(
                        query_text="submarine buoyancy ballast",
                        source="query",
                        gap_plan={
                            "unsupported_terms": ["submarine"],
                            "gap_terms": [{"term": "submarine", "weight": 2.0}],
                            "retrieval_queries": ["submarine buoyancy ballast"],
                            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
                            "weak_concepts": [
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.6,
                                    "drift": 0.2,
                                    "top_terms": ["submarine", "ballast", "buoyancy"],
                                    "match_count": 1,
                                }
                            ],
                            "grounded_fraction": 0.0,
                        },
                    )
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]

                self.assertEqual(
                    runtime["autonomy"]["recent_query_gaps"][0]["weak_concepts"][0]["label"],
                    "buoyancy control",
                )
                self.assertEqual(
                    runtime["autonomy"]["focus_plan"]["weak_concepts"][0]["label"],
                    "buoyancy control",
                )
                self.assertEqual(
                    runtime["autonomy"]["focus_plan"]["weak_concepts"][0]["top_terms"],
                    ["submarine", "ballast", "buoyancy"],
                )
            finally:
                manager.close()

    def test_fully_grounded_query_does_not_persist_recent_gap_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_fully_grounded_query_gap")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "trigger_interval_tokens": 50,
                    },
                )
                with manager._lock:
                    manager._interaction_pipeline.record_recent_query_gap(
                        query_text="what corrects submarine trim",
                        source="query",
                        gap_plan={
                            "unsupported_terms": [],
                            "gap_terms": [{"term": "submarine", "weight": 1.0}],
                            "retrieval_queries": ["submarine ballast trim"],
                            "follow_up_questions": ["What grounded evidence would reduce drift for buoyancy control?"],
                            "weak_concepts": [
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.4,
                                    "drift": 0.3,
                                    "top_terms": ["submarine", "ballast", "trim"],
                                    "match_count": 2,
                                }
                            ],
                            "grounded_fraction": 1.0,
                        },
                    )
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["recent_query_gaps"], [])
                self.assertIsNone(runtime["autonomy"]["focus_plan"])
            finally:
                manager.close()

    def test_fully_grounded_query_skips_next_autonomy_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_grounded_query_autonomy_skip")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "trigger_interval_tokens": 1,
                    },
                )
                with manager._lock:
                    manager._interaction_pipeline.record_recent_query_gap(
                        query_text="what corrects submarine trim",
                        source="query",
                        gap_plan={
                            "unsupported_terms": [],
                            "gap_terms": [{"term": "submarine", "weight": 1.0}],
                            "retrieval_queries": ["submarine ballast trim"],
                            "follow_up_questions": ["What grounded evidence would reduce drift for buoyancy control?"],
                            "weak_concepts": [
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.4,
                                    "drift": 0.3,
                                    "top_terms": ["submarine", "ballast", "trim"],
                                    "match_count": 2,
                                }
                            ],
                            "grounded_fraction": 1.0,
                        },
                    )
                with patch("hecsn.service.brain_runtime.run_live_acquisition") as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                mocked_acquire.assert_not_called()
                self.assertIsNone(runtime["autonomy"]["last_acquisition_summary"])
            finally:
                manager.close()

    def test_focus_pressure_accelerates_autonomy_trigger_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_adaptive_autonomy_balance")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["arxiv", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "provider_curriculum": {
                            "wikipedia": {
                                "attempts": 2,
                                "commits": 2,
                                "successes": 2,
                                "diagnostic_gain_ema": 0.22,
                                "semantic_relevance_ema": 0.85,
                                "answerability_gain_ema": 0.42,
                                "uncertainty_reduction_ema": 0.31,
                                "weak_concept_stabilization_ema": 0.24,
                                "topic_terms": {"submarine": 1.0, "ballast": 0.9, "trim": 0.7},
                                "topic_families": {
                                    "submarine": {
                                        "commits": 2,
                                        "successes": 2,
                                        "semantic_relevance_ema": 0.85,
                                        "answerability_gain_ema": 0.42,
                                        "uncertainty_reduction_ema": 0.31,
                                        "weak_concept_stabilization_ema": 0.24,
                                    }
                                },
                            },
                            "arxiv": {
                                "attempts": 2,
                                "commits": 2,
                                "successes": 2,
                                "diagnostic_gain_ema": 0.26,
                                "semantic_relevance_ema": 0.88,
                                "answerability_gain_ema": 0.39,
                                "uncertainty_reduction_ema": 0.28,
                                "weak_concept_stabilization_ema": 0.18,
                                "topic_terms": {"protein": 1.0, "enzyme": 0.8},
                                "topic_families": {
                                    "protein": {
                                        "commits": 2,
                                        "successes": 2,
                                        "semantic_relevance_ema": 0.88,
                                        "answerability_gain_ema": 0.39,
                                        "uncertainty_reduction_ema": 0.28,
                                        "weak_concept_stabilization_ema": 0.18,
                                    }
                                },
                            },
                        },
                        "trigger_interval_tokens": 24,
                        "candidate_train_tokens": 96,
                        "probe_tokens": 48,
                        "acquisition_tokens": 48,
                        "acquisition_slots": 1,
                    },
                )
                manager.runtime_facade.query(query_text="What corrects submarine ballast trim?", top_k_memories=6)
                manager._brain_recent_query_gaps[0]["weak_concepts"] = [
                    {
                        "label": "buoyancy control",
                        "weakness": 0.92,
                        "uncertainty": 0.84,
                        "drift": 0.22,
                        "top_terms": ["submarine", "ballast", "trim"],
                        "match_count": 1,
                    }
                ]
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {"unsupported_terms": ["submarine", "ballast", "trim"]},
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                adaptive = runtime["autonomy"]["adaptive_learning"]
                kwargs = mocked_acquire.call_args.kwargs

                mocked_acquire.assert_called_once()
                self.assertLess(int(adaptive["effective_trigger_interval_tokens"]), 24)
                self.assertGreater(int(adaptive["effective_acquisition_tokens"]), 48)
                self.assertEqual(int(adaptive["effective_acquisition_slots"]), 2)
                self.assertGreater(float(adaptive["focus_pressure"]), 0.5)
                self.assertEqual(adaptive["provider_priority_details"]["provider"], "wikipedia")
                self.assertEqual(int(kwargs["acquisition_tokens"]), int(adaptive["effective_acquisition_tokens"]))
                self.assertEqual(int(kwargs["acquisition_slots"]), int(adaptive["effective_acquisition_slots"]))
                self.assertGreater(float(runtime["autonomy"]["adaptive_learning"]["targeted_learning_share_target"]), 0.5)
            finally:
                manager.close()

    def test_terminus_autonomy_uses_concept_store_focus_without_recent_query_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_concept_store_focus")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text("submarine ballast buoyancy pressure " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(candidate_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.feed(text=("submarine ballast buoyancy pressure control " * 16).strip())
                concept_store = manager.runtime_facade.status()["concept_store"]
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]

                self.assertGreater(concept_store["growth"]["expansion_events"], 0)
                self.assertGreater(concept_store["abstraction"]["requested_output_dim"], 8)
                self.assertEqual(runtime["autonomy"]["recent_query_gaps"], [])
                self.assertIsNotNone(runtime["autonomy"]["focus_plan"])
                self.assertEqual(runtime["autonomy"]["focus_plan"]["planner_mode"], "concept_store_abstraction_focus")
                self.assertIn("submarine", " ".join(runtime["autonomy"]["focus_plan"]["retrieval_queries"]).lower())
                self.assertGreater(
                    runtime["autonomy"]["focus_plan"]["structural_growth"]["expansion_events"],
                    0,
                )

                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": runtime["autonomy"]["focus_plan"],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["semantic_plan"]["planner_mode"], "concept_store_abstraction_focus")
                self.assertIn("submarine", " ".join(kwargs["semantic_plan"]["retrieval_queries"]).lower())
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["metadata"]["query_text"].lower())
            finally:
                manager.close()

    def test_terminus_autonomy_surfaces_geometric_curiosity_focus_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                enable_abstraction_layer=True,
            )
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial_abstraction.pt",
                trainer,
                metadata={"test_case": "service_manager_geometric_curiosity_focus"},
            )
            manager = HECSNServiceManager(
                checkpoint_path,
                trace_dir=root / "traces",
            )
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text("river stream water current bank " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(candidate_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.feed(text=("river stream water current bank " * 12).strip())
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                autonomy = runtime["autonomy"]

                self.assertTrue(autonomy["geometric_curiosity"]["enabled"])
                self.assertTrue(autonomy["geometric_curiosity"]["has_focus_plan"])
                self.assertIsNotNone(autonomy["focus_plan"])
                self.assertIn("geometric_gaps", autonomy["focus_plan"])
                self.assertTrue(autonomy["focus_plan"]["retrieval_queries"])
            finally:
                manager.close()

    def test_terminus_live_remote_search_learns_geometric_query_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                enable_abstraction_layer=True,
            )
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial_geometric_curriculum.pt",
                trainer,
                metadata={"test_case": "service_manager_geometric_query_families"},
            )
            manager = HECSNServiceManager(
                checkpoint_path,
                trace_dir=root / "traces",
            )
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.feed(text=("river stream water current bank loan credit " * 12).strip())
                focus_plan = manager.runtime_facade.terminus_status()["terminus_runtime"]["autonomy"]["focus_plan"]
                self.assertTrue(focus_plan["geometric_gaps"])
                selected_query = str(focus_plan["retrieval_queries"][0]).strip().lower()

                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    side_effect=[
                        {
                            "policy": "active",
                            "tokens_trained_total": 32,
                            "acquired_sources": ["wikipedia_gap_source"],
                            "semantic_plan": focus_plan,
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_gap_source",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": selected_query,
                                    "selected_semantic_relevance": 0.91,
                                    "selected_gap_reduction": 0.22,
                                    "selected_diagnostic_gap_reduction": 0.31,
                                    "tokens_trained": 32,
                                    "selected_metadata": {
                                        "provider": "wikipedia",
                                        "query_text": selected_query,
                                        "semantic_relevance": 0.91,
                                        "catalog_terms": ["river current", "bank finance"],
                                    },
                                    "candidate_snapshot": {
                                        "wikipedia_gap_source": {
                                            "semantic_answerability": 0.22,
                                            "concept_uncertainty": 0.72,
                                            "concept_support": 0.18,
                                            "semantic_weak_concept_pressure": 0.76,
                                        }
                                    },
                                    "selected_semantic_answerability_after": 0.64,
                                    "selected_concept_uncertainty_after": 0.28,
                                    "selected_concept_support_after": 0.58,
                                    "selected_weak_concept_pressure_after": 0.18,
                                }
                            ],
                        },
                        {
                            "policy": "active",
                            "tokens_trained_total": 0,
                            "acquired_sources": [],
                            "semantic_plan": focus_plan,
                            "acquisition_history": [],
                        },
                    ],
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()
                    manager.runtime_facade.terminus_tick()

                first_kwargs = mocked_acquire.call_args_list[0].kwargs
                second_kwargs = mocked_acquire.call_args_list[1].kwargs
                spec = second_kwargs["candidate_bank_specs"][0]
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertGreaterEqual(
                    int(spec["catalog_queries_per_provider"]),
                    int(first_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"]),
                )
                self.assertEqual(int(spec["catalog_query_family_budget_bonus"]), 1)
                self.assertIn("catalog_provider_query_families", spec)
                self.assertIn(selected_query, spec["catalog_provider_query_families"]["wikipedia"])
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["query_family_strength"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["query_family_focus_score"]),
                    0.0,
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["query_family_query_bonus"]),
                    1,
                )
                self.assertIn(
                    selected_query,
                    provider_curriculum["ranked_providers"][0]["matched_query_families"],
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["query_families"][selected_query]["commits"]),
                    1,
                )
            finally:
                manager.close()

    def test_query_passes_query_conditioned_concept_focus_into_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_query_conditioned_abstraction")
            try:
                manager.runtime_facade.feed(text=("submarine ballast buoyancy pressure control " * 16).strip())
                captured: dict[str, object] = {}

                def _fake_build_query_result(**kwargs: object) -> dict[str, object]:
                    captured.update(kwargs)
                    return {
                        "checkpoint": "test://service-manager",
                        "checkpoint_metadata": {},
                        "config": {},
                        "feed_summary": None,
                        "context_summary": None,
                        "context_comparison": None,
                        "query_summary": {
                            "query_text": "submarine control depth",
                            "memory_matches": [],
                            "memory_episodes": [],
                            "native_decode": {"available": False},
                        },
                    }

                with patch("hecsn.service.operator_interaction.build_query_result", side_effect=_fake_build_query_result):
                    result = manager.runtime_facade.query(query_text="submarine control depth", top_k_memories=4)

                self.assertIn("ballast", " ".join(captured["retrieval_focus_terms"]).lower())
                self.assertTrue(captured["memory_priority"])
                self.assertEqual(
                    result["query_summary"]["abstraction_focus"]["planner_mode"],
                    "concept_store_abstraction_focus",
                )
                self.assertIn(
                    "submarine",
                    " ".join(result["query_summary"]["abstraction_focus"]["retrieval_queries"]).lower(),
                )
            finally:
                manager.close()

    def test_terminus_autonomy_passes_recent_query_focus_into_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_autonomy_query_focus_bridge")
            source_path = root / "terminus_source.txt"
            related_path = root / "submarine_source.txt"
            unrelated_path = root / "garden_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            related_path.write_text("submarine buoyancy ballast pressure " * 24, encoding="utf-8")
            unrelated_path.write_text("garden tomato soil sunlight " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "submarine_source",
                                "source": str(related_path),
                                "source_type": "file",
                            },
                            {
                                "name": "garden_source",
                                "source": str(unrelated_path),
                                "source_type": "file",
                            },
                        ],
                        "trigger_interval_tokens": 1,
                        "semantic_shortlist_size": 1,
                        "semantic_shortlist_gap_weight": 0.0,
                        "semantic_shortlist_affinity_weight": 1.0,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertIn("submarine", kwargs["semantic_plan"]["unsupported_terms"])
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["metadata"]["query_text"].lower())
                self.assertIn("ballast", kwargs["candidate_bank_specs"][1]["metadata"]["query_text"].lower())
            finally:
                manager.close()

    def test_terminus_autonomy_auto_enables_focus_shortlist_for_broader_candidate_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_focus_shortlist_auto")
            source_path = root / "terminus_source.txt"
            submarine_path = root / "submarine_source.txt"
            garden_path = root / "garden_source.txt"
            astronomy_path = root / "astronomy_source.txt"
            cooking_path = root / "cooking_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            submarine_path.write_text("submarine buoyancy ballast pressure " * 24, encoding="utf-8")
            garden_path.write_text("garden tomato soil sunlight " * 24, encoding="utf-8")
            astronomy_path.write_text("planet orbit telescope observatory " * 24, encoding="utf-8")
            cooking_path.write_text("kitchen simmer recipe broth " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "submarine_source",
                                "source": str(submarine_path),
                                "source_type": "file",
                            },
                            {
                                "name": "garden_source",
                                "source": str(garden_path),
                                "source_type": "file",
                            },
                            {
                                "name": "astronomy_source",
                                "source": str(astronomy_path),
                                "source_type": "file",
                            },
                            {
                                "name": "cooking_source",
                                "source": str(cooking_path),
                                "source_type": "file",
                            },
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["semantic_shortlist_size"], 2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_preserves_registry_candidate_bank_and_shortlists_estimated_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_catalog_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "registry_pool",
                                "catalog_mode": "semantic_registry",
                                "catalog_limit": 4,
                                "catalog_entries": [
                                    {
                                        "name": "submarine_source",
                                        "source": "https://example.test/submarine",
                                        "source_type": "web",
                                        "summary": "submarine buoyancy ballast pressure",
                                    },
                                    {
                                        "name": "garden_source",
                                        "source": "https://example.test/garden",
                                        "source_type": "web",
                                        "summary": "garden tomato soil sunlight",
                                    },
                                    {
                                        "name": "astronomy_source",
                                        "source": "https://example.test/astronomy",
                                        "source_type": "web",
                                        "summary": "planet orbit telescope observatory",
                                    },
                                    {
                                        "name": "cooking_source",
                                        "source": "https://example.test/cooking",
                                        "source_type": "web",
                                        "summary": "kitchen simmer recipe broth",
                                    },
                                ],
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                self.assertEqual(len(runtime["autonomy"]["candidate_bank"][0]["catalog_entries"]), 4)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "semantic_registry")
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_preserves_live_remote_search_candidate_bank_and_shortlists_estimated_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_live_remote_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["wikipedia", "arxiv"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    runtime["autonomy"]["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["wikipedia", "arxiv"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 3)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_defaults_to_live_remote_search_candidate_bank_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_default_live_remote_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_count"], 1)
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    runtime["autonomy"]["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 1)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.0)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 1.0)
            finally:
                manager.close()

    def test_terminus_default_live_remote_search_grows_query_budget_for_multiple_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_default_live_remote_query_growth")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "trigger_interval_tokens": 1,
                    },
                )
                with manager._lock:
                    manager._interaction_pipeline.record_recent_query_gap(
                        query_text="submarine buoyancy ballast",
                        source="query",
                        gap_plan={
                            "unsupported_terms": ["submarine"],
                            "gap_terms": [{"term": "submarine", "weight": 2.0}],
                            "retrieval_queries": [],
                            "follow_up_questions": [],
                            "weak_concepts": [
                                {
                                    "label": "garden soil",
                                    "weakness": 0.9,
                                    "uncertainty": 0.5,
                                    "drift": 0.1,
                                    "top_terms": ["garden", "tomato", "soil"],
                                    "match_count": 1,
                                },
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.6,
                                    "drift": 0.2,
                                    "top_terms": ["submarine", "ballast", "buoyancy"],
                                    "match_count": 1,
                                },
                            ],
                            "grounded_fraction": 0.0,
                        },
                    )
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine"],
                        },
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 3)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
            finally:
                manager.close()

    def test_terminus_live_remote_search_learns_provider_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_curriculum")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["arxiv", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.runtime_facade.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                manager._brain_recent_query_gaps[0]["weak_concepts"] = [
                    {
                        "label": "buoyancy control",
                        "weakness": 0.9,
                        "uncertainty": 0.8,
                        "drift": 0.1,
                        "top_terms": ["submarine", "ballast", "buoyancy"],
                        "match_count": 1,
                    }
                ]
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    side_effect=[
                        {
                            "policy": "active",
                            "tokens_trained_total": 32,
                            "acquired_sources": ["wikipedia_submarine_source"],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_submarine_source",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": "submarine buoyancy ballast",
                                    "selected_semantic_relevance": 0.9,
                                    "selected_gap_reduction": 0.25,
                                    "selected_diagnostic_gap_reduction": 0.35,
                                    "tokens_trained": 32,
                                     "selected_metadata": {
                                         "provider": "wikipedia",
                                         "query_text": "submarine buoyancy ballast",
                                         "semantic_relevance": 0.9,
                                         "catalog_terms": ["marine engineering", "ballast tank"],
                                      },
                                     "candidate_snapshot": {
                                         "wikipedia_submarine_source": {
                                             "semantic_answerability": 0.20,
                                             "concept_uncertainty": 0.70,
                                             "concept_support": 0.15,
                                             "semantic_weak_concept_pressure": 0.80,
                                         }
                                    },
                                     "selected_semantic_answerability_after": 0.65,
                                     "selected_concept_uncertainty_after": 0.25,
                                     "selected_concept_support_after": 0.60,
                                     "selected_weak_concept_pressure_after": 0.20,
                                   }
                               ],
                           },
                        {
                            "policy": "active",
                            "tokens_trained_total": 24,
                            "acquired_sources": ["wikipedia_submarine_follow_up"],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_submarine_follow_up",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": "submarine buoyancy ballast trim",
                                    "selected_semantic_relevance": 0.92,
                                    "selected_gap_reduction": 0.18,
                                    "selected_diagnostic_gap_reduction": 0.21,
                                    "tokens_trained": 24,
                                    "selected_metadata": {
                                        "provider": "wikipedia",
                                        "query_text": "submarine buoyancy ballast trim",
                                        "semantic_relevance": 0.92,
                                        "catalog_terms": ["marine engineering", "ballast tank", "trim control"],
                                    },
                                    "candidate_snapshot": {
                                        "wikipedia_submarine_follow_up": {
                                            "semantic_answerability": 0.45,
                                            "concept_uncertainty": 0.42,
                                            "concept_support": 0.32,
                                            "semantic_weak_concept_pressure": 0.46,
                                        }
                                    },
                                    "selected_semantic_answerability_after": 0.79,
                                    "selected_concept_uncertainty_after": 0.18,
                                    "selected_concept_support_after": 0.74,
                                    "selected_weak_concept_pressure_after": 0.10,
                                }
                            ],
                        },
                        {
                            "policy": "active",
                            "tokens_trained_total": 0,
                            "acquired_sources": [],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [],
                        },
                    ],
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()
                    manager.runtime_facade.terminus_tick()
                    manager.runtime_facade.terminus_tick()

                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                first_kwargs = mocked_acquire.call_args_list[0].kwargs
                second_kwargs = mocked_acquire.call_args_list[1].kwargs
                third_kwargs = mocked_acquire.call_args_list[2].kwargs
                self.assertEqual(
                    first_kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["arxiv", "wikipedia"],
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_providers"][0],
                    "wikipedia",
                )
                self.assertGreater(
                    float(second_kwargs["candidate_bank_specs"][0]["catalog_provider_priority_map"]["wikipedia"]),
                    float(second_kwargs["candidate_bank_specs"][0]["catalog_provider_priority_map"]["arxiv"]),
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"],
                    3,
                )
                self.assertEqual(
                    third_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"],
                    4,
                )
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]
                self.assertIsNotNone(provider_curriculum)
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertEqual(provider_curriculum["ranked_providers"][0]["successes"], 2)
                self.assertIn("submarine", provider_curriculum["focus_terms"])
                self.assertIn(
                    "marine engineering",
                    provider_curriculum["ranked_providers"][0]["topic_terms"],
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["answerability_gain_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["uncertainty_reduction_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["weak_concept_stabilization_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["utility_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["focus_alignment_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["topic_family_strength"]),
                    0.0,
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["topic_family_query_bonus"]),
                    1,
                )
                self.assertIn(
                    "submarine",
                    provider_curriculum["ranked_providers"][0]["matched_topic_families"],
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["topic_families"]["submarine"]["commits"]),
                    2,
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_provider_topic_terms"]["wikipedia"][0],
                    "submarine",
                )
                self.assertIn(
                    "marine engineering",
                    second_kwargs["candidate_bank_specs"][0]["catalog_provider_topic_terms"]["wikipedia"],
                )
                self.assertEqual(
                    int(third_kwargs["candidate_bank_specs"][0]["catalog_topic_family_budget_bonus"]),
                    1,
                )
            finally:
                manager.close()

    def test_terminus_live_remote_search_avoids_off_topic_provider_term_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_topic_filter")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.2,
                        "semantic_relevance_ema": 0.9,
                        "answerability_gain_ema": 0.4,
                        "uncertainty_reduction_ema": 0.3,
                        "weak_concept_stabilization_ema": 0.2,
                        "topic_terms": {"submarine": 1.0, "buoyancy": 0.5},
                        "topic_families": {
                            "submarine": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.9,
                                "answerability_gain_ema": 0.4,
                                "uncertainty_reduction_ema": 0.3,
                                "weak_concept_stabilization_ema": 0.2,
                            }
                        },
                    },
                    "openalex": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.25,
                        "semantic_relevance_ema": 0.92,
                        "answerability_gain_ema": 0.45,
                        "uncertainty_reduction_ema": 0.35,
                        "weak_concept_stabilization_ema": 0.25,
                        "topic_terms": {"octopus": 1.0, "jars": 0.5},
                        "topic_families": {
                            "octopus": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.92,
                                "answerability_gain_ema": 0.45,
                                "uncertainty_reduction_ema": 0.35,
                                "weak_concept_stabilization_ema": 0.25,
                            }
                        },
                    },
                }
                manager.runtime_facade.query(query_text="What opens jars and solves puzzles?", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["opens", "jars", "solves", "puzzles"],
                        },
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                self.assertEqual(spec["catalog_providers"][0], "openalex")
                self.assertNotIn("catalog_provider_topic_terms", spec)
            finally:
                manager.close()

    def test_terminus_live_remote_search_penalizes_off_topic_history_under_strong_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_focus_penalty")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["openalex", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "openalex": {
                        "attempts": 5,
                        "commits": 5,
                        "successes": 5,
                        "diagnostic_gain_ema": 0.42,
                        "semantic_relevance_ema": 0.95,
                        "answerability_gain_ema": 0.58,
                        "uncertainty_reduction_ema": 0.47,
                        "weak_concept_stabilization_ema": 0.34,
                        "topic_terms": {"octopus": 1.0, "jars": 0.8, "puzzles": 0.6},
                        "topic_families": {
                            "octopus": {
                                "commits": 5,
                                "successes": 5,
                                "semantic_relevance_ema": 0.95,
                                "answerability_gain_ema": 0.58,
                                "uncertainty_reduction_ema": 0.47,
                                "weak_concept_stabilization_ema": 0.34,
                            }
                        },
                    },
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 1,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.18,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.16,
                        "uncertainty_reduction_ema": 0.14,
                        "weak_concept_stabilization_ema": 0.10,
                        "topic_terms": {"submarine": 1.0, "ballast": 0.9, "buoyancy": 0.7},
                        "topic_families": {
                            "submarine": {
                                "commits": 1,
                                "successes": 1,
                                "semantic_relevance_ema": 0.72,
                                "answerability_gain_ema": 0.16,
                                "uncertainty_reduction_ema": 0.14,
                                "weak_concept_stabilization_ema": 0.10,
                            }
                        },
                    },
                }
                manager.runtime_facade.query(query_text="How do submarine ballast tanks control buoyancy?", top_k_memories=6)
                manager._brain_recent_query_gaps[0]["weak_concepts"] = [
                    {
                        "label": "buoyancy control",
                        "weakness": 0.88,
                        "uncertainty": 0.82,
                        "drift": 0.16,
                        "top_terms": ["submarine", "ballast", "buoyancy"],
                        "match_count": 1,
                    }
                ]
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {"unsupported_terms": ["submarine", "ballast", "buoyancy"]},
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                ranked = runtime["autonomy"]["provider_curriculum"]["ranked_providers"]
                wikipedia = next(item for item in ranked if item["provider"] == "wikipedia")
                openalex = next(item for item in ranked if item["provider"] == "openalex")

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertGreater(
                    float(spec["catalog_provider_priority_map"]["wikipedia"]),
                    float(spec["catalog_provider_priority_map"]["openalex"]),
                )
                self.assertEqual(ranked[0]["provider"], "wikipedia")
                self.assertGreater(float(wikipedia["focus_alignment"]), float(openalex["focus_alignment"]))
                self.assertGreater(float(openalex["off_topic_penalty"]), 0.0)
            finally:
                manager.close()

    def test_verified_action_outcome_reinforces_provider_grounded_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_action_outcome")
            source_path = root / "terminus_source.txt"
            notes_path = root / "notes.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            notes_path.write_text("Submarine ballast tanks fill with water to reduce buoyancy.\n", encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["openalex", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "openalex": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.18,
                        "semantic_relevance_ema": 0.64,
                        "answerability_gain_ema": 0.14,
                        "uncertainty_reduction_ema": 0.12,
                        "weak_concept_stabilization_ema": 0.08,
                        "utility_ema": 0.22,
                        "focus_alignment_ema": 0.30,
                        "grounded_outcome_ema": 0.0,
                        "last_query_text": "octopus jar puzzles",
                        "topic_terms": {"octopus": 1.0, "jars": 0.8},
                    },
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.18,
                        "semantic_relevance_ema": 0.64,
                        "answerability_gain_ema": 0.14,
                        "uncertainty_reduction_ema": 0.12,
                        "weak_concept_stabilization_ema": 0.08,
                        "utility_ema": 0.22,
                        "focus_alignment_ema": 0.30,
                        "grounded_outcome_ema": 0.0,
                        "last_query_text": "submarine ballast buoyancy",
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8},
                    },
                }
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "workspace_search",
                        "query_text": "ballast tanks",
                        "predicted_outcome": "I expect workspace search to find grounded submarine ballast evidence.",
                    },
                    trigger_reason="query_gap_auto_search",
                    trigger_query_text="How do submarine ballast tanks control buoyancy?",
                )
                runtime = result["terminus_runtime"]
                ranked = runtime["autonomy"]["provider_curriculum"]["ranked_providers"]
                wikipedia = next(item for item in ranked if item["provider"] == "wikipedia")
                openalex = next(item for item in ranked if item["provider"] == "openalex")

                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertGreater(float(wikipedia["grounded_outcome_ema"]), 0.0)
                self.assertEqual(float(openalex["grounded_outcome_ema"]), 0.0)
                self.assertGreater(float(wikipedia["utility_ema"]), float(openalex["utility_ema"]))
            finally:
                manager.close()

    def test_response_selected_evidence_exposes_provider_provenance_and_reinforces_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_evidence_provenance")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text(
                "Submarine ballast tanks fill with water to reduce buoyancy and surface by pumping air. " * 24,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "submarine ballast buoyancy",
                                    "catalog_terms": ["submarine", "ballast", "buoyancy"],
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 48,
                        "probe_tokens": 24,
                        "acquisition_tokens": 48,
                        "acquisition_slots": 1,
                    },
                )
                manager.runtime_facade.query(query_text="How do submarine ballast tanks control buoyancy?", top_k_memories=6)
                manager.runtime_facade.terminus_tick()
                before_runtime = manager.runtime_facade.status()["terminus_runtime"]
                before_provider = before_runtime["autonomy"]["provider_curriculum"]["ranked_providers"][0]

                with patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    response = manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks control buoyancy?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                selected = response["response"]["selected_evidence"]
                after_runtime = manager.runtime_facade.status()["terminus_runtime"]
                after_provider = after_runtime["autonomy"]["provider_curriculum"]["ranked_providers"][0]

                self.assertTrue(selected)
                self.assertIn("wikipedia", selected[0].get("providers") or [selected[0].get("provider")])
                self.assertEqual(selected[0]["provider"], "wikipedia")
                self.assertGreater(float(after_provider["grounded_outcome_ema"]), float(before_provider["grounded_outcome_ema"]))
            finally:
                manager.close()

    def test_follow_up_query_improvement_reinforces_provider_delayed_consequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_delayed_consequence")
            source_path = root / "terminus_source.txt"
            first_candidate_path = root / "candidate_first.txt"
            second_candidate_path = root / "candidate_second.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            first_candidate_path.write_text(
                "Submarine ballast tanks fill with water to reduce buoyancy. " * 4,
                encoding="utf-8",
            )
            second_candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory_first",
                                "source": str(first_candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "submarine ballast buoyancy",
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                query_text = "How does compressed air raise the submarine to the surface?"
                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                manager.runtime_facade.terminus_tick()
                with patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    initial = manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                before_runtime = manager.runtime_facade.status()["terminus_runtime"]
                before_provider = before_runtime["autonomy"]["provider_curriculum"]["ranked_providers"][0]

                self.assertIn("wikipedia", initial["response"]["delayed_consequence_candidate"]["providers"])
                self.assertEqual(float(before_provider["delayed_consequence_ema"]), 0.0)

                manager._brain_config["autonomy"]["candidate_bank"] = [
                    {
                        "name": "candidate_memory_second",
                        "source": str(second_candidate_path),
                        "source_type": "file",
                        "metadata": {
                            "provider": "wikipedia",
                            "query_text": "compressed air surface submarine",
                        },
                    }
                ]
                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                manager.runtime_facade.terminus_tick()
                follow_up = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                after_runtime = manager.runtime_facade.status()["terminus_runtime"]
                after_provider = after_runtime["autonomy"]["provider_curriculum"]["ranked_providers"][0]

                self.assertGreater(
                    float(follow_up["gap_plan"]["grounded_fraction"]),
                    float(initial["query_result"]["gap_plan"]["grounded_fraction"]),
                )
                self.assertGreater(int(follow_up["delayed_consequence"]["credited_records"]), 0)
                self.assertIn("wikipedia", follow_up["delayed_consequence"]["credited_providers"])
                self.assertGreater(float(after_provider["delayed_consequence_ema"]), float(before_provider["delayed_consequence_ema"]))
                self.assertGreater(
                    int(after_runtime["autonomy"]["delayed_consequence_tracking"]["credited_record_count"]),
                    0,
                )
            finally:
                manager.close()

    def test_follow_up_query_regression_penalizes_provider_long_horizon_utility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_long_horizon_penalty")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "last_query_text": "submarine ballast buoyancy",
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8, "buoyancy": 0.6},
                    }
                }
                query_text = "How do submarine ballast tanks control buoyancy?"
                focus_plan = {
                    "query_terms": ["submarine", "ballast", "buoyancy"],
                    "unsupported_terms": ["submarine", "ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 1.0}],
                    "retrieval_queries": [query_text],
                    "follow_up_questions": [],
                    "weak_concepts": [],
                }

                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Submarine ballast tanks regulate buoyancy.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Submarine ballast tanks regulate buoyancy.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                before_priority, _before_details = manager._provider_curriculum_priority_locked(
                    "wikipedia",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )
                follow_up = manager.runtime_facade.query(query_text=query_text, top_k_memories=6)
                after_priority, after_details = manager._provider_curriculum_priority_locked(
                    "wikipedia",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )

                self.assertGreater(int(follow_up["delayed_consequence"]["penalized_records"]), 0)
                self.assertIn("wikipedia", follow_up["delayed_consequence"]["penalized_providers"])
                self.assertGreater(float(after_details["contradiction_decay_ema"]), 0.0)
                self.assertLess(float(after_priority), float(before_priority))
            finally:
                manager.close()

    def test_later_grounded_improvement_forgives_provider_long_horizon_penalty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_long_horizon_forgiveness")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "compressed air submarine surface ballast",
                                    "catalog_terms": ["compressed", "air", "submarine", "surface"],
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "contradiction_decay_ema": 0.0,
                        "last_query_text": "compressed air submarine surface",
                        "topic_terms": {"compressed": 1.0, "air": 0.8, "submarine": 0.6, "surface": 0.4},
                    }
                }
                query_text = "How does compressed air raise the submarine to the surface?"
                focus_plan = {
                    "query_terms": ["compressed", "air", "submarine", "surface"],
                    "unsupported_terms": ["compressed", "air"],
                    "gap_terms": [{"term": "compressed", "weight": 1.0}],
                    "retrieval_queries": [query_text],
                    "follow_up_questions": [],
                    "weak_concepts": [],
                }
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Compressed air raises the submarine to the surface.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Compressed air raises the submarine to the surface.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                penalized = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                penalty_priority, penalty_details = manager._provider_curriculum_priority_locked(
                    "wikipedia",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )

                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)
                self.assertIn("wikipedia", penalized["delayed_consequence"]["penalized_providers"])
                self.assertGreater(float(penalty_details["contradiction_decay_ema"]), 0.0)

                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                manager.runtime_facade.terminus_tick()
                forgiven = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                forgiven_priority, forgiven_details = manager._provider_curriculum_priority_locked(
                    "wikipedia",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )
                forgiven_runtime = manager.runtime_facade.status()["terminus_runtime"]

                self.assertGreater(
                    float(forgiven["gap_plan"]["grounded_fraction"]),
                    float(penalized["gap_plan"]["grounded_fraction"]),
                )
                self.assertGreater(int(forgiven["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(forgiven["delayed_consequence"]["forgiven_records"]), 0)
                self.assertIn("wikipedia", forgiven["delayed_consequence"]["forgiven_providers"])
                self.assertLess(
                    float(forgiven_details["contradiction_decay_ema"]),
                    float(penalty_details["contradiction_decay_ema"]),
                )
                self.assertGreater(float(forgiven_priority), float(penalty_priority))
                self.assertGreater(
                    int(forgiven_runtime["autonomy"]["delayed_consequence_tracking"]["forgiven_record_count"]),
                    0,
                )
            finally:
                manager.close()

    def test_stale_provider_consequence_state_cools_and_retires(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_consequence_cooling")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "contradiction_decay_ema": 0.0,
                        "last_query_text": "submarine ballast buoyancy",
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8, "buoyancy": 0.6},
                    }
                }
                query_text = "How do submarine ballast tanks control buoyancy?"
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Submarine ballast tanks regulate buoyancy.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Submarine ballast tanks regulate buoyancy.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                manager.runtime_facade.query(query_text=query_text, top_k_memories=6)
                penalty_runtime = manager.runtime_facade.status()["terminus_runtime"]
                penalty_tracking = penalty_runtime["autonomy"]["delayed_consequence_tracking"]
                penalty_record = penalty_tracking["recent_records"][0]

                self.assertGreater(int(penalty_tracking["penalized_record_count"]), 0)
                self.assertGreater(float(penalty_record["unresolved_penalty_balance"]), 0.0)

                manager._trainer.token_count += (
                    delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS
                    + delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS
                )
                cooled_runtime = manager.runtime_facade.status()["terminus_runtime"]
                cooled_tracking = cooled_runtime["autonomy"]["delayed_consequence_tracking"]
                cooled_record = cooled_tracking["recent_records"][0]

                self.assertLess(
                    float(cooled_record["unresolved_penalty_balance"]),
                    float(penalty_record["unresolved_penalty_balance"]),
                )
                self.assertGreater(int(cooled_record["cooling_events"]), 0)
                self.assertGreater(int(cooled_tracking["cooled_record_count_total"]), 0)

                manager._trainer.token_count += delayed_consequence_module.DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS * 2
                retired_runtime = manager.runtime_facade.status()["terminus_runtime"]
                retired_tracking = retired_runtime["autonomy"]["delayed_consequence_tracking"]

                self.assertEqual(int(retired_tracking["record_count"]), 0)
                self.assertGreater(int(retired_tracking["retired_record_count_total"]), 0)
            finally:
                manager.close()

    def test_repeated_provider_consequence_records_compact_query_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_consequence_compaction")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Submarine ballast tanks control buoyancy.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Submarine ballast tanks control buoyancy.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    first = manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks control buoyancy?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    second = manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks change buoyancy?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                tracking = manager.runtime_facade.status()["terminus_runtime"]["autonomy"]["delayed_consequence_tracking"]
                record = tracking["recent_records"][0]

                self.assertEqual(int(second["response"]["delayed_consequence_candidate"]["aggregate_count"]), 2)
                self.assertEqual(int(tracking["record_count"]), 1)
                self.assertGreater(int(tracking["aggregated_record_count"]), 0)
                self.assertGreaterEqual(int(tracking["aggregate_occurrence_count"]), 2)
                self.assertGreater(int(tracking["compacted_record_count_total"]), 0)
                self.assertEqual(int(record["aggregate_count"]), 2)
                self.assertGreater(float(record["aggregate_support_multiplier"]), 1.0)
                self.assertIn("How do submarine ballast tanks control buoyancy?", record["query_examples"])
                self.assertIn("How do submarine ballast tanks change buoyancy?", record["query_examples"])
            finally:
                manager.close()

    def test_provider_consequence_family_trajectory_summary_tracks_penalty_then_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_consequence_trajectory")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "compressed air submarine surface ballast",
                                    "catalog_terms": ["compressed", "air", "submarine", "surface"],
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Compressed air raises the submarine to the surface.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Compressed air raises the submarine to the surface.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How does compressed air raise the submarine to the surface?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    manager.runtime_facade.respond(
                        query_text="How does compressed air lift the submarine to the surface?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "contradiction_decay_ema": 0.0,
                        "last_query_text": "compressed air submarine surface",
                        "topic_terms": {"compressed": 1.0, "air": 0.8, "submarine": 0.6, "surface": 0.4},
                    }
                }
                query_text = "How does compressed air raise the submarine to the surface?"
                penalized = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                penalty_runtime = manager.runtime_facade.status()["terminus_runtime"]
                penalty_record = penalty_runtime["autonomy"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)
                self.assertEqual(int(penalty_record["aggregate_count"]), 2)
                self.assertEqual(str(penalty_record["trajectory_state"]), "negative")
                self.assertGreater(float(penalty_record["trajectory_penalty_total"]), 0.0)
                self.assertLess(float(penalty_record["trajectory_support_multiplier"]), 1.0)
                self.assertGreater(float(penalty_record["trajectory_penalty_multiplier"]), 1.0)

                manager.runtime_facade.terminus_tick()
                recovered = manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                recovered_runtime = manager.runtime_facade.status()["terminus_runtime"]
                recovered_record = recovered_runtime["autonomy"]["delayed_consequence_tracking"]["recent_records"][0]

                self.assertGreater(int(recovered["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(recovered["delayed_consequence"]["forgiven_records"]), 0)
                self.assertGreater(float(recovered_record["trajectory_credit_total"]), 0.0)
                self.assertGreater(float(recovered_record["trajectory_forgiveness_total"]), 0.0)
                self.assertGreater(
                    float(recovered_record["trajectory_net_score"]),
                    float(penalty_record["trajectory_net_score"]),
                )
                self.assertGreater(
                    float(recovered_record["trajectory_recent_delta_ema"]),
                    float(penalty_record["trajectory_recent_delta_ema"]),
                )
                self.assertGreater(
                    float(recovered_record["trajectory_support_multiplier"]),
                    float(penalty_record["trajectory_support_multiplier"]),
                )
                self.assertLess(
                    float(recovered_record["trajectory_penalty_multiplier"]),
                    float(penalty_record["trajectory_penalty_multiplier"]),
                )
                self.assertEqual(str(recovered_record["trajectory_state"]), "recovering")
            finally:
                manager.close()

    def test_provider_consequence_family_divergence_split_separates_mixed_query_branches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_consequence_split")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "compressed air submarine surface ballast",
                                    "catalog_terms": ["compressed", "air", "submarine", "surface"],
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Submarine buoyancy depends on ballast tanks and compressed air.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Submarine buoyancy depends on ballast tanks and compressed air.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks control buoyancy underwater?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    second = manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks use compressed air to rise?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                self.assertEqual(int(second["response"]["delayed_consequence_candidate"]["aggregate_count"]), 2)
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "contradiction_decay_ema": 0.0,
                        "last_query_text": "compressed air submarine surface",
                        "topic_terms": {"compressed": 1.0, "air": 0.8, "submarine": 0.6, "surface": 0.4},
                    }
                }

                penalized = manager.runtime_facade.query(
                    query_text="How do submarine ballast tanks control buoyancy underwater?",
                    top_k_memories=8,
                )
                self.assertGreater(int(penalized["delayed_consequence"]["penalized_records"]), 0)

                manager.runtime_facade.terminus_tick()
                manager.runtime_facade.terminus_tick()
                manager.runtime_facade.terminus_tick()
                recovered = manager.runtime_facade.query(
                    query_text="How do submarine ballast tanks use compressed air to rise?",
                    top_k_memories=8,
                )
                tracking = manager.runtime_facade.status()["terminus_runtime"]["autonomy"]["delayed_consequence_tracking"]
                records = {
                    str(record.get("split_branch", "")): record
                    for record in tracking["recent_records"]
                    if str(record.get("split_branch", ""))
                }

                self.assertGreater(int(recovered["delayed_consequence"]["credited_records"]), 0)
                self.assertGreater(int(recovered["delayed_consequence"]["forgiven_records"]), 0)
                self.assertEqual(int(tracking["record_count"]), 2)
                self.assertGreater(int(tracking["split_record_count_total"]), 0)
                self.assertEqual(int(tracking["aggregate_occurrence_count"]), 2)
                self.assertIn("supportive", records)
                self.assertIn("adverse", records)
                supportive = records["supportive"]
                adverse = records["adverse"]
                self.assertIn("How do submarine ballast tanks use compressed air to rise?", supportive["query_examples"])
                self.assertIn("How do submarine ballast tanks control buoyancy underwater?", adverse["query_examples"])
                self.assertEqual(str(adverse["trajectory_state"]), "negative")
                self.assertGreater(
                    float(supportive["trajectory_net_score"]),
                    float(adverse["trajectory_net_score"]),
                )
                self.assertEqual(str(supportive["split_group_id"]), str(adverse["split_group_id"]))
            finally:
                manager.close()

    def test_provider_split_lineage_remerges_after_aligned_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_consequence_remerge")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "compressed air submarine surface ballast",
                                    "catalog_terms": ["compressed", "air", "submarine", "surface"],
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                with patch.object(
                    manager._responder,
                    "build_response",
                    return_value={
                        "response_text": "Submarine buoyancy depends on ballast tanks and compressed air.",
                        "response_mode": "grounded_synthesis",
                        "selected_evidence": [
                            {
                                "text": "Submarine buoyancy depends on ballast tanks and compressed air.",
                                "provider": "wikipedia",
                                "providers": ["wikipedia"],
                                "term_coverage": 1.0,
                                "score": 0.9,
                            }
                        ],
                        "evidence_coverage": 1.0,
                        "unsupported_terms": [],
                    },
                ), patch.object(
                    manager._interaction_pipeline,
                    "_plan_gaps_fn",
                    return_value={
                        "grounded_fraction": 1.0,
                        "unsupported_terms": [],
                        "gap_terms": [],
                        "retrieval_queries": [],
                        "follow_up_questions": [],
                        "weak_concepts": [],
                    },
                ), patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks control buoyancy underwater?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                    manager.runtime_facade.respond(
                        query_text="How do submarine ballast tanks use compressed air to rise?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.20,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.46,
                        "focus_alignment_ema": 0.72,
                        "grounded_outcome_ema": 0.0,
                        "delayed_consequence_ema": 0.0,
                        "contradiction_decay_ema": 0.0,
                        "last_query_text": "compressed air submarine surface",
                        "topic_terms": {"compressed": 1.0, "air": 0.8, "submarine": 0.6, "surface": 0.4},
                    }
                }

                manager.runtime_facade.query(
                    query_text="How do submarine ballast tanks control buoyancy underwater?",
                    top_k_memories=8,
                )
                manager.runtime_facade.terminus_tick()
                manager.runtime_facade.terminus_tick()
                split_result = manager.runtime_facade.query(
                    query_text="How do submarine ballast tanks use compressed air to rise?",
                    top_k_memories=8,
                )
                split_tracking = manager.runtime_facade.status()["terminus_runtime"]["autonomy"]["delayed_consequence_tracking"]

                self.assertGreater(int(split_result["delayed_consequence"]["split_records"]), 0)
                self.assertEqual(int(split_tracking["record_count"]), 2)

                remerged = manager.runtime_facade.query(
                    query_text="How do submarine ballast tanks use compressed air to rise?",
                    top_k_memories=8,
                )
                tracking = manager.runtime_facade.status()["terminus_runtime"]["autonomy"]["delayed_consequence_tracking"]
                record = tracking["recent_records"][0]

                self.assertGreater(int(remerged["delayed_consequence"]["remerged_records"]), 0)
                self.assertEqual(int(tracking["record_count"]), 1)
                self.assertGreater(int(tracking["remerged_record_count_total"]), 0)
                self.assertEqual(str(record["split_branch"]), "")
                self.assertGreater(int(record["remerge_events"]), 0)
                self.assertTrue(str(record["last_remerged_at"]))
                self.assertIn("How do submarine ballast tanks use compressed air to rise?", record["query_examples"])
                self.assertIn("How do submarine ballast tanks control buoyancy underwater?", record["query_examples"])
            finally:
                manager.close()

    def test_grounded_family_summary_biases_equal_focus_provider_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_family_summary")
            source_path = root / "terminus_source.txt"
            first_candidate_path = root / "candidate_first.txt"
            second_candidate_path = root / "candidate_second.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            first_candidate_path.write_text(
                "Submarine ballast tanks fill with water to reduce buoyancy. " * 4,
                encoding="utf-8",
            )
            second_candidate_path.write_text(
                "Compressed air expels water from ballast tanks so the submarine rises to the surface. " * 5,
                encoding="utf-8",
            )
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory_first",
                                "source": str(first_candidate_path),
                                "source_type": "file",
                                "metadata": {
                                    "provider": "wikipedia",
                                    "query_text": "submarine ballast buoyancy",
                                },
                            }
                        ],
                        "trigger_interval_tokens": 1,
                        "candidate_train_tokens": 120,
                        "probe_tokens": 48,
                        "acquisition_tokens": 120,
                        "acquisition_slots": 1,
                    },
                )
                query_text = "How does compressed air raise the submarine to the surface?"
                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                manager.runtime_facade.terminus_tick()
                with patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text=query_text,
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                manager._brain_config["autonomy"]["candidate_bank"] = [
                    {
                        "name": "candidate_memory_second",
                        "source": str(second_candidate_path),
                        "source_type": "file",
                        "metadata": {
                            "provider": "wikipedia",
                            "query_text": "compressed air surface submarine",
                        },
                    }
                ]
                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                manager.runtime_facade.terminus_tick()
                manager.runtime_facade.query(query_text=query_text, top_k_memories=8)
                runtime = manager.runtime_facade.status()["terminus_runtime"]
                wikipedia_details = next(
                    item
                    for item in runtime["autonomy"]["provider_curriculum"]["ranked_providers"]
                    if item["provider"] == "wikipedia"
                )

                self.assertGreater(float(wikipedia_details["grounded_family_summary_ema"]), 0.0)

                focus_plan = {
                    "query_terms": ["submarine", "ballast", "compressed", "air", "surface"],
                    "unsupported_terms": ["compressed", "air"],
                    "gap_terms": [{"term": "compressed", "weight": 1.0}],
                    "retrieval_queries": [query_text],
                    "follow_up_questions": [],
                    "weak_concepts": [],
                }
                shared_provider_entry = {
                    "attempts": 3,
                    "commits": 3,
                    "successes": 2,
                    "diagnostic_gain_ema": 0.22,
                    "semantic_relevance_ema": 0.72,
                    "answerability_gain_ema": 0.18,
                    "uncertainty_reduction_ema": 0.16,
                    "weak_concept_stabilization_ema": 0.10,
                    "utility_ema": 0.18,
                    "focus_alignment_ema": 0.65,
                    "grounded_outcome_ema": 0.10,
                    "delayed_consequence_ema": 0.10,
                    "contradiction_decay_ema": 0.0,
                    "topic_terms": {"submarine": 1.0, "ballast": 0.8, "compressed": 0.6},
                }
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "openalex": {**shared_provider_entry, "grounded_family_summary_ema": 0.0},
                    "wikipedia": {
                        **shared_provider_entry,
                        "grounded_family_summary_ema": float(wikipedia_details["grounded_family_summary_ema"]),
                    },
                }
                wikipedia_priority, wikipedia_details = manager._provider_curriculum_priority_locked(
                    "wikipedia",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )
                openalex_priority, openalex_details = manager._provider_curriculum_priority_locked(
                    "openalex",
                    focus_plan,
                    autonomy=manager._brain_config["autonomy"],
                )

                self.assertGreater(float(wikipedia_details["grounded_family_summary_ema"]), float(openalex_details["grounded_family_summary_ema"]))
                self.assertGreater(float(wikipedia_priority), float(openalex_priority))
            finally:
                manager.close()

    def test_provider_utility_ema_biases_equal_focus_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_utility_ema")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["openalex", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "openalex": {
                        "attempts": 3,
                        "commits": 3,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.22,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.18,
                        "focus_alignment_ema": 0.65,
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8},
                        "topic_families": {
                            "submarine": {
                                "commits": 2,
                                "successes": 1,
                                "semantic_relevance_ema": 0.72,
                                "answerability_gain_ema": 0.18,
                                "uncertainty_reduction_ema": 0.16,
                                "weak_concept_stabilization_ema": 0.10,
                            }
                        },
                    },
                    "wikipedia": {
                        "attempts": 3,
                        "commits": 3,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.22,
                        "semantic_relevance_ema": 0.72,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.16,
                        "weak_concept_stabilization_ema": 0.10,
                        "utility_ema": 0.74,
                        "focus_alignment_ema": 0.65,
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8},
                        "topic_families": {
                            "submarine": {
                                "commits": 2,
                                "successes": 1,
                                "semantic_relevance_ema": 0.72,
                                "answerability_gain_ema": 0.18,
                                "uncertainty_reduction_ema": 0.16,
                                "weak_concept_stabilization_ema": 0.10,
                            }
                        },
                    },
                }
                manager.runtime_facade.query(query_text="How do submarine ballast tanks control buoyancy?", top_k_memories=6)
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {"unsupported_terms": ["submarine", "ballast", "buoyancy"]},
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                ranked = runtime["autonomy"]["provider_curriculum"]["ranked_providers"]

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertEqual(ranked[0]["provider"], "wikipedia")
                self.assertGreater(float(ranked[0]["utility_ema"]), float(ranked[1]["utility_ema"]))
            finally:
                manager.close()

    def test_terminus_live_remote_search_prefers_matched_topic_family_on_revisit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_revisit_alignment")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 1,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.18,
                        "semantic_relevance_ema": 0.70,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.12,
                        "weak_concept_stabilization_ema": 0.08,
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8, "buoyancy": 0.6},
                        "topic_families": {
                            "submarine": {
                                "commits": 1,
                                "successes": 1,
                                "semantic_relevance_ema": 0.70,
                                "answerability_gain_ema": 0.18,
                                "uncertainty_reduction_ema": 0.12,
                                "weak_concept_stabilization_ema": 0.08,
                            }
                        },
                    },
                    "openalex": {
                        "attempts": 2,
                        "commits": 1,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.28,
                        "semantic_relevance_ema": 0.92,
                        "answerability_gain_ema": 0.42,
                        "uncertainty_reduction_ema": 0.30,
                        "weak_concept_stabilization_ema": 0.18,
                        "topic_terms": {"octopus": 1.0, "jars": 0.7},
                        "topic_families": {
                            "octopus": {
                                "commits": 1,
                                "successes": 1,
                                "semantic_relevance_ema": 0.92,
                                "answerability_gain_ema": 0.42,
                                "uncertainty_reduction_ema": 0.30,
                                "weak_concept_stabilization_ema": 0.18,
                            }
                        },
                    },
                }
                manager._brain_recent_query_gaps.appendleft(
                    manager._normalize_recent_query_gap(
                        {
                            "source": "query",
                            "query_text": "What corrects submarine trim?",
                            "unsupported_terms": ["corrects", "trim"],
                            "gap_terms": [
                                {"term": "corrects", "weight": 2.0},
                                {"term": "trim", "weight": 2.0},
                            ],
                            "retrieval_queries": ["corrects trim"],
                            "follow_up_questions": [
                                "What grounded evidence is still missing for corrects?",
                                "What grounded evidence is still missing for trim?",
                            ],
                            "weak_concepts": [
                                {
                                    "label": "terms / octopuses",
                                    "weakness": 0.55,
                                    "uncertainty": 0.54,
                                    "drift": 0.0,
                                    "top_terms": ["terms", "octopuses", "open"],
                                    "match_count": 1,
                                }
                            ],
                            "grounded_fraction": 0.3333333333333333,
                        }
                    )
                )
                with patch(
                    "hecsn.service.brain_runtime.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["corrects", "submarine", "trim"],
                        },
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.runtime_facade.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                runtime = manager.runtime_facade.terminus_status()["terminus_runtime"]
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertGreater(
                    float(spec["catalog_provider_priority_map"]["wikipedia"]),
                    float(spec["catalog_provider_priority_map"]["openalex"]),
                )
                self.assertIn("submarine", str(spec["catalog_focus_text"]))
                self.assertIn("catalog_provider_topic_terms", spec)
                self.assertEqual(spec["catalog_provider_topic_terms"]["wikipedia"][0], "submarine")
                self.assertNotIn("openalex", spec["catalog_provider_topic_terms"])
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["topic_family_focus_score"]),
                    0.0,
                )
            finally:
                manager.close()

    def test_save_restore_round_trips_terminus_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_checkpoint")
            source_path = root / "terminus_source.txt"
            source_path.write_text("hebbian memory consolidation " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "checkpoint_source",
                            "source": str(source_path),
                            "source_type": "file",
                            "metadata": {"label": "memory consolidation hebbian signal"},
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                    ingestion={"queue_target_tokens": 24, "prewarm_on_startup": False},
                )
                manager.runtime_facade.query(query_text="How does hebbian memory consolidation work?", top_k_memories=6)
                before_runtime = manager.runtime_facade.terminus_tick()["terminus_runtime"]
                with patch.object(manager._interaction_pipeline, "_maybe_auto_action_assist_fn", return_value=None):
                    manager.runtime_facade.respond(
                        query_text="How does hebbian memory consolidation work?",
                        max_evidence_items=3,
                        learn_mode="none",
                    )
                saved = manager.runtime_facade.save_checkpoint(str(root / "terminus_service.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    terminus_runtime = restored.runtime_facade.status()["terminus_runtime"]

                    self.assertTrue(terminus_runtime["configured"])
                    self.assertEqual(terminus_runtime["source_bank"][0]["name"], "checkpoint_source")
                    self.assertEqual(terminus_runtime["autonomy"]["candidate_count"], 1)
                    self.assertEqual(terminus_runtime["ingestion"]["queue_target_tokens"], 24)
                    self.assertFalse(terminus_runtime["ingestion"]["prewarm_on_startup"])
                    self.assertGreater(float(before_runtime["source_progress"][0]["utility_ema"]), 0.0)
                    self.assertGreater(
                        int(terminus_runtime["background_source_routing"]["delayed_consequence_tracking"]["record_count"]),
                        0,
                    )
                    self.assertGreater(
                        int(terminus_runtime["autonomy"]["delayed_consequence_tracking"]["record_count"]),
                        0,
                    )
                    self.assertGreaterEqual(
                        float(terminus_runtime["source_progress"][0]["utility_ema"]),
                        float(before_runtime["source_progress"][0]["utility_ema"]),
                    )
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_save_restore_round_trips_recent_brain_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_brain_event_checkpoint")
            source_path = root / "brain_source.txt"
            source_path.write_text("brain event persistence signal " * 32, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "brain_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                    ingestion={"queue_target_tokens": 16, "prewarm_on_startup": False},
                )
                manager.runtime_facade.terminus_tick(steps=2)
                before_runtime = manager.runtime_facade.status()["terminus_runtime"]
                self.assertGreater(len(before_runtime["recent_events"]), 0)
                self.assertEqual(before_runtime["recent_events"][0], before_runtime["last_event"])

                saved = manager.runtime_facade.save_checkpoint(str(root / "brain_events.pt"))
            finally:
                manager.close()

            restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
            try:
                after_runtime = restored.runtime_facade.status()["terminus_runtime"]

                self.assertEqual(after_runtime["last_event"], before_runtime["last_event"])
                self.assertEqual(after_runtime["recent_events"], before_runtime["recent_events"])
            finally:
                restored.close()

    def test_save_restore_round_trips_catalog_candidate_bank_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_catalog_checkpoint")
            source_path = root / "terminus_source.txt"
            source_path.write_text("hebbian memory consolidation " * 24, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "checkpoint_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "registry_pool",
                                "catalog_mode": "semantic_registry",
                                "catalog_limit": 2,
                                "catalog_entries": [
                                    {
                                        "name": "submarine_source",
                                        "source": "https://example.test/submarine",
                                        "source_type": "web",
                                        "summary": "submarine buoyancy ballast pressure",
                                    },
                                    {
                                        "name": "garden_source",
                                        "source": "https://example.test/garden",
                                        "source_type": "web",
                                        "summary": "garden tomato soil sunlight",
                                    },
                                ],
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                saved = manager.runtime_facade.save_checkpoint(str(root / "terminus_catalog_service.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    terminus_runtime = restored.runtime_facade.status()["terminus_runtime"]

                    self.assertTrue(terminus_runtime["configured"])
                    self.assertEqual(terminus_runtime["source_bank"][0]["name"], "checkpoint_source")
                    self.assertEqual(terminus_runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                    self.assertEqual(len(terminus_runtime["autonomy"]["candidate_bank"][0]["catalog_entries"]), 2)
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_start_and_stop_terminus_loop_updates_running_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_loop")
            source_path = root / "terminus_source.txt"
            source_path.write_text("unsupervised knowledge accumulation " * 64, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "loop_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=16,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )

                started = manager.runtime_facade.start_terminus()
                self.assertTrue(started["terminus_runtime"]["running"])
                self.assertIsNotNone(started["terminus_runtime"]["running_since"])
                time.sleep(0.5)
                stopped = manager.runtime_facade.stop_terminus()

                self.assertFalse(stopped["terminus_runtime"]["running"])
                self.assertIsNone(stopped["terminus_runtime"]["running_since"])
                self.assertGreaterEqual(stopped["terminus_runtime"]["background_tokens_processed"], 8)
                self.assertEqual(stopped["terminus_runtime"]["recent_events"][0]["type"], "stopped")
            finally:
                manager.close()

    def test_terminus_tick_rejected_while_runtime_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_tick_running")
            source_path = root / "terminus_source.txt"
            source_path.write_text("runtime ownership safety signal " * 64, encoding="utf-8")
            try:
                manager.runtime_facade.configure_terminus(
                    source_bank=[
                        {
                            "name": "loop_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=16,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )

                started = manager.runtime_facade.start_terminus()
                self.assertTrue(started["terminus_runtime"]["running"])
                with self.assertRaisesRegex(ValueError, "background runtime is active"):
                    manager.runtime_facade.terminus_tick()
            finally:
                try:
                    manager.runtime_facade.stop_terminus()
                except Exception:
                    pass
                manager.close()

    def test_stop_timeout_records_shutdown_state_and_interrupts_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_stop_timeout")

            class _ClosableIterator:
                def __init__(self) -> None:
                    self.closed = False

                def __iter__(self):
                    return self

                def __next__(self):
                    raise StopIteration

                def close(self) -> None:
                    self.closed = True

            class _StuckThread:
                def __init__(self) -> None:
                    self._alive = True
                    self.join_calls: list[float | None] = []

                def is_alive(self) -> bool:
                    return self._alive

                def join(self, timeout: float | None = None) -> None:
                    self.join_calls.append(timeout)

            brain_stream = _ClosableIterator()
            sensory_stream = _ClosableIterator()
            stuck_thread = _StuckThread()

            manager._brain_source_runtimes = [
                _BrainSourceRuntime(
                    spec={"name": "stuck_source", "source_type": "file"},
                    stream=brain_stream,
                )
            ]
            manager._sensory_source_runtimes = [
                _SensorySourceRuntime(
                    spec={"name": "stuck_sensory", "adapter": "audiocaps"},
                    stream=sensory_stream,
                )
            ]
            manager._brain_thread = stuck_thread  # type: ignore[assignment]
            manager._brain_stop_event = Event()
            manager._brain_running = True
            manager._brain_running_since = "2026-04-22T00:00:00+00:00"

            try:
                with self.assertRaisesRegex(RuntimeError, "did not stop within"):
                    manager.runtime_facade.stop_terminus()

                runtime = manager.runtime_facade.status()["terminus_runtime"]
                self.assertTrue(runtime["running"])
                self.assertTrue(runtime["shutdown"]["stop_requested"])
                self.assertTrue(runtime["shutdown"]["stop_timed_out"])
                self.assertTrue(runtime["shutdown"]["thread_alive"])
                self.assertIn("did not stop within", runtime["last_error"])
                self.assertTrue(brain_stream.closed)
                self.assertTrue(sensory_stream.closed)
                self.assertEqual(runtime["recent_events"][0]["type"], "stop_timeout")
            finally:
                stuck_thread._alive = False
                with manager._lock:
                    manager._finalize_brain_stop_locked(stuck_thread)  # type: ignore[arg-type]
                manager.close()

    def test_close_suppresses_stop_timeout_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_close_timeout")

            class _StuckThread:
                def __init__(self) -> None:
                    self._alive = True

                def is_alive(self) -> bool:
                    return self._alive

                def join(self, timeout: float | None = None) -> None:
                    return None

            manager._brain_thread = _StuckThread()  # type: ignore[assignment]
            manager._brain_stop_event = Event()
            manager._brain_running = True
            manager._brain_running_since = "2026-04-22T00:00:00+00:00"

            manager.close()
            self.assertIn("did not stop within", manager._brain_last_error or "")


class CortexIntegrationTests(unittest.TestCase):
    """Test deleted Cortex boundaries with service manager."""

    def test_manager_creation_status_do_not_eagerly_initialize_cortex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_lazy_manager_startup", env_root=root)
            try:
                status = manager.runtime_facade.status()
            finally:
                manager.close()

            self.assertIn("terminus_runtime", status)
            self.assertNotIn("cortex", status["terminus_runtime"])
            self.assertNotIn("retired_runtime_path", status["terminus_runtime"])

    def test_cortex_methods_are_removed_from_operator_facade(self) -> None:
        """The LLM Cortex API is retired and no longer part of the operator facade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_no_nim_key")
            try:
                self.assertFalse(hasattr(manager, "cortex_ask"))
                self.assertFalse(hasattr(manager, "cortex_sleep"))
                self.assertFalse(hasattr(manager, "cortex_thoughts"))
                self.assertFalse(hasattr(manager, "cortex_snapshot"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_ask"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_sleep"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_thoughts"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_snapshot"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_signal_state"))
            finally:
                manager.close()

    def test_cognitive_signal_is_canonical_operator_signal_surface(self) -> None:
        """RuntimeFacade exposes Cognitive Signal while retired names stay absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cognitive_signal_canonical_surface")
            try:
                facade_signal = manager.runtime_facade.cognitive_signal_state()
                self.assertEqual(facade_signal["schema_version"], "cognitive_signal.v1")
                self.assertIn("subcortical_language", facade_signal)
                self.assertIn("subcortical_deliberation", facade_signal)
                self.assertFalse(hasattr(manager, "cognitive_signal_state"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_signal_state"))
                self.assertFalse(hasattr(manager.runtime_facade, "cortex_thoughts"))
            finally:
                manager.close()

    def test_runtime_snapshot_excludes_retired_runtime_path(self) -> None:
        """Regression: terminus runtime snapshot omits the former Cortex path.

        Verdict and payload key coverage is in test_status_read_model.py::StatusReadModelStatusTests
        and StatusReadModelPayloadCompatibilityTests. This stub confirms the manager wiring survives.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_snapshot_key")
            try:
                status = manager.runtime_facade.status()
                self.assertNotIn("cortex", status["terminus_runtime"])
                self.assertNotIn("retired_runtime_path", status["terminus_runtime"])
            finally:
                manager.close()

    def test_status_exposes_runtime_truth_contract(self) -> None:
        """Regression: manager delegates runtime truth to the Status Read Model.

        Detailed verdict progression, payload keys, and safety flag coverage lives in
        test_status_read_model.py::StatusReadModelRuntimeTruthVerdictTests and
        StatusReadModelPayloadCompatibilityTests. This stub confirms manager-level delegation.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="status_runtime_truth_contract")
            try:
                status = manager.runtime_facade.status()
                truth = status["runtime_truth"]
                self.assertEqual(truth["schema_version"], 1)
                self.assertIn("verdict", truth)
            finally:
                manager.close()

    def test_status_and_terminus_status_read_runtime_state_directly(self) -> None:
        """Regression: manager surfaces runtime state through the read model delegation seam.

        Per-surface dirty_state and state_revision propagation is covered by
        test_status_read_model.py::StatusReadModelRuntimeStatePropagationTests.
        Event path normalization is a manager-level integration concern.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="status_runtime_state_seam")
            try:
                manager._runtime_state.dirty_state = True
                manager._runtime_state.state_revision = 7
                manager._runtime_state.record_event(
                    {
                        "type": "runtime-state-seam",
                        "path": Path("reports/runtime/event.json"),
                        "items": ["alpha", Path("nested/item.txt")],
                    }
                )
                manager.__dict__.pop("_cached_status", None)
                manager.__dict__.pop("_cached_terminus_status", None)

                status = manager.runtime_facade.status()
                self.assertTrue(status["dirty_state"])
                self.assertEqual(status["state_revision"], 7)
                last_event = status["terminus_runtime"]["last_event"]
                self.assertEqual(last_event["type"], "runtime-state-seam")
                self.assertEqual(
                    Path(last_event["path"]).as_posix(),
                    "reports/runtime/event.json",
                )
                self.assertEqual(
                    Path(last_event["items"][1]).as_posix(),
                    "nested/item.txt",
                )
            finally:
                manager.close()

    def test_manager_does_not_expose_runtime_state_compatibility_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="status_runtime_state_no_compat_aliases")
            try:
                aliases = ("dirty_state", "state_revision", "last_event", "recent_events")
                for alias in aliases:
                    with self.subTest(alias=alias):
                        self.assertFalse(hasattr(manager, alias), f"manager unexpectedly exposes {alias}")
                        self.assertNotIn(alias, manager.__dict__, f"manager unexpectedly stores {alias}")
            finally:
                manager.close()

    def test_runtime_truth_reports_alive_after_subcortex_progress(self) -> None:
        """Regression: alive verdict flows through the manager after a real tick.

        Verdict logic coverage is in test_status_read_model.py::StatusReadModelRuntimeTruthVerdictTests.
        This confirms the integration path from configure_terminus -> terminus_tick -> status().
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "truth_source.txt"
            source_path.write_text("submarine ballast pressure control " * 8, encoding="utf-8")
            manager = _build_manager(root, test_case="status_runtime_truth_alive")
            try:
                runtime = manager.runtime_facade
                runtime.configure_terminus(
                    source_bank=[
                        {
                            "name": "truth_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=8,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                runtime.terminus_tick()

                truth = runtime.status()["runtime_truth"]
                self.assertEqual(truth["verdict"], "alive")
            finally:
                manager.close()

    def test_status_fresh_wait_ignores_stale_cached_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "stream.txt"
            source_path.write_text("character stream learning " * 24, encoding="utf-8")
            manager = _build_manager(root, test_case="status_fresh_wait")
            try:
                runtime = manager.runtime_facade
                stale = runtime.status()
                self.assertFalse(stale["terminus_runtime"]["configured"])
                runtime.configure_terminus(
                    source_bank=[
                        {
                            "name": "local_stream",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=20,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                    ingestion={
                        "enabled": True,
                        "queue_target_tokens": 40,
                        "prewarm_on_startup": False,
                        "prewarm_max_seconds": 0.2,
                    },
                )

                ready = Event()
                release = Event()

                def _hold_lock() -> None:
                    with manager._lock:
                        ready.set()
                        release.wait(timeout=1.0)

                thread = threading.Thread(target=_hold_lock, daemon=True)
                thread.start()
                self.assertTrue(ready.wait(timeout=1.0))
                try:
                    fresh = runtime.status(fresh_wait_seconds=0.05)
                finally:
                    release.set()
                    thread.join(timeout=1.0)

                self.assertTrue(fresh["terminus_runtime"]["configured"])
                self.assertEqual(fresh["terminus_runtime"]["source_bank"][0]["name"], "local_stream")
            finally:
                manager.close()

    def test_telemetry_excludes_retired_runtime_path(self) -> None:
        """Regression: telemetry snapshot omits the former Cortex path.

        Detailed telemetry payload coverage is in
        test_status_read_model.py::StatusReadModelTelemetryTests. This stub confirms
        manager-level facade wiring.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_telemetry")
            try:
                telemetry = manager.runtime_facade.telemetry_snapshot()
                self.assertNotIn("cortex", telemetry["terminus_runtime"])
                self.assertNotIn("retired_runtime_path", telemetry["terminus_runtime"])
            finally:
                manager.close()

class ServiceManagerActionLoopTests(unittest.TestCase):
    def test_execute_digital_action_persists_verified_workspace_search_and_persists_action_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_workspace_search",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    }
                )
                self.assertTrue(result["accepted"])
                runtime = result["terminus_runtime"]
                self.assertEqual(runtime["action_loop"]["verified_actions"], 1)
                self.assertEqual(runtime["action_loop"]["contradicted_actions"], 0)
                self.assertEqual(runtime["action_loop"]["ledger_scope"], "subcortex_action_ledger")
                self.assertNotIn("retired_loop_sync", runtime["action_loop"])
                self.assertFalse(hasattr(manager, "_thought_loop_actual"))
                self.assertEqual(runtime["recent_events"][0]["type"], "digital_action_executed")
                self.assertEqual(result["result"]["verification"]["status"], "verified")

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["verification"]["status"], "verified")
            finally:
                restored.close()

    def test_record_runtime_feedback_updates_action_history_and_missing_target_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            manager = _build_manager(
                root,
                test_case="service_manager_runtime_feedback_action",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    }
                )
                action_id = result["result"]["action_id"]
                feedback = manager.runtime_facade.record_runtime_feedback(
                    {
                        "target_type": "action",
                        "target_id": action_id,
                        "verdict": "verified",
                        "confidence": 0.77,
                        "summary": "Manual evaluator verified the action result.",
                        "evidence": [{"source": "review"}],
                        "tags": ["reviewed"],
                        "evaluator_id": "qa-bot",
                    }
                )

                self.assertTrue(feedback["accepted"])
                self.assertEqual(feedback["target"]["verification"]["status"], "verified")
                self.assertEqual(feedback["target"]["provenance"], "verified")
                self.assertEqual(feedback["target"]["feedback"][0]["evidence"][0]["source"], "review")
                with self.assertRaisesRegex(ValueError, "Runtime feedback target not found"):
                    manager.runtime_facade.record_runtime_feedback(
                        {
                            "target_type": "action",
                            "target_id": "missing-action",
                            "verdict": "unverified",
                            "confidence": 0.1,
                        }
                    )
                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()

            restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces", env_root=root)
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["actions"][0]["action_id"], action_id)
                self.assertEqual(history["actions"][0]["verification"]["status"], "verified")
                self.assertEqual(history["actions"][0]["feedback"][0]["evaluator_id"], "qa-bot")
            finally:
                restored.close()

    def test_execute_digital_action_persists_parameterized_api_request_and_persists_action_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            port = _free_port()
            server = ThreadingHTTPServer(("127.0.0.1", port), _EchoJsonApiHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_parameterized_api_request",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/api/echo",
                        "method": "POST",
                        "params": {"kind": "feline"},
                        "json_body": {"topic": "cats", "fact": "mice at night"},
                        "query_text": "cats mice night feline",
                        "predicted_outcome": "I expect the API request to return structured JSON about cats and mice at night.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["inputs"]["method"], "POST")
                self.assertEqual(result["result"]["inputs"]["params"]["kind"], "feline")
                self.assertEqual(result["result"]["inputs"]["json_body"]["topic"], "cats")
                self.assertEqual(result["result"]["verification"]["status"], "verified")

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["action_type"], "api_request")
                self.assertEqual(history["actions"][0]["inputs"]["method"], "POST")
                self.assertEqual(history["actions"][0]["inputs"]["params"]["kind"], "feline")
            finally:
                restored.close()

    def test_execute_digital_action_persists_structured_api_verification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_structured_api_verification",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "query_text": "cat mice night",
                        "predicted_outcome": "I expect the JSON endpoint to identify the animal that hunts mice at night.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["verification"]["evidence"][0]["json_path"], "$.animals[0]")
                self.assertEqual(result["result"]["verification"]["evidence"][0]["structure_kind"], "object")
                self.assertGreaterEqual(result["result"]["verification"]["evidence"][0]["field_count"], 3)

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["action_type"], "api_request")
                self.assertEqual(history["actions"][0]["verification"]["evidence"][0]["json_path"], "$.animals[0]")
                self.assertEqual(history["actions"][0]["verification"]["evidence"][0]["structure_kind"], "object")
            finally:
                restored.close()

    def test_execute_digital_action_persists_explicit_api_assertion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_assertions",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "expected_json_paths": ["$.animals[0]", "$.animals[0].diet"],
                        "expected_response_shape": "object",
                        "query_text": "cat mice night",
                        "predicted_outcome": "I expect the JSON endpoint to expose the first animal entry and its diet.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["inputs"]["expected_json_paths"], ["$.animals[0]", "$.animals[0].diet"])
                self.assertEqual(result["result"]["inputs"]["expected_response_shape"], "object")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_path" for item in result["result"]["verification"]["evidence"]))
                self.assertTrue(any(item.get("assertion_kind") == "expected_response_shape" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_response_shape"], "object")
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_paths"], ["$.animals[0]", "$.animals[0].diet"])
                self.assertTrue(any(item.get("assertion_kind") == "expected_response_shape" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_execute_digital_action_persists_explicit_api_value_assertion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_value_assertions",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/animals.json",
                        "expected_json_values": {
                            "$.animals[0].diet": "mice",
                            "$.animals[0].active_time": "night",
                        },
                        "query_text": "cat mice night",
                        "predicted_outcome": "I expect the JSON endpoint to confirm the first animal diet and active time.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["inputs"]["expected_json_values"]["$.animals[0].diet"], "mice")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_value" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_values"]["$.animals[0].diet"], "mice")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_value" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_execute_digital_action_persists_explicit_api_predicate_assertion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91, "count": 7},
                        "animals": [{"name": "cat", "diet": "mice at night"}],
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_predicate_assertions",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                            {"path": "$.metrics.score", "op": "gte", "value": 0.9},
                        ],
                        "query_text": "cat metrics night",
                        "predicted_outcome": "I expect the JSON endpoint to satisfy text and score predicates.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["inputs"]["expected_json_predicates"][0]["op"], "contains")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_predicates"][0]["op"], "contains")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_execute_digital_action_persists_composite_api_predicate_assertion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91, "count": 7},
                        "animals": [{"name": "cat", "diet": "mice at night"}],
                        "tags": ["night-hunter", "feline-companion"],
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_composite_predicate_assertions",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/metrics.json",
                        "expected_json_predicates": [
                            {"path": "$.metrics.score", "op": "between", "value": {"min": 0.9, "max": 1.0}},
                            {"path": "$.animals[0].diet", "op": "startswith", "value": "mice"},
                            {"path": "$.tags", "op": "any_contains", "value": "hunter"},
                        ],
                        "query_text": "cat metrics night",
                        "predicted_outcome": "I expect the JSON endpoint to satisfy composite predicate checks.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["inputs"]["expected_json_predicates"][0]["op"], "between")
                self.assertTrue(any(item.get("predicate_op") == "between" for item in result["result"]["verification"]["evidence"]))
                self.assertTrue(any(item.get("predicate_op") == "any_contains" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_predicates"][0]["op"], "between")
                self.assertTrue(any(item.get("predicate_op") == "between" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_execute_digital_action_persists_logical_api_group_assertion_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "logic.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91},
                        "animals": [{"diet": "mice at night"}],
                        "traits": {"primary": "night-hunter", "secondary": "feline-companion"},
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_logical_group_assertions",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/logic.json",
                        "expected_json_predicates": [
                            {"path": "$.traits", "op": "all_regex", "value": "^[a-z-]+$"},
                        ],
                        "expected_json_predicate_groups": [
                            {
                                "logic": "any",
                                "predicates": [
                                    {"path": "$.metrics.score", "op": "lt", "value": 0.5},
                                    {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                ],
                            }
                        ],
                        "query_text": "logic",
                        "predicted_outcome": "I expect the JSON endpoint to satisfy logical predicate groups and object quantifiers.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertEqual(result["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "any")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in result["result"]["verification"]["evidence"]))
                self.assertTrue(any(item.get("predicate_op") == "all_regex" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_predicate_groups"][0]["logic"], "any")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_execute_digital_action_persists_wildcard_and_nested_group_api_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "wild.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice at night", "traits": ["hunter", "feline"]},
                            {"name": "owl", "diet": "mice at dawn", "traits": ["bird", "night"]},
                        ],
                        "groups": {"predators": [{"name": "cat"}, {"name": "owl"}]},
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            manager = _build_manager(
                root,
                test_case="service_manager_action_loop_api_wildcard_nested_groups",
                env_root=root,
            )
            try:
                result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": f"http://127.0.0.1:{port}/wild.json",
                        "expected_json_paths": ["$.animals[*].diet"],
                        "expected_json_values": {"$.animals[*].name": "owl"},
                        "expected_json_predicate_groups": [
                            {
                                "logic": "all",
                                "groups": [
                                    {
                                        "logic": "any",
                                        "predicates": [
                                            {"path": "$.animals[*].diet", "op": "contains", "value": "night"},
                                            {"path": "$.animals[*].diet", "op": "contains", "value": "reptile"},
                                        ],
                                    },
                                    {
                                        "logic": "none",
                                        "predicates": [
                                            {"path": "$.animals[*].traits[*]", "op": "contains", "value": "reptile"},
                                        ],
                                    },
                                ],
                            }
                        ],
                        "query_text": "wild json nested",
                        "predicted_outcome": "I expect wildcard JSON checks and nested groups to persist on the maintained path.",
                    }
                )
                self.assertTrue(result["accepted"])
                self.assertEqual(result["result"]["action_type"], "api_request")
                self.assertEqual(result["result"]["verification"]["status"], "verified")
                self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].diet" for item in result["result"]["verification"]["evidence"]))
                self.assertEqual(result["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "all")
                self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in result["result"]["verification"]["evidence"]))

                saved = manager.runtime_facade.save_checkpoint()
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            restored = HECSNServiceManager(
                saved["path"],
                trace_dir=root / "restored_traces",
                env_root=root,
            )
            try:
                history = restored.runtime_facade.action_history(limit=4)
                self.assertEqual(history["count"], 1)
                self.assertEqual(history["actions"][0]["inputs"]["expected_json_predicate_groups"][0]["logic"], "all")
                self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].diet" for item in history["actions"][0]["verification"]["evidence"]))
            finally:
                restored.close()

    def test_respond_auto_executes_workspace_web_and_api_action_assist_for_gap_queries(self) -> None:
        cases = [
            _RespondActionAssistCase(
                label="workspace_search",
                test_case="service_manager_action_loop_auto_query_gap",
                query_text="What do cats chase at night?",
                reason="query_gap_auto_search",
                expected_response_fragment="cats chase mice at night",
                setup=_write_query_gap_notes,
            ),
            _RespondActionAssistCase(
                label="workspace_read",
                test_case="service_manager_action_loop_auto_query_read",
                query_text="What does notes.md say cats chase at night?",
                reason="query_gap_auto_read",
                expected_action_type="workspace_read",
                expected_response_fragment="cats chase mice at night",
                expected_input_path="notes.md",
                setup=_write_query_gap_notes,
            ),
            _RespondActionAssistCase(
                label="web_fetch",
                test_case="service_manager_action_loop_auto_query_fetch",
                query_text="What does {url} say cats chase at night?",
                reason="query_gap_auto_fetch",
                expected_action_type="web_fetch",
                expected_response_fragment="cats chase mice at night",
                setup=_write_query_gap_page,
                served_path="page.html",
            ),
            _RespondActionAssistCase(
                label="api_request",
                test_case="service_manager_action_loop_auto_query_api",
                query_text="What does {url} say cats chase at night?",
                reason="query_gap_auto_api_request",
                expected_action_type="api_request",
                expected_response_fragment="mice at night",
                setup=_write_query_gap_data,
                served_path="data.json",
            ),
        ]
        for case in cases:
            with self.subTest(action_type=case.label):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    case.setup(root)
                    query_text = case.query_text
                    server_context = _serve_directory(root) if case.served_path is not None else None
                    if server_context is not None:
                        with server_context as base_url:
                            query_text = query_text.format(url=f"{base_url}/{case.served_path}")
                            self._assert_auto_query_action_assist_case(case, root, query_text)
                        continue
                    self._assert_auto_query_action_assist_case(case, root, query_text)

    def _assert_auto_query_action_assist_case(
        self,
        case: _RespondActionAssistCase,
        root: Path,
        query_text: str,
    ) -> None:
        manager = _build_manager(
            root,
            test_case=case.test_case,
            env_root=root,
        )
        try:
            response = manager.runtime_facade.respond(
                query_text=query_text,
                max_evidence_items=3,
                learn_mode="none",
            )
            assist = response["query_result"]["action_assist"]
            self.assertEqual(assist["reason"], case.reason)
            self.assertTrue(assist["executed"])
            self.assertEqual(assist["result"]["trigger_reason"], case.reason)
            if case.expected_action_type is not None:
                self.assertEqual(assist["result"]["action_type"], case.expected_action_type)
            self.assertIn(case.expected_response_fragment, response["response"]["response_text"].lower())
            if case.expected_input_path is not None:
                self.assertEqual(assist["result"]["inputs"]["path"], case.expected_input_path)
            if case.served_path is not None:
                self.assertIn("http://127.0.0.1", assist["result"]["inputs"]["url"])
            history = manager.runtime_facade.action_history(limit=4)
            self.assertEqual(history["count"], 1)
            if case.expected_action_type is not None:
                self.assertEqual(history["actions"][0]["action_type"], case.expected_action_type)
        finally:
            manager.close()

    def test_respond_does_not_reuse_parameterized_api_request_for_explicit_url_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            port = _free_port()
            server = ThreadingHTTPServer(("127.0.0.1", port), _EchoJsonApiHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            manager = _build_manager(
                root,
                test_case="service_manager_parameterized_api_request_reuse_guard",
                env_root=root,
            )
            url = f"http://127.0.0.1:{port}/api/echo"
            try:
                operator_result = manager.runtime_facade.execute_digital_action(
                    {
                        "action_type": "api_request",
                        "url": url,
                        "method": "POST",
                        "params": {"kind": "feline"},
                        "json_body": {"topic": "cats", "fact": "mice at night"},
                        "query_text": "cats mice night feline",
                        "predicted_outcome": "I expect the API request to return structured JSON about cats and mice at night.",
                    }
                )
                self.assertTrue(operator_result["accepted"])
                self.assertEqual(operator_result["result"]["inputs"]["method"], "POST")

                response = manager.runtime_facade.respond(
                    query_text=f"What does {url} say cats mice at night?",
                    max_evidence_items=3,
                    learn_mode="none",
                )
                assist = response["query_result"]["action_assist"]
                self.assertEqual(assist["reason"], "query_gap_auto_api_request")
                self.assertTrue(assist["executed"])
                self.assertFalse(assist["reused_recent_action"])
                self.assertEqual(assist["result"]["action_type"], "api_request")
                self.assertEqual(assist["result"]["trigger_reason"], "query_gap_auto_api_request")
                self.assertEqual(assist["result"]["verification"]["status"], "contradicted")
                history = manager.runtime_facade.action_history(limit=8)
                self.assertEqual(history["count"], 2)
                self.assertEqual(history["actions"][0]["trigger_reason"], "query_gap_auto_api_request")
                self.assertEqual(history["actions"][0]["inputs"]["method"], "GET")
                self.assertEqual(history["actions"][1]["inputs"]["method"], "POST")
            finally:
                manager.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

if __name__ == "__main__":
    unittest.main()
