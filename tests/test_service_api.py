from __future__ import annotations

import json
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from hecsn.config.model_config import HECSNConfig
from hecsn.service.api import DEFAULT_WEB_DIST_DIR, create_app
from hecsn.service.server import build_arg_parser
from hecsn.training.runner_utils import set_seed
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_checkpoint(root: Path, *, test_case: str) -> Path:
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
    return save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )


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


class ServiceApiTerminusRuntimeTests(unittest.TestCase):
    def test_static_ui_default_points_to_built_frontend_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_static_default")
            app = create_app(ckpt, trace_dir=root / "traces")
            self.assertEqual(app.state.web_dist_dir, DEFAULT_WEB_DIST_DIR)
            app.state.hecsn_manager.close()

        self.assertEqual(DEFAULT_WEB_DIST_DIR, Path("HECSN_UI") / "dist")

        parser = build_arg_parser()
        args = parser.parse_args(["--checkpoint", "checkpoints\\terminus\\model.pt"])
        self.assertEqual(args.web_dist_dir, DEFAULT_WEB_DIST_DIR)

    def test_terminus_configure_and_tick_endpoint_train_from_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("character stream learning " * 32, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_tick"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "api_terminus_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 20,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": False,
                        "ingestion": {"queue_target_tokens": 40, "prewarm_on_startup": False, "prewarm_max_seconds": 0.2},
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 2})
                status_response = client.get("/status")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            self.assertTrue(configure_response.json()["terminus_runtime"]["configured"])
            self.assertGreater(tick_response.json()["token_count"], configure_response.json()["token_count"])
            self.assertEqual(status_response.json()["terminus_runtime"]["source_bank"][0]["name"], "api_terminus_source")
            self.assertEqual(status_response.json()["terminus_runtime"]["ingestion"]["queue_target_tokens"], 40)
            self.assertFalse(status_response.json()["terminus_runtime"]["ingestion"]["prewarm_on_startup"])
            self.assertAlmostEqual(float(status_response.json()["terminus_runtime"]["ingestion"]["prewarm_max_seconds"]), 0.2, places=6)
            self.assertGreater(tick_response.json()["terminus_runtime"]["last_tick_token_delta"], 0)
            self.assertTrue(
                any(event.get("type") == "tick" for event in status_response.json()["terminus_runtime"]["recent_events"])
            )
            self.assertGreater(
                status_response.json()["terminus_runtime"]["source_progress"][0]["tick_visits"],
                0,
            )

    def test_terminus_action_endpoint_executes_workspace_search_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_terminus_action"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    },
                )
                history_response = client.get("/terminus/actions")

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["terminus_runtime"]["action_loop"]["verified_actions"], 1)
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "workspace_search")
            self.assertEqual(history_body["actions"][0]["verification"]["status"], "verified")

    def test_terminus_action_endpoint_executes_workspace_read_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_terminus_action_read"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_read",
                        "path": "notes.md",
                        "query_text": "cats chase night",
                        "predicted_outcome": "I expect notes.md to say what cats chase at night.",
                    },
                )
                history_response = client.get("/terminus/actions")

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "workspace_read")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["result"]["inputs"]["path"], "notes.md")
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "workspace_read")

    def test_terminus_action_endpoint_executes_web_fetch_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "page.html").write_text(
                "<html><body><main><p>Cats chase mice at night.</p><p>Cats rest indoors during the day.</p></main></body></html>",
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_fetch"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "web_fetch",
                            "url": f"http://127.0.0.1:{port}/page.html",
                            "query_text": "cats chase night",
                            "predicted_outcome": "I expect the page to say what cats chase at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "web_fetch")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertIn("http://127.0.0.1", action_body["result"]["inputs"]["url"])
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "web_fetch")

    def test_terminus_action_endpoint_executes_api_request_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data.json").write_text(
                '{"facts": {"chase": "mice at night", "rest": "indoors during the day"}}',
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/data.json",
                            "query_text": "cats chase night",
                            "predicted_outcome": "I expect the JSON endpoint to say what cats chase at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertIn("http://127.0.0.1", action_body["result"]["inputs"]["url"])
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "api_request")

    def test_terminus_action_endpoint_executes_parameterized_api_request_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            port = _free_port()
            server = ThreadingHTTPServer(("127.0.0.1", port), _EchoJsonApiHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_parameterized_api"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/api/echo",
                            "method": "POST",
                            "params": {"kind": "feline"},
                            "json_body": {"topic": "cats", "fact": "mice at night"},
                            "query_text": "cats mice night feline",
                            "predicted_outcome": "I expect the JSON endpoint to echo a structured request about cats and mice at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["method"], "POST")
            self.assertEqual(action_body["result"]["inputs"]["params"]["kind"], "feline")
            self.assertEqual(action_body["result"]["inputs"]["json_body"]["topic"], "cats")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "api_request")
            self.assertEqual(history_body["actions"][0]["inputs"]["method"], "POST")

    def test_terminus_action_endpoint_preserves_structured_api_verification_evidence(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_structured_api_verification"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to identify the animal that hunts mice at night.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["result"]["verification"]["evidence"][0]["json_path"], "$.animals[0]")
            self.assertEqual(action_body["result"]["verification"]["evidence"][0]["structure_kind"], "object")
            self.assertGreaterEqual(action_body["result"]["verification"]["evidence"][0]["field_count"], 3)

    def test_terminus_action_endpoint_verifies_expected_json_paths_and_response_shape(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "expected_json_paths": ["$.animals[0]", "$.animals[0].diet"],
                            "expected_response_shape": "object",
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to expose the first animal entry and its diet.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_paths"], ["$.animals[0]", "$.animals[0].diet"])
            self.assertEqual(action_body["result"]["inputs"]["expected_response_shape"], "object")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_path" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("assertion_kind") == "expected_response_shape" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_expected_json_values(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_value_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "expected_json_values": {
                                "$.animals[0].diet": "mice",
                                "$.animals[0].active_time": "night",
                            },
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to confirm the first animal diet and active time.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_values"]["$.animals[0].diet"], "mice")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_value" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_expected_json_predicates(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_predicate_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/metrics.json",
                            "expected_json_predicates": [
                                {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                {"path": "$.metrics.score", "op": "gte", "value": 0.9},
                            ],
                            "query_text": "cat metrics night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy text and score predicates.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicates"][0]["op"], "contains")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_composite_json_predicates(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_composite_predicates"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/metrics.json",
                            "expected_json_predicates": [
                                {"path": "$.metrics.score", "op": "between", "value": {"min": 0.9, "max": 1.0}},
                                {"path": "$.animals[0].diet", "op": "startswith", "value": "mice"},
                                {"path": "$.tags", "op": "any_contains", "value": "hunter"},
                            ],
                            "query_text": "cat metrics night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy composite predicate checks.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicates"][0]["op"], "between")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("predicate_op") == "between" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("predicate_op") == "any_contains" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_logical_predicate_groups(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_logical_groups"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/logic.json",
                            "expected_json_predicates": [
                                {"path": "$.traits", "op": "all_regex", "value": "^[a-z-]+$"},
                                {"path": "$.traits", "op": "any_contains", "value": "hunter"},
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
                            "query_text": "logic night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy logical predicate groups and object quantifiers.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "any")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("predicate_op") == "all_regex" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_wildcard_and_nested_group_assertions(self) -> None:
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
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_wildcard_nested_groups"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
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
                            "predicted_outcome": "I expect wildcard JSON checks and nested groups to pass on the maintained path.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].diet" for item in action_body["result"]["verification"]["evidence"]))
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "all")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_cortex_sleep_endpoint_requests_sleep_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.cortex.core import MockCortex
            from hecsn.cortex.episodic_memory import SimpleEmbedder

            with patch("hecsn.cortex.multi_cortex.create_cortex_from_env", return_value=MockCortex()), patch(
                "hecsn.cortex.multi_cortex.create_embedder_from_env",
                return_value=SimpleEmbedder(),
            ):
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_cortex_sleep"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
            manager = app.state.hecsn_manager
            manager._thought_loop.sleep_dream_count = 1
            manager._thought_loop.start()
            try:
                with TestClient(app) as client:
                    sleep_response = client.post(
                        "/terminus/cortex/sleep",
                        json={"reason": "Operator requested a consolidation cycle."},
                    )

                    deadline = time.time() + 2.0
                    cortex_snapshot = {}
                    terminus_runtime = {}
                    while time.time() < deadline:
                        cortex_snapshot = client.get("/terminus/cortex").json()
                        terminus_runtime = client.get("/terminus").json()["terminus_runtime"]
                        if cortex_snapshot.get("sleep_control", {}).get("requested_cycles_completed", 0) >= 1:
                            break
                        time.sleep(0.02)

                self.assertEqual(sleep_response.status_code, 200)
                sleep_body = sleep_response.json()
                self.assertTrue(sleep_body["accepted"])
                self.assertEqual(sleep_body["request"]["source"], "operator")
                self.assertEqual(cortex_snapshot.get("sleep_control", {}).get("last_cycle", {}).get("trigger"), "requested")
                self.assertTrue(any(event.get("type") == "cortex_sleep_requested" for event in terminus_runtime.get("recent_events", [])))
                self.assertTrue(any(event.get("type") == "cortex_sleep_completed" for event in terminus_runtime.get("recent_events", [])))
            finally:
                manager.close()

    def test_respond_endpoint_auto_executes_workspace_action_for_gap_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            (root / "page.html").write_text(
                "<html><body><main><p>Cats chase mice at night.</p><p>Cats rest indoors during the day.</p></main></body></html>",
                encoding="utf-8",
            )
            (root / "data.json").write_text(
                '{"facts": {"chase": "mice at night", "rest": "indoors during the day"}}',
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_respond_auto_action"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    respond_response = client.post(
                        "/respond",
                        json={
                            "query_text": "What do cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    read_response = client.post(
                        "/respond",
                        json={
                            "query_text": "What does notes.md say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    fetch_response = client.post(
                        "/respond",
                        json={
                            "query_text": f"What does http://127.0.0.1:{port}/page.html say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    api_response = client.post(
                        "/respond",
                        json={
                            "query_text": f"What does http://127.0.0.1:{port}/data.json say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    actions_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(respond_response.status_code, 200)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(fetch_response.status_code, 200)
            self.assertEqual(api_response.status_code, 200)
            self.assertEqual(actions_response.status_code, 200)
            body = respond_response.json()
            read_body = read_response.json()
            fetch_body = fetch_response.json()
            api_body = api_response.json()
            actions = actions_response.json()
            self.assertEqual(body["query_result"]["action_assist"]["reason"], "query_gap_auto_search")
            self.assertTrue(body["query_result"]["action_assist"]["executed"])
            self.assertIn("cats chase mice at night", body["response"]["response_text"].lower())
            self.assertEqual(read_body["query_result"]["action_assist"]["reason"], "query_gap_auto_read")
            self.assertEqual(read_body["query_result"]["action_assist"]["result"]["action_type"], "workspace_read")
            self.assertIn("cats chase mice at night", read_body["response"]["response_text"].lower())
            self.assertEqual(fetch_body["query_result"]["action_assist"]["reason"], "query_gap_auto_fetch")
            self.assertEqual(fetch_body["query_result"]["action_assist"]["result"]["action_type"], "web_fetch")
            self.assertIn("cats chase mice at night", fetch_body["response"]["response_text"].lower())
            self.assertEqual(api_body["query_result"]["action_assist"]["reason"], "query_gap_auto_api_request")
            self.assertEqual(api_body["query_result"]["action_assist"]["result"]["action_type"], "api_request")
            self.assertIn("mice at night", api_body["response"]["response_text"].lower())
            self.assertEqual(actions["count"], 4)
            self.assertIn(actions["actions"][0]["trigger_reason"], {"query_gap_auto_api_request", "query_gap_auto_fetch", "query_gap_auto_read", "query_gap_auto_search"})

    def test_terminus_tick_endpoint_rejects_when_runtime_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("runtime ownership safety signal " * 32, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_tick_running"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "api_terminus_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 20,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                start_response = client.post("/terminus/start")
                tick_response = client.post("/terminus/tick", json={"steps": 1})
                stop_response = client.post("/terminus/stop")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(start_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 422)
            self.assertIn("background runtime is active", tick_response.json()["detail"])
            self.assertEqual(stop_response.status_code, 200)

    def test_terminus_tick_then_respond_returns_grounded_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_grounding.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "cats rest indoors.",
                            "cats chase mice at night.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_grounded_synthesis"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_grounding_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 48,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 6})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "Where do cats rest and what do they chase at night?",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertIn(body["response"]["response_mode"], {"quote", "grounded_synthesis", "stitch"})
            self.assertIn("indoors", body["response"]["response_text"].lower())
            self.assertIn("mice", body["response"]["response_text"].lower())

    def test_terminus_tick_then_respond_handles_unsegmented_character_stream_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_unsegmented_query.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "submarines regulate buoyancy with ballast tanks.",
                            "ballast water shifts pressure and buoyancy inside a submarine.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_unsegmented_character_stream_query"),
                trace_dir=root / "traces",
            )
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_unsegmented_query_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 48,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 6})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "submarineballast",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertIn(body["response"]["response_mode"], {"quote", "grounded_synthesis", "stitch"})
            self.assertTrue(
                any(term in body["response"]["response_text"].lower() for term in ("submarine", "ballast", "buoyancy"))
            )
            self.assertEqual(body["response"]["unsupported_terms"], [])

    def test_terminus_tick_then_respond_handles_mixed_world_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_mixed_world.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "mercury is the closest planet to the sun.",
                            "volcanoes release ash and lava during eruptions.",
                            "octopuses solve puzzles and open jars.",
                            "rainbows form when sunlight passes through water droplets.",
                            "libraries lend books and provide quiet reading rooms.",
                            "moss grows on damp forest stones.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_mixed_world"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_mixed_world_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 64,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 10})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "What is closest to the sun and what do volcanoes release?",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertEqual(body["response"]["response_mode"], "grounded_synthesis")
            self.assertIn("mercury", body["response"]["response_text"].lower())
            self.assertTrue(
                any(term in body["response"]["response_text"].lower() for term in ("ash", "lava"))
            )

    def test_terminus_configure_accepts_registry_backed_candidate_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_catalog_candidate_bank"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "api_terminus_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 20,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": False,
                        "autonomy": {
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
                            "trigger_interval_tokens": 200,
                        },
                    },
                )
                status_response = client.get("/status")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            autonomy = status_response.json()["terminus_runtime"]["autonomy"]
            self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "semantic_registry")
            self.assertEqual(len(autonomy["candidate_bank"][0]["catalog_entries"]), 2)
            self.assertEqual(autonomy["candidate_names"], ["registry_pool"])

    def test_terminus_tick_commits_live_remote_search_candidate_from_recent_query_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(_build_checkpoint(root, test_case="service_api_live_remote_search_commit"), trace_dir=root / "traces")

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "candidate_bank": [
                                        {
                                            "name": "live_remote_pool",
                                            "catalog_mode": "live_remote_search",
                                            "catalog_providers": ["wikipedia"],
                                            "catalog_queries_per_provider": 1,
                                            "catalog_provider_result_limit": 3,
                                            "catalog_limit": 3,
                                            "catalog_probe_pool_limit": 3,
                                        }
                                    ],
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                    "semantic_shortlist_size": 1,
                                    "semantic_shortlist_gap_weight": 0.0,
                                    "semantic_shortlist_affinity_weight": 1.0,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)

                query_body = query_response.json()
                tick_body = tick_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertIn("submarine", query_body["gap_plan"]["unsupported_terms"])
                self.assertIn("submarine", autonomy["focus_plan"]["unsupported_terms"])
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_recovers_grounded_answer_from_recent_query_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_answer_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                query_body = query_response.json()
                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    autonomy["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertIn("submarine", query_body["gap_plan"]["unsupported_terms"])
                self.assertIn("submarine", autonomy["focus_plan"]["unsupported_terms"])
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_uses_catalog_summary_for_first_tick_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine_delayed.txt"
            submarine_path.write_text(
                (
                    "submarines travel underwater for naval operations and long endurance patrols near ocean fleets. "
                    * 10
                )
                + (
                    "submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. "
                    * 16
                ),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_catalog_summary_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine_delayed.txt",
                        "source_type": "web",
                        "summary": (
                            "Submarines regulate buoyancy with ballast tanks. "
                            "Ballast water shifts pressure and buoyancy inside a submarine."
                        ),
                        "terms": ["marine engineering", "ballast tank"],
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_focuses_late_summary_fragments_for_first_tick_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            ballast_path = root / "ballast_delayed.txt"
            ballast_path.write_text(
                (
                    "ballast tanks are mechanical compartments for marine systems and vessel trim adjustments. "
                    * 10
                )
                + (
                    "ballast water reduces buoyancy in a submarine and supports underwater trim control. "
                    * 16
                ),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_focus_fragment_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "ballast_tank_source",
                        "source": f"http://127.0.0.1:{content_port}/ballast_delayed.txt",
                        "source_type": "web",
                        "summary": (
                            "A ballast tank is a compartment within a boat, ship, or floating structure that holds water, "
                            "ballast water reduces buoyancy in a submarine, and helps correct trim."
                        ),
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["ballast_tank_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("ballast", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "submarine"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_live_remote_search_probes_cross_provider_page_content_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            ballast_paper_path = root / "ballast_paper.txt"
            ballast_paper_path.write_text(
                (
                    "marine engineering reports compare underwater trim and ballast systems. "
                    "ballast tanks reduce submarine buoyancy and support underwater trim control. "
                )
                * 18,
                encoding="utf-8",
            )
            cable_path = root / "submarine_cable.txt"
            cable_path.write_text(
                "submarine cables carry internet traffic between continents and coastal landing stations. " * 24,
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_live_remote_cross_provider_content_probe"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider == "wikipedia":
                    return [
                        {
                            "name": "cable_source",
                            "source": f"http://127.0.0.1:{content_port}/submarine_cable.txt",
                            "source_type": "web",
                            "summary": "Submarine infrastructure and communications systems.",
                            "query_text": query,
                            "catalog_priority": 0.70,
                            "provider": provider,
                        }
                    ][:result_limit]
                if provider == "openalex":
                    return [
                        {
                            "name": "ballast_paper_source",
                            "source": f"http://127.0.0.1:{content_port}/ballast_paper.txt",
                            "source_type": "web",
                            "summary": "Marine systems analysis of vessel stability and trim.",
                            "terms": ["marine engineering"],
                            "query_text": query,
                            "catalog_priority": 0.35,
                            "provider": provider,
                        }
                    ][:result_limit]
                return []

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "candidate_bank": [
                                        {
                                            "name": "live_remote_pool",
                                            "catalog_mode": "live_remote_search",
                                            "catalog_providers": ["wikipedia", "openalex"],
                                            "catalog_limit": 1,
                                            "catalog_probe_pool_limit": 2,
                                            "catalog_prior_weight": 0.4,
                                            "catalog_queries_per_provider": 1,
                                            "catalog_provider_result_limit": 1,
                                        }
                                    ],
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["ballast_paper_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_can_recover_via_follow_up_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_follow_up_probe"),
                trace_dir=root / "traces",
            )
            provider_queries: list[str] = []

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                provider_queries.append(f"{provider}:{query}")
                if provider != "wikipedia":
                    return []
                if query.strip().lower() != "submarine":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    }
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    autonomy["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(autonomy["candidate_bank"][0]["catalog_queries_per_provider"], 2)
                self.assertIn("wikipedia:submarine buoyancy ballast", provider_queries)
                self.assertIn("wikipedia:submarine", provider_queries)
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_grows_query_budget_from_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_query_growth"),
                trace_dir=root / "traces",
            )
            provider_queries: list[str] = []

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                provider_queries.append(f"{provider}:{query}")
                if provider != "wikipedia":
                    return []
                if query.strip().lower() != "submarine ballast buoyancy":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    }
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        with app.state.hecsn_manager._lock:
                            app.state.hecsn_manager._record_recent_query_gap_locked(
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
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(autonomy["candidate_bank"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(
                    autonomy["focus_plan"]["weak_concepts"][0]["top_terms"],
                    ["garden", "tomato", "soil"],
                )
                self.assertIn("wikipedia:submarine", provider_queries)
                self.assertIn("wikipedia:garden tomato soil", provider_queries)
                self.assertIn("wikipedia:submarine ballast buoyancy", provider_queries)
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_start_rejects_missing_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_start_guard"), trace_dir=root / "traces")
            with TestClient(app) as client:
                response = client.post("/terminus/start")

            self.assertEqual(response.status_code, 422)
            self.assertIn("source_bank", response.text)

    def test_legacy_brain_report_and_preset_endpoints_are_not_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_legacy_surface_removed"), trace_dir=root / "traces")
            with TestClient(app) as client:
                brain_response = client.get("/brain")
                presets_response = client.get("/acquisition/presets")
                reports_response = client.get("/reports/benchmarks")

            self.assertEqual(brain_response.status_code, 404)
            self.assertEqual(presets_response.status_code, 404)
            self.assertEqual(reports_response.status_code, 404)

    def test_architecture_endpoint_returns_layer_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_architecture"), trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/architecture")

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model_name"], "Terminus")
            self.assertEqual(data["core_name"], "GPCSN")
            self.assertEqual(data["version"], "current")
            layers = data["layers"]
            self.assertIsInstance(layers, list)
            self.assertGreater(len(layers), 0)
            layer_ids = [l["id"] for l in layers]
            self.assertIn("input_encoding", layer_ids)
            self.assertIn("competitive_routing", layer_ids)
            self.assertIn("predictive_columns", layer_ids)
            self.assertIn("memory_consolidation", layer_ids)
            self.assertIn("nim_cortex", layer_ids)
            for layer in layers:
                self.assertIn("id", layer)
                self.assertIn("name", layer)
                self.assertIn("enabled", layer)
                self.assertIn("type", layer)
                self.assertIn("params", layer)
            enabled_layers = [l for l in layers if l["enabled"]]
            self.assertGreaterEqual(len(enabled_layers), 3)

    def test_grounding_probe_endpoint_returns_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_grounding_probe"), trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.post("/grounding-probe/run")

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("total_accuracy", data)
            self.assertIn("concrete_accuracy", data)
            self.assertIn("abstract_accuracy", data)
            self.assertIn("concreteness_gap", data)
            self.assertIsInstance(data["total_accuracy"], float)
            self.assertGreaterEqual(data["total_accuracy"], 0.0)
            self.assertLessEqual(data["total_accuracy"], 1.0)

    def test_telemetry_snapshot_includes_animation_data(self) -> None:
        """telemetry_snapshot includes animation sub-object for SSE consumers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            mgr = HECSNServiceManager(
                _build_checkpoint(root, test_case="service_api_animation"),
                trace_dir=root / "traces",
            )
            snapshot = mgr.telemetry_snapshot()
            mgr.close()

        self.assertIn("animation", snapshot)
        anim = snapshot["animation"]
        self.assertIn("n_columns", anim)
        self.assertIn("activations", anim)
        self.assertIn("spike_counts", anim)
        self.assertIn("memory_fill", anim)
        self.assertEqual(len(anim["activations"]), anim["n_columns"])

    def test_quick_start_presets_endpoint(self) -> None:
        """GET /terminus/presets returns available preset list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_presets")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/terminus/presets")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            ids = [p["id"] for p in data]
            self.assertEqual(ids, ["curriculum"])
            self.assertNotIn("multimodal", ids)
            self.assertNotIn("multimodal_fast", ids)
            self.assertEqual(data[0]["id"], "curriculum")
            self.assertTrue(data[0].get("default"))
            for preset in data:
                self.assertIn("label", preset)
                self.assertIn("description", preset)
                self.assertIn("source_count", preset)
                self.assertIn("default", preset)
                self.assertIn("legacy", preset)

    def test_quick_start_configures_and_starts_terminus(self) -> None:
        """POST /terminus/quick-start uses the recommended curriculum preset by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "quick_start_source.txt"
            source_path.write_text("The platypus is an egg-laying mammal found in eastern Australia. " * 5)
            ckpt = _build_checkpoint(root, test_case="service_api_quick_start")
            app = create_app(ckpt, trace_dir=root / "traces")

            def _fake_start(self):
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    "dirty_state": bool(self._dirty_state),
                    "state_revision": int(self._state_revision),
                    "token_count": int(self._trainer.token_count),
                }

            with patch("hecsn.service.manager.HECSNServiceManager.start_terminus", autospec=True, side_effect=_fake_start):
                with TestClient(app) as client:
                    resp = client.post("/terminus/quick-start")
                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    self.assertTrue(data["terminus_runtime"]["configured"])
                    self.assertFalse(data.get("already_running", False))
                    self.assertEqual(data.get("preset_applied"), "curriculum")
                    self.assertEqual(data["terminus_runtime"]["source_count"], 3)
                    self.assertLessEqual(data["terminus_runtime"]["tick_tokens"], 64)
                    self.assertTrue(data["terminus_runtime"]["autonomy"]["enabled"])
                    self.assertEqual(data["terminus_runtime"]["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                    manager = app.state.hecsn_manager
                    self.assertTrue(manager._trainer.config.enable_context_layer)
                    self.assertTrue(manager._trainer.config.enable_binding_layer)
                    self.assertNotIn("curriculum", data["terminus_runtime"])
                    stop_resp = client.post("/terminus/stop")
                    self.assertEqual(stop_resp.status_code, 200)

    def test_datasets_endpoint_reports_current_runtime_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_datasets")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/datasets")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            names = [entry["name"] for entry in data["datasets"]]
            self.assertIn("fineweb_edu", names)
            self.assertIn("wikipedia_en", names)
            self.assertIn("s2orc_arxiv_abstracts", names)
            self.assertIn("science_figures", names)
            self.assertIn("environmental_audio", names)
            self.assertNotIn("nim_curriculum", names)
            self.assertNotIn("N-MNIST", names)
            self.assertNotIn("FSDD", names)
            self.assertIn("huggingface", data)
            self.assertIn("token_configured", data["huggingface"])

    def test_sensory_recent_endpoint_returns_media_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_sensory_recent")
            app = create_app(ckpt, trace_dir=root / "traces")
            manager = app.state.hecsn_manager
            with manager._lock:
                manager._sensory_preview_history.appendleft(
                    {
                        "preview_id": "preview-1",
                        "captured_at": "2026-04-21T00:00:00+00:00",
                        "source_name": "science_figures",
                        "adapter": "s1_mmalign",
                        "text": "example figure",
                        "semantic_match": 0.8,
                        "modality_need": 0.5,
                        "selection_score": 0.9,
                        "window_budget": 6,
                        "topics": ["scientific", "diagram"],
                        "focus_terms": ["scientific", "diagram"],
                        "metadata": {"title": "phase diagram"},
                        "visual": {"mime_type": "image/png", "bytes": b"fakepng", "width": 16, "height": 16},
                        "audio": None,
                    }
                )
            with TestClient(app) as client:
                resp = client.get("/terminus/sensory/recent?limit=1")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["latest_preview_id"], "preview-1")
            self.assertEqual(len(data["previews"]), 1)
            self.assertTrue(data["previews"][0]["visual"]["data_url"].startswith("data:image/png;base64,"))

    def test_quick_start_rejects_unknown_preset(self) -> None:
        """POST /terminus/quick-start with bad preset returns 422."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_qs_bad")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.post("/terminus/quick-start?preset=nonexistent")
            self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
